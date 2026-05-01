import { getModuleLogger } from "../logging/logger.js";
import { normalizeCoreModelName, resolveCoreModelUrl } from "../model-catalog.js";

export interface TensorShape extends ReadonlyArray<number> {}

export interface InferenceTensorLike {
  data: Float32Array;
  dims: number[];
}

export interface ORTFeeds {
  [name: string]: InferenceTensorLike;
}

export interface ORTNamespace {
  Tensor: new (
    dataType: 'float32',
    data: Float32Array,
    dims: TensorShape,
  ) => InferenceTensorLike;
}

export interface InferenceSessionLike {
  inputNames?: readonly string[];
  outputNames: readonly string[];
  getInputs?(): Array<{ name?: string }>;
  run(feeds: ORTFeeds): Promise<Record<string, InferenceTensorLike>>;
}

export interface BrowserORTNamespace extends ORTNamespace {
  env?: {
    wasm?: {
      wasmPaths?: string;
    };
  };
  InferenceSession?: {
    create(
      modelUrl: string,
      options?: { executionProviders?: readonly string[] },
    ): Promise<InferenceSessionLike>;
  };
}

export interface SegmentLike {
  validT: number;
}

export const MODEL_IO = Object.freeze({
  BATCH: 1,
  T_PER_BATCH: 128,
  C: 3,
  FREQ: 360,
  INPUT_SHAPE: Object.freeze([1, 3, 360, 128]) as TensorShape,
  OUTPUT_SHAPE: Object.freeze([1, 361, 128]) as TensorShape,
});

const inferenceLogger = getModuleLogger("core.inference");

export function resolveSessionInputName(session: InferenceSessionLike | null | undefined): string {
  if (!session) {
    return '';
  }
  const directName = session.inputNames?.[0];
  if (directName) {
    return directName;
  }
  if (typeof session.getInputs === 'function') {
    const inputs = session.getInputs();
    const name = inputs?.[0]?.name;
    if (name) {
      return name;
    }
  }
  return '';
}

export function createORTFeeds(
  ort: ORTNamespace,
  inputName: string | undefined,
  batchData: Float32Array,
  shape: TensorShape = MODEL_IO.INPUT_SHAPE,
): ORTFeeds {
  const key = String(inputName || 'inputs').trim() || 'inputs';
  if (ort && typeof ort.Tensor === 'function') {
    return { [key]: new ort.Tensor('float32', batchData, shape) };
  }
  return {
    [key]: {
      data: batchData,
      dims: [...shape],
    },
  };
}

export function computeArgmaxIndices(
  data: Float32Array,
  B: number,
  F: number,
  T: number,
): Int32Array {
  const result = new Int32Array(B * T);
  for (let b = 0; b < B; b += 1) {
    for (let t = 0; t < T; t += 1) {
      let best = -Infinity;
      let bestF = 0;
      for (let f = 0; f < F; f += 1) {
        const idx = b * F * T + f * T + t;
        const v = Number(data[idx] ?? Number.NEGATIVE_INFINITY);
        if (v > best) {
          best = v;
          bestF = f;
        }
      }
      result[b * T + t] = bestF;
    }
  }
  return result;
}

export function collectBatchArgmax(
  batchArgmax: Int32Array | number[],
  chunk: readonly SegmentLike[],
  B: number,
  T: number,
): number[] {
  const batch: number[] = [];
  for (let b = 0; b < B && b < chunk.length; b += 1) {
    const current = chunk[b];
    if (!current) {
      continue;
    }
    const keepT = Math.max(0, Math.floor(Number(current.validT) || 0));
    for (let t = 0; t < keepT; t += 1) {
      const raw = Number(batchArgmax[b * T + t] ?? 0);
      const normalized = Number.isFinite(raw) ? Math.floor(raw) - 1 : -1;
      batch.push(normalized);
    }
  }
  return batch;
}

export function marginToConfidence(best: number, secondBest: number): number {
  const margin = Number(best) - Number(secondBest);
  if (!Number.isFinite(margin)) {
    return 0;
  }
  if (margin >= 20) {
    return 1;
  }
  if (margin <= -20) {
    return 0;
  }
  return 1 / (1 + Math.exp(-margin));
}

export function collectBatchConfidenceFromScores(
  scoreData: Float32Array,
  chunk: readonly SegmentLike[],
  B: number,
  F: number,
  T: number,
): number[] {
  const batch: number[] = [];
  for (let b = 0; b < B && b < chunk.length; b += 1) {
    const current = chunk[b];
    if (!current) {
      continue;
    }
    const keepT = Math.max(0, Math.floor(Number(current.validT) || 0));
    for (let t = 0; t < keepT; t += 1) {
      let best = -Infinity;
      let second = -Infinity;
      for (let f = 0; f < F; f += 1) {
        const idx = b * F * T + f * T + t;
        const v = Number(scoreData[idx] ?? Number.NEGATIVE_INFINITY);
        if (v > best) {
          second = best;
          best = v;
        } else if (v > second) {
          second = v;
        }
      }
      batch.push(marginToConfidence(best, second));
    }
  }
  return batch;
}

function now(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

export async function inferBatchWithRetry({
  session,
  ort,
  batchData,
  retryWarnMessage,
  retryErrorMessage,
  createWasmSession,
}: {
  session: InferenceSessionLike;
  ort: ORTNamespace;
  batchData: Float32Array;
  retryWarnMessage: string;
  retryErrorMessage: string;
  createWasmSession?: (() => Promise<InferenceSessionLike | null>) | null;
}): Promise<{
  session: InferenceSessionLike;
  outTensor: { data: Float32Array; dims: number[] };
  elapsedMs: number;
}> {
  const inputName = resolveSessionInputName(session);
  const feeds = createORTFeeds(ort, inputName, batchData, MODEL_IO.INPUT_SHAPE);
  let modelOut: Record<string, InferenceTensorLike>;
  const tInferStart = now();

  try {
    modelOut = await session.run(feeds);
  } catch (error) {
    if (typeof createWasmSession === 'function') {
      const newSession = await createWasmSession();
      if (newSession) {
        session = newSession;
        try {
          modelOut = await session.run(feeds);
        } catch (retryError) {
          throw new Error(
            `${retryErrorMessage}: ${retryError instanceof Error ? retryError.message : String(retryError)}`,
          );
        }
      } else {
        throw error instanceof Error ? error : new Error(String(error));
      }
    } else {
      throw new Error(
        `${retryWarnMessage}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  const outName = session.outputNames[0] ?? '';
  const outTensor = (outName && modelOut[outName]) || Object.values(modelOut)[0] || null;
  if (!outTensor || !outTensor.data || outTensor.dims.length !== 3) {
    throw new Error(
      `模型输出异常，预期[${MODEL_IO.BATCH},361,${MODEL_IO.T_PER_BATCH}]，实际=${outTensor?.dims ? `[${outTensor.dims.join(",")}]` : "unknown"}`,
    );
  }

  return {
    session,
    outTensor,
    elapsedMs: now() - tInferStart,
  };
}

export async function collectBatchPredictions({
  outTensor,
  chunk,
}: {
  outTensor: { data: Float32Array; dims: number[] };
  chunk: readonly SegmentLike[];
}): Promise<{
  batchArgmax: number[];
  batchConfidence: number[];
}> {
  const B = outTensor.dims[0] ?? 0;
  const F = outTensor.dims[1] ?? 0;
  const T = outTensor.dims[2] ?? 0;
  const outData = outTensor.data;

  return {
    batchArgmax: collectBatchArgmax(computeArgmaxIndices(outData, B, F, T), chunk, B, T),
    batchConfidence: collectBatchConfidenceFromScores(outData, chunk, B, F, T),
  };
}

const BROWSER_ORT_SCRIPT_URL =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.min.js";
const BROWSER_ORT_WASM_PATH =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";

export const CORE_INFERENCE_WORKER_MODULE_URL = new URL(
  "./worker.js",
  import.meta.url,
).toString();

let browserOrtRuntimePromise: Promise<BrowserORTNamespace> | null = null;

function resolveBrowserOrtRuntime(): BrowserORTNamespace | null {
  const runtimeGlobal = globalThis as typeof globalThis & {
    ort?: BrowserORTNamespace;
  };
  return runtimeGlobal.ort ?? null;
}

function loadBrowserOrtRuntime(): Promise<BrowserORTNamespace> {
  const existingOrt = resolveBrowserOrtRuntime();
  if (existingOrt?.InferenceSession?.create) {
    inferenceLogger.info("browser ORT runtime already available");
    return Promise.resolve(existingOrt);
  }

  if (browserOrtRuntimePromise) {
    return browserOrtRuntimePromise;
  }

  if (typeof document === "undefined") {
    browserOrtRuntimePromise = (async () => {
      const runtimeGlobal = globalThis as typeof globalThis & {
        ort?: BrowserORTNamespace;
        importScripts?: (...urls: string[]) => void;
      };
      if (typeof runtimeGlobal.importScripts === "function") {
        runtimeGlobal.importScripts(BROWSER_ORT_SCRIPT_URL);
      } else {
        const response = await fetch(BROWSER_ORT_SCRIPT_URL);
        if (!response.ok) {
          throw new Error(
            `failed to load onnxruntime-web runtime (HTTP ${response.status})`,
          );
        }
        const code = await response.text();
        (0, eval)(`${code}\n//# sourceURL=${BROWSER_ORT_SCRIPT_URL}`);
      }
      const ort = resolveBrowserOrtRuntime();
      if (!ort?.InferenceSession?.create) {
        throw new Error("onnxruntime-web did not register global ort");
      }
      if (ort.env?.wasm) {
        ort.env.wasm.wasmPaths = BROWSER_ORT_WASM_PATH;
      }
      runtimeGlobal.ort = ort;
      return ort;
    })().catch((error: unknown) => {
      browserOrtRuntimePromise = null;
      throw error;
    });

    return browserOrtRuntimePromise;
  }

  browserOrtRuntimePromise = new Promise<BrowserORTNamespace>((resolve, reject) => {
    const runtimeGlobal = globalThis as typeof globalThis & {
      ort?: BrowserORTNamespace;
    };
    const existing = document.querySelector<HTMLScriptElement>(
      `script[data-ohm-ort="true"]`,
    );
    const finish = () => {
      const ort = resolveBrowserOrtRuntime();
      if (ort?.InferenceSession?.create) {
        if (ort.env?.wasm) {
          ort.env.wasm.wasmPaths = BROWSER_ORT_WASM_PATH;
        }
        resolve(ort);
        return;
      }
      reject(new Error("onnxruntime-web did not register global ort"));
    };

    if (existing) {
      if (runtimeGlobal.ort?.InferenceSession?.create) {
        finish();
        return;
      }
      existing.addEventListener("load", finish, { once: true });
      existing.addEventListener(
        "error",
        () => reject(new Error("failed to load onnxruntime-web")),
        { once: true },
      );
      return;
    }

    const script = document.createElement("script");
    script.dataset.ohmOrt = "true";
    script.src = BROWSER_ORT_SCRIPT_URL;
    script.async = true;
    script.onload = finish;
    script.onerror = () => reject(new Error("failed to load onnxruntime-web"));
    document.head.appendChild(script);
  }).catch((error: unknown) => {
    browserOrtRuntimePromise = null;
    throw error;
  });

  return browserOrtRuntimePromise;
}

export async function ensureBrowserInferenceSession({
  modelName,
  session,
  ort,
  currentModelName,
}: {
  modelName: string;
  session?: InferenceSessionLike | null;
  ort?: BrowserORTNamespace | null;
  currentModelName?: string | null;
}): Promise<{
  session: InferenceSessionLike;
  ort: BrowserORTNamespace;
  currentModelName: string;
}> {
  const safeModelName = normalizeCoreModelName(modelName);
  if (
    session &&
    ort &&
    currentModelName === safeModelName &&
    ort.InferenceSession?.create
  ) {
    inferenceLogger.info(`browser ORT session reused for model=${safeModelName}`);
    return {
      session,
      ort,
      currentModelName: safeModelName,
    };
  }

  const loadedOrt = ort?.InferenceSession?.create
    ? ort
    : await loadBrowserOrtRuntime();
  if (loadedOrt.env?.wasm) {
    loadedOrt.env.wasm.wasmPaths = BROWSER_ORT_WASM_PATH;
  }

  const modelUrl = resolveCoreModelUrl(safeModelName);
  inferenceLogger.info(`browser ORT loading model=${safeModelName} url=${modelUrl}`);
  const createSession = loadedOrt.InferenceSession?.create;
  if (!createSession) {
    throw new Error("onnxruntime-web did not register InferenceSession.create");
  }

  const nextSession = await createSession(modelUrl, {
    executionProviders: ["wasm"],
  });
  inferenceLogger.info(
    `browser ORT session ready model=${safeModelName} inputShape=${MODEL_IO.INPUT_SHAPE.join(",")} outputShape=${MODEL_IO.OUTPUT_SHAPE.join(",")}`,
  );

  return {
    session: nextSession,
    ort: loadedOrt,
    currentModelName: safeModelName,
  };
}
