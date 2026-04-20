const INFERENCE_LOG_PREFIX = "[core.inference.worker]";
const ORT_CDN_VERSION = "1.24.3";
const DEFAULT_ORT_WEBGPU_SCRIPT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_CDN_VERSION}/dist/ort.webgpu.min.mjs`;
const DEFAULT_ORT_WASM_SCRIPT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_CDN_VERSION}/dist/ort.min.mjs`;
const DEFAULT_ORT_WASM_BASE_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_CDN_VERSION}/dist/`;
const DEFAULT_MODEL_NAME = "msnet.onnx";
const MODEL_NAMES = CORE_MODEL_NAMES = [
  "mamba_a.onnx",
  "mamba_b.onnx",
  "mftfa_a.onnx",
  "mftfa_b.onnx",
  "msnet.onnx",
];
const MODEL_URL_BASE = new URL("../../models/", import.meta.url).toString();
const MODEL_IO = Object.freeze({
  BATCH: 1,
  T_PER_BATCH: 128,
  C: 3,
  FREQ: 360,
  INPUT_SHAPE: Object.freeze([1, 3, 360, 128]),
  OUTPUT_SHAPE: Object.freeze([1, 361, 128]),
});

let webgpuScriptUrl = DEFAULT_ORT_WEBGPU_SCRIPT_URL;
let wasmScriptUrl = DEFAULT_ORT_WASM_SCRIPT_URL;
let ortWasmBaseUrl = DEFAULT_ORT_WASM_BASE_URL;
let ortRuntime = null;
let session = null;
let providerUsed = null;
let currentModelName = "";
const loadedScripts = new Set();

function logInfo(message, payload) {
  if (payload) {
    console.log(`${INFERENCE_LOG_PREFIX} ${message}`, payload);
    return;
  }
  console.log(`${INFERENCE_LOG_PREFIX} ${message}`);
}

function logWarn(message, payload) {
  if (payload) {
    console.warn(`${INFERENCE_LOG_PREFIX} ${message}`, payload);
    return;
  }
  console.warn(`${INFERENCE_LOG_PREFIX} ${message}`);
}

function buildEmptyInferenceResult() {
  return {
    totalArgmax: [],
    totalConfidence: [],
    visibleArgmax: [],
    visibleConfidence: [],
    totalExpectedFrames: 0,
    totalBatchCount: 0,
  };
}

function normalizeCoreModelName(value) {
  const safeValue = String(value || "").trim();
  if (MODEL_NAMES.includes(safeValue)) {
    return safeValue;
  }
  return DEFAULT_MODEL_NAME;
}

function resolveModelUrl(modelName) {
  return `${MODEL_URL_BASE}${encodeURIComponent(normalizeCoreModelName(modelName))}`;
}

function resolveSessionInputName(nextSession) {
  if (!nextSession) {
    return "";
  }
  const directName = nextSession.inputNames?.[0];
  if (directName) {
    return directName;
  }
  if (typeof nextSession.getInputs === "function") {
    const inputs = nextSession.getInputs();
    const name = inputs?.[0]?.name;
    if (name) {
      return name;
    }
  }
  return "";
}

function getOrtRuntime() {
  const runtimeGlobal = globalThis;
  return runtimeGlobal.ort ?? ortRuntime ?? null;
}

function setOrtRuntime(nextOrt) {
  ortRuntime = nextOrt;
  globalThis.ort = nextOrt;
  if (nextOrt?.env?.wasm) {
    nextOrt.env.wasm.wasmPaths = ortWasmBaseUrl;
  }
  return nextOrt;
}

function loadOrtRuntime(scriptUrl) {
  const existing = getOrtRuntime();
  if (existing?.InferenceSession?.create && loadedScripts.has(scriptUrl)) {
    return Promise.resolve(existing);
  }

  return import(scriptUrl)
    .then((moduleNamespace) => {
      const ort = moduleNamespace?.InferenceSession?.create
        ? moduleNamespace
        : moduleNamespace?.default ?? null;
      if (!ort?.InferenceSession?.create) {
        throw new Error("onnxruntime-web did not register global ort");
      }
      loadedScripts.add(scriptUrl);
      return setOrtRuntime(ort);
    })
    .catch((error) => {
      throw error;
    });
}

function snapshotWasmEnv() {
  const wasmEnv = getOrtRuntime()?.env?.wasm;
  if (!wasmEnv) {
    return null;
  }
  return {
    proxy: wasmEnv.proxy,
    numThreads: wasmEnv.numThreads,
  };
}

function restoreWasmEnv(snapshot) {
  const wasmEnv = getOrtRuntime()?.env?.wasm;
  if (!wasmEnv || !snapshot) {
    return;
  }
  if (typeof snapshot.proxy !== "undefined") {
    wasmEnv.proxy = snapshot.proxy;
  }
  if (typeof snapshot.numThreads !== "undefined") {
    wasmEnv.numThreads = snapshot.numThreads;
  }
}

function applyWasmEnv({ proxy, numThreads }) {
  const wasmEnv = getOrtRuntime()?.env?.wasm;
  if (!wasmEnv) {
    return;
  }
  if (typeof proxy === "boolean") {
    wasmEnv.proxy = proxy;
  }
  if (Number.isFinite(numThreads)) {
    wasmEnv.numThreads = Math.max(1, Math.floor(numThreads));
  }
}

function computeArgmaxIndices(data, B, F, T) {
  const result = new Int32Array(B * T);
  for (let b = 0; b < B; b += 1) {
    for (let t = 0; t < T; t += 1) {
      let best = -Infinity;
      let bestF = 0;
      for (let f = 0; f < F; f += 1) {
        const idx = b * F * T + f * T + t;
        const value = Number(data[idx] ?? Number.NEGATIVE_INFINITY);
        if (value > best) {
          best = value;
          bestF = f;
        }
      }
      result[b * T + t] = bestF;
    }
  }
  return result;
}

function collectVisibleArgmax(batchArgmax, chunk, B, T) {
  const visible = [];
  for (let b = 0; b < B && b < chunk.length; b += 1) {
    const current = chunk[b];
    if (!current) {
      continue;
    }
    const keepT = Math.max(0, Math.floor(Number(current.validT) || 0));
    for (let t = 0; t < keepT; t += 1) {
      const raw = Number(batchArgmax[b * T + t] ?? 0);
      visible.push(Number.isFinite(raw) ? Math.floor(raw) - 1 : -1);
    }
  }
  return visible;
}

function marginToConfidence(best, secondBest) {
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

function collectVisibleConfidenceFromScores(scoreData, chunk, B, F, T) {
  const visible = [];
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
        const value = Number(scoreData[idx] ?? Number.NEGATIVE_INFINITY);
        if (value > best) {
          second = best;
          best = value;
        } else if (value > second) {
          second = value;
        }
      }
      visible.push(marginToConfidence(best, second));
    }
  }
  return visible;
}

function collectBatchPredictions(outTensor, chunk) {
  const B = outTensor.dims[0] ?? 0;
  const F = outTensor.dims[1] ?? 0;
  const T = outTensor.dims[2] ?? 0;
  const outData = outTensor.data;

  return {
    batchArgmax: collectVisibleArgmax(
      computeArgmaxIndices(outData, B, F, T),
      chunk,
      B,
      T,
    ),
    batchConfidence: collectVisibleConfidenceFromScores(outData, chunk, B, F, T),
  };
}

function segmentBatchTo128(batch) {
  const data = batch && batch.data instanceof Float32Array ? batch.data : null;
  const shape = batch && batch.shape ? Array.from(batch.shape) : [];
  if (!data || shape.length < 3) {
    return [];
  }

  const c0 = Math.max(1, Math.floor(Number(shape[0]) || 1));
  const f0 = Math.max(1, Math.floor(Number(shape[1]) || 1));
  const t0 = Math.max(1, Math.floor(Number(shape[2]) || 1));
  const useC = Math.min(MODEL_IO.C, c0);
  const useF = Math.min(MODEL_IO.FREQ, f0);
  const segments = [];

  for (let startT = 0; startT < t0; startT += MODEL_IO.T_PER_BATCH) {
    const validT = Math.min(MODEL_IO.T_PER_BATCH, t0 - startT);
    const segData = new Float32Array(MODEL_IO.C * MODEL_IO.FREQ * MODEL_IO.T_PER_BATCH);
    for (let c = 0; c < useC; c += 1) {
      for (let f = 0; f < useF; f += 1) {
        const srcBase = c * f0 * t0 + f * t0 + startT;
        const dstBase = c * MODEL_IO.FREQ * MODEL_IO.T_PER_BATCH + f * MODEL_IO.T_PER_BATCH;
        for (let t = 0; t < validT; t += 1) {
          segData[dstBase + t] = data[srcBase + t];
        }
      }
    }
    segments.push({ data: segData, validT });
  }

  return segments;
}

function createResultFromSegments(inputName, batches) {
  const ort = getOrtRuntime();
  if (!session) {
    throw new Error("session not initialized");
  }
  if (!ort?.Tensor) {
    throw new Error("onnxruntime-web tensor constructor is unavailable");
  }

  const totalArgmax = [];
  const totalConfidence = [];
  let totalExpectedFrames = 0;
  let totalBatchCount = 0;
  const resolvedInputName = resolveSessionInputName(session) || inputName || "input";

  return (async () => {
    for (const batch of batches) {
      const batchShape = batch && batch.shape ? Array.from(batch.shape) : [];
      const batchFrames = Math.max(0, Math.floor(Number(batchShape[2]) || 0));
      totalExpectedFrames += batchFrames;
      const segments = segmentBatchTo128(batch);
      for (const segment of segments) {
        const feeds = {
          [resolvedInputName]: new ort.Tensor(
            "float32",
            segment.data,
            MODEL_IO.INPUT_SHAPE,
          ),
        };
        const out = await session.run(feeds);
        const outName = session.outputNames && session.outputNames[0];
        const outTensor = out[outName];
        if (!outTensor || !outTensor.data || outTensor.dims.length !== 3) {
          throw new Error(
            `模型输出异常，预期[${MODEL_IO.BATCH},361,${MODEL_IO.T_PER_BATCH}]，实际=${outTensor?.dims ? `[${outTensor.dims.join(",")}]` : "unknown"}`,
          );
        }
        const chunk = [{ validT: segment.validT }];
        const { batchArgmax, batchConfidence } = collectBatchPredictions(outTensor, chunk);
        totalArgmax.push(...batchArgmax);
        totalConfidence.push(...batchConfidence);
        totalBatchCount += 1;
      }
    }

    return {
      totalArgmax,
      totalConfidence,
      visibleArgmax: totalArgmax.slice(),
      visibleConfidence: totalConfidence.slice(),
      totalExpectedFrames,
      totalBatchCount,
    };
  })();
}

function getModelUrl(modelName) {
  return resolveModelUrl(normalizeCoreModelName(modelName));
}

async function createSessionWithProvider(modelUrl, provider, mode = "default") {
  const ort = getOrtRuntime();
  if (!ort?.InferenceSession?.create) {
    throw new Error("onnxruntime-web runtime is unavailable");
  }

  if (provider === "default") {
    const session = await ort.InferenceSession.create(modelUrl);
    return {
      session,
      provider: "default",
    };
  }

  const opts = {
    executionProviders: [provider],
    graphOptimizationLevel: "disabled",
  };
  if (provider === "webgpu") {
    const session = await ort.InferenceSession.create(modelUrl, opts);
    return {
      session,
      provider: "webgpu",
    };
  }

  if (provider === "wasm") {
    const session = await ort.InferenceSession.create(modelUrl, opts);
    return {
      session,
      provider: mode === "multi" ? "wasm-multi" : "wasm-single",
    };
  }

  throw new Error(`Unsupported provider: ${provider}`);
}

async function createPrioritySession(modelUrl) {
  try {
    await loadOrtRuntime(webgpuScriptUrl);
    try {
      return await createSessionWithProvider(modelUrl, "webgpu");
    } catch (error) {
      logWarn("WebGPU session init failed, falling back to wasm.", error);
    }
  } catch (error) {
    logWarn("failed to load WebGPU ORT bundle, falling back to wasm bundle.", error);
  }

  try {
    await loadOrtRuntime(wasmScriptUrl);
    const wasmEnvSnapshot = snapshotWasmEnv();
    const fallbackThreads = Math.max(1, globalThis.navigator?.hardwareConcurrency || 1);
    const attempts = [
      { proxy: true, numThreads: fallbackThreads, label: "multi" },
      { proxy: false, numThreads: 1, label: "single" },
    ];
    let lastError = null;
    try {
      for (const attempt of attempts) {
        applyWasmEnv(attempt);
        try {
          return await createSessionWithProvider(modelUrl, "wasm", attempt.label);
        } catch (error) {
          lastError = error;
          logWarn(
            `wasm session create failed mode=${attempt.label}, trying next provider fallback`,
            error,
          );
        }
      }
    } finally {
      restoreWasmEnv(wasmEnvSnapshot);
    }
    if (lastError) {
      throw lastError;
    }
  } catch (error) {
    logWarn("wasm provider init failed, falling back to default ORT session.", error);
  }

  await loadOrtRuntime(wasmScriptUrl);
  return await createSessionWithProvider(modelUrl, "default");
}

self.onmessage = async (ev) => {
  const m = ev.data || {};
  try {
    if (m.cmd === "init") {
      webgpuScriptUrl =
        typeof m.webgpuScriptUrl === "string" && m.webgpuScriptUrl
          ? m.webgpuScriptUrl
          : webgpuScriptUrl;
      wasmScriptUrl =
        typeof m.wasmScriptUrl === "string" && m.wasmScriptUrl
          ? m.wasmScriptUrl
          : wasmScriptUrl;
      ortWasmBaseUrl =
        typeof m.wasmAssetPath === "string" && m.wasmAssetPath
          ? m.wasmAssetPath
          : ortWasmBaseUrl;

      currentModelName = normalizeCoreModelName(m.modelName);
      currentResult = buildEmptyInferenceResult();
      const modelUrl = getModelUrl(m.modelName);
      const resolved = await createPrioritySession(modelUrl);
      session = resolved.session;
      providerUsed = resolved.provider;
      self.postMessage({ cmd: "inited", provider: providerUsed });
    } else if (m.cmd === "process") {
      if (!session) throw new Error("session not initialized");
      const id = m.id;
      const batches = Array.isArray(m.batches) ? m.batches : [];
      const inputName = resolveSessionInputName(session) || m.inputName || "input";
      if (batches.length) {
        logInfo(`worker process begin: id=${id} model=${currentModelName || "unknown"} batches=${batches.length}`);
        const result = await createResultFromSegments(inputName, batches);
        currentResult = result;
        logInfo(
          `worker process done: id=${id} model=${currentModelName || "unknown"} batches=${result.totalBatchCount} totalFrames=${result.totalExpectedFrames}`,
        );
        self.postMessage({ cmd: "result", id, result });
        return;
      }

      const ort = getOrtRuntime();
      if (!ort?.Tensor) {
        throw new Error("onnxruntime-web tensor constructor is unavailable");
      }
      const shape = Array.isArray(m.shape) && m.shape.length
        ? m.shape
        : MODEL_IO.INPUT_SHAPE;
      const buf = m.buffer;
      const floatData = new Float32Array(buf);
      const feeds = { [inputName]: new ort.Tensor("float32", floatData, shape) };
      const out = await session.run(feeds);
      const outName = session.outputNames && session.outputNames[0];
      const outTensor = out[outName];
      if (!outTensor || !outTensor.data || outTensor.dims.length !== 3) {
        throw new Error(
          `模型输出异常，预期[${MODEL_IO.BATCH},361,${MODEL_IO.T_PER_BATCH}]，实际=${outTensor?.dims ? `[${outTensor.dims.join(",")}]` : "unknown"}`,
        );
      }
      const validT = Math.max(1, Math.min(MODEL_IO.T_PER_BATCH, Number(shape?.[3] || MODEL_IO.T_PER_BATCH)));
      const { batchArgmax, batchConfidence } = collectBatchPredictions(outTensor, [{ validT }]);
      const result = {
        totalArgmax: batchArgmax.slice(),
        totalConfidence: batchConfidence.slice(),
        visibleArgmax: batchArgmax.slice(),
        visibleConfidence: batchConfidence.slice(),
        totalExpectedFrames: validT,
        totalBatchCount: 1,
      };
      currentResult = result;
      self.postMessage({ cmd: "result", id, result });
    }
  } catch (err) {
    self.postMessage({
      cmd: "error",
      error: err && err.message ? err.message : String(err),
      orig: m,
    });
  }
};
