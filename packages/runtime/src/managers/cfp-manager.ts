import type { CFPBatch } from "@ohm/core/cache/cache.js";
import {
  buildCFPAnalysisCacheKey,
  CFPIndexedDBCache,
} from "@ohm/core/cache/cache.js";
import { isAbortError, throwIfAborted } from "@ohm/core/abort/abort.js";
import { isPyodideOOMError, splitCFPRangeOnOOM } from "@ohm/core/cfp/cfp.js";
import { runCFPChunkInPyodide } from "@ohm/core/cfp/chunk.js";
import {
  type CFPBootstrapEnvironment,
  initializePyodideForCFP,
  type PyodideLike,
} from "@ohm/core/cfp/pyodide-bootstrap.js";
import { createCFPWorkerManager } from "@ohm/core/cfp/worker-manager.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import type {
  CFPChunkInput,
  CFPWorkerErrorMessage,
  CFPWorkerMessage,
  CFPWorkerResultMessage,
  CFPWorkerTimingMessage,
  WorkerLike,
} from "@ohm/core/cfp/types.js";
import type { AnalyzeExecutionOptions, AnalyzeInput } from "../types.js";

const cfpLogger = getModuleLogger("core.runtime.cfp");
const processLogger = getModuleLogger("core.runtime.cfp.process");

const PYODIDE_CDN_VERSION = "0.25.1";
const defaultPyodideScriptUrl = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/pyodide.js`;
const defaultPyodideIndexURL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/`;
const defaultCFPScriptUrl = new URL("../../../core/cfp.py", import.meta.url).toString();
const cfpCacheBackend = "runtime-v2";

type PyodideWorkerLike = PyodideLike & {
  toPy(value: Float32Array): { destroy?: () => void };
  globals: { set(name: string, value: unknown): void };
  runPython(code: string): unknown;
  runPythonAsync(code: string): Promise<unknown>;
  FS: PyodideLike["FS"] & {
    readFile(path: string): Uint8Array;
    unlink?(path: string): void;
  };
};

interface CFPProcessOptions {
  signal?: AbortSignal | null;
  pauseController?: {
    waitForResume?: (signal?: AbortSignal | null) => Promise<void> | void;
  } | null;
  workerInitTimeoutMs?: number;
  requireWorker?: boolean;
}

type PendingChunkPromise = {
  resolve: (value: CFPBatch) => void;
  reject: (reason?: unknown) => void;
};

type WorkerError = Error & { oom?: boolean };

function deriveMinChunkSamples(fs: number): number {
  return Math.max(1, Math.floor(Math.max(1, Number(fs) || 1) * 0.5));
}

async function processCFPInputRecursive({
  input,
  minChunkSamples,
  signal,
  waitIfPaused,
  processChunk,
}: {
  input: CFPChunkInput;
  minChunkSamples: number;
  signal?: AbortSignal | null;
  waitIfPaused: () => Promise<void>;
  processChunk: (input: CFPChunkInput) => Promise<CFPBatch>;
}): Promise<CFPBatch[]> {
  await waitIfPaused();
  throwIfAborted(signal);
  try {
    const batch = await processChunk(input);
    return [batch];
  } catch (error) {
    const oom = !!(error as WorkerError)?.oom || isPyodideOOMError(error);
    if (oom && input.pcm.length > minChunkSamples * 2) {
      const splitRange = splitCFPRangeOnOOM(0, input.pcm.length, minChunkSamples);
      if (splitRange) {
        processLogger.warn(
          `CFP OOM retry: [0,${input.pcm.length}) -> [${splitRange.left.start},${splitRange.left.end}) + [${splitRange.right.start},${splitRange.right.end})`,
        );
        const leftInput: CFPChunkInput = {
          pcm: input.pcm.slice(splitRange.left.start, splitRange.left.end),
          fs: input.fs,
        };
        const rightInput: CFPChunkInput = {
          pcm: input.pcm.slice(splitRange.right.start, splitRange.right.end),
          fs: input.fs,
        };
        const left = await processCFPInputRecursive({
          input: leftInput,
          minChunkSamples,
          signal: signal ?? null,
          waitIfPaused,
          processChunk,
        });
        const right = await processCFPInputRecursive({
          input: rightInput,
          minChunkSamples,
          signal: signal ?? null,
          waitIfPaused,
          processChunk,
        });
        return [...left, ...right];
      }
    }
    throw error;
  }
}

function createWorkerMessageHandler(
  pending: Map<string, PendingChunkPromise>,
): {
  handleTimingMessage: (message: CFPWorkerTimingMessage) => void;
  handleWorkerResultMessage: (message: CFPWorkerResultMessage) => void;
  handleWorkerErrorMessage: (message: CFPWorkerErrorMessage) => void;
  rejectAllPending: (error: unknown) => void;
} {
  function clearPending(id: string): PendingChunkPromise | null {
    const pendingPromise = pending.get(id);
    if (!pendingPromise) {
      return null;
    }
    pending.delete(id);
    return pendingPromise;
  }

  function resolvePendingResult(id: string, result: CFPBatch): void {
    const pendingPromise = clearPending(id);
    if (!pendingPromise) {
      return;
    }
    pendingPromise.resolve(result);
  }

  function rejectPendingResult(id: string, error: unknown): void {
    const pendingPromise = clearPending(id);
    if (!pendingPromise) {
      return;
    }
    pendingPromise.reject(error);
  }

  function rejectAllPending(error: unknown): void {
    if (!pending.size) {
      return;
    }
    for (const [id, pendingPromise] of pending.entries()) {
      pending.delete(id);
      pendingPromise.reject(error);
    }
  }

  function handleTimingMessage(message: CFPWorkerTimingMessage): void {
    const timingLine =
      typeof message.phase === "string" ? message.phase : "timing";
    if (timingLine) {
      processLogger.info(
        `${timingLine} [${message.start ?? "?"},${message.end ?? "?"})`,
      );
    }
  }

  function handleWorkerResultMessage(message: CFPWorkerResultMessage): void {
    const data = new Float32Array(message.dataBuf);
    const shape = new Int32Array(message.shapeBuf);
    const batch: CFPBatch = { data, shape };
    if (typeof message.id === "string" && message.id) {
      resolvePendingResult(message.id, batch);
      return;
    }
    rejectAllPending(new Error("CFP worker result missing id"));
  }

  function handleWorkerErrorMessage(message: CFPWorkerErrorMessage): void {
    const errObj = new Error(message.error || "cfp worker error") as WorkerError;
    errObj.oom = !!message.oom;
    if (typeof message.id === "string" && message.id) {
      rejectPendingResult(message.id, errObj);
      return;
    }
    rejectAllPending(errObj);
  }

  return {
    handleTimingMessage,
    handleWorkerResultMessage,
    handleWorkerErrorMessage,
    rejectAllPending,
  };
}

async function runCFPOnMainThread({
  input,
  pyodide,
  signal = null,
  pauseController = null,
}: {
  input: CFPChunkInput;
  pyodide: PyodideLike | null;
  signal?: AbortSignal | null;
  pauseController?: CFPProcessOptions["pauseController"];
}): Promise<CFPBatch[]> {
  if (!pyodide) {
    throw new Error("Pyodide is not initialized");
  }

  async function waitIfPaused(): Promise<void> {
    await pauseController?.waitForResume?.(signal);
  }

  return await processCFPInputRecursive({
    input,
    minChunkSamples: deriveMinChunkSamples(input.fs),
    signal,
    waitIfPaused,
    processChunk: async (segment) => {
      const result = await runCFPChunkInPyodide({
        pyodide,
        pcm: segment.pcm,
        fs: segment.fs,
        phase: "main-thread",
      });
      return {
        data: result.data,
        shape: result.shape,
      };
    },
  });
}

async function runCFPWithResidentWorker({
  input,
  worker,
  signal = null,
  pauseController = null,
}: {
  input: CFPChunkInput;
  worker: WorkerLike;
  signal?: AbortSignal | null;
  pauseController?: CFPProcessOptions["pauseController"];
}): Promise<CFPBatch[]> {
  const { pcm, fs } = input;
  processLogger.info(
    `CFP worker process begin: samples=${pcm.length} fs=${fs}`,
  );
  const pending = new Map<string, PendingChunkPromise>();
  const workerRef: { current: WorkerLike } = { current: worker };
  const handlers = createWorkerMessageHandler(pending);
  let nextId = 1;

  processLogger.info("CFP worker resident hit, using worker parallel processing.");

  async function processSegment(segment: CFPChunkInput): Promise<CFPBatch> {
    throwIfAborted(signal);
    const currentWorker = workerRef.current;
    if (!currentWorker) {
      throw new Error("CFP worker is not available");
    }
    const id = String(nextId++);
    return new Promise<CFPBatch>((resolve, reject) => {
      const arr = segment.pcm.slice();
      pending.set(id, { resolve, reject });
      try {
        currentWorker.postMessage(
          { cmd: "process", id, pcmBuffer: arr.buffer, fs: segment.fs },
          [arr.buffer],
        );
      } catch (error) {
        pending.delete(id);
        reject(error);
      }
    });
  }

  function attachWorkerHandlers(nextWorker: WorkerLike): void {
    if (!nextWorker) {
      return;
    }
    nextWorker.onmessage = (ev: MessageEvent<CFPWorkerMessage>) => {
      const message = ev.data || ({} as CFPWorkerMessage);
      if (message.cmd === "timing") {
        handlers.handleTimingMessage(message);
        return;
      }
      if (message.cmd === "result") {
        handlers.handleWorkerResultMessage(message);
        return;
      }
      if (message.cmd === "error") {
        handlers.handleWorkerErrorMessage(message);
      }
    };
    nextWorker.onerror = () => {
      handlers.rejectAllPending(new Error("CFP worker runtime error"));
    };
  }

  attachWorkerHandlers(worker);

  async function waitIfPaused(): Promise<void> {
    await pauseController?.waitForResume?.(signal);
  }

  try {
    return await processCFPInputRecursive({
      input: { pcm, fs },
      minChunkSamples: deriveMinChunkSamples(fs),
      signal: signal ?? null,
      waitIfPaused,
      processChunk: processSegment,
    });
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    processLogger.warn(
      `CFP worker processing interrupted: ${error instanceof Error ? error.message : String(error)}`,
    );
    throw error;
  }
}

function resolveAnalysisFileKey(input: AnalyzeInput): string {
  const sourceLabel = String(input.source.label || "").trim();
  return String(input.fileKey || sourceLabel || "analysis").trim();
}

export interface CFPManagerOptions {
  label?: string;
  pyodideScriptUrl?: string;
  pyodideIndexURL?: string;
  cfpScriptUrl?: string;
  createWorkerInstance?: (() => WorkerLike) | undefined;
}

export interface CFPCheckCacheInput {
  input: AnalyzeInput;
  execution: Readonly<AnalyzeExecutionOptions>;
}

export interface CFPCheckCacheResult {
  fileKey: string;
  batches: readonly CFPBatch[] | null;
}

export interface CFPProcessInput {
  input: AnalyzeInput;
  execution: Readonly<AnalyzeExecutionOptions>;
  previousBatches: readonly CFPBatch[];
  segment: CFPChunkInput;
  signal: AbortSignal | null;
  complete: boolean;
}

export interface CFPProcessResult {
  fileKey: string;
  batches: readonly CFPBatch[];
  allBatches: readonly CFPBatch[];
  complete: boolean;
}

export class CFPManager {
  private readonly analyzerLabel: string;
  private readonly pyodideScriptUrl: string;
  private readonly pyodideIndexURL: string;
  private readonly cfpScriptUrl: string;
  private readonly createWorkerInstance: (() => WorkerLike) | undefined;
  private readonly cache: CFPIndexedDBCache;
  private readonly workerManager: ReturnType<typeof createCFPWorkerManager>;
  private residentWorker: WorkerLike | null = null;
  private residentWorkerReadyPromise: Promise<WorkerLike | null> | null = null;
  private processQueue: Promise<void> = Promise.resolve();
  private readonly workerState: {
    pyodide: PyodideWorkerLike | null;
    ready: boolean;
  } = {
    pyodide: null,
    ready: false,
  };

  constructor({
    label = "runtime",
    pyodideScriptUrl = defaultPyodideScriptUrl,
    pyodideIndexURL = defaultPyodideIndexURL,
    cfpScriptUrl = defaultCFPScriptUrl,
    createWorkerInstance,
  }: CFPManagerOptions = {}) {
    this.analyzerLabel = String(label || "runtime").trim() || "runtime";
    this.pyodideScriptUrl = pyodideScriptUrl;
    this.pyodideIndexURL = pyodideIndexURL;
    this.cfpScriptUrl = cfpScriptUrl;
    this.createWorkerInstance = createWorkerInstance;
    cfpLogger.info(
      `runtime cfp manager ready: worker=${String(new URL("../../../core/dist/cfp/worker.js", import.meta.url))} pyodide=${this.pyodideScriptUrl} cfp=${this.cfpScriptUrl}`,
    );
    this.cache = new CFPIndexedDBCache({
      normalizeCFPBatches: normalizeCFPBatches,
    });
    this.workerManager = createCFPWorkerManager({
      disableWorker: typeof Worker === "undefined",
      createWorkerInstance: () => {
        if (this.createWorkerInstance) {
          try {
            return this.createWorkerInstance();
          } catch {
            return null;
          }
        }
        try {
          return new Worker(
            new URL("../../../core/dist/cfp/worker.js?worker", import.meta.url),
            { type: "module" },
          );
        } catch {
          return null;
        }
      },
      resolveCFPScriptUrl: () => this.cfpScriptUrl,
      resolvePyodideIndexURL: () => this.pyodideIndexURL,
      resolvePyodideScriptUrl: () => this.pyodideScriptUrl,
    });
    void this.workerManager.prewarmCFPWorker().then((ready) => {
      cfpLogger.info(
        ready
          ? `runtime cfp worker prewarm ready: ${String(new URL("../../../core/dist/cfp/worker.js", import.meta.url))}`
          : `runtime cfp worker prewarm skipped or unavailable: ${String(new URL("../../../core/dist/cfp/worker.js", import.meta.url))}`,
      );
    });
  }

  private enqueueProcess<T>(task: () => Promise<T>): Promise<T> {
    const next = this.processQueue.then(task);
    this.processQueue = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  async checkCache({
    input,
    execution,
  }: CFPCheckCacheInput): Promise<CFPCheckCacheResult> {
    const fileKey = resolveAnalysisFileKey(input);
    const cacheKey = buildCFPAnalysisCacheKey({
      namespace: this.analyzerLabel,
      fileKey,
      backend: cfpCacheBackend,
    });

    if (execution.allowCache !== false && !execution.forceRefresh && cacheKey) {
      const cached = await this.cache.getCFPCache(cacheKey, {
        allowPartial: false,
      });
      if (cached && cached.length) {
        cfpLogger.info(
          `runtime cfp cache hit: ${cacheKey}, batches=${cached.length}`,
        );
        return {
          fileKey,
          batches: cached,
        };
      }
    }

    cfpLogger.info(`runtime cfp keys ready: ${cacheKey}`);
    return {
      fileKey,
      batches: null,
    };
  }

  private async ensureResidentWorker(
    timeoutMs: number,
  ): Promise<WorkerLike | null> {
    if (this.residentWorker) {
      return this.residentWorker;
    }
    if (this.residentWorkerReadyPromise) {
      return await this.residentWorkerReadyPromise;
    }

    this.residentWorkerReadyPromise = (async () => {
      const prewarmPromise = this.workerManager.getWorkerPrewarmPromise();
      if (prewarmPromise) {
        try {
          await prewarmPromise;
        } catch {}
      }

      let worker = this.workerManager.getPrewarmedWorker();
      if (worker) {
        this.residentWorker = worker;
        cfpLogger.info(
          `runtime cfp resident worker ready: ${String(new URL("../../../core/dist/cfp/worker.js?worker", import.meta.url))}`,
        );
        return worker;
      }

      try {
        worker = this.createWorkerInstance
          ? this.createWorkerInstance()
          : new Worker(
              new URL("../../../core/dist/cfp/worker.js?worker", import.meta.url),
              { type: "module" },
            );
      } catch {
        return null;
      }

      let initFailureReason = "";
      const ready = await this.workerManager.waitWorkerInit(
        worker,
        timeoutMs,
        (reason) => {
          initFailureReason = reason;
        },
      );
      if (!ready) {
        cfpLogger.warn(
          `runtime cfp resident worker init failed: ${String(new URL("../../../core/dist/cfp/worker.js?worker", import.meta.url))}${initFailureReason ? ` reason=${initFailureReason}` : ""}`,
        );
        this.workerManager.terminateWorkerSafely(worker);
        return null;
      }

      this.residentWorker = worker;
      if (typeof this.workerManager.setPrewarmedWorker === "function") {
        this.workerManager.setPrewarmedWorker(worker);
      }
      cfpLogger.info(
        `runtime cfp resident worker ready: ${String(new URL("../../../core/dist/cfp/worker.js?worker", import.meta.url))}`,
      );
      return worker;
    })().finally(() => {
      this.residentWorkerReadyPromise = null;
    });

    return await this.residentWorkerReadyPromise;
  }

  async process({
    input,
    execution,
    previousBatches,
    segment,
    signal,
    complete,
  }: CFPProcessInput): Promise<CFPProcessResult> {
    return await this.enqueueProcess(async () => {
      const fileKey = resolveAnalysisFileKey(input);
      const cacheKey = buildCFPAnalysisCacheKey({
        namespace: this.analyzerLabel,
        fileKey,
        backend: cfpCacheBackend,
      });

      if (!previousBatches.length) {
        const cacheState = await this.checkCache({
          input,
          execution,
        });
        if (cacheState.batches?.length) {
          return {
            fileKey: cacheState.fileKey,
            batches: cacheState.batches,
            allBatches: cacheState.batches,
            complete: true,
          };
        }
      }

      const batches = await this.processCFP({
        input: segment,
        signal,
      });
      const allBatches = [...previousBatches, ...batches];
      await this.commitCache({
        cacheKey,
        batches: allBatches,
        allowCache: execution.allowCache !== false,
        complete,
      });

      return {
        fileKey,
        batches,
        allBatches,
        complete: false,
      };
    });
  }

  async processCFP({
    input,
    signal,
  }: {
    input: CFPChunkInput;
    signal: AbortSignal | null;
  }): Promise<readonly CFPBatch[]> {
    const canUseWorker = typeof Worker !== "undefined";
    cfpLogger.info(
      `runtime cfp process start: worker=${canUseWorker} samples=${input.pcm.length} fs=${input.fs}`,
    );
    if (canUseWorker) {
      const worker = await this.ensureResidentWorker(30000);
      if (worker) {
        const workerResults = await runCFPWithResidentWorker({
          input,
          worker,
          signal,
        });
        cfpLogger.info(
          `runtime cfp process done via worker: batches=${workerResults.length}`,
        );
        return workerResults;
      }
      cfpLogger.warn(
        "runtime cfp worker unavailable, falling back to main-thread Pyodide",
      );
    }

    cfpLogger.warn("runtime cfp falling back to main-thread Pyodide");
    const pyodide = await this.ensurePyodide();
    const result = await runCFPOnMainThread({
      input,
      pyodide,
      signal,
    });
    cfpLogger.info(
      `runtime cfp process done via main-thread: batches=${result.length}`,
    );
    return result;
  }

  async commitCache({
    cacheKey,
    batches,
    allowCache,
    complete,
  }: {
    cacheKey: string;
    batches: readonly CFPBatch[];
    allowCache: boolean;
    complete: boolean;
  }): Promise<void> {
    if (!allowCache) {
      return;
    }
    if (!cacheKey || !batches.length) {
      return;
    }
    try {
      for (let index = 0; index < batches.length; index += 1) {
        const batch = batches[index];
        if (!batch) {
          continue;
        }
        await this.cache.appendCFPCacheChunk(cacheKey, batch, {
          index,
          reset: index === 0,
          complete: complete === true && index === batches.length - 1,
          expectedChunkCount: batches.length,
        });
      }
      if (complete) {
        await this.cache.finalizeCFPCache(cacheKey, {
          chunkCount: batches.length,
        });
      }
      cfpLogger.info(
        `runtime cfp cache written: ${cacheKey}, batches=${batches.length}, complete=${complete ? "true" : "false"}`,
      );
    } catch (error) {
      cfpLogger.warn(
        `runtime cfp cache write skipped: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  private async ensurePyodide(): Promise<PyodideWorkerLike> {
    if (this.workerState.ready && this.workerState.pyodide) {
      return this.workerState.pyodide;
    }

    cfpLogger.info(
      `runtime cfp loading pyodide on main-thread: script=${this.pyodideScriptUrl} index=${this.pyodideIndexURL}`,
    );
    const browserPyodide = await loadBrowserPyodideRuntime(
      this.pyodideScriptUrl,
      this.pyodideIndexURL,
    );

    this.workerState.pyodide = (await initializePyodideForCFP({
      pyodideScriptUrl: this.pyodideScriptUrl,
      pyodideIndexURL: this.pyodideIndexURL,
      cfpScriptUrl: this.cfpScriptUrl,
      packages: ["numpy", "scipy"],
      environment: {
        loadScript: async () => true,
        loadPyodide: async () => browserPyodide,
      } satisfies CFPBootstrapEnvironment,
    })) as PyodideWorkerLike;
    this.workerState.ready = true;
    cfpLogger.info("runtime cfp pyodide ready on main-thread");
    return this.workerState.pyodide;
  }
}

function normalizeCFPBatches(rawBatches: unknown): CFPBatch[] {
  if (!Array.isArray(rawBatches)) {
    return [];
  }
  const normalized: CFPBatch[] = [];
  for (const item of rawBatches) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const typed = item as { data?: unknown; shape?: unknown };
    if (
      !(typed.data instanceof Float32Array) ||
      !(typed.shape instanceof Int32Array)
    ) {
      continue;
    }
    normalized.push({
      data: typed.data,
      shape: typed.shape,
    });
  }
  return normalized;
}

async function loadBrowserPyodideRuntime(
  pyodideScriptUrl: string,
  indexURL: string,
): Promise<PyodideLike> {
  const runtimeGlobal = globalThis as typeof globalThis & {
    loadPyodide?: (options: { indexURL: string }) => Promise<PyodideLike>;
  };

  if (typeof runtimeGlobal.loadPyodide !== "function") {
    if (typeof document === "undefined") {
      throw new Error("Pyodide is not available in current environment");
    }

    await new Promise<void>((resolve, reject) => {
      const existingScript = document.querySelector<HTMLScriptElement>(
        'script[data-ohm-pyodide="true"]',
      );
      const finish = () => resolve();
      const fail = () => reject(new Error("failed to load pyodide runtime"));

      if (existingScript) {
        if (typeof runtimeGlobal.loadPyodide === "function") {
          resolve();
          return;
        }
        existingScript.addEventListener("load", finish, { once: true });
        existingScript.addEventListener("error", fail, { once: true });
        return;
      }

      const script = document.createElement("script");
      script.dataset.ohmPyodide = "true";
      script.src = pyodideScriptUrl;
      script.async = true;
      script.onload = finish;
      script.onerror = fail;
      document.head.appendChild(script);
    });
  }

  const loadPyodide = runtimeGlobal.loadPyodide?.bind(runtimeGlobal);
  if (!loadPyodide) {
    throw new Error("Pyodide is not available in current environment");
  }
  return await loadPyodide({ indexURL });
}
