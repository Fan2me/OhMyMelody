import type { CFPBatch } from "@ohm/core/cache/cache.js";
import {
  buildPredictionCacheKey,
  CFPIndexedDBCache,
  normalizePredictionCacheEntry,
  type PredictionCacheEntry,
} from "@ohm/core/cache/cache.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import {
  normalizeCoreModelName,
  resolveCoreModelUrl,
} from "@ohm/core/model-catalog.js";
import type { WorkerLike } from "@ohm/core/cfp/types.js";
import type { InferenceProgress, InferenceResult } from "../analysis.js";

const inferenceLogger = getModuleLogger("core.runtime.inference");
const defaultWorkerScriptUrl = new URL(
  "../../../core/dist/inference/worker.js",
  import.meta.url,
).toString();

type InferenceWorkerInitMessage = {
  cmd: "init";
  modelName: string;
  modelUrl: string;
};

type InferenceWorkerResetMessage = {
  cmd: "reset";
};

type InferenceWorkerProcessMessage = {
  cmd: "process";
  id: string;
  modelName: string;
  batches: readonly CFPBatch[];
};

type InferenceWorkerResultMessage = {
  cmd: "result";
  id: string;
  result: InferenceResult;
};

type InferenceWorkerErrorMessage = {
  cmd: "error";
  id?: string;
  error: string;
};

type InferenceWorkerInitedMessage = {
  cmd: "inited";
  provider?: string;
};

type InferenceWorkerMessage =
  | InferenceWorkerInitMessage
  | InferenceWorkerResetMessage
  | InferenceWorkerProcessMessage
  | InferenceWorkerResultMessage
  | InferenceWorkerErrorMessage
  | InferenceWorkerInitedMessage;

type PendingPromise = {
  resolve: (value: InferenceResult) => void;
  reject: (reason?: unknown) => void;
};

function buildEmptyInferenceResult(): InferenceResult {
  return {
    totalArgmax: [],
    totalConfidence: [],
    visibleArgmax: [],
    visibleConfidence: [],
    totalExpectedFrames: 0,
    totalBatchCount: 0,
  };
}

function mergeInferenceResults(
  previous: InferenceResult,
  next: InferenceResult,
): InferenceResult {
  return {
    totalArgmax: [...(previous.totalArgmax || []), ...(next.totalArgmax || [])],
    totalConfidence: [...(previous.totalConfidence || []), ...(next.totalConfidence || [])],
    visibleArgmax: [...(previous.visibleArgmax || []), ...(next.visibleArgmax || [])],
    visibleConfidence: [...(previous.visibleConfidence || []), ...(next.visibleConfidence || [])],
    totalExpectedFrames:
      Math.max(0, Math.floor(Number(previous.totalExpectedFrames) || 0)) +
      Math.max(0, Math.floor(Number(next.totalExpectedFrames) || 0)),
    totalBatchCount:
      Math.max(0, Math.floor(Number(previous.totalBatchCount) || 0)) +
      Math.max(0, Math.floor(Number(next.totalBatchCount) || 0)),
  };
}

export interface InferenceManagerOptions {
  onProgress?: ((progress: InferenceProgress) => void) | null;
  workerModuleUrl?: string | URL;
}

export class InferenceManager {
  private readonly workerModuleUrl: string | URL;
  private readonly options: InferenceManagerOptions;
  private readonly cache: CFPIndexedDBCache;
  private residentWorker: WorkerLike | null = null;
  private workerReadyPromise: Promise<WorkerLike | null> | null = null;
  private workerInitialized = false;
  private currentModelName = "";
  private currentProvider = "";
  private sessionModelName = "";
  private sessionId = 0;
  private useCachedResult = false;
  private result: InferenceResult = buildEmptyInferenceResult();
  private pending = new Map<string, PendingPromise>();
  private nextId = 1;
  private processQueue: Promise<void> = Promise.resolve();
  private readonly workerInitTimeoutMs = 30000;

  constructor(options: InferenceManagerOptions = {}) {
    this.options = options;
    this.workerModuleUrl = options.workerModuleUrl ?? defaultWorkerScriptUrl;
    this.cache = new CFPIndexedDBCache({
      normalizeCFPBatches: () => [],
    });
  }

  private buildInferenceCacheKey(fileKey: string, modelName: string): string {
    return buildPredictionCacheKey({
      fileKey,
      modelName: normalizeCoreModelName(modelName),
      backend: "runtime-inference-v1",
    });
  }

  private toInferenceResultFromCache(entry: PredictionCacheEntry, totalBatchCount: number): InferenceResult {
    const totalArgmax = Array.from(entry.totalArgmax || []);
    const totalConfidence = Array.from(entry.totalConfidence || []);
    const visibleArgmax = Array.from(entry.visibleArgmax || []);
    const visibleConfidence = Array.from(entry.visibleConfidence || []);
    return {
      totalArgmax,
      totalConfidence,
      visibleArgmax,
      visibleConfidence,
      totalExpectedFrames:
        Number.isFinite(Number(entry.totalExpectedFrames)) && Number(entry.totalExpectedFrames) > 0
          ? Math.floor(Number(entry.totalExpectedFrames))
          : Math.max(totalArgmax.length, visibleArgmax.length),
      totalBatchCount:
        Number.isFinite(Number(entry.totalBatchCount)) && Number(entry.totalBatchCount) > 0
          ? Math.floor(Number(entry.totalBatchCount))
          : totalBatchCount,
    };
  }

  private isCacheEntryUsable(entry: PredictionCacheEntry, expectedBatchCount = 0): boolean {
    if (!entry || entry.complete !== true) {
      return false;
    }
    const totalFrames =
      Number.isFinite(Number(entry.totalExpectedFrames)) && Number(entry.totalExpectedFrames) > 0
        ? Math.floor(Number(entry.totalExpectedFrames))
        : Math.max(entry.totalArgmax.length, entry.visibleArgmax.length);
    const batchCount =
      Number.isFinite(Number(entry.totalBatchCount)) && Number(entry.totalBatchCount) > 0
        ? Math.floor(Number(entry.totalBatchCount))
        : 0;
    if (totalFrames <= 0) {
      return false;
    }
    if (entry.totalConfidence.length < entry.totalArgmax.length) {
      return false;
    }
    if (entry.visibleConfidence.length < entry.visibleArgmax.length) {
      return false;
    }
    if (expectedBatchCount > 0 && batchCount > 0 && batchCount < expectedBatchCount) {
      return false;
    }
    return true;
  }

  private toPredictionCacheEntry(result: InferenceResult): PredictionCacheEntry {
    return {
      totalArgmax: Int32Array.from(Array.isArray(result.totalArgmax) ? result.totalArgmax : []),
      visibleArgmax: Int32Array.from(Array.isArray(result.visibleArgmax) ? result.visibleArgmax : []),
      totalConfidence: Float32Array.from(Array.isArray(result.totalConfidence) ? result.totalConfidence : []),
      visibleConfidence: Float32Array.from(Array.isArray(result.visibleConfidence) ? result.visibleConfidence : []),
      totalExpectedFrames: Math.max(0, Math.floor(Number(result.totalExpectedFrames) || 0)),
      totalBatchCount: Math.max(0, Math.floor(Number(result.totalBatchCount) || 0)),
      complete: true,
    };
  }

  async hasCache(fileKey: string, modelName: string, expectedBatchCount = 0): Promise<boolean> {
    const safeFileKey = String(fileKey || "").trim();
    if (!safeFileKey) {
      return false;
    }
    const cacheKey = this.buildInferenceCacheKey(safeFileKey, modelName);
    try {
      const cachedRaw = await this.cache.getPredictionCache(cacheKey);
      const cached = normalizePredictionCacheEntry(cachedRaw);
      return !!cached && this.isCacheEntryUsable(cached, expectedBatchCount);
    } catch {
      return false;
    }
  }

  private createWorkerInstance(): WorkerLike {
    if (typeof Worker === "undefined") {
      return null;
    }
    try {
      const workerUrl = new URL(String(this.workerModuleUrl), import.meta.url);
      return new Worker(workerUrl, { type: "module" });
    } catch {
      return null;
    }
  }

  private attachWorkerHandlers(worker: WorkerLike): void {
    if (!worker) {
      return;
    }
    worker.onmessage = (ev: MessageEvent<InferenceWorkerMessage>) => {
      const message = ev.data || ({} as InferenceWorkerMessage);
      if (message.cmd === "inited") {
        this.currentProvider = message.provider || this.currentProvider || "unknown";
        inferenceLogger.info(
          `runtime inference worker inited: model=${this.currentModelName || "unknown"} provider=${this.currentProvider || "unknown"}`,
        );
        this.workerInitialized = true;
        return;
      }
      if (message.cmd === "result") {
        const pending = this.pending.get(message.id);
        if (pending) {
          this.pending.delete(message.id);
          pending.resolve(message.result);
        }
        return;
      }
      if (message.cmd === "error") {
        if (typeof message.id === "string" && message.id) {
          const pending = this.pending.get(message.id);
          if (pending) {
            this.pending.delete(message.id);
            pending.reject(new Error(message.error));
          }
          return;
        }
        for (const [id, pending] of this.pending.entries()) {
          this.pending.delete(id);
          pending.reject(new Error(message.error));
        }
      }
    };
    worker.onerror = () => {
      for (const [id, pending] of this.pending.entries()) {
        this.pending.delete(id);
        pending.reject(new Error("Inference worker runtime error"));
      }
      this.workerInitialized = false;
    };
  }

  private async ensureWorker(): Promise<WorkerLike | null> {
    if (this.residentWorker) {
      return this.residentWorker;
    }
    if (this.workerReadyPromise) {
      return await this.workerReadyPromise;
    }

    this.workerReadyPromise = (async () => {
      const worker = this.createWorkerInstance();
      if (!worker) {
        return null;
      }
      this.attachWorkerHandlers(worker);
      this.residentWorker = worker;
      inferenceLogger.info(
        `runtime inference worker created: ${String(this.workerModuleUrl)}`,
      );
      return worker;
    })().finally(() => {
      this.workerReadyPromise = null;
    });

    return await this.workerReadyPromise;
  }

  private async ensureInitialized(modelName: string): Promise<WorkerLike | null> {
    const worker = await this.ensureWorker();
    if (!worker) {
      return null;
    }

    const safeModelName = normalizeCoreModelName(modelName);
    if (this.workerInitialized && this.currentModelName === safeModelName) {
      inferenceLogger.info(
        `runtime inference worker reuse initialized model=${safeModelName}`,
      );
      return worker;
    }

    inferenceLogger.info(
      `runtime inference worker init begin: model=${safeModelName} timeoutMs=${this.workerInitTimeoutMs}`,
    );
    const modelUrl = resolveCoreModelUrl(safeModelName);
    const initMessage: InferenceWorkerInitMessage = {
      cmd: "init",
      modelName: safeModelName,
      modelUrl,
    };
    await new Promise<void>((resolve, reject) => {
      let settled = false;
      let timeout: ReturnType<typeof setTimeout> | null = null;
      const finish = (err?: unknown) => {
        if (settled) {
          return;
        }
        settled = true;
        if (timeout) {
          clearTimeout(timeout);
          timeout = null;
        }
        worker?.removeEventListener("message", onMessage);
        worker?.removeEventListener("error", onError);
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      };
      const onMessage = (ev: MessageEvent<InferenceWorkerMessage>) => {
        const message = ev.data || ({} as InferenceWorkerMessage);
        if (message.cmd === "inited") {
          finish();
        }
        if (message.cmd === "error") {
          finish(new Error(message.error));
        }
      };
      const onError = () => {
        finish(new Error("Inference worker init failed"));
      };
      timeout = setTimeout(
        () => finish(new Error("Inference worker init timed out")),
        this.workerInitTimeoutMs,
      );
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", onError);
      try {
        worker.postMessage(initMessage);
      } catch (error) {
        finish(error);
      }
    });

    this.currentModelName = safeModelName;
    this.workerInitialized = true;
    inferenceLogger.info(
      `runtime inference worker init done: model=${safeModelName}`,
    );
    return worker;
  }

  private async sendReset(): Promise<void> {
    const worker = await this.ensureWorker();
    if (!worker) {
      return;
    }
    inferenceLogger.info("runtime inference worker reset");
    worker.postMessage({ cmd: "reset" } satisfies InferenceWorkerResetMessage);
  }

  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    const next = this.processQueue.then(task);
    this.processQueue = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  private async startNewSession(modelName: string, reason: string): Promise<void> {
    const previousModel = this.sessionModelName || "none";
    this.sessionId += 1;
    this.sessionModelName = modelName;
    this.useCachedResult = false;
    this.result = buildEmptyInferenceResult();
    await this.sendReset();
    inferenceLogger.info(
      `runtime inference session started: session=${this.sessionId} reason=${reason} prevModel=${previousModel} model=${modelName}`,
    );
  }

  async process({
    batches,
    modelName,
    fileKey,
    allowCache = true,
    forceRefresh = false,
    complete = false,
  }: {
    batches: readonly CFPBatch[];
    modelName: string;
    fileKey?: string;
    allowCache?: boolean;
    forceRefresh?: boolean;
    complete?: boolean;
  }): Promise<InferenceResult> {
    return await this.enqueue(async () => {
      const safeModelName = normalizeCoreModelName(modelName);
      if (!this.sessionModelName) {
        await this.startNewSession(safeModelName, "initial");
      } else if (this.sessionModelName !== safeModelName) {
        await this.startNewSession(safeModelName, "model-changed");
      }
      inferenceLogger.info(
        `runtime inference process start: session=${this.sessionId} model=${safeModelName} provider=${this.currentProvider || "unknown"} batches=${batches.length}`,
      );
      if (!batches.length) {
        return this.result;
      }

      const canUseCache =
        allowCache !== false &&
        typeof fileKey === "string" &&
        fileKey.trim().length > 0;
      const cacheKey = canUseCache ? this.buildInferenceCacheKey(String(fileKey), safeModelName) : "";
      if (this.useCachedResult && this.result.totalExpectedFrames > 0) {
        return this.result;
      }
      if (canUseCache && cacheKey && forceRefresh !== true && this.result.totalExpectedFrames <= 0) {
        try {
          const cachedRaw = await this.cache.getPredictionCache(cacheKey);
          const cached = normalizePredictionCacheEntry(cachedRaw);
          if (cached && this.isCacheEntryUsable(cached, batches.length)) {
            const hit = this.toInferenceResultFromCache(cached, batches.length);
            this.result = hit;
            this.useCachedResult = true;
            inferenceLogger.info(
              `runtime inference cache hit: key=${cacheKey} totalFrames=${hit.totalExpectedFrames}`,
            );
            return hit;
          }
          if (cached && !this.isCacheEntryUsable(cached, batches.length)) {
            inferenceLogger.warn(
              `runtime inference cache ignored: key=${cacheKey} reason=incomplete-or-mismatch`,
            );
          }
        } catch (error) {
          inferenceLogger.warn(
            `runtime inference cache read failed: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      }

      const worker = await this.ensureInitialized(modelName);
      if (!worker) {
        throw new Error("Inference worker is not available");
      }

      const id = String(this.nextId++);
      inferenceLogger.info(
        `runtime inference worker process begin: session=${this.sessionId} id=${id} model=${safeModelName} provider=${this.currentProvider || "unknown"} batches=${batches.length}`,
      );
      const resultPromise = new Promise<InferenceResult>((resolve, reject) => {
        this.pending.set(id, { resolve, reject });
      });

      worker.postMessage({
        cmd: "process",
        id,
        modelName: safeModelName,
        batches,
      } satisfies InferenceWorkerProcessMessage);

      const result = await resultPromise;
      this.useCachedResult = false;
      this.result = mergeInferenceResults(this.result, result);
      const mergedResult = this.result;
      if (canUseCache && cacheKey && complete === true) {
        try {
          await this.cache.setPredictionCache(cacheKey, this.toPredictionCacheEntry(mergedResult));
          inferenceLogger.info(
            `runtime inference cache written: key=${cacheKey} totalFrames=${mergedResult.totalExpectedFrames}`,
          );
        } catch (error) {
          inferenceLogger.warn(
            `runtime inference cache write failed: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      }
      inferenceLogger.info(
        `runtime inference worker process done: session=${this.sessionId} id=${id} model=${safeModelName} provider=${this.currentProvider || "unknown"} batches=${mergedResult.totalBatchCount} totalFrames=${mergedResult.totalExpectedFrames} visible=${mergedResult.visibleArgmax.length}`,
      );
      return mergedResult;
    });
  }

  buildEmptyResult(): InferenceResult {
    return buildEmptyInferenceResult();
  }

  reset(): void {
    void this.enqueue(async () => {
      this.sessionModelName = "";
      this.sessionId += 1;
      this.useCachedResult = false;
      this.result = buildEmptyInferenceResult();
      await this.sendReset();
    });
  }
}
