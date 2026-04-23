import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import {
  normalizePredictionCacheEntry,
  PredictionIndexedDBCache,
  type PredictionCacheEntry,
} from "@ohm/core/cache/prediction.js";
import { CORE_INFERENCE_WORKER_MODULE_URL } from "@ohm/core/inference/inference.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import {
  normalizeCoreModelName,
  resolveCoreModelUrl,
} from "@ohm/core/model-catalog.js";
import type { WorkerLike } from "@ohm/core/cfp/types.js";
import type { InferenceProgress, InferenceResult } from "../analysis.js";

const inferenceLogger = getModuleLogger("core.runtime.inference");
const defaultWorkerScriptUrl = CORE_INFERENCE_WORKER_MODULE_URL;

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

type WorkerState = {
  worker: WorkerLike | null;
  readyPromise: Promise<WorkerLike | null> | null;
  initializedModelName: string;
  provider: string;
};

type SessionState = {
  modelName: string;
  id: number;
  useCachedResult: boolean;
  result: InferenceResult;
};

type InferenceProcessInput = {
  batches: readonly CFPBatch[];
  modelName: string;
  fileKey?: string;
  allowCache?: boolean;
  forceRefresh?: boolean;
  complete?: boolean;
};

function buildEmptyInferenceResult(): InferenceResult {
  return {
    totalArgmax: [],
    totalConfidence: [],
  };
}

function mergeInferenceResults(
  previous: InferenceResult,
  next: InferenceResult,
): InferenceResult {
  return {
    totalArgmax: [...(previous.totalArgmax || []), ...(next.totalArgmax || [])],
    totalConfidence: [...(previous.totalConfidence || []), ...(next.totalConfidence || [])],
  };
}

function buildInferenceCacheKey(
  fileKey: string,
  modelName: string,
  backend = "runtime-inference-v3",
): string {
  return [
    String(fileKey || "").trim(),
    normalizeCoreModelName(modelName),
    String(backend || "").trim(),
  ].join("::");
}

export interface InferenceManagerOptions {
  onProgress?: ((progress: InferenceProgress) => void) | null;
  workerModuleUrl?: string | URL;
}

export class InferenceManager {
  private readonly workerModuleUrl: string | URL;
  private readonly options: InferenceManagerOptions;
  private readonly cache: PredictionIndexedDBCache;
  private readonly workerState: WorkerState = {
    worker: null,
    readyPromise: null,
    initializedModelName: "",
    provider: "",
  };
  private readonly sessionState: SessionState = {
    modelName: "",
    id: 0,
    useCachedResult: false,
    result: buildEmptyInferenceResult(),
  };
  private pending = new Map<string, PendingPromise>();
  private nextId = 1;
  private processQueue: Promise<void> = Promise.resolve();
  private readonly workerInitTimeoutMs = 30000;

  constructor(options: InferenceManagerOptions = {}) {
    this.options = options;
    this.workerModuleUrl = options.workerModuleUrl ?? defaultWorkerScriptUrl;
    this.cache = new PredictionIndexedDBCache();
  }

  private get sessionLabel(): string {
    return `session=${this.sessionState.id}`;
  }

  private buildInferenceCacheKey(fileKey: string, modelName: string): string {
    return buildInferenceCacheKey(fileKey, modelName);
  }

  private toInferenceResultFromCache(entry: PredictionCacheEntry): InferenceResult {
    const totalArgmax = Array.from(entry.totalArgmax || []);
    const totalConfidence = Array.from(entry.totalConfidence || []);
    return {
      totalArgmax,
      totalConfidence,
    };
  }

  private isCacheEntryUsable(entry: PredictionCacheEntry): boolean {
    if (!entry) {
      return false;
    }
    if (entry.totalArgmax.length <= 0) {
      return false;
    }
    if (entry.totalConfidence.length < entry.totalArgmax.length) {
      return false;
    }
    return true;
  }

  private toPredictionCacheEntry(result: InferenceResult): PredictionCacheEntry {
    return {
      totalArgmax: Int32Array.from(Array.isArray(result.totalArgmax) ? result.totalArgmax : []),
      totalConfidence: Float32Array.from(Array.isArray(result.totalConfidence) ? result.totalConfidence : []),
    };
  }

  async hasCache(fileKey: string, modelName: string): Promise<boolean> {
    const safeFileKey = String(fileKey || "").trim();
    if (!safeFileKey) {
      return false;
    }
    const cacheKey = this.buildInferenceCacheKey(safeFileKey, modelName);
    try {
      const cachedRaw = await this.cache.getPredictionCache(cacheKey);
      const cached = normalizePredictionCacheEntry(cachedRaw);
      return !!cached && this.isCacheEntryUsable(cached);
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
        this.workerState.provider =
          message.provider || this.workerState.provider || "unknown";
        inferenceLogger.info(
          `runtime inference worker inited: model=${this.workerState.initializedModelName || "unknown"} provider=${this.workerState.provider || "unknown"}`,
        );
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
      this.rejectAllPending(new Error("Inference worker runtime error"));
      this.workerState.initializedModelName = "";
    };
  }

  private rejectAllPending(error: unknown): void {
    for (const [id, pending] of this.pending.entries()) {
      this.pending.delete(id);
      pending.reject(error);
    }
  }

  private async ensureWorker(): Promise<WorkerLike | null> {
    if (this.workerState.worker) {
      return this.workerState.worker;
    }
    if (this.workerState.readyPromise) {
      return await this.workerState.readyPromise;
    }

    this.workerState.readyPromise = (async () => {
      const worker = this.createWorkerInstance();
      if (!worker) {
        return null;
      }
      this.attachWorkerHandlers(worker);
      this.workerState.worker = worker;
      inferenceLogger.info(
        `runtime inference worker created: ${String(this.workerModuleUrl)}`,
      );
      return worker;
    })().finally(() => {
      this.workerState.readyPromise = null;
    });

    return await this.workerState.readyPromise;
  }

  private async ensureInitialized(modelName: string): Promise<WorkerLike | null> {
    const worker = await this.ensureWorker();
    if (!worker) {
      return null;
    }

    const safeModelName = normalizeCoreModelName(modelName);
    if (this.workerState.initializedModelName === safeModelName) {
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

    this.workerState.initializedModelName = safeModelName;
    inferenceLogger.info(
      `runtime inference worker init done: model=${safeModelName}`,
    );
    return worker;
  }

  private async postWorkerRequest(
    worker: NonNullable<WorkerLike>,
    message: InferenceWorkerProcessMessage,
  ): Promise<InferenceResult> {
    return await new Promise<InferenceResult>((resolve, reject) => {
      this.pending.set(message.id, { resolve, reject });
      try {
        worker.postMessage(message);
      } catch (error) {
        this.pending.delete(message.id);
        reject(error);
      }
    });
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

  private resetSessionState(nextModelName = ""): void {
    this.sessionState.modelName = nextModelName;
    this.sessionState.id += 1;
    this.sessionState.useCachedResult = false;
    this.sessionState.result = buildEmptyInferenceResult();
  }

  private async startNewSession(modelName: string, reason: string): Promise<void> {
    const previousModel = this.sessionState.modelName || "none";
    this.resetSessionState(modelName);
    await this.sendReset();
    inferenceLogger.info(
      `runtime inference session started: session=${this.sessionState.id} reason=${reason} prevModel=${previousModel} model=${modelName}`,
    );
  }

  private async ensureSession(modelName: string): Promise<string> {
    const safeModelName = normalizeCoreModelName(modelName);
    if (!this.sessionState.modelName) {
      await this.startNewSession(safeModelName, "initial");
    } else if (this.sessionState.modelName !== safeModelName) {
      await this.startNewSession(safeModelName, "model-changed");
    }
    return safeModelName;
  }

  private resolveCacheKey(
    fileKey: string | undefined,
    modelName: string,
    allowCache: boolean,
  ): string {
    if (allowCache === false || typeof fileKey !== "string" || !fileKey.trim()) {
      return "";
    }
    return this.buildInferenceCacheKey(String(fileKey), modelName);
  }

  private async tryReadCachedResult(
    cacheKey: string,
    forceRefresh: boolean,
  ): Promise<InferenceResult | null> {
    if (
      !cacheKey ||
      forceRefresh === true ||
      this.sessionState.result.totalArgmax.length > 0
    ) {
      return null;
    }
    if (
      this.sessionState.useCachedResult &&
      this.sessionState.result.totalArgmax.length > 0
    ) {
      return this.sessionState.result;
    }

    try {
      const cachedRaw = await this.cache.getPredictionCache(cacheKey);
      const cached = normalizePredictionCacheEntry(cachedRaw);
      if (cached && this.isCacheEntryUsable(cached)) {
        const hit = this.toInferenceResultFromCache(cached);
        this.sessionState.result = hit;
        this.sessionState.useCachedResult = true;
        inferenceLogger.info(`runtime inference cache hit: key=${cacheKey}`);
        return hit;
      }
      if (cached && !this.isCacheEntryUsable(cached)) {
        inferenceLogger.warn(
          `runtime inference cache ignored: key=${cacheKey} reason=invalid-cache-entry`,
        );
      }
    } catch (error) {
      inferenceLogger.warn(
        `runtime inference cache read failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    return null;
  }

  private async writeCachedResult(
    cacheKey: string,
    complete: boolean,
    result: InferenceResult,
  ): Promise<void> {
    if (!cacheKey || complete !== true) {
      return;
    }
    try {
      await this.cache.setPredictionCache(
        cacheKey,
        this.toPredictionCacheEntry(result),
      );
      inferenceLogger.info(`runtime inference cache written: key=${cacheKey}`);
    } catch (error) {
      inferenceLogger.warn(
        `runtime inference cache write failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async process({
    batches,
    modelName,
    fileKey,
    allowCache = true,
    forceRefresh = false,
    complete = false,
  }: InferenceProcessInput): Promise<InferenceResult> {
    return await this.enqueue(async () => {
      const safeModelName = await this.ensureSession(modelName);
      inferenceLogger.info(
        `runtime inference process start: session=${this.sessionState.id} model=${safeModelName} provider=${this.workerState.provider || "unknown"} batches=${batches.length}`,
      );
      if (!batches.length) {
        return this.sessionState.result;
      }

      const cacheKey = this.resolveCacheKey(fileKey, safeModelName, allowCache);
      const cachedResult = await this.tryReadCachedResult(cacheKey, forceRefresh);
      if (cachedResult) {
        return cachedResult;
      }

      const worker = await this.ensureInitialized(modelName);
      if (!worker) {
        throw new Error("Inference worker is not available");
      }

      const id = String(this.nextId++);
      inferenceLogger.info(
        `runtime inference worker process begin: session=${this.sessionState.id} id=${id} model=${safeModelName} provider=${this.workerState.provider || "unknown"} batches=${batches.length}`,
      );
      const result = await this.postWorkerRequest(worker, {
        cmd: "process",
        id,
        modelName: safeModelName,
        batches,
      } satisfies InferenceWorkerProcessMessage);
      this.sessionState.useCachedResult = false;
      this.sessionState.result = mergeInferenceResults(
        this.sessionState.result,
        result,
      );
      const mergedResult = this.sessionState.result;
      await this.writeCachedResult(cacheKey, complete, mergedResult);
      inferenceLogger.info(
        `runtime inference worker process done: session=${this.sessionState.id} id=${id} model=${safeModelName} provider=${this.workerState.provider || "unknown"}`,
      );
      return mergedResult;
    });
  }

  buildEmptyResult(): InferenceResult {
    return buildEmptyInferenceResult();
  }

  reset(): void {
    void this.enqueue(async () => {
      this.resetSessionState();
      await this.sendReset();
    });
  }
}
