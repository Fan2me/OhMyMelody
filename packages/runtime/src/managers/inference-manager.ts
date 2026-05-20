import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import {
  normalizePredictionCacheEntry,
  PredictionIndexedDBCache,
} from "@ohm/core/cache/prediction.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import { normalizeCoreModelName } from "@ohm/core/model-catalog.js";
import type { InferenceProgress, InferenceResult } from "../analysis.js";
import {
  buildInferenceCacheKey,
  isPredictionCacheEntryUsable,
  toInferenceResultFromCache,
  toPredictionCacheEntry,
} from "../inference/cache.js";
import {
  createInferenceSessionState,
  resetInferenceSessionState,
  type InferenceSessionState,
} from "../inference/session.js";
import {
  DEFAULT_INFERENCE_WORKER_SCRIPT_URL,
  InferenceWorkerClient,
} from "../inference/worker-client.js";
import {
  buildEmptyInferenceResult,
  mergeInferenceResults,
} from "../inference/result.js";

const inferenceLogger = getModuleLogger("core.runtime.inference");

type InferenceProcessInput = {
  batches: readonly CFPBatch[];
  modelName: string;
  fileKey?: string;
  forceRefresh?: boolean;
  complete?: boolean;
  onProgress?: ((progress: InferenceResult) => void) | null;
};

export interface InferenceManagerOptions {
  onProgress?: ((progress: InferenceProgress) => void) | null;
  workerModuleUrl?: string | URL;
}

export class InferenceManager {
  private readonly cache: PredictionIndexedDBCache;
  private readonly workerClient: InferenceWorkerClient;
  private readonly sessionState: InferenceSessionState =
    createInferenceSessionState();
  private processQueue: Promise<void> = Promise.resolve();

  constructor(options: InferenceManagerOptions = {}) {
    this.cache = new PredictionIndexedDBCache();
    this.workerClient = new InferenceWorkerClient(
      options.workerModuleUrl ?? DEFAULT_INFERENCE_WORKER_SCRIPT_URL,
    );
  }

  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    const next = this.processQueue.then(task);
    this.processQueue = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  private async startNewSession(
    modelName: string,
    reason: string,
  ): Promise<void> {
    const previousModel = this.sessionState.modelName || "none";
    resetInferenceSessionState(this.sessionState, modelName);
    await this.workerClient.reset();
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
    } else if (this.sessionState.complete) {
      await this.startNewSession(safeModelName, "previous-complete");
    }
    return safeModelName;
  }

  private resolveCacheKey(
    fileKey: string | undefined,
    modelName: string,
  ): string {
    if (typeof fileKey !== "string" || !fileKey.trim()) {
      return "";
    }
    return buildInferenceCacheKey(String(fileKey), modelName);
  }

  async hasCache(fileKey: string, modelName: string): Promise<boolean> {
    const safeFileKey = String(fileKey || "").trim();
    if (!safeFileKey) {
      return false;
    }
    const cacheKey = buildInferenceCacheKey(safeFileKey, modelName);
    try {
      const cachedRaw = await this.cache.getPredictionCache(cacheKey);
      const cached = normalizePredictionCacheEntry(cachedRaw);
      return isPredictionCacheEntryUsable(cached);
    } catch {
      return false;
    }
  }

  private async tryReadCachedResult(
    cacheKey: string,
    forceRefresh: boolean,
    onProgress?: ((progress: InferenceResult) => void) | null,
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
      if (isPredictionCacheEntryUsable(cached)) {
        const hit = toInferenceResultFromCache(cached);
        this.sessionState.result = hit;
        this.sessionState.useCachedResult = true;
        inferenceLogger.info(`runtime inference cache hit: key=${cacheKey}`);
        onProgress?.(hit);
        return hit;
      }
      if (cached) {
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
        toPredictionCacheEntry(result),
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
    forceRefresh = false,
    complete = false,
    onProgress = null,
  }: InferenceProcessInput): Promise<InferenceResult> {
    return this.enqueue(async () => {
      const safeModelName = await this.ensureSession(modelName);
      inferenceLogger.info(
        `runtime inference process start: session=${this.sessionState.id} model=${safeModelName} provider=${this.workerClient.provider} batches=${batches.length}`,
      );
      if (!batches.length) {
        return this.sessionState.result;
      }

      const cacheKey = this.resolveCacheKey(fileKey, safeModelName);
      const cachedResult = await this.tryReadCachedResult(
        cacheKey,
        forceRefresh,
        onProgress,
      );
      if (cachedResult) {
        return cachedResult;
      }

      inferenceLogger.info(
        `runtime inference worker process begin: session=${this.sessionState.id} model=${safeModelName} provider=${this.workerClient.provider} batches=${batches.length}`,
      );
      const { id, result } = await this.workerClient.process({
        batches,
        modelName: safeModelName,
        onProgress,
      });
      this.sessionState.useCachedResult = false;
      this.sessionState.result = mergeInferenceResults(
        this.sessionState.result,
        result,
      );
      this.sessionState.complete = complete === true;
      const mergedResult = this.sessionState.result;
      await this.writeCachedResult(cacheKey, complete, mergedResult);
      inferenceLogger.info(
        `runtime inference worker process done: session=${this.sessionState.id} id=${id} model=${safeModelName} provider=${this.workerClient.provider}`,
      );
      return mergedResult;
    });
  }

  buildEmptyResult(): InferenceResult {
    return buildEmptyInferenceResult();
  }

  reset(): void {
    void this.enqueue(async () => {
      resetInferenceSessionState(this.sessionState);
      await this.workerClient.reset();
    });
  }
}
