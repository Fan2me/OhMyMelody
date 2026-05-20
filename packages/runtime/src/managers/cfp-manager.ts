import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { CFPIndexedDBCache } from "@ohm/core/cache/cfp.js";
import {
  CORE_CFP_SCRIPT_URL,
  CORE_CFP_WORKER_MODULE_URL,
} from "@ohm/core/cfp/index.js";
import { createCFPWorkerManager } from "@ohm/core/cfp/worker-manager.js";
import type { CFPChunkInput, WorkerLike } from "@ohm/core/cfp/types.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import {
  buildCFPCacheKey,
  commitCFPCache,
  normalizeCFPBatches,
} from "../cfp/cache.js";
import { runCFPOnMainThread } from "../cfp/main-thread-runner.js";
import {
  initializeBrowserCFPPyodide,
  type PyodideWorkerLike,
} from "../cfp/pyodide-runtime.js";
import { runCFPWithResidentWorker } from "../cfp/worker-runner.js";

const cfpLogger = getModuleLogger("core.runtime.cfp");

const PYODIDE_CDN_VERSION = "0.25.1";
const defaultPyodideScriptUrl = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/pyodide.js`;
const defaultPyodideIndexURL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/`;
const defaultCFPScriptUrl = CORE_CFP_SCRIPT_URL;

export interface CFPManagerOptions {
  label?: string;
  pyodideScriptUrl?: string;
  pyodideIndexURL?: string;
  cfpScriptUrl?: string;
  createWorkerInstance?: (() => WorkerLike) | undefined;
}

export interface CFPCheckCacheInput {
  fileKey: string;
  forceRefresh?: boolean;
}

export interface CFPCheckCacheResult {
  fileKey: string;
  batches: readonly CFPBatch[] | null;
}

export interface CFPProcessInput {
  fileKey: string;
  batchOffset?: number;
  segment: CFPChunkInput;
  signal?: AbortSignal | null;
  complete?: boolean;
  forceRefresh?: boolean;
}

export type CFPProcessResult =
  | {
      kind: "cache-hit";
      fileKey: string;
      batches: readonly CFPBatch[];
      complete: true;
    }
  | {
      kind: "segment";
      fileKey: string;
      batches: readonly CFPBatch[];
      complete: boolean;
    };

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
  private readonly pyodideState: {
    runtime: PyodideWorkerLike | null;
    ready: boolean;
  } = {
    runtime: null,
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
      `runtime cfp manager ready: worker=${CORE_CFP_WORKER_MODULE_URL} pyodide=${this.pyodideScriptUrl} cfp=${this.cfpScriptUrl}`,
    );
    this.cache = new CFPIndexedDBCache({
      normalizeCFPBatches,
    });
    this.workerManager = createCFPWorkerManager({
      disableWorker: typeof Worker === "undefined",
      createWorkerInstance: () => this.createDefaultWorkerInstance(),
      resolveCFPScriptUrl: () => this.cfpScriptUrl,
      resolvePyodideIndexURL: () => this.pyodideIndexURL,
      resolvePyodideScriptUrl: () => this.pyodideScriptUrl,
    });
    void this.workerManager.prewarmCFPWorker().then((ready) => {
      cfpLogger.info(
        ready
          ? `runtime cfp worker prewarm ready: ${CORE_CFP_WORKER_MODULE_URL}`
          : `runtime cfp worker prewarm skipped or unavailable: ${CORE_CFP_WORKER_MODULE_URL}`,
      );
    });
  }

  private createDefaultWorkerInstance(): WorkerLike {
    if (this.createWorkerInstance) {
      try {
        return this.createWorkerInstance();
      } catch {
        return null;
      }
    }
    try {
      return new Worker(CORE_CFP_WORKER_MODULE_URL, { type: "module" });
    } catch {
      return null;
    }
  }

  private enqueueProcess<T>(task: () => Promise<T>): Promise<T> {
    const next = this.processQueue.then(task);
    this.processQueue = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  private buildCacheKey(fileKey: string): string {
    return buildCFPCacheKey({
      namespace: this.analyzerLabel,
      fileKey,
    });
  }

  async checkCache({
    fileKey,
    forceRefresh = false,
  }: CFPCheckCacheInput): Promise<CFPCheckCacheResult> {
    const cacheKey = this.buildCacheKey(fileKey);

    if (!forceRefresh && cacheKey) {
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
      return this.residentWorkerReadyPromise;
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
          `runtime cfp resident worker ready: ${CORE_CFP_WORKER_MODULE_URL}`,
        );
        return worker;
      }

      worker = this.createDefaultWorkerInstance();
      if (!worker) {
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
          `runtime cfp resident worker init failed: ${CORE_CFP_WORKER_MODULE_URL}${initFailureReason ? ` reason=${initFailureReason}` : ""}`,
        );
        this.workerManager.terminateWorkerSafely(worker);
        return null;
      }

      this.residentWorker = worker;
      if (typeof this.workerManager.setPrewarmedWorker === "function") {
        this.workerManager.setPrewarmedWorker(worker);
      }
      cfpLogger.info(
        `runtime cfp resident worker ready: ${CORE_CFP_WORKER_MODULE_URL}`,
      );
      return worker;
    })().finally(() => {
      this.residentWorkerReadyPromise = null;
    });

    return this.residentWorkerReadyPromise;
  }

  async process({
    fileKey,
    batchOffset = 0,
    segment,
    signal = null,
    complete = false,
    forceRefresh = false,
  }: CFPProcessInput): Promise<CFPProcessResult> {
    return this.enqueueProcess(async () => {
      const cacheKey = this.buildCacheKey(fileKey);

      if (batchOffset <= 0) {
        const cacheState = await this.checkCache({
          fileKey,
          forceRefresh,
        });
        if (cacheState.batches?.length) {
          return {
            kind: "cache-hit",
            fileKey: cacheState.fileKey,
            batches: cacheState.batches,
            complete: true,
          };
        }
      }

      const batches = await this.processCFP({
        input: segment,
        signal,
      });
      await this.commitCache({
        cacheKey,
        batches,
        startIndex: batchOffset,
        complete,
      });

      return {
        kind: "segment",
        fileKey,
        batches,
        complete,
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
    startIndex,
    complete,
  }: {
    cacheKey: string;
    batches: readonly CFPBatch[];
    startIndex: number;
    complete: boolean;
  }): Promise<void> {
    await commitCFPCache({
      cache: this.cache,
      cacheKey,
      batches,
      startIndex,
      complete,
    });
  }

  private async ensurePyodide(): Promise<PyodideWorkerLike> {
    if (this.pyodideState.ready && this.pyodideState.runtime) {
      return this.pyodideState.runtime;
    }

    cfpLogger.info(
      `runtime cfp loading pyodide on main-thread: script=${this.pyodideScriptUrl} index=${this.pyodideIndexURL}`,
    );
    this.pyodideState.runtime = await initializeBrowserCFPPyodide({
      pyodideScriptUrl: this.pyodideScriptUrl,
      pyodideIndexURL: this.pyodideIndexURL,
      cfpScriptUrl: this.cfpScriptUrl,
    });
    this.pyodideState.ready = true;
    cfpLogger.info("runtime cfp pyodide ready on main-thread");
    return this.pyodideState.runtime;
  }
}
