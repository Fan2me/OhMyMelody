import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { CORE_INFERENCE_WORKER_MODULE_URL } from "@ohm/core/inference/inference.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import {
  normalizeCoreModelName,
  resolveCoreModelUrl,
} from "@ohm/core/model-catalog.js";
import type { WorkerLike } from "@ohm/core/cfp/types.js";
import type { InferenceResult } from "../analysis.js";

const inferenceLogger = getModuleLogger("core.runtime.inference");

export const DEFAULT_INFERENCE_WORKER_SCRIPT_URL =
  CORE_INFERENCE_WORKER_MODULE_URL;

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

type InferenceWorkerProgressMessage = {
  cmd: "progress";
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
  | InferenceWorkerProgressMessage
  | InferenceWorkerResultMessage
  | InferenceWorkerErrorMessage
  | InferenceWorkerInitedMessage;

type PendingPromise = {
  resolve: (value: InferenceResult) => void;
  reject: (reason?: unknown) => void;
  onProgress: ((progress: InferenceResult) => void) | null;
};

type WorkerState = {
  worker: WorkerLike | null;
  readyPromise: Promise<WorkerLike | null> | null;
  initializedModelName: string;
  provider: string;
};

export class InferenceWorkerClient {
  private readonly workerModuleUrl: string | URL;
  private readonly workerInitTimeoutMs = 30000;
  private readonly workerState: WorkerState = {
    worker: null,
    readyPromise: null,
    initializedModelName: "",
    provider: "",
  };
  private pending = new Map<string, PendingPromise>();
  private nextId = 1;

  constructor(workerModuleUrl: string | URL = DEFAULT_INFERENCE_WORKER_SCRIPT_URL) {
    this.workerModuleUrl = workerModuleUrl;
  }

  get provider(): string {
    return this.workerState.provider || "unknown";
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
          `runtime inference worker inited: model=${this.workerState.initializedModelName || "unknown"} provider=${this.provider}`,
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
      if (message.cmd === "progress") {
        const pending = this.pending.get(message.id);
        if (pending?.onProgress) {
          try {
            pending.onProgress(message.result);
          } catch (error) {
            inferenceLogger.warn(
              `runtime inference progress handler failed: ${error instanceof Error ? error.message : String(error)}`,
            );
          }
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
        this.rejectAllPending(new Error(message.error));
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
      return this.workerState.readyPromise;
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

    return this.workerState.readyPromise;
  }

  async ensureInitialized(modelName: string): Promise<WorkerLike | null> {
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
    const initMessage: InferenceWorkerInitMessage = {
      cmd: "init",
      modelName: safeModelName,
      modelUrl: resolveCoreModelUrl(safeModelName),
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
        worker.removeEventListener("message", onMessage);
        worker.removeEventListener("error", onError);
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
    onProgress?: ((progress: InferenceResult) => void) | null,
  ): Promise<InferenceResult> {
    return new Promise<InferenceResult>((resolve, reject) => {
      this.pending.set(message.id, {
        resolve,
        reject,
        onProgress: onProgress ?? null,
      });
      try {
        worker.postMessage(message);
      } catch (error) {
        this.pending.delete(message.id);
        reject(error);
      }
    });
  }

  async process({
    batches,
    modelName,
    onProgress = null,
  }: {
    batches: readonly CFPBatch[];
    modelName: string;
    onProgress?: ((progress: InferenceResult) => void) | null;
  }): Promise<{ id: string; result: InferenceResult }> {
    const worker = await this.ensureInitialized(modelName);
    if (!worker) {
      throw new Error("Inference worker is not available");
    }

    const safeModelName = normalizeCoreModelName(modelName);
    const id = String(this.nextId++);
    const result = await this.postWorkerRequest(
      worker,
      {
        cmd: "process",
        id,
        modelName: safeModelName,
        batches,
      },
      onProgress,
    );
    return { id, result };
  }

  async reset(): Promise<void> {
    const worker = await this.ensureWorker();
    if (!worker) {
      return;
    }
    inferenceLogger.info("runtime inference worker reset");
    worker.postMessage({ cmd: "reset" } satisfies InferenceWorkerResetMessage);
  }
}
