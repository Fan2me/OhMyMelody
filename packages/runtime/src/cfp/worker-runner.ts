import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { isAbortError, throwIfAborted } from "@ohm/core/abort/abort.js";
import {
  deriveMinChunkSamples,
  processCFPInputRecursive,
  type CFPProcessPauseController,
} from "@ohm/core/cfp/process.js";
import type {
  CFPChunkInput,
  CFPWorkerErrorMessage,
  CFPWorkerMessage,
  CFPWorkerResultMessage,
  CFPWorkerTimingMessage,
  WorkerLike,
} from "@ohm/core/cfp/types.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";

const processLogger = getModuleLogger("core.runtime.cfp.process");

type PendingChunkPromise = {
  resolve: (value: CFPBatch) => void;
  reject: (reason?: unknown) => void;
};

type WorkerError = Error & { oom?: boolean };

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
      const pendingPromise = clearPending(message.id);
      pendingPromise?.resolve(batch);
      return;
    }
    rejectAllPending(new Error("CFP worker result missing id"));
  }

  function handleWorkerErrorMessage(message: CFPWorkerErrorMessage): void {
    const errObj = new Error(message.error || "cfp worker error") as WorkerError;
    errObj.oom = !!message.oom;
    if (typeof message.id === "string" && message.id) {
      const pendingPromise = clearPending(message.id);
      pendingPromise?.reject(errObj);
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

export async function runCFPWithResidentWorker({
  input,
  worker,
  signal = null,
  pauseController = null,
}: {
  input: CFPChunkInput;
  worker: WorkerLike;
  signal?: AbortSignal | null;
  pauseController?: CFPProcessPauseController;
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

  if (worker) {
    worker.onmessage = (ev: MessageEvent<CFPWorkerMessage>) => {
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
    worker.onerror = () => {
      handlers.rejectAllPending(new Error("CFP worker runtime error"));
    };
  }

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
