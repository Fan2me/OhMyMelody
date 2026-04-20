import { toPositiveFinite } from "./common.js";
import type { CFPWorkerMessage, WorkerLike } from "./types.js";

export interface CFPWorkerManagerOptions {
  disableWorker?: boolean;
  createWorkerInstance?: () => WorkerLike;
  resolveCFPScriptUrl?: () => string;
  resolvePyodideIndexURL?: () => string;
  resolvePyodideScriptUrl?: (indexURL?: string) => string;
}

export interface CFPWorkerPrewarmOptions {
  workerInitTimeoutMs?: number;
}

export function createCFPWorkerManager({
  disableWorker = false,
  createWorkerInstance = () => null,
  resolveCFPScriptUrl = () => "",
  resolvePyodideIndexURL = () => "",
  resolvePyodideScriptUrl = (indexURL = "") => indexURL,
}: CFPWorkerManagerOptions = {}) {
  let prewarmedWorker: WorkerLike = null;
  let workerPrewarmPromise: Promise<boolean> | null = null;

  function terminateWorkerSafely(worker: WorkerLike): void {
    try {
      if (worker) {
        worker.terminate();
      }
    } catch {}
  }

  async function waitWorkerInit(
    worker: WorkerLike,
    timeoutMs = 30000,
    onFailure?: (reason: string) => void,
  ): Promise<boolean> {
    if (!worker) {
      return false;
    }
    const ms = Math.max(5000, Math.floor(toPositiveFinite(timeoutMs, 30000) ?? 30000));
    return await new Promise<boolean>((resolve) => {
      let settled = false;
      let timeout: ReturnType<typeof setTimeout> | null = null;
      const finish = (ok: boolean, reason?: string) => {
        if (settled) {
          return;
        }
        settled = true;
        if (timeout) {
          clearTimeout(timeout);
          timeout = null;
        }
        worker.removeEventListener("message", onmsg);
        worker.removeEventListener("error", onerror);
        worker.removeEventListener("messageerror", onmessageerror);
        if (!ok && reason) {
          onFailure?.(reason);
        }
        resolve(ok);
      };
      const onmsg = (ev: MessageEvent<CFPWorkerMessage>) => {
        const message = ev.data ?? null;
        if (!message || typeof message !== "object") {
          return;
        }
        if (message.cmd === "inited") {
          finish(true);
          return;
        }
        if (message.cmd === "error") {
          const reason =
            typeof message.error === "string" && message.error
              ? message.error
              : "CFP worker init reported an error";
          finish(false, reason);
        }
      };
      const onerror = (event: ErrorEvent) => {
        const reason =
          event?.message ||
          (event?.error instanceof Error ? event.error.message : "") ||
          "CFP worker runtime error";
        finish(false, reason);
      };
      const onmessageerror = () => finish(false, "CFP worker messageerror");
      timeout = setTimeout(() => finish(false, `CFP worker init timed out after ${ms}ms`), ms);
      worker.addEventListener("message", onmsg);
      worker.addEventListener("error", onerror);
      worker.addEventListener("messageerror", onmessageerror);
      try {
        const pyodideIndexURL = resolvePyodideIndexURL();
        worker.postMessage({
          cmd: "init",
          cfpScriptUrl: resolveCFPScriptUrl(),
          pyodideIndexURL,
          pyodideScriptUrl: resolvePyodideScriptUrl(pyodideIndexURL),
        });
      } catch {
        finish(false, "CFP worker init postMessage failed");
      }
    });
  }

  async function prewarmCFPWorker(
    options: CFPWorkerPrewarmOptions = {},
  ): Promise<boolean> {
    if (disableWorker || typeof Worker === "undefined") {
      return false;
    }
    if (prewarmedWorker) {
      return true;
    }
    if (workerPrewarmPromise) {
      return await workerPrewarmPromise;
    }

    const workerInitTimeoutMs = Math.max(
      5000,
      Math.floor(toPositiveFinite(options.workerInitTimeoutMs, 30000) ?? 30000),
    );
    workerPrewarmPromise = (async () => {
      const worker = createWorkerInstance();
      if (!worker) {
        return false;
      }
      const ready = await waitWorkerInit(worker, workerInitTimeoutMs);
      if (!ready) {
        terminateWorkerSafely(worker);
        return false;
      }
      prewarmedWorker = worker;
      return true;
    })().finally(() => {
      workerPrewarmPromise = null;
    });

    return await workerPrewarmPromise;
  }

  return {
    terminateWorkerSafely,
    waitWorkerInit,
    prewarmCFPWorker,
    getPrewarmedWorker: () => prewarmedWorker,
    setPrewarmedWorker: (nextWorker: WorkerLike) => {
      prewarmedWorker = nextWorker;
    },
    getWorkerPrewarmPromise: () => workerPrewarmPromise,
  };
}
