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

  async function waitWorkerInit(worker: WorkerLike, timeoutMs = 30000): Promise<boolean> {
    if (!worker) {
      return false;
    }
    const ms = Math.max(5000, Math.floor(toPositiveFinite(timeoutMs, 30000) ?? 30000));
    return await new Promise<boolean>((resolve) => {
      let settled = false;
      let timeout: ReturnType<typeof setTimeout> | null = null;
      const finish = (ok: boolean) => {
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
          finish(false);
        }
      };
      const onerror = () => finish(false);
      timeout = setTimeout(() => finish(false), ms);
      worker.addEventListener("message", onmsg);
      worker.addEventListener("error", onerror);
      try {
        const pyodideIndexURL = resolvePyodideIndexURL();
        worker.postMessage({
          cmd: "init",
          cfpScriptUrl: resolveCFPScriptUrl(),
          pyodideIndexURL,
          pyodideScriptUrl: resolvePyodideScriptUrl(pyodideIndexURL),
        });
      } catch {
        finish(false);
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
