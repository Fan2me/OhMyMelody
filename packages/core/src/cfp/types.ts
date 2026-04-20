import type { CFPBatch } from "../cache/cache.js";

export interface CFPChunkInput {
  pcm: Float32Array;
  fs: number;
}

export interface CFPWorkerBackend {
  createWorkerInstance: () => WorkerLike;
  waitWorkerInit: (worker: WorkerLike, timeoutMs?: number) => Promise<boolean>;
  terminateWorkerSafely: (worker: WorkerLike) => void;
  getPrewarmedWorker?: () => WorkerLike;
  setPrewarmedWorker?: (worker: WorkerLike) => void;
  getWorkerPrewarmPromise?: () => Promise<boolean> | null;
}

export type WorkerLike = Worker | null;

export type CFPChunkResult = CFPBatch;

export type CFPWorkerInitMessage = {
  cmd: "init";
  cfpScriptUrl?: string;
  pyodideScriptUrl?: string;
  pyodideIndexURL?: string;
};

export type CFPWorkerProcessMessage = {
  cmd: "process";
  id: string;
  pcmBuffer: ArrayBuffer;
  fs: number;
};

export type CFPWorkerInitedMessage = {
  cmd: "inited";
};

export type CFPWorkerTimingMessage = {
  cmd: "timing";
  id?: string;
  start?: number;
  end?: number;
  error?: string;
  oom?: boolean;
  dataBuf?: ArrayBuffer;
  shapeBuf?: ArrayBuffer;
  cfpProfile?: unknown;
  phase?: string;
  t0?: number;
  t1?: number;
  tToPyStart?: number;
  tToPyEnd?: number;
  tPyStart?: number;
  tPyEnd?: number;
};

export type CFPWorkerResultMessage = {
  cmd: "result";
  id?: string;
  start?: number;
  end?: number;
  shapeBuf: ArrayBuffer;
  dataBuf: ArrayBuffer;
};

export type CFPWorkerErrorMessage = {
  cmd: "error";
  id?: string;
  error?: string;
  oom?: boolean;
  start?: number;
  end?: number;
};

export type CFPWorkerMessage =
  | CFPWorkerInitMessage
  | CFPWorkerProcessMessage
  | CFPWorkerInitedMessage
  | CFPWorkerTimingMessage
  | CFPWorkerResultMessage
  | CFPWorkerErrorMessage;
