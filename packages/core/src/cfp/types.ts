import type { CFPBatch } from "../cache/cfp.js";
import type { PyodideLike } from "./pyodide-bootstrap.js";

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

export type PyodideWorkerLike = PyodideLike & {
  toPy(value: Float32Array): { destroy?: () => void };
  globals: { set(name: string, value: unknown): void };
  runPython(code: string): unknown;
  runPythonAsync(code: string): Promise<unknown>;
  FS: PyodideLike["FS"] & {
    readFile(path: string): Uint8Array;
    unlink?(path: string): void;
  };
};

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
