import type {
  CFPWorkerErrorMessage,
  CFPWorkerInitMessage,
  CFPWorkerMessage,
  CFPWorkerProcessMessage,
  CFPWorkerTimingMessage,
} from "./types.js";
import {
  type PyodideLike,
  initializePyodideForCFP,
  type CFPBootstrapEnvironment,
} from "./pyodide-bootstrap.js";
import { toErrorMessage } from "./common.js";
import { isPyodideOOMError } from "./cfp.js";
import { runCFPChunkInPyodide } from "./chunk.js";
import { getCFPProfileReadPython } from "../script/script.js";

type PyodideWorkerLike = PyodideLike & {
  toPy(value: Float32Array): { destroy?: () => void };
  globals: { set(name: string, value: unknown): void };
  runPython(code: string): unknown;
  runPythonAsync(code: string): Promise<unknown>;
  FS: PyodideLike["FS"] & {
    readFile(path: string): Uint8Array;
    unlink?(path: string): void;
  };
};

let pyodide: PyodideWorkerLike | null = null;
let ready = false;
let cfpScriptUrl: string | null = null;
const PYODIDE_CDN_VERSION = "0.25.1";
let pyodideScriptUrl = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/pyodide.js`;
let pyodideIndexURL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/`;

type CFPWorkerChunkResult = {
  shapeCopy: Int32Array;
  dataCopy: Float32Array;
  timing: {
    phase: string;
    tStart: number;
    tToPyStart: number;
    tToPyEnd: number;
    tPyStart: number;
    tPyEnd: number;
    tEnd: number;
    cfpProfile: unknown;
  };
};

export function readCFPProfile(pyodide: PyodideLike): unknown {
  try {
    if (!pyodide) {
      return null;
    }
    const jsonText = pyodide.runPython(getCFPProfileReadPython());
    return typeof jsonText === "string" ? JSON.parse(jsonText) : null;
  } catch {
    return null;
  }
}

async function loadWorkerScript(url: string): Promise<boolean> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to load external script ${url} (HTTP ${resp.status}).`);
  }
  const code = await resp.text();
  (0, eval)(`${code}\n//# sourceURL=${url}`);
  return true;
}

async function initPyodideWorker(): Promise<void> {
  if (ready) {
    return;
  }
  pyodide = (await initializePyodideForCFP({
    pyodideScriptUrl,
    pyodideIndexURL,
    cfpScriptUrl: cfpScriptUrl || new URL("../../cfp.py", import.meta.url).toString(),
    packages: ["numpy", "scipy"],
    environment: {
      loadScript: loadWorkerScript,
      loadPyodide: async ({ indexURL }) => {
        const workerGlobal = globalThis as typeof globalThis & {
          loadPyodide?: (options: { indexURL: string }) => Promise<PyodideLike>;
        };
        const loadPyodide =
          typeof workerGlobal.loadPyodide === "function"
            ? workerGlobal.loadPyodide.bind(workerGlobal)
            : null;
        if (!loadPyodide) {
          throw new Error(
            "Pyodide is not available in worker environment. Ensure pyodide.js has been loaded inside the worker before init.",
          );
        }
        return await loadPyodide({ indexURL });
      },
    } satisfies CFPBootstrapEnvironment,
  })) as PyodideWorkerLike;
  ready = true;
}

function copyResultBuffers(
  shape: Int32Array,
  data: Float32Array,
): { shapeCopy: Int32Array; dataCopy: Float32Array } {
  const shapeCopy = new Int32Array(shape.length);
  shapeCopy.set(shape);
  const dataCopy = new Float32Array(data.length);
  dataCopy.set(data);
  return { shapeCopy, dataCopy };
}

function postCFPResult(
  result: CFPWorkerChunkResult,
  meta: Record<string, unknown> = {},
): void {
  const timingMessage: CFPWorkerTimingMessage = {
    cmd: "timing",
    phase: result.timing.phase,
    tToPyStart: result.timing.tToPyStart,
    tToPyEnd: result.timing.tToPyEnd,
    tPyStart: result.timing.tPyStart,
    tPyEnd: result.timing.tPyEnd,
    cfpProfile: result.timing.cfpProfile,
    ...meta,
  };

  if (Object.prototype.hasOwnProperty.call(meta, "id")) {
    timingMessage.t0 = result.timing.tStart;
    timingMessage.t1 = result.timing.tEnd;
  }

  self.postMessage(timingMessage);
  const workerGlobal = self as typeof self & {
    postMessage(message: unknown, transfer?: Transferable[]): void;
  };
  workerGlobal.postMessage(
    {
      cmd: "result",
      ...meta,
      shapeBuf: result.shapeCopy.buffer,
      dataBuf: result.dataCopy.buffer,
    },
    [result.shapeCopy.buffer as ArrayBuffer, result.dataCopy.buffer as ArrayBuffer],
  );
}

function postCFPError(err: unknown, meta: Record<string, unknown> = {}): void {
  const oom = isPyodideOOMError(err);
  const payload: CFPWorkerErrorMessage = {
    cmd: "error",
    ...meta,
    error: toErrorMessage(err),
    oom,
  };
  self.postMessage(payload);
}

async function handleInitMessage(message: CFPWorkerInitMessage): Promise<void> {
  cfpScriptUrl =
    typeof message.cfpScriptUrl === "string" && message.cfpScriptUrl
      ? message.cfpScriptUrl
      : cfpScriptUrl;
  pyodideScriptUrl =
    typeof message.pyodideScriptUrl === "string" && message.pyodideScriptUrl
      ? message.pyodideScriptUrl
      : pyodideScriptUrl;
  pyodideIndexURL =
    typeof message.pyodideIndexURL === "string" && message.pyodideIndexURL
      ? message.pyodideIndexURL
      : pyodideIndexURL;
  await initPyodideWorker();
  self.postMessage({ cmd: "inited" });
}

async function handleProcessMessage(message: CFPWorkerProcessMessage): Promise<void> {
  if (!ready) {
    await initPyodideWorker();
  }
  const floatArr = new Float32Array(message.pcmBuffer);
  const pyodideRuntime = pyodide as PyodideWorkerLike | null;
  if (!pyodideRuntime) {
    throw new Error("Pyodide is not initialized");
  }
  const result = await runCFPChunkInPyodide({
    pyodide: pyodideRuntime,
    pcm: floatArr,
    fs: message.fs,
    phase: "process",
  });
  const { shapeCopy, dataCopy } = copyResultBuffers(result.shape, result.data);
  postCFPResult(
    {
      shapeCopy,
      dataCopy,
      timing: result.timing,
    },
    { id: message.id },
  );
}

self.onmessage = async (ev: MessageEvent<CFPWorkerMessage>) => {
  const message = ev.data || ({} as CFPWorkerMessage);
  try {
    if (message.cmd === "init") {
      await handleInitMessage(message);
      return;
    }
    if (message.cmd === "process") {
      await handleProcessMessage(message);
      return;
    }
  } catch (error) {
    const errId = "id" in message && typeof message.id === "string" ? message.id : undefined;
    postCFPError(error, errId ? { id: errId } : {});
  }
};
