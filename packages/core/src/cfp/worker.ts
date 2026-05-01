import {
  initializePyodideForCFP,
  type CFPBootstrapEnvironment,
  type PyodideLike,
} from "./pyodide-bootstrap.js";
import type {
  CFPWorkerErrorMessage,
  CFPWorkerInitMessage,
  CFPWorkerMessage,
  CFPWorkerProcessMessage,
  CFPWorkerTimingMessage,
} from "./types.js";

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

function toErrorMessage(err: unknown): string {
  return err && typeof err === "object" && "toString" in err
    ? String(err)
    : String(err);
}

function isPyodideOOMError(err: unknown): boolean {
  const msg =
    err && typeof err === "object" && "toString" in err ? String(err) : "";
  return /ArrayMemoryError|Unable to allocate|out of memory|MemoryError/i.test(
    msg,
  );
}

let pyodide: PyodideWorkerLike | null = null;
let ready = false;
let cfpScriptUrl: string | null = null;
const PYODIDE_CDN_VERSION = "0.25.1";
let pyodideScriptUrl = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/pyodide.js`;
let pyodideIndexURL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/`;

type CFPWorkerChunkResult = {
  shape: Int32Array;
  data: Float32Array;
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

function cleanupPyodideChunkArtifacts(
  pyodide: { runPython?: (code: string) => unknown; FS?: { unlink?: (path: string) => void } | null } | null,
): void {
  if (!pyodide) {
    return;
  }

  try {
    if (typeof pyodide.runPython === "function") {
      pyodide.runPython(`
import gc
for _name in ("x_pcm", "fs_pcm", "W"):
    try:
        del globals()[_name]
    except KeyError:
        pass
gc.collect()
`);
    }
  } catch {}

  try {
    const fs = pyodide.FS;
    if (fs && typeof fs.unlink === "function") {
      try {
        fs.unlink("cfp_out_shape.bin");
      } catch {}
      try {
        fs.unlink("cfp_out.bin");
      } catch {}
    }
  } catch {}
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

function copyResultBuffers(
  shape: Int32Array,
  data: Float32Array,
): { shape: Int32Array; data: Float32Array } {
  const shapeCopy = new Int32Array(shape.length);
  shapeCopy.set(shape);
  const dataCopy = new Float32Array(data.length);
  dataCopy.set(data);
  return { shape: shapeCopy, data: dataCopy };
}

async function runCFPChunkInPyodide({
  pyodide,
  pcm,
  fs,
  phase,
}: {
  pyodide: PyodideLike;
  pcm: Float32Array;
  fs: number;
  phase: string;
}): Promise<CFPWorkerChunkResult> {
  if (!pyodide) {
    throw new Error("Pyodide is not initialized");
  }

  const tStart = performance.now();
  const tToPyStart = performance.now();
  const np_pcm = pyodide.toPy(pcm) as { destroy?: () => void } | null;
    const tToPyEnd = performance.now();

  try {
    pyodide.globals.set("x_pcm", np_pcm);
    pyodide.globals.set("fs_pcm", fs);
    const tPyStart = performance.now();
    await pyodide.runPythonAsync(`
import cfp
W = cfp.cfp_process_from_array(x_pcm, fs_pcm, model_type="melody")
`);
    const tPyEnd = performance.now();
    let cfpProfile: unknown = null;
    try {
      const jsonText = pyodide.runPython("import cfp\ncfp.get_last_cfp_profile_json()");
      cfpProfile = typeof jsonText === "string" ? JSON.parse(jsonText) : null;
    } catch {
      cfpProfile = null;
    }
    const shapeBuf = pyodide.FS?.readFile("cfp_out_shape.bin");
    const dataBuf = pyodide.FS?.readFile("cfp_out.bin");
    if (!shapeBuf || !dataBuf) {
      throw new Error("Failed to read CFP output from Pyodide FS");
    }
    const shape = new Int32Array(
      shapeBuf.buffer,
      shapeBuf.byteOffset,
      shapeBuf.byteLength / 4,
    );
    const data = new Float32Array(
      dataBuf.buffer,
      dataBuf.byteOffset,
      dataBuf.byteLength / 4,
    );
    return {
      shape,
      data,
      timing: {
        phase,
        tStart,
        tToPyStart,
        tToPyEnd,
        tPyStart,
        tPyEnd,
        tEnd: performance.now(),
        cfpProfile,
      },
    };
  } finally {
    try {
      if (np_pcm && typeof np_pcm.destroy === "function") {
        np_pcm.destroy();
      }
    } catch {}
    cleanupPyodideChunkArtifacts(pyodide);
  }
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
      shapeBuf: result.shape.buffer,
      dataBuf: result.data.buffer,
    },
    [result.shape.buffer as ArrayBuffer, result.data.buffer as ArrayBuffer],
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
        return loadPyodide({ indexURL });
      },
    } satisfies CFPBootstrapEnvironment,
  })) as PyodideWorkerLike;
  ready = true;
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
  const { shape, data } = copyResultBuffers(result.shape, result.data);
  postCFPResult(
    {
      shape,
      data,
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
