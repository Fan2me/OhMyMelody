import type { PyodideLike } from "./pyodide-bootstrap.js";
import { cleanupPyodideChunkArtifacts } from "./common.js";
import { getCFPChunkExecutionPython } from "../script/script.js";
import { readCFPProfile } from "./worker.js";

export type CFPChunkTiming = {
  phase: string;
  tStart: number;
  tToPyStart: number;
  tToPyEnd: number;
  tPyStart: number;
  tPyEnd: number;
  tEnd: number;
  cfpProfile: unknown;
};

export type CFPChunkPyodideResult = {
  shape: Int32Array;
  data: Float32Array;
  timing: CFPChunkTiming;
};

export async function runCFPChunkInPyodide({
  pyodide,
  pcm,
  fs,
  phase,
}: {
  pyodide: PyodideLike;
  pcm: Float32Array;
  fs: number;
  phase: string;
}): Promise<CFPChunkPyodideResult> {
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
    await pyodide.runPythonAsync(getCFPChunkExecutionPython());
    const tPyEnd = performance.now();
    const cfpProfile = readCFPProfile(pyodide);
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
