import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { runCFPChunkInPyodide } from "@ohm/core/cfp/chunk.js";
import {
  deriveMinChunkSamples,
  processCFPInputRecursive,
  type CFPProcessPauseController,
} from "@ohm/core/cfp/process.js";
import type { PyodideLike } from "@ohm/core/cfp/pyodide-bootstrap.js";
import type { CFPChunkInput } from "@ohm/core/cfp/types.js";

export async function runCFPOnMainThread({
  input,
  pyodide,
  signal = null,
  pauseController = null,
}: {
  input: CFPChunkInput;
  pyodide: PyodideLike | null;
  signal?: AbortSignal | null;
  pauseController?: CFPProcessPauseController;
}): Promise<CFPBatch[]> {
  if (!pyodide) {
    throw new Error("Pyodide is not initialized");
  }

  async function waitIfPaused(): Promise<void> {
    await pauseController?.waitForResume?.(signal);
  }

  return processCFPInputRecursive({
    input,
    minChunkSamples: deriveMinChunkSamples(input.fs),
    signal,
    waitIfPaused,
    processChunk: async (segment) => {
      const result = await runCFPChunkInPyodide({
        pyodide,
        pcm: segment.pcm,
        fs: segment.fs,
        phase: "main-thread",
      });
      return {
        data: result.data,
        shape: result.shape,
      };
    },
  });
}
