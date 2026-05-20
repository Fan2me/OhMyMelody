import type { CFPBatch } from "../cache/cfp.js";
import { throwIfAborted } from "../abort/abort.js";
import { getModuleLogger } from "../logging/logger.js";
import { isPyodideOOMError, splitCFPRangeOnOOM } from "./cfp.js";
import type { CFPChunkInput } from "./types.js";

const processLogger = getModuleLogger("core.cfp.process");

export type CFPProcessPauseController = {
  waitForResume?: (signal?: AbortSignal | null) => Promise<void> | void;
} | null;

type WorkerError = Error & { oom?: boolean };

export function deriveMinChunkSamples(fs: number): number {
  return Math.max(1, Math.floor(Math.max(1, Number(fs) || 1) * 0.5));
}

export async function processCFPInputRecursive({
  input,
  minChunkSamples,
  signal,
  waitIfPaused,
  processChunk,
}: {
  input: CFPChunkInput;
  minChunkSamples: number;
  signal?: AbortSignal | null;
  waitIfPaused: () => Promise<void>;
  processChunk: (input: CFPChunkInput) => Promise<CFPBatch>;
}): Promise<CFPBatch[]> {
  await waitIfPaused();
  throwIfAborted(signal);
  try {
    const batch = await processChunk(input);
    return [batch];
  } catch (error) {
    const oom = !!(error as WorkerError)?.oom || isPyodideOOMError(error);
    if (oom && input.pcm.length > minChunkSamples * 2) {
      const splitRange = splitCFPRangeOnOOM(0, input.pcm.length, minChunkSamples);
      if (splitRange) {
        processLogger.warn(
          `CFP OOM retry: [0,${input.pcm.length}) -> [${splitRange.left.start},${splitRange.left.end}) + [${splitRange.right.start},${splitRange.right.end})`,
        );
        const leftInput: CFPChunkInput = {
          pcm: input.pcm.slice(splitRange.left.start, splitRange.left.end),
          fs: input.fs,
        };
        const rightInput: CFPChunkInput = {
          pcm: input.pcm.slice(splitRange.right.start, splitRange.right.end),
          fs: input.fs,
        };
        const left = await processCFPInputRecursive({
          input: leftInput,
          minChunkSamples,
          signal: signal ?? null,
          waitIfPaused,
          processChunk,
        });
        const right = await processCFPInputRecursive({
          input: rightInput,
          minChunkSamples,
          signal: signal ?? null,
          waitIfPaused,
          processChunk,
        });
        return [...left, ...right];
      }
    }
    throw error;
  }
}
