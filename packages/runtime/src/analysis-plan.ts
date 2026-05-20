import { toPositiveFinite } from "@ohm/core/cfp/common.js";
import type { AnalysisPlan, AnalysisPlanTask } from "./analysis.js";
export {
  createSampleQueue,
  drainSamples,
  enqueueSamples,
  padSamplesToLength,
  takeSamples,
  type SampleQueue,
} from "./sample-queue.js";
export {
  normalizeStreamingChunk,
  takeNextStreamingSegment,
} from "./streaming-segments.js";

export const ANALYSIS_WARMUP_CHUNK_SEC = 1.28;
export const ANALYSIS_WARMUP_ROUNDS = 3;
export const ANALYSIS_STEADY_CHUNK_SEC = 1.28 * 4;
export const ANALYSIS_STREAM_CHUNK_SEC = 1.28;

export function buildAnalysisPlan(
  totalSamples: number,
  sampleRate: number,
): AnalysisPlan {
  const total = Math.max(0, Math.floor(toPositiveFinite(totalSamples, 0)));
  const sr = Math.max(1, Math.floor(toPositiveFinite(sampleRate, 1)));
  const warmUpChunkSize = ANALYSIS_WARMUP_CHUNK_SEC * sr;
  const steadyChunkSize = ANALYSIS_STEADY_CHUNK_SEC * sr;

  const tasks: AnalysisPlanTask[] = [];
  if (total <= 0) {
    return tasks;
  }

  let start = 0;
  for (let round = 0; round < ANALYSIS_WARMUP_ROUNDS && start < total; round++) {
    const end = Math.min(start + warmUpChunkSize, total);
    tasks.push({ start, end });
    start = end;
  }
  while (start < total) {
    const end = Math.min(start + steadyChunkSize, total);
    tasks.push({ start, end });
    start = end;
  }

  return tasks;
}
