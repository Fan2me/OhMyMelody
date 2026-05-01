import { toPositiveFinite } from "@ohm/core/cfp/common.js";
import {
  DEFAULT_TARGET_SAMPLE_RATE,
  mixDownChannelsToMono,
  postProcessDecodedAudio,
} from "@ohm/core/audio/pcm.js";
import type { AnalysisPlan, AnalysisPlanTask } from "./analysis.js";

export const ANALYSIS_WARMUP_CHUNK_SEC = 1.28;
export const ANALYSIS_WARMUP_ROUNDS = 20;
export const ANALYSIS_STEADY_CHUNK_SEC = 1.28*20;
export const ANALYSIS_STREAM_CHUNK_SEC = 1.28;

export type SampleQueue = {
  chunks: Float32Array[];
  totalSamples: number;
};

export function createSampleQueue(): SampleQueue {
  return {
    chunks: [],
    totalSamples: 0,
  };
}

export function enqueueSamples(queue: SampleQueue, samples: Float32Array): void {
  if (!samples.length) {
    return;
  }
  queue.chunks.push(samples);
  queue.totalSamples += samples.length;
}

export function takeSamples(
  queue: SampleQueue,
  sampleCount: number,
): Float32Array | null {
  const take = Math.max(0, Math.floor(sampleCount));
  if (!take || queue.totalSamples < take) {
    return null;
  }

  const out = new Float32Array(take);
  let offset = 0;
  while (offset < take && queue.chunks.length > 0) {
    const head = queue.chunks[0];
    if (!head) {
      queue.chunks.shift();
      continue;
    }
    const available = head.length;
    const need = take - offset;
    const copyCount = Math.min(available, need);
    out.set(head.subarray(0, copyCount), offset);
    offset += copyCount;
    queue.totalSamples -= copyCount;
    if (copyCount === available) {
      queue.chunks.shift();
    } else {
      queue.chunks[0] = head.subarray(copyCount);
    }
  }

  return offset === take ? out : out.slice(0, offset);
}

export function drainSamples(queue: SampleQueue): Float32Array {
  if (!queue.totalSamples || !queue.chunks.length) {
    return new Float32Array(0);
  }
  return takeSamples(queue, queue.totalSamples) ?? new Float32Array(0);
}

export function padSamplesToLength(
  samples: Float32Array,
  targetLength: number,
): Float32Array {
  const target = Math.max(0, Math.floor(targetLength));
  if (target <= 0) {
    return new Float32Array(0);
  }
  if (samples.length >= target) {
    return samples;
  }
  const padded = new Float32Array(target);
  padded.set(samples);
  return padded;
}

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

export function normalizeStreamingChunk(
  rawChunk: readonly Float32Array[],
  sampleRate: number,
): Float32Array {
  const monoChunk = mixDownChannelsToMono(rawChunk);
  return postProcessDecodedAudio({
    decodedPcm: monoChunk,
    sampleRate,
    targetSampleRate: DEFAULT_TARGET_SAMPLE_RATE,
  }).pcm;
}

export function takeNextStreamingSegment({
  queue,
  streamEnded,
  targetSegmentSamples,
}: {
  queue: SampleQueue;
  streamEnded: boolean;
  targetSegmentSamples: number;
}): { segment: Float32Array | null; final: boolean } {
  if (queue.totalSamples >= targetSegmentSamples) {
    return {
      segment: takeSamples(queue, targetSegmentSamples),
      final: false,
    };
  }
  if (streamEnded && queue.totalSamples > 0) {
    const tail = drainSamples(queue);
    return {
      segment: padSamplesToLength(tail, targetSegmentSamples),
      final: true,
    };
  }
  return {
    segment: null,
    final: false,
  };
}
