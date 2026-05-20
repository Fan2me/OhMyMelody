import {
  DEFAULT_TARGET_SAMPLE_RATE,
  mixDownChannelsToMono,
  postProcessDecodedAudio,
} from "@ohm/core/audio/pcm.js";
import {
  drainSamples,
  padSamplesToLength,
  takeSamples,
  type SampleQueue,
} from "./sample-queue.js";

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
