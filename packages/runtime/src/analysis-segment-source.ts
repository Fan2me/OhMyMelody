import { DEFAULT_TARGET_SAMPLE_RATE } from "@ohm/core/audio/pcm.js";
import { ANALYSIS_STREAM_CHUNK_SEC } from "./analysis-plan.js";
import { createSampleQueue, enqueueSamples } from "./sample-queue.js";
import {
  normalizeStreamingChunk,
  takeNextStreamingSegment,
} from "./streaming-segments.js";
import type { AnalysisSessionStore } from "./analysis-session-store.js";
import type { AnalyzeInput } from "./types.js";

export type AnalyzerAudioManagerLike = {
  setAudio(input: AnalyzeInput): Promise<unknown>;
  getAudio(): { pcm: Float32Array; fs: number; mode?: string } | null;
  getPcmChunk(start: number, end: number): Float32Array;
};

export type AudioSnapshot = {
  pcm: Float32Array;
  fs: number;
  mode?: string;
};

export type PcmSegment = {
  index: number;
  segment: { pcm: Float32Array; fs: number };
  final: boolean;
  outputIndex: number;
  appendToStreamingAudio: boolean;
};

export type PcmSegmentSource = {
  audio: AudioSnapshot;
  reuseCFP: boolean;
  forceRefresh: boolean;
  next(generation: number): Promise<PcmSegment | null>;
  afterSegment?(segment: PcmSegment, generation: number): void;
};

export type StreamingPcmSegmentSource = PcmSegmentSource & {
  readonly processedCount: number;
  scheduleChunk(
    token: number,
    rawChunk: readonly Float32Array[],
    sampleRate: number,
  ): void;
  finish(): void;
  fail(error: unknown): void;
};

export function snapshotDecodedAudio(store: AnalysisSessionStore): AudioSnapshot {
  const audio = store.requireDecodedAudio();
  return audio.mode
    ? { pcm: audio.pcm, fs: audio.fs, mode: audio.mode }
    : { pcm: audio.pcm, fs: audio.fs };
}

export function createDecodedAudioSegmentSource({
  store,
  audioManager,
}: {
  store: AnalysisSessionStore;
  audioManager: Pick<AnalyzerAudioManagerLike, "getPcmChunk">;
}): PcmSegmentSource {
  const decodedAudio = store.requireDecodedAudio();
  const run = store.run;
  if (!run) {
    throw new Error("runtime run is unavailable");
  }

  return {
    audio: snapshotDecodedAudio(store),
    reuseCFP: run.reuseCFP,
    forceRefresh: run.execution.forceRefresh ?? false,
    async next(generation: number): Promise<PcmSegment | null> {
      if (generation !== store.generation) {
        return null;
      }
      const plan = store.requirePlan();
      if (store.state.nextPlanIndex >= plan.length) {
        return null;
      }
      const index = store.state.nextPlanIndex;
      const task = store.requirePlanTask(plan);
      return {
        index,
        segment: {
          pcm: audioManager.getPcmChunk(task.start, task.end),
          fs: decodedAudio.fs,
        },
        final: index + 1 >= plan.length,
        outputIndex: plan.length,
        appendToStreamingAudio: false,
      };
    },
    afterSegment(segment: PcmSegment, generation: number): void {
      if (generation !== store.generation) {
        return;
      }
      const plan = store.requirePlan();
      const runAfterSegment = store.run;
      if (!runAfterSegment) {
        throw new Error("runtime run is unavailable");
      }
      store.setPlanIndex(
        runAfterSegment.cfpComplete ? plan.length : segment.index + 1,
      );
    },
  };
}

export function createStreamingSegmentSource({
  isAborted,
}: {
  isAborted(token: number): boolean;
}): StreamingPcmSegmentSource {
  const targetSegmentSamples = Math.max(
    1,
    Math.round(DEFAULT_TARGET_SAMPLE_RATE * ANALYSIS_STREAM_CHUNK_SEC),
  );
  const pendingSamples = createSampleQueue();

  let streamEnded = false;
  let segmentIndex = 0;
  let processedCount = 0;
  let streamError: unknown = null;
  let pendingWake: (() => void) | null = null;

  function wake(): void {
    const notify = pendingWake;
    pendingWake = null;
    notify?.();
  }

  function waitForSamples(): Promise<void> {
    return new Promise((resolve) => {
      pendingWake = resolve;
    });
  }

  return {
    audio: {
      pcm: new Float32Array(0),
      fs: DEFAULT_TARGET_SAMPLE_RATE,
    },
    reuseCFP: false,
    forceRefresh: true,
    get processedCount(): number {
      return processedCount;
    },
    async next(token: number): Promise<PcmSegment | null> {
      while (!isAborted(token)) {
        if (streamError) {
          throw streamError;
        }
        const next = takeNextStreamingSegment({
          queue: pendingSamples,
          streamEnded,
          targetSegmentSamples,
        });
        if (next.segment && next.segment.length) {
          const index = segmentIndex;
          segmentIndex += 1;
          processedCount += 1;
          if (next.final) {
            streamEnded = true;
          }
          return {
            index,
            segment: {
              pcm: next.segment,
              fs: DEFAULT_TARGET_SAMPLE_RATE,
            },
            final: next.final,
            outputIndex: index + 1,
            appendToStreamingAudio: true,
          };
        }
        if (streamEnded) {
          return null;
        }
        await waitForSamples();
      }
      return null;
    },
    scheduleChunk(
      token: number,
      rawChunk: readonly Float32Array[],
      sampleRate: number,
    ): void {
      if (isAborted(token)) {
        return;
      }
      enqueueSamples(pendingSamples, normalizeStreamingChunk(rawChunk, sampleRate));
      wake();
    },
    finish(): void {
      streamEnded = true;
      wake();
    },
    fail(error: unknown): void {
      streamError = error;
      streamEnded = true;
      wake();
    },
  };
}
