import { DEFAULT_TARGET_SAMPLE_RATE } from "@ohm/core/audio/pcm.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import { captureAudioFromMediaStream } from "@ohm/core/media/stream-capture.js";
import {
  createSampleQueue,
  enqueueSamples,
  normalizeStreamingChunk,
  takeNextStreamingSegment,
  ANALYSIS_STREAM_CHUNK_SEC,
} from "./analysis-plan.js";
import { AnalysisPhase } from "./analysis.js";
import type { AnalysisSessionStore } from "./analysis-session-store.js";

export interface CreateAnalysisStreamRunnerOptions {
  store: AnalysisSessionStore;
  emitPhaseEnd<P extends AnalysisPhase>(
    phase: P,
    index: number,
    data: import("./analysis.js").AnalyzerPhaseEventDataMap[P],
  ): void;
}

export function createAnalysisStreamRunner({
  store,
  emitPhaseEnd,
}: CreateAnalysisStreamRunnerOptions) {
  const logger = getModuleLogger("core.runtime.analyzer");
  const analysisChunkSec = ANALYSIS_STREAM_CHUNK_SEC;

  function createStreamingSession({
    processAnalysisSegment,
  }: {
    processAnalysisSegment: (input: {
      generation: number;
      stepIndex: number;
      segment: { pcm: Float32Array; fs: number };
      final: boolean;
      completeRun: boolean;
      forceRefresh: boolean;
      reuseCFP?: boolean;
    }) => Promise<void>;
  }) {
    const targetSegmentSamples = Math.max(
      1,
      Math.round(DEFAULT_TARGET_SAMPLE_RATE * analysisChunkSec),
    );
    const pendingSamples = createSampleQueue();

    let streamEnded = false;
    let streamingDrainPromise: Promise<void> = Promise.resolve();
    let segmentIndex = 0;

    function isStreamingAborted(token: number): boolean {
      const run = store.run;
      return token !== store.generation || run?.execution.signal?.aborted === true;
    }

    async function processQueuedSegment(
      token: number,
      segment: Float32Array,
      final: boolean,
    ): Promise<void> {
      if (isStreamingAborted(token)) {
        return;
      }
      const run = store.run;
      if (!run) {
        throw new Error("runtime run is unavailable");
      }
      store.appendStreamingAudio(run, segment);
      await processAnalysisSegment({
        generation: token,
        stepIndex: segmentIndex,
        segment: {
          pcm: segment,
          fs: DEFAULT_TARGET_SAMPLE_RATE,
        },
        final,
        completeRun: final,
        forceRefresh: true,
      });
      segmentIndex += 1;
    }

    async function drain(token: number): Promise<void> {
      while (!isStreamingAborted(token)) {
        const next = takeNextStreamingSegment({
          queue: pendingSamples,
          streamEnded,
          targetSegmentSamples,
        });
        if (!next.segment || !next.segment.length) {
          break;
        }

        await processQueuedSegment(token, next.segment, next.final);
        if (isStreamingAborted(token)) {
          return;
        }

        if (next.final) {
          streamEnded = true;
          break;
        }
      }
    }

    function requestDrain(token: number): void {
      if (isStreamingAborted(token)) {
        return;
      }
      streamingDrainPromise = streamingDrainPromise
        .then(() => drain(token))
        .catch((error) => {
          if (isStreamingAborted(token)) {
            return;
          }
          throw error;
        });
    }

    return {
      get segmentIndex(): number {
        return segmentIndex;
      },
      scheduleChunk(token: number, rawChunk: readonly Float32Array[], sampleRate: number): void {
        if (isStreamingAborted(token)) {
          return;
        }
        const resampled = normalizeStreamingChunk(rawChunk, sampleRate);
        enqueueSamples(pendingSamples, resampled);
        requestDrain(token);
      },
      async flushTail(token: number): Promise<void> {
        if (isStreamingAborted(token)) {
          return;
        }
        streamEnded = true;
        requestDrain(token);
        await streamingDrainPromise.catch(() => undefined);
      },
    };
  }

  async function runStreamingAnalysis(
    token: number,
    processAnalysisSegment: (input: {
      generation: number;
      stepIndex: number;
      segment: { pcm: Float32Array; fs: number };
      final: boolean;
      completeRun: boolean;
      forceRefresh: boolean;
      reuseCFP?: boolean;
    }) => Promise<void>,
  ): Promise<void> {
    const run = store.run;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }
    const stream =
      run.input.source.kind === "stream" ? run.input.source.stream : null;
      if (!stream) {
        return;
      }

    const session = createStreamingSession({ processAnalysisSegment });
    store.setPlanIndex(0);

    emitPhaseEnd(AnalysisPhase.AUDIO, 0, {
      audio: {
        pcm: new Float32Array(0),
        fs: DEFAULT_TARGET_SAMPLE_RATE,
      },
      reuseCFP: false,
    });

    logger.info(`runtime analyzer streaming prepared: ${store.analyzerLabel}`);
    try {
      await captureAudioFromMediaStream(stream, {
        signal: run.execution.signal ?? null,
        onChunk: (chunk, sampleRate) => session.scheduleChunk(token, chunk, sampleRate),
      });
      if (token !== store.generation) {
        return;
      }
      await session.flushTail(token);
      if (token !== store.generation) {
        return;
      }
      if (session.segmentIndex === 0) {
        emitPhaseEnd(AnalysisPhase.OUTPUT, 0, {
          audio: {
            pcm: new Float32Array(0),
            fs: DEFAULT_TARGET_SAMPLE_RATE,
          },
          cfp: [],
          inference: null,
        });
      }
    } catch (error) {
      logger.warn(
        `runtime analyzer streaming failed: ${error instanceof Error ? error.message : String(error)}`,
      );
      throw error;
    }
  }

  return {
    runStreamingAnalysis,
  };
}
