import { DEFAULT_TARGET_SAMPLE_RATE } from "@ohm/core/audio/pcm.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import { captureAudioFromMediaStream } from "@ohm/core/media/stream-capture.js";
import type { AnalyzeExecutionOptions, AnalyzeInput } from "./types.js";
import type {
  Analyzer,
  AnalyzerEventListener,
} from "./analysis.js";
import { AnalysisPhase } from "./analysis.js";
import { buildAnalysisPlan } from "./analysis-plan.js";
import { createPhaseEmitter } from "./analysis-emitter.js";
import { createAnalysisSessionStore } from "./analysis-session-store.js";
import type { AnalyzerRunState } from "./analysis-session-store.js";
import {
  createDecodedAudioSegmentSource,
  createStreamingSegmentSource,
  snapshotDecodedAudio,
  type AnalyzerAudioManagerLike,
  type PcmSegment,
  type PcmSegmentSource,
  type StreamingPcmSegmentSource,
} from "./analysis-segment-source.js";
import {
  createAnalysisSessionPipeline,
  type AnalyzerCFPManagerLike,
  type AnalyzerInferenceManagerLike,
} from "./analysis-session-pipeline.js";
import { AudioManager } from "./managers/audio-manager.js";
import { CFPManager } from "./managers/cfp-manager.js";
import { InferenceManager } from "./managers/inference-manager.js";

export interface CreateAnalysisSessionOptions {
  label?: string;
  audioManager?: AnalyzerAudioManagerLike;
  cfpManager?: AnalyzerCFPManagerLike;
  inferenceManager?: AnalyzerInferenceManagerLike;
}

export function createAnalysisSession(
  opt: CreateAnalysisSessionOptions = {},
): Analyzer {
  const analyzerLabel = opt.label ?? "runtime";
  const audioManager = opt.audioManager ?? new AudioManager();
  const cfpManager = opt.cfpManager ?? new CFPManager({ label: analyzerLabel });
  const inferenceManager = opt.inferenceManager ?? new InferenceManager();
  const analyzerLogger = getModuleLogger("core.runtime.analyzer");
  const store = createAnalysisSessionStore(analyzerLabel);
  const phaseListeners = new Set<AnalyzerEventListener>();

  const { emitPhaseEnd } = createPhaseEmitter({
    listeners: phaseListeners,
    getContext: () => store.context,
    getState: () => store.state,
  });
  const pipeline = createAnalysisSessionPipeline({
    store,
    cfpManager,
    inferenceManager,
    emitPhaseEnd,
    snapshotOutputAudio,
  });

  let stepQueue: Promise<void> = Promise.resolve();
  let streamingAnalysisGeneration: number | null = null;
  let activeSegmentSource: PcmSegmentSource | null = null;

  function resetSchedulingState(): void {
    stepQueue = Promise.resolve();
    streamingAnalysisGeneration = null;
    activeSegmentSource = null;
  }

  function snapshotOutputAudio(run: AnalyzerRunState): {
    pcm: Float32Array;
    fs: number;
    mode?: string;
  } {
    return run.input.source.kind === "stream"
      ? store.snapshotStreamingAudio(run)
      : snapshotDecodedAudio(store);
  }

  function isStreamingAborted(token: number): boolean {
    const run = store.run;
    return token !== store.generation || run?.execution.signal?.aborted === true;
  }

  async function prepareAudioAndPlan(
    input: AnalyzeInput,
    execution: AnalyzeExecutionOptions,
  ): Promise<void> {
    store.resetActiveState(input, execution);
    inferenceManager.reset();

    store.setPhase(AnalysisPhase.AUDIO);
    await audioManager.setAudio(input);
    const decodedAudio = audioManager.getAudio();
    if (!decodedAudio) {
      throw new Error("decoded audio is unavailable");
    }
    store.setDecodedAudio(decodedAudio);
    store.setAnalysisPlan(
      buildAnalysisPlan(decodedAudio.pcm.length, decodedAudio.fs),
    );
    store.setPlanIndex(0);

    activeSegmentSource = createDecodedAudioSegmentSource({
      store,
      audioManager,
    });
    emitPhaseEnd(AnalysisPhase.AUDIO, 0, {
      audio: activeSegmentSource.audio,
      reuseCFP: activeSegmentSource.reuseCFP,
    });
    analyzerLogger.info(`runtime analyzer prepared: ${store.analyzerLabel}`);
  }

  async function runAnalysisFromSegments(
    generation: number,
    source: PcmSegmentSource,
    mode: "single" | "drain",
  ): Promise<void> {
    while (generation === store.generation) {
      const segment = await source.next(generation);
      if (!segment) {
        return;
      }
      appendSegmentAudio(segment);
      await pipeline.processSegment({
        generation,
        stepIndex: segment.index,
        segment: segment.segment,
        final: segment.final,
        isFinalSegment: segment.final,
        outputIndex: segment.outputIndex,
        forceRefresh: source.forceRefresh,
        reuseCFP: source.reuseCFP,
      });
      if (generation !== store.generation) {
        return;
      }
      source.afterSegment?.(segment, generation);
      if (mode === "single" || segment.final) {
        return;
      }
    }
  }

  function appendSegmentAudio(segment: PcmSegment): void {
    if (!segment.appendToStreamingAudio) {
      return;
    }
    const run = store.run;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }
    store.appendStreamingAudio(run, segment.segment.pcm);
  }

  async function runStreamingAnalysis(token: number): Promise<void> {
    const run = store.run;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }
    const stream =
      run.input.source.kind === "stream" ? run.input.source.stream : null;
    if (!stream) {
      return;
    }

    const source = createStreamingSegmentSource({
      isAborted: isStreamingAborted,
    });
    activeSegmentSource = source;
    store.setPlanIndex(0);
    emitPhaseEnd(AnalysisPhase.AUDIO, 0, {
      audio: source.audio,
      reuseCFP: source.reuseCFP,
    });

    analyzerLogger.info(
      `runtime analyzer streaming prepared: ${store.analyzerLabel}`,
    );
    try {
      const capturePromise = captureStreamingAudio(token, stream, source);
      await runAnalysisFromSegments(token, source, "drain");
      if (token !== store.generation) {
        return;
      }
      await capturePromise;
      if (source.processedCount === 0) {
        pipeline.emitOutputPhase(0, {
          audio: {
            pcm: new Float32Array(0),
            fs: DEFAULT_TARGET_SAMPLE_RATE,
          },
          cfp: [],
          inference: null,
        });
      }
    } catch (error) {
      analyzerLogger.warn(
        `runtime analyzer streaming failed: ${error instanceof Error ? error.message : String(error)}`,
      );
      throw error;
    }
  }

  function captureStreamingAudio(
    token: number,
    stream: MediaStream,
    source: StreamingPcmSegmentSource,
  ): Promise<void> {
    const run = store.run;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }
    return captureAudioFromMediaStream(stream, {
      signal: run.execution.signal ?? null,
      onChunk: (chunk, sampleRate) =>
        source.scheduleChunk(token, chunk, sampleRate),
    }).then(
      () => {
        source.finish();
      },
      (error) => {
        source.fail(error);
      },
    );
  }

  async function startStreamingAnalysis(
    input: AnalyzeInput,
    execution: AnalyzeExecutionOptions,
  ): Promise<void> {
    store.resetActiveState(input, execution);
    inferenceManager.reset();
    const token = store.generation;
    streamingAnalysisGeneration = token;
    await runStreamingAnalysis(token)
      .then(() => undefined)
      .catch((error) => {
        if (token !== store.generation) {
          return;
        }
        throw error;
      })
      .finally(() => {
        if (
          token === store.generation &&
          streamingAnalysisGeneration === token
        ) {
          streamingAnalysisGeneration = null;
        }
      });
  }

  return {
    subscribe(listener: AnalyzerEventListener): () => void {
      phaseListeners.add(listener);
      return () => {
        phaseListeners.delete(listener);
      };
    },
    async setAudio(
      input: AnalyzeInput,
      execution: AnalyzeExecutionOptions = {},
    ): Promise<void> {
      resetSchedulingState();
      if (input.source.kind === "stream") {
        await startStreamingAnalysis(input, execution);
        return;
      }
      await prepareAudioAndPlan(input, execution);
      if (store.state.analysisPlan?.length === 0) {
        pipeline.emitOutputPhase(0, {
          audio: snapshotDecodedAudio(store),
          cfp: [],
          inference: null,
        });
        analyzerLogger.info(
          `runtime analyzer ready with empty plan: ${store.analyzerLabel}`,
        );
      }
    },
    step(): Promise<void> {
      if (
        streamingAnalysisGeneration !== null &&
        streamingAnalysisGeneration === store.generation
      ) {
        return Promise.resolve();
      }
      const generation = store.generation;
      const source = activeSegmentSource;
      if (!source) {
        throw new Error("runtime segment source is unavailable");
      }
      const next = stepQueue.then(() =>
        runAnalysisFromSegments(generation, source, "single"),
      );
      stepQueue = next.then(
        () => undefined,
        () => undefined,
      );
      return next;
    },
  };
}

export type {
  AnalysisContext,
  AnalysisPlan,
  AnalysisPlanTask,
  AnalysisState,
  AnalyzerEventListener,
  AnalyzerPhaseEvent,
  InferenceResult,
} from "./analysis.js";
export { AnalysisPhase } from "./analysis.js";
