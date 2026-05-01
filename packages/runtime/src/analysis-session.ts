import { getModuleLogger } from "@ohm/core/logging/logger.js";
import type { AnalyzeExecutionOptions, AnalyzeInput } from "./types.js";
import type {
  AnalysisContext,
  AnalysisPlan,
  AnalysisPlanTask,
  AnalysisState,
  Analyzer,
  AnalyzerEventListener,
  AnalyzerPhaseEvent,
  InferenceResult,
} from "./analysis.js";
import { createAnalysisSessionStore } from "./analysis-session-store.js";
import { createPhaseEmitter } from "./analysis-emitter.js";
import { createAnalysisSegmentRunner } from "./analysis-segment-runner.js";
import { createAnalysisStepRunner } from "./analysis-step-runner.js";
import { createAnalysisStreamRunner } from "./analysis-stream-runner.js";
import { AudioManager } from "./managers/audio-manager.js";
import { CFPManager } from "./managers/cfp-manager.js";
import { InferenceManager } from "./managers/inference-manager.js";

type AnalyzerAudioManagerLike = Pick<
  AudioManager,
  "getAudio" | "getPcmChunk"
> & {
  setAudio(input: AnalyzeInput): Promise<unknown>;
};

type AnalyzerCFPManagerLike = {
  process(input: {
    fileKey: string;
    batchOffset?: number;
    segment: { pcm: Float32Array; fs: number };
    signal?: AbortSignal | null;
    complete?: boolean;
    forceRefresh?: boolean;
  }): Promise<{
    kind: "cache-hit" | "segment";
    fileKey: string;
    batches: readonly import("@ohm/core/cache/cfp.js").CFPBatch[];
    complete: boolean;
  }>;
};

type AnalyzerInferenceManagerLike = Pick<
  InferenceManager,
  "process" | "reset"
> & {
  hasCache?: (fileKey: string, modelName: string) => Promise<boolean>;
};

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

  const { setPhase, emitPhaseEnd } = createPhaseEmitter({
    listeners: phaseListeners,
    getContext: () => store.context,
    getState: () => store.state,
  });

  const segmentRunner = createAnalysisSegmentRunner({
    store,
    cfpManager,
    inferenceManager,
    setPhase,
    emitPhaseEnd,
  });

  const stepRunner = createAnalysisStepRunner({
    store,
    audioManager,
    inferenceManager,
    setPhase,
    emitPhaseEnd,
  });

  const streamRunner = createAnalysisStreamRunner({
    store,
    emitPhaseEnd,
  });

  let stepQueue: Promise<void> = Promise.resolve();
  let streamingAnalysisGeneration: number | null = null;

  function resetSchedulingState(): void {
    stepQueue = Promise.resolve();
    streamingAnalysisGeneration = null;
  }

  async function startStreamingAnalysis(
    input: AnalyzeInput,
    execution: AnalyzeExecutionOptions,
  ): Promise<void> {
    store.resetActiveState(input, execution);
    inferenceManager.reset();
    const token = store.generation;
    streamingAnalysisGeneration = token;
    const streamingAnalysisPromise = streamRunner
      .runStreamingAnalysis(token, segmentRunner.processAnalysisSegment)
      .then(() => undefined)
      .catch((error) => {
        if (token !== store.generation) {
          return;
        }
        throw error;
      })
      .finally(() => {
        if (token === store.generation && streamingAnalysisGeneration === token) {
          streamingAnalysisGeneration = null;
        }
      });
    await streamingAnalysisPromise;
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
      await stepRunner.prepareAudioAndPlan(input, execution);
      if (store.state.analysisPlan?.length === 0) {
        const decodedAudio = store.requireDecodedAudio();
        const audioSnapshot = decodedAudio.mode
          ? { pcm: decodedAudio.pcm, fs: decodedAudio.fs, mode: decodedAudio.mode }
          : { pcm: decodedAudio.pcm, fs: decodedAudio.fs };
        segmentRunner.emitOutputPhase(0, {
          audio: audioSnapshot,
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
      const next = stepQueue.then(() =>
        stepRunner.runStep(generation, segmentRunner.processAnalysisSegment),
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
