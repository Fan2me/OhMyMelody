import { buildAnalysisPlan } from "./analysis-plan.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import { AnalysisPhase } from "./analysis.js";
import { AudioManager } from "./managers/audio-manager.js";
import type { AnalyzeExecutionOptions, AnalyzeInput } from "./types.js";
import type { AnalysisSessionStore } from "./analysis-session-store.js";
type AnalyzerAudioManagerLike = Pick<
  AudioManager,
  "getAudio" | "getPcmChunk"
> & {
  setAudio(input: AnalyzeInput): Promise<unknown>;
};

type AnalyzerInferenceManagerLike = {
  reset(): void;
};

export interface CreateAnalysisStepRunnerOptions {
  store: AnalysisSessionStore;
  audioManager: AnalyzerAudioManagerLike;
  inferenceManager: AnalyzerInferenceManagerLike;
  emitPhaseEnd<P extends AnalysisPhase>(
    phase: P,
    index: number,
    data: import("./analysis.js").AnalyzerPhaseEventDataMap[P],
  ): void;
  setPhase(phase: AnalysisPhase): void;
}

export function createAnalysisStepRunner({
  store,
  audioManager,
  inferenceManager,
  emitPhaseEnd,
  setPhase,
}: CreateAnalysisStepRunnerOptions) {
  const logger = getModuleLogger("core.runtime.analyzer");
  async function prepareAudioAndPlan(
    input: AnalyzeInput,
    execution: AnalyzeExecutionOptions,
  ): Promise<void> {
    store.resetActiveState(input, execution);
    inferenceManager.reset();

    setPhase(AnalysisPhase.AUDIO);
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

    const run = store.run;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }
    const audioSnapshot = decodedAudio.mode
      ? { pcm: decodedAudio.pcm, fs: decodedAudio.fs, mode: decodedAudio.mode }
      : { pcm: decodedAudio.pcm, fs: decodedAudio.fs };
    emitPhaseEnd(AnalysisPhase.AUDIO, 0, {
      audio: audioSnapshot,
      reuseCFP: run.reuseCFP,
    });
    logger.info(`runtime analyzer prepared: ${store.analyzerLabel}`);
  }

  async function runStep(
    generation: number,
    processAnalysisSegment: (input: {
      generation: number;
      stepIndex: number;
      segment: { pcm: Float32Array; fs: number };
      final: boolean;
      completeRun: boolean;
      outputIndex?: number;
      forceRefresh?: boolean;
      reuseCFP?: boolean;
    }) => Promise<void>,
  ): Promise<void> {
    if (generation !== store.generation) {
      return;
    }

    const plan = store.requirePlan();
    if (store.state.nextPlanIndex >= plan.length) {
      return;
    }

    const stepIndex = store.state.nextPlanIndex;
    const isLastStep = stepIndex + 1 >= plan.length;
    const currentPlanTask = store.requirePlanTask(plan);
    const decodedAudio = store.requireDecodedAudio();
    const run = store.run;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }
    const segment = {
      pcm: audioManager.getPcmChunk(currentPlanTask.start, currentPlanTask.end),
      fs: decodedAudio.fs,
    };

    await processAnalysisSegment({
      generation,
      stepIndex,
      segment,
      final: isLastStep,
      completeRun: isLastStep,
      outputIndex: plan.length,
      reuseCFP: run.reuseCFP,
    });

    if (generation !== store.generation) {
      return;
    }

    const runAfterSegment = store.run;
    if (!runAfterSegment) {
      throw new Error("runtime run is unavailable");
    }
    store.setPlanIndex(runAfterSegment.cfpComplete ? plan.length : stepIndex + 1);
  }

  return {
    prepareAudioAndPlan,
    runStep,
  };
}
