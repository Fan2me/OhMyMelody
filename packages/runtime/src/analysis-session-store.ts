import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { DEFAULT_TARGET_SAMPLE_RATE, concatFloat32Chunks } from "@ohm/core/audio/pcm.js";
import {
  createAnalysisState,
  resolveAnalyzeFileKey,
  type AnalysisContext,
  type AnalysisPlan,
  type AnalysisPlanTask,
  type AnalysisState,
  AnalysisPhase,
} from "./analysis.js";
import type { AnalyzeExecutionOptions, AnalyzeInput } from "./types.js";

export type AnalyzerRunState = {
  input: AnalyzeInput;
  execution: Readonly<AnalyzeExecutionOptions>;
  fileKey: string;
  cfpBatches: readonly CFPBatch[];
  cfpComplete: boolean;
  reuseCFP: boolean;
  pendingInferenceCount: number;
  outputEmitted: boolean;
  streamingAudioChunks: Float32Array[];
  streamingAudioTotalSamples: number;
};

export interface AnalysisSessionStore {
  readonly state: AnalysisState;
  readonly context: AnalysisContext | null;
  readonly run: AnalyzerRunState | null;
  readonly generation: number;
  readonly analyzerLabel: string;
  resetActiveState(
    input: AnalyzeInput,
    execution: AnalyzeExecutionOptions,
  ): void;
  requireContext(): AnalysisContext;
  requirePlan(): AnalysisPlan;
  requireDecodedAudio(): NonNullable<AnalysisState["decodedAudio"]>;
  requirePlanTask(plan: AnalysisPlan): AnalysisPlanTask;
  setPhase(phase: AnalysisPhase): AnalysisContext;
  applyCFPResult(
    run: AnalyzerRunState,
    result: {
      kind: "cache-hit" | "segment";
      fileKey: string;
      batches: readonly CFPBatch[];
      complete: boolean;
    },
  ): void;
  appendStreamingAudio(run: AnalyzerRunState, segment: Float32Array): void;
  snapshotStreamingAudio(run: AnalyzerRunState): {
    pcm: Float32Array;
    fs: number;
  };
  markOutputEmitted(run: AnalyzerRunState): void;
  setDecodedAudio(decodedAudio: NonNullable<AnalysisState["decodedAudio"]>): void;
  setAnalysisPlan(plan: AnalysisPlan): void;
  setPlanIndex(index: number): void;
}

export function createAnalysisSessionStore(label = "runtime"): AnalysisSessionStore {
  let activeState = createAnalysisState();
  let activeContext: AnalysisContext | null = null;
  let activeRun: AnalyzerRunState | null = null;
  let generation = 0;

  function requireContext(): AnalysisContext {
    const context = activeContext;
    if (!context) {
      throw new Error("runtime context is unavailable");
    }
    return context;
  }

  function requirePlan(): AnalysisPlan {
    const plan = activeState.analysisPlan;
    if (!plan) {
      throw new Error("runtime plan is unavailable");
    }
    return plan;
  }

  function requireDecodedAudio(): NonNullable<AnalysisState["decodedAudio"]> {
    const decodedAudio = activeState.decodedAudio;
    if (!decodedAudio) {
      throw new Error("decoded audio is unavailable");
    }
    return decodedAudio;
  }

  function requirePlanTask(plan: AnalysisPlan): AnalysisPlanTask {
    const task = plan[activeState.nextPlanIndex];
    if (!task) {
      throw new Error("runtime plan step is unavailable");
    }
    return task;
  }

  function setPhase(phase: AnalysisPhase): AnalysisContext {
    const context = requireContext();
    context.phase = phase;
    return context;
  }

  function resetActiveState(
    input: AnalyzeInput,
    execution: AnalyzeExecutionOptions,
  ): void {
    const previousRun = activeRun;
    const nextFileKey = resolveAnalyzeFileKey(input);
    const previousFileKey = previousRun?.fileKey ?? "";
    const forceRefresh = execution.forceRefresh === true;
    const shouldReuseCFP =
      !forceRefresh &&
      (previousRun?.cfpBatches.length ?? 0) > 0 &&
      previousRun?.cfpComplete === true &&
      nextFileKey.length > 0 &&
      nextFileKey === previousFileKey;

    generation += 1;
    activeState = createAnalysisState();
    activeRun = {
      input,
      execution,
      fileKey: nextFileKey,
      cfpBatches: shouldReuseCFP ? [...(previousRun?.cfpBatches ?? [])] : [],
      cfpComplete: shouldReuseCFP,
      reuseCFP: shouldReuseCFP,
      pendingInferenceCount: 0,
      outputEmitted: false,
      streamingAudioChunks: [],
      streamingAudioTotalSamples: 0,
    };
    activeContext = {
      state: activeState,
      phase: AnalysisPhase.AUDIO,
    };
  }

  function applyCFPResult(
    run: AnalyzerRunState,
    result: {
      kind: "cache-hit" | "segment";
      fileKey: string;
      batches: readonly CFPBatch[];
      complete: boolean;
    },
  ): void {
    run.cfpBatches =
      result.kind === "cache-hit"
        ? [...result.batches]
        : [...run.cfpBatches, ...result.batches];
    run.cfpComplete = result.complete;
  }

  function appendStreamingAudio(
    run: AnalyzerRunState,
    segment: Float32Array,
  ): void {
    run.streamingAudioChunks.push(segment);
    run.streamingAudioTotalSamples += segment.length;
  }

  function snapshotStreamingAudio(run: AnalyzerRunState): {
    pcm: Float32Array;
    fs: number;
  } {
    return {
      pcm: concatFloat32Chunks(run.streamingAudioChunks, run.streamingAudioTotalSamples),
      fs: DEFAULT_TARGET_SAMPLE_RATE,
    };
  }

  function markOutputEmitted(run: AnalyzerRunState): void {
    run.outputEmitted = true;
  }

  function setDecodedAudio(
    decodedAudio: NonNullable<AnalysisState["decodedAudio"]>,
  ): void {
    activeState.decodedAudio = decodedAudio;
  }

  function setAnalysisPlan(plan: AnalysisPlan): void {
    activeState.analysisPlan = plan;
  }

  function setPlanIndex(index: number): void {
    activeState.nextPlanIndex = Math.max(0, Math.floor(index));
  }

  return {
    get state() {
      return activeState;
    },
    get context() {
      return activeContext;
    },
    get run() {
      return activeRun;
    },
    get generation() {
      return generation;
    },
    analyzerLabel: label,
    resetActiveState,
    requireContext,
    requirePlan,
    requireDecodedAudio,
    requirePlanTask,
    setPhase,
    applyCFPResult,
    appendStreamingAudio,
    snapshotStreamingAudio,
    markOutputEmitted,
    setDecodedAudio,
    setAnalysisPlan,
    setPlanIndex,
  };
}
