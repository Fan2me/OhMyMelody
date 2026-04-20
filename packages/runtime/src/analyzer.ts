import type { CFPBatch } from "@ohm/core/cache/cache.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import { toPositiveFinite } from "@ohm/core/cfp/common.js";
import {
  createAnalysisState,
  type AnalyzerEventListener,
  type AnalyzerPhaseEventDataMap,
  type AnalyzerPhaseEvent,
  type AnalyzerAudioEventData,
  type AnalyzerCfpEventData,
  type AnalyzerInferenceEventData,
  type AnalyzerOutputEventData,
  type AnalysisPlan,
  type AnalysisPlanTask,
  type AnalysisState,
  type AnalysisContext,
  type InferenceResult,
  type Analyzer,
  AnalysisPhase,
} from "./analysis.js";
import type { AnalyzeExecutionOptions, AnalyzeInput } from "./types.js";
import { AudioManager } from "./managers/audio-manager.js";
import { CFPManager } from "./managers/cfp-manager.js";
import { InferenceManager } from "./managers/inference-manager.js";

type AnalyzerAudioManagerLike = Pick<
  AudioManager,
  "setAudio" | "getAudio" | "getPcmChunk"
>;

type AnalyzerCFPManagerLike = {
  process(input: {
    input: AnalyzeInput;
    execution: Readonly<AnalyzeExecutionOptions>;
    previousBatches: readonly CFPBatch[];
    segment: { pcm: Float32Array; fs: number };
    signal: AbortSignal | null;
    complete: boolean;
  }): Promise<{
    fileKey: string;
    batches: readonly CFPBatch[];
    allBatches: readonly CFPBatch[];
    complete: boolean;
  }>;
};

type AnalyzerInferenceManagerLike = Pick<
  InferenceManager,
  "process" | "reset"
> & {
  hasCache?: (fileKey: string, modelName: string, expectedBatchCount?: number) => Promise<boolean>;
};

export interface CreateAnalyzerOptions {
  audioManager?: AnalyzerAudioManagerLike;
  cfpManager?: AnalyzerCFPManagerLike;
  inferenceManager?: AnalyzerInferenceManagerLike;
}

export type {
  AnalysisPlan,
  AnalysisPlanTask,
  AnalysisState,
  AnalysisContext,
  Analyzer,
  AnalyzerEventListener,
  AnalyzerPhaseEvent,
} from "./analysis.js";

export { AnalysisPhase } from "./analysis.js";

export { AudioManager } from "./managers/audio-manager.js";
export { CFPManager } from "./managers/cfp-manager.js";
export { InferenceManager } from "./managers/inference-manager.js";

const analyzerLogger = getModuleLogger("core.runtime.analyzer");

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function buildAnalysisPlan(
  totalSamples: number,
  sampleRate: number,
  options: {
    chunkSec?: number;
    steadyChunkSec?: number;
    adaptiveChunkSec?: boolean;
    firstChunkSec?: number;
    warmupChunkSec?: number;
    warmupChunks?: number;
    rampChunks?: number;
  } = {},
  batchSec = 10,
): AnalysisPlan {
  const total = Math.max(0, Math.floor(toPositiveFinite(totalSamples, 0) ?? 0));
  const sr = Math.max(1, Math.floor(toPositiveFinite(sampleRate, 1) ?? 1));
  const explicitChunkSec = toPositiveFinite(options.chunkSec, null);
  const defaultSteadyChunkSec = clampNumber(
    toPositiveFinite(
      options.steadyChunkSec,
      toPositiveFinite(batchSec, 8) ?? 8,
    ) ?? 8,
    1,
    100,
  );
  const shouldAdaptive = options.adaptiveChunkSec === true;

  const tasks: AnalysisPlanTask[] = [];
  if (total <= 0) {
    return tasks;
  }

  if (explicitChunkSec) {
    const fixedSamples = Math.max(
      1,
      Math.floor(sr * clampNumber(explicitChunkSec, 0.5, 30)),
    );
    for (let start = 0; start < total; start += fixedSamples) {
      const end = Math.min(start + fixedSamples, total);
      tasks.push({ start, end });
    }
    return tasks;
  }

  if (!shouldAdaptive) {
    const fixedSamples = Math.max(1, Math.floor(sr * defaultSteadyChunkSec));
    for (let start = 0; start < total; start += fixedSamples) {
      const end = Math.min(start + fixedSamples, total);
      tasks.push({ start, end });
    }
    return tasks;
  }

  const steadySec = defaultSteadyChunkSec;
  const firstSec = clampNumber(
    toPositiveFinite(options.firstChunkSec, 0.8) ?? 0.8,
    0.5,
    steadySec,
  );
  const warmupSec = clampNumber(
    toPositiveFinite(options.warmupChunkSec, 1.6) ?? 1.6,
    firstSec,
    steadySec,
  );
  const warmupChunks = Math.max(
    1,
    Math.floor(toPositiveFinite(options.warmupChunks, 2) ?? 2),
  );
  const rampChunks = Math.max(
    0,
    Math.floor(toPositiveFinite(options.rampChunks, 2) ?? 2),
  );

  let start = 0;
  let chunkIdx = 0;
  while (start < total) {
    let sec = steadySec;
    if (chunkIdx === 0) {
      sec = firstSec;
    } else if (chunkIdx < warmupChunks) {
      sec = warmupSec;
    } else if (chunkIdx < warmupChunks + rampChunks) {
      const rampIdx = chunkIdx - warmupChunks + 1;
      const rampFrac = rampIdx / (rampChunks + 1);
      sec = warmupSec + (steadySec - warmupSec) * rampFrac;
    }
    const size = Math.max(1, Math.floor(sr * sec));
    const end = Math.min(start + size, total);
    tasks.push({ start, end });
    start = end;
    chunkIdx += 1;
  }

  return tasks;
}

export function createAnalyzer(
  options: CreateAnalyzerOptions = {},
): Analyzer {
  const analyzerLabel = "runtime";
  const batchSec = 4;

  const audioManager = options.audioManager ?? new AudioManager();
  const cfpManager = options.cfpManager ?? new CFPManager({ label: analyzerLabel });
  const inferenceManager =
    options.inferenceManager ?? new InferenceManager();
  const phaseListeners = new Set<AnalyzerEventListener>();

  let activeState = createAnalysisState();
  let activeContext: AnalysisContext | null = null;
  let lastInferenceResult: InferenceResult | null = null;
  let currentCFPBatches: readonly CFPBatch[] = [];
  let currentCFPComplete = false;
  let currentInput: AnalyzeInput | null = null;
  let currentExecution: Readonly<AnalyzeExecutionOptions> | null = null;
  let reuseCFPBatchesForCurrentRun = false;
  let outputEmitted = false;
  let analysisGeneration = 0;
  let stepQueue: Promise<void> = Promise.resolve();
  let inferenceQueue: Promise<void> = Promise.resolve();

  function resetActiveState(
    input: AnalyzeInput,
    execution: AnalyzeExecutionOptions,
  ): void {
    const previousInput = currentInput;
    const previousCFPBatches = currentCFPBatches;
    const previousCFPComplete = currentCFPComplete;
    const nextFileKey = String(input.fileKey || input.source.label || "").trim();
    const previousFileKey = String(previousInput?.fileKey || previousInput?.source.label || "").trim();
    const shouldReuseCFP =
      execution.forceRefresh !== true &&
      previousCFPBatches.length > 0 &&
      previousCFPComplete === true &&
      nextFileKey.length > 0 &&
      nextFileKey === previousFileKey;

    analysisGeneration += 1;
    stepQueue = Promise.resolve();
    inferenceQueue = Promise.resolve();
    activeState = createAnalysisState();
    currentCFPBatches = shouldReuseCFP ? [...previousCFPBatches] : [];
    currentCFPComplete = shouldReuseCFP;
    reuseCFPBatchesForCurrentRun = shouldReuseCFP;
    currentInput = input;
    currentExecution = execution;
    outputEmitted = false;
    inferenceManager.reset();
    lastInferenceResult = null;
    activeContext = {
      state: activeState,
      phase: AnalysisPhase.AUDIO,
    };
  }

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

  function emitPhaseEnd<P extends AnalysisPhase>(
    phase: P,
    index: number,
    data: AnalyzerPhaseEventDataMap[P],
  ): void {
    const context = {
      ...requireContext(),
      phase,
    };
    let summary = `phase=${phase} index=${index}`;
    switch (phase) {
      case AnalysisPhase.AUDIO:
        {
          const payload = data as AnalyzerAudioEventData;
          summary = `phase=${phase} index=${index} audio pcm=${payload.audio.pcm.length} fs=${payload.audio.fs}`;
        }
        break;
      case AnalysisPhase.CFP:
        {
          const payload = data as AnalyzerCfpEventData;
          summary = `phase=${phase} index=${index} cfp batches=${payload.cfp.length} complete=${payload.complete}`;
        }
        break;
      case AnalysisPhase.INFERENCE:
        {
          const payload = data as AnalyzerInferenceEventData;
          summary = `phase=${phase} index=${index} cfp batches=${payload.cfp.length} inferenceFrames=${payload.inference.totalExpectedFrames}`;
        }
        break;
      case AnalysisPhase.OUTPUT:
        {
          const payload = data as AnalyzerOutputEventData;
          summary = `phase=${phase} index=${index} audio pcm=${payload.audio.pcm.length} fs=${payload.audio.fs} cfp batches=${payload.cfp.length} inferenceFrames=${payload.inference?.totalExpectedFrames ?? 0}`;
        }
        break;
    }
    analyzerLogger.info(`runtime analyzer emit phase-end: ${summary}`);
    const event = {
      type: "phase-end",
      phase,
      index,
      context: { ...context },
      state: activeState,
      data,
    } as AnalyzerPhaseEvent;
    for (const listener of phaseListeners) {
      try {
        listener(event);
      } catch (error) {
        analyzerLogger.warn(
          `runtime analyzer phase listener failed: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    }
  }

  function emitOutputPhase(index: number): void {
    if (outputEmitted) {
      return;
    }
    const audio = requireDecodedAudio();
    const audioSnapshot = audio.mode
      ? { pcm: audio.pcm, fs: audio.fs, mode: audio.mode }
      : { pcm: audio.pcm, fs: audio.fs };
    setPhase(AnalysisPhase.OUTPUT);
    emitPhaseEnd(AnalysisPhase.OUTPUT, index, {
      audio: audioSnapshot,
      cfp: currentCFPBatches,
      inference: lastInferenceResult,
    });
    outputEmitted = true;
  }

  async function prepareAudioAndPlan(
    input: AnalyzeInput,
    execution: Readonly<AnalyzeExecutionOptions>,
  ): Promise<void> {
    resetActiveState(input, execution);
    const context = requireContext();

    setPhase(AnalysisPhase.AUDIO);
    await audioManager.setAudio(input);

    const decodedAudio = audioManager.getAudio();
    if (!decodedAudio) {
      throw new Error("decoded audio is unavailable");
    }
    context.state.decodedAudio = decodedAudio;
    context.state.analysisPlan = buildAnalysisPlan(
      decodedAudio.pcm.length,
      decodedAudio.fs,
      {
        adaptiveChunkSec: true,
        steadyChunkSec: batchSec,
        firstChunkSec: 0.8,
        warmupChunkSec: 1.6,
        warmupChunks: 2,
        rampChunks: 3,
      },
      batchSec,
    );

    const analysisPlan = context.state.analysisPlan;
    if (!analysisPlan) {
      throw new Error("runtime plan is unavailable");
    }

    context.state.nextPlanIndex = 0;

    const audioSnapshot = decodedAudio.mode
      ? { pcm: decodedAudio.pcm, fs: decodedAudio.fs, mode: decodedAudio.mode }
      : { pcm: decodedAudio.pcm, fs: decodedAudio.fs };
    emitPhaseEnd(AnalysisPhase.AUDIO, 0, {
      audio: audioSnapshot,
      reuseCFP: reuseCFPBatchesForCurrentRun,
    });
    analyzerLogger.info(`runtime analyzer prepared: ${analyzerLabel}`);
    if (analysisPlan.length === 0) {
      emitOutputPhase(0);
      analyzerLogger.info(
        `runtime analyzer ready with empty plan: ${analyzerLabel}`,
      );
    }
  }

  async function runStep(generation: number): Promise<void> {
    const input = currentInput;
    const execution = currentExecution;
    if (!input || !execution) {
      throw new Error("runtime input is unavailable");
    }
    if (generation !== analysisGeneration) {
      return;
    }

    const plan = requirePlan();
    if (activeState.nextPlanIndex >= plan.length) {
      return;
    }

    const stepIndex = activeState.nextPlanIndex;
    const isLastStep = stepIndex + 1 >= plan.length;
    const currentPlanTask = requirePlanTask(plan);
    const decodedAudio = requireDecodedAudio();
    const segment = {
      pcm: audioManager.getPcmChunk(currentPlanTask.start, currentPlanTask.end),
      fs: decodedAudio.fs,
    };

    let cfpResult:
      | {
          fileKey: string;
          batches: readonly CFPBatch[];
          allBatches: readonly CFPBatch[];
          complete: boolean;
        }
      | null = null;

    if (reuseCFPBatchesForCurrentRun && stepIndex === 0 && currentCFPBatches.length > 0) {
      cfpResult = {
        fileKey: String(input.fileKey || input.source.label || "").trim(),
        batches: currentCFPBatches,
        allBatches: currentCFPBatches,
        complete: true,
      };
      analyzerLogger.info(
        `runtime analyzer reuse cfp batches: count=${currentCFPBatches.length}`,
      );
    } else {
      setPhase(AnalysisPhase.CFP);
      cfpResult = await cfpManager.process({
        input,
        execution,
        previousBatches: currentCFPBatches,
        segment,
        signal: execution.signal ?? null,
        complete: isLastStep,
      });
    }

    if (generation !== analysisGeneration) {
      return;
    }

    currentCFPBatches = [...cfpResult.allBatches];
    currentCFPComplete = cfpResult.complete === true;
    activeState.nextPlanIndex = cfpResult.complete
      ? plan.length
      : stepIndex + 1;
    if (!(reuseCFPBatchesForCurrentRun && stepIndex === 0)) {
      emitPhaseEnd(AnalysisPhase.CFP, stepIndex, {
        cfp: cfpResult.batches,
        allCfp: currentCFPBatches,
        complete: cfpResult.complete,
      });
    }
    inferenceQueue = inferenceQueue
      .then(async () => {
        if (generation !== analysisGeneration) {
          return;
        }
        setPhase(AnalysisPhase.INFERENCE);
        const fileKey = String(input.fileKey || input.source.label || "").trim();
        analyzerLogger.info(
          `runtime analyzer inference start: model=${input.model.name} batches=${cfpResult.batches.length}`,
        );
        const completeRun = isLastStep || cfpResult.complete;
        let inferenceCacheHit = false;
        const shouldProbeInferenceCache =
          execution.allowCache !== false &&
          execution.forceRefresh !== true &&
          cfpResult.complete &&
          cfpResult.batches.length > 1 &&
          typeof inferenceManager.hasCache === "function" &&
          fileKey.length > 0;

        if (shouldProbeInferenceCache) {
          try {
            inferenceCacheHit = await inferenceManager.hasCache!(fileKey, input.model.name, cfpResult.batches.length);
          } catch {
            inferenceCacheHit = false;
          }
        }

        const shouldProgressiveReplay =
          cfpResult.complete &&
          cfpResult.batches.length > 1 &&
          !inferenceCacheHit;

        if (shouldProgressiveReplay) {
          analyzerLogger.info(
            `runtime analyzer inference progressive replay: model=${input.model.name} batches=${cfpResult.batches.length}`,
          );
          for (let batchIndex = 0; batchIndex < cfpResult.batches.length; batchIndex += 1) {
            if (generation !== analysisGeneration) {
              return;
            }
            const batch = cfpResult.batches[batchIndex];
            if (!batch) {
              continue;
            }
            const isLastBatch = batchIndex + 1 >= cfpResult.batches.length;
            const inferenceResult = await inferenceManager.process({
              batches: [batch],
              modelName: input.model.name,
              fileKey,
              allowCache: execution.allowCache !== false,
              forceRefresh: execution.forceRefresh === true,
              complete: completeRun && isLastBatch,
            });
            if (generation !== analysisGeneration) {
              return;
            }
            lastInferenceResult = inferenceResult;
            emitPhaseEnd(AnalysisPhase.INFERENCE, stepIndex, {
              cfp: [batch],
              allCfp: currentCFPBatches,
              inference: inferenceResult,
            });
          }
          analyzerLogger.info(
            `runtime analyzer inference progressive replay done: model=${input.model.name} batches=${cfpResult.batches.length}`,
          );
        } else {
          const inferenceResult = await inferenceManager.process({
            batches: cfpResult.batches,
            modelName: input.model.name,
            fileKey,
            allowCache: execution.allowCache !== false,
            forceRefresh: execution.forceRefresh === true,
            complete: completeRun,
          });
          if (generation !== analysisGeneration) {
            return;
          }
          lastInferenceResult = inferenceResult;
          analyzerLogger.info(
            `runtime analyzer inference done: model=${input.model.name} batches=${inferenceResult.totalBatchCount}`,
          );
          emitPhaseEnd(AnalysisPhase.INFERENCE, stepIndex, {
            cfp: cfpResult.batches,
            allCfp: currentCFPBatches,
            inference: inferenceResult,
          });
        }
        if (completeRun) {
          emitOutputPhase(plan.length);
        } else {
          setPhase(AnalysisPhase.CFP);
        }
      })
      .then(
        () => undefined,
        () => undefined,
      );
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
      await prepareAudioAndPlan(input, execution);
    },
    step(): Promise<void> {
      const generation = analysisGeneration;
      const next = stepQueue.then(() => runStep(generation));
      stepQueue = next.then(
        () => undefined,
        () => undefined,
      );
      return next;
    },
  };
}
