import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import { toPositiveFinite } from "@ohm/core/cfp/common.js";
import {
  DEFAULT_TARGET_SAMPLE_RATE,
  concatFloat32Chunks,
  mixDownChannelsToMono,
  postProcessDecodedAudio,
} from "@ohm/core/audio/pcm.js";
import { captureAudioFromMediaStream } from "@ohm/core/media/stream-capture.js";
import {
  createAnalysisState,
  resolveAnalyzeFileKey,
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
    fileKey: string;
    batchOffset?: number;
    segment: { pcm: Float32Array; fs: number };
    signal?: AbortSignal | null;
    complete?: boolean;
    allowCache?: boolean;
    forceRefresh?: boolean;
  }): Promise<{
    kind: "cache-hit" | "segment";
    fileKey: string;
    batches: readonly CFPBatch[];
    complete: boolean;
  }>;
};

type AnalyzerInferenceManagerLike = Pick<
  InferenceManager,
  "process" | "reset"
> & {
  hasCache?: (fileKey: string, modelName: string) => Promise<boolean>;
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

type SampleQueue = {
  chunks: Float32Array[];
  totalSamples: number;
};

type AnalyzerRunState = {
  input: AnalyzeInput;
  execution: Readonly<AnalyzeExecutionOptions>;
  fileKey: string;
  cfpBatches: readonly CFPBatch[];
  cfpComplete: boolean;
  reuseCFP: boolean;
  outputEmitted: boolean;
  lastInferenceResult: InferenceResult | null;
  streamingAudioChunks: Float32Array[];
  streamingAudioTotalSamples: number;
};

type AnalyzerCFPProcessResult = Awaited<
  ReturnType<AnalyzerCFPManagerLike["process"]>
>;
function createSampleQueue(): SampleQueue {
  return {
    chunks: [],
    totalSamples: 0,
  };
}

function enqueueSamples(queue: SampleQueue, samples: Float32Array): void {
  if (!samples.length) {
    return;
  }
  queue.chunks.push(samples);
  queue.totalSamples += samples.length;
}

function takeSamples(
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

function drainSamples(queue: SampleQueue): Float32Array {
  if (!queue.totalSamples || !queue.chunks.length) {
    return new Float32Array(0);
  }
  return takeSamples(queue, queue.totalSamples) ?? new Float32Array(0);
}

function padSamplesToLength(
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
  const warmUpChunkSec = 1.28;
  const warmUpChunkSize = warmUpChunkSec * sr;
  const warmUpRounds = 3;
  const steadyChunkSec = 5.12;
  const steadyChunkSize = steadyChunkSec * sr;

  const tasks: AnalysisPlanTask[] = [];
  if (total <= 0) {
    return tasks;
  }

  let start = 0;
  for (let round = 0; round < warmUpRounds && start < total; round++) {
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

export function createAnalyzer(opt: CreateAnalyzerOptions = {}): Analyzer {
  const analyzerLabel = "runtime";
  const analysisChunkSec = 1.28;

  const audioManager = opt.audioManager ?? new AudioManager();
  const cfpManager = opt.cfpManager ?? new CFPManager({ label: analyzerLabel });
  const inferenceManager = opt.inferenceManager ?? new InferenceManager();
  const phaseListeners = new Set<AnalyzerEventListener>();

  let activeState = createAnalysisState();
  let activeContext: AnalysisContext | null = null;
  let activeRun: AnalyzerRunState | null = null;
  let analysisGeneration = 0;
  let stepQueue: Promise<void> = Promise.resolve();
  let inferenceQueue: Promise<void> = Promise.resolve();
  let streamingAnalysisActive = false;
  let streamingAnalysisPromise: Promise<void> | null = null;

  function resetActiveState(
    input: AnalyzeInput,
    execution: AnalyzeExecutionOptions,
  ): void {
    const previousRun = activeRun;
    const nextFileKey = resolveAnalyzeFileKey(input);
    const previousFileKey = previousRun?.fileKey ?? "";
    const shouldReuseCFP =
      execution.forceRefresh !== true &&
      (previousRun?.cfpBatches.length ?? 0) > 0 &&
      previousRun?.cfpComplete === true &&
      nextFileKey.length > 0 &&
      nextFileKey === previousFileKey;

    analysisGeneration += 1;
    stepQueue = Promise.resolve();
    inferenceQueue = Promise.resolve();
    activeState = createAnalysisState();
    activeRun = {
      input,
      execution,
      fileKey: nextFileKey,
      cfpBatches: shouldReuseCFP ? [...(previousRun?.cfpBatches ?? [])] : [],
      cfpComplete: shouldReuseCFP,
      reuseCFP: shouldReuseCFP,
      outputEmitted: false,
      lastInferenceResult: null,
      streamingAudioChunks: [],
      streamingAudioTotalSamples: 0,
    };
    inferenceManager.reset();
    streamingAnalysisActive = false;
    streamingAnalysisPromise = null;
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

  function requireRun(): AnalyzerRunState {
    const run = activeRun;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }
    return run;
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

  function applyCFPResult(
    run: AnalyzerRunState,
    result: AnalyzerCFPProcessResult,
  ): void {
    run.cfpBatches =
      result.kind === "cache-hit"
        ? [...result.batches]
        : [...run.cfpBatches, ...result.batches];
    run.cfpComplete = result.complete;
  }

  function emitCFPPhase(
    index: number,
    result: AnalyzerCFPProcessResult,
  ): void {
    const run = requireRun();
    emitPhaseEnd(AnalysisPhase.CFP, index, {
      cfp: result.batches,
      allCfp: run.cfpBatches,
      complete: result.complete,
    });
  }

  async function emitInferenceProgress({
    generation,
    stepIndex,
    batches,
    completeRun,
  }: {
    generation: number;
    stepIndex: number;
    batches: readonly CFPBatch[];
    completeRun: boolean;
  }): Promise<void> {
    const run = requireRun();
    setPhase(AnalysisPhase.INFERENCE);
    analyzerLogger.info(
      `runtime analyzer inference start: model=${run.input.model.name}`,
    );

    let inferenceCacheHit = false;
    const shouldProbeInferenceCache =
      run.execution.allowCache !== false &&
      run.execution.forceRefresh !== true &&
      completeRun &&
      batches.length > 1 &&
      typeof inferenceManager.hasCache === "function" &&
      run.fileKey.length > 0;

    if (shouldProbeInferenceCache) {
      try {
        inferenceCacheHit = await inferenceManager.hasCache!(
          run.fileKey,
          run.input.model.name,
        );
      } catch {
        inferenceCacheHit = false;
      }
    }

    const shouldProgressiveReplay =
      completeRun &&
      batches.length > 1 &&
      !inferenceCacheHit;

    if (shouldProgressiveReplay) {
      analyzerLogger.info(
        `runtime analyzer inference progressive replay: model=${run.input.model.name}`,
      );
      for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
        if (generation !== analysisGeneration) {
          return;
        }
        const batch = batches[batchIndex];
        if (!batch) {
          continue;
        }
        const isLastBatch = batchIndex + 1 >= batches.length;
        const inferenceResult = await inferenceManager.process({
          batches: [batch],
          modelName: run.input.model.name,
          fileKey: run.fileKey,
          allowCache: run.execution.allowCache !== false,
          forceRefresh: run.execution.forceRefresh === true,
          complete: completeRun && isLastBatch,
        });
        if (generation !== analysisGeneration) {
          return;
        }
        run.lastInferenceResult = inferenceResult;
        emitPhaseEnd(AnalysisPhase.INFERENCE, stepIndex, {
          cfp: [batch],
          allCfp: run.cfpBatches,
          inference: inferenceResult,
        });
      }
      analyzerLogger.info(
        `runtime analyzer inference progressive replay done: model=${run.input.model.name}`,
      );
      return;
    }

    const inferenceResult = await inferenceManager.process({
      batches,
      modelName: run.input.model.name,
      fileKey: run.fileKey,
      allowCache: run.execution.allowCache !== false,
      forceRefresh: run.execution.forceRefresh === true,
      complete: completeRun,
    });
    if (generation !== analysisGeneration) {
      return;
    }
    run.lastInferenceResult = inferenceResult;
    analyzerLogger.info(
      `runtime analyzer inference done: model=${run.input.model.name}`,
    );
    emitPhaseEnd(AnalysisPhase.INFERENCE, stepIndex, {
      cfp: batches,
      allCfp: run.cfpBatches,
      inference: inferenceResult,
    });
  }

  async function applyCFPAndRunInference({
    generation,
    stepIndex,
    result,
    completeRun,
    emitCFP = true,
  }: {
    generation: number;
    stepIndex: number;
    result: AnalyzerCFPProcessResult;
    completeRun: boolean;
    emitCFP?: boolean;
  }): Promise<void> {
    if (generation !== analysisGeneration) {
      return;
    }
    const run = requireRun();
    applyCFPResult(run, result);
    if (emitCFP) {
      emitCFPPhase(stepIndex, result);
    }
    await emitInferenceProgress({
      generation,
      stepIndex,
      batches: result.batches,
      completeRun,
    });
    if (generation !== analysisGeneration) {
      return;
    }
    if (completeRun) {
      emitOutputPhase(
        activeState.analysisPlan?.length ?? stepIndex + 1,
      );
    } else {
      setPhase(AnalysisPhase.CFP);
    }
  }

  async function processStreamingSegment({
    token,
    segment,
    index,
    final,
  }: {
    token: number;
    segment: Float32Array;
    index: number;
    final: boolean;
  }): Promise<void> {
    if (token !== analysisGeneration || !segment.length) {
      return;
    }

    const run = requireRun();

    const segmentResult = await processCFPSegment(
      run,
      {
        pcm: segment,
        fs: DEFAULT_TARGET_SAMPLE_RATE,
      },
      final,
      {
        allowCache: false,
        forceRefresh: true,
      },
    );

    if (token !== analysisGeneration) {
      return;
    }

    await applyCFPAndRunInference({
      generation: token,
      stepIndex: index,
      result: {
        ...segmentResult,
        kind: "segment",
      },
      completeRun: final,
    });
  }

  async function processCFPSegment(
    run: AnalyzerRunState,
    segment: { pcm: Float32Array; fs: number },
    complete: boolean,
    options: {
      allowCache: boolean;
      forceRefresh: boolean;
    },
  ): Promise<AnalyzerCFPProcessResult> {
    return await cfpManager.process({
      fileKey: run.fileKey,
      batchOffset: run.cfpBatches.length,
      segment,
      signal: run.execution.signal ?? null,
      complete,
      allowCache: options.allowCache,
      forceRefresh: options.forceRefresh,
    });
  }

  function finalizeStreamingOutput(token: number, segmentIndex: number): void {
    const run = requireRun();
    if (token !== analysisGeneration || run.outputEmitted) {
      return;
    }
    emitPhaseEnd(AnalysisPhase.OUTPUT, segmentIndex, {
      audio: buildStreamingAudioSnapshot(),
      cfp: run.cfpBatches,
      inference: run.lastInferenceResult,
    });
    run.outputEmitted = true;
  }

  function buildStreamingAudioSnapshot(): { pcm: Float32Array; fs: number } {
    const run = requireRun();
    return {
      pcm: concatFloat32Chunks(
        run.streamingAudioChunks,
        run.streamingAudioTotalSamples,
      ),
      fs: DEFAULT_TARGET_SAMPLE_RATE,
    };
  }

  function appendStreamingAudio(run: AnalyzerRunState, segment: Float32Array): void {
    run.streamingAudioChunks.push(segment);
    run.streamingAudioTotalSamples += segment.length;
  }

  function normalizeStreamingChunk(
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

  function takeNextStreamingSegment({
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

  async function runStreamingAnalysis(token: number): Promise<void> {
    const run = requireRun();
    const stream =
      run.input.source.kind === "stream" ? run.input.source.stream : null;
    if (!stream) {
      return;
    }
    const isStreamingAborted = (): boolean =>
      token !== analysisGeneration || run.execution.signal?.aborted === true;

    streamingAnalysisActive = true;
    const targetSegmentSamples = Math.max(
      1,
      Math.round(DEFAULT_TARGET_SAMPLE_RATE * analysisChunkSec),
    );
    run.streamingAudioChunks = [];
    run.streamingAudioTotalSamples = 0;
    const streamingSession = (() => {
      const pendingSamples = createSampleQueue();
      let streamEnded = false;
      let streamingDrainPromise: Promise<void> = Promise.resolve();
      let segmentIndex = 0;

      async function processQueuedSegment(
        segment: Float32Array,
        final: boolean,
      ): Promise<void> {
        if (isStreamingAborted()) {
          return;
        }
        appendStreamingAudio(run, segment);
        await processStreamingSegment({
          token,
          segment,
          index: segmentIndex,
          final,
        });
        segmentIndex += 1;
      }

      async function drain(): Promise<void> {
        while (!isStreamingAborted()) {
          const next = takeNextStreamingSegment({
            queue: pendingSamples,
            streamEnded,
            targetSegmentSamples,
          });
          if (!next.segment || !next.segment.length) {
            break;
          }

          await processQueuedSegment(next.segment, next.final);
          if (isStreamingAborted()) {
            return;
          }

          if (next.final) {
            streamEnded = true;
            break;
          }
        }
      }

      function requestDrain(): void {
        if (isStreamingAborted()) {
          return;
        }
        streamingDrainPromise = streamingDrainPromise
          .then(() => drain())
          .catch((error) => {
            if (isStreamingAborted()) {
              return;
            }
            throw error;
          });
      }

      return {
        get segmentIndex(): number {
          return segmentIndex;
        },
        scheduleChunk(
          rawChunk: readonly Float32Array[],
          sampleRate: number,
        ): void {
          if (isStreamingAborted()) {
            return;
          }
          const resampled = normalizeStreamingChunk(rawChunk, sampleRate);
          enqueueSamples(pendingSamples, resampled);
          requestDrain();
        },
        async flushTail(): Promise<void> {
          if (isStreamingAborted()) {
            return;
          }

          streamEnded = true;
          requestDrain();
          await streamingDrainPromise.catch(() => undefined);
        },
      };
    })();

    emitPhaseEnd(AnalysisPhase.AUDIO, 0, {
      audio: {
        pcm: new Float32Array(0),
        fs: DEFAULT_TARGET_SAMPLE_RATE,
      },
      reuseCFP: false,
    });

    try {
      await captureAudioFromMediaStream(stream, {
        signal: run.execution.signal ?? null,
        onChunk: streamingSession.scheduleChunk,
      });
      if (isStreamingAborted()) {
        return;
      }
      await streamingSession.flushTail();
      if (isStreamingAborted()) {
        return;
      }
      finalizeStreamingOutput(token, streamingSession.segmentIndex);
    } finally {
      streamingAnalysisActive = false;
      streamingAnalysisPromise = null;
    }
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
          summary = `phase=${phase} index=${index} cfp batches=${payload.cfp.length}`;
        }
        break;
      case AnalysisPhase.OUTPUT:
        {
          const payload = data as AnalyzerOutputEventData;
          summary = `phase=${phase} index=${index} audio pcm=${payload.audio.pcm.length} fs=${payload.audio.fs} cfp batches=${payload.cfp.length}`;
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
    const run = requireRun();
    if (run.outputEmitted) {
      return;
    }
    const audio = requireDecodedAudio();
    const audioSnapshot = audio.mode
      ? { pcm: audio.pcm, fs: audio.fs, mode: audio.mode }
      : { pcm: audio.pcm, fs: audio.fs };
    setPhase(AnalysisPhase.OUTPUT);
    emitPhaseEnd(AnalysisPhase.OUTPUT, index, {
      audio: audioSnapshot,
      cfp: run.cfpBatches,
      inference: run.lastInferenceResult,
    });
    run.outputEmitted = true;
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
      reuseCFP: requireRun().reuseCFP,
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
    const run = requireRun();
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

    let cfpResult: AnalyzerCFPProcessResult | null = null;

    if (run.reuseCFP && stepIndex === 0 && run.cfpBatches.length > 0) {
      cfpResult = {
        kind: "cache-hit",
        fileKey: run.fileKey,
        batches: run.cfpBatches,
        complete: true,
      };
      analyzerLogger.info(
        `runtime analyzer reuse cfp batches: count=${run.cfpBatches.length}`,
      );
    } else {
      setPhase(AnalysisPhase.CFP);
      cfpResult = await processCFPSegment(
        run,
        segment,
        isLastStep,
        {
          allowCache: run.execution.allowCache !== false,
          forceRefresh: run.execution.forceRefresh === true,
        },
      );
    }

    if (generation !== analysisGeneration) {
      return;
    }

    activeState.nextPlanIndex = cfpResult.complete
      ? plan.length
      : stepIndex + 1;
    inferenceQueue = inferenceQueue
      .then(async () => {
        if (generation !== analysisGeneration) {
          return;
        }
        const completeRun = isLastStep || cfpResult.complete;
        await applyCFPAndRunInference({
          generation,
          stepIndex,
          result: cfpResult,
          completeRun,
          emitCFP: !(run.reuseCFP && stepIndex === 0),
        });
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
      if (input.source.kind === "stream") {
        resetActiveState(input, execution);
        const token = analysisGeneration;
        streamingAnalysisPromise = runStreamingAnalysis(token)
          .then(() => undefined)
          .catch((error) => {
            if (token !== analysisGeneration) {
              return;
            }
            throw error;
          });
        await streamingAnalysisPromise;
      } else {
        await prepareAudioAndPlan(input, execution);
      }
    },
    step(): Promise<void> {
      if (streamingAnalysisActive || streamingAnalysisPromise) {
        return Promise.resolve();
      }
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
