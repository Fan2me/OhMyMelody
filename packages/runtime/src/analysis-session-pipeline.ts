import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import {
  AnalysisPhase,
  type AnalyzerOutputEventData,
  type AnalyzerPhaseEventDataMap,
  type InferenceResult,
} from "./analysis.js";
import type {
  AnalysisSessionStore,
  AnalyzerRunState,
} from "./analysis-session-store.js";

export type AnalyzerCFPProcessResult = {
  kind: "cache-hit" | "segment";
  fileKey: string;
  batches: readonly CFPBatch[];
  complete: boolean;
};

export type AnalyzerCFPManagerLike = {
  process(input: {
    fileKey: string;
    batchOffset?: number;
    segment: { pcm: Float32Array; fs: number };
    signal?: AbortSignal | null;
    complete?: boolean;
    forceRefresh?: boolean;
  }): Promise<AnalyzerCFPProcessResult>;
};

export type AnalyzerInferenceManagerLike = {
  reset(): void;
  process(input: {
    batches: readonly CFPBatch[];
    modelName: string;
    fileKey?: string;
    forceRefresh?: boolean;
    complete?: boolean;
    onProgress?: (progress: InferenceResult) => void;
  }): Promise<InferenceResult>;
  hasCache?: (fileKey: string, modelName: string) => Promise<boolean>;
};

export type SegmentJob = {
  generation: number;
  stepIndex: number;
  segment: { pcm: Float32Array; fs: number };
  final: boolean;
  isFinalSegment: boolean;
  outputIndex: number;
  forceRefresh: boolean;
  reuseCFP: boolean;
};

type InferenceJob = {
  generation: number;
  stepIndex: number;
  run: AnalyzerRunState;
  cfpBatches: readonly CFPBatch[];
  allCfp: readonly CFPBatch[];
  isFinalSegment: boolean;
  forceRefresh: boolean;
  outputIndex: number;
};

type EmitPhaseEnd = <Phase extends AnalysisPhase>(
  phase: Phase,
  index: number,
  data: AnalyzerPhaseEventDataMap[Phase],
) => void;

export function createAnalysisSessionPipeline({
  store,
  cfpManager,
  inferenceManager,
  emitPhaseEnd,
  snapshotOutputAudio,
}: {
  store: AnalysisSessionStore;
  cfpManager: AnalyzerCFPManagerLike;
  inferenceManager: AnalyzerInferenceManagerLike;
  emitPhaseEnd: EmitPhaseEnd;
  snapshotOutputAudio(run: AnalyzerRunState): AnalyzerOutputEventData["audio"];
}): {
  emitOutputPhase(index: number, output: AnalyzerOutputEventData): void;
  processSegment(job: SegmentJob): Promise<void>;
} {
  const analyzerLogger = getModuleLogger("core.runtime.analyzer");

  function emitOutputPhase(
    index: number,
    output: AnalyzerOutputEventData,
  ): void {
    const run = store.run;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }
    if (run.outputEmitted) {
      return;
    }
    store.setPhase(AnalysisPhase.OUTPUT);
    emitPhaseEnd(AnalysisPhase.OUTPUT, index, output);
    store.markOutputEmitted(run);
  }

  function startInferenceJob(job: InferenceJob): void {
    job.run.pendingInferenceCount += 1;
    void runInferenceJob(job);
  }

  async function runInferenceJob(job: InferenceJob): Promise<void> {
    try {
      const inferenceResult = await inferenceManager.process({
        batches: job.cfpBatches,
        modelName: job.run.input.model.name,
        fileKey: job.run.fileKey,
        forceRefresh: job.forceRefresh,
        complete: job.isFinalSegment,
        onProgress: (progress) => emitInferenceProgress(job, progress),
      });
      finishInferenceJob(job, inferenceResult);
    } catch (error) {
      failInferenceJob(job, error);
    }
  }

  function emitInferenceProgress(
    job: InferenceJob,
    progress: InferenceResult,
  ): void {
    if (job.generation !== store.generation) {
      return;
    }
    emitPhaseEnd(AnalysisPhase.INFERENCE, job.stepIndex, {
      cfp: job.cfpBatches,
      allCfp: job.allCfp,
      inference: progress,
    });
  }

  function finishInferenceJob(
    job: InferenceJob,
    inferenceResult: InferenceResult,
  ): void {
    const run = markInferenceJobSettled(job.generation);
    if (!run) {
      return;
    }
    analyzerLogger.info(
      `runtime analyzer inference finished: model=${run.input.model.name}`,
    );
    maybeEmitOutputAfterInference(job, run, inferenceResult);
  }

  function failInferenceJob(job: InferenceJob, error: unknown): void {
    const run = markInferenceJobSettled(job.generation);
    if (!run) {
      return;
    }
    analyzerLogger.warn(
      `runtime analyzer inference failed: model=${run.input.model.name} error=${error instanceof Error ? error.message : String(error)}`,
    );
  }

  function markInferenceJobSettled(
    generation: number,
  ): AnalyzerRunState | null {
    if (generation !== store.generation) {
      return null;
    }
    const run = store.run;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }
    run.pendingInferenceCount = Math.max(0, run.pendingInferenceCount - 1);
    return run;
  }

  function maybeEmitOutputAfterInference(
    job: InferenceJob,
    run: AnalyzerRunState,
    inferenceResult: InferenceResult,
  ): void {
    if (!run.cfpComplete || run.pendingInferenceCount > 0) {
      return;
    }
    emitOutputPhase(job.outputIndex, {
      audio: snapshotOutputAudio(run),
      cfp: run.cfpBatches,
      inference: inferenceResult,
    });
    analyzerLogger.info(
      `runtime analyzer output emitted: model=${run.input.model.name} cfpComplete=${run.cfpComplete}`,
    );
  }

  async function processSegment(job: SegmentJob): Promise<void> {
    const run = store.run;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }

    analyzerLogger.info(
      `runtime analyzer cfp start: reuseCFP=${job.reuseCFP}, forceRefresh=${job.forceRefresh}`,
    );
    let cfpResult: AnalyzerCFPProcessResult;
    if (job.reuseCFP && job.stepIndex === 0 && run.cfpBatches.length > 0) {
      cfpResult = {
        kind: "cache-hit",
        fileKey: run.fileKey,
        batches: run.cfpBatches,
        complete: true,
      };
    } else {
      store.setPhase(AnalysisPhase.CFP);
      cfpResult = await cfpManager.process({
        fileKey: run.fileKey,
        batchOffset: run.cfpBatches.length,
        segment: job.segment,
        signal: run.execution.signal ?? null,
        complete: job.final,
        forceRefresh: job.forceRefresh,
      });
    }
    if (cfpResult.kind === "cache-hit") {
      analyzerLogger.info(
        `runtime analyzer reuse cfp batches: count=${run.cfpBatches.length}`,
      );
    }

    if (job.generation !== store.generation) {
      return;
    }

    store.applyCFPResult(run, cfpResult);
    analyzerLogger.info(
      `runtime analyzer cfp done: batches=${cfpResult.batches.length}, complete=${cfpResult.complete}`,
    );
    if (!(job.reuseCFP && job.stepIndex === 0)) {
      emitPhaseEnd(AnalysisPhase.CFP, job.stepIndex, {
        cfp: cfpResult.batches,
        allCfp: run.cfpBatches,
        complete: cfpResult.complete,
      });
    }

    store.setPhase(AnalysisPhase.INFERENCE);
    analyzerLogger.info(
      `runtime analyzer inference start: model=${run.input.model.name}`,
    );
    startInferenceJob({
      generation: job.generation,
      stepIndex: job.stepIndex,
      run,
      cfpBatches: cfpResult.batches.slice(),
      allCfp: run.cfpBatches.slice(),
      isFinalSegment: job.isFinalSegment,
      forceRefresh: job.forceRefresh,
      outputIndex: job.outputIndex,
    });
  }

  return {
    emitOutputPhase,
    processSegment,
  };
}
