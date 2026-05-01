import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import {
  AnalysisPhase,
  type InferenceResult,
  type AnalyzerPhaseEventDataMap,
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

type AnalyzerInferenceManagerLike = {
  process(input: {
    batches: readonly CFPBatch[];
    modelName: string;
    fileKey?: string;
    forceRefresh?: boolean;
    complete?: boolean;
    onProgress?: ((progress: InferenceResult) => void) | null;
  }): Promise<InferenceResult>;
};

type AnalyzerCFPManagerLike = {
  process(input: {
    fileKey: string;
    batchOffset?: number;
    segment: { pcm: Float32Array; fs: number };
    signal?: AbortSignal | null;
    complete?: boolean;
    forceRefresh?: boolean;
  }): Promise<AnalyzerCFPProcessResult>;
};

export interface CreateAnalysisSegmentRunnerOptions {
  store: AnalysisSessionStore;
  cfpManager: AnalyzerCFPManagerLike;
  inferenceManager: AnalyzerInferenceManagerLike;
  setPhase(phase: AnalysisPhase): void;
  emitPhaseEnd<P extends AnalysisPhase>(
    phase: P,
    index: number,
    data: AnalyzerPhaseEventDataMap[P],
  ): void;
}

export function createAnalysisSegmentRunner({
  store,
  cfpManager,
  inferenceManager,
  setPhase,
  emitPhaseEnd,
}: CreateAnalysisSegmentRunnerOptions) {
  const logger = getModuleLogger("core.runtime.analyzer");

  function scheduleInferencePhase(input: {
    generation: number;
    stepIndex: number;
    run: AnalyzerRunState;
    cfpBatches: readonly CFPBatch[];
    allCfp: readonly CFPBatch[];
    completeRun: boolean;
    forceRefresh: boolean;
    outputIndex: number;
  }): void {
    input.run.pendingInferenceCount += 1;
    void inferenceManager
      .process({
        batches: input.cfpBatches,
        modelName: input.run.input.model.name,
        fileKey: input.run.fileKey,
        forceRefresh: input.forceRefresh,
        complete: input.completeRun,
        onProgress: (progress) => {
          if (input.generation !== store.generation) {
            return;
          }
          emitPhaseEnd(AnalysisPhase.INFERENCE, input.stepIndex, {
            cfp: input.cfpBatches,
            allCfp: input.allCfp,
            inference: progress,
          });
        },
      })
      .then((inferenceResult) => {
        if (input.generation !== store.generation) {
          return;
        }

        const run = store.run;
        if (!run) {
          throw new Error("runtime run is unavailable");
        }
        run.pendingInferenceCount = Math.max(
          0,
          run.pendingInferenceCount - 1,
        );
        logger.info(
          `runtime analyzer inference done: model=${run.input.model.name}`,
        );
        if (!run.cfpComplete || run.pendingInferenceCount > 0) {
          return;
        }

        const outputAudio =
          run.input.source.kind === "stream"
            ? store.snapshotStreamingAudio(run)
            : (() => {
                const audio = store.requireDecodedAudio();
                return audio.mode
                  ? { pcm: audio.pcm, fs: audio.fs, mode: audio.mode }
                  : { pcm: audio.pcm, fs: audio.fs };
              })();
        emitOutputPhase(input.outputIndex, {
          audio: outputAudio,
          cfp: run.cfpBatches,
          inference: inferenceResult,
        });
        logger.info(
          `runtime analyzer output done: model=${run.input.model.name} cfpComplete=${run.cfpComplete}`,
        );
      })
      .catch((error) => {
        if (input.generation !== store.generation) {
          return;
        }
        const run = store.run;
        if (run) {
          run.pendingInferenceCount = Math.max(
            0,
            run.pendingInferenceCount - 1,
          );
        }
        logger.warn(
          `runtime analyzer inference failed: ${error instanceof Error ? error.message : String(error)}`,
        );
      });
  }

  function emitOutputPhase(
    index: number,
    output: {
      audio: { pcm: Float32Array; fs: number; mode?: string };
      cfp: readonly CFPBatch[];
      inference: InferenceResult | null;
    },
  ): void {
    const run = store.run;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }
    if (run.outputEmitted) {
      return;
    }
    setPhase(AnalysisPhase.OUTPUT);
    emitPhaseEnd(AnalysisPhase.OUTPUT, index, output);
    store.markOutputEmitted(run);
  }

  async function processAnalysisSegment({
    generation,
    stepIndex,
    segment,
    final,
    completeRun,
    outputIndex = stepIndex + 1,
    forceRefresh: forceRefreshOverride,
    reuseCFP = false,
  }: {
    generation: number;
    stepIndex: number;
    segment: { pcm: Float32Array; fs: number };
    final: boolean;
    completeRun: boolean;
    outputIndex?: number;
    forceRefresh?: boolean;
    reuseCFP?: boolean;
  }): Promise<void> {
    const run = store.run;
    if (!run) {
      throw new Error("runtime run is unavailable");
    }
    const forceRefresh =
      forceRefreshOverride ?? run.execution.forceRefresh ?? false;

    logger.info(
      `runtime analyzer cfp start: reuseCFP=${reuseCFP}, forceRefresh=${forceRefresh}`,
    );
    let cfpResult: AnalyzerCFPProcessResult;
    if (reuseCFP && stepIndex === 0 && run.cfpBatches.length > 0) {
      cfpResult = {
        kind: "cache-hit",
        fileKey: run.fileKey,
        batches: run.cfpBatches,
        complete: true,
      };
    } else {
      setPhase(AnalysisPhase.CFP);
      cfpResult = await cfpManager.process({
        fileKey: run.fileKey,
        batchOffset: run.cfpBatches.length,
        segment,
        signal: run.execution.signal ?? null,
        complete: final,
        forceRefresh,
      });
    }
    if (cfpResult.kind === "cache-hit") {
      logger.info(
        `runtime analyzer reuse cfp batches: count=${run.cfpBatches.length}`,
      );
    }

    if (generation !== store.generation) {
      return;
    }

    store.applyCFPResult(run, cfpResult);
    logger.info(
      `runtime analyzer cfp done: batches=${cfpResult.batches.length}, complete=${cfpResult.complete}`,
    );
    if (!(reuseCFP && stepIndex === 0)) {
      emitPhaseEnd(AnalysisPhase.CFP, stepIndex, {
        cfp: cfpResult.batches,
        allCfp: run.cfpBatches,
        complete: cfpResult.complete,
      });
    }

    setPhase(AnalysisPhase.INFERENCE);
    logger.info(
      `runtime analyzer inference start: model=${run.input.model.name}`,
    );
    const cfpBatches = cfpResult.batches.slice();
    const allCfp = run.cfpBatches.slice();
    scheduleInferencePhase({
      generation,
      stepIndex,
      run,
      cfpBatches,
      allCfp,
      completeRun,
      forceRefresh,
      outputIndex,
    });
  }

  return {
    emitOutputPhase,
    processAnalysisSegment,
  };
}
