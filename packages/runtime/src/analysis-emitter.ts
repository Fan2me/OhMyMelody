import { getModuleLogger } from "@ohm/core/logging/logger.js";
import type {
  AnalysisContext,
  AnalyzerEventListener,
  AnalyzerPhaseEvent,
  AnalyzerPhaseEventDataMap,
  AnalysisState,
} from "./analysis.js";
import { AnalysisPhase } from "./analysis.js";

export interface PhaseEmitterOptions {
  listeners: Set<AnalyzerEventListener>;
  getContext: () => AnalysisContext | null;
  getState: () => AnalysisState;
}

function buildPhaseSummary<P extends AnalysisPhase>(
  phase: P,
  index: number,
  data: AnalyzerPhaseEventDataMap[P],
): string {
  let summary = `phase=${phase} index=${index}`;
  switch (phase) {
    case AnalysisPhase.AUDIO: {
      const payload = data as AnalyzerPhaseEventDataMap[AnalysisPhase.AUDIO];
      summary = `phase=${phase} index=${index} audio pcm=${payload.audio.pcm.length} fs=${payload.audio.fs}`;
      break;
    }
    case AnalysisPhase.CFP: {
      const payload = data as AnalyzerPhaseEventDataMap[AnalysisPhase.CFP];
      summary = `phase=${phase} index=${index} cfp batches=${payload.cfp.length} complete=${payload.complete}`;
      break;
    }
    case AnalysisPhase.INFERENCE: {
      const payload = data as AnalyzerPhaseEventDataMap[AnalysisPhase.INFERENCE];
      summary = `phase=${phase} index=${index} cfp batches=${payload.cfp.length}`;
      break;
    }
    case AnalysisPhase.OUTPUT: {
      const payload = data as AnalyzerPhaseEventDataMap[AnalysisPhase.OUTPUT];
      summary = `phase=${phase} index=${index} audio pcm=${payload.audio.pcm.length} fs=${payload.audio.fs} cfp batches=${payload.cfp.length}`;
      break;
    }
  }
  return summary;
}

export function createPhaseEmitter({
  listeners,
  getContext,
  getState,
}: PhaseEmitterOptions) {
  const logger = getModuleLogger("core.runtime.analyzer");
  function requireContext(): AnalysisContext {
    const context = getContext();
    if (!context) {
      throw new Error("runtime context is unavailable");
    }
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
    logger.info(
      `runtime analyzer emit phase-end: ${buildPhaseSummary(phase, index, data)}`,
    );
    const event = {
      type: "phase-end",
      phase,
      index,
      context: { ...context },
      state: getState(),
      data,
    } as AnalyzerPhaseEvent;
    for (const listener of listeners) {
      try {
        listener(event);
      } catch (error) {
        logger.warn(
          `runtime analyzer phase listener failed: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    }
  }

  return {
    emitPhaseEnd,
  };
}
