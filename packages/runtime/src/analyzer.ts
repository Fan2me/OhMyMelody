import {
  createAnalysisSession,
  type CreateAnalysisSessionOptions,
} from "./analysis-session.js";
import type {
  AnalysisPlan,
  AnalysisPlanTask,
  AnalysisState,
  AnalysisContext,
  InferenceResult,
  Analyzer,
  AnalyzerEventListener,
  AnalyzerPhaseEvent,
} from "./analysis.js";
import { AnalysisPhase } from "./analysis.js";

export interface CreateAnalyzerOptions extends CreateAnalysisSessionOptions {}

export {
  AnalysisPhase,
  type AnalysisPlan,
  type AnalysisPlanTask,
  type AnalysisState,
  type AnalysisContext,
  type InferenceResult,
  type Analyzer,
  type AnalyzerEventListener,
  type AnalyzerPhaseEvent,
} from "./analysis.js";

export {
  buildAnalysisPlan,
  createSampleQueue,
  drainSamples,
  enqueueSamples,
  normalizeStreamingChunk,
  padSamplesToLength,
  takeNextStreamingSegment,
  takeSamples,
  type SampleQueue,
} from "./analysis-plan.js";

export { createAnalysisSession } from "./analysis-session.js";
export { AudioManager } from "./managers/audio-manager.js";
export { CFPManager } from "./managers/cfp-manager.js";
export { InferenceManager } from "./managers/inference-manager.js";

export const createAnalyzer = createAnalysisSession;
