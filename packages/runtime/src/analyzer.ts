import {
  createAnalysisSession,
  type CreateAnalysisSessionOptions,
} from "./analysis-session.js";

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
} from "./analysis-plan.js";

export {
  createSampleQueue,
  drainSamples,
  enqueueSamples,
  padSamplesToLength,
  takeSamples,
  type SampleQueue,
} from "./sample-queue.js";

export {
  normalizeStreamingChunk,
  takeNextStreamingSegment,
} from "./streaming-segments.js";

export { createAnalysisSession } from "./analysis-session.js";
export { AudioManager } from "./managers/audio-manager.js";
export { CFPManager } from "./managers/cfp-manager.js";
export { InferenceManager } from "./managers/inference-manager.js";

export const createAnalyzer = createAnalysisSession;
