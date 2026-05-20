export type {
  SourceDescriptor,
  ModelDescriptor,
  AnalyzeInput,
  AnalyzeExecutionOptions,
  AnalysisResult,
} from "./types.js";

export type {
  AnalysisPlan,
  AnalysisPlanTask,
  AnalysisState,
  AnalysisContext,
  InferenceResult,
  Analyzer,
  AnalyzerEventListener,
  AnalyzerPhaseEvent,
} from "./analysis.js";

export { AnalysisPhase } from "./analysis.js";

export {
  createAnalysisState,
  createAnalysisResult,
} from "./analysis.js";
export { decodeInputAudio, sourceToBlob } from "./audio-input.js";

export {
  createAnalyzer,
  createAnalysisSession,
  buildAnalysisPlan,
} from "./analyzer.js";
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
export {
  AudioManager,
  type AudioResult,
} from "./managers/audio-manager.js";
export { CFPManager } from "./managers/cfp-manager.js";
export {
  InferenceManager,
  type InferenceManagerOptions,
} from "./managers/inference-manager.js";
