import type { CFPBatch } from "@ohm/core/cache/cache.js";
import {
  estimateDurationSecFromCFPBatches,
} from "@ohm/core/cache/cache.js";
import { decodeAudioRaw } from "@ohm/core/audio/decoder.js";
import type {
  AnalyzeExecutionOptions,
  AnalyzeInput,
  AnalysisPhase,
  AnalysisResult,
} from "./types.js";

export { AnalysisPhase } from "./types.js";

export interface InferenceProgress {
  batchIdx: number;
  totalArgmax: number[];
  totalConfidence: number[];
  done: boolean;
}

export interface InferenceResult {
  totalArgmax: number[];
  totalConfidence: number[];
}

export interface AnalysisPlanTask {
  start: number;
  end: number;
}

export type AnalysisPlan = readonly AnalysisPlanTask[];

export interface AnalysisState {
  decodedAudio: { pcm: Float32Array; fs: number; mode?: string } | null;
  analysisPlan: AnalysisPlan | null;
  nextPlanIndex: number;
}

export interface AnalysisContext {
  state: AnalysisState;
  phase: AnalysisPhase;
}

export interface AnalyzerAudioEventData {
  audio: { pcm: Float32Array; fs: number; mode?: string };
  reuseCFP?: boolean;
}

export interface AnalyzerCfpEventData {
  cfp: readonly CFPBatch[];
  allCfp: readonly CFPBatch[];
  complete: boolean;
}

export interface AnalyzerInferenceEventData {
  cfp: readonly CFPBatch[];
  allCfp: readonly CFPBatch[];
  inference: InferenceResult;
}

export interface AnalyzerOutputEventData {
  audio: { pcm: Float32Array; fs: number; mode?: string };
  cfp: readonly CFPBatch[];
  inference: InferenceResult | null;
}

export interface AnalyzerPhaseEventDataMap {
  [AnalysisPhase.AUDIO]: AnalyzerAudioEventData;
  [AnalysisPhase.CFP]: AnalyzerCfpEventData;
  [AnalysisPhase.INFERENCE]: AnalyzerInferenceEventData;
  [AnalysisPhase.OUTPUT]: AnalyzerOutputEventData;
}

export type AnalyzerPhaseEvent = {
  [Phase in AnalysisPhase]: {
    type: "phase-end";
    phase: Phase;
    index: number;
    context: Readonly<AnalysisContext>;
    state: AnalysisState;
    data: AnalyzerPhaseEventDataMap[Phase];
  };
}[AnalysisPhase];

export type AnalyzerEventListener = (
  event: AnalyzerPhaseEvent,
) => void;

export interface Analyzer {
  subscribe(listener: AnalyzerEventListener): () => void;
  setAudio(
    input: AnalyzeInput,
    execution?: AnalyzeExecutionOptions,
  ): Promise<void>;
  step(): Promise<void>;
}

export function createAnalysisState(): AnalysisState {
  return {
    decodedAudio: null,
    analysisPlan: null,
    nextPlanIndex: 0,
  };
}

export function sourceToBlob(source: AnalyzeInput["source"]): Blob {
  if (source.kind === "file") {
    return source.file;
  }
  if (source.kind === "blob") {
    return source.blob;
  }
  if (source.kind === "buffer") {
    return new Blob([source.buffer as unknown as BlobPart]);
  }
  throw new Error(`Unsupported analysis source kind: ${source.kind}`);
}

export function createAnalysisResult<TArtifact>(
  input: AnalyzeInput,
  fileKey: string,
  batches: CFPBatch[],
): AnalysisResult<TArtifact> {
  return {
    fileKey,
    modelName: String(input.model.name || "").trim(),
    durationSec: estimateDurationSecFromCFPBatches(batches, 0.01),
    artifacts: batches as unknown as readonly TArtifact[],
  };
}

export async function decodeInputAudio(
  input: AnalyzeInput,
): Promise<{ pcm: Float32Array; fs: number; mode?: string }> {
  const blob = sourceToBlob(input.source);
  return await decodeAudioRaw(blob);
}
