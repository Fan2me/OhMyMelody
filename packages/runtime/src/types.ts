import type { CFPBatch } from "@ohm/core/cache/cache.js";
import type { InferenceResult } from "./analysis.js";

export type SourceDescriptor =
  | {
      kind: "file";
      file: File;
      label?: string;
    }
  | {
      kind: "blob";
      blob: Blob;
      label?: string;
    }
  | {
      kind: "url";
      url: string;
      label?: string;
    }
  | {
      kind: "stream";
      stream: MediaStream;
      label?: string;
    }
  | {
      kind: "buffer";
      buffer: ArrayBuffer | ArrayBufferView;
      label?: string;
    };

export interface ModelDescriptor {
  name: string;
}

export interface AnalyzeInput {
  source: SourceDescriptor;
  model: ModelDescriptor;
  fileKey?: string;
}

export interface AnalyzeExecutionOptions {
  allowCache?: boolean;
  forceRefresh?: boolean;
  signal?: AbortSignal | null;
}

export interface AnalysisResult<TArtifact = unknown> {
  fileKey: string;
  modelName: string;
  durationSec: number;
  artifacts: readonly TArtifact[];
}

export enum AnalysisPhase {
  AUDIO = "audio",
  CFP = "cfp",
  INFERENCE = "inference",
  OUTPUT = "output",
}

export interface AnalyzerPhaseEvent {
  type: "phase-end";
  phase: AnalysisPhase;
  index: number;
  context: {
    state: unknown;
    phase: AnalysisPhase;
  };
  state: unknown;
  data:
    | {
        audio: { pcm: Float32Array; fs: number; mode?: string };
        reuseCFP?: boolean;
      }
    | {
        cfp: readonly CFPBatch[];
        allCfp: readonly CFPBatch[];
      }
    | {
        cfp: readonly CFPBatch[];
        allCfp: readonly CFPBatch[];
        inference: InferenceResult | null;
      }
    | {
        audio: { pcm: Float32Array; fs: number; mode?: string };
        cfp: readonly CFPBatch[];
        allCfp: readonly CFPBatch[];
        inference: InferenceResult | null;
      };
}

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
