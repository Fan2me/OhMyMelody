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
