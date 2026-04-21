import type { CFPBatch } from "@ohm/core/cache/cache.js";
import type {
  AnalyzeExecutionOptions,
  AnalyzeInput,
  InferenceResult,
} from "../../runtime/dist/index.js";
import type { DisplaySamplingConfig, RepresentativeMode } from "./display-sampling.js";
import { DEFAULT_DISPLAY_SAMPLING_CONFIG } from "./display-sampling.js";
import type { HeatmapBenchmarkEntry } from "./heatmap-render-core.js";

export const DEFAULT_MAIN_FPS = 60;
export const DEFAULT_OVERVIEW_FPS = 24;
export const DEFAULT_MAIN_HEIGHT = 360;
export const DEFAULT_OVERVIEW_HEIGHT = 120;

export interface SpectrumSectionConfig {
  enabled: boolean;
  overlay: boolean;
  fps: number;
  overlayFps: number;
}

export interface SpectrumPitchRange {
  minBin: number;
  maxBin: number;
}

export interface SpectrumUiState {
  status:
    | "idle"
    | "running"
    | "succeeded"
    | "cancelled"
    | "failed"
    | "disposed";
  message: string | null;
  mount: HTMLElement | null;
  audioElement: HTMLAudioElement | null;
  audio: { pcm: Float32Array; fs: number; mode?: string } | null;
  cfp: readonly CFPBatch[];
  inference: InferenceResult | null;
  displaySampling: DisplaySamplingConfig;
  pitchRange: SpectrumPitchRange;
  sections: {
    main: SpectrumSectionConfig;
    overview: SpectrumSectionConfig;
  };
}

export interface SpectrumUiOptions {
  analyzer: {
    subscribe: (listener: (event: any) => void) => () => void;
    setAudio: (
      input: AnalyzeInput,
      execution?: AnalyzeExecutionOptions,
    ) => Promise<void>;
    step: () => Promise<void>;
  };
  mount?: HTMLElement | null;
  audioElement?: HTMLAudioElement | null;
  sections?: Partial<{
    main: boolean | Partial<SpectrumSectionConfig>;
    overview: boolean | Partial<SpectrumSectionConfig>;
  }>;
  displaySampling?: Partial<DisplaySamplingConfig>;
  pitchRange?: Partial<SpectrumPitchRange>;
  heatmapWorkerFactory?: (() => Worker | null) | null;
  heatmapWorkerUrl?: string | URL;
  onStateChange?: ((state: SpectrumUiState) => void) | null;
  onRender?: ((mount: HTMLElement | null, state: SpectrumUiState) => void) | null;
}

export interface SpectrumUiAnalyzeOptions {
  input: AnalyzeInput;
  execution?: AnalyzeExecutionOptions;
}

export interface SpectrumUiDebugState {
  timing: {
    durationSec: number;
    currentTimeSec: number;
    currentFrame: number;
  };
  viewport: {
    totalFrames: number;
    viewFrames: number;
    offsetFrames: number;
    zoom: number;
  };
  spectral: {
    totalBins: number;
    xPerFrame: number;
    yPerBin: number;
    displayMinBin: number;
    displayMaxBin: number;
  };
  prediction: {
    frameCount: number;
    confidenceCount: number;
    inferenceTotalCount: number;
  };
  hover: {
    active: boolean;
    canvasX: number | null;
    canvasY: number | null;
    frame: number;
    frameX: number | null;
    frameBin: number | null;
    frameY: number | null;
  };
}

export interface SpectrumUi {
  analyze(options: SpectrumUiAnalyzeOptions): Promise<void>;
  cancel(reason?: unknown): void;
  destroy(reason?: unknown): void;
  getState(): SpectrumUiState;
  getDebugState(): SpectrumUiDebugState;
  mount(nextMount: HTMLElement | null): void;
  setAudioElement(nextAudioElement: HTMLAudioElement | null): void;
  setRefreshRate(
    next: Partial<{
      main: number;
      mainOverlay: number;
      overview: number;
      overviewOverlay: number;
    }>,
  ): void;
  setDisplaySampling(next: Partial<DisplaySamplingConfig>): void;
  setPitchRange(next: Partial<SpectrumPitchRange>): void;
  runHeatmapBenchmark(rounds?: number): HeatmapBenchmarkEntry[];
  setSections(
    next: Partial<{
      main: boolean | Partial<SpectrumSectionConfig>;
      overview: boolean | Partial<SpectrumSectionConfig>;
    }>,
  ): void;
  subscribe(listener: (state: SpectrumUiState) => void): () => void;
}

export type UiControllerStatus = SpectrumUiState["status"];
export type UiControllerState = SpectrumUiState;
export type UiControllerOptions = SpectrumUiOptions;
export type UiAnalyzeOptions = SpectrumUiAnalyzeOptions;
export type UiController = SpectrumUi;

export interface SpectrumInteractionState {
  spectrumW: number;
  spectrumOverviewW: number;
  spectrumH: number;
  spectrumOffset: number;
  spectrumZoom: number;
  spectrumDuration: number;
  spectrumDragging: boolean;
  spectrumHoverFrame: number;
  spectrumHoverActive: boolean;
  spectrumHoverX: number | null;
  spectrumHoverY: number | null;
  overviewDragging: boolean;
}

export const DIRTY = {
  MAIN_BASE: 1 << 0,
  MAIN_OVERLAY: 1 << 1,
  OVERVIEW_BASE: 1 << 2,
  OVERVIEW_OVERLAY: 1 << 3,
} as const;

export function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function toPositiveFinite(value: unknown, fallback: number): number {
  const next = Number(value);
  return Number.isFinite(next) && next > 0 ? next : fallback;
}

export function normalizeSectionConfig(
  value: boolean | Partial<SpectrumSectionConfig> | undefined,
  fallbackFps: number,
): SpectrumSectionConfig {
  if (value === false) {
    return { enabled: false, overlay: false, fps: fallbackFps, overlayFps: fallbackFps };
  }
  if (value === true || value === undefined) {
    return { enabled: true, overlay: true, fps: fallbackFps, overlayFps: fallbackFps };
  }
  const enabled = value.enabled !== false;
  const fps = Number.isFinite(Number(value.fps)) ? Math.max(0, Math.floor(Number(value.fps))) : fallbackFps;
  const overlayFps = Number.isFinite(Number((value as Partial<SpectrumSectionConfig> & { overlayFps?: number }).overlayFps))
    ? Math.max(0, Math.floor(Number((value as Partial<SpectrumSectionConfig> & { overlayFps?: number }).overlayFps)))
    : fps;
  return {
    enabled,
    overlay: enabled && value.overlay !== false,
    fps,
    overlayFps,
  };
}

export function mergeSectionConfig(
  base: SpectrumSectionConfig,
  override: boolean | Partial<SpectrumSectionConfig> | undefined,
): SpectrumSectionConfig {
  if (override === undefined) return base;
  return normalizeSectionConfig(
    {
      enabled:
        typeof override === "object"
          ? (override.enabled ?? base.enabled)
          : base.enabled,
      overlay:
        typeof override === "object"
          ? (override.overlay ?? base.overlay)
          : base.overlay,
      fps: typeof override === "object" ? (override.fps ?? base.fps) : base.fps,
      overlayFps:
        typeof override === "object"
          ? (override.overlayFps ?? base.overlayFps)
          : base.overlayFps,
    },
    base.fps,
  );
}

export function createDefaultSections(
  sections: SpectrumUiOptions["sections"] = {},
): SpectrumUiState["sections"] {
  return {
    main: mergeSectionConfig(
      { enabled: true, overlay: true, fps: DEFAULT_MAIN_FPS, overlayFps: DEFAULT_MAIN_FPS },
      sections?.main,
    ),
    overview: mergeSectionConfig(
      { enabled: true, overlay: true, fps: DEFAULT_OVERVIEW_FPS, overlayFps: DEFAULT_OVERVIEW_FPS },
      sections?.overview,
    ),
  };
}

export function createDefaultDisplaySampling(
  displaySampling: SpectrumUiOptions["displaySampling"] = {},
): DisplaySamplingConfig {
  return {
    minUnitsPerSecond: Number.isFinite(displaySampling?.minUnitsPerSecond)
      ? Math.max(1, Math.floor(Number(displaySampling.minUnitsPerSecond)))
      : DEFAULT_DISPLAY_SAMPLING_CONFIG.minUnitsPerSecond,
    maxUnitsPerSecond: Number.isFinite(displaySampling?.maxUnitsPerSecond)
      ? Math.max(1, Math.floor(Number(displaySampling.maxUnitsPerSecond)))
      : DEFAULT_DISPLAY_SAMPLING_CONFIG.maxUnitsPerSecond,
    representativeMode:
      (displaySampling?.representativeMode as RepresentativeMode | undefined) ??
      DEFAULT_DISPLAY_SAMPLING_CONFIG.representativeMode,
  };
}

export function normalizePitchRange(
  pitchRange: Partial<SpectrumPitchRange> | undefined,
): SpectrumPitchRange {
  const rawMin = Math.floor(Number(pitchRange?.minBin));
  const rawMax = Math.floor(Number(pitchRange?.maxBin));
  const minBin = Number.isFinite(rawMin) ? Math.max(0, rawMin) : 0;
  const maxBin = Number.isFinite(rawMax) ? Math.max(minBin, rawMax) : Number.MAX_SAFE_INTEGER;
  return {
    minBin,
    maxBin,
  };
}

export function createDefaultInteractionState(): SpectrumInteractionState {
  return {
    spectrumW: 0,
    spectrumOverviewW: 0,
    spectrumH: 0,
    spectrumOffset: 0,
    spectrumZoom: 1,
    spectrumDuration: 0,
    spectrumDragging: false,
    spectrumHoverFrame: -1,
    spectrumHoverActive: false,
    spectrumHoverX: null,
    spectrumHoverY: null,
    overviewDragging: false,
  };
}

export function copyState(state: SpectrumUiState): SpectrumUiState {
  return {
    status: state.status,
    message: state.message,
    mount: state.mount,
    audioElement: state.audioElement,
    audio: state.audio,
    cfp: state.cfp,
    inference: state.inference,
    displaySampling: { ...state.displaySampling },
    pitchRange: { ...state.pitchRange },
    sections: {
      main: { ...state.sections.main },
      overview: { ...state.sections.overview },
    },
  };
}
