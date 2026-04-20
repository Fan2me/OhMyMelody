import type { CFPBatch } from "@ohm/core/cache/cache.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import {
  AnalysisPhase,
  type Analyzer,
} from "../../runtime/dist/index.js";
import { buildSpectrumTimeline } from "./heatmap-render-core.js";
import {
  type DisplaySamplingConfig,
} from "./display-sampling.js";
import { getMainViewFrameCount } from "./spectrum-layout.js";
import {
  createDefaultInteractionState,
  createDefaultSections,
  createDefaultDisplaySampling,
  normalizePitchRange,
  copyState,
  mergeSectionConfig,
  DIRTY,
  type SpectrumInteractionState,
  type SpectrumUiAnalyzeOptions,
  type SpectrumUiOptions,
  type SpectrumUiState,
  type SpectrumUiDebugState,
  type SpectrumUi,
  type SpectrumPitchRange,
  DEFAULT_MAIN_FPS,
  DEFAULT_OVERVIEW_FPS,
  type SpectrumSectionConfig,
} from "./spectrum-state.js";
import { createSpectrumMainOverlayRenderer } from "./spectrum-main-overlay.js";
import { createSpectrumOverviewOverlayRenderer } from "./spectrum-overview-overlay.js";
import {
  createSpectrumRenderController,
  type SpectrumRenderController,
} from "./spectrum-render.js";
import {
  createProgressiveSpectrumVisualizer,
  type ProgressiveSpectrumVisualizer,
} from "./progressive-spectrum-visualizer.js";
import {
  createSpectrumInteractionController,
  type SpectrumInteractionController,
} from "./spectrum-interaction.js";

export type {
  SpectrumSectionConfig,
  SpectrumUiState,
  SpectrumUiOptions,
  SpectrumUiAnalyzeOptions,
  SpectrumUi,
  UiControllerStatus,
  UiControllerState,
  UiControllerOptions,
  UiAnalyzeOptions,
  UiController,
  SpectrumInteractionState,
} from "./spectrum-state.js";

const uiLogger = getModuleLogger("core.ui.spectrum");
const FRAME_SEC = 0.01;

function isAbortLikeError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  if (error instanceof DOMException && error.name === "AbortError") return true;
  const typed = error as { name?: unknown; message?: unknown };
  return (
    typed.name === "AbortError" ||
    (typeof typed.message === "string" && /abort|cancel/i.test(typed.message))
  );
}

function getCancelMessage(reason: unknown): string {
  if (reason instanceof Error) return reason.message;
  if (typeof reason === "string") return reason;
  return "Cancelled";
}

function createInitialState(options: SpectrumUiOptions): SpectrumUiState {
  return {
    status: "idle",
    message: null,
    mount: options.mount ?? null,
    audioElement: options.audioElement ?? null,
    audio: null,
    cfp: [],
    inference: null,
    displaySampling: createDefaultDisplaySampling((options as { displaySampling?: any }).displaySampling ?? {}),
    pitchRange: normalizePitchRange(options.pitchRange),
    sections: createDefaultSections(options.sections),
  };
}

export function createSpectrumUi(options: SpectrumUiOptions): SpectrumUi {
  const analyzer: Analyzer = options.analyzer as Analyzer;
  const documentRef = typeof document !== "undefined" ? document : null;
  const windowRef = typeof window !== "undefined" ? window : null;
  const listeners = new Set<(state: SpectrumUiState) => void>();
  const state = createInitialState(options);
  const interactionState = createDefaultInteractionState();

  let renderController!: SpectrumRenderController;
  let interactionController!: SpectrumInteractionController;
  let progressiveVisualizer!: ProgressiveSpectrumVisualizer;
  let mountTarget: HTMLElement | null = options.mount ?? null;
  let audioElement: HTMLAudioElement | null = options.audioElement ?? null;
  let analysisCleanup: (() => void) | null = null;
  let abortController: AbortController | null = null;
  let analysisToken = 0;
  let timeline = buildSpectrumTimeline([]);
  let cfpBatches: readonly CFPBatch[] = [];
  let predictionWriteOffset = 0;
  let predictionVisibleCount = 0;
  let scheduledStepIndex = -1;
  let finished = false;
  let analysisDoneResolve: (() => void) | null = null;
  let analysisDoneReject: ((error: unknown) => void) | null = null;
  let analysisCompletionPromise: Promise<void> = Promise.resolve();

  const getMainCanvasRef = () => renderController?.getMainOverlayCanvas() ?? null;
  const getOverviewCanvasRef = () => renderController?.getOverviewOverlayCanvas() ?? null;

  function rebuildTimelineFromBatches(): void {
    const nextSource = cfpBatches.map((batch) => [batch] as const);
    timeline = buildSpectrumTimeline(nextSource);
    interactionState.spectrumOverviewW = Math.max(0, timeline.totalSlots);
    interactionState.spectrumH = Math.max(1, timeline.freqCount || 1);
    const viewW = getMainViewFrameCount({
      spectrumW: Math.max(1, interactionState.spectrumW),
      spectrumZoom: interactionState.spectrumZoom,
      spectrumDuration: interactionState.spectrumDuration,
    });
    interactionState.spectrumOffset = Math.max(
      0,
      Math.min(
        interactionState.spectrumOffset,
        Math.max(0, interactionState.spectrumW - viewW),
      ),
    );
    renderController.setTimeline(nextSource);
  }

  function writePredictionChunk(
    visibleArgmax: readonly number[],
    visibleConfidence: readonly number[] | Float32Array | null | undefined,
  ): void {
    if (!progressiveVisualizer) {
      return;
    }
    const nextVisibleCount = Math.max(0, Math.floor(visibleArgmax.length || 0));
    const nextOffset = predictionWriteOffset;
    const startIndex = Math.max(0, Math.min(predictionVisibleCount, nextVisibleCount));
    const nextArgmax = visibleArgmax.slice(startIndex);
    const nextConfidence = visibleConfidence
      ? Array.prototype.slice.call(visibleConfidence, startIndex)
      : null;
    progressiveVisualizer.applyInferenceProgress({
      visibleArgmax: nextArgmax,
      visibleConfidence: nextConfidence,
      visibleOffset: nextOffset,
    });
    predictionWriteOffset = nextOffset + nextArgmax.length;
    predictionVisibleCount = nextVisibleCount;
  }

  function scheduleNextStep(nextIndex: number, token: number): void {
    if (finished || token !== analysisToken) {
      return;
    }
    if (nextIndex <= scheduledStepIndex) {
      return;
    }
    scheduledStepIndex = nextIndex;
    void analyzer.step().catch((error) => {
      if (finished || token !== analysisToken) {
        return;
      }
      analysisDoneReject?.(error);
      analysisDoneResolve = null;
      analysisDoneReject = null;
    });
  }

  const mainOverlayRenderer = createSpectrumMainOverlayRenderer({
    getCanvas: () => {
      const canvas = getMainCanvasRef();
      const ctx = canvas?.getContext("2d") || null;
      return canvas && ctx ? { canvas, ctx } : null;
    },
    getState: () => state,
    getInteractionState: () => interactionState,
    getAudioElement: () => audioElement,
    getPredictionFrames: () => progressiveVisualizer?.state.progressiveArgmax ?? [],
    getPredictionConfidence: () => progressiveVisualizer?.state.progressiveConfidence ?? null,
  });

  const overviewOverlayRenderer = createSpectrumOverviewOverlayRenderer({
    getCanvas: () => {
      const canvas = getOverviewCanvasRef();
      const ctx = canvas?.getContext("2d") || null;
      return canvas && ctx ? { canvas, ctx } : null;
    },
    getState: () => state,
    getInteractionState: () => interactionState,
  });

  renderController = createSpectrumRenderController({
    windowRef,
    documentRef,
    getState: () => state,
    getInteractionState: () => interactionState,
    mainOverlayRenderer,
    overviewOverlayRenderer,
    heatmapWorkerFactory: options.heatmapWorkerFactory ?? null,
    heatmapWorkerUrl: options.heatmapWorkerUrl ?? null,
  });

  progressiveVisualizer = createProgressiveSpectrumVisualizer({
    setSpectrumPayload: () => {
      renderController.requestSpectrumRedraw({ force: true, includeOverviewBase: true });
    },
    setSpectrumDuration: (duration: number) => {
      interactionState.spectrumDuration = Math.max(0, duration);
    },
    requestSpectrumRedraw: (next) => {
      renderController.requestSpectrumRedraw(
        typeof next === "boolean" ? next : next?.force ?? false,
      );
    },
    markSpectrumDataDirty: () => {
      renderController.requestSpectrumRedraw({ force: true, includeOverviewBase: true });
    },
    frameSec: FRAME_SEC,
    maxWindowFrames: 240000,
    spectrumHeight: 360,
  });

  interactionController = createSpectrumInteractionController({
    windowRef,
    getMainCanvas: () => renderController.getMainCanvas(),
    getOverviewCanvas: () => renderController.getOverviewCanvas(),
    getAxisX: () => 0,
    getMainTimelineHeight: () => 0,
    getState: () => interactionState,
    getDisplaySamplingConfig: () => state.displaySampling,
    setState: (partial) => {
      Object.assign(interactionState, partial);
    },
    seekAudioTime: async (timeSec: number) => {
      if (audioElement) {
        audioElement.currentTime = Math.max(0, timeSec);
      }
    },
    markAutoPanSuppressed: (nowTs?: number, durationMs?: number) => {
      renderController.markAutoPanSuppressed(nowTs, durationMs);
    },
    requestSpectrumRedraw: (next) => {
      renderController.requestSpectrumRedraw(next);
    },
    requestOverviewOverlayRedraw: () => {
      renderController.requestOverviewOverlayRedraw();
    },
  });

  function emit(next: Partial<SpectrumUiState>, redrawMask = 0): SpectrumUiState {
    if (Object.prototype.hasOwnProperty.call(next, "status")) {
      state.status = next.status ?? state.status;
    }
    if (Object.prototype.hasOwnProperty.call(next, "message")) {
      state.message = next.message ?? null;
    }
    if (Object.prototype.hasOwnProperty.call(next, "mount")) {
      state.mount = next.mount ?? null;
    }
    if (Object.prototype.hasOwnProperty.call(next, "audioElement")) {
      state.audioElement = next.audioElement ?? null;
      audioElement = state.audioElement;
      renderController.attachAudioElement(audioElement);
    }
    if (Object.prototype.hasOwnProperty.call(next, "audio")) {
      state.audio = next.audio ?? null;
      interactionState.spectrumDuration = state.audio
        ? state.audio.pcm.length / Math.max(1, state.audio.fs)
        : 0;
    }
    if (Object.prototype.hasOwnProperty.call(next, "cfp")) {
      state.cfp = next.cfp ?? [];
    }
    if (Object.prototype.hasOwnProperty.call(next, "inference")) {
      state.inference = next.inference ?? null;
    }
    if (Object.prototype.hasOwnProperty.call(next, "displaySampling") && next.displaySampling) {
      state.displaySampling = {
        ...state.displaySampling,
        ...next.displaySampling,
      };
    }
    if (Object.prototype.hasOwnProperty.call(next, "pitchRange") && next.pitchRange) {
      state.pitchRange = normalizePitchRange({
        ...state.pitchRange,
        ...next.pitchRange,
      });
    }
    if (
      Object.prototype.hasOwnProperty.call(next, "sections") &&
      next.sections
    ) {
      state.sections = {
        main: { ...next.sections.main },
        overview: { ...next.sections.overview },
      };
    }

    const snapshot = copyState(state);
    for (const listener of listeners) {
      try {
        listener(snapshot);
      } catch {}
    }
    options.onStateChange?.(snapshot);
    options.onRender?.(mountTarget, snapshot);
    if (redrawMask > 0) {
      renderController.requestSpectrumRedraw({ dirtyMask: redrawMask, force: false });
    }
    return snapshot;
  }

  function ensureMounted(): void {
    renderController.mount(mountTarget);
    interactionController.bind();
    if (audioElement) {
      renderController.attachAudioElement(audioElement);
    }
    renderController.requestSpectrumRedraw({ force: true });
  }

  function updateSections(
    next: Partial<{
      main: boolean | Partial<SpectrumSectionConfig>;
      overview: boolean | Partial<SpectrumSectionConfig>;
    }>,
  ): void {
    state.sections = {
      main: mergeSectionConfig(state.sections.main, next.main),
      overview: mergeSectionConfig(state.sections.overview, next.overview),
    };
  }

  async function analyze({
    input,
    execution = {},
  }: SpectrumUiAnalyzeOptions): Promise<void> {
    const token = ++analysisToken;
    analysisCleanup?.();
    analysisCleanup = null;
    abortController?.abort("UI analyze restarted");
    abortController = new AbortController();
    finished = false;
    scheduledStepIndex = -1;
    analysisCompletionPromise = new Promise<void>((resolve, reject) => {
      analysisDoneResolve = resolve;
      analysisDoneReject = reject;
    });
    emit({ status: "running", message: "Analyzing audio..." });

    analysisCleanup = analyzer.subscribe((event) => {
      if (token !== analysisToken) return;
      if (event.phase === AnalysisPhase.AUDIO) {
        const reuseCFP = (event.data as { reuseCFP?: boolean }).reuseCFP === true;
        const predictionFrameCount = Math.max(
          1,
          Math.round(
            (event.data.audio.pcm.length / Math.max(1, event.data.audio.fs)) /
              FRAME_SEC,
          ),
        );
        if (!reuseCFP) {
          cfpBatches = [];
          progressiveVisualizer?.reset({
            expectedFrames: predictionFrameCount,
            durationSec: event.data.audio.pcm.length / Math.max(1, event.data.audio.fs),
            preserveExisting: false,
            pushToUi: true,
          });
        } else {
          cfpBatches = [...state.cfp];
          progressiveVisualizer?.ensureBase(
            predictionFrameCount,
            event.data.audio.pcm.length / Math.max(1, event.data.audio.fs),
            true,
            true,
          );
          rebuildTimelineFromBatches();
        }
        predictionWriteOffset = 0;
        predictionVisibleCount = 0;
        interactionState.spectrumW = Math.max(
          1,
          Math.round((event.data.audio.pcm.length / Math.max(1, event.data.audio.fs)) / FRAME_SEC),
        );
        interactionState.spectrumOffset = Math.max(
          0,
          Math.min(
            interactionState.spectrumOffset,
            Math.max(
              0,
              interactionState.spectrumW -
                getMainViewFrameCount({
                  spectrumW: Math.max(1, interactionState.spectrumW),
                  spectrumZoom: interactionState.spectrumZoom,
                  spectrumDuration: interactionState.spectrumDuration,
                }),
            ),
          ),
        );
        if (!reuseCFP) {
          emit(
            { audio: event.data.audio, cfp: [], inference: null },
            DIRTY.MAIN_BASE | DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_BASE | DIRTY.OVERVIEW_OVERLAY,
          );
        } else {
          emit(
            { audio: event.data.audio, cfp: cfpBatches, inference: null },
            DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_OVERLAY,
          );
        }
        uiLogger.info(
          `audio phase-end: index=${event.index} pcm=${event.data.audio.pcm.length} fs=${event.data.audio.fs}`,
        );
        return;
      }
      if (event.phase === AnalysisPhase.CFP) {
        cfpBatches = [...event.data.allCfp];
        if (progressiveVisualizer) {
          for (const batch of event.data.cfp) {
            progressiveVisualizer.enqueueChunk(batch);
          }
        }
        rebuildTimelineFromBatches();
        emit({ cfp: cfpBatches }, DIRTY.MAIN_BASE | DIRTY.OVERVIEW_BASE);
        uiLogger.info(
          `cfp phase-end: index=${event.index} batches=${event.data.cfp.length} all=${cfpBatches.length}`,
        );
        if (!event.data.complete) {
          scheduleNextStep(event.index + 1, token);
        }
        return;
      }
      if (event.phase === AnalysisPhase.INFERENCE) {
        writePredictionChunk(
          event.data.inference.visibleArgmax,
          event.data.inference.visibleConfidence,
        );
        emit({ cfp: cfpBatches, inference: event.data.inference }, DIRTY.MAIN_OVERLAY);
        uiLogger.info(
          `inference phase-end: index=${event.index} visible=${event.data.inference?.visibleArgmax.length || 0}`,
        );
        return;
      }
      if (event.phase === AnalysisPhase.OUTPUT) {
        progressiveVisualizer?.flush();
        cfpBatches = [...event.data.cfp];
        rebuildTimelineFromBatches();
        emit({
          audio: event.data.audio,
          cfp: cfpBatches,
          inference: event.data.inference,
        }, DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_BASE);
        uiLogger.info(
          `output phase-end: index=${event.index} audio=${event.data.audio.pcm.length} cfp=${event.data.cfp.length} inference=${event.data.inference?.visibleArgmax.length || 0}`,
        );
        finished = true;
        analysisDoneResolve?.();
        analysisDoneResolve = null;
        analysisDoneReject = null;
      }
    });

    try {
      await analyzer.setAudio(input, {
        ...execution,
        signal: abortController.signal,
      });
      if (!finished) {
        scheduleNextStep(0, token);
      }
      await analysisCompletionPromise;
      emit({ status: "succeeded", message: null });
    } catch (error) {
      const aborted = isAbortLikeError(error);
      if (!finished) {
        emit({
          status: aborted ? "cancelled" : "failed",
          message: aborted
            ? "Cancelled"
            : error instanceof Error
              ? error.message
              : "Analysis failed",
        });
        finished = true;
      }
      analysisDoneResolve = null;
      analysisDoneReject = null;
      throw error;
    } finally {
      analysisCleanup?.();
      analysisCleanup = null;
      abortController = null;
    }
  }

  function cancel(reason: unknown = "UI controller cancelled"): void {
    analysisToken += 1;
    analysisCleanup?.();
    analysisCleanup = null;
    abortController?.abort(reason);
    abortController = null;
    finished = true;
    analysisDoneReject?.(
      reason instanceof Error
        ? reason
        : new Error(getCancelMessage(reason)),
    );
    analysisDoneResolve = null;
    analysisDoneReject = null;
      emit({ status: "cancelled", message: getCancelMessage(reason) });
    renderController.requestSpectrumRedraw({ force: true });
  }

  function destroy(reason: unknown = "UI controller destroyed"): void {
    cancel(reason);
    interactionController.destroy();
    renderController.destroy();
    if (mountTarget) {
      mountTarget.replaceChildren();
    }
    mountTarget = null;
    emit({ status: "disposed", message: null });
  }

  function mount(nextMount: HTMLElement | null): void {
    mountTarget = nextMount;
    emit({ mount: mountTarget }, DIRTY.MAIN_BASE | DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_BASE | DIRTY.OVERVIEW_OVERLAY);
    ensureMounted();
  }

  function setAudioElement(nextAudioElement: HTMLAudioElement | null): void {
    audioElement = nextAudioElement;
    emit({ audioElement }, DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_OVERLAY);
    renderController.attachAudioElement(audioElement);
    renderController.requestSpectrumRedraw({ force: true });
  }

  function setRefreshRate(
    next: Partial<{
      main: number;
      mainOverlay: number;
      overview: number;
      overviewOverlay: number;
    }>,
  ): void {
    if (Object.prototype.hasOwnProperty.call(next, "main")) {
      state.sections.main.fps = Number.isFinite(next.main as number)
        ? Math.max(0, Math.floor(Number(next.main)))
        : state.sections.main.fps || DEFAULT_MAIN_FPS;
    }
    if (Object.prototype.hasOwnProperty.call(next, "mainOverlay")) {
      state.sections.main.overlayFps = Number.isFinite(next.mainOverlay as number)
        ? Math.max(0, Math.floor(Number(next.mainOverlay)))
        : state.sections.main.overlayFps || state.sections.main.fps || DEFAULT_MAIN_FPS;
    }
    if (Object.prototype.hasOwnProperty.call(next, "overview")) {
      state.sections.overview.fps = Number.isFinite(next.overview as number)
        ? Math.max(0, Math.floor(Number(next.overview)))
        : state.sections.overview.fps || DEFAULT_OVERVIEW_FPS;
    }
    if (Object.prototype.hasOwnProperty.call(next, "overviewOverlay")) {
      state.sections.overview.overlayFps = Number.isFinite(next.overviewOverlay as number)
        ? Math.max(0, Math.floor(Number(next.overviewOverlay)))
        : state.sections.overview.overlayFps || state.sections.overview.fps || DEFAULT_OVERVIEW_FPS;
    }
    emit({ sections: state.sections }, DIRTY.MAIN_BASE | DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_BASE | DIRTY.OVERVIEW_OVERLAY);
    renderController.requestSpectrumRedraw({ force: true });
  }

  function setDisplaySampling(
    next: Partial<DisplaySamplingConfig>,
  ): void {
    state.displaySampling = {
      ...state.displaySampling,
      ...next,
    };
    emit({ displaySampling: state.displaySampling }, DIRTY.MAIN_BASE | DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_BASE | DIRTY.OVERVIEW_OVERLAY);
    renderController.requestSpectrumRedraw({ force: true });
  }

  function setSections(
    next: Partial<{
      main: boolean | Partial<SpectrumSectionConfig>;
      overview: boolean | Partial<SpectrumSectionConfig>;
    }>,
  ): void {
    updateSections(next);
    emit({ sections: state.sections }, DIRTY.MAIN_BASE | DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_BASE | DIRTY.OVERVIEW_OVERLAY);
    renderController.requestSpectrumRedraw({ force: true });
  }

  function subscribe(listener: (state: SpectrumUiState) => void): () => void {
    if (typeof listener !== "function") return () => {};
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function getState(): SpectrumUiState {
    return copyState(state);
  }

  function getDebugState(): SpectrumUiDebugState {
    const audioDurationSec = state.audio
      ? state.audio.pcm.length / Math.max(1, state.audio.fs)
      : 0;
    const durationSec = Math.max(
      0,
      interactionState.spectrumDuration || audioDurationSec || audioElement?.duration || 0,
    );
    const currentTimeSec = Math.max(0, Number(audioElement?.currentTime || 0));
    const currentFrame = Math.max(0, currentTimeSec / FRAME_SEC);
    const totalFrames = Math.max(
      1,
      interactionState.spectrumW ||
        progressiveVisualizer?.state.spectrumW ||
        state.inference?.visibleArgmax.length ||
        state.inference?.totalArgmax.length ||
        Math.round(durationSec / FRAME_SEC),
    );
    const viewFrames = getMainViewFrameCount({
      spectrumW: totalFrames,
      spectrumZoom: interactionState.spectrumZoom,
      spectrumDuration: durationSec,
    });
    const offsetFrames = Math.max(
      0,
      Math.min(Math.max(0, totalFrames - viewFrames), interactionState.spectrumOffset || 0),
    );
    const totalBins = Math.max(1, interactionState.spectrumH || progressiveVisualizer?.state.spectrumH || 1);
    const displayMinBin = Math.max(0, Math.min(totalBins - 1, Math.floor(state.pitchRange.minBin || 0)));
    const displayMaxBin = Math.max(displayMinBin, Math.min(totalBins - 1, Math.floor(state.pitchRange.maxBin || (totalBins - 1))));
    const displayBinCount = Math.max(1, displayMaxBin - displayMinBin + 1);

    const overlayCanvas = renderController.getMainOverlayCanvas();
    const dpr = Math.max(1, windowRef?.devicePixelRatio || 1);
    const canvasW = Math.max(1, (overlayCanvas?.width || overlayCanvas?.clientWidth || 1) / dpr);
    const canvasH = Math.max(1, (overlayCanvas?.height || overlayCanvas?.clientHeight || 1) / dpr);
    const xPerFrame = canvasW / Math.max(1, viewFrames);
    const yPerBin = canvasH / displayBinCount;

    const hoverFrame = Number.isFinite(interactionState.spectrumHoverFrame)
      ? Math.max(0, Math.min(totalFrames - 1, Math.floor(interactionState.spectrumHoverFrame)))
      : -1;
    const hoverFrameX = hoverFrame >= 0
      ? ((hoverFrame - offsetFrames) / Math.max(1, viewFrames)) * canvasW
      : null;
    const predictionFrames = progressiveVisualizer?.state.progressiveArgmax;
    const hoverFrameBinRaw =
      hoverFrame >= 0 && predictionFrames && hoverFrame < predictionFrames.length
        ? Number(predictionFrames[hoverFrame])
        : NaN;
    const hoverFrameBin = Number.isFinite(hoverFrameBinRaw) && hoverFrameBinRaw >= 0
      ? hoverFrameBinRaw
      : null;
    const hoverFrameY = hoverFrameBin === null
      ? null
      : hoverFrameBin < displayMinBin || hoverFrameBin > displayMaxBin
        ? null
        : canvasH - (hoverFrameBin - displayMinBin + 0.5) * (canvasH / displayBinCount);

    return {
      timing: {
        durationSec,
        currentTimeSec,
        currentFrame,
      },
      viewport: {
        totalFrames,
        viewFrames,
        offsetFrames,
        zoom: Math.max(1, Number(interactionState.spectrumZoom || 1)),
      },
      spectral: {
        totalBins,
        xPerFrame,
        yPerBin,
        displayMinBin,
        displayMaxBin,
      },
      prediction: {
        frameCount: progressiveVisualizer?.state.progressiveArgmax.length || 0,
        confidenceCount: progressiveVisualizer?.state.progressiveConfidence.length || 0,
        inferenceVisibleCount: state.inference?.visibleArgmax.length || 0,
        inferenceTotalCount: state.inference?.totalArgmax.length || 0,
      },
      hover: {
        active: !!interactionState.spectrumHoverActive,
        canvasX: interactionState.spectrumHoverX,
        canvasY: interactionState.spectrumHoverY,
        frame: hoverFrame,
        frameX: hoverFrameX,
        frameBin: hoverFrameBin,
        frameY: hoverFrameY,
      },
    };
  }

  function setPitchRange(next: Partial<SpectrumPitchRange>): void {
    const merged = normalizePitchRange({
      ...state.pitchRange,
      ...next,
    });
    if (
      merged.minBin === state.pitchRange.minBin &&
      merged.maxBin === state.pitchRange.maxBin
    ) {
      return;
    }
    state.pitchRange = merged;
    emit(
      { pitchRange: state.pitchRange },
      DIRTY.MAIN_BASE | DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_BASE | DIRTY.OVERVIEW_OVERLAY,
    );
    renderController.requestSpectrumRedraw({ force: true });
  }

  function runHeatmapBenchmark(rounds = 3) {
    return renderController.runHeatmapBenchmark(rounds);
  }

  if (mountTarget) {
    ensureMounted();
  } else {
    emit({ sections: state.sections }, DIRTY.MAIN_BASE | DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_BASE | DIRTY.OVERVIEW_OVERLAY);
  }
  setAudioElement(audioElement);

  return {
    analyze,
    cancel,
    destroy,
    getState,
    getDebugState,
    mount,
    setAudioElement,
    setDisplaySampling,
    setPitchRange,
    runHeatmapBenchmark,
    setRefreshRate,
    setSections,
    subscribe,
  };
}

export function createUiController(options: SpectrumUiOptions): SpectrumUi {
  return createSpectrumUi(options);
}
