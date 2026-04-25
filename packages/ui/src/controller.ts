import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import {
  AnalysisPhase,
  type Analyzer,
  type AnalyzerPhaseEvent,
} from "@ohm/runtime";
import { buildSpectrumTimeline } from "./heatmap-render-core.js";
import {
  type DisplaySamplingConfig,
} from "./display-sampling.js";
import { getMainViewFrameCount } from "./spectrum-layout.js";
import {
  createDefaultInteractionState,
  createDefaultSections,
  createDefaultDisplaySampling,
  DEFAULT_OVERVIEW_RATIO,
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
type AnalyzerAudioPhaseEvent = Extract<
  AnalyzerPhaseEvent,
  { phase: AnalysisPhase.AUDIO }
>;
type AnalyzerCfpPhaseEvent = Extract<
  AnalyzerPhaseEvent,
  { phase: AnalysisPhase.CFP }
>;
type AnalyzerInferencePhaseEvent = Extract<
  AnalyzerPhaseEvent,
  { phase: AnalysisPhase.INFERENCE }
>;
type AnalyzerOutputPhaseEvent = Extract<
  AnalyzerPhaseEvent,
  { phase: AnalysisPhase.OUTPUT }
>;

type AnalysisSessionState = {
  token: number;
  scheduledStepIndex: number;
  finished: boolean;
  stopRequested: boolean;
  predictionOffset: number;
  predictionChunkCount: number;
};

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

function isEditableShortcutTarget(target: EventTarget | null): boolean {
  if (!target || !(target instanceof Element)) {
    return false;
  }
  if (target instanceof HTMLInputElement) {
    return true;
  }
  if (target instanceof HTMLTextAreaElement) {
    return true;
  }
  if (target instanceof HTMLSelectElement) {
    return true;
  }
  return target instanceof HTMLElement && target.isContentEditable;
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
    spectrumOverviewRatio: DEFAULT_OVERVIEW_RATIO,
    spectrumMainFullscreen: false,
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
  let keyboardCleanup: (() => void) | null = null;
  let abortController: AbortController | null = null;
  let analysisToken = 0;
  let timeline = buildSpectrumTimeline([]);
  let cfpBatches: readonly CFPBatch[] = [];
  const analysisSession: AnalysisSessionState = {
    token: 0,
    scheduledStepIndex: -1,
    finished: false,
    stopRequested: false,
    predictionOffset: 0,
    predictionChunkCount: 0,
  };
  let timelineRebuildRaf: number | null = null;
  let analysisDoneResolve: (() => void) | null = null;
  let analysisDoneReject: ((error: unknown) => void) | null = null;
  let analysisCompletionPromise: Promise<void> = Promise.resolve();

  const getMainCanvasRef = () => renderController?.getMainOverlayCanvas() ?? null;
  const getOverviewCanvasRef = () => renderController?.getOverviewOverlayCanvas() ?? null;

  function rebuildTimelineFromBatches(): void {
    const nextSource = cfpBatches.length ? [cfpBatches] : [];
    timeline = buildSpectrumTimeline(nextSource);
    interactionState.spectrumOverviewW = Math.max(0, timeline.totalFrames);
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
    renderController.setTimeline(timeline);
  }

  function scheduleTimelineRebuild(force = false): void {
    if (force || !windowRef) {
      if (timelineRebuildRaf !== null && windowRef) {
        windowRef.cancelAnimationFrame(timelineRebuildRaf);
        timelineRebuildRaf = null;
      }
      rebuildTimelineFromBatches();
      return;
    }

    if (timelineRebuildRaf !== null) {
      return;
    }

    timelineRebuildRaf = windowRef.requestAnimationFrame(() => {
      timelineRebuildRaf = null;
      rebuildTimelineFromBatches();
    });
  }

  function writePredictionChunk(
    predictionArgmax: readonly number[],
    predictionConfidence: readonly number[] | Float32Array | null | undefined,
  ): void {
    if (!progressiveVisualizer) {
      return;
    }
    const nextChunkCount = Math.max(0, Math.floor(predictionArgmax.length || 0));
    const startIndex = Math.max(
      0,
      Math.min(analysisSession.predictionChunkCount, nextChunkCount),
    );
    const nextArgmax = predictionArgmax.slice(startIndex);
    const nextConfidence = predictionConfidence
      ? Array.prototype.slice.call(predictionConfidence, startIndex)
      : null;
    progressiveVisualizer.applyPredictionChunk({
      predictionArgmax: nextArgmax,
      predictionConfidence: nextConfidence,
      predictionOffset: startIndex,
    });
    analysisSession.predictionChunkCount = nextChunkCount;
  }

  function countCFPFrames(batches: readonly CFPBatch[]): number {
    let total = 0;
    for (const batch of batches) {
      const frameCount = Math.max(
        0,
        Math.floor(Number(batch?.shape?.[2]) || 0),
      );
      total += frameCount;
    }
    return total;
  }

  function ensureVisualizerFrameCapacity(frameCount: number): void {
    if (!progressiveVisualizer) {
      return;
    }
    const nextFrameCount = Math.max(1, Math.floor(frameCount || 0));
    if (nextFrameCount <= progressiveVisualizer.state.spectrumW) {
      return;
    }
    const nextDurationSec = Math.max(
      interactionState.spectrumDuration || 0,
      nextFrameCount * FRAME_SEC,
    );
    progressiveVisualizer.ensureBase(
      nextFrameCount,
      nextDurationSec,
      true,
      true,
    );
    updateInteractionFrameMetrics(nextFrameCount);
  }

  function scheduleNextStep(nextIndex: number, token: number): void {
    if (analysisSession.finished || analysisSession.stopRequested || token !== analysisToken) {
      return;
    }
    if (nextIndex <= analysisSession.scheduledStepIndex) {
      return;
    }
    analysisSession.scheduledStepIndex = nextIndex;
    void analyzer.step().catch((error) => {
      if (analysisSession.finished || token !== analysisToken) {
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
    getPredictionFrames: () => progressiveVisualizer?.state.predictionArgmax ?? [],
    getPredictionConfidence: () => progressiveVisualizer?.state.predictionConfidence ?? null,
    getPredictionRevision: () => progressiveVisualizer?.state.predictionRevision ?? 0,
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
  });

  progressiveVisualizer = createProgressiveSpectrumVisualizer({
    setSpectrumPayload: () => {
      renderController.requestSpectrumRedraw({ force: true, includeOverviewBase: false });
    },
    setSpectrumDuration: (duration: number) => {
      interactionState.spectrumDuration = Math.max(0, duration);
    },
    requestSpectrumRedraw: (next) => {
      renderController.requestSpectrumRedraw(next);
    },
    markSpectrumDataDirty: () => {
      renderController.requestSpectrumRedraw({ dirtyMask: DIRTY.MAIN_BASE, force: false });
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

  if (documentRef && typeof documentRef.addEventListener === "function") {
    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.repeat || event.altKey || event.ctrlKey || event.metaKey) {
        return;
      }
      if (isEditableShortcutTarget(event.target)) {
        return;
      }
      if (event.code === "KeyF" || event.key === "f" || event.key === "F") {
        event.preventDefault();
        void renderController.toggleMainFullscreen();
        return;
      }
      if (event.code === "Space" || event.key === " ") {
        if (!audioElement) {
          return;
        }
        event.preventDefault();
        if (audioElement.paused || audioElement.ended) {
          void audioElement.play().catch(() => undefined);
        } else {
          audioElement.pause();
        }
      }
    };
    documentRef.addEventListener("keydown", handleGlobalKeyDown);
    keyboardCleanup = () => {
      documentRef.removeEventListener("keydown", handleGlobalKeyDown);
    };
  }

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

  function resolveAudioDurationSec(audio: { pcm: Float32Array; fs: number }): number {
    return audio.pcm.length / Math.max(1, audio.fs);
  }

  function completeAnalysisSession(): void {
    analysisSession.finished = true;
    analysisDoneResolve?.();
    analysisDoneResolve = null;
    analysisDoneReject = null;
  }

  function rejectAnalysisSession(error: unknown): void {
    analysisDoneReject?.(error);
    analysisDoneResolve = null;
    analysisDoneReject = null;
  }

  function resetAnalysisSession(token: number): void {
    analysisSession.token = token;
    analysisSession.finished = false;
    analysisSession.stopRequested = false;
    analysisSession.scheduledStepIndex = -1;
    analysisSession.predictionOffset = 0;
    analysisSession.predictionChunkCount = 0;
  }

  function clearAnalysisSubscription(): void {
    analysisCleanup?.();
    analysisCleanup = null;
  }

  function resetTimelineSchedule(): void {
    if (timelineRebuildRaf !== null && windowRef) {
      windowRef.cancelAnimationFrame(timelineRebuildRaf);
      timelineRebuildRaf = null;
    }
  }

  function beginAnalysis(token: number): void {
    clearAnalysisSubscription();
    abortController?.abort("UI analyze restarted");
    abortController = new AbortController();
    resetAnalysisSession(token);
    analysisCompletionPromise = new Promise<void>((resolve, reject) => {
      analysisDoneResolve = resolve;
      analysisDoneReject = reject;
    });
    void analysisCompletionPromise.catch(() => undefined);
    emit({ status: "running", message: "Analyzing audio..." });
  }

  function updateInteractionFrameMetrics(frameCount: number): void {
    interactionState.spectrumW = Math.max(1, frameCount);
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
  }

  function handleAudioPhase(event: AnalyzerAudioPhaseEvent): void {
    const reuseCFP = (event.data as { reuseCFP?: boolean }).reuseCFP === true;
    const durationSec = resolveAudioDurationSec(event.data.audio);
    const predictionFrameCount = Math.max(1, Math.round(durationSec / FRAME_SEC));

    if (!reuseCFP) {
      cfpBatches = [];
      progressiveVisualizer?.reset({
        expectedFrames: predictionFrameCount,
        durationSec,
        preserveExisting: false,
        pushToUi: true,
      });
    } else {
      cfpBatches = [...state.cfp];
      progressiveVisualizer?.ensureBase(
        predictionFrameCount,
        durationSec,
        true,
        true,
      );
      scheduleTimelineRebuild(true);
    }

    analysisSession.predictionOffset = 0;
    analysisSession.predictionChunkCount = 0;
    updateInteractionFrameMetrics(predictionFrameCount);
    emit(
      {
        audio: event.data.audio,
        cfp: reuseCFP ? cfpBatches : [],
        inference: null,
      },
      reuseCFP
        ? DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_OVERLAY
        : DIRTY.MAIN_BASE |
            DIRTY.MAIN_OVERLAY |
            DIRTY.OVERVIEW_BASE |
            DIRTY.OVERVIEW_OVERLAY,
    );
    uiLogger.info(
      `audio phase-end: index=${event.index} pcm=${event.data.audio.pcm.length} fs=${event.data.audio.fs}`,
    );
  }

  function handleCFPPhase(event: AnalyzerCfpPhaseEvent): void {
    cfpBatches = [...event.data.allCfp];
    ensureVisualizerFrameCapacity(countCFPFrames(cfpBatches));
    if (progressiveVisualizer) {
      for (const batch of event.data.cfp) {
        progressiveVisualizer.enqueueChunk(batch);
      }
    }
    scheduleTimelineRebuild(event.data.complete === true);
    emit({ cfp: cfpBatches });
    uiLogger.info(
      `cfp phase-end: index=${event.index} batches=${event.data.cfp.length} all=${cfpBatches.length}`,
    );
    if (!event.data.complete && !analysisSession.stopRequested) {
      scheduleNextStep(event.index + 1, analysisSession.token);
    }
  }

  function handleInferencePhase(event: AnalyzerInferencePhaseEvent): void {
    writePredictionChunk(
      event.data.inference.totalArgmax,
      event.data.inference.totalConfidence,
    );
    emit(
      { cfp: cfpBatches, inference: event.data.inference },
      DIRTY.MAIN_OVERLAY,
    );
    uiLogger.info(
      `inference phase-end: index=${event.index} prediction=${event.data.inference?.totalArgmax.length || 0}`,
    );
    if (analysisSession.stopRequested && !analysisSession.finished) {
      progressiveVisualizer?.flush();
      completeAnalysisSession();
    }
  }

  function handleOutputPhase(event: AnalyzerOutputPhaseEvent): void {
    progressiveVisualizer?.flush();
    uiLogger.info(
      `output phase-end: index=${event.index} audio=${event.data.audio.pcm.length} cfp=${event.data.cfp.length} prediction=${event.data.inference?.totalArgmax.length || 0}`,
    );
    completeAnalysisSession();
  }

  async function analyze({
    input,
    execution = {},
  }: SpectrumUiAnalyzeOptions): Promise<void> {
    const token = ++analysisToken;
    beginAnalysis(token);

    analysisCleanup = analyzer.subscribe((event) => {
      if (token !== analysisToken) return;
      if (event.phase === AnalysisPhase.AUDIO) {
        handleAudioPhase(event);
        return;
      }
      if (event.phase === AnalysisPhase.CFP) {
        handleCFPPhase(event);
        return;
      }
      if (event.phase === AnalysisPhase.INFERENCE) {
        handleInferencePhase(event);
        return;
      }
      if (event.phase === AnalysisPhase.OUTPUT) {
        handleOutputPhase(event);
      }
    });

    try {
      const signal = abortController?.signal ?? null;
      await analyzer.setAudio(input, {
        ...execution,
        signal,
      });
      if (!analysisSession.finished) {
        scheduleNextStep(0, token);
      }
      await analysisCompletionPromise;
      emit({ status: "succeeded", message: null });
    } catch (error) {
      const aborted = isAbortLikeError(error);
      if (!analysisSession.finished) {
        emit({
          status: aborted ? "cancelled" : "failed",
          message: aborted
            ? "Cancelled"
            : error instanceof Error
              ? error.message
              : "Analysis failed",
        });
        analysisSession.finished = true;
      }
      analysisDoneResolve = null;
      analysisDoneReject = null;
      throw error;
    } finally {
      clearAnalysisSubscription();
      abortController = null;
    }
  }

  function requestStop(reason: unknown = "UI controller stop requested"): void {
    if (analysisSession.finished) {
      return;
    }
    analysisSession.stopRequested = true;
    uiLogger.info(`analysis stop requested: ${String(reason)}`);
  }

  function cancel(reason: unknown = "UI controller cancelled"): void {
    analysisToken += 1;
    analysisSession.stopRequested = false;
    resetTimelineSchedule();
    clearAnalysisSubscription();
    abortController?.abort(reason);
    abortController = null;
    analysisSession.finished = true;
    rejectAnalysisSession(
      reason instanceof Error
        ? reason
        : new Error(getCancelMessage(reason)),
    );
    emit({ status: "cancelled", message: getCancelMessage(reason) });
    renderController.requestSpectrumRedraw({ force: true });
  }

  function destroy(reason: unknown = "UI controller destroyed"): void {
    cancel(reason);
    keyboardCleanup?.();
    keyboardCleanup = null;
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
    const predictionFrames = progressiveVisualizer?.state.predictionArgmax;
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
        frameCount: progressiveVisualizer?.state.predictionArgmax.length || 0,
        confidenceCount: progressiveVisualizer?.state.predictionConfidence.length || 0,
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
    requestStop,
    cancel,
    destroy,
    toggleMainFullscreen: (next?: boolean) => renderController.toggleMainFullscreen(next),
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
