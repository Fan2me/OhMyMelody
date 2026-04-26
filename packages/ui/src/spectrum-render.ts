import {
  benchmarkHeatmapTimelineRender,
  renderHeatmapTimeline,
  type HeatmapBenchmarkEntry,
  type HeatmapFrequencyViewport,
  type HeatmapOptimizationLevel,
  type HeatmapTimeline,
} from "./heatmap-render-core.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
import { RafScheduler } from "@ohm/core/scheduler/scheduler.js";
import {
  DEFAULT_MAIN_FPS,
  DEFAULT_OVERVIEW_FPS,
  DEFAULT_OVERVIEW_RATIO,
  DIRTY,
  clampNumber,
  persistOverviewRatio,
  type SpectrumInteractionState,
  type SpectrumUiState,
} from "./spectrum-state.js";
import { getDisplayStrideFramesForZoom } from "./display-sampling.js";
import { getMainViewFrameCount } from "./spectrum-layout.js";
import {
  createSpectrumRootContainer,
  createSpectrumSection,
} from "./spectrum-canvas-shell.js";
import type {
  SpectrumMainOverlayRenderer,
} from "./spectrum-main-overlay.js";
import type {
  SpectrumOverviewOverlayRenderer,
} from "./spectrum-overview-overlay.js";

export interface SpectrumRenderControllerDeps {
  windowRef: Window | null;
  documentRef: Document | null;
  getState: () => SpectrumUiState;
  getInteractionState: () => SpectrumInteractionState;
  mainOverlayRenderer: SpectrumMainOverlayRenderer;
  overviewOverlayRenderer: SpectrumOverviewOverlayRenderer;
}

const FRAME_SEC = 0.01;
const MIN_MAIN_SECTION_HEIGHT = 240;
const MIN_OVERVIEW_SECTION_HEIGHT = 96;
const SPLITTER_HIT_HEIGHT = 14;
const OVERVIEW_RATIO_FALLBACK = DEFAULT_OVERVIEW_RATIO;
const spectrumRenderLogger = getModuleLogger("core.ui.spectrum.render");

export interface SpectrumRenderController {
  mount(nextMount: HTMLElement | null): void;
  attachAudioElement(nextAudioElement: HTMLAudioElement | null): void;
  setTimeline(nextTimeline: HeatmapTimeline | null): void;
  toggleMainFullscreen(next?: boolean): Promise<void>;
  requestSpectrumRedraw(next?: { includeOverviewBase?: boolean; dirtyMask?: number }): void;
  requestOverviewOverlayRedraw(): void;
  runHeatmapBenchmark(rounds?: number): HeatmapBenchmarkEntry[];
  markAutoPanSuppressed(nowTs?: number, durationMs?: number): void;
  destroy(): void;
  getMainCanvas(): HTMLCanvasElement | null;
  getOverviewCanvas(): HTMLCanvasElement | null;
  getMainOverlayCanvas(): HTMLCanvasElement | null;
  getOverviewOverlayCanvas(): HTMLCanvasElement | null;
  getRenderStats(): SpectrumRenderStats;
}

export interface SpectrumRenderTaskStats {
  targetFps: number;
  actualFps: number;
  runCount: number;
  nextRunInMs: number;
}

export interface SpectrumRenderCanvasStats extends SpectrumRenderTaskStats {
  canvasWidth: number;
  canvasHeight: number;
  dpr: number;
}

export interface SpectrumRenderStats {
  mainBase: SpectrumRenderCanvasStats;
  mainOverlay: SpectrumRenderTaskStats;
  overviewBase: SpectrumRenderCanvasStats;
  overviewOverlay: SpectrumRenderTaskStats;
  dirty: {
    currentMask: number;
    playing: boolean;
    schedulerRunning: boolean;
  };
}

function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  timeline: HeatmapTimeline | null,
  viewport: { startSlot: number; endSlot: number } | null,
  viewportMode: "slots" | "frames" = "slots",
  representativeMode = "first-valid",
  frequencyViewport: HeatmapFrequencyViewport | null = null,
  sampleStrideFrames = 1,
  optimizationLevel: HeatmapOptimizationLevel = "u32-region",
): void {
  renderHeatmapTimeline(
    ctx,
    canvas.width || canvas.clientWidth || 1,
    canvas.height || canvas.clientHeight || 1,
    timeline,
    viewport,
    viewportMode,
    representativeMode,
    frequencyViewport,
    sampleStrideFrames,
    optimizationLevel,
  );
}

function getClientPointFromEvent(event: MouseEvent | PointerEvent | TouchEvent | null) {
  if (!event) return null;
  const mouseLike = event as MouseEvent | PointerEvent;
  if (Number.isFinite(mouseLike.clientX) && Number.isFinite(mouseLike.clientY)) {
    return {
      clientX: mouseLike.clientX,
      clientY: mouseLike.clientY,
    };
  }
  const touchEvent = event as TouchEvent;
  const touch =
    (touchEvent.touches && touchEvent.touches[0]) ||
    (touchEvent.changedTouches && touchEvent.changedTouches[0]) ||
    null;
  if (!touch) return null;
  return {
    clientX: touch.clientX,
    clientY: touch.clientY,
  };
}

function preventDefaultIfPossible(event: unknown) {
  if (
    event &&
    typeof (event as { preventDefault?: () => void }).preventDefault === "function"
  ) {
    (event as { preventDefault: () => void }).preventDefault();
  }
}

export function createSpectrumRenderController(
  deps: SpectrumRenderControllerDeps,
): SpectrumRenderController {
  const {
    windowRef,
    documentRef,
    getState,
    getInteractionState,
    mainOverlayRenderer,
    overviewOverlayRenderer,
  } = deps;

  let root: HTMLDivElement | null = null;
  let mainSection: HTMLElement | null = null;
  let overviewSection: HTMLElement | null = null;
  let mainBase: HTMLCanvasElement | null = null;
  let mainOverlay: HTMLCanvasElement | null = null;
  let overviewBase: HTMLCanvasElement | null = null;
  let overviewOverlay: HTMLCanvasElement | null = null;
  let mainBaseCtx: CanvasRenderingContext2D | null = null;
  let mainOverlayCtx: CanvasRenderingContext2D | null = null;
  let overviewBaseCtx: CanvasRenderingContext2D | null = null;
  let overviewOverlayCtx: CanvasRenderingContext2D | null = null;
  let placeholder: HTMLDivElement | null = null;
  let splitBar: HTMLDivElement | null = null;
  let splitBarHandle: HTMLDivElement | null = null;
  let toolbar: HTMLDivElement | null = null;
  let mainFullscreenToggle: HTMLButtonElement | null = null;
  let resizeObserver: ResizeObserver | null = null;
  let audioCleanup: (() => void) | null = null;
  let audioElement: HTMLAudioElement | null = null;
  const rafScheduler = new RafScheduler();
  let schedulerStarted = false;
  let schedulerTasksReady = false;
  let frameTaskId: string | null = null;
  let mainBaseTaskId: string | null = null;
  let mainOverlayTaskId: string | null = null;
  let overviewBaseTaskId: string | null = null;
  let overviewOverlayTaskId: string | null = null;
  let finalizeTaskId: string | null = null;
  let taskRateSignature = "";
  let dirtyMask = 0;
  
  let playing = false;
  let lastAutoPanAt = 0;
  let autoPanSuppressedUntil = 0;
  let timeline: HeatmapTimeline | null = null;
  let fullscreenCleanup: (() => void) | null = null;
  let controlsVisible = false;
  let lastAutoPanLogKey = "";
  let lastHeatmapRenderLogKey = "";
  const recentMainBaseDrawTimes: number[] = [];
  const recentMainOverlayDrawTimes: number[] = [];
  const recentOverviewBaseDrawTimes: number[] = [];
  const recentOverviewOverlayDrawTimes: number[] = [];
  const layoutMetrics = {
    rootHeight: 0,
    mainWidth: 0,
    mainHeight: 0,
    overviewWidth: 0,
    overviewHeight: 0,
  };
  let splitDragRootTop = 0;
  let splitDragRootHeight = 0;
  let lastLayoutSignature = "";

  const AUTO_PAN_MIN_INTERVAL_MS = 33;
  const AUTO_PAN_EDGE_RATIO = 1 / 10;
  const AUTO_PAN_SUPPRESS_MS = 250;
  const SCHEDULER_FRAME_FPS = 60;
  const cleanups: Array<() => void> = [];

  function refreshLayoutMetricsFromElements(): void {
    layoutMetrics.rootHeight = Math.max(
      0,
      root?.clientHeight || root?.offsetHeight || 0,
    );
    layoutMetrics.mainWidth = Math.max(
      0,
      mainSection?.clientWidth || mainSection?.offsetWidth || 0,
    );
    layoutMetrics.mainHeight = Math.max(
      0,
      mainSection?.clientHeight || mainSection?.offsetHeight || 0,
    );
    layoutMetrics.overviewWidth = Math.max(
      0,
      overviewSection?.clientWidth || overviewSection?.offsetWidth || 0,
    );
    layoutMetrics.overviewHeight = Math.max(
      0,
      overviewSection?.clientHeight || overviewSection?.offsetHeight || 0,
    );
  }

  function refreshLayoutMetricsFromObserver(entries: readonly ResizeObserverEntry[]): void {
    for (const entry of entries) {
      const target = entry.target;
      if (target === root) {
        layoutMetrics.rootHeight = Math.max(0, Math.floor(entry.contentRect.height || 0));
        continue;
      }
      if (target === mainSection) {
        layoutMetrics.mainWidth = Math.max(0, Math.floor(entry.contentRect.width || 0));
        layoutMetrics.mainHeight = Math.max(0, Math.floor(entry.contentRect.height || 0));
        continue;
      }
      if (target === overviewSection) {
        layoutMetrics.overviewWidth = Math.max(0, Math.floor(entry.contentRect.width || 0));
        layoutMetrics.overviewHeight = Math.max(0, Math.floor(entry.contentRect.height || 0));
      }
    }
  }

  function destroyHeatmapWorker(): void {
    return;
  }

  function getOverviewRatioBounds(): { min: number; max: number } {
    const rootHeight = Math.max(0, layoutMetrics.rootHeight || 0);
    const availableHeight = Math.max(1, rootHeight - SPLITTER_HIT_HEIGHT);
    const minRatio = rootHeight > 0
      ? clampNumber(MIN_OVERVIEW_SECTION_HEIGHT / availableHeight, 0.08, 0.92)
      : 0.12;
    const maxRatio = rootHeight > 0
      ? clampNumber(1 - MIN_MAIN_SECTION_HEIGHT / availableHeight, minRatio, 0.92)
      : OVERVIEW_RATIO_FALLBACK;
    return {
      min: Math.min(minRatio, maxRatio),
      max: Math.max(minRatio, maxRatio),
    };
  }

  function resolveOverviewRatio(): number {
    const interactionState = getInteractionState();
    const bounds = getOverviewRatioBounds();
    const currentRatio = Number(interactionState.spectrumOverviewRatio);
    if (!Number.isFinite(currentRatio) || currentRatio <= 0) {
      const nextRatio = clampNumber(
        root?.isConnected ? OVERVIEW_RATIO_FALLBACK : DEFAULT_OVERVIEW_RATIO,
        bounds.min,
        bounds.max,
      );
      if (root?.isConnected) {
        interactionState.spectrumOverviewRatio = nextRatio;
        persistOverviewRatio(nextRatio);
      }
      return nextRatio;
    }
    const clampedRatio = clampNumber(currentRatio, bounds.min, bounds.max);
    if (clampedRatio !== interactionState.spectrumOverviewRatio) {
      interactionState.spectrumOverviewRatio = clampedRatio;
      persistOverviewRatio(clampedRatio);
    }
    return clampedRatio;
  }

  function applySectionLayout(): void {
    const state = getState();
    const interactionState = getInteractionState();
    const mainVisible = !!state.audio && state.sections.main.enabled;
    const overviewVisible = !!state.audio && state.sections.overview.enabled;
    const ratio = resolveOverviewRatio();
    const isFullscreen = interactionState.spectrumMainFullscreen === true;
    const signature = [
      mainVisible ? 1 : 0,
      overviewVisible ? 1 : 0,
      isFullscreen ? 1 : 0,
      controlsVisible ? 1 : 0,
      ratio.toFixed(6),
    ].join("|");
    if (signature === lastLayoutSignature) {
      return;
    }
    lastLayoutSignature = signature;

    if (placeholder) {
      placeholder.style.display = mainVisible || overviewVisible ? "none" : "flex";
    }

    if (mainFullscreenToggle) {
      mainFullscreenToggle.disabled = !mainVisible;
      mainFullscreenToggle.textContent = isFullscreen ? "⤡" : "⛶";
      mainFullscreenToggle.title = isFullscreen ? "退出主图全屏 (F)" : "主图全屏 (F)";
      mainFullscreenToggle.setAttribute("aria-pressed", isFullscreen ? "true" : "false");
    }

    if (toolbar) {
      toolbar.style.display = mainVisible ? "flex" : "none";
      toolbar.style.opacity = controlsVisible && mainVisible ? "1" : "0";
      toolbar.style.transform = controlsVisible && mainVisible ? "translateY(0)" : "translateY(-4px)";
      toolbar.style.pointerEvents = controlsVisible && mainVisible ? "auto" : "none";
    }

    if (splitBar) {
      splitBar.style.display = mainVisible && overviewVisible && !isFullscreen ? "" : "none";
      splitBar.style.opacity = isFullscreen ? "0" : "0.06";
      splitBar.style.cursor = isFullscreen ? "default" : "row-resize";
    }
    if (splitBarHandle) {
      splitBarHandle.style.opacity = isFullscreen ? "0" : "0.16";
    }

    if (overviewSection) {
      overviewSection.style.display = overviewVisible ? "" : "none";
      overviewSection.style.flex = mainVisible && overviewVisible
        ? `${ratio} ${ratio} 0px`
        : "0 0 auto";
      overviewSection.style.minHeight = `${MIN_OVERVIEW_SECTION_HEIGHT}px`;
    }

    if (mainSection) {
      mainSection.style.display = mainVisible ? "" : "none";
      mainSection.style.flex = mainVisible && overviewVisible
        ? `${1 - ratio} ${1 - ratio} 0px`
        : "1 1 auto";
      mainSection.style.minHeight = isFullscreen ? "0px" : `${MIN_MAIN_SECTION_HEIGHT}px`;
      mainSection.style.border = isFullscreen ? "0" : "1px solid rgba(54, 80, 130, 0.08)";
      mainSection.style.borderRadius = isFullscreen ? "0" : "0";
      mainSection.style.background = isFullscreen ? "transparent" : "rgba(245, 248, 255, 0.9)";
      mainSection.style.boxShadow = isFullscreen ? "none" : "inset 0 1px 0 rgba(255, 255, 255, 0.72), 0 10px 24px rgba(19, 37, 70, 0.08)";
    }
  }

  function syncFullscreenStateFromDocument(): void {
    const interactionState = getInteractionState();
    const isFullscreen = documentRef?.fullscreenElement === mainSection;
    if (interactionState.spectrumMainFullscreen === isFullscreen) {
      return;
    }
    interactionState.spectrumMainFullscreen = isFullscreen;
    applySectionLayout();
    schedule(DIRTY.MAIN_BASE | DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_OVERLAY);
  }

  async function setMainFullscreen(nextFullscreen: boolean): Promise<void> {
    const next = !!nextFullscreen;
    const isFullscreen = documentRef?.fullscreenElement === mainSection;
    if (next === isFullscreen) {
      return;
    }
    if (!mainSection) {
      return;
    }
    try {
      if (next) {
        await mainSection.requestFullscreen();
      } else if (documentRef?.fullscreenElement) {
        await documentRef.exitFullscreen();
      }
    } catch (error) {
      console.warn(
        `ui spectrum fullscreen unavailable: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    syncFullscreenStateFromDocument();
  }

  function updateOverviewRatioFromPoint(clientY: number): void {
    const interactionState = getInteractionState();
    if (splitDragRootHeight <= 0) {
      return;
    }
    const bounds = getOverviewRatioBounds();
    const usableHeight = Math.max(1, splitDragRootHeight - SPLITTER_HIT_HEIGHT);
    const nextRatio = clampNumber(
      (clientY - splitDragRootTop) / usableHeight,
      bounds.min,
      bounds.max,
    );
    if (nextRatio === interactionState.spectrumOverviewRatio) {
      return;
    }
    interactionState.spectrumOverviewRatio = nextRatio;
    persistOverviewRatio(nextRatio);
    applySectionLayout();
    schedule();
  }

  function syncSectionVisibility(): void {
    const state = getState();
    const hasAudio = !!state.audio;
    if (placeholder) {
      const nextDisplay = hasAudio ? "none" : "flex";
      if (placeholder.style.display !== nextDisplay) {
        placeholder.style.display = nextDisplay;
      }
    }
    applySectionLayout();
    if (mainOverlay) {
      const nextDisplay =
        hasAudio && state.sections.main.enabled && state.sections.main.overlay ? "" : "none";
      if (mainOverlay.style.display !== nextDisplay) {
        mainOverlay.style.display = nextDisplay;
      }
    }
    if (overviewOverlay) {
      const nextDisplay =
        hasAudio && state.sections.overview.enabled && !getInteractionState().spectrumMainFullscreen && state.sections.overview.overlay ? "" : "none";
      if (overviewOverlay.style.display !== nextDisplay) {
        overviewOverlay.style.display = nextDisplay;
      }
    }
  }

  function getPlaybackFrame(): number {
    const currentTime = Number(audioElement?.currentTime ?? NaN);
    if (!Number.isFinite(currentTime) || currentTime < 0) {
      return 0;
    }
    return currentTime / FRAME_SEC;
  }

  function syncToolbarVisibility(nextVisible?: boolean): void {
    if (typeof nextVisible === "boolean") {
      controlsVisible = nextVisible;
    }
    applySectionLayout();
  }

  function markAutoPanSuppressed(nowTs = performance.now(), durationMs = AUTO_PAN_SUPPRESS_MS): void {
    autoPanSuppressedUntil = Math.max(
      autoPanSuppressedUntil,
      nowTs + Math.max(0, durationMs || 0),
    );
  }

  function maybeAutoPanToPlaybackHead(nowTs = performance.now()): boolean {
    const state = getState();
    if (!state.audio) {
      return false;
    }
    if (nowTs < autoPanSuppressedUntil) {
      return false;
    }

    const totalFrames = Math.max(0, getInteractionState().spectrumW || 0);
    if (totalFrames <= 0) {
      return false;
    }

    const viewW = Math.max(
      1,
      getMainViewFrameCount(getInteractionState()),
    );
    const maxOffset = Math.max(0, totalFrames - viewW);
    const offset = Math.max(0, Math.min(maxOffset, getInteractionState().spectrumOffset));
    const frameFloat = getPlaybackFrame();
    if (!Number.isFinite(frameFloat) || frameFloat <= 0) {
      return false;
    }

    const playheadMode = "follow";
    let desiredOffset = offset;
    const edgeThreshold = Math.max(1, viewW * AUTO_PAN_EDGE_RATIO);
    if (frameFloat >= offset + viewW - edgeThreshold) {
      desiredOffset = Math.floor(frameFloat - edgeThreshold);
    } else if (frameFloat < offset + edgeThreshold) {
      desiredOffset = Math.floor(frameFloat - edgeThreshold);
    } else {
      return false;
    }

    desiredOffset = Math.max(0, Math.min(maxOffset, desiredOffset));
    if (desiredOffset === offset) {
      return false;
    }

    if (nowTs - lastAutoPanAt < AUTO_PAN_MIN_INTERVAL_MS) {
      return false;
    }

    const interactionState = getInteractionState();
    interactionState.spectrumOffset = desiredOffset;
    dirtyMask |= DIRTY.MAIN_BASE | DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_OVERLAY;
    lastAutoPanAt = nowTs;
    const autoPanLogKey = [
      playheadMode,
      `frame=${frameFloat.toFixed(2)}`,
      `offset=${offset}`,
      `desired=${desiredOffset}`,
      `viewW=${viewW}`,
      `max=${maxOffset}`,
    ].join("|");
    if (autoPanLogKey !== lastAutoPanLogKey) {
      lastAutoPanLogKey = autoPanLogKey;
      spectrumRenderLogger.info("auto-pan updated", {
        playheadMode,
        frameFloat,
        offset,
        desiredOffset,
        viewW,
        maxOffset,
      });
    }
    return true;
  }

  function resolveSpectrumFrequencyViewport(
    state: SpectrumUiState,
  ): HeatmapFrequencyViewport {
    const spectrumBinCount = Math.max(
      1,
      Math.floor(getInteractionState().spectrumH || timeline?.freqCount || 1),
    );
    const frequencyViewport: HeatmapFrequencyViewport = {
      minBin: Math.max(
        0,
        Math.min(spectrumBinCount - 1, Math.floor(state.pitchRange.minBin || 0)),
      ),
      maxBin: Math.max(
        0,
        Math.min(
          spectrumBinCount - 1,
          Math.floor(state.pitchRange.maxBin || (spectrumBinCount - 1)),
        ),
      ),
    };
    if (frequencyViewport.maxBin < frequencyViewport.minBin) {
      frequencyViewport.maxBin = frequencyViewport.minBin;
    }
    return frequencyViewport;
  }

  function renderBaseHeatmapTarget(args: {
    target: "main" | "overview";
    now: number;
    state: SpectrumUiState;
    canvas: HTMLCanvasElement | null;
    ctx: CanvasRenderingContext2D | null;
    viewport: { startSlot: number; endSlot: number };
    representativeMode: string;
    frequencyViewport: HeatmapFrequencyViewport;
    sampleStrideFrames: number;
  }): void {
    const { target, now, canvas, ctx, viewport, representativeMode, frequencyViewport, sampleStrideFrames } = args;
    if (!canvas || !ctx) {
      return;
    }
    const renderLogKey = [
      args.target,
      `vp=${args.viewport.startSlot}-${args.viewport.endSlot}`,
      `stride=${args.sampleStrideFrames}`,
      `freq=${args.frequencyViewport.minBin}-${args.frequencyViewport.maxBin}`,
    ].join("|");
    if (renderLogKey !== lastHeatmapRenderLogKey) {
      lastHeatmapRenderLogKey = renderLogKey;
      spectrumRenderLogger.info("heatmap render request", {
        target: args.target,
        viewportStart: args.viewport.startSlot,
        viewportEnd: args.viewport.endSlot,
        sampleStrideFrames: args.sampleStrideFrames,
        frequencyViewport: {
          minBin: args.frequencyViewport.minBin,
          maxBin: args.frequencyViewport.maxBin,
        },
      });
    }
    drawHeatmap(
      ctx,
      canvas,
      timeline,
      viewport,
      "frames",
      representativeMode,
      frequencyViewport,
      sampleStrideFrames,
      "u32-region",
    );
  }

  function recordDraw(recentDrawTimes: number[], now: number): void {
    recentDrawTimes.push(now);
    const cutoff = now - 1000;
    while (recentDrawTimes.length && (recentDrawTimes[0] ?? 0) < cutoff) {
      recentDrawTimes.shift();
    }
  }

  function clearDirtyBits(mask: number): void {
    dirtyMask &= ~mask;
  }

  function resolveTaskFps(rawFps: number | null | undefined, fallbackFps: number): number {
    const fallback = Math.max(1, Math.floor(Number(fallbackFps) || SCHEDULER_FRAME_FPS));
    const resolved = Number(rawFps);
    if (!Number.isFinite(resolved) || resolved <= 0) {
      return fallback;
    }
    return Math.max(1, Math.floor(resolved));
  }

  function getScheduledTaskRates() {
    const state = getState();
    const mainFps = resolveTaskFps(state.sections.main.fps, DEFAULT_MAIN_FPS);
    const mainOverlayFps = resolveTaskFps(
      state.sections.main.overlayFps,
      state.sections.main.fps || mainFps,
    );
    const overviewFps = resolveTaskFps(state.sections.overview.fps, DEFAULT_OVERVIEW_FPS);
    const overviewOverlayFps = resolveTaskFps(
      state.sections.overview.overlayFps,
      state.sections.overview.fps || overviewFps,
    );
    return {
      frameFps: SCHEDULER_FRAME_FPS,
      mainFps,
      mainOverlayFps,
      overviewFps,
      overviewOverlayFps,
    };
  }

  function getTaskSnapshot(
    id: string | null,
    recentDrawTimes: number[],
  ): SpectrumRenderTaskStats | null {
    if (!id) {
      return null;
    }
    const snapshot = rafScheduler.getTaskSnapshot(id);
    if (!snapshot) {
      return null;
    }
    return {
      targetFps: snapshot.targetFps ?? 0,
      actualFps: recentDrawTimes.length,
      runCount: snapshot.runCount,
      nextRunInMs: Math.max(0, (snapshot.nextRunTs || 0) - performance.now()),
    };
  }

  function ensureScheduledTasks(): void {
    if (schedulerTasksReady) {
      return;
    }
    const rates = getScheduledTaskRates();
    frameTaskId = rafScheduler.addTask(() => {
      runFrameTick();
    }, {
      priority: 0,
      targetFps: rates.frameFps,
    });
    mainOverlayTaskId = rafScheduler.addTask(() => {
      runMainOverlayTask();
    }, {
      priority: 5,
      targetFps: rates.mainOverlayFps,
    });
    overviewOverlayTaskId = rafScheduler.addTask(() => {
      runOverviewOverlayTask();
    }, {
      priority: 6,
      targetFps: rates.overviewOverlayFps,
    });
    mainBaseTaskId = rafScheduler.addTask(() => {
      runMainBaseTask();
    }, {
      priority: 10,
      targetFps: rates.mainFps,
    });
    overviewBaseTaskId = rafScheduler.addTask(() => {
      runOverviewBaseTask();
    }, {
      priority: 11,
      targetFps: rates.overviewFps,
    });
    finalizeTaskId = rafScheduler.addTask(() => {
      finishScheduledFrame();
    }, {
      priority: 100,
      targetFps: rates.frameFps,
    });
    schedulerTasksReady = true;
    taskRateSignature = "";
  }

  function syncScheduledTaskRates(forceWake = false): void {
    ensureScheduledTasks();
    const rates = getScheduledTaskRates();
    const signature = [
      rates.frameFps,
      rates.mainFps,
      rates.mainOverlayFps,
      rates.overviewFps,
      rates.overviewOverlayFps,
    ].join("|");
    if (!forceWake && signature === taskRateSignature) {
      return;
    }
    taskRateSignature = signature;
    if (frameTaskId) {
      rafScheduler.updateTask(frameTaskId, {
        priority: 0,
        targetFps: rates.frameFps,
      });
    }
    if (mainOverlayTaskId) {
      rafScheduler.updateTask(mainOverlayTaskId, {
        priority: 5,
        targetFps: rates.mainOverlayFps,
      });
    }
    if (overviewOverlayTaskId) {
      rafScheduler.updateTask(overviewOverlayTaskId, {
        priority: 6,
        targetFps: rates.overviewOverlayFps,
      });
    }
    if (mainBaseTaskId) {
      rafScheduler.updateTask(mainBaseTaskId, {
        priority: 10,
        targetFps: rates.mainFps,
      });
    }
    if (overviewBaseTaskId) {
      rafScheduler.updateTask(overviewBaseTaskId, {
        priority: 11,
        targetFps: rates.overviewFps,
      });
    }
    if (finalizeTaskId) {
      rafScheduler.updateTask(finalizeTaskId, {
        priority: 100,
        targetFps: rates.frameFps,
      });
    }
  }

  function wakeScheduledTasks(mask: number): void {
    ensureScheduledTasks();
    const rates = getScheduledTaskRates();
    const wakeAll = mask === 0;
    if (wakeAll || (mask & DIRTY.MAIN_OVERLAY) !== 0) {
      if (mainOverlayTaskId) {
        rafScheduler.updateTask(mainOverlayTaskId, {
          priority: 5,
          targetFps: rates.mainOverlayFps,
        });
      }
    }
    if (wakeAll || (mask & DIRTY.OVERVIEW_OVERLAY) !== 0) {
      if (overviewOverlayTaskId) {
        rafScheduler.updateTask(overviewOverlayTaskId, {
          priority: 6,
          targetFps: rates.overviewOverlayFps,
        });
      }
    }
    if (wakeAll || (mask & DIRTY.MAIN_BASE) !== 0) {
      if (mainBaseTaskId) {
        rafScheduler.updateTask(mainBaseTaskId, {
          priority: 10,
          targetFps: rates.mainFps,
        });
      }
    }
    if (wakeAll || (mask & DIRTY.OVERVIEW_BASE) !== 0) {
      if (overviewBaseTaskId) {
        rafScheduler.updateTask(overviewBaseTaskId, {
          priority: 11,
          targetFps: rates.overviewFps,
        });
      }
    }
    if (wakeAll && frameTaskId) {
      rafScheduler.updateTask(frameTaskId, {
        priority: 0,
        targetFps: rates.frameFps,
      });
    }
    if (wakeAll && finalizeTaskId) {
      rafScheduler.updateTask(finalizeTaskId, {
        priority: 100,
        targetFps: rates.frameFps,
      });
    }
  }

  function startScheduler(): void {
    ensureScheduledTasks();
    if (schedulerStarted) {
      return;
    }
    syncScheduledTaskRates(true);
    schedulerStarted = true;
    rafScheduler.start();
  }

  function stopScheduler(): void {
    if (!schedulerStarted) {
      return;
    }
    rafScheduler.stop();
    schedulerStarted = false;
  }

  function runFrameTick(): void {
    syncSectionVisibility();
    const state = getState();
    if (!state.audio) {
      dirtyMask = 0;
      stopScheduler();
      return;
    }
    if (playing) {
      dirtyMask |= DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_OVERLAY;
    }
    maybeAutoPanToPlaybackHead();
    syncCanvasSizes();
  }

  function runMainOverlayTask(): void {
    const state = getState();
    if (!state.audio || !state.sections.main.enabled || !state.sections.main.overlay) {
      clearDirtyBits(DIRTY.MAIN_OVERLAY);
      return;
    }
    if (!mainOverlay || !mainOverlayCtx) {
      clearDirtyBits(DIRTY.MAIN_OVERLAY);
      return;
    }
    const shouldDraw = playing || (dirtyMask & DIRTY.MAIN_OVERLAY) !== 0;
    if (!shouldDraw) {
      return;
    }
    mainOverlayRenderer.drawOverlay();
    recordDraw(recentMainOverlayDrawTimes, performance.now());
    dirtyMask &= ~DIRTY.MAIN_OVERLAY;
  }

  function runOverviewOverlayTask(): void {
    const state = getState();
    if (!state.audio || !state.sections.overview.enabled || !state.sections.overview.overlay) {
      clearDirtyBits(DIRTY.OVERVIEW_OVERLAY);
      return;
    }
    if (!overviewOverlay || !overviewOverlayCtx) {
      clearDirtyBits(DIRTY.OVERVIEW_OVERLAY);
      return;
    }
    const shouldDraw = playing || (dirtyMask & DIRTY.OVERVIEW_OVERLAY) !== 0;
    if (!shouldDraw) {
      return;
    }
    overviewOverlayRenderer.drawOverviewOverlay();
    recordDraw(recentOverviewOverlayDrawTimes, performance.now());
    dirtyMask &= ~DIRTY.OVERVIEW_OVERLAY;
  }

  function runMainBaseTask(): void {
    const state = getState();
    if (!state.audio || !state.sections.main.enabled) {
      clearDirtyBits(DIRTY.MAIN_BASE);
      return;
    }
    if (!mainBase || !mainBaseCtx) {
      clearDirtyBits(DIRTY.MAIN_BASE);
      return;
    }
    const shouldDraw = (dirtyMask & DIRTY.MAIN_BASE) !== 0;
    if (!shouldDraw) {
      return;
    }
    const frequencyViewport = resolveSpectrumFrequencyViewport(state);
    const interactionState = getInteractionState();
    const mainViewW = getMainViewFrameCount(interactionState);
    const mainSampleStrideFrames = Math.max(
      1,
      getDisplayStrideFramesForZoom({
        zoom: interactionState.spectrumZoom,
        minZoom: 1,
        maxZoom: 20,
        minUnitsPerSecond: state.displaySampling.minUnitsPerSecond,
        maxUnitsPerSecond: state.displaySampling.maxUnitsPerSecond,
        frameRateHz: Math.max(1, Math.round(1 / FRAME_SEC)),
      }),
    );
    const mainViewport = {
      startSlot: Math.max(0, Math.floor(interactionState.spectrumOffset)),
      endSlot: Math.max(
        1,
        Math.min(
          interactionState.spectrumW || timeline?.totalFrames || 0,
          Math.floor(interactionState.spectrumOffset + mainViewW),
        ),
      ),
    };
    renderBaseHeatmapTarget({
      target: "main",
      now: performance.now(),
      state,
      canvas: mainBase,
      ctx: mainBaseCtx,
      viewport: mainViewport,
      representativeMode: state.displaySampling?.representativeMode || "first-valid",
      frequencyViewport,
      sampleStrideFrames: mainSampleStrideFrames,
    });
    recordDraw(recentMainBaseDrawTimes, performance.now());
    dirtyMask &= ~DIRTY.MAIN_BASE;
  }

  function runOverviewBaseTask(): void {
    const state = getState();
    if (!state.audio || !state.sections.overview.enabled) {
      clearDirtyBits(DIRTY.OVERVIEW_BASE);
      return;
    }
    if (!overviewBase || !overviewBaseCtx) {
      clearDirtyBits(DIRTY.OVERVIEW_BASE);
      return;
    }
    const shouldDraw = (dirtyMask & DIRTY.OVERVIEW_BASE) !== 0;
    if (!shouldDraw) {
      return;
    }
    const frequencyViewport = resolveSpectrumFrequencyViewport(state);
    const interactionState = getInteractionState();
    const overviewTotalFrames = Math.max(
      1,
      Math.floor(interactionState.spectrumW || timeline?.totalFrames || 0),
    );
    const overviewSampleStrideFrames = Math.max(
      1,
      getDisplayStrideFramesForZoom({
        zoom: 1,
        minZoom: 1,
        maxZoom: 20,
        minUnitsPerSecond: 1,
        maxUnitsPerSecond: state.displaySampling.maxUnitsPerSecond,
        frameRateHz: Math.max(1, Math.round(1 / FRAME_SEC)),
      }),
    );

    renderBaseHeatmapTarget({
      target: "overview",
      now: performance.now(),
      state,
      canvas: overviewBase,
      ctx: overviewBaseCtx,
      viewport: {
        startSlot: 0,
        endSlot: overviewTotalFrames,
      },
      representativeMode: state.displaySampling?.representativeMode || "first-valid",
      frequencyViewport,
      sampleStrideFrames: overviewSampleStrideFrames,
    });
    recordDraw(recentOverviewBaseDrawTimes, performance.now());
    dirtyMask &= ~DIRTY.OVERVIEW_BASE;
  }

  function finishScheduledFrame(): void {
    if (!dirtyMask && !playing) {
      stopScheduler();
      return;
    }
    // nothing to reset for removed `force` semantics
  }

  function schedule(
    mask = DIRTY.MAIN_BASE |
      DIRTY.MAIN_OVERLAY |
      DIRTY.OVERVIEW_BASE |
      DIRTY.OVERVIEW_OVERLAY,
  ): void {
    dirtyMask |= mask;
    startScheduler();
    syncScheduledTaskRates();
    wakeScheduledTasks(mask);
  }

  function stopLoop(): void {
    stopScheduler();
  }

  function resizeCanvas(
    canvas: HTMLCanvasElement,
    width: number,
    height: number,
    maxDpr = Number.POSITIVE_INFINITY,
  ): void {
    const dpr = Math.max(
      1,
      Math.min(maxDpr, windowRef?.devicePixelRatio || 1),
    );
    const nextWidth = Math.max(1, Math.floor(Math.max(1, width) * dpr));
    const nextHeight = Math.max(1, Math.floor(Math.max(1, height) * dpr));
    if (canvas.width !== nextWidth) canvas.width = nextWidth;
    if (canvas.height !== nextHeight) canvas.height = nextHeight;
  }

  function syncCanvasSizes(): void {
    if (mainSection && mainBase && mainOverlay) {
      resizeCanvas(mainBase, layoutMetrics.mainWidth, layoutMetrics.mainHeight);
      resizeCanvas(mainOverlay, layoutMetrics.mainWidth, layoutMetrics.mainHeight);
    }
    if (overviewSection && overviewBase && overviewOverlay) {
      resizeCanvas(
        overviewBase,
        layoutMetrics.overviewWidth,
        layoutMetrics.overviewHeight,
      );
      resizeCanvas(
        overviewOverlay,
        layoutMetrics.overviewWidth,
        layoutMetrics.overviewHeight,
      );
    }
  }

  function createRoot(): HTMLDivElement | null {
    if (!documentRef) return null;
    if (root) return root;

    const nextRoot = createSpectrumRootContainer(documentRef);
    nextRoot.style.gap = "0";
    const nextPlaceholder = documentRef.createElement("div");
    nextPlaceholder.style.position = "absolute";
    nextPlaceholder.style.inset = "12px";
    nextPlaceholder.style.display = "flex";
    nextPlaceholder.style.alignItems = "center";
    nextPlaceholder.style.justifyContent = "center";
    nextPlaceholder.style.flexDirection = "column";
    nextPlaceholder.style.gap = "10px";
    nextPlaceholder.style.color = "rgba(94, 106, 123, 0.96)";
    nextPlaceholder.style.background = "rgba(245, 248, 255, 0.82)";
    nextPlaceholder.style.border = "1px dashed rgba(54, 80, 130, 0.16)";
    nextPlaceholder.style.pointerEvents = "none";
    nextPlaceholder.textContent = "选择音频后开始显示频谱";
    placeholder = nextPlaceholder;

    const nextSplitter = documentRef.createElement("div");
    nextSplitter.style.height = "2px";
    nextSplitter.style.flex = "0 0 auto";
    nextSplitter.style.position = "relative";
    nextSplitter.style.borderRadius = "999px";
    nextSplitter.style.cursor = "row-resize";
    nextSplitter.style.touchAction = "none";
    nextSplitter.style.userSelect = "none";
    nextSplitter.style.display = "flex";
    nextSplitter.style.alignItems = "center";
    nextSplitter.style.justifyContent = "center";
    nextSplitter.style.padding = "0";
    nextSplitter.style.background = "transparent";
    nextSplitter.style.opacity = "0.06";
    nextSplitter.style.transition = "opacity 0.12s ease";
    nextSplitter.setAttribute("role", "separator");
    nextSplitter.setAttribute("aria-orientation", "horizontal");
    nextSplitter.title = "拖拽调整 overview / main 比例";
    const nextSplitterHandle = documentRef.createElement("div");
    nextSplitterHandle.style.width = "100%";
    nextSplitterHandle.style.height = "2px";
    nextSplitterHandle.style.borderRadius = "999px";
    nextSplitterHandle.style.background = "rgba(17, 87, 216, 0.16)";
    nextSplitterHandle.style.boxShadow = "none";
    nextSplitter.appendChild(nextSplitterHandle);
    splitBar = nextSplitter;
    splitBarHandle = nextSplitterHandle;

    const overview = createSpectrumSection(documentRef, "overview", MIN_OVERVIEW_SECTION_HEIGHT);
    const main = createSpectrumSection(documentRef, "main", MIN_MAIN_SECTION_HEIGHT);
    main.section.style.flex = "1 1 auto";
    overview.section.style.flex = "1 1 auto";
    main.section.style.minHeight = `${MIN_MAIN_SECTION_HEIGHT}px`;
    overview.section.style.minHeight = `${MIN_OVERVIEW_SECTION_HEIGHT}px`;

    const nextToolbar = documentRef.createElement("div");
    nextToolbar.style.position = "absolute";
    nextToolbar.style.top = "12px";
    nextToolbar.style.right = "12px";
    nextToolbar.style.zIndex = "4";
    nextToolbar.style.display = "flex";
    nextToolbar.style.gap = "8px";
    nextToolbar.style.alignItems = "center";
    nextToolbar.style.pointerEvents = "none";
    nextToolbar.style.opacity = "0";
    nextToolbar.style.transform = "translateY(-4px)";
    nextToolbar.style.transition = "opacity 0.12s ease, transform 0.12s ease";
    const makeToolbarButton = (title: string) => {
      const button = documentRef.createElement("button");
      button.type = "button";
      button.style.width = "32px";
      button.style.height = "32px";
      button.style.display = "inline-flex";
      button.style.alignItems = "center";
      button.style.justifyContent = "center";
      button.style.border = "1px solid rgba(17, 87, 216, 0.14)";
      button.style.borderRadius = "999px";
      button.style.background = "rgba(255, 255, 255, 0.88)";
      button.style.color = "var(--primary-strong)";
      button.style.boxShadow = "0 10px 18px rgba(19, 37, 70, 0.08)";
      button.style.cursor = "pointer";
      button.style.userSelect = "none";
      button.style.touchAction = "manipulation";
      button.style.pointerEvents = "auto";
      button.title = title;
      button.setAttribute("aria-label", title);
      return button;
    };
    const nextFullscreenToggle = makeToolbarButton("主图全屏 (F)");
    nextFullscreenToggle.addEventListener("click", () => {
      const interactionState = getInteractionState();
      void setMainFullscreen(!interactionState.spectrumMainFullscreen);
    });
    nextToolbar.appendChild(nextFullscreenToggle);
    main.section.appendChild(nextToolbar);

    nextRoot.appendChild(nextPlaceholder);
    nextRoot.appendChild(overview.section);
    nextRoot.appendChild(nextSplitter);
    nextRoot.appendChild(main.section);
    root = nextRoot;
    overviewSection = overview.section;
    overviewBase = overview.base;
    overviewOverlay = overview.overlay;
    overviewBaseCtx = overview.baseCtx;
    overviewOverlayCtx = overview.overlayCtx;
    mainSection = main.section;
    toolbar = nextToolbar;
    mainBase = main.base;
    mainOverlay = main.overlay;
    mainBaseCtx = main.baseCtx;
    mainOverlayCtx = main.overlayCtx;
    mainFullscreenToggle = nextFullscreenToggle;
    refreshLayoutMetricsFromElements();

    if (typeof ResizeObserver !== "undefined") {
      resizeObserver?.disconnect();
      resizeObserver = new ResizeObserver((entries) => {
        refreshLayoutMetricsFromObserver(entries);
        schedule();
      });
      resizeObserver.observe(nextRoot);
      resizeObserver.observe(main.section);
      resizeObserver.observe(overview.section);
      resizeObserver.observe(nextSplitter);
    }
    if (documentRef && typeof documentRef.addEventListener === "function") {
      const handleFullscreenChange = () => {
        syncFullscreenStateFromDocument();
      };
      const handleFullscreenError = () => {
        syncFullscreenStateFromDocument();
      };
      documentRef.addEventListener("fullscreenchange", handleFullscreenChange);
      documentRef.addEventListener("fullscreenerror", handleFullscreenError);
      fullscreenCleanup = () => {
        documentRef.removeEventListener("fullscreenchange", handleFullscreenChange);
        documentRef.removeEventListener("fullscreenerror", handleFullscreenError);
      };
      cleanups.push(() => {
        fullscreenCleanup?.();
        fullscreenCleanup = null;
      });
    }
    const splitDraggingState = {
      active: false,
      pointerId: -1,
    };
    const getSplitPoint = (event: MouseEvent | PointerEvent | TouchEvent | null) => {
      const point = getClientPointFromEvent(event);
      if (!point) {
        return null;
      }
      return point;
    };
    const beginSplitDrag = (event: MouseEvent | PointerEvent | TouchEvent | null) => {
      if (!mainSection || !overviewSection) {
        return;
      }
      const state = getState();
      if (!state.audio || state.sections.main.enabled === false || state.sections.overview.enabled === false) {
        return;
      }
      preventDefaultIfPossible(event);
      const point = getSplitPoint(event);
      if (!point) {
        return;
      }
      const rootRect = root?.getBoundingClientRect() || null;
      if (!rootRect || rootRect.height <= 0) {
        return;
      }
      splitDragRootTop = rootRect.top;
      splitDragRootHeight = rootRect.height;
      splitDraggingState.active = true;
      nextSplitter.style.opacity = "0.14";
      nextSplitterHandle.style.opacity = "0.3";
      const pointerEvent = event as PointerEvent | null;
      splitDraggingState.pointerId =
        pointerEvent && Number.isFinite(pointerEvent.pointerId) ? pointerEvent.pointerId : -1;
      updateOverviewRatioFromPoint(point.clientY);
      try {
        if (
          splitDraggingState.pointerId >= 0 &&
          nextSplitter.setPointerCapture
        ) {
          nextSplitter.setPointerCapture(splitDraggingState.pointerId);
        }
      } catch {}
    };
    const moveSplitDrag = (event: MouseEvent | PointerEvent | TouchEvent | null) => {
      if (!splitDraggingState.active) {
        return;
      }
      preventDefaultIfPossible(event);
      const point = getSplitPoint(event);
      if (!point) {
        return;
      }
      updateOverviewRatioFromPoint(point.clientY);
    };
    const endSplitDrag = (event: MouseEvent | PointerEvent | TouchEvent | null) => {
      if (!splitDraggingState.active) {
        return;
      }
      preventDefaultIfPossible(event);
      const point = getSplitPoint(event);
      if (point) {
        updateOverviewRatioFromPoint(point.clientY);
      }
      splitDraggingState.active = false;
      splitDragRootTop = 0;
      splitDragRootHeight = 0;
      try {
        const pointerEvent = event as PointerEvent | null;
        if (
          splitDraggingState.pointerId >= 0 &&
          nextSplitter.releasePointerCapture &&
          pointerEvent &&
          pointerEvent.pointerId === splitDraggingState.pointerId
        ) {
          nextSplitter.releasePointerCapture(pointerEvent.pointerId);
        }
      } catch {}
      splitDraggingState.pointerId = -1;
      nextSplitter.style.opacity = "0.06";
      nextSplitterHandle.style.opacity = "0.16";
    };
    const handleSplitterPointerEnter = () => {
      if (!splitDraggingState.active) {
        nextSplitter.style.opacity = "0.12";
        nextSplitterHandle.style.opacity = "0.2";
      }
    };
    const handleSplitterPointerLeave = () => {
      if (!splitDraggingState.active) {
        nextSplitter.style.opacity = "0.06";
        nextSplitterHandle.style.opacity = "0.16";
      }
    };
    nextSplitter.onpointerdown = null;
    nextSplitter.onpointermove = null;
    nextSplitter.onpointerup = null;
    nextSplitter.onpointercancel = null;
    nextSplitter.onmousedown = null;
    nextSplitter.onmousemove = null;
    nextSplitter.onmouseup = null;
    nextSplitter.ontouchstart = null;
    nextSplitter.ontouchmove = null;
    nextSplitter.ontouchend = null;
    nextSplitter.ontouchcancel = null;
    const hasPointerEvents =
      !!windowRef &&
      typeof (
        windowRef as Window & { PointerEvent?: typeof PointerEvent }
      ).PointerEvent === "function";
    if (hasPointerEvents) {
      nextSplitter.onpointerdown = beginSplitDrag;
      nextSplitter.onpointermove = moveSplitDrag;
      nextSplitter.onpointerup = endSplitDrag;
      nextSplitter.onpointercancel = endSplitDrag;
    } else {
      nextSplitter.onmousedown = beginSplitDrag;
      nextSplitter.onmousemove = moveSplitDrag;
      nextSplitter.onmouseup = endSplitDrag;
      nextSplitter.ontouchstart = beginSplitDrag;
      nextSplitter.ontouchmove = moveSplitDrag;
      nextSplitter.ontouchend = endSplitDrag;
      nextSplitter.ontouchcancel = endSplitDrag;
    }
    nextSplitter.addEventListener("pointerenter", handleSplitterPointerEnter);
    nextSplitter.addEventListener("pointerleave", handleSplitterPointerLeave);
    const handleMainPointerEnter = () => {
      syncToolbarVisibility(true);
    };
    const handleMainPointerLeave = () => {
      syncToolbarVisibility(false);
    };
    main.section.addEventListener("pointerenter", handleMainPointerEnter);
    main.section.addEventListener("pointerleave", handleMainPointerLeave);
    cleanups.push(() => {
      nextSplitter.removeEventListener("pointerenter", handleSplitterPointerEnter);
      nextSplitter.removeEventListener("pointerleave", handleSplitterPointerLeave);
      main.section.removeEventListener("pointerenter", handleMainPointerEnter);
      main.section.removeEventListener("pointerleave", handleMainPointerLeave);
    });
    syncSectionVisibility();
    schedule();
    return nextRoot;
  }

  function attachAudioElement(nextAudioElement: HTMLAudioElement | null): void {
    if (audioCleanup) {
      audioCleanup();
      audioCleanup = null;
    }
    audioElement = nextAudioElement;
    if (!audioElement || typeof audioElement.addEventListener !== "function") {
      playing = false;
      schedule();
      return;
    }
    const currentAudioElement = audioElement;

    const refresh = () => {
      playing = !currentAudioElement.paused && !currentAudioElement.ended;
      schedule(DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_OVERLAY);
    };
    const onTime = () => schedule(DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_OVERLAY);
    const pairs: Array<[string, EventListener]> = [
      ["timeupdate", onTime],
      ["play", refresh],
      ["playing", refresh],
      ["pause", refresh],
      ["ended", refresh],
      ["seeked", onTime],
      ["seeking", onTime],
      ["loadedmetadata", refresh],
      ["ratechange", refresh],
    ];
    for (const [type, listener] of pairs) {
      currentAudioElement.addEventListener(type, listener);
    }
    audioCleanup = () => {
      for (const [type, listener] of pairs) {
        currentAudioElement.removeEventListener(type, listener);
      }
    };
    refresh();
  }

  function mount(nextMount: HTMLElement | null): void {
    const nextRoot = createRoot();
    if (nextRoot) {
      if (nextMount) {
        if (nextRoot.parentElement !== nextMount) {
          nextMount.appendChild(nextRoot);
        }
      } else if (nextRoot.parentElement) {
        nextRoot.parentElement.removeChild(nextRoot);
      }
    }
    schedule();
  }

  function setTimeline(nextTimeline: HeatmapTimeline | null): void {
    timeline = nextTimeline;
    schedule(DIRTY.MAIN_BASE | DIRTY.OVERVIEW_BASE);
  }

  function requestSpectrumRedraw(
    next?: { includeOverviewBase?: boolean; dirtyMask?: number },
  ): void {
    const dirtyMask =
      typeof next === "object" && next && Number.isFinite(next.dirtyMask)
        ? Math.max(0, Math.floor(Number(next.dirtyMask)))
        : null;
    const includeOverviewBase =
      dirtyMask !== null
        ? (dirtyMask & DIRTY.OVERVIEW_BASE) !== 0
        : next?.includeOverviewBase !== false;
    const resolvedMask =
      dirtyMask !== null
        ? dirtyMask
        : DIRTY.MAIN_BASE |
          DIRTY.MAIN_OVERLAY |
          (includeOverviewBase ? DIRTY.OVERVIEW_BASE : 0) |
          DIRTY.OVERVIEW_OVERLAY;
    schedule(resolvedMask);
  }

  function requestOverviewOverlayRedraw(): void {
    schedule(DIRTY.OVERVIEW_OVERLAY);
  }

  function runHeatmapBenchmark(rounds = 3): HeatmapBenchmarkEntry[] {
    if (!timeline || !mainBase || !mainBaseCtx) {
      return [];
    }
    const state = getState();
    const interactionState = getInteractionState();
    const spectrumBinCount = Math.max(1, Math.floor(interactionState.spectrumH || timeline.freqCount || 1));
    const frequencyViewport: HeatmapFrequencyViewport = {
      minBin: Math.max(0, Math.min(spectrumBinCount - 1, Math.floor(state.pitchRange.minBin || 0))),
      maxBin: Math.max(0, Math.min(spectrumBinCount - 1, Math.floor(state.pitchRange.maxBin || (spectrumBinCount - 1)))),
    };
    if (frequencyViewport.maxBin < frequencyViewport.minBin) {
      frequencyViewport.maxBin = frequencyViewport.minBin;
    }
    const mainViewW = getMainViewFrameCount(interactionState);
    const mainViewport = {
      startSlot: Math.max(0, Math.floor(interactionState.spectrumOffset)),
      endSlot: Math.max(
        1,
        Math.min(
          interactionState.spectrumW || timeline.totalFrames || 0,
          Math.floor(interactionState.spectrumOffset + mainViewW),
        ),
      ),
    };
    const sampleStrideFrames = Math.max(
      1,
      getDisplayStrideFramesForZoom({
        zoom: interactionState.spectrumZoom,
        minZoom: 1,
        maxZoom: 20,
        minUnitsPerSecond: state.displaySampling.minUnitsPerSecond,
        maxUnitsPerSecond: state.displaySampling.maxUnitsPerSecond,
        frameRateHz: Math.max(1, Math.round(1 / FRAME_SEC)),
      }),
    );
    return benchmarkHeatmapTimelineRender(
      mainBaseCtx,
      mainBase.width || mainBase.clientWidth || 1,
      mainBase.height || mainBase.clientHeight || 1,
      timeline,
      mainViewport,
      "frames",
      state.displaySampling?.representativeMode || "first-valid",
      frequencyViewport,
      sampleStrideFrames,
      rounds,
    );
  }

  function markAutoPanSuppressedPublic(nowTs?: number, durationMs?: number): void {
    markAutoPanSuppressed(nowTs, durationMs);
  }

  function destroy(): void {
    if (documentRef?.fullscreenElement === mainSection && documentRef.exitFullscreen) {
      void documentRef.exitFullscreen().catch(() => undefined);
    }
    resizeObserver?.disconnect();
    resizeObserver = null;
    audioCleanup?.();
    audioCleanup = null;
    while (cleanups.length > 0) {
      const cleanup = cleanups.pop();
      try {
        cleanup?.();
      } catch {}
    }
    stopLoop();
    destroyHeatmapWorker();
    frameTaskId = null;
    mainBaseTaskId = null;
    mainOverlayTaskId = null;
    overviewBaseTaskId = null;
    overviewOverlayTaskId = null;
    finalizeTaskId = null;
    schedulerTasksReady = false;
    schedulerStarted = false;
    taskRateSignature = "";
    dirtyMask = 0;
    
    if (root?.parentElement) {
      root.parentElement.removeChild(root);
    }
    root = null;
    placeholder = null;
    splitBar = null;
    splitBarHandle = null;
    toolbar = null;
    mainFullscreenToggle = null;
    controlsVisible = false;
    mainSection = null;
    overviewSection = null;
    mainBase = null;
    mainOverlay = null;
    overviewBase = null;
    overviewOverlay = null;
    mainBaseCtx = null;
    mainOverlayCtx = null;
    overviewBaseCtx = null;
    overviewOverlayCtx = null;
  }

  function getMainCanvas(): HTMLCanvasElement | null {
    return mainBase;
  }

  function getOverviewCanvas(): HTMLCanvasElement | null {
    return overviewBase;
  }

  function getMainOverlayCanvas(): HTMLCanvasElement | null {
    return mainOverlay;
  }

  function getOverviewOverlayCanvas(): HTMLCanvasElement | null {
    return overviewOverlay;
  }

  function buildCanvasStats(
    canvas: HTMLCanvasElement | null,
    taskId: string | null,
  ): SpectrumRenderCanvasStats {
    const task = getTaskSnapshot(
      taskId,
      canvas === mainBase
        ? recentMainBaseDrawTimes
        : canvas === overviewBase
          ? recentOverviewBaseDrawTimes
          : [],
    );
    const cssWidth = Math.max(1, canvas?.clientWidth || Math.floor((canvas?.width || 1) / Math.max(1, windowRef?.devicePixelRatio || 1)) || 1);
    const cssHeight = Math.max(1, canvas?.clientHeight || Math.floor((canvas?.height || 1) / Math.max(1, windowRef?.devicePixelRatio || 1)) || 1);
    const dpr = canvas ? Math.max(1, canvas.width / cssWidth, canvas.height / cssHeight) : Math.max(1, windowRef?.devicePixelRatio || 1);
    return {
      targetFps: task?.targetFps || 0,
      actualFps: task?.actualFps || 0,
      runCount: task?.runCount || 0,
      nextRunInMs: task?.nextRunInMs || 0,
      canvasWidth: canvas?.width || 0,
      canvasHeight: canvas?.height || 0,
      dpr,
    };
  }

  function getRenderStats(): SpectrumRenderStats {
    return {
      mainBase: buildCanvasStats(mainBase, mainBaseTaskId),
      mainOverlay: getTaskSnapshot(mainOverlayTaskId, recentMainOverlayDrawTimes) || {
        targetFps: 0,
        actualFps: 0,
        runCount: 0,
        nextRunInMs: 0,
      },
      overviewBase: buildCanvasStats(overviewBase, overviewBaseTaskId),
      overviewOverlay: getTaskSnapshot(overviewOverlayTaskId, recentOverviewOverlayDrawTimes) || {
        targetFps: 0,
        actualFps: 0,
        runCount: 0,
        nextRunInMs: 0,
      },
      dirty: {
        currentMask: dirtyMask,
        playing,
        schedulerRunning: schedulerStarted,
      },
    };
  }

  return {
    mount,
    attachAudioElement,
    setTimeline,
    toggleMainFullscreen: setMainFullscreen,
    requestSpectrumRedraw,
    requestOverviewOverlayRedraw,
    runHeatmapBenchmark,
    markAutoPanSuppressed: markAutoPanSuppressedPublic,
    destroy,
    getMainCanvas,
    getOverviewCanvas,
    getMainOverlayCanvas,
    getOverviewOverlayCanvas,
    getRenderStats,
  };
}
