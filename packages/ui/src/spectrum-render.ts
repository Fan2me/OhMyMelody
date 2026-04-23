import {
  benchmarkHeatmapTimelineRender,
  renderHeatmapTimeline,
  type HeatmapBenchmarkEntry,
  type HeatmapFrequencyViewport,
  type HeatmapOptimizationLevel,
  type HeatmapTimeline,
} from "./heatmap-render-core.js";
import {
  DEFAULT_MAIN_HEIGHT,
  DEFAULT_OVERVIEW_HEIGHT,
  DEFAULT_MAIN_FPS,
  DEFAULT_OVERVIEW_FPS,
  DIRTY,
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

export interface SpectrumRenderController {
  mount(nextMount: HTMLElement | null): void;
  attachAudioElement(nextAudioElement: HTMLAudioElement | null): void;
  setTimeline(nextTimeline: HeatmapTimeline | null): void;
  requestSpectrumRedraw(next?: boolean | { force?: boolean; includeOverviewBase?: boolean; dirtyMask?: number }): void;
  requestOverviewOverlayRedraw(): void;
  runHeatmapBenchmark(rounds?: number): HeatmapBenchmarkEntry[];
  markAutoPanSuppressed(nowTs?: number, durationMs?: number): void;
  destroy(): void;
  getMainCanvas(): HTMLCanvasElement | null;
  getOverviewCanvas(): HTMLCanvasElement | null;
  getMainOverlayCanvas(): HTMLCanvasElement | null;
  getOverviewOverlayCanvas(): HTMLCanvasElement | null;
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
  let resizeObserver: ResizeObserver | null = null;
  let audioCleanup: (() => void) | null = null;
  let audioElement: HTMLAudioElement | null = null;
  let rafId: number | null = null;
  let dirtyMask = 0;
  let forceRender = false;
  let playing = false;
  let lastAutoPanAt = 0;
  let autoPanSuppressedUntil = 0;
  let heatmapTimelineSeq = 0;
  let timeline: HeatmapTimeline | null = null;
  const lastDrawAt = {
    mainBase: 0,
    mainOverlay: 0,
    overviewBase: 0,
    overviewOverlay: 0,
  };

  const AUTO_PAN_MIN_INTERVAL_MS = 33;
  const AUTO_PAN_EDGE_RATIO = 1 / 10;
  const AUTO_PAN_SUPPRESS_MS = 250;

  function destroyHeatmapWorker(): void {
    return;
  }

  function clearQueuedHeatmapRequests(): void {
    return;
  }

  function attachHeatmapWorkerHandlers(nextWorker: Worker): Worker {
    return nextWorker;
  }

  function flushQueuedHeatmapRequest(): void {
    return;
  }

  function buildHeatmapRenderSignature(args: {
    target: "main" | "overview";
    width: number;
    height: number;
    viewport: { startSlot: number; endSlot: number } | null;
    viewportMode: "slots" | "frames";
    representativeMode: string;
    frequencyViewport: HeatmapFrequencyViewport | null;
    sampleStrideFrames: number;
    optimizationLevel: HeatmapOptimizationLevel;
  }): string {
    const vp = args.viewport;
    const fvp = args.frequencyViewport;
    return [
      `seq:${heatmapTimelineSeq}`,
      `target:${args.target}`,
      `size:${args.width}x${args.height}`,
      `mode:${args.viewportMode}`,
      `rep:${args.representativeMode}`,
      `stride:${args.sampleStrideFrames}`,
      `opt:${args.optimizationLevel}`,
      `vp:${vp ? `${vp.startSlot}-${vp.endSlot}` : "none"}`,
      `fvp:${fvp ? `${fvp.minBin}-${fvp.maxBin}` : "none"}`,
    ].join("|");
  }

  function syncSectionVisibility(): void {
    const state = getState();
    const hasAudio = !!state.audio;
    if (placeholder) {
      placeholder.style.display = hasAudio ? "none" : "flex";
    }
    const mainVisible = hasAudio && state.sections.main.enabled;
    const overviewVisible = hasAudio && state.sections.overview.enabled;
    if (mainSection) {
      mainSection.style.display = mainVisible ? "" : "none";
      mainSection.style.flex =
        mainVisible && overviewVisible ? "1 1 auto" : mainVisible ? "1 1 auto" : "0 0 auto";
    }
    if (overviewSection) {
      overviewSection.style.display = overviewVisible ? "" : "none";
      overviewSection.style.flex =
        overviewVisible && mainVisible ? "0 0 auto" : overviewVisible ? "1 1 auto" : "0 0 auto";
    }
    if (mainOverlay) {
      mainOverlay.style.display =
        mainVisible && state.sections.main.overlay ? "" : "none";
    }
    if (overviewOverlay) {
      overviewOverlay.style.display =
        overviewVisible && state.sections.overview.overlay ? "" : "none";
    }
  }

  function getPlaybackFrame(): number {
    const currentTime = Number(audioElement?.currentTime ?? NaN);
    if (!Number.isFinite(currentTime) || currentTime < 0) {
      return 0;
    }
    return currentTime / FRAME_SEC;
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

    const edgeThreshold = Math.max(1, viewW * AUTO_PAN_EDGE_RATIO);
    let desiredOffset = offset;

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
    return true;
  }

  function ensureHeatmapWorker(): Worker | null {
    return null;
  }

  function postHeatmapTimeline(): void {
    return;
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
    if (target === "main") {
      lastDrawAt.mainBase = now;
    } else {
      lastDrawAt.overviewBase = now;
    }
  }

  function schedule(
    mask = DIRTY.MAIN_BASE |
      DIRTY.MAIN_OVERLAY |
      DIRTY.OVERVIEW_BASE |
      DIRTY.OVERVIEW_OVERLAY,
    force = false,
  ): void {
    dirtyMask |= mask;
    forceRender = forceRender || force;
    if (!windowRef) {
      renderFrame(performance.now());
      return;
    }
    if (rafId === null) {
      rafId = windowRef.requestAnimationFrame(renderFrame);
    }
  }

  function stopLoop(): void {
    if (rafId !== null && windowRef) {
      windowRef.cancelAnimationFrame(rafId);
    }
    rafId = null;
  }

  function resizeCanvas(canvas: HTMLCanvasElement, host: HTMLElement): void {
    const dpr = Math.max(1, windowRef?.devicePixelRatio || 1);
    const width = Math.max(1, Math.floor(host.clientWidth || host.getBoundingClientRect().width || 1));
    const height = Math.max(1, Math.floor(host.clientHeight || host.getBoundingClientRect().height || 1));
    const nextWidth = Math.max(1, Math.floor(width * dpr));
    const nextHeight = Math.max(1, Math.floor(height * dpr));
    if (canvas.width !== nextWidth) canvas.width = nextWidth;
    if (canvas.height !== nextHeight) canvas.height = nextHeight;
  }

  function syncCanvasSizes(): void {
    if (mainSection && mainBase && mainOverlay) {
      resizeCanvas(mainBase, mainSection);
      resizeCanvas(mainOverlay, mainSection);
    }
    if (overviewSection && overviewBase && overviewOverlay) {
      resizeCanvas(overviewBase, overviewSection);
      resizeCanvas(overviewOverlay, overviewSection);
    }
  }

  function renderFrame(now: number): void {
    rafId = null;
    syncSectionVisibility();
    const state = getState();
    if (!state.audio) {
      forceRender = false;
      dirtyMask = 0;
      return;
    }
    const representativeMode = state.displaySampling?.representativeMode || "first-valid";
    if (playing) {
      dirtyMask |= DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_OVERLAY;
    }
    maybeAutoPanToPlaybackHead(now);
    syncCanvasSizes();

    const pending = dirtyMask;
    dirtyMask = 0;

    const mainFps = Number.isFinite(state.sections.main.fps)
      ? Math.max(0, Math.floor(state.sections.main.fps))
      : DEFAULT_MAIN_FPS;
    const mainOverlayFps = Number.isFinite(state.sections.main.overlayFps)
      ? Math.max(0, Math.floor(state.sections.main.overlayFps))
      : mainFps;
    const overviewFps = Number.isFinite(state.sections.overview.fps)
      ? Math.max(0, Math.floor(state.sections.overview.fps))
      : DEFAULT_OVERVIEW_FPS;
    const overviewOverlayFps = Number.isFinite(state.sections.overview.overlayFps)
      ? Math.max(0, Math.floor(state.sections.overview.overlayFps))
      : overviewFps;

    if (
      pending & DIRTY.MAIN_OVERLAY &&
      state.sections.main.enabled &&
      state.sections.main.overlay &&
      mainOverlay &&
      mainOverlayCtx
    ) {
      if (forceRender || mainOverlayFps <= 0 || now - lastDrawAt.mainOverlay >= 1000 / mainOverlayFps) {
        mainOverlayRenderer.drawOverlay();
        lastDrawAt.mainOverlay = now;
      } else {
        dirtyMask |= DIRTY.MAIN_OVERLAY;
      }
    }
    if (
      pending & DIRTY.OVERVIEW_OVERLAY &&
      state.sections.overview.enabled &&
      state.sections.overview.overlay &&
      overviewOverlay &&
      overviewOverlayCtx
    ) {
      if (forceRender || overviewOverlayFps <= 0 || now - lastDrawAt.overviewOverlay >= 1000 / overviewOverlayFps) {
        overviewOverlayRenderer.drawOverviewOverlay();
        lastDrawAt.overviewOverlay = now;
      } else {
        dirtyMask |= DIRTY.OVERVIEW_OVERLAY;
      }
    }
    if (
      pending & DIRTY.MAIN_BASE &&
      state.sections.main.enabled &&
      mainBase &&
      mainBaseCtx
    ) {
      if (forceRender || mainFps <= 0 || now - lastDrawAt.mainBase >= 1000 / mainFps) {
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
          now,
          state,
          canvas: mainBase,
          ctx: mainBaseCtx,
          viewport: mainViewport,
          representativeMode,
          frequencyViewport,
          sampleStrideFrames: mainSampleStrideFrames,
        });
      } else {
        dirtyMask |= DIRTY.MAIN_BASE;
      }
    }
    if (
      pending & DIRTY.OVERVIEW_BASE &&
      state.sections.overview.enabled &&
      overviewBase &&
      overviewBaseCtx
    ) {
      if (forceRender || overviewFps <= 0 || now - lastDrawAt.overviewBase >= 1000 / overviewFps) {
        const frequencyViewport = resolveSpectrumFrequencyViewport(state);
        const interactionState = getInteractionState();
        const overviewTotalFrames = Math.max(
          1,
          Math.floor(interactionState.spectrumW || timeline?.totalFrames || 0),
        );
        const overviewViewport = {
          startSlot: 0,
          endSlot: overviewTotalFrames,
        };
        renderBaseHeatmapTarget({
          target: "overview",
          now,
          state,
          canvas: overviewBase,
          ctx: overviewBaseCtx,
          viewport: overviewViewport,
          representativeMode,
          frequencyViewport,
          sampleStrideFrames: 1,
        });
      } else {
        dirtyMask |= DIRTY.OVERVIEW_BASE;
      }
    }

    forceRender = false;
    if (dirtyMask || playing) {
      schedule(0);
    }
  }

  function createRoot(): HTMLDivElement | null {
    if (!documentRef) return null;
    if (root) return root;

    const nextRoot = createSpectrumRootContainer(documentRef);
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

    const overview = createSpectrumSection(documentRef, "overview", DEFAULT_OVERVIEW_HEIGHT);
    const main = createSpectrumSection(documentRef, "main", DEFAULT_MAIN_HEIGHT);
    main.section.style.flex = "1 1 auto";
    overview.section.style.flex = "0 0 auto";

    nextRoot.appendChild(nextPlaceholder);
    nextRoot.appendChild(overview.section);
    nextRoot.appendChild(main.section);
    root = nextRoot;
    overviewSection = overview.section;
    overviewBase = overview.base;
    overviewOverlay = overview.overlay;
    overviewBaseCtx = overview.baseCtx;
    overviewOverlayCtx = overview.overlayCtx;
    mainSection = main.section;
    mainBase = main.base;
    mainOverlay = main.overlay;
    mainBaseCtx = main.baseCtx;
    mainOverlayCtx = main.overlayCtx;

    if (typeof ResizeObserver !== "undefined") {
      resizeObserver?.disconnect();
      resizeObserver = new ResizeObserver(() => schedule(undefined, true));
      resizeObserver.observe(nextRoot);
      resizeObserver.observe(main.section);
      resizeObserver.observe(overview.section);
    }
    syncSectionVisibility();
    schedule(undefined, true);
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
      schedule(undefined, true);
      return;
    }
    const currentAudioElement = audioElement;

    const refresh = () => {
      playing = !currentAudioElement.paused && !currentAudioElement.ended;
      schedule(DIRTY.MAIN_OVERLAY | DIRTY.OVERVIEW_OVERLAY, true);
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
    schedule(undefined, true);
  }

  function setTimeline(nextTimeline: HeatmapTimeline | null): void {
    timeline = nextTimeline;
    schedule(DIRTY.MAIN_BASE | DIRTY.OVERVIEW_BASE, true);
  }

  function requestSpectrumRedraw(
    next?: boolean | { force?: boolean; includeOverviewBase?: boolean; dirtyMask?: number },
  ): void {
    const force = typeof next === "boolean" ? next : !!next?.force;
    const dirtyMask =
      typeof next === "object" && next && Number.isFinite(next.dirtyMask)
        ? Math.max(0, Math.floor(Number(next.dirtyMask)))
        : null;
    const includeOverviewBase =
      dirtyMask !== null
        ? (dirtyMask & DIRTY.OVERVIEW_BASE) !== 0
        : typeof next === "boolean"
          ? true
          : next?.includeOverviewBase !== false;
    const resolvedMask =
      dirtyMask !== null
        ? dirtyMask
        : DIRTY.MAIN_BASE |
          DIRTY.MAIN_OVERLAY |
          (includeOverviewBase ? DIRTY.OVERVIEW_BASE : 0) |
          DIRTY.OVERVIEW_OVERLAY;
    schedule(
      resolvedMask,
      force,
    );
  }

  function requestOverviewOverlayRedraw(): void {
    schedule(DIRTY.OVERVIEW_OVERLAY, false);
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
    resizeObserver?.disconnect();
    resizeObserver = null;
    audioCleanup?.();
    audioCleanup = null;
    stopLoop();
    destroyHeatmapWorker();
    if (root?.parentElement) {
      root.parentElement.removeChild(root);
    }
    root = null;
    placeholder = null;
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

  return {
    mount,
    attachAudioElement,
    setTimeline,
    requestSpectrumRedraw,
    requestOverviewOverlayRedraw,
    runHeatmapBenchmark,
    markAutoPanSuppressed: markAutoPanSuppressedPublic,
    destroy,
    getMainCanvas,
    getOverviewCanvas,
    getMainOverlayCanvas,
    getOverviewOverlayCanvas,
  };
}
