import type { DisplaySamplingConfig } from "./display-sampling.js";
import type { SpectrumInteractionState } from "./spectrum-state.js";
import { getMainViewFrameCount, clampSpectrumZoom, getPlotMetrics } from "./spectrum-layout.js";

export interface SpectrumInteractionControllerDeps {
  windowRef: Window | null;
  getMainCanvas: () => HTMLCanvasElement | null;
  getOverviewCanvas: () => HTMLCanvasElement | null;
  getAxisX: () => number;
  getMainTimelineHeight: () => number;
  getState: () => SpectrumInteractionState;
  getDisplaySamplingConfig: () => DisplaySamplingConfig;
  setState: (partial: Partial<SpectrumInteractionState>) => void;
  seekAudioTime: (timeSec: number) => void | Promise<void>;
  markAutoPanSuppressed: (nowTs?: number, durationMs?: number) => void;
  requestSpectrumRedraw: (force?: boolean) => void;
  requestOverviewOverlayRedraw: () => void;
}

export interface SpectrumInteractionController {
  bind(): void;
  destroy(): void;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
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

export function createSpectrumInteractionController(
  deps: SpectrumInteractionControllerDeps,
): SpectrumInteractionController {
  const {
    windowRef,
    getMainCanvas,
    getOverviewCanvas,
    getState,
    setState,
    getAxisX,
    getMainTimelineHeight,
    seekAudioTime,
    markAutoPanSuppressed,
    requestSpectrumRedraw,
    requestOverviewOverlayRedraw,
    getDisplaySamplingConfig,
  } = deps;

  const cleanups: Array<() => void> = [];

  function getMainPlotMetrics(canvas: HTMLCanvasElement) {
    return getPlotMetrics(canvas, getAxisX(), getMainTimelineHeight());
  }

  function bindMainCanvas(): void {
    const canvas = getMainCanvas();
    if (!canvas) return;
    const targetCanvas: HTMLCanvasElement = canvas;
    targetCanvas.style.touchAction = "none";
    const windowWithPointer = windowRef as (Window & { PointerEvent?: typeof PointerEvent }) | null;
    const hasPointerEvents = !!windowWithPointer && typeof windowWithPointer.PointerEvent === "function";
    const activePointers = new Map<number, { clientX: number; clientY: number }>();
    let pinchGesture: null | {
      startDistance: number;
      startZoom: number;
      anchorFrac: number;
      anchorFrame: number;
    } = null;

    const getTouchPoints = (event: TouchEvent | null) => {
      const touches = event && event.touches ? Array.from(event.touches) : [];
      return touches.map((touch) => ({
        clientX: touch.clientX,
        clientY: touch.clientY,
      }));
    };

    const getActivePointerPoints = () => Array.from(activePointers.values()).slice(0, 2);

    const clearPinchGesture = () => {
      pinchGesture = null;
    };

    const releaseCapturedPointers = () => {
      for (const pointerId of activePointers.keys()) {
        try {
          if (hasPointerEvents && targetCanvas.releasePointerCapture) {
            targetCanvas.releasePointerCapture(pointerId);
          }
        } catch {}
      }
    };

    const clearMainInteractionState = () => {
      setState({
        spectrumDragging: false,
        spectrumHoverActive: false,
        spectrumHoverFrame: -1,
        spectrumHoverX: null,
        spectrumHoverY: null,
      });
    };

    const clearOverviewInteractionState = () => {
      setState({ overviewDragging: false });
      requestOverviewOverlayRedraw();
    };

    const clearAllInteractionState = () => {
      clearPinchGesture();
      releaseCapturedPointers();
      activePointers.clear();
      clearMainInteractionState();
      clearOverviewInteractionState();
      requestSpectrumRedraw(true);
    };

    const buildPinchGesture = (points: Array<{ clientX: number; clientY: number }>) => {
      const state = getState();
      if (!Array.isArray(points) || points.length < 2 || state.spectrumW <= 0) return null;
      const [p1, p2] = points;
      if (!p1 || !p2) return null;
      const distance = Math.hypot(p2.clientX - p1.clientX, p2.clientY - p1.clientY);
      if (!Number.isFinite(distance) || distance < 8) return null;
      const centerX = (p1.clientX + p2.clientX) * 0.5;
      const { rect, axisX, plotW } = getMainPlotMetrics(targetCanvas);
      const relX = clampNumber(centerX - rect.left - axisX, 0, plotW - 1);
      const anchorFrac = relX / plotW;
      const viewW = getMainViewFrameCount(state);
      const anchorFrame = state.spectrumOffset + anchorFrac * viewW;
      return {
        startDistance: distance,
        startZoom: state.spectrumZoom,
        anchorFrac,
        anchorFrame,
      };
    };

    const applySpectrumZoomFromAnchor = ({
      nextZoom,
      anchorFrame,
      anchorFrac,
    }: {
      nextZoom: number;
      anchorFrame: number;
      anchorFrac: number;
    }) => {
      const state = getState();
      if (state.spectrumW <= 0) return false;
      const clampedZoom = clampSpectrumZoom(nextZoom, state, getDisplaySamplingConfig());
      const safeFrac = clampNumber(Number(anchorFrac) || 0, 0, 1);
      const safeAnchorFrame = clampNumber(Number(anchorFrame) || 0, 0, state.spectrumW);
      const prevZoom = state.spectrumZoom;
      const prevOffset = state.spectrumOffset;
      const newViewW = getMainViewFrameCount({
        spectrumW: state.spectrumW,
        spectrumZoom: clampedZoom,
        spectrumDuration: state.spectrumDuration,
      });
      const maxOffset = Math.max(0, state.spectrumW - newViewW);
      const nextOffset = Math.max(
        0,
        Math.min(maxOffset, Math.round(safeAnchorFrame - safeFrac * newViewW)),
      );
      if (clampedZoom === prevZoom && nextOffset === prevOffset) return false;
      setState({
        spectrumZoom: clampedZoom,
        spectrumOffset: nextOffset,
      });
      requestSpectrumRedraw(true);
      return true;
    };

    const startPinch = (
      points: Array<{ clientX: number; clientY: number }>,
      event: MouseEvent | PointerEvent | TouchEvent | null = null,
    ) => {
      const nextPinchGesture = buildPinchGesture(points);
      if (!nextPinchGesture) return false;
      markAutoPanSuppressed();
      preventDefaultIfPossible(event);
      pinchGesture = nextPinchGesture;
      setState({
        spectrumDragging: false,
        spectrumHoverActive: false,
        spectrumHoverFrame: -1,
        spectrumHoverX: null,
        spectrumHoverY: null,
      });
      requestSpectrumRedraw(true);
      return true;
    };

    const updatePinch = (
      points: Array<{ clientX: number; clientY: number }>,
      event: MouseEvent | PointerEvent | TouchEvent | null = null,
    ) => {
      if (!pinchGesture || !Array.isArray(points) || points.length < 2) return false;
      const [p1, p2] = points;
      if (!p1 || !p2) return false;
      const distance = Math.hypot(p2.clientX - p1.clientX, p2.clientY - p1.clientY);
      if (!Number.isFinite(distance) || distance < 8) return false;
      preventDefaultIfPossible(event);
      const scale = distance / Math.max(1, pinchGesture.startDistance);
      return applySpectrumZoomFromAnchor({
        nextZoom: pinchGesture.startZoom * scale,
        anchorFrame: pinchGesture.anchorFrame,
        anchorFrac: pinchGesture.anchorFrac,
      });
    };

    const handleDrag = (e: MouseEvent | PointerEvent | TouchEvent | null) => {
      const state = getState();
      if (state.spectrumW <= 0) return;
      markAutoPanSuppressed();
      const point = getClientPointFromEvent(e);
      if (!point) return;
      const { rect, axisX, plotW } = getMainPlotMetrics(targetCanvas);
      const relX = clampNumber(point.clientX - rect.left - axisX, 0, plotW - 1);
      const viewW = getMainViewFrameCount(state);
      const frameFloat = state.spectrumOffset + (relX / plotW) * viewW;
      const t = Math.max(0, Math.min(state.spectrumW - 1, Math.floor(frameFloat)));
      if (state.spectrumDuration > 0) {
        void seekAudioTime((t / Math.max(1, state.spectrumW - 1)) * state.spectrumDuration);
      }
    };

    const markHoverFromEvent = (e: MouseEvent | PointerEvent | TouchEvent | null) => {
      const state = getState();
      if (state.spectrumW <= 0) return;
      const point = getClientPointFromEvent(e);
      if (!point) return;
      const { rect, axisX, plotW, plotH } = getMainPlotMetrics(targetCanvas);
      const x = clampNumber(point.clientX - rect.left - axisX, 0, plotW - 1);
      const y = clampNumber(point.clientY - rect.top, 0, plotH - 1);
      const viewW = getMainViewFrameCount(state);
      const frame = Math.max(
        0,
        Math.min(
          Math.max(0, state.spectrumW - 1),
          Math.floor(state.spectrumOffset + (x / plotW) * viewW),
        ),
      );
      setState({
        spectrumHoverActive: true,
        spectrumHoverFrame: frame,
        spectrumHoverX: axisX + x,
        spectrumHoverY: y,
      });
      requestSpectrumRedraw(false);
    };

    const startInteraction = (e: MouseEvent | PointerEvent | TouchEvent | any) => {
      if (
        hasPointerEvents &&
        e &&
        e.pointerId !== undefined &&
        Number.isFinite(e.clientX) &&
        Number.isFinite(e.clientY)
      ) {
        activePointers.set(e.pointerId, {
          clientX: e.clientX,
          clientY: e.clientY,
        });
        if (activePointers.size >= 2) {
          startPinch(getActivePointerPoints(), e);
          try {
            if (targetCanvas.setPointerCapture && e.pointerId !== undefined) {
              targetCanvas.setPointerCapture(e.pointerId);
            }
          } catch {}
          return;
        }
      } else if (!hasPointerEvents) {
        const touchPoints = getTouchPoints(e);
        if (touchPoints.length >= 2) {
          startPinch(touchPoints.slice(0, 2), e);
          return;
        }
      }
      preventDefaultIfPossible(e);
      setState({
        spectrumHoverActive: true,
        spectrumDragging: true,
      });
      markAutoPanSuppressed();
      markHoverFromEvent(e);
      try {
        if (hasPointerEvents && targetCanvas.setPointerCapture && e.pointerId !== undefined) {
          targetCanvas.setPointerCapture(e.pointerId);
        }
      } catch {}
    };

    const moveInteraction = (e: MouseEvent | PointerEvent | TouchEvent | any) => {
      const state = getState();
      if (
        hasPointerEvents &&
        e &&
        e.pointerId !== undefined &&
        Number.isFinite(e.clientX) &&
        Number.isFinite(e.clientY)
      ) {
        activePointers.set(e.pointerId, {
          clientX: e.clientX,
          clientY: e.clientY,
        });
        if (pinchGesture || activePointers.size >= 2) {
          if (!pinchGesture) startPinch(getActivePointerPoints(), e);
          updatePinch(getActivePointerPoints(), e);
          return;
        }
      } else if (!hasPointerEvents) {
        const touchPoints = getTouchPoints(e);
        if (pinchGesture || touchPoints.length >= 2) {
          if (!pinchGesture) startPinch(touchPoints.slice(0, 2), e);
          updatePinch(touchPoints.slice(0, 2), e);
          return;
        }
      }
      setState({ spectrumHoverActive: true });
      if (state.spectrumDragging) {
        markAutoPanSuppressed();
        preventDefaultIfPossible(e);
        handleDrag(e);
      }
      markHoverFromEvent(e);
    };

    const stopInteraction = (
      e: MouseEvent | PointerEvent | TouchEvent | any,
      { commit = true } = {},
    ) => {
      preventDefaultIfPossible(e);
      markAutoPanSuppressed();
      if (hasPointerEvents && e && e.pointerId !== undefined) {
        activePointers.delete(e.pointerId);
      }
      try {
        if (hasPointerEvents && targetCanvas.releasePointerCapture && e && e.pointerId !== undefined) {
          targetCanvas.releasePointerCapture(e.pointerId);
        }
      } catch {}
      const state = getState();
      if (pinchGesture) {
        if (hasPointerEvents && activePointers.size >= 2) {
          startPinch(getActivePointerPoints(), e);
          return;
        }
        if (!hasPointerEvents) {
          const touchPoints = getTouchPoints(e);
          if (touchPoints.length >= 2) {
            startPinch(touchPoints.slice(0, 2), e);
            return;
          }
        }
        clearPinchGesture();
        clearMainInteractionState();
        requestSpectrumRedraw(true);
        return;
      }
      if (state.spectrumDragging) {
        if (commit) handleDrag(e);
        clearMainInteractionState();
        requestSpectrumRedraw(true);
      }
    };

    const leaveSpectrumHover = () => {
      const state = getState();
      if (pinchGesture || state.spectrumDragging) return;
      setState({
        spectrumHoverActive: false,
        spectrumHoverFrame: -1,
        spectrumHoverX: null,
        spectrumHoverY: null,
      });
      requestSpectrumRedraw(true);
    };

    targetCanvas.onpointerdown = null;
    targetCanvas.onpointermove = null;
    targetCanvas.onpointerup = null;
    targetCanvas.onpointercancel = null;
    targetCanvas.onpointerleave = null;
    targetCanvas.onmousedown = null;
    targetCanvas.onmousemove = null;
    targetCanvas.onmouseup = null;
    targetCanvas.onmouseleave = null;
    targetCanvas.ontouchstart = null;
    targetCanvas.ontouchmove = null;
    targetCanvas.ontouchend = null;
    targetCanvas.ontouchcancel = null;

    if (hasPointerEvents) {
      targetCanvas.onpointerdown = startInteraction;
      targetCanvas.onpointermove = moveInteraction;
      targetCanvas.onpointerup = (e) => stopInteraction(e, { commit: true });
      targetCanvas.onpointercancel = (e) => {
        stopInteraction(e, { commit: false });
        clearMainInteractionState();
        requestSpectrumRedraw(true);
      };
      targetCanvas.onpointerleave = leaveSpectrumHover;
    } else {
      targetCanvas.onmousedown = startInteraction;
      targetCanvas.onmousemove = moveInteraction;
      targetCanvas.onmouseup = (e) => stopInteraction(e, { commit: true });
      targetCanvas.onmouseleave = () => {
        clearMainInteractionState();
        requestSpectrumRedraw(true);
      };
      targetCanvas.ontouchstart = startInteraction;
      targetCanvas.ontouchmove = moveInteraction;
      targetCanvas.ontouchend = (e) => stopInteraction(e, { commit: true });
      targetCanvas.ontouchcancel = () => {
        stopInteraction(null, { commit: false });
        clearMainInteractionState();
        requestSpectrumRedraw(true);
      };
    }

    const handleWindowBlur = () => {
      clearAllInteractionState();
    };
    const handleVisibilityChange = () => {
      const documentRef = windowRef?.document ?? null;
      if (documentRef?.visibilityState === "hidden") {
        clearAllInteractionState();
      }
    };
    windowRef?.addEventListener("blur", handleWindowBlur);
    windowRef?.document?.addEventListener("visibilitychange", handleVisibilityChange);
    cleanups.push(() => {
      windowRef?.removeEventListener("blur", handleWindowBlur);
      windowRef?.document?.removeEventListener("visibilitychange", handleVisibilityChange);
    });

    const wheelListener = (e: WheelEvent) => {
      e.preventDefault();
      const state = getState();
      if (state.spectrumW <= 0) return;
      markAutoPanSuppressed();
      const prevZoom = state.spectrumZoom;
      const nextZoom = e.deltaY < 0 ? prevZoom * 1.2 : prevZoom / 1.2;
      const { rect, axisX, plotW } = getMainPlotMetrics(targetCanvas);
      const relX = clampNumber(e.clientX - rect.left - axisX, 0, plotW - 1);
      const frac = relX / plotW;
      const viewW = getMainViewFrameCount(state);
      const anchorFrame = state.spectrumOffset + frac * viewW;
      applySpectrumZoomFromAnchor({
        nextZoom,
        anchorFrame,
        anchorFrac: frac,
      });
    };
    targetCanvas.addEventListener("wheel", wheelListener, { passive: false });
    cleanups.push(() => targetCanvas.removeEventListener("wheel", wheelListener));
  }

  function bindOverviewCanvas(): void {
    const canvas = getOverviewCanvas();
    if (!canvas) return;
    const targetCanvas: HTMLCanvasElement = canvas;
    targetCanvas.style.touchAction = "none";
    targetCanvas.style.cursor = "default";
    const windowWithPointer = windowRef as (Window & { PointerEvent?: typeof PointerEvent }) | null;
    const hasPointerEvents = !!windowWithPointer && typeof windowWithPointer.PointerEvent === "function";
    let dragging = false;
    let dragType: "left" | "right" | "move" | "jump" | null = null;
    let dragStartX = 0;
    let dragStartOffset = 0;
    let dragStartViewW = 0;
    const EDGE_HIT_PX = 8;

    const updateOverviewCursor = (point: { clientX: number; clientY: number } | null = null) => {
      if (dragging) {
        targetCanvas.style.cursor =
          dragType === "left" || dragType === "right"
            ? "ew-resize"
            : "grabbing";
        return;
      }
      const state = getState();
      if (state.spectrumW <= 0 || !point) {
        targetCanvas.style.cursor = "default";
        return;
      }
      const rect = targetCanvas.getBoundingClientRect();
      if (!rect.width) {
        targetCanvas.style.cursor = "default";
        return;
      }
      const x = point.clientX - rect.left;
      const pxPerFrame = rect.width / Math.max(1, state.spectrumW);
      const viewW = getMainViewFrameCount(state);
      const x0 = state.spectrumOffset * pxPerFrame;
      const x1 = (state.spectrumOffset + viewW) * pxPerFrame;
      if (Math.abs(x - x0) < EDGE_HIT_PX || Math.abs(x - x1) < EDGE_HIT_PX) {
        targetCanvas.style.cursor = "ew-resize";
      } else if (x > x0 && x < x1) {
        targetCanvas.style.cursor = "grab";
      } else {
        targetCanvas.style.cursor = "default";
      }
    };

    const handleOverviewDrag = (e: MouseEvent | PointerEvent | TouchEvent | null) => {
      const state = getState();
      const overviewW = Math.max(0, state.spectrumW);
      if (overviewW <= 0) return;
      markAutoPanSuppressed();
      const point = getClientPointFromEvent(e);
      if (!point) return;
      const rect = targetCanvas.getBoundingClientRect();
      const x = point.clientX - rect.left;
      const totalW = overviewW;
      const pxPerFrame = rect.width / totalW;
      const viewW = getMainViewFrameCount(state);
      const prevOffset = state.spectrumOffset;
      const prevZoom = state.spectrumZoom;

      if (dragType === "left") {
        let newOffset = Math.round(x / pxPerFrame);
        let newViewW = dragStartViewW + (dragStartOffset - newOffset);
        if (newViewW < 10) {
          newViewW = 10;
          newOffset = dragStartOffset + dragStartViewW - 10;
        }
        if (newOffset < 0) {
          newOffset = 0;
          newViewW = dragStartViewW + dragStartOffset;
        }
        if (newOffset + newViewW > state.spectrumW) newViewW = state.spectrumW - newOffset;
        setState({
          spectrumOffset: newOffset,
          spectrumZoom: clampSpectrumZoom(state.spectrumW / newViewW, state, getDisplaySamplingConfig()),
        });
      } else if (dragType === "right") {
        let newViewW = Math.round((x - dragStartOffset * pxPerFrame) / pxPerFrame);
        if (newViewW < 10) newViewW = 10;
        if (dragStartOffset + newViewW > state.spectrumW) newViewW = state.spectrumW - dragStartOffset;
        setState({
          spectrumZoom: clampSpectrumZoom(state.spectrumW / newViewW, state, getDisplaySamplingConfig()),
        });
      } else if (dragType === "move") {
        const delta = Math.round((x - dragStartX) / pxPerFrame);
        let newOffset = dragStartOffset + delta;
        const maxOffset = state.spectrumW - dragStartViewW;
        if (newOffset < 0) newOffset = 0;
        if (newOffset > maxOffset) newOffset = maxOffset;
        setState({ spectrumOffset: newOffset });
      } else if (dragType === "jump") {
        const frame = Math.max(0, Math.min(totalW - 1, Math.floor(x / pxPerFrame)));
        setState({
          spectrumOffset: Math.max(0, Math.min(state.spectrumW - viewW, frame - Math.floor(viewW / 2))),
        });
      }

      const nextState = getState();
      if (nextState.spectrumOffset === prevOffset && nextState.spectrumZoom === prevZoom) return;
      requestOverviewOverlayRedraw();
      requestSpectrumRedraw(true);
      updateOverviewCursor(point);
    };

    const startOverviewDrag = (e: MouseEvent | PointerEvent | TouchEvent | null) => {
      const state = getState();
      const overviewW = Math.max(0, state.spectrumW);
      if (overviewW <= 0) return;
      const point = getClientPointFromEvent(e);
      if (!point) return;
      preventDefaultIfPossible(e);
      markAutoPanSuppressed();
      setState({ overviewDragging: true });
      const rect = targetCanvas.getBoundingClientRect();
      const x = point.clientX - rect.left;
      const totalW = overviewW;
      const pxPerFrame = rect.width / totalW;
      const viewW = getMainViewFrameCount(state);
      const offset = state.spectrumOffset;
      const x0 = offset * pxPerFrame;
      const x1 = (offset + viewW) * pxPerFrame;
      if (Math.abs(x - x0) < 8) dragType = "left";
      else if (Math.abs(x - x1) < 8) dragType = "right";
      else if (x > x0 && x < x1) dragType = "move";
      else dragType = "jump";
      dragging = true;
      dragStartX = x;
      dragStartOffset = offset;
      dragStartViewW = viewW;
      try {
        if (
          hasPointerEvents &&
          targetCanvas.setPointerCapture &&
          (e as PointerEvent | null) &&
          (e as PointerEvent).pointerId !== undefined
        ) {
          targetCanvas.setPointerCapture((e as PointerEvent).pointerId);
        }
      } catch {}
      handleOverviewDrag(e);
      updateOverviewCursor(point);
    };

    const moveOverviewDrag = (e: MouseEvent | PointerEvent | TouchEvent | null) => {
      if (dragging) {
        handleOverviewDrag(e);
        return;
      }
      updateOverviewCursor(getClientPointFromEvent(e));
    };

    const stopOverviewDrag = (
      e: MouseEvent | PointerEvent | TouchEvent | null,
      { commit = true } = {},
    ) => {
      preventDefaultIfPossible(e);
      if (dragging) {
        if (commit) handleOverviewDrag(e);
        dragging = false;
        dragType = null;
      }
      try {
        const pointerEvent = e as PointerEvent | null;
        if (
          hasPointerEvents &&
          targetCanvas.releasePointerCapture &&
          pointerEvent &&
          pointerEvent.pointerId !== undefined
        ) {
          targetCanvas.releasePointerCapture(pointerEvent.pointerId);
        }
      } catch {}
      setState({ overviewDragging: false });
      requestOverviewOverlayRedraw();
      updateOverviewCursor(getClientPointFromEvent(e));
    };

    targetCanvas.onpointerdown = null;
    targetCanvas.onpointermove = null;
    targetCanvas.onpointerup = null;
    targetCanvas.onpointercancel = null;
    targetCanvas.onpointerleave = null;
    targetCanvas.onmousedown = null;
    targetCanvas.onmousemove = null;
    targetCanvas.onmouseup = null;
    targetCanvas.onmouseleave = null;
    targetCanvas.ontouchstart = null;
    targetCanvas.ontouchmove = null;
    targetCanvas.ontouchend = null;
    targetCanvas.ontouchcancel = null;

    if (hasPointerEvents) {
      targetCanvas.onpointerdown = startOverviewDrag;
      targetCanvas.onpointermove = moveOverviewDrag;
      targetCanvas.onpointerup = (e) => {
        markAutoPanSuppressed(performance.now(), 1000);
        stopOverviewDrag(e, { commit: true });
      };
      targetCanvas.onpointercancel = (e) => {
        markAutoPanSuppressed(performance.now(), 1000);
        stopOverviewDrag(e, { commit: false });
      };
      targetCanvas.onpointerleave = () => {
        if (!dragging) {
          updateOverviewCursor(null);
        }
      };
      return;
    }

    targetCanvas.onmousedown = startOverviewDrag;
    targetCanvas.onmousemove = moveOverviewDrag;
    targetCanvas.onmouseup = (e) => {
      markAutoPanSuppressed(performance.now(), 1000);
      stopOverviewDrag(e, { commit: true });
    };
    targetCanvas.onmouseleave = (e) => {
      markAutoPanSuppressed(performance.now(), 1000);
      stopOverviewDrag(e, { commit: false });
    };
    targetCanvas.ontouchstart = startOverviewDrag;
    targetCanvas.ontouchmove = moveOverviewDrag;
    targetCanvas.ontouchend = (e) => {
      markAutoPanSuppressed(performance.now(), 1000);
      stopOverviewDrag(e, { commit: true });
    };
    targetCanvas.ontouchcancel = (e) => {
      markAutoPanSuppressed(performance.now(), 1000);
      stopOverviewDrag(e, { commit: false });
    };
  }

  function bind(): void {
    destroy();
    bindMainCanvas();
    bindOverviewCanvas();
  }

  function destroy(): void {
    for (const cleanup of cleanups.splice(0, cleanups.length)) {
      try {
        cleanup();
      } catch {}
    }
  }

  return {
    bind,
    destroy,
  };
}
