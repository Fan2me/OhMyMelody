import { DEFAULT_DISPLAY_SAMPLING_CONFIG, getDisplayZoomBounds } from "./display-sampling.js";
import type { DisplaySamplingConfig } from "./display-sampling.js";
import type { SpectrumInteractionState } from "./spectrum-state.js";

export type PlotMetrics = {
  rect: DOMRect;
  axisX: number;
  timelineHeight: number;
  plotW: number;
  plotH: number;
};

export type CanvasLogicalSize = {
  width: number;
  height: number;
  dpr: number;
};

type CanvasSizeSource = Pick<
  HTMLCanvasElement,
  "width" | "height" | "clientWidth" | "clientHeight" | "getBoundingClientRect"
>;

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function getCanvasLogicalSize(canvas: CanvasSizeSource): CanvasLogicalSize {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Number(rect.width) || Number(canvas.clientWidth) || Number(canvas.width) || 1);
  const height = Math.max(1, Number(rect.height) || Number(canvas.clientHeight) || Number(canvas.height) || 1);
  const dpr = Math.max(
    1,
    Number(canvas.width) > 0 ? Number(canvas.width) / width : 1,
    Number(canvas.height) > 0 ? Number(canvas.height) / height : 1,
  );
  return { width, height, dpr };
}

export function spectrumFrameToTime(
  frame: number,
  totalFrames: number,
  durationSec: number,
): number {
  const safeTotalFrames = Math.max(1, Math.floor(Number(totalFrames) || 0));
  const safeDurationSec = Math.max(0, Number(durationSec) || 0);
  if (safeTotalFrames <= 1 || safeDurationSec <= 0) return 0;
  const safeFrame = clampNumber(Number(frame) || 0, 0, safeTotalFrames - 1);
  return (safeFrame / (safeTotalFrames - 1)) * safeDurationSec;
}

export function spectrumTimeToFrame(
  timeSec: number,
  totalFrames: number,
  durationSec: number,
): number {
  const safeTotalFrames = Math.max(1, Math.floor(Number(totalFrames) || 0));
  const safeDurationSec = Math.max(0, Number(durationSec) || 0);
  if (safeTotalFrames <= 1 || safeDurationSec <= 0) return 0;
  const safeTimeSec = clampNumber(Number(timeSec) || 0, 0, safeDurationSec);
  return (safeTimeSec / safeDurationSec) * (safeTotalFrames - 1);
}

export function getMainViewFrameCount(state: Pick<SpectrumInteractionState, "spectrumW" | "spectrumZoom" | "spectrumDuration">): number {
  const safeTotalFrames = Math.max(1, Math.floor(Number(state.spectrumW) || 0));
  const safeZoom = Math.max(1, Number(state.spectrumZoom) || 1);
  const baseViewFrames = Math.max(1, Math.floor(safeTotalFrames / safeZoom));
  return Math.max(1, Math.min(safeTotalFrames, baseViewFrames));
}

export function clampSpectrumZoom(
  nextZoom: number,
  state: Pick<SpectrumInteractionState, "spectrumW" | "spectrumDuration">,
  displaySampling: Pick<DisplaySamplingConfig, "maxUnitsPerSecond"> = DEFAULT_DISPLAY_SAMPLING_CONFIG,
): number {
  const { maxZoom } = getDisplayZoomBounds({
    totalFrames: state.spectrumW || 1,
    durationSec: state.spectrumDuration || 0,
    maxUnitsPerSecond: displaySampling.maxUnitsPerSecond,
  });
  const safeZoom = Number.isFinite(nextZoom) ? nextZoom : 1;
  return Math.max(1, Math.min(Math.max(1, maxZoom), safeZoom));
}

export function getPlotMetrics(
  canvas: HTMLCanvasElement,
  axisX: number,
  timelineHeight: number,
): PlotMetrics {
  const rect = canvas.getBoundingClientRect();
  const logicalSize = getCanvasLogicalSize(canvas);
  const safeAxisX = Math.max(0, Number(axisX) || 0);
  const safeTimelineHeight = Math.max(0, Number(timelineHeight) || 0);
  return {
    rect,
    axisX: safeAxisX,
    timelineHeight: safeTimelineHeight,
    plotW: Math.max(1, logicalSize.width - safeAxisX),
    plotH: Math.max(1, logicalSize.height - safeTimelineHeight),
  };
}
