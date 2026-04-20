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

export function getMainViewFrameCount(state: Pick<SpectrumInteractionState, "spectrumW" | "spectrumZoom" | "spectrumDuration">): number {
  let viewW = Math.floor(Math.max(0, Number(state.spectrumW) || 0) / Math.max(1, Number(state.spectrumZoom) || 1));
  if (viewW > state.spectrumW) viewW = state.spectrumW;
  if (viewW < 1) viewW = 1;
  return Math.max(1, Math.floor(viewW));
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
  const safeAxisX = Math.max(0, Number(axisX) || 0);
  const safeTimelineHeight = Math.max(0, Number(timelineHeight) || 0);
  return {
    rect,
    axisX: safeAxisX,
    timelineHeight: safeTimelineHeight,
    plotW: Math.max(1, Math.floor((rect.width || canvas.clientWidth || 1) - safeAxisX)),
    plotH: Math.max(1, Math.floor((rect.height || canvas.clientHeight || 1) - safeTimelineHeight)),
  };
}
