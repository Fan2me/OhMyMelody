import type { SpectrumInteractionState, SpectrumUiState } from "./spectrum-state.js";
import type { SpectrumOverlayCanvasRef } from "./spectrum-main-overlay.js";
import { getMainViewFrameCount } from "./spectrum-layout.js";

export interface SpectrumOverviewOverlayRendererDeps {
  getCanvas: () => SpectrumOverlayCanvasRef | null;
  getState: () => SpectrumUiState | null;
  getInteractionState: () => SpectrumInteractionState | null;
}

export interface SpectrumOverviewOverlayRenderer {
  drawOverviewOverlay: () => void;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function niceTickStep(rawStep: number): number {
  const safe = Math.max(0.0001, rawStep);
  const magnitude = Math.pow(10, Math.floor(Math.log10(safe)));
  const ratio = safe / magnitude;
  const rounded =
    ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 5 ? 5 : 10;
  return rounded * magnitude;
}

function formatTimeLabel(seconds: number, fractionDigits = 2): string {
  const safe = Math.max(0, Number(seconds) || 0);
  return `${safe.toFixed(fractionDigits)}s`;
}

function drawPlayhead(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  currentTime: number,
  duration: number,
  color = "rgba(255, 78, 78, 0.95)",
): void {
  if (!Number.isFinite(currentTime) || !Number.isFinite(duration) || duration <= 0) {
    return;
  }
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.floor((canvas.width || canvas.clientWidth || 1) / dpr));
  const height = Math.max(1, Math.floor((canvas.height || canvas.clientHeight || 1) / dpr));
  const x = Math.round(clampNumber(currentTime / duration, 0, 1) * Math.max(0, width - 1)) + 0.5;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, height);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawTimeAxis(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  duration: number,
): void {
  if (!Number.isFinite(duration) || duration <= 0) {
    return;
  }
  const axisH = Math.min(18, Math.max(14, Math.round(height * 0.12)));
  const axisY = Math.max(0, height - axisH);
  ctx.fillStyle = "rgba(0, 0, 0, 0.24)";
  ctx.fillRect(0, axisY, width, axisH);
  ctx.strokeStyle = "rgba(255,255,255,0.14)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, axisY + 0.5);
  ctx.lineTo(width, axisY + 0.5);
  ctx.stroke();

  const targetTicks = Math.max(8, Math.floor(width / 45));
  const tickStep = niceTickStep(duration / targetTicks);
  const start = 0;
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.66)";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let t = start; t <= duration + 1e-6; t += tickStep) {
    const x = Math.round((t / Math.max(1, duration)) * Math.max(0, width - 1)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, axisY);
    ctx.lineTo(x, axisY + 5);
    ctx.stroke();
    ctx.fillText(formatTimeLabel(t, 2), x, axisY + 5);
  }
}

export function createSpectrumOverviewOverlayRenderer(
  deps: SpectrumOverviewOverlayRendererDeps,
): SpectrumOverviewOverlayRenderer {
  const {
    getCanvas,
    getState,
    getInteractionState,
  } = deps;

  function drawOverviewOverlay(): void {
    const refs = getCanvas();
    if (!refs || !refs.canvas || !refs.ctx) return;
    const state = getState();
    const interaction = getInteractionState();
    if (!state || !interaction) return;

    const { canvas, ctx } = refs;
    const {
      spectrumW,
      spectrumDuration,
      spectrumOffset,
      spectrumZoom,
    } = interaction;
    const audioElement = state.audioElement;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const overviewUnits = Math.max(1, spectrumW);
    if (!overviewUnits) return;

    const viewW = getMainViewFrameCount({
      spectrumW: overviewUnits,
      spectrumZoom,
      spectrumDuration,
    });
    const offset = Math.max(0, Math.min(overviewUnits - viewW, spectrumOffset));
    const boxX = (offset / Math.max(1, overviewUnits)) * w;
    const boxW = (viewW / Math.max(1, overviewUnits)) * w;
    ctx.fillStyle = "rgba(0,0,0,0.18)";
    ctx.fillRect(boxX, 0, boxW, h);
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(boxX + 0.5, 0.5, Math.max(1, boxW - 1), Math.max(1, h - 1));

    if (audioElement) {
      drawPlayhead(
        ctx,
        canvas,
        audioElement.currentTime || 0,
        audioElement.duration || 0,
        "rgba(255, 78, 78, 0.85)",
      );
    } else if (spectrumDuration > 0) {
      drawPlayhead(
        ctx,
        canvas,
        0,
        spectrumDuration,
        "rgba(120, 200, 255, 0.65)",
      );
    }

    drawTimeAxis(ctx, w, h, spectrumDuration || (audioElement?.duration || 0));
  }

  return {
    drawOverviewOverlay,
  };
}
