import type { SpectrumInteractionState, SpectrumUiState } from "./spectrum-state.js";
import { getMainViewFrameCount } from "./spectrum-layout.js";
import {
  getDisplayStrideFramesForZoom,
  pickRepresentativeIndex,
} from "./display-sampling.js";

export interface SpectrumOverlayCanvasRef {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
}

export interface SpectrumMainOverlayRendererDeps {
  getCanvas: () => SpectrumOverlayCanvasRef | null;
  getState: () => SpectrumUiState | null;
  getInteractionState: () => SpectrumInteractionState | null;
  getAudioElement: () => HTMLAudioElement | null;
  getPredictionFrames: () => ArrayLike<number> | readonly number[];
  getPredictionConfidence?: () => ArrayLike<number> | readonly number[] | null;
  getPredictionRevision?: () => number;
}

export interface SpectrumMainOverlayRenderer {
  drawOverlay: () => void;
}

const FRAME_SEC = 0.01;
const BINS_PER_SEMITONE = 5;
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const AXIS_X = 0;

type CachedPitchAxis = {
  key: string;
  canvas: HTMLCanvasElement;
};

type CachedPredictionSeries = {
  key: string;
  points: Array<{
    centerFrame: number;
    bin: number;
  }>;
};

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

function binToNoteName(bin: number): string {
  const b = Number(bin);
  if (!Number.isFinite(b) || b < 0) return "Rest";
  const semi = Math.max(0, Math.round(b / BINS_PER_SEMITONE));
  const octave = 1 + Math.floor(semi / 12);
  return `${NOTE_NAMES[semi % 12]}${octave}`;
}

function binToY(
  bin: number,
  height: number,
  displayStart: number,
  displayEnd: number,
): number | null {
  const safeBin = Number(bin);
  if (!Number.isFinite(safeBin)) return null;
  if (safeBin < displayStart || safeBin > displayEnd) return null;
  const displayBins = Math.max(1, displayEnd - displayStart + 1);
  return height - (safeBin - displayStart + 0.5) * (height / displayBins);
}

function drawPitchAxis(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  displayStart: number,
  displayEnd: number,
): void {
  const displayBins = Math.max(1, displayEnd - displayStart + 1);
  const plotH = Math.max(1, height);
  const pixelPerBin = plotH / displayBins;
  const semitoneStart = Math.ceil(displayStart / BINS_PER_SEMITONE);
  const semitoneEnd = Math.floor(displayEnd / BINS_PER_SEMITONE);

  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.20)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let s = semitoneStart; s <= semitoneEnd; s += 1) {
    const bin = s * BINS_PER_SEMITONE;
    if (bin < displayStart || bin > displayEnd) continue;
    const y = Math.round(plotH - (bin - displayStart + 0.5) * pixelPerBin) + 0.5;
    ctx.moveTo(AXIS_X, y);
    ctx.lineTo(width, y);
  }
  ctx.stroke();
  ctx.restore();

  ctx.font = "10px sans-serif";
  ctx.fillStyle = "#fff";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (let s = semitoneStart; s <= semitoneEnd; s += 1) {
    const bin = s * BINS_PER_SEMITONE;
    if (bin < displayStart || bin > displayEnd) continue;
    const y = Math.round(plotH - (bin - displayStart + 0.5) * pixelPerBin);
    ctx.fillText(binToNoteName(bin), AXIS_X + 4, y);
    ctx.strokeStyle = "rgba(255,255,255,0.20)";
    ctx.beginPath();
    ctx.moveTo(AXIS_X, y);
    ctx.lineTo(AXIS_X + 6, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255,255,255,0.20)";
  ctx.beginPath();
  ctx.moveTo(AXIS_X + 0.5, 0);
  ctx.lineTo(AXIS_X + 0.5, plotH);
  ctx.stroke();
}

function drawTimeAxis(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  axisX: number,
  startTimeSec: number,
  visibleTimeSec: number,
): void {
  if (!Number.isFinite(visibleTimeSec) || visibleTimeSec <= 0) {
    return;
  }
  const axisY = Math.max(0.5, height - 0.5);
  const plotW = Math.max(1, width - axisX);

  const targetTicks = Math.max(4, Math.floor(plotW / 80));
  const tickStep = niceTickStep(visibleTimeSec / targetTicks);
  const start = Math.max(0, Math.ceil(startTimeSec / tickStep) * tickStep);
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.72)";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  for (let t = start; t <= startTimeSec + visibleTimeSec + 1e-6; t += tickStep) {
    const x =
      axisX +
      Math.round(((t - startTimeSec) / Math.max(1, visibleTimeSec)) * Math.max(0, plotW - 1)) +
      0.5;
    ctx.fillText(formatTimeLabel(t, 2), x, axisY - 8);
  }
}

export function createSpectrumMainOverlayRenderer(
  deps: SpectrumMainOverlayRendererDeps,
): SpectrumMainOverlayRenderer {
  const {
    getCanvas,
    getState,
    getInteractionState,
    getAudioElement,
    getPredictionFrames,
    getPredictionConfidence = () => null,
    getPredictionRevision = () => 0,
  } = deps;
  let cachedPitchAxis: CachedPitchAxis | null = null;
  let cachedPredictionSeries: CachedPredictionSeries | null = null;

  function drawPitchAxisCached(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    displayStart: number,
    displayEnd: number,
  ): void {
    const safeWidth = Math.max(1, Math.round(width));
    const safeHeight = Math.max(1, Math.round(height));
    const key = `${safeWidth}x${safeHeight}:${displayStart}-${displayEnd}`;
    if (!cachedPitchAxis || cachedPitchAxis.key !== key) {
      const axisCanvas =
        cachedPitchAxis?.canvas ??
        (typeof document !== "undefined" ? document.createElement("canvas") : null);
      if (!axisCanvas) {
        drawPitchAxis(ctx, width, height, displayStart, displayEnd);
        return;
      }
      axisCanvas.width = safeWidth;
      axisCanvas.height = safeHeight;
      const axisCtx = axisCanvas.getContext("2d");
      if (!axisCtx) {
        drawPitchAxis(ctx, width, height, displayStart, displayEnd);
        return;
      }
      axisCtx.setTransform(1, 0, 0, 1, 0, 0);
      axisCtx.clearRect(0, 0, axisCanvas.width, axisCanvas.height);
      drawPitchAxis(axisCtx, width, height, displayStart, displayEnd);
      cachedPitchAxis = {
        key,
        canvas: axisCanvas,
      };
    }
    ctx.drawImage(cachedPitchAxis.canvas, 0, 0, width, height);
  }

  function buildPredictionSeriesKey(args: {
    totalFrames: number;
    stride: number;
    representativeMode: string;
    predictionRevision: number;
    predictionFrames: ArrayLike<number> | readonly number[];
    predictionConfidence: ArrayLike<number> | readonly number[] | null;
  }): string {
    const { totalFrames, stride, representativeMode, predictionRevision, predictionFrames, predictionConfidence } = args;
    const frameLength = Math.max(0, Math.floor(Number(predictionFrames.length) || 0));
    const confidenceLength = Math.max(0, Math.floor(Number(predictionConfidence?.length) || 0));
    const frameFirst = frameLength > 0 ? Number(predictionFrames[0]) : NaN;
    const frameMid = frameLength > 0 ? Number(predictionFrames[Math.floor(frameLength / 2)]) : NaN;
    const frameLast = frameLength > 0 ? Number(predictionFrames[frameLength - 1]) : NaN;
    const confFirst = confidenceLength > 0 ? Number(predictionConfidence?.[0]) : NaN;
    const confMid = confidenceLength > 0 ? Number(predictionConfidence?.[Math.floor(confidenceLength / 2)]) : NaN;
    const confLast = confidenceLength > 0 ? Number(predictionConfidence?.[confidenceLength - 1]) : NaN;
    return [
      `frames:${totalFrames}`,
      `stride:${stride}`,
      `rep:${representativeMode}`,
      `rev:${predictionRevision}`,
      `len:${frameLength}`,
      `conf:${confidenceLength}`,
      `sig:${frameFirst.toFixed(4)}|${frameMid.toFixed(4)}|${frameLast.toFixed(4)}`,
      `csig:${confFirst.toFixed(4)}|${confMid.toFixed(4)}|${confLast.toFixed(4)}`,
    ].join("|");
  }

  function getCachedPredictionSeries(args: {
    totalFrames: number;
    stride: number;
    representativeMode: string;
    predictionRevision: number;
    predictionFrames: ArrayLike<number> | readonly number[];
    predictionConfidence: ArrayLike<number> | readonly number[] | null;
  }): CachedPredictionSeries {
    const key = buildPredictionSeriesKey(args);
    if (cachedPredictionSeries && cachedPredictionSeries.key === key) {
      return cachedPredictionSeries;
    }
    const totalFrames = Math.max(1, Math.floor(Number(args.totalFrames) || 1));
    const stride = Math.max(1, Math.floor(Number(args.stride) || 1));
    const points: CachedPredictionSeries["points"] = [];
    const confidenceAt = (idx: number): number => {
      const value = args.predictionConfidence?.[idx];
      return Number.isFinite(value) ? Number(value) : NaN;
    };
    const validAt = (idx: number): boolean => {
      const value = args.predictionFrames[idx];
      return Number.isFinite(value) && Number(value) >= 0;
    };
    for (let regionStart = 0; regionStart < totalFrames; regionStart += stride) {
      const regionEnd = Math.min(totalFrames, regionStart + stride);
      const frameIndex = pickRepresentativeIndex({
        start: regionStart,
        end: regionEnd,
        mode: args.representativeMode,
        isValidAt: validAt,
        scoreAt: confidenceAt,
      });
      const rawBin = Number(args.predictionFrames[frameIndex]);
      points.push({
        centerFrame: (regionStart + regionEnd) * 0.5,
        bin: Number.isFinite(rawBin) && rawBin >= 0 ? Math.floor(rawBin) : NaN,
      });
    }
    cachedPredictionSeries = { key, points };
    return cachedPredictionSeries;
  }

  function drawOverlay(): void {
    const refs = getCanvas();
    if (!refs || !refs.canvas || !refs.ctx) return;
    const state = getState();
    const interaction = getInteractionState();
    if (!state || !interaction) return;

    const { canvas, ctx } = refs;
    const audioElement = getAudioElement();

    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(canvas.width / dpr));
    const h = Math.max(1, Math.round(canvas.height / dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const spectrumBinCount = Math.max(1, Math.floor(interaction.spectrumH || 1));
    const displayStart = Math.max(0, Math.min(spectrumBinCount - 1, Math.floor(state.pitchRange.minBin || 0)));
    const displayEnd = Math.max(
      displayStart,
      Math.min(spectrumBinCount - 1, Math.floor(state.pitchRange.maxBin || (spectrumBinCount - 1))),
    );
    const plotH = Math.max(1, h);
    const plotW = Math.max(1, w);
    drawPitchAxisCached(ctx, w, h, displayStart, displayEnd);

    const predictionFrames = getPredictionFrames();
    const predictionConfidence = getPredictionConfidence?.() || null;
    const fallbackDuration =
      state.audio ? state.audio.pcm.length / Math.max(1, state.audio.fs) : 0;
    const totalFrames = Math.max(
      1,
      interaction.spectrumW ||
        predictionFrames.length ||
        state.inference?.totalArgmax.length ||
        Math.round((interaction.spectrumDuration || fallbackDuration) / FRAME_SEC),
    );
    const viewW = getMainViewFrameCount({
      spectrumW: totalFrames,
      spectrumZoom: interaction.spectrumZoom,
      spectrumDuration: interaction.spectrumDuration,
    });
    const offset = clampNumber(
      interaction.spectrumOffset,
      0,
      Math.max(0, totalFrames - viewW),
    );
    const currentTime = audioElement?.currentTime || 0;
    const playheadFrame = Math.max(0, currentTime / FRAME_SEC);
    if (Number.isFinite(playheadFrame) && playheadFrame >= offset && playheadFrame <= offset + viewW) {
      const playheadX = totalFrames > 1
        ? Math.round(((playheadFrame - offset) / viewW) * plotW) + 0.5
        : plotW * 0.5;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, h);
      ctx.strokeStyle = audioElement
        ? "rgba(255, 78, 78, 0.95)"
        : "rgba(120, 200, 255, 0.75)";
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    if (predictionFrames.length) {
      const displayConfig = state.displaySampling;
      const stride = Math.max(
        1,
        getDisplayStrideFramesForZoom({
          zoom: interaction.spectrumZoom,
          minZoom: 1,
          maxZoom: 20,
          minUnitsPerSecond: displayConfig.minUnitsPerSecond,
          maxUnitsPerSecond: displayConfig.maxUnitsPerSecond,
          frameRateHz: Math.max(1, Math.round(1 / FRAME_SEC)),
        }),
      );
      const series = getCachedPredictionSeries({
        totalFrames,
        stride,
        representativeMode: displayConfig.representativeMode,
        predictionRevision: getPredictionRevision(),
        predictionFrames,
        predictionConfidence,
      });
      ctx.beginPath();
      let drawing = false;
      const visibleStart = offset;
      const visibleEnd = Math.min(totalFrames, offset + viewW);
      for (const point of series.points) {
        if (point.centerFrame < visibleStart || point.centerFrame >= visibleEnd) {
          continue;
        }
        if (!Number.isFinite(point.bin) || point.bin < 0) {
          drawing = false;
          continue;
        }
        const y = binToY(point.bin, plotH, displayStart, displayEnd);
        if (y === null) {
          drawing = false;
          continue;
        }
        const x =
          totalFrames > 1
            ? (((point.centerFrame - offset) / viewW) * plotW)
            : plotW * 0.5;
        if (!drawing) {
          ctx.moveTo(x, y);
          drawing = true;
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.strokeStyle = "rgba(255, 214, 102, 0.95)";
      ctx.lineWidth = 2;
      ctx.stroke();

      const frameIdx = Math.max(0, Math.min(totalFrames - 1, Math.floor(playheadFrame)));
      const rawBin = Number(predictionFrames[frameIdx]);
      if (Number.isFinite(rawBin) && rawBin >= 0) {
        const curBin = Math.floor(rawBin);
        const label = `${binToNoteName(curBin)} (bin=${curBin})`;
        const y = binToY(curBin, plotH, displayStart, displayEnd);
        if (y !== null) {
          const x = totalFrames > 1
            ? Math.round(((frameIdx - offset) / viewW) * plotW) + 0.5
            : plotW * 0.5;
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, Math.PI * 2);
          ctx.fillStyle = "#ffd166";
          ctx.fill();

          ctx.font = "bold 13px sans-serif";
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          const padding = 8;
          const labelH = 24;
          const labelW = Math.max(44, Math.ceil(ctx.measureText(label).width + padding * 2));
          let labelX = x + 10;
          let labelY = y - labelH - 8;
          if (labelX + labelW > w) labelX = x - labelW - 10;
          if (labelY < 2) labelY = y + 8;
          if (labelY + labelH > h) labelY = Math.max(2, h - labelH - 2);
          ctx.fillStyle = "rgba(0,0,0,0.62)";
          ctx.fillRect(labelX, labelY, labelW, labelH);
          ctx.fillStyle = "#ffd166";
          ctx.fillText(label, labelX + padding, labelY + labelH / 2);
        }
      }
    }

    if (
      interaction.spectrumHoverActive &&
      interaction.spectrumHoverX !== null &&
      interaction.spectrumHoverY !== null
    ) {
      const vx = Math.max(0, Math.min(w - 1, interaction.spectrumHoverX));
      const vy = Math.max(0, Math.min(h - 1, interaction.spectrumHoverY));
      ctx.beginPath();
      ctx.moveTo(vx + 0.5, 0);
      ctx.lineTo(vx + 0.5, h);
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, vy + 0.5);
      ctx.lineTo(vx, vy + 0.5);
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(vx, vy, 5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,200,0,1)";
      ctx.fill();

      const hoverFrame = interaction.spectrumHoverFrame;
      const hoverBinValue = Number.isFinite(hoverFrame) ? Number(predictionFrames[hoverFrame]) : NaN;
      if (Number.isFinite(hoverBinValue) && hoverBinValue >= 0) {
        const bin = Math.floor(hoverBinValue);
        const note = binToNoteName(bin);
        const hoverY = binToY(bin, plotH, displayStart, displayEnd);
        if (bin >= 0 && hoverY !== null) {
          ctx.font = "12px sans-serif";
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          const padding = 6;
          const text = `${note} (bin=${Math.floor(bin)})`;
          const metrics = ctx.measureText(text);
          const boxW = metrics.width + padding * 2;
          const boxH = 20;
          let boxX = vx + 8;
          let boxY = hoverY - boxH - 8;
          if (boxX + boxW > w) boxX = vx - boxW - 8;
          if (boxY < 0) boxY = hoverY + 8;
          ctx.fillStyle = "rgba(0,0,0,0.62)";
          ctx.fillRect(boxX, boxY, boxW, boxH);
          ctx.fillStyle = "#ffd166";
          ctx.fillText(text, boxX + padding, boxY + boxH / 2);
        }
      }
    }

    const visibleTimeSec = Math.max(1 / 100, viewW * FRAME_SEC);
    const startTimeSec = Math.max(0, offset * FRAME_SEC);
    drawTimeAxis(ctx, w, h, AXIS_X, startTimeSec, visibleTimeSec);

    void FRAME_SEC;
  }

  return {
    drawOverlay,
  };
}
