import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { pickRepresentativeIndex } from "./display-sampling.js";

export type HeatmapTimelineSlot = readonly CFPBatch[] | null;

export type HeatmapTimelineSegment = {
  slotIndex: number;
  batches: readonly CFPBatch[] | null;
  empty: boolean;
  frameCount: number;
  freqCount: number;
  min: number;
  max: number;
  batchFrameStarts: number[];
  batchFrameCounts: number[];
  batchFreqCounts: number[];
};

export type HeatmapTimeline = {
  segments: HeatmapTimelineSegment[];
  totalSlots: number;
  totalFrames: number;
  freqCount: number;
  min: number;
  max: number;
};

export type HeatmapTimelineViewport = {
  startSlot: number;
  endSlot: number;
};

export type HeatmapFrequencyViewport = {
  minBin: number;
  maxBin: number;
};

export type HeatmapOptimizationLevel = "baseline-rgba" | "u32" | "u32-region";

export type HeatmapBenchmarkEntry = {
  name: string;
  avgMs: number;
  sampleStrideFrames: number;
  optimizationLevel: HeatmapOptimizationLevel;
};

type BatchSummary = {
  frameCount: number;
  freqCount: number;
  min: number;
  max: number;
};

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

const HEATMAP_PALETTE_SIZE = 256;
const HEATMAP_MAX_DRAW_ROWS = 220;

const IS_LITTLE_ENDIAN = (() => {
  const u32 = new Uint32Array([0x0a0b0c0d]);
  const u8 = new Uint8Array(u32.buffer);
  return u8[0] === 0x0d;
})();

function rgbaToPackedU32(r: number, g: number, b: number, a: number): number {
  if (IS_LITTLE_ENDIAN) {
    return ((a & 0xff) << 24) | ((b & 0xff) << 16) | ((g & 0xff) << 8) | (r & 0xff);
  }
  return ((r & 0xff) << 24) | ((g & 0xff) << 16) | ((b & 0xff) << 8) | (a & 0xff);
}

function buildHeatmapPaletteRGBA(): Uint8Array {
  const palette = new Uint8Array(HEATMAP_PALETTE_SIZE * 4);
  for (let i = 0; i < HEATMAP_PALETTE_SIZE; i += 1) {
    const t = i / (HEATMAP_PALETTE_SIZE - 1);
    const warm = Math.pow(t, 0.9);
    const base = i * 4;
    palette[base] = Math.round(16 + 224 * warm);
    palette[base + 1] = Math.round(20 + 180 * t);
    palette[base + 2] = Math.round(28 + 90 * (1 - t));
    palette[base + 3] = 255;
  }
  return palette;
}

const HEATMAP_PALETTE_RGBA = buildHeatmapPaletteRGBA();

const HEATMAP_PALETTE_U32 = (() => {
  const out = new Uint32Array(HEATMAP_PALETTE_SIZE);
  for (let i = 0; i < HEATMAP_PALETTE_SIZE; i += 1) {
    const base = i * 4;
    out[i] = rgbaToPackedU32(
      HEATMAP_PALETTE_RGBA[base] ?? 0,
      HEATMAP_PALETTE_RGBA[base + 1] ?? 0,
      HEATMAP_PALETTE_RGBA[base + 2] ?? 0,
      HEATMAP_PALETTE_RGBA[base + 3] ?? 0,
    );
  }
  return out;
})();

const BATCH_SUMMARY_CACHE = new WeakMap<CFPBatch, BatchSummary>();

function summarizeBatch(batch: CFPBatch): BatchSummary {
  const cached = BATCH_SUMMARY_CACHE.get(batch);
  if (cached) {
    return cached;
  }
  const shape = batch.shape;
  const freqCount = Math.max(1, Math.floor(Number(shape?.[1]) || 1));
  const frameCount = Math.max(1, Math.floor(Number(shape?.[2]) || 1));
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < batch.data.length; i += 1) {
    const value = Number(batch.data[i]);
    if (!Number.isFinite(value)) {
      continue;
    }
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max)) max = 0;
  const summary = {
    frameCount,
    freqCount,
    min,
    max,
  };
  BATCH_SUMMARY_CACHE.set(batch, summary);
  return summary;
}

export function buildSpectrumTimeline(
  slots: readonly HeatmapTimelineSlot[] = [],
): HeatmapTimeline {
  const segments: HeatmapTimelineSegment[] = [];
  let freqCount = 0;
  let totalFrames = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (let slotIndex = 0; slotIndex < slots.length; slotIndex += 1) {
    const slot = slots[slotIndex] ?? null;
    if (!slot || !slot.length) {
      segments.push({
        slotIndex,
        batches: null,
        empty: true,
        frameCount: 0,
        freqCount: 0,
        min: 0,
        max: 0,
        batchFrameStarts: [],
        batchFrameCounts: [],
        batchFreqCounts: [],
      });
      continue;
    }

    const batchFrameStarts: number[] = [];
    const batchFrameCounts: number[] = [];
    const batchFreqCounts: number[] = [];
    let frameCount = 0;
    let slotFreqCount = 0;
    let slotMin = Number.POSITIVE_INFINITY;
    let slotMax = Number.NEGATIVE_INFINITY;

    for (const batch of slot) {
      if (!batch || !(batch.data instanceof Float32Array)) {
        continue;
      }
      const summary = summarizeBatch(batch);
      batchFrameStarts.push(frameCount);
      batchFrameCounts.push(summary.frameCount);
      batchFreqCounts.push(summary.freqCount);
      frameCount += summary.frameCount;
      slotFreqCount = Math.max(slotFreqCount, summary.freqCount);
      slotMin = Math.min(slotMin, summary.min);
      slotMax = Math.max(slotMax, summary.max);
    }

    if (!batchFrameStarts.length) {
      segments.push({
        slotIndex,
        batches: null,
        empty: true,
        frameCount: 0,
        freqCount: 0,
        min: 0,
        max: 0,
        batchFrameStarts: [],
        batchFrameCounts: [],
        batchFreqCounts: [],
      });
      continue;
    }

    const segmentMin = Number.isFinite(slotMin) ? slotMin : 0;
    const segmentMax = Number.isFinite(slotMax) ? slotMax : 0;
    segments.push({
      slotIndex,
      batches: slot,
      empty: false,
      frameCount,
      freqCount: slotFreqCount,
      min: segmentMin,
      max: segmentMax,
      batchFrameStarts,
      batchFrameCounts,
      batchFreqCounts,
    });
    freqCount = Math.max(freqCount, slotFreqCount);
    totalFrames += frameCount;
    min = Math.min(min, segmentMin);
    max = Math.max(max, segmentMax);
  }

  return {
    segments,
    totalSlots: slots.length,
    totalFrames,
    freqCount,
    min: Number.isFinite(min) ? min : 0,
    max: Number.isFinite(max) ? max : 0,
  };
}

function fillEmptySlot(
  data32: Uint32Array,
  width: number,
  height: number,
  xStart: number,
  xEnd: number,
): void {
  const left = Math.max(0, Math.floor(xStart));
  const right = Math.max(left + 1, Math.ceil(xEnd));
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * width + left;
    const rowEnd = y * width + right;
    data32.fill(0, rowStart, rowEnd);
  }
}

function resolveBatchForLocalFrame(
  segment: HeatmapTimelineSegment,
  localFrame: number,
): {
  batch: CFPBatch | null;
  batchFrameStart: number;
  batchFrameCount: number;
  batchFreqCount: number;
  batchLocalFrame: number;
} {
  let batchIndex = 0;
  while (
    batchIndex < segment.batchFrameStarts.length - 1 &&
    localFrame >= segment.batchFrameStarts[batchIndex]! + segment.batchFrameCounts[batchIndex]!
  ) {
    batchIndex += 1;
  }

  const batch = segment.batches?.[batchIndex] ?? null;
  const batchFrameStart = segment.batchFrameStarts[batchIndex] ?? 0;
  const batchFrameCount = segment.batchFrameCounts[batchIndex] ?? 1;
  const batchFreqCount = segment.batchFreqCounts[batchIndex] ?? 1;
  const batchLocalFrame = clampNumber(
    localFrame - batchFrameStart,
    0,
    Math.max(0, batchFrameCount - 1),
  );

  return {
    batch,
    batchFrameStart,
    batchFrameCount,
    batchFreqCount,
    batchLocalFrame,
  };
}

type HeatmapXRegion = {
  xStart: number;
  xEnd: number;
  batch: CFPBatch;
  batchFrameCount: number;
  batchFreqCount: number;
  batchLocalFrame: number;
};

function appendHeatmapRegion(
  regions: HeatmapXRegion[],
  next: HeatmapXRegion,
): void {
  const last = regions.length ? regions[regions.length - 1] : null;
  if (
    last &&
    last.xEnd === next.xStart &&
    last.batch === next.batch &&
    last.batchFrameCount === next.batchFrameCount &&
    last.batchFreqCount === next.batchFreqCount &&
    last.batchLocalFrame === next.batchLocalFrame
  ) {
    last.xEnd = next.xEnd;
    return;
  }
  regions.push(next);
}

function toPaletteIndex(normalized: number): number {
  if (normalized <= 0) return 0;
  if (normalized >= 1) return HEATMAP_PALETTE_SIZE - 1;
  return Math.floor(normalized * (HEATMAP_PALETTE_SIZE - 1));
}

function drawHeatmapRegion(
  data32: Uint32Array,
  width: number,
  height: number,
  region: HeatmapXRegion,
  yStarts: readonly number[],
  yRowHeights: readonly number[],
  yFreqIndexes: readonly number[],
  minFreq: number,
  visibleMin: number,
  invValueSpan: number,
  hasValueSpan: boolean,
): void {
  const { batch, batchFrameCount, batchFreqCount, batchLocalFrame, xStart, xEnd } = region;
  if (!batch || !(batch.data instanceof Float32Array)) {
    return;
  }
  for (let yi = 0; yi < yStarts.length; yi += 1) {
    const y = yStarts[yi] ?? 0;
    const rowHeight = yRowHeights[yi] ?? 1;
    const freqIndex = yFreqIndexes[yi] ?? minFreq;
    const clampedFreq = Math.min(freqIndex, Math.max(0, batchFreqCount - 1));
    const src = clampedFreq * batchFrameCount + batchLocalFrame;
    const value = batch.data[src];
    if (value === undefined || !Number.isFinite(value)) {
      continue;
    }
    const normalized = hasValueSpan ? (value - visibleMin) * invValueSpan : 0;
    const packed = HEATMAP_PALETTE_U32[toPaletteIndex(normalized)] ?? 0;
    const yEnd = Math.min(height, y + rowHeight);
    for (let yy = y; yy < yEnd; yy += 1) {
      const rowOffset = yy * width;
      data32.fill(packed, rowOffset + xStart, rowOffset + xEnd);
    }
  }
}

function colorFromNormalized(normalized: number): [number, number, number, number] {
  const t = Math.max(0, Math.min(1, normalized));
  const warm = Math.pow(t, 0.9);
  return [
    Math.round(16 + 224 * warm),
    Math.round(20 + 180 * t),
    Math.round(28 + 90 * (1 - t)),
    255,
  ];
}

function renderHeatmapTimelineBaseline(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  safeWidth: number,
  safeHeight: number,
  timeline: HeatmapTimeline,
  viewport: HeatmapTimelineViewport | null,
  viewportMode: "slots" | "frames",
  representativeMode: string,
  frequencyViewport: HeatmapFrequencyViewport | null,
): void {
  const image = ctx.createImageData(safeWidth, safeHeight);
  const data = image.data;
  const fullMaxFreq = Math.max(0, timeline.freqCount - 1);
  const minFreq = frequencyViewport
    ? clampNumber(Math.floor(Number(frequencyViewport.minBin) || 0), 0, fullMaxFreq)
    : 0;
  const maxFreq = frequencyViewport
    ? clampNumber(Math.floor(Number(frequencyViewport.maxBin) || fullMaxFreq), minFreq, fullMaxFreq)
    : fullMaxFreq;
  const visibleFreqSpan = Math.max(1, maxFreq - minFreq + 1);
  const visibleMin = Number.isFinite(timeline.min) ? timeline.min : 0;
  const visibleMax = Number.isFinite(timeline.max) ? timeline.max : visibleMin + 1;
  const valueSpan = Math.max(1e-9, visibleMax - visibleMin);

  if (viewportMode === "frames") {
    const totalFrames = Math.max(1, timeline.totalFrames || 0);
    const startFrame = Math.max(0, Math.floor(Number(viewport?.startSlot) || 0));
    const endFrame = Math.max(startFrame + 1, Math.floor(Number(viewport?.endSlot) || totalFrames));
    const visibleFrames = Math.max(1, endFrame - startFrame);
    let frameCursor = 0;
    for (const segment of timeline.segments) {
      const segFrameStart = frameCursor;
      frameCursor += Math.max(0, segment.frameCount);
      const segFrameEnd = frameCursor;
      if (!segment || segment.empty || segment.frameCount <= 0 || segment.freqCount <= 0) continue;
      const visStart = Math.max(startFrame, segFrameStart);
      const visEnd = Math.min(endFrame, segFrameEnd);
      if (visEnd <= visStart) continue;
      const visStartPx = Math.floor(((visStart - startFrame) / visibleFrames) * safeWidth);
      const visEndPx = Math.max(visStartPx + 1, Math.floor(((visEnd - startFrame) / visibleFrames) * safeWidth));
      const spanPx = Math.max(1, visEndPx - visStartPx);
      for (let x = visStartPx; x < visEndPx; x += 1) {
        const localSpan = Math.max(1, spanPx - 1);
        const frame = Math.max(
          visStart,
          Math.min(
            visEnd - 1,
            Math.floor(visStart + ((x - visStartPx) / localSpan) * Math.max(1, visEnd - visStart - 1)),
          ),
        );
        const localFrame = pickRepresentativeIndex({
          start: Math.max(segFrameStart, frame),
          end: Math.min(segFrameEnd, frame + 1),
          mode: representativeMode,
        });
        const resolved = resolveBatchForLocalFrame(segment, localFrame - segFrameStart);
        if (!resolved.batch || !(resolved.batch.data instanceof Float32Array)) continue;
        for (let y = 0; y < safeHeight; y += 1) {
          const yRatio = safeHeight > 1 ? (safeHeight - 1 - y) / (safeHeight - 1) : 0;
          const freqIndex = clampNumber(
            Math.floor(minFreq + yRatio * (visibleFreqSpan - 1)),
            minFreq,
            maxFreq,
          );
          const clampedFreq = Math.min(freqIndex, Math.max(0, resolved.batchFreqCount - 1));
          const src = clampedFreq * resolved.batchFrameCount + resolved.batchLocalFrame;
          const value = resolved.batch.data[src];
          if (value === undefined || !Number.isFinite(value)) continue;
          const normalized = (value - visibleMin) / valueSpan;
          const [r, g, b, a] = colorFromNormalized(normalized);
          const dst = (y * safeWidth + x) * 4;
          data[dst] = r;
          data[dst + 1] = g;
          data[dst + 2] = b;
          data[dst + 3] = a;
        }
      }
    }
  } else {
    // For benchmark baseline, slots mode falls back to frame-like traversal by slots.
    const totalSlots = Math.max(1, timeline.totalSlots);
    const startSlot = viewport
      ? clampNumber(Math.floor(Number(viewport.startSlot) || 0), 0, Math.max(0, totalSlots - 1))
      : 0;
    const endSlot = viewport
      ? clampNumber(
          Math.floor(Number(viewport.endSlot) || totalSlots),
          Math.min(totalSlots, startSlot + 1),
          totalSlots,
        )
      : totalSlots;
    const visibleSlots = Math.max(1, endSlot - startSlot);
    for (let slotIndex = startSlot; slotIndex < endSlot; slotIndex += 1) {
      const segment = timeline.segments[slotIndex];
      if (!segment || segment.empty || segment.frameCount <= 0) continue;
      const slotRelativeIndex = slotIndex - startSlot;
      const slotStartPx = Math.floor((slotRelativeIndex / visibleSlots) * safeWidth);
      const slotEndPx = Math.max(slotStartPx + 1, Math.floor(((slotRelativeIndex + 1) / visibleSlots) * safeWidth));
      const slotSpanPx = Math.max(1, slotEndPx - slotStartPx);
      for (let x = slotStartPx; x < slotEndPx; x += 1) {
        const localSpan = Math.max(1, slotSpanPx - 1);
        const localFrame = Math.max(
          0,
          Math.min(
            Math.max(0, segment.frameCount - 1),
            Math.floor(((x - slotStartPx) / localSpan) * Math.max(1, segment.frameCount - 1)),
          ),
        );
        const resolved = resolveBatchForLocalFrame(segment, localFrame);
        if (!resolved.batch || !(resolved.batch.data instanceof Float32Array)) continue;
        for (let y = 0; y < safeHeight; y += 1) {
          const yRatio = safeHeight > 1 ? (safeHeight - 1 - y) / (safeHeight - 1) : 0;
          const freqIndex = clampNumber(
            Math.floor(minFreq + yRatio * (visibleFreqSpan - 1)),
            minFreq,
            maxFreq,
          );
          const clampedFreq = Math.min(freqIndex, Math.max(0, resolved.batchFreqCount - 1));
          const src = clampedFreq * resolved.batchFrameCount + resolved.batchLocalFrame;
          const value = resolved.batch.data[src];
          if (value === undefined || !Number.isFinite(value)) continue;
          const normalized = (value - visibleMin) / valueSpan;
          const [r, g, b, a] = colorFromNormalized(normalized);
          const dst = (y * safeWidth + x) * 4;
          data[dst] = r;
          data[dst + 1] = g;
          data[dst + 2] = b;
          data[dst + 3] = a;
        }
      }
    }
  }

  ctx.putImageData(image, 0, 0);
}

export function renderHeatmapTimeline(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null | undefined,
  width: number,
  height: number,
  timeline: HeatmapTimeline | null,
  viewport: HeatmapTimelineViewport | null = null,
  viewportMode: "slots" | "frames" = "slots",
  representativeMode = "first-valid",
  frequencyViewport: HeatmapFrequencyViewport | null = null,
  sampleStrideFrames = 1,
  optimizationLevel: HeatmapOptimizationLevel = "u32-region",
): void {
  if (!ctx) return;
  const safeWidth = Math.max(1, Math.floor(Number(width) || 1));
  const safeHeight = Math.max(1, Math.floor(Number(height) || 1));
  ctx.clearRect(0, 0, safeWidth, safeHeight);
  const safeTimeline = timeline as HeatmapTimeline;
  if (!safeTimeline || !safeTimeline.totalSlots) return;

  if (optimizationLevel === "baseline-rgba") {
    renderHeatmapTimelineBaseline(
      ctx,
      safeWidth,
      safeHeight,
      safeTimeline,
      viewport,
      viewportMode,
      representativeMode,
      frequencyViewport,
    );
    return;
  }

  const image = ctx.createImageData(safeWidth, safeHeight);
  const data32 = new Uint32Array(image.data.buffer);
  const fullMaxFreq = Math.max(0, safeTimeline.freqCount - 1);
  const minFreq = frequencyViewport
    ? clampNumber(Math.floor(Number(frequencyViewport.minBin) || 0), 0, fullMaxFreq)
    : 0;
  const maxFreq = frequencyViewport
    ? clampNumber(Math.floor(Number(frequencyViewport.maxBin) || fullMaxFreq), minFreq, fullMaxFreq)
    : fullMaxFreq;
  const visibleFreqCount = Math.max(1, maxFreq - minFreq + 1);
  const visibleFreqSpan = Math.max(0, visibleFreqCount - 1);
  const yStepDraw = Math.max(1, Math.floor(visibleFreqCount / HEATMAP_MAX_DRAW_ROWS) || 1);
  const heightDenominator = Math.max(1, safeHeight - 1);
  const yStarts: number[] = [];
  const yRowHeights: number[] = [];
  const yFreqIndexes: number[] = [];
  for (let y = 0; y < safeHeight; y += yStepDraw) {
    const rowHeight = Math.max(1, Math.min(yStepDraw, safeHeight - y));
    let freqIndex = minFreq;
    if (safeHeight > 1) {
      const yRatio = (safeHeight - 1 - y) / heightDenominator;
      freqIndex = Math.floor(minFreq + yRatio * visibleFreqSpan);
      if (freqIndex < minFreq) freqIndex = minFreq;
      if (freqIndex > maxFreq) freqIndex = maxFreq;
    }
    yStarts.push(y);
    yRowHeights.push(rowHeight);
    yFreqIndexes.push(freqIndex);
  }

  const visibleMin = Number.isFinite(safeTimeline.min) ? safeTimeline.min : 0;
  const visibleMax = Number.isFinite(safeTimeline.max) ? safeTimeline.max : visibleMin + 1;
  const valueSpan = visibleMax - visibleMin;
  const hasValueSpan = valueSpan > 0;
  const invValueSpan = hasValueSpan ? 1 / valueSpan : 0;
  const minSampleStride = Math.max(1, Math.floor(Number(sampleStrideFrames) || 1));

  if (viewportMode === "frames") {
    const totalFrames = Math.max(1, safeTimeline.totalFrames || 0);
    const rawStartFrame = viewport
      ? Math.floor(Number(viewport.startSlot) || 0)
      : 0;
    const startFrame = Math.max(0, rawStartFrame);
    const rawEndFrame = viewport
      ? Math.floor(Number(viewport.endSlot) || totalFrames)
      : totalFrames;
    const endFrame = Math.max(startFrame + 1, rawEndFrame);
    const visibleFrames = Math.max(1, endFrame - startFrame);
    const sampleStep = Math.max(1, minSampleStride);

    let frameCursor = 0;
    for (const segment of safeTimeline.segments) {
      const segFrameStart = frameCursor;
      frameCursor += Math.max(0, segment.frameCount);
      const segFrameEnd = frameCursor;
      if (!segment || segment.empty || segment.frameCount <= 0 || segment.freqCount <= 0) {
        continue;
      }

      const visStart = Math.max(startFrame, segFrameStart);
      const visEnd = Math.min(endFrame, segFrameEnd);
      if (visEnd <= visStart) continue;

      const visStartPx = Math.floor(((visStart - startFrame) / visibleFrames) * safeWidth);
      const visEndPx =
        segFrameEnd >= endFrame
          ? safeWidth
          : Math.max(
              visStartPx + 1,
              Math.floor(((visEnd - startFrame) / visibleFrames) * safeWidth),
            );
      const spanPx = Math.max(1, visEndPx - visStartPx);
      const segVisibleFrames = Math.max(1, visEnd - visStart);
      const framesPerPixel = segVisibleFrames / Math.max(1, spanPx);
      const xStepPx = Math.max(
        1,
        Math.floor(sampleStep / Math.max(1e-6, framesPerPixel)),
      );
      const regions: HeatmapXRegion[] = [];

      for (let x = visStartPx; x < visEndPx; x += xStepPx) {
        const xEnd = Math.min(visEndPx, x + xStepPx);
        const localSpan = Math.max(1, spanPx - 1);
        const xCenter = x + (xEnd - x) * 0.5;
        const regionStartFrame = Math.max(
          visStart,
          Math.min(
            visEnd - 1,
            Math.floor(
              visStart +
                ((xCenter - visStartPx) / localSpan) * Math.max(1, visEnd - visStart - 1),
            ),
          ),
        );
        const regionEndFrame = Math.max(
          regionStartFrame + 1,
          Math.min(visEnd, regionStartFrame + sampleStep),
        );
        const localFrame = pickRepresentativeIndex({
          start: Math.max(segFrameStart, regionStartFrame),
          end: Math.min(segFrameEnd, regionEndFrame),
          mode: representativeMode,
        });
        const {
          batch,
          batchFrameCount,
          batchFreqCount,
          batchLocalFrame,
        } = resolveBatchForLocalFrame(segment, localFrame - segFrameStart);
        if (!batch || !(batch.data instanceof Float32Array)) {
          continue;
        }
        if (optimizationLevel === "u32") {
          regions.push({
            xStart: x,
            xEnd,
            batch,
            batchFrameCount,
            batchFreqCount,
            batchLocalFrame,
          });
        } else {
          appendHeatmapRegion(regions, {
            xStart: x,
            xEnd,
            batch,
            batchFrameCount,
            batchFreqCount,
            batchLocalFrame,
          });
        }
      }

      for (let i = 0; i < regions.length; i += 1) {
        drawHeatmapRegion(
          data32,
          safeWidth,
          safeHeight,
          regions[i]!,
          yStarts,
          yRowHeights,
          yFreqIndexes,
          minFreq,
          visibleMin,
          invValueSpan,
          hasValueSpan,
        );
      }
    }
  } else {
    const totalSlots = Math.max(1, safeTimeline.totalSlots);
    const startSlot = viewport
      ? clampNumber(Math.floor(Number(viewport.startSlot) || 0), 0, Math.max(0, totalSlots - 1))
      : 0;
    const endSlot = viewport
      ? clampNumber(
          Math.floor(Number(viewport.endSlot) || totalSlots),
          Math.min(totalSlots, startSlot + 1),
          totalSlots,
        )
      : totalSlots;
    const visibleSlots = Math.max(1, endSlot - startSlot);

    for (let slotIndex = startSlot; slotIndex < endSlot; slotIndex += 1) {
      const segment = safeTimeline.segments[slotIndex];
      const slotRelativeIndex = slotIndex - startSlot;
      const slotStartPx = Math.floor((slotRelativeIndex / visibleSlots) * safeWidth);
      const slotEndPx =
        slotIndex >= endSlot - 1
          ? safeWidth
          : Math.max(
              slotStartPx + 1,
              Math.floor(((slotRelativeIndex + 1) / visibleSlots) * safeWidth),
            );

      if (!segment || segment.empty || segment.frameCount <= 0 || segment.freqCount <= 0) {
        fillEmptySlot(data32, safeWidth, safeHeight, slotStartPx, slotEndPx);
        continue;
      }

      const slotSpanPx = Math.max(1, slotEndPx - slotStartPx);
      const sampleStep = Math.max(1, minSampleStride);
      const slotFramesPerPixel = Math.max(1, segment.frameCount) / Math.max(1, slotSpanPx);
      const xStepPx = Math.max(
        1,
        Math.floor(sampleStep / Math.max(1e-6, slotFramesPerPixel)),
      );
      const regions: HeatmapXRegion[] = [];
      for (let x = slotStartPx; x < slotEndPx; x += xStepPx) {
        const xEnd = Math.min(slotEndPx, x + xStepPx);
        const localSpan = Math.max(1, slotSpanPx - 1);
        const xCenter = x + (xEnd - x) * 0.5;
        const regionStartFrame = Math.max(
          0,
          Math.min(
            Math.max(0, segment.frameCount - 1),
            Math.floor(
              ((xCenter - slotStartPx) / localSpan) * Math.max(1, segment.frameCount - 1),
            ),
          ),
        );
        const regionEndFrame = Math.max(
          regionStartFrame + 1,
          Math.min(segment.frameCount, regionStartFrame + sampleStep),
        );
        const localFrame = pickRepresentativeIndex({
          start: regionStartFrame,
          end: regionEndFrame,
          mode: representativeMode,
        });
        const {
          batch,
          batchFrameCount,
          batchFreqCount,
          batchLocalFrame,
        } = resolveBatchForLocalFrame(segment, localFrame);
        if (!batch || !(batch.data instanceof Float32Array)) {
          continue;
        }
        if (optimizationLevel === "u32") {
          regions.push({
            xStart: x,
            xEnd,
            batch,
            batchFrameCount,
            batchFreqCount,
            batchLocalFrame,
          });
        } else {
          appendHeatmapRegion(regions, {
            xStart: x,
            xEnd,
            batch,
            batchFrameCount,
            batchFreqCount,
            batchLocalFrame,
          });
        }
      }

      for (let i = 0; i < regions.length; i += 1) {
        drawHeatmapRegion(
          data32,
          safeWidth,
          safeHeight,
          regions[i]!,
          yStarts,
          yRowHeights,
          yFreqIndexes,
          minFreq,
          visibleMin,
          invValueSpan,
          hasValueSpan,
        );
      }
    }
  }

  ctx.putImageData(image, 0, 0);

  ctx.save?.();
  try {
    if ("strokeStyle" in ctx) {
      ctx.strokeStyle = "rgba(255,255,255,0.05)";
      ctx.lineWidth = 1;
      if (viewportMode === "frames") {
        const totalFrames = Math.max(1, safeTimeline.totalFrames || 0);
        const rawStartFrame = viewport
          ? Math.floor(Number(viewport.startSlot) || 0)
          : 0;
        const startFrame = Math.max(0, rawStartFrame);
        const rawEndFrame = viewport
          ? Math.floor(Number(viewport.endSlot) || totalFrames)
          : totalFrames;
        const endFrame = Math.max(startFrame + 1, rawEndFrame);
        const visibleFrames = Math.max(1, endFrame - startFrame);
        let frameCursor = 0;
        ctx.beginPath();
        for (const segment of safeTimeline.segments) {
          const segFrameStart = frameCursor;
          frameCursor += Math.max(0, segment.frameCount);
          const segFrameEnd = frameCursor;
          const visStart = Math.max(startFrame, segFrameStart);
          const visEnd = Math.min(endFrame, segFrameEnd);
          if (visEnd <= visStart) continue;
          const x =
            Math.round(((visStart - startFrame) / visibleFrames) * safeWidth) + 0.5;
          ctx.moveTo(x, 0);
          ctx.lineTo(x, safeHeight);
        }
        ctx.stroke();
      } else {
        const totalSlots = Math.max(1, safeTimeline.totalSlots);
        const startSlot = viewport
          ? clampNumber(Math.floor(Number(viewport.startSlot) || 0), 0, Math.max(0, totalSlots - 1))
          : 0;
        const endSlot = viewport
          ? clampNumber(
              Math.floor(Number(viewport.endSlot) || totalSlots),
              Math.min(totalSlots, startSlot + 1),
              totalSlots,
            )
          : totalSlots;
        const visibleSlots = Math.max(1, endSlot - startSlot);
        ctx.beginPath();
        for (let slotIndex = startSlot + 1; slotIndex < endSlot; slotIndex += 1) {
          const x =
            Math.round(((slotIndex - startSlot) / visibleSlots) * safeWidth) + 0.5;
          ctx.moveTo(x, 0);
          ctx.lineTo(x, safeHeight);
        }
        ctx.stroke();
      }
    }
  } catch {}
  try {
    ctx.restore?.();
  } catch {}
}

export function benchmarkHeatmapTimelineRender(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null | undefined,
  width: number,
  height: number,
  timeline: HeatmapTimeline | null,
  viewport: HeatmapTimelineViewport | null,
  viewportMode: "slots" | "frames",
  representativeMode: string,
  frequencyViewport: HeatmapFrequencyViewport | null,
  sampleStrideFrames: number,
  rounds = 3,
): HeatmapBenchmarkEntry[] {
  const now = () =>
    typeof performance !== "undefined" && typeof performance.now === "function"
      ? performance.now()
      : Date.now();
  const variants: Array<{ name: string; optimizationLevel: HeatmapOptimizationLevel; stride: number }> = [
    { name: "baseline", optimizationLevel: "baseline-rgba", stride: 1 },
    { name: "stage1-u32", optimizationLevel: "u32", stride: 1 },
    {
      name: "stage2-u32-region-stride",
      optimizationLevel: "u32-region",
      stride: Math.max(1, Math.floor(Number(sampleStrideFrames) || 1)),
    },
  ];
  const results: HeatmapBenchmarkEntry[] = [];
  for (const variant of variants) {
    let totalMs = 0;
    const totalRounds = Math.max(1, Math.floor(Number(rounds) || 1));
    for (let i = 0; i < totalRounds; i += 1) {
      const t0 = now();
      renderHeatmapTimeline(
        ctx,
        width,
        height,
        timeline,
        viewport,
        viewportMode,
        representativeMode,
        frequencyViewport,
        variant.stride,
        variant.optimizationLevel,
      );
      totalMs += now() - t0;
    }
    results.push({
      name: variant.name,
      avgMs: totalMs / Math.max(1, totalRounds),
      sampleStrideFrames: variant.stride,
      optimizationLevel: variant.optimizationLevel,
    });
  }
  return results;
}
