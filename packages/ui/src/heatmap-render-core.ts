import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";
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
type HeatmapRenderContext = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

const HEATMAP_RENDER_CACHE = new WeakMap<HeatmapRenderContext, HeatmapRenderCache>();
const BATCH_RENDER_ID_CACHE = new WeakMap<CFPBatch, number>();
const heatmapRenderLogger = getModuleLogger("core.ui.heatmap.render");

let nextBatchRenderId = 1;

type HeatmapRenderCache = {
  timelineKey: string;
  width: number;
  height: number;
  viewportStart: number;
  viewportEnd: number;
  viewportMode: "slots" | "frames";
  representativeMode: string;
  sampleStrideFrames: number;
  optimizationLevel: HeatmapOptimizationLevel;
  minFreq: number;
  maxFreq: number;
  visibleMin: number;
  visibleMax: number;
  representativeFrameCache: Map<string, number>;
};

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

function getBatchRenderId(batch: CFPBatch): number {
  const cached = BATCH_RENDER_ID_CACHE.get(batch);
  if (typeof cached === "number") {
    return cached;
  }
  const next = nextBatchRenderId;
  nextBatchRenderId += 1;
  BATCH_RENDER_ID_CACHE.set(batch, next);
  return next;
}

function getSegmentRenderKey(segment: HeatmapTimelineSegment): string {
  const batchIds = Array.isArray(segment.batches)
    ? segment.batches.map((batch) => getBatchRenderId(batch)).join(",")
    : "none";
  return `slot:${segment.slotIndex}|frames:${segment.frameCount}|freq:${segment.freqCount}|batches:${batchIds}`;
}

function getTimelineRenderKey(timeline: HeatmapTimeline): string {
  const segmentKeys = timeline.segments.map((segment) => getSegmentRenderKey(segment)).join(";");
  return [
    `slots:${timeline.totalSlots}`,
    `frames:${timeline.totalFrames}`,
    `freq:${timeline.freqCount}`,
    `min:${timeline.min.toFixed(6)}`,
    `max:${timeline.max.toFixed(6)}`,
    `segments:${segmentKeys}`,
  ].join("|");
}

function buildRepresentativeRegionCacheKey(args: {
  viewportMode: "slots" | "frames";
  representativeMode: string;
  sampleStrideFrames: number;
  segmentKey: string;
  regionBucketStart: number;
  regionBucketEnd: number;
  minFreq: number;
  maxFreq: number;
  visibleMin: number;
  visibleMax: number;
}): string {
  return [
    `mode:${args.viewportMode}`,
    `rep:${args.representativeMode}`,
    `stride:${args.sampleStrideFrames}`,
    `seg:${args.segmentKey}`,
    `bucket:${args.regionBucketStart}-${args.regionBucketEnd}`,
    `freq:${args.minFreq}-${args.maxFreq}`,
    `value:${args.visibleMin.toFixed(6)}-${args.visibleMax.toFixed(6)}`,
  ].join("|");
}

function resolveRepresentativeFrameIndex(args: {
  segment: HeatmapTimelineSegment;
  viewportMode: "slots" | "frames";
  representativeMode: string;
  sampleStrideFrames: number;
  regionStartFrame: number;
  regionEndFrame: number;
  minFreq: number;
  maxFreq: number;
  visibleMin: number;
  visibleMax: number;
  cache: Map<string, number>;
}): number {
  const segmentKey = getSegmentRenderKey(args.segment);
  const stride = Math.max(1, Math.floor(args.sampleStrideFrames) || 1);
  const regionStart = Math.max(0, Math.floor(args.regionStartFrame));
  const regionEnd = Math.max(regionStart + 1, Math.floor(args.regionEndFrame));
  const bucketStart = Math.floor(regionStart / stride);
  const bucketEnd = Math.floor((regionEnd - 1) / stride) + 1;
  const cacheKey = buildRepresentativeRegionCacheKey({
    viewportMode: args.viewportMode,
    representativeMode: args.representativeMode,
    sampleStrideFrames: stride,
    segmentKey,
    regionBucketStart: bucketStart,
    regionBucketEnd: bucketEnd,
    minFreq: args.minFreq,
    maxFreq: args.maxFreq,
    visibleMin: args.visibleMin,
    visibleMax: args.visibleMax,
  });
  const cached = args.cache.get(cacheKey);
  if (Number.isFinite(cached)) {
    return Math.max(0, Math.floor(Number(cached)));
  }
  const picked = pickRepresentativeIndex({
    start: regionStart,
    end: regionEnd,
    mode: args.representativeMode,
  });
  args.cache.set(cacheKey, picked);
  return picked;
}

function paintHeatmapRegion(
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
  drawHeatmapRegion(
    data32,
    width,
    height,
    region,
    yStarts,
    yRowHeights,
    yFreqIndexes,
    minFreq,
    visibleMin,
    invValueSpan,
    hasValueSpan,
  );
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


type HeatmapViewportMode = "slots" | "frames";

type HeatmapYMapping = {
  yStarts: number[];
  yRowHeights: number[];
  yFreqIndexes: number[];
};

type HeatmapDebugRegion = {
  probeX?: number;
  segmentKey: string;
  xStart: number;
  xEnd: number;
  regionStartFrame: number;
  regionEndFrame: number;
  localFrame: number;
  batchLocalFrame: number;
  batchFrameCount: number;
  sampleStrideFrames: number;
  viewportMoved: boolean;
  frameBias: number;
  regionBucketStart: number;
  regionBucketEnd: number;
};

type RenderState = {
  ctx: HeatmapRenderContext;
  width: number;
  height: number;
  timeline: HeatmapTimeline;
  viewport: HeatmapTimelineViewport | null;
  viewportMode: HeatmapViewportMode;
  representativeMode: string;
  frequencyViewport: HeatmapFrequencyViewport | null;
  sampleStrideFrames: number;
  optimizationLevel: HeatmapOptimizationLevel;
  minFreq: number;
  maxFreq: number;
  visibleMin: number;
  visibleMax: number;
  hasValueSpan: boolean;
  invValueSpan: number;
  timelineKey: string;
};

type CacheState = {
  canUseRegionCache: boolean;
  previousCache: HeatmapRenderCache | null;
  viewportMoved: boolean;
  cacheIsCompatible: boolean;
  representativeFrameCache: Map<string, number>;
};

type RenderResult = {
  viewportStart: number;
  viewportEnd: number;
  firstDebugRegion: HeatmapDebugRegion | null;
  debugProbeRegions: HeatmapDebugRegion[];
  visibleProbeRegions: HeatmapDebugRegion[];
};

type ViewSpan = {
  mode: HeatmapViewportMode;
  segment: HeatmapTimelineSegment | undefined;
  xStart: number;
  xEnd: number;
  viewportStartFrame: number;
  viewportEndFrame: number;
  viewportFrameCount: number;
  segmentFrameStart: number;
  segmentFrameEnd: number;
  visibleFrameStart: number;
  visibleFrameEnd: number;
};

function frequencyRange(
  timeline: HeatmapTimeline,
  frequencyViewport: HeatmapFrequencyViewport | null,
): { minFreq: number; maxFreq: number; visibleFreqSpan: number } {
  const fullMaxFreq = Math.max(0, timeline.freqCount - 1);
  const minFreq = frequencyViewport
    ? clampNumber(Math.floor(Number(frequencyViewport.minBin) || 0), 0, fullMaxFreq)
    : 0;
  const maxFreq = frequencyViewport
    ? clampNumber(Math.floor(Number(frequencyViewport.maxBin) || fullMaxFreq), minFreq, fullMaxFreq)
    : fullMaxFreq;
  return {
    minFreq,
    maxFreq,
    visibleFreqSpan: Math.max(0, maxFreq - minFreq),
  };
}

function valueRange(timeline: HeatmapTimeline): {
  visibleMin: number;
  visibleMax: number;
  hasValueSpan: boolean;
  invValueSpan: number;
  baselineValueSpan: number;
} {
  const visibleMin = Number.isFinite(timeline.min) ? timeline.min : 0;
  const visibleMax = Number.isFinite(timeline.max) ? timeline.max : visibleMin + 1;
  const valueSpan = visibleMax - visibleMin;
  return {
    visibleMin,
    visibleMax,
    hasValueSpan: valueSpan > 0,
    invValueSpan: valueSpan > 0 ? 1 / valueSpan : 0,
    baselineValueSpan: Math.max(1e-9, valueSpan),
  };
}

function createRenderState(args: {
  ctx: HeatmapRenderContext;
  width: number;
  height: number;
  timeline: HeatmapTimeline;
  viewport: HeatmapTimelineViewport | null;
  viewportMode: HeatmapViewportMode;
  representativeMode: string;
  frequencyViewport: HeatmapFrequencyViewport | null;
  sampleStrideFrames: number;
  optimizationLevel: HeatmapOptimizationLevel;
}): RenderState {
  const freq = frequencyRange(args.timeline, args.frequencyViewport);
  const value = valueRange(args.timeline);
  return {
    ...args,
    ...freq,
    visibleMin: value.visibleMin,
    visibleMax: value.visibleMax,
    hasValueSpan: value.hasValueSpan,
    invValueSpan: value.invValueSpan,
    sampleStrideFrames: Math.max(1, Math.floor(Number(args.sampleStrideFrames) || 1)),
    timelineKey: getTimelineRenderKey(args.timeline),
  };
}

function createYMapping(height: number, minFreq: number, maxFreq: number): HeatmapYMapping {
  const visibleFreqSpan = Math.max(0, maxFreq - minFreq);
  const heightDenominator = Math.max(1, height - 1);
  const yStarts: number[] = [];
  const yRowHeights: number[] = [];
  const yFreqIndexes: number[] = [];

  for (let y = 0; y < height; y += 1) {
    const yRatio = height > 1 ? (height - 1 - y) / heightDenominator : 0;
    yStarts.push(y);
    yRowHeights.push(1);
    yFreqIndexes.push(clampNumber(Math.floor(minFreq + yRatio * visibleFreqSpan), minFreq, maxFreq));
  }

  return { yStarts, yRowHeights, yFreqIndexes };
}

function createCacheState(state: RenderState): CacheState {
  const canUseRegionCache = state.viewportMode !== "frames";
  const previousCache = canUseRegionCache ? HEATMAP_RENDER_CACHE.get(state.ctx) ?? null : null;
  const viewportMoved =
    !!previousCache && previousCache.viewportMode === state.viewportMode && state.viewport
      ? Math.floor(Number(state.viewport.startSlot) || 0) !== previousCache.viewportStart
      : false;
  const cacheIsCompatible =
    !!previousCache &&
    previousCache.timelineKey === state.timelineKey &&
    previousCache.width === state.width &&
    previousCache.height === state.height &&
    previousCache.viewportMode === state.viewportMode &&
    previousCache.representativeMode === state.representativeMode &&
    previousCache.sampleStrideFrames === state.sampleStrideFrames &&
    previousCache.optimizationLevel === state.optimizationLevel &&
    previousCache.minFreq === state.minFreq &&
    previousCache.maxFreq === state.maxFreq &&
    previousCache.visibleMin === state.visibleMin &&
    previousCache.visibleMax === state.visibleMax;

  return {
    canUseRegionCache,
    previousCache,
    viewportMoved,
    cacheIsCompatible,
    representativeFrameCache:
      cacheIsCompatible && previousCache
        ? new Map(previousCache.representativeFrameCache)
        : new Map<string, number>(),
  };
}

function eachVisibleSpan(
  timeline: HeatmapTimeline,
  viewport: HeatmapTimelineViewport | null,
  viewportMode: HeatmapViewportMode,
  width: number,
  visit: (span: ViewSpan) => void,
): { viewportStart: number; viewportEnd: number } {
  if (viewportMode === "frames") {
    const totalFrames = Math.max(1, timeline.totalFrames || 0);
    const startFrame = Math.max(0, Math.floor(Number(viewport?.startSlot) || 0));
    const endFrame = Math.max(startFrame + 1, Math.floor(Number(viewport?.endSlot) || totalFrames));
    const visibleFrames = Math.max(1, endFrame - startFrame);
    let frameCursor = 0;

    for (const segment of timeline.segments) {
      const segmentFrameStart = frameCursor;
      frameCursor += Math.max(0, segment.frameCount);
      const segmentFrameEnd = frameCursor;
      const visibleFrameStart = Math.max(startFrame, segmentFrameStart);
      const visibleFrameEnd = Math.min(endFrame, segmentFrameEnd);
      if (visibleFrameEnd <= visibleFrameStart) continue;

      const xStart = Math.floor(((visibleFrameStart - startFrame) / visibleFrames) * width);
      const xEnd =
        segmentFrameEnd >= endFrame
          ? width
          : Math.max(xStart + 1, Math.floor(((visibleFrameEnd - startFrame) / visibleFrames) * width));
      visit({
        mode: "frames",
        segment,
        xStart,
        xEnd,
        viewportStartFrame: startFrame,
        viewportEndFrame: endFrame,
        viewportFrameCount: visibleFrames,
        segmentFrameStart,
        segmentFrameEnd,
        visibleFrameStart,
        visibleFrameEnd,
      });
    }
    return { viewportStart: startFrame, viewportEnd: endFrame };
  }

  const totalSlots = Math.max(1, timeline.totalSlots);
  const startSlot = viewport
    ? clampNumber(Math.floor(Number(viewport.startSlot) || 0), 0, Math.max(0, totalSlots - 1))
    : 0;
  const endSlot = viewport
    ? clampNumber(Math.floor(Number(viewport.endSlot) || totalSlots), Math.min(totalSlots, startSlot + 1), totalSlots)
    : totalSlots;
  const visibleSlots = Math.max(1, endSlot - startSlot);

  for (let slotIndex = startSlot; slotIndex < endSlot; slotIndex += 1) {
    const relativeSlot = slotIndex - startSlot;
    const segment = timeline.segments[slotIndex];
    const xStart = Math.floor((relativeSlot / visibleSlots) * width);
    const xEnd =
      slotIndex >= endSlot - 1
        ? width
        : Math.max(xStart + 1, Math.floor(((relativeSlot + 1) / visibleSlots) * width));
    const segmentFrameEnd = Math.max(0, segment?.frameCount ?? 0);

    visit({
      mode: "slots",
      segment,
      xStart,
      xEnd,
      viewportStartFrame: startSlot,
      viewportEndFrame: endSlot,
      viewportFrameCount: visibleSlots,
      segmentFrameStart: 0,
      segmentFrameEnd,
      visibleFrameStart: 0,
      visibleFrameEnd: segmentFrameEnd,
    });
  }
  return { viewportStart: startSlot, viewportEnd: endSlot };
}

function isRenderableSegment(segment: HeatmapTimelineSegment | undefined): segment is HeatmapTimelineSegment {
  return !!segment && !segment.empty && segment.frameCount > 0 && segment.freqCount > 0;
}

function resolveBaselineLocalFrame(span: ViewSpan, x: number, representativeMode: string): number {
  const segment = span.segment;
  if (!segment) return 0;

  if (span.mode === "slots") {
    const localSpan = Math.max(1, span.xEnd - span.xStart - 1);
    return clampNumber(
      Math.floor(((x - span.xStart) / localSpan) * Math.max(1, segment.frameCount - 1)),
      0,
      Math.max(0, segment.frameCount - 1),
    );
  }

  const localSpan = Math.max(1, span.xEnd - span.xStart - 1);
  const frame = clampNumber(
    Math.floor(
      span.visibleFrameStart +
        ((x - span.xStart) / localSpan) * Math.max(1, span.visibleFrameEnd - span.visibleFrameStart - 1),
    ),
    span.visibleFrameStart,
    span.visibleFrameEnd - 1,
  );
  return (
    pickRepresentativeIndex({
      start: Math.max(span.segmentFrameStart, frame),
      end: Math.min(span.segmentFrameEnd, frame + 1),
      mode: representativeMode,
    }) - span.segmentFrameStart
  );
}

function paintBaselineColumn(args: {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  x: number;
  resolved: ReturnType<typeof resolveBatchForLocalFrame>;
  minFreq: number;
  maxFreq: number;
  visibleFreqSpan: number;
  visibleMin: number;
  valueSpan: number;
}): void {
  const { data, width, height, x, resolved } = args;
  if (!resolved.batch || !(resolved.batch.data instanceof Float32Array)) return;

  for (let y = 0; y < height; y += 1) {
    const yRatio = height > 1 ? (height - 1 - y) / (height - 1) : 0;
    const freqIndex = clampNumber(Math.floor(args.minFreq + yRatio * args.visibleFreqSpan), args.minFreq, args.maxFreq);
    const src = Math.min(freqIndex, Math.max(0, resolved.batchFreqCount - 1)) * resolved.batchFrameCount + resolved.batchLocalFrame;
    const value = resolved.batch.data[src];
    if (value === undefined || !Number.isFinite(value)) continue;

    const [r, g, b, a] = colorFromNormalized((value - args.visibleMin) / args.valueSpan);
    const dst = (y * width + x) * 4;
    data[dst] = r;
    data[dst + 1] = g;
    data[dst + 2] = b;
    data[dst + 3] = a;
  }
}

function renderHeatmapTimelineBaseline(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  safeWidth: number,
  safeHeight: number,
  timeline: HeatmapTimeline,
  viewport: HeatmapTimelineViewport | null,
  viewportMode: HeatmapViewportMode,
  representativeMode: string,
  frequencyViewport: HeatmapFrequencyViewport | null,
): void {
  const image = ctx.createImageData(safeWidth, safeHeight);
  const freq = frequencyRange(timeline, frequencyViewport);
  const value = valueRange(timeline);

  eachVisibleSpan(timeline, viewport, viewportMode, safeWidth, (span) => {
    if (!isRenderableSegment(span.segment)) return;
    for (let x = span.xStart; x < span.xEnd; x += 1) {
      const resolved = resolveBatchForLocalFrame(
        span.segment,
        resolveBaselineLocalFrame(span, x, representativeMode),
      );
      paintBaselineColumn({
        data: image.data,
        width: safeWidth,
        height: safeHeight,
        x,
        resolved,
        minFreq: freq.minFreq,
        maxFreq: freq.maxFreq,
        visibleFreqSpan: freq.visibleFreqSpan,
        visibleMin: value.visibleMin,
        valueSpan: value.baselineValueSpan,
      });
    }
  });

  ctx.putImageData(image, 0, 0);
}

function createDebugRegion(args: {
  probeX?: number;
  segment: HeatmapTimelineSegment;
  xStart: number;
  xEnd: number;
  regionStartFrame: number;
  regionEndFrame: number;
  localFrame: number;
  batchLocalFrame: number;
  batchFrameCount: number;
  sampleStrideFrames: number;
  viewportMoved: boolean;
  frameBias: number;
  regionBucketStart: number;
  regionBucketEnd: number;
}): HeatmapDebugRegion {
  return {
    ...(args.probeX === undefined ? {} : { probeX: args.probeX }),
    segmentKey: getSegmentRenderKey(args.segment),
    xStart: args.xStart,
    xEnd: args.xEnd,
    regionStartFrame: args.regionStartFrame,
    regionEndFrame: args.regionEndFrame,
    localFrame: args.localFrame,
    batchLocalFrame: args.batchLocalFrame,
    batchFrameCount: args.batchFrameCount,
    sampleStrideFrames: args.sampleStrideFrames,
    viewportMoved: args.viewportMoved,
    frameBias: args.frameBias,
    regionBucketStart: args.regionBucketStart,
    regionBucketEnd: args.regionBucketEnd,
  };
}

function regionSeed(span: ViewSpan, x: number, xEnd: number, sampleStrideFrames: number, width: number): {
  xStart: number;
  xEnd: number;
  regionStartFrame: number;
  regionEndFrame: number;
  sampleStrideFrames: number;
  frameBias: number;
  batchLookupOffset: number;
} | null {
  if (span.mode === "frames") {
    const xCenter = x + (xEnd - x) * 0.5;
    const progress = clampNumber((xCenter - 0.5) / Math.max(1, width), 0, 1);
    const exactFrame = clampNumber(
      Math.round(span.viewportStartFrame + progress * Math.max(1, span.viewportFrameCount - 1)),
      span.viewportStartFrame,
      span.viewportEndFrame - 1,
    );
    if (exactFrame < span.segmentFrameStart || exactFrame >= span.segmentFrameEnd) return null;
    const regionStartFrame = Math.max(
      span.segmentFrameStart,
      Math.min(span.segmentFrameEnd - 1, Math.floor(exactFrame / sampleStrideFrames) * sampleStrideFrames),
    );
    return {
      xStart: x,
      xEnd,
      regionStartFrame,
      regionEndFrame: Math.min(span.segmentFrameEnd, regionStartFrame + sampleStrideFrames),
      sampleStrideFrames,
      frameBias: 0,
      batchLookupOffset: span.segmentFrameStart,
    };
  }

  const segment = span.segment;
  if (!segment) return null;
  const slotSpanPx = Math.max(1, span.xEnd - span.xStart);
  const xCenter = x + (xEnd - x) * 0.5;
  const progress = clampNumber((xCenter - span.xStart) / Math.max(1, slotSpanPx - 1), 0, 1);
  const regionStartFrame = clampNumber(
    Math.round(progress * Math.max(1, segment.frameCount - 1)),
    0,
    Math.max(0, segment.frameCount - 1),
  );
  return {
    xStart: x,
    xEnd,
    regionStartFrame,
    regionEndFrame: Math.max(regionStartFrame + 1, Math.min(segment.frameCount, regionStartFrame + sampleStrideFrames)),
    sampleStrideFrames,
    frameBias: 0,
    batchLookupOffset: 0,
  };
}

function regionStepPx(span: ViewSpan, sampleStrideFrames: number, width: number): number {
  if (span.mode === "frames") {
    return Math.max(1, Math.floor(sampleStrideFrames / Math.max(1e-6, span.viewportFrameCount / Math.max(1, width))));
  }
  const slotSpanPx = Math.max(1, span.xEnd - span.xStart);
  const framesPerPixel = Math.max(1, span.segment?.frameCount ?? 0) / Math.max(1, slotSpanPx);
  return Math.max(1, Math.floor(sampleStrideFrames / Math.max(1e-6, framesPerPixel)));
}

function resolveRegion(args: {
  state: RenderState;
  cacheState: CacheState;
  span: ViewSpan;
  seed: ReturnType<typeof regionSeed>;
}): { region: HeatmapXRegion; debugRegion: HeatmapDebugRegion } | null {
  const { state, cacheState, span, seed } = args;
  const segment = span.segment;
  if (!seed || !isRenderableSegment(segment)) return null;

  const regionBucketStart = Math.floor(seed.regionStartFrame / seed.sampleStrideFrames);
  const regionBucketEnd = Math.floor(Math.max(seed.regionStartFrame, seed.regionEndFrame - 1) / seed.sampleStrideFrames) + 1;
  const localFrame = resolveRepresentativeFrameIndex({
    segment,
    viewportMode: state.viewportMode,
    representativeMode: state.representativeMode,
    sampleStrideFrames: seed.sampleStrideFrames,
    regionStartFrame: Math.min(Math.max(0, segment.frameCount - 1), seed.regionStartFrame + seed.frameBias),
    regionEndFrame: seed.regionEndFrame,
    minFreq: state.minFreq,
    maxFreq: state.maxFreq,
    visibleMin: state.visibleMin,
    visibleMax: state.visibleMax,
    cache: cacheState.representativeFrameCache,
  });
  const batchLookupFrame = localFrame - seed.batchLookupOffset;
  const { batch, batchFrameCount, batchFreqCount, batchLocalFrame } = resolveBatchForLocalFrame(segment, batchLookupFrame);
  if (!batch || !(batch.data instanceof Float32Array)) return null;

  return {
    region: { xStart: seed.xStart, xEnd: seed.xEnd, batch, batchFrameCount, batchFreqCount, batchLocalFrame },
    debugRegion: createDebugRegion({
      segment,
      xStart: seed.xStart,
      xEnd: seed.xEnd,
      regionStartFrame: seed.regionStartFrame,
      regionEndFrame: seed.regionEndFrame,
      localFrame,
      batchLocalFrame: Math.max(0, batchLookupFrame),
      batchFrameCount: segment.frameCount,
      sampleStrideFrames: seed.sampleStrideFrames,
      viewportMoved: cacheState.viewportMoved,
      frameBias: seed.frameBias,
      regionBucketStart,
      regionBucketEnd,
    }),
  };
}

function pushRegion(regions: HeatmapXRegion[], region: HeatmapXRegion, optimizationLevel: HeatmapOptimizationLevel): void {
  if (optimizationLevel === "u32") regions.push(region);
  else appendHeatmapRegion(regions, region);
}

function paintRegions(data32: Uint32Array, state: RenderState, yMapping: HeatmapYMapping, regions: readonly HeatmapXRegion[]): void {
  for (let i = 0; i < regions.length; i += 1) {
    paintHeatmapRegion(
      data32,
      state.width,
      state.height,
      regions[i]!,
      yMapping.yStarts,
      yMapping.yRowHeights,
      yMapping.yFreqIndexes,
      state.minFreq,
      state.visibleMin,
      state.invValueSpan,
      state.hasValueSpan,
    );
  }
}

function renderViewport(state: RenderState, cacheState: CacheState, yMapping: HeatmapYMapping, data32: Uint32Array): RenderResult {
  const result: RenderResult = {
    viewportStart: 0,
    viewportEnd: 0,
    firstDebugRegion: null,
    debugProbeRegions: [],
    visibleProbeRegions: [],
  };
  const probeXs = [
    Math.max(0, Math.floor(state.width * 0.1)),
    Math.max(0, Math.floor(state.width * 0.5)),
    Math.max(0, Math.floor(state.width * 0.9)),
  ];
  const bounds = eachVisibleSpan(state.timeline, state.viewport, state.viewportMode, state.width, (span) => {
    if (!isRenderableSegment(span.segment)) {
      if (span.mode === "slots") fillEmptySlot(data32, state.width, state.height, span.xStart, span.xEnd);
      return;
    }

    const regions: HeatmapXRegion[] = [];
    const stepPx = regionStepPx(span, state.sampleStrideFrames, state.width);
    for (let x = span.xStart; x < span.xEnd; x += stepPx) {
      const xEnd = Math.min(span.xEnd, x + stepPx);
      const resolved = resolveRegion({
        state,
        cacheState,
        span,
        seed: regionSeed(span, x, xEnd, state.sampleStrideFrames, state.width),
      });
      if (!resolved) continue;

      if (!result.firstDebugRegion) result.firstDebugRegion = resolved.debugRegion;
      if (span.mode === "slots") {
        const probeX = probeXs.find((probe) => probe >= resolved.region.xStart && probe < resolved.region.xEnd);
        if (probeX !== undefined && !result.visibleProbeRegions.some((probe) => probe.probeX === probeX)) {
          result.visibleProbeRegions.push({ ...resolved.debugRegion, probeX });
        }
      }
      pushRegion(regions, resolved.region, state.optimizationLevel);
    }
    paintRegions(data32, state, yMapping, regions);
  });

  result.viewportStart = bounds.viewportStart;
  result.viewportEnd = bounds.viewportEnd;
  return result;
}

function commitRenderCache(state: RenderState, cacheState: CacheState, result: RenderResult): void {
  if (!cacheState.canUseRegionCache) return;
  HEATMAP_RENDER_CACHE.set(state.ctx, {
    timelineKey: state.timelineKey,
    width: state.width,
    height: state.height,
    viewportStart: result.viewportStart,
    viewportEnd: result.viewportEnd,
    viewportMode: state.viewportMode,
    representativeMode: state.representativeMode,
    sampleStrideFrames: state.sampleStrideFrames,
    optimizationLevel: state.optimizationLevel,
    minFreq: state.minFreq,
    maxFreq: state.maxFreq,
    visibleMin: state.visibleMin,
    visibleMax: state.visibleMax,
    representativeFrameCache: cacheState.representativeFrameCache,
  });
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
  if (!timeline || !timeline.totalSlots) return;

  if (optimizationLevel === "baseline-rgba") {
    renderHeatmapTimelineBaseline(ctx, safeWidth, safeHeight, timeline, viewport, viewportMode, representativeMode, frequencyViewport);
    return;
  }

  const state = createRenderState({
    ctx,
    width: safeWidth,
    height: safeHeight,
    timeline,
    viewport,
    viewportMode,
    representativeMode,
    frequencyViewport,
    sampleStrideFrames,
    optimizationLevel,
  });
  const cacheState = createCacheState(state);
  const image = ctx.createImageData(state.width, state.height);
  const result = renderViewport(
    state,
    cacheState,
    createYMapping(state.height, state.minFreq, state.maxFreq),
    new Uint32Array(image.data.buffer),
  );

  if (state.viewportMode === "frames") {
    heatmapRenderLogger.info("heatmap frames render", {
      viewportStart: result.viewportStart,
      viewportEnd: result.viewportEnd,
      sampleStrideFrames: state.sampleStrideFrames,
      viewportMoved: cacheState.viewportMoved,
      cacheHit: cacheState.cacheIsCompatible && !!cacheState.previousCache,
      firstRegion: result.firstDebugRegion,
      probeRegions: result.debugProbeRegions,
      visibleProbeRegions: result.visibleProbeRegions,
    });
  }

  ctx.putImageData(image, 0, 0);
  commitRenderCache(state, cacheState, result);
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
      if (ctx) HEATMAP_RENDER_CACHE.delete(ctx);
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
