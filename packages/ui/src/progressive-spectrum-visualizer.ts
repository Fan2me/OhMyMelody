import type { CFPBatch } from "@ohm/core/cache/cfp.js";
import { DIRTY } from "./spectrum-state.js";

export interface ProgressiveSpectrumState {
  spectrumData: Float32Array;
  spectrumW: number;
  spectrumH: number;
  sourceFrameOffset: number;
  windowFrameCount: number;
  predictionArgmax: Float32Array;
  predictionConfidence: Float32Array;
  predictionRevision: number;
  writeFrameOffset: number;
  renderedFrames: number;
}

export interface ProgressiveSpectrumPayload {
  data: Float32Array;
  width: number;
  height: number;
  argmax: Float32Array;
  confidence: Float32Array;
}

export interface SpectrumRedrawRequest {
  includeOverviewBase?: boolean;
  dirtyMask?: number;
}

export interface ProgressiveSpectrumVisualizerOptions {
  setSpectrumPayload?: ((payload: ProgressiveSpectrumPayload) => void) | null;
  setSpectrumDuration?: ((duration: number) => void) | null;
  requestSpectrumRedraw?: ((opts?: SpectrumRedrawRequest) => void) | null;
  markSpectrumDataDirty?: (() => void) | null;
  frameSec?: number;
  maxWindowFrames?: number;
  initialExpectedFrames?: number;
  initialDurationSec?: number;
  spectrumHeight?: number;
}

export interface ProgressiveSpectrumVisualizer {
  state: ProgressiveSpectrumState;
  reset(next?: {
    expectedFrames?: number;
    durationSec?: number;
    preserveExisting?: boolean;
    pushToUi?: boolean;
  }): void;
  ensureBase(expectedFrames: number, durationSec?: number, pushToUi?: boolean, preserveExisting?: boolean): void;
  enqueueChunk(one: CFPBatch): void;
  applyPredictionChunk(args: {
    predictionArgmax: ArrayLike<number> | null | undefined;
    predictionConfidence?: ArrayLike<number> | null | undefined;
    predictionOffset?: number;
  }): void;
  flush(): void;
  setFrameSec(nextFrameSec: number): void;
}

const EMPTY_FLOAT32 = new Float32Array(0);
const DEFAULT_FRAME_SEC = 0.01;
const DEFAULT_HEIGHT = 360;
const DEFAULT_MAX_WINDOW_FRAMES = 120000;

function createBlankProgressiveSpectrumPayload(totalFrames: number, height = DEFAULT_HEIGHT) {
  const spectrumW = Math.max(1, Math.floor(totalFrames || 1));
  const spectrumH = Math.max(1, Math.floor(height || DEFAULT_HEIGHT));
  const spectrumData = new Float32Array(spectrumH * spectrumW);
  spectrumData.fill(NaN);
  const predictionArgmax = new Float32Array(spectrumW);
  predictionArgmax.fill(NaN);
  const predictionConfidence = new Float32Array(spectrumW);
  predictionConfidence.fill(NaN);
  return {
    spectrumData,
    spectrumW,
    spectrumH,
    predictionArgmax,
    predictionConfidence,
  };
}

function getBatchShape(batch: CFPBatch): [number, number, number] {
  const shape = batch.shape;
  const C = Math.max(1, Math.floor(Number(shape?.[0]) || 1));
  const F = Math.max(1, Math.floor(Number(shape?.[1]) || 1));
  const T = Math.max(1, Math.floor(Number(shape?.[2]) || 1));
  return [C, F, T];
}

function copyPredictionSeries(
  destination: Float32Array,
  source: Float32Array,
): void {
  if (!destination.length || !source.length) {
    return;
  }
  destination.set(source.subarray(0, Math.min(destination.length, source.length)));
}

function copySpectrumRows(
  destination: Float32Array,
  destinationWidth: number,
  source: Float32Array,
  sourceWidth: number,
  rowCount: number,
  frameCount: number,
): void {
  if (!destinationWidth || !sourceWidth || !rowCount || !frameCount) {
    return;
  }

  const rows = Math.max(0, Math.min(rowCount, Math.floor(destination.length / destinationWidth)));
  const frames = Math.max(0, Math.min(frameCount, destinationWidth, sourceWidth));
  if (!rows || !frames) {
    return;
  }

  for (let y = 0; y < rows; y += 1) {
    const dstStart = y * destinationWidth;
    const srcStart = y * sourceWidth;
    destination.set(source.subarray(srcStart, srcStart + frames), dstStart);
  }
}

export function createProgressiveSpectrumVisualizer(
  options: ProgressiveSpectrumVisualizerOptions = {},
): ProgressiveSpectrumVisualizer {
  const {
    setSpectrumPayload = null,
    setSpectrumDuration = null,
    requestSpectrumRedraw = null,
    markSpectrumDataDirty = null,
    frameSec = DEFAULT_FRAME_SEC,
    maxWindowFrames = DEFAULT_MAX_WINDOW_FRAMES,
    initialExpectedFrames = 0,
    initialDurationSec = 0,
    spectrumHeight = DEFAULT_HEIGHT,
  } = options;

  let currentFrameSec = Number.isFinite(frameSec) && frameSec > 0 ? frameSec : DEFAULT_FRAME_SEC;
  let totalFrames = 0;
  let height = Math.max(1, Math.floor(spectrumHeight || DEFAULT_HEIGHT));
  let spectrumData = EMPTY_FLOAT32;
  let predictionArgmax = EMPTY_FLOAT32;
  let predictionConfidence = EMPTY_FLOAT32;
  let pendingDurationSec = 0;
  let queued = false;
  const pendingChunks: CFPBatch[] = [];
  let pendingChunkReadIndex = 0;
  const state: ProgressiveSpectrumState = {
    spectrumData,
    spectrumW: 0,
    spectrumH: height,
    sourceFrameOffset: 0,
    windowFrameCount: 0,
    predictionArgmax,
    predictionConfidence,
    predictionRevision: 0,
    writeFrameOffset: 0,
    renderedFrames: 0,
  };

  function syncState(next: Partial<ProgressiveSpectrumState>) {
    if (Object.prototype.hasOwnProperty.call(next, "spectrumData")) state.spectrumData = next.spectrumData ?? state.spectrumData;
    if (Object.prototype.hasOwnProperty.call(next, "spectrumW")) state.spectrumW = next.spectrumW ?? state.spectrumW;
    if (Object.prototype.hasOwnProperty.call(next, "spectrumH")) state.spectrumH = next.spectrumH ?? state.spectrumH;
    if (Object.prototype.hasOwnProperty.call(next, "sourceFrameOffset")) state.sourceFrameOffset = next.sourceFrameOffset ?? state.sourceFrameOffset;
    if (Object.prototype.hasOwnProperty.call(next, "windowFrameCount")) state.windowFrameCount = next.windowFrameCount ?? state.windowFrameCount;
    if (Object.prototype.hasOwnProperty.call(next, "predictionArgmax")) state.predictionArgmax = next.predictionArgmax ?? state.predictionArgmax;
    if (Object.prototype.hasOwnProperty.call(next, "predictionConfidence")) state.predictionConfidence = next.predictionConfidence ?? state.predictionConfidence;
    if (Object.prototype.hasOwnProperty.call(next, "predictionRevision")) state.predictionRevision = next.predictionRevision ?? state.predictionRevision;
    if (Object.prototype.hasOwnProperty.call(next, "writeFrameOffset")) state.writeFrameOffset = next.writeFrameOffset ?? state.writeFrameOffset;
    if (Object.prototype.hasOwnProperty.call(next, "renderedFrames")) state.renderedFrames = next.renderedFrames ?? state.renderedFrames;
  }

  function applySpectrumPayload(payload: ProgressiveSpectrumPayload): void {
    if (typeof setSpectrumPayload !== "function") return;
    setSpectrumPayload(payload);
  }

  function applyDuration(durationSec: number) {
    pendingDurationSec = Math.max(0, Number(durationSec || 0));
    if (typeof setSpectrumDuration === "function") {
      setSpectrumDuration(pendingDurationSec);
    }
  }

  function markDirty() {
    if (typeof markSpectrumDataDirty === "function") {
      markSpectrumDataDirty();
    }
  }

  function requestRedraw(fullOrOptions: boolean | SpectrumRedrawRequest = false): void {
    if (typeof requestSpectrumRedraw !== "function") return;
    if (typeof fullOrOptions === "boolean") {
      requestSpectrumRedraw();
    } else {
      requestSpectrumRedraw(fullOrOptions);
    }
  }

  function ensureBase(expectedFrames: number, durationSec = 0, pushToUi = true, preserveExisting = true) {
    const safeTotalFrames = Math.max(1, Math.min(maxWindowFrames, Math.floor(expectedFrames || 1)));
    const prevData = spectrumData;
    const prevArgmax = predictionArgmax;
    const prevConfidence = predictionConfidence;
    const prevHeight = height;
    totalFrames = safeTotalFrames;

    const blank = createBlankProgressiveSpectrumPayload(safeTotalFrames, height);
    if (preserveExisting && prevData.length > 0 && prevData !== EMPTY_FLOAT32) {
      const prevWidth = prevHeight > 0 ? Math.floor(prevData.length / prevHeight) : 0;
      copySpectrumRows(
        blank.spectrumData,
        blank.spectrumW,
        prevData,
        prevWidth,
        Math.min(blank.spectrumH, prevHeight),
        prevWidth,
      );
      copyPredictionSeries(blank.predictionArgmax, prevArgmax);
      copyPredictionSeries(blank.predictionConfidence, prevConfidence);
    }

    spectrumData = blank.spectrumData;
    predictionArgmax = blank.predictionArgmax;
    predictionConfidence = blank.predictionConfidence;
    syncState({
      spectrumData,
      spectrumW: blank.spectrumW,
      spectrumH: blank.spectrumH,
      predictionArgmax,
      predictionConfidence,
      predictionRevision: state.predictionRevision + 1,
      sourceFrameOffset: 0,
      windowFrameCount: 0,
      writeFrameOffset: 0,
      renderedFrames: 0,
    });

    const resolvedDurationSec = Math.max(
      0,
      Number(durationSec || 0),
      safeTotalFrames * currentFrameSec,
      pendingDurationSec,
    );
    applyDuration(resolvedDurationSec);

    if (pushToUi) {
      applySpectrumPayload({
        data: blank.spectrumData,
        width: blank.spectrumW,
        height: blank.spectrumH,
        argmax: blank.predictionArgmax,
        confidence: blank.predictionConfidence,
      });
      requestRedraw({ includeOverviewBase: true });
    }
  }

  function scheduleChunkDrain(): void {
    if (queued) {
      return;
    }
    queued = true;
    queueMicrotask(drainQueuedChunks);
  }

  function drainQueuedChunks() {
    queued = false;
    if (pendingChunkReadIndex >= pendingChunks.length) return;
    let touched = false;
    while (pendingChunkReadIndex < pendingChunks.length) {
      const one = pendingChunks[pendingChunkReadIndex];
      pendingChunkReadIndex += 1;
      if (!one) continue;
      const [C, F, T] = getBatchShape(one);
      if (!Number.isFinite(C) || !Number.isFinite(F) || !Number.isFinite(T) || T <= 0) continue;

      if (totalFrames <= 0 || spectrumData === EMPTY_FLOAT32 || spectrumData.length !== state.spectrumH * totalFrames) {
        ensureBase(totalFrames > 0 ? totalFrames : T, pendingDurationSec, true, true);
      }

      const useC = Math.max(0, Math.min(C - 1, 0));
      const useF = Math.min(state.spectrumH, F);
      const currentCount = state.windowFrameCount;
      let xOffset = currentCount;
      if (T >= state.spectrumW) {
        const srcStartT = T - state.spectrumW;
        for (let f = 0; f < useF; f += 1) {
          const srcStart = useC * F * T + f * T + srcStartT;
          const srcEnd = useC * F * T + f * T + T;
          const dstStart = f * state.spectrumW;
          state.spectrumData.set(one.data.subarray(srcStart, srcEnd), dstStart);
        }
        state.windowFrameCount = state.spectrumW;
        state.writeFrameOffset = state.spectrumW;
        xOffset = state.spectrumW;
      } else {
        const overflow = currentCount + T - state.spectrumW;
        if (overflow > 0) {
          for (let f = 0; f < state.spectrumH; f += 1) {
            const rowStart = f * state.spectrumW;
            state.spectrumData.copyWithin(rowStart, rowStart + overflow, rowStart + currentCount);
          }
          state.predictionArgmax.copyWithin(0, overflow, currentCount);
          state.predictionConfidence.copyWithin(0, overflow, currentCount);
          xOffset = currentCount - overflow;
        }

        for (let t = 0; t < Math.min(T, Math.max(0, state.spectrumW - xOffset)); t += 1) {
          for (let f = 0; f < useF; f += 1) {
            const src = useC * F * T + f * T + t;
            const dst = f * state.spectrumW + (xOffset + t);
            state.spectrumData[dst] = Number(one.data[src] ?? NaN);
          }
        }
        state.windowFrameCount = Math.min(state.spectrumW, xOffset + T);
        state.writeFrameOffset = state.windowFrameCount;
      }
      state.sourceFrameOffset += T;
      touched = true;
    }
    pendingChunks.length = 0;
    pendingChunkReadIndex = 0;
    if (touched) {
      markDirty();
      requestRedraw({ dirtyMask: DIRTY.MAIN_BASE });
    }
  }

  function enqueueChunk(one: CFPBatch) {
    pendingChunks.push(one);
    scheduleChunkDrain();
  }

  function applyPredictionChunk(args: {
    predictionArgmax: ArrayLike<number> | null | undefined;
    predictionConfidence?: ArrayLike<number> | null | undefined;
    predictionOffset?: number;
  }) {
    const predictionArgmax = args.predictionArgmax;
    const predictionConfidence = args.predictionConfidence;
    if (!predictionArgmax) return;

    const start = Math.max(0, Math.floor(Number.isFinite(args.predictionOffset as number) ? Number(args.predictionOffset) : 0));
    const maxCopy = Math.min(state.spectrumW - start, Number(predictionArgmax.length) || 0);
    for (let i = 0; i < maxCopy; i += 1) {
      const value = Number(predictionArgmax[i]);
      state.predictionArgmax[start + i] = Number.isFinite(value) ? value : NaN;
    }
    if (predictionConfidence) {
      const confCopy = Math.min(state.spectrumW - start, Number(predictionConfidence.length) || 0);
      for (let i = 0; i < confCopy; i += 1) {
        state.predictionConfidence[start + i] = Number(predictionConfidence[i] ?? 0);
      }
    }
    state.predictionRevision += 1;
    state.renderedFrames = Math.max(state.renderedFrames, start + maxCopy);
    requestRedraw({ dirtyMask: DIRTY.MAIN_OVERLAY });
  }

  function reset(next: {
    expectedFrames?: number;
    durationSec?: number;
    preserveExisting?: boolean;
    pushToUi?: boolean;
  } = {}) {
    pendingChunks.length = 0;
    pendingChunkReadIndex = 0;
    queued = false;
    state.sourceFrameOffset = 0;
    state.windowFrameCount = 0;
    state.writeFrameOffset = 0;
    state.renderedFrames = 0;
    state.predictionRevision += 1;
    if (typeof next.expectedFrames === "number" && next.expectedFrames > 0) {
      ensureBase(
        next.expectedFrames,
        typeof next.durationSec === "number" ? next.durationSec : pendingDurationSec,
        next.pushToUi !== false,
        next.preserveExisting !== false,
      );
    } else {
      spectrumData = EMPTY_FLOAT32;
      predictionArgmax = EMPTY_FLOAT32;
      predictionConfidence = EMPTY_FLOAT32;
      totalFrames = 0;
      height = Math.max(1, Math.floor(next.expectedFrames || height || DEFAULT_HEIGHT));
      syncState({
        spectrumData,
        spectrumW: 0,
        spectrumH: height,
        predictionArgmax,
        predictionConfidence,
        predictionRevision: state.predictionRevision,
      });
    }
  }

  if (initialExpectedFrames > 0) {
    ensureBase(initialExpectedFrames, initialDurationSec, true, false);
  }

  return {
    state,
    reset,
    ensureBase,
    enqueueChunk,
    applyPredictionChunk,
    flush: drainQueuedChunks,
    setFrameSec(nextFrameSec: number) {
      if (Number.isFinite(nextFrameSec) && nextFrameSec > 0) {
        currentFrameSec = nextFrameSec;
      }
    },
  };
}
