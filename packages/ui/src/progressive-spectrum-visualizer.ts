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

  /**
   * 处理队列中的所有 CFP 数据块，将频谱数据写入可视化缓冲区
   * 实现滑动窗口机制：数据溢出时自动左移旧数据
   */
  function drainQueuedChunks() {
    queued = false;  // 清除调度标志
    if (pendingChunkReadIndex >= pendingChunks.length) return;  // 无待处理数据

    let touched = false;  // 标记是否有数据被实际处理

    // 遍历所有待处理的 chunk
    while (pendingChunkReadIndex < pendingChunks.length) {
      const one = pendingChunks[pendingChunkReadIndex];
      pendingChunkReadIndex += 1;
      if (!one) continue;  // 跳过无效数据

      // 获取 batch 形状：C=通道数, F=频率数, T=时间帧数
      const [C, F, T] = getBatchShape(one);
      if (!Number.isFinite(C) || !Number.isFinite(F) || !Number.isFinite(T) || T <= 0) continue;  // 验证数据有效性

      // 缓冲区未初始化或尺寸不匹配时，重新初始化
      if (totalFrames <= 0 || spectrumData === EMPTY_FLOAT32 || spectrumData.length !== state.spectrumH * totalFrames) {
        ensureBase(totalFrames > 0 ? totalFrames : T, pendingDurationSec, true, true);
      }

      // 计算实际使用的通道和频率维度
      const useC = Math.max(0, Math.min(C - 1, 0));  // 使用第一个通道
      const useF = Math.min(state.spectrumH, F);    // 限制频率数不超过窗口高度
      const currentCount = state.windowFrameCount;  // 当前窗口中已有的帧数

      // 模式 A：当前 chunk 足够填满整个窗口
      if (T >= state.spectrumW) {
        // 取 chunk 的最后 spectrumW 帧，直接填满窗口
        const srcStartT = T - state.spectrumW;
        for (let f = 0; f < useF; f += 1) {
          const channelOffset = useC * F * T;
          const freqOffset = f * T;
          const srcStart = channelOffset + freqOffset + srcStartT;  // 源数据起始索引
          const srcEnd = channelOffset + freqOffset + T;              // 源数据结束索引
          const dstStart = f * state.spectrumW;                // 目标数据起始索引
          state.spectrumData.set(one.data.subarray(srcStart, srcEnd), dstStart);
        }
        state.windowFrameCount = state.spectrumW;  // 窗口已满
        state.writeFrameOffset = state.spectrumW;
      }
      // 模式 B：chunk 较小，需要滚动窗口
      else {
        let xOffset = currentCount;  // 新数据的写入起始位置
        // 计算溢出量：当前帧数 + 新帧数 - 窗口宽度
        const overflow = currentCount + T - state.spectrumW;
        if (overflow > 0) {
          // 左移频谱数据，为新数据腾出空间
          for (let f = 0; f < state.spectrumH; f += 1) {
            const rowStart = f * state.spectrumW;
            state.spectrumData.copyWithin(rowStart, rowStart + overflow, rowStart + currentCount);
          }
          // 同样左移预测数据
          state.predictionArgmax.copyWithin(0, overflow, currentCount);
          state.predictionConfidence.copyWithin(0, overflow, currentCount);
          xOffset = currentCount - overflow;  // 更新写入位置
        }

        // 将新 chunk 的数据写入窗口右侧空闲区域
        for (let t = 0; t < Math.min(T, Math.max(0, state.spectrumW - xOffset)); t += 1) {
          for (let f = 0; f < useF; f += 1) {
            const src = useC * F * T + f * T + t;  // 源数据索引
            const dst = f * state.spectrumW + (xOffset + t);  // 目标数据索引
            state.spectrumData[dst] = Number(one.data[src] ?? NaN);
          }
        }
        state.windowFrameCount = Math.min(state.spectrumW, xOffset + T);  // 更新窗口帧数
        state.writeFrameOffset = state.windowFrameCount;
      }

      state.sourceFrameOffset += T;  // 更新已处理的源数据帧偏移
      touched = true;
    }

    // 清空队列
    pendingChunks.length = 0;
    pendingChunkReadIndex = 0;

    // 如果有数据被处理，触发重绘
    if (touched) {
      markDirty();
      requestRedraw({ dirtyMask: DIRTY.MAIN_BASE });
    }
  }

  function enqueueChunk(one: CFPBatch) {
    pendingChunks.push(one);
    scheduleChunkDrain();
  }

  /**
   * 应用预测数据块到可视化状态
   * 将模型的预测结果（argmax 和 confidence）写入缓冲区
   */
  function applyPredictionChunk(args: {
    predictionArgmax: ArrayLike<number> | null | undefined;
    predictionConfidence?: ArrayLike<number> | null | undefined;
    predictionOffset?: number; 
  }) {
    const predictionArgmax = args.predictionArgmax;
    const predictionConfidence = args.predictionConfidence;
    if (!predictionArgmax) return;

    const start = Math.max(0, Math.floor(Number.isFinite(args.predictionOffset as number) ? Number(args.predictionOffset) : 0));
    // 计算最大可复制长度：不超过窗口剩余空间和预测数据长度
    const maxCopy = Math.min(state.spectrumW - start, Number(predictionArgmax.length) || 0);

    // 写入预测的音高类别索引
    for (let i = 0; i < maxCopy; i += 1) {
      const value = Number(predictionArgmax[i]);
      state.predictionArgmax[start + i] = Number.isFinite(value) ? value : NaN;
    }

    // 写入预测置信度（如果有）
    if (predictionConfidence) {
      const confCopy = Math.min(state.spectrumW - start, Number(predictionConfidence.length) || 0);
      for (let i = 0; i < confCopy; i += 1) {
        state.predictionConfidence[start + i] = Number(predictionConfidence[i] ?? 0);
      }
    }

    // 更新预测版本号，用于缓存失效
    state.predictionRevision += 1;
    // 更新已渲染帧数
    state.renderedFrames = Math.max(state.renderedFrames, start + maxCopy);
    // 触发主视图覆盖层重绘（显示预测曲线）
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
  };
}
