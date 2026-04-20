import {
  buildSpectrumTimeline,
  renderHeatmapTimeline,
  type HeatmapFrequencyViewport,
  type HeatmapOptimizationLevel,
  type HeatmapTimeline,
  type HeatmapTimelineSlot,
} from "./heatmap-render-core.js";

type HeatmapRenderCommand =
  | {
      cmd: "set-timeline";
      cfp: readonly HeatmapTimelineSlot[];
      seq: number;
    }
  | {
      cmd: "render";
      target: "main" | "overview";
      width: number;
      height: number;
      viewport?: {
        startSlot: number;
        endSlot: number;
      } | null;
      viewportMode?: "slots" | "frames";
      frequencyViewport?: HeatmapFrequencyViewport | null;
      representativeMode?: string;
      sampleStrideFrames?: number;
      optimizationLevel?: HeatmapOptimizationLevel;
      requestId: number;
    };

type HeatmapRenderResponse =
  | {
      cmd: "ready";
      seq: number;
    }
  | {
      cmd: "frame";
      target: "main" | "overview";
      requestId: number;
      bitmap: ImageBitmap;
    }
  | {
      cmd: "error";
      target?: "main" | "overview";
      requestId?: number;
      error: string;
    };

let timeline: HeatmapTimeline | null = null;

const workerScope = globalThis as typeof globalThis & {
  postMessage(message: unknown, transfer?: Transferable[]): void;
};

function postReady(seq: number): void {
  const payload: HeatmapRenderResponse = {
    cmd: "ready",
    seq,
  };
  workerScope.postMessage(payload);
}

self.onmessage = (event: MessageEvent<HeatmapRenderCommand>) => {
  const message = event.data || ({} as HeatmapRenderCommand);

  try {
    if (message.cmd === "set-timeline") {
      timeline = buildSpectrumTimeline(message.cfp);
      postReady(message.seq);
      return;
    }

    if (message.cmd === "render") {
      const width = Math.max(1, Math.floor(Number(message.width) || 1));
      const height = Math.max(1, Math.floor(Number(message.height) || 1));
      const offscreen = new OffscreenCanvas(width, height);
      const ctx = offscreen.getContext("2d");
      renderHeatmapTimeline(
        ctx,
        width,
        height,
        timeline,
        message.viewport ?? null,
        message.viewportMode ?? (message.target === "main" ? "frames" : "slots"),
        message.representativeMode ?? "first-valid",
        message.frequencyViewport ?? null,
        Math.max(1, Math.floor(Number(message.sampleStrideFrames) || 1)),
        message.optimizationLevel ?? "u32-region",
      );
      const bitmap = offscreen.transferToImageBitmap();
      const payload: HeatmapRenderResponse = {
        cmd: "frame",
        target: message.target,
        requestId: message.requestId,
        bitmap,
      };
      workerScope.postMessage(payload, [bitmap]);
      return;
    }
  } catch (error) {
    const payload: HeatmapRenderResponse =
      "target" in message && "requestId" in message
        ? {
            cmd: "error",
            target: message.target,
            requestId: message.requestId,
          error: error instanceof Error ? error.message : String(error),
          }
        : {
            cmd: "error",
            error: error instanceof Error ? error.message : String(error),
          };
    workerScope.postMessage(payload);
  }
};
