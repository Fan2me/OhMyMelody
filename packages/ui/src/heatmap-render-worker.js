import {
  buildSpectrumTimeline,
  renderHeatmapTimeline,
} from "./heatmap-render-core.js";

let timeline = null;

const workerScope = globalThis;

function postReady(seq) {
  workerScope.postMessage({
    cmd: "ready",
    seq,
  });
}

self.onmessage = (event) => {
  const message = event.data || {};

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
      workerScope.postMessage(
        {
          cmd: "frame",
          target: message.target,
          requestId: message.requestId,
          bitmap,
        },
        [bitmap],
      );
      return;
    }
  } catch (error) {
    workerScope.postMessage(
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
          },
    );
  }
};
