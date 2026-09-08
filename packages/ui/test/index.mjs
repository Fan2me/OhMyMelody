import assert from "node:assert/strict";
import test from "node:test";

const previousSelf = globalThis.self;
globalThis.self = globalThis;
const controllerExports = await import("../dist/controller.js");
const stateExports = await import("../dist/spectrum-state.js");
const heatmapExports = await import("../dist/heatmap-render-core.js");
const layoutExports = await import("../dist/spectrum-layout.js");
const mainOverlayExports = await import("../dist/spectrum-main-overlay.js");
globalThis.self = previousSelf;

test("normalizeSectionConfig handles booleans and partial overrides", () => {
  const { normalizeSectionConfig } = stateExports;

  assert.deepEqual(
    normalizeSectionConfig(false, 30),
    { enabled: false, overlay: false, fps: 30, overlayFps: 30 },
  );

  assert.deepEqual(
    normalizeSectionConfig(true, 24),
    { enabled: true, overlay: true, fps: 24, overlayFps: 24 },
  );

  assert.deepEqual(
    normalizeSectionConfig({ enabled: true, overlay: false, fps: 12 }, 60),
    { enabled: true, overlay: false, fps: 12, overlayFps: 12 },
  );
});

test("buildSpectrumTimeline keeps CFP frame layout intact", () => {
  const { buildSpectrumTimeline } = heatmapExports;
  const slots = [
    [
      {
        data: new Float32Array([0, 1, 2, 3, 4, 5]),
        shape: new Int32Array([3, 3, 2]),
      },
    ],
    [
      {
        data: new Float32Array([5, 4, 3, 2, 1, 0]),
        shape: new Int32Array([3, 3, 2]),
      },
    ],
  ];

  const timeline = buildSpectrumTimeline(slots);

  assert.equal(timeline.segments.length, 2);
  assert.equal(timeline.frames.length, 4);
  assert.equal(timeline.totalFrames, 4);
  assert.equal(timeline.freqCount, 3);
  assert.equal(timeline.min, 0);
  assert.equal(timeline.max, 5);
  assert.equal(timeline.segments[0].slotIndex, 0);
  assert.equal(timeline.segments[0].frameCount, 2);
  assert.equal(timeline.segments[0].freqCount, 3);
  assert.deepEqual(timeline.segments[0].batchFrameStarts, [0]);
  assert.equal(timeline.frames[0].globalFrame, 0);
  assert.equal(timeline.frames[3].globalFrame, 3);
  assert.equal(timeline.segments[1].slotIndex, 1);
  assert.equal(timeline.segments[1].frameCount, 2);
});

test("controller entrypoint exposes createUiController", () => {
  assert.equal(typeof controllerExports.createUiController, "function");
});

test("canvas logical size uses the rendered CSS rect instead of rounded client dimensions", () => {
  const { getCanvasLogicalSize, getPlotMetrics } = layoutExports;
  const canvas = {
    width: 1601,
    height: 721,
    clientWidth: 800,
    clientHeight: 360,
    getBoundingClientRect() {
      return { width: 800.5, height: 360.5 };
    },
  };

  assert.deepEqual(getCanvasLogicalSize(canvas), {
    width: 800.5,
    height: 360.5,
    dpr: 2,
  });
  assert.equal(getPlotMetrics(canvas, 0, 0).plotW, 800.5);
  assert.equal(getPlotMetrics(canvas, 0, 0).plotH, 360.5);
});

test("spectrum frame/time conversion is reversible for a non-frame-aligned duration", () => {
  const { spectrumFrameToTime, spectrumTimeToFrame } = layoutExports;
  const totalFrames = 1201;
  const durationSec = 12.345;
  const frame = 777.25;
  const time = spectrumFrameToTime(frame, totalFrames, durationSec);

  assert.ok(Math.abs(spectrumTimeToFrame(time, totalFrames, durationSec) - frame) < 1e-9);
});

test("main overlay playhead stays aligned with the interaction frame for non-frame-aligned durations", () => {
  const { createSpectrumMainOverlayRenderer } = mainOverlayExports;
  const { spectrumFrameToTime } = layoutExports;
  const totalFrames = 1201;
  const durationSec = 12.345;
  const targetFrame = 777.25;
  const targetWidth = 800.5;
  const strokes = [];
  let currentPath = [];
  const ctx = {
    setTransform() {},
    clearRect() {},
    save() {},
    restore() {},
    beginPath() { currentPath = []; },
    moveTo(x, y) { currentPath.push([x, y]); },
    lineTo(x, y) { currentPath.push([x, y]); },
    stroke() { strokes.push({ style: this.strokeStyle, path: currentPath }); },
    fillText() {},
    strokeStyle: "",
    lineWidth: 1,
    font: "",
    fillStyle: "",
    textAlign: "",
    textBaseline: "",
  };
  const canvas = {
    width: 1601,
    height: 721,
    clientWidth: 800,
    clientHeight: 360,
    getBoundingClientRect() {
      return { width: targetWidth, height: 360.5 };
    },
  };
  const renderer = createSpectrumMainOverlayRenderer({
    getCanvas: () => ({ canvas, ctx }),
    getState: () => ({
      audio: { pcm: new Float32Array(12345), fs: 1000 },
      pitchRange: { minBin: 0, maxBin: 10 },
      displaySampling: {},
      inference: null,
    }),
    getInteractionState: () => ({
      spectrumW: totalFrames,
      spectrumH: 11,
      spectrumZoom: 1,
      spectrumDuration: durationSec,
      spectrumOffset: 0,
      spectrumHoverActive: false,
      spectrumHoverX: null,
      spectrumHoverY: null,
      spectrumHoverFrame: -1,
    }),
    getAudioElement: () => ({ currentTime: spectrumFrameToTime(targetFrame, totalFrames, durationSec) }),
    getPredictionFrames: () => [],
  });

  renderer.drawOverlay();

  const playheadStroke = strokes.find((stroke) => stroke.style.includes("255, 78, 78"));
  assert.ok(playheadStroke);
  assert.ok(
    Math.abs(playheadStroke.path[0][0] - (targetFrame / totalFrames) * targetWidth) < 1e-9,
    `actual=${playheadStroke.path[0][0]} expected=${(targetFrame / totalFrames) * targetWidth}`,
  );
});
