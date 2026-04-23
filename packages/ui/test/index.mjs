import assert from "node:assert/strict";
import test from "node:test";

const previousSelf = globalThis.self;
globalThis.self = globalThis;
const controllerExports = await import("../dist/controller.js");
const stateExports = await import("../dist/spectrum-state.js");
const heatmapExports = await import("../dist/heatmap-render-core.js");
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
  assert.equal(timeline.totalFrames, 4);
  assert.equal(timeline.freqCount, 3);
  assert.equal(timeline.min, 0);
  assert.equal(timeline.max, 5);
  assert.equal(timeline.segments[0].slotIndex, 0);
  assert.equal(timeline.segments[0].frameCount, 2);
  assert.equal(timeline.segments[0].freqCount, 3);
  assert.deepEqual(timeline.segments[0].batchFrameStarts, [0]);
  assert.equal(timeline.segments[1].slotIndex, 1);
  assert.equal(timeline.segments[1].frameCount, 2);
});

test("controller entrypoint exposes createUiController", () => {
  assert.equal(typeof controllerExports.createUiController, "function");
});
