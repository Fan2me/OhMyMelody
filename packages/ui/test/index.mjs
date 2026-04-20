import assert from "node:assert/strict";
import test from "node:test";

const previousSelf = globalThis.self;
globalThis.self = globalThis;
const uiExports = await import("../dist/controller.js");
globalThis.self = previousSelf;

test("normalizeSpectrumSectionConfig handles booleans and partial overrides", () => {
  const { normalizeSpectrumSectionConfig } = uiExports;

  assert.deepEqual(
    normalizeSpectrumSectionConfig(false, 30),
    { enabled: false, overlay: false, fps: 30 },
  );

  assert.deepEqual(
    normalizeSpectrumSectionConfig(true, 24),
    { enabled: true, overlay: true, fps: 24 },
  );

  assert.deepEqual(
    normalizeSpectrumSectionConfig({ enabled: true, overlay: false, fps: 12 }, 60),
    { enabled: true, overlay: false, fps: 12 },
  );
});

test("buildSpectrumTimeline keeps CFP frame layout intact", () => {
  const { buildSpectrumTimeline } = uiExports;
  const batches = [
    {
      data: new Float32Array([0, 1, 2, 3, 4, 5]),
      shape: new Int32Array([3, 3, 2]),
    },
    {
      data: new Float32Array([5, 4, 3, 2, 1, 0]),
      shape: new Int32Array([3, 3, 2]),
    },
  ];

  const timeline = buildSpectrumTimeline(batches);

  assert.equal(timeline.segments.length, 2);
  assert.equal(timeline.totalFrames, 4);
  assert.equal(timeline.freqCount, 3);
  assert.equal(timeline.min, 0);
  assert.equal(timeline.max, 5);
  assert.equal(timeline.segments[0].startFrame, 0);
  assert.equal(timeline.segments[0].frameCount, 2);
  assert.equal(timeline.segments[0].freqCount, 3);
  assert.equal(timeline.segments[1].startFrame, 2);
  assert.equal(timeline.segments[1].frameCount, 2);
});
