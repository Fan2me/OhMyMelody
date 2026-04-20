import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import test from "node:test";

const previousSelf = globalThis.self;
globalThis.self = globalThis;
const runtimeExports = await import("../dist/analyzer.js");
const coreInferenceExports = await import("../../core/dist/inference/inference.js");
globalThis.self = previousSelf;
const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));
const realAudioPath = `${repoRoot}20260405-211059.wav`;

async function decodeWavPcm16(path) {
  const bytes = new Uint8Array(await readFile(path));
  const text = (start, length) =>
    String.fromCharCode(...bytes.slice(start, start + length));
  if (text(0, 4) !== "RIFF" || text(8, 4) !== "WAVE") {
    throw new Error("fixture is not a WAV file");
  }

  let offset = 12;
  let audioFormat = 0;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataOffset = 0;
  let dataLength = 0;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  while (offset + 8 <= bytes.length) {
    const chunkId = text(offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkDataOffset = offset + 8;
    if (chunkId === "fmt ") {
      audioFormat = view.getUint16(chunkDataOffset + 0, true);
      channels = view.getUint16(chunkDataOffset + 2, true);
      sampleRate = view.getUint32(chunkDataOffset + 4, true);
      bitsPerSample = view.getUint16(chunkDataOffset + 14, true);
    } else if (chunkId === "data") {
      dataOffset = chunkDataOffset;
      dataLength = chunkSize;
      break;
    }
    offset = chunkDataOffset + chunkSize + (chunkSize % 2);
  }

  if (audioFormat !== 1 || bitsPerSample !== 16 || !channels || !sampleRate) {
    throw new Error("fixture WAV format is not supported by this test");
  }
  if (!dataOffset || !dataLength) {
    throw new Error("fixture WAV data chunk is missing");
  }

  const frameCount = Math.floor(dataLength / 2 / channels);
  const pcm = new Float32Array(frameCount);
  let cursor = dataOffset;
  for (let frameIdx = 0; frameIdx < frameCount; frameIdx += 1) {
    let sampleSum = 0;
    for (let channelIdx = 0; channelIdx < channels; channelIdx += 1) {
      sampleSum += view.getInt16(cursor, true) / 32768;
      cursor += 2;
    }
    pcm[frameIdx] = sampleSum / channels;
  }

  return { pcm, fs: sampleRate, mode: "wav" };
}

test("buildAnalysisPlan returns fixed chunk tasks", () => {
  const { buildAnalysisPlan } = runtimeExports;
  const plan = buildAnalysisPlan(8, 2, { chunkSec: 1 });

  assert.deepEqual(plan, [
    { start: 0, end: 2 },
    { start: 2, end: 4 },
    { start: 4, end: 6 },
    { start: 6, end: 8 },
  ]);
});

test("model io indexes CFP tensors as B,C,F,T", async () => {
  const { MODEL_IO, collectBatchPredictions } = coreInferenceExports;
  const outData = new Float32Array(MODEL_IO.BATCH * 361 * MODEL_IO.T_PER_BATCH);
  for (let t = 0; t < MODEL_IO.T_PER_BATCH; t += 1) {
    outData[0 * 361 * MODEL_IO.T_PER_BATCH + 12 * MODEL_IO.T_PER_BATCH + t] =
      10;
  }

  const { batchArgmax, batchConfidence } = await collectBatchPredictions({
    outTensor: {
      data: outData,
      dims: [MODEL_IO.BATCH, 361, MODEL_IO.T_PER_BATCH],
    },
    chunk: [{ validT: MODEL_IO.T_PER_BATCH }],
  });

  assert.equal(batchArgmax.length, MODEL_IO.T_PER_BATCH);
  assert.equal(batchConfidence.length, MODEL_IO.T_PER_BATCH);
  assert.ok(batchArgmax.every((value) => value === 12));
  assert.ok(batchConfidence.every((value) => value >= 0 && value <= 1));
});

test("analyzer emits phase events through the whole flow", async () => {
  const { AnalysisPhase, createAnalyzer } = runtimeExports;
  const decodedAudio = await decodeWavPcm16(realAudioPath);
  const audioManager = {
    async setAudio() {
      return decodedAudio;
    },
    getAudio() {
      return decodedAudio;
    },
    getPcmChunk(start, end) {
      return decodedAudio.pcm.slice(start, end);
    },
  };
  const cfpBatch = {
    data: new Float32Array([1]),
    shape: new Int32Array([1]),
  };
  const inferenceResult = {
    totalArgmax: [1],
    totalConfidence: [0.9],
    visibleArgmax: [1],
    visibleConfidence: [0.9],
    totalExpectedFrames: 1,
    totalBatchCount: 1,
  };
  const analyzer = createAnalyzer({
    audioManager,
    cfpManager: {
      async process() {
        return {
          fileKey: "demo-key",
          batches: [cfpBatch],
          allBatches: [cfpBatch],
          complete: true,
        };
      },
    },
    inferenceManager: {
      async process() {
        return inferenceResult;
      },
    },
  });

  const phases = [];
  let outputData = null;
  analyzer.subscribe((event) => {
    phases.push(event.phase);
    if (event.phase === AnalysisPhase.OUTPUT) {
      outputData = event.data;
    }
  });

  await analyzer.setAudio({
    source: { kind: "buffer", buffer: new ArrayBuffer(8), label: "demo" },
    model: { name: "demo-model" },
  });
  await analyzer.step();

  assert.deepEqual(phases, [
    AnalysisPhase.AUDIO,
    AnalysisPhase.CFP,
    AnalysisPhase.INFERENCE,
    AnalysisPhase.OUTPUT,
  ]);
  assert.ok(outputData);
  assert.equal(outputData.audio.fs, 44100);
  assert.equal(outputData.audio.mode, "wav");
  assert.equal(outputData.audio.pcm.length, decodedAudio.pcm.length);
  assert.equal(outputData.cfp.length, 1);
  assert.deepEqual(outputData.inference, inferenceResult);
});
