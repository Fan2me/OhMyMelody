import assert from "node:assert/strict";
import test from "node:test";

const previousSelf = globalThis.self;
globalThis.self = globalThis;
const runtimeExports = await import("../dist/analyzer.js");
const coreInferenceExports = await import("../../core/dist/inference/inference.js");
globalThis.self = previousSelf;

function createDecodedAudio(sampleCount, fs = 44100) {
  const pcm = new Float32Array(sampleCount);
  for (let index = 0; index < sampleCount; index += 1) {
    pcm[index] = index / Math.max(1, sampleCount);
  }
  return { pcm, fs, mode: "mock" };
}

function createInput(label = "demo") {
  return {
    source: { kind: "buffer", buffer: new ArrayBuffer(8), label },
    model: { name: "demo-model" },
  };
}

function waitForOutput(analyzer, phase) {
  return new Promise((resolve) => {
    const unsubscribe = analyzer.subscribe((event) => {
      if (event.phase !== phase) {
        return;
      }
      unsubscribe();
      resolve(event);
    });
  });
}

test("buildAnalysisPlan uses warmup chunks before steady chunks", () => {
  const { buildAnalysisPlan } = runtimeExports;
  const warmupChunkSize = 56448;
  const steadyChunkSize = 225792;
  const totalSamples = warmupChunkSize * 3 + steadyChunkSize * 2 + 100;

  const plan = buildAnalysisPlan(totalSamples, 44100);

  assert.deepEqual(plan, [
    { start: 0, end: 56448 },
    { start: 56448, end: 112896 },
    { start: 112896, end: 169344 },
    { start: 169344, end: 395136 },
    { start: 395136, end: 620928 },
    { start: 620928, end: 621028 },
  ]);
});

test("model io indexes CFP tensors as B,C,F,T and normalizes argmax", async () => {
  const { MODEL_IO, collectBatchPredictions } = coreInferenceExports;
  const outData = new Float32Array(MODEL_IO.BATCH * 361 * MODEL_IO.T_PER_BATCH);
  for (let t = 0; t < MODEL_IO.T_PER_BATCH; t += 1) {
    outData[12 * MODEL_IO.T_PER_BATCH + t] = 10;
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
  assert.ok(batchArgmax.every((value) => value === 11));
  assert.ok(batchConfidence.every((value) => value >= 0 && value <= 1));
});

test("analyzer emits phase events across multiple plan steps", async () => {
  const { AnalysisPhase, createAnalyzer } = runtimeExports;
  const decodedAudio = createDecodedAudio(56448 * 3 + 100);
  const cfpCalls = [];
  const inferenceCalls = [];
  const analyzer = createAnalyzer({
    audioManager: {
      async setAudio() {
        return decodedAudio;
      },
      getAudio() {
        return decodedAudio;
      },
      getPcmChunk(start, end) {
        return decodedAudio.pcm.slice(start, end);
      },
    },
    cfpManager: {
      async process({ batchOffset, segment, complete }) {
        const batch = {
          data: new Float32Array([(batchOffset ?? 0) + 1]),
          shape: new Int32Array([1]),
        };
        cfpCalls.push({ previousCount: batchOffset ?? 0, size: segment.pcm.length, complete });
        return {
          kind: "segment",
          fileKey: "demo-key",
          batches: [batch],
          complete,
        };
      },
    },
    inferenceManager: {
      reset() {},
      async process({ batches, complete }) {
        inferenceCalls.push({ count: batches.length, complete });
        return {
          totalArgmax: batches.map((_, index) => index),
          totalConfidence: batches.map(() => 0.9),
        };
      },
    },
  });

  const phases = [];
  const outputDone = waitForOutput(analyzer, AnalysisPhase.OUTPUT);
  analyzer.subscribe((event) => {
    phases.push(`${event.phase}:${event.index}`);
  });

  await analyzer.setAudio(createInput());
  await analyzer.step();
  await analyzer.step();
  await analyzer.step();
  await analyzer.step();
  const outputEvent = await outputDone;

  assert.deepEqual(
    phases,
    [
      "audio:0",
      "cfp:0",
      "inference:0",
      "cfp:1",
      "inference:1",
      "cfp:2",
      "inference:2",
      "cfp:3",
      "inference:3",
      "output:4",
    ],
  );
  assert.deepEqual(
    cfpCalls.map((call) => call.size),
    [56448, 56448, 56448, 100],
  );
  assert.deepEqual(
    inferenceCalls,
    [
      { count: 1, complete: false },
      { count: 1, complete: false },
      { count: 1, complete: false },
      { count: 1, complete: true },
    ],
  );
  assert.equal(outputEvent.data.audio.fs, 44100);
  assert.equal(outputEvent.data.audio.mode, "mock");
  assert.equal(outputEvent.data.audio.pcm.length, decodedAudio.pcm.length);
  assert.equal(outputEvent.data.cfp.length, 4);
});

test("analyzer reuses completed CFP batches for the same file key", async () => {
  const { AnalysisPhase, createAnalyzer } = runtimeExports;
  const decodedAudio = createDecodedAudio(32000);
  let cfpProcessCount = 0;
  const inferenceBatchCounts = [];
  const analyzer = createAnalyzer({
    audioManager: {
      async setAudio() {
        return decodedAudio;
      },
      getAudio() {
        return decodedAudio;
      },
      getPcmChunk(start, end) {
        return decodedAudio.pcm.slice(start, end);
      },
    },
    cfpManager: {
      async process({ complete }) {
        cfpProcessCount += 1;
        const batch = {
          data: new Float32Array([1]),
          shape: new Int32Array([1]),
        };
        return {
          kind: "segment",
          fileKey: "shared-key",
          batches: [batch],
          complete,
        };
      },
    },
    inferenceManager: {
      reset() {},
      async process({ batches }) {
        inferenceBatchCounts.push(batches.length);
        return {
          totalArgmax: [1],
          totalConfidence: [0.9],
        };
      },
    },
  });

  let firstOutput = waitForOutput(analyzer, AnalysisPhase.OUTPUT);
  await analyzer.setAudio(createInput("shared"));
  await analyzer.step();
  await firstOutput;

  const secondPhases = [];
  analyzer.subscribe((event) => {
    secondPhases.push(event.phase);
  });
  const secondOutput = waitForOutput(analyzer, AnalysisPhase.OUTPUT);
  await analyzer.setAudio(createInput("shared"));
  await analyzer.step();
  await secondOutput;

  assert.equal(cfpProcessCount, 1);
  assert.deepEqual(inferenceBatchCounts, [1, 1]);
  assert.ok(!secondPhases.includes(AnalysisPhase.CFP));
});

test("analyzer progressively replays complete CFP batches when inference cache is cold", async () => {
  const { AnalysisPhase, createAnalyzer } = runtimeExports;
  const decodedAudio = createDecodedAudio(1000);
  const batchA = { data: new Float32Array([1]), shape: new Int32Array([1]) };
  const batchB = { data: new Float32Array([2]), shape: new Int32Array([1]) };
  const inferenceCalls = [];
  const inferenceEvents = [];
  const analyzer = createAnalyzer({
    audioManager: {
      async setAudio() {
        return decodedAudio;
      },
      getAudio() {
        return decodedAudio;
      },
      getPcmChunk(start, end) {
        return decodedAudio.pcm.slice(start, end);
      },
    },
    cfpManager: {
      async process() {
        return {
          kind: "cache-hit",
          fileKey: "cold-key",
          batches: [batchA, batchB],
          complete: true,
        };
      },
    },
    inferenceManager: {
      reset() {},
      async hasCache() {
        return false;
      },
      async process({ batches, complete }) {
        inferenceCalls.push({ values: batches.map((batch) => batch.data[0]), complete });
        return {
          totalArgmax: batches.map((batch) => batch.data[0]),
          totalConfidence: batches.map(() => 0.9),
        };
      },
    },
  });

  analyzer.subscribe((event) => {
    if (event.phase === AnalysisPhase.INFERENCE) {
      inferenceEvents.push(event.data.inference.totalArgmax);
    }
  });
  const outputDone = waitForOutput(analyzer, AnalysisPhase.OUTPUT);

  await analyzer.setAudio(createInput("cold"));
  await analyzer.step();
  await outputDone;

  assert.deepEqual(inferenceCalls, [
    { values: [1], complete: false },
    { values: [2], complete: true },
  ]);
  assert.deepEqual(inferenceEvents, [[1], [2]]);
});

test("analyzer skips progressive replay when inference cache is warm", async () => {
  const { AnalysisPhase, createAnalyzer } = runtimeExports;
  const decodedAudio = createDecodedAudio(1000);
  const batchA = { data: new Float32Array([1]), shape: new Int32Array([1]) };
  const batchB = { data: new Float32Array([2]), shape: new Int32Array([1]) };
  const inferenceCalls = [];
  const inferenceEvents = [];
  const analyzer = createAnalyzer({
    audioManager: {
      async setAudio() {
        return decodedAudio;
      },
      getAudio() {
        return decodedAudio;
      },
      getPcmChunk(start, end) {
        return decodedAudio.pcm.slice(start, end);
      },
    },
    cfpManager: {
      async process() {
        return {
          kind: "cache-hit",
          fileKey: "warm-key",
          batches: [batchA, batchB],
          complete: true,
        };
      },
    },
    inferenceManager: {
      reset() {},
      async hasCache() {
        return true;
      },
      async process({ batches, complete }) {
        inferenceCalls.push({ values: batches.map((batch) => batch.data[0]), complete });
        return {
          totalArgmax: batches.map((batch) => batch.data[0]),
          totalConfidence: batches.map(() => 0.9),
        };
      },
    },
  });

  analyzer.subscribe((event) => {
    if (event.phase === AnalysisPhase.INFERENCE) {
      inferenceEvents.push(event.data.inference.totalArgmax);
    }
  });
  const outputDone = waitForOutput(analyzer, AnalysisPhase.OUTPUT);

  await analyzer.setAudio(createInput("warm"));
  await analyzer.step();
  await outputDone;

  assert.deepEqual(inferenceCalls, [
    { values: [1, 2], complete: true },
  ]);
  assert.deepEqual(inferenceEvents, [[1, 2]]);
});

test("InferenceManager merges incremental results and resets on model change", async () => {
  const { InferenceManager } = runtimeExports;
  const previousWorker = globalThis.Worker;
  const initModels = [];

  class FakeInferenceWorker {
    constructor() {
      this.listeners = new Map();
      this.onmessage = null;
      this.onerror = null;
    }

    addEventListener(type, listener) {
      const current = this.listeners.get(type) ?? [];
      current.push(listener);
      this.listeners.set(type, current);
    }

    removeEventListener(type, listener) {
      const current = this.listeners.get(type) ?? [];
      this.listeners.set(
        type,
        current.filter((item) => item !== listener),
      );
    }

    emit(type, payload) {
      for (const listener of this.listeners.get(type) ?? []) {
        listener(payload);
      }
      if (type === "message" && typeof this.onmessage === "function") {
        this.onmessage(payload);
      }
      if (type === "error" && typeof this.onerror === "function") {
        this.onerror(payload);
      }
    }

    postMessage(message) {
      if (message?.cmd === "init") {
        initModels.push(message.modelName);
        queueMicrotask(() => {
          this.emit("message", { data: { cmd: "inited", provider: "fake" } });
        });
        return;
      }
      if (message?.cmd === "process") {
        const values = message.batches.map((batch) => batch.data[0]);
        queueMicrotask(() => {
          this.emit("message", {
            data: {
              cmd: "result",
              id: message.id,
              result: {
                totalArgmax: values,
                totalConfidence: values.map(() => 0.9),
              },
            },
          });
        });
      }
    }
  }

  globalThis.Worker = FakeInferenceWorker;

  try {
    const manager = new InferenceManager();
    const batch1 = { data: new Float32Array([1]), shape: new Int32Array([1]) };
    const batch2 = { data: new Float32Array([2]), shape: new Int32Array([1]) };
    const batch3 = { data: new Float32Array([3]), shape: new Int32Array([1]) };

    const first = await manager.process({
      batches: [batch1],
      modelName: "mamba_a.onnx",
      allowCache: false,
    });
    const second = await manager.process({
      batches: [batch2],
      modelName: "mamba_a.onnx",
      allowCache: false,
      complete: true,
    });
    const third = await manager.process({
      batches: [batch3],
      modelName: "mamba_b.onnx",
      allowCache: false,
    });

    assert.deepEqual(first.totalArgmax, [1]);
    assert.deepEqual(second.totalArgmax, [1, 2]);
    assert.deepEqual(third.totalArgmax, [3]);
    assert.deepEqual(initModels, ["mamba_a.onnx", "mamba_b.onnx"]);
  } finally {
    globalThis.Worker = previousWorker;
  }
});

test("InferenceManager reuses cached results before hitting worker process", async () => {
  const { InferenceManager } = runtimeExports;
  const previousWorker = globalThis.Worker;
  let processCalls = 0;

  class FakeInferenceWorker {
    constructor() {
      this.listeners = new Map();
      this.onmessage = null;
      this.onerror = null;
    }

    addEventListener(type, listener) {
      const current = this.listeners.get(type) ?? [];
      current.push(listener);
      this.listeners.set(type, current);
    }

    removeEventListener(type, listener) {
      const current = this.listeners.get(type) ?? [];
      this.listeners.set(
        type,
        current.filter((item) => item !== listener),
      );
    }

    emit(type, payload) {
      for (const listener of this.listeners.get(type) ?? []) {
        listener(payload);
      }
      if (type === "message" && typeof this.onmessage === "function") {
        this.onmessage(payload);
      }
    }

    postMessage(message) {
      if (message?.cmd === "init") {
        queueMicrotask(() => {
          this.emit("message", { data: { cmd: "inited", provider: "fake" } });
        });
        return;
      }
      if (message?.cmd === "process") {
        processCalls += 1;
      }
    }
  }

  globalThis.Worker = FakeInferenceWorker;

  try {
    const manager = new InferenceManager();
    manager.cache = {
      async getPredictionCache() {
        return {
          totalArgmax: Int32Array.from([7, 8]),
          totalConfidence: Float32Array.from([0.7, 0.8]),
        };
      },
    };

    const result = await manager.process({
      batches: [{ data: new Float32Array([1]), shape: new Int32Array([1]) }],
      modelName: "mamba_a.onnx",
      fileKey: "cached-demo",
      allowCache: true,
    });

    assert.deepEqual(result.totalArgmax, [7, 8]);
    assert.ok(Math.abs(result.totalConfidence[0] - 0.7) < 1e-6);
    assert.ok(Math.abs(result.totalConfidence[1] - 0.8) < 1e-6);
    assert.equal(processCalls, 0);
  } finally {
    globalThis.Worker = previousWorker;
  }
});
