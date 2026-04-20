import assert from "node:assert/strict";
import test from "node:test";

import { createCFPWorkerManager } from "../dist/cfp/worker-manager.js";

class FakeWorker {
  constructor() {
    this.listeners = new Map();
    this.terminated = false;
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

  postMessage(message) {
    if (message?.cmd !== "init") {
      return;
    }
    queueMicrotask(() => {
      for (const listener of this.listeners.get("message") ?? []) {
        listener({ data: { cmd: "inited" } });
      }
    });
  }

  terminate() {
    this.terminated = true;
  }
}

test("CFP worker manager prewarms a worker instance", async () => {
  const previousWorker = globalThis.Worker;
  let created = 0;
  globalThis.Worker = FakeWorker;

  try {
    const manager = createCFPWorkerManager({
      createWorkerInstance: () => {
        created += 1;
        return new FakeWorker();
      },
      resolveCFPScriptUrl: () => "/packages/core/cfp.py",
      resolvePyodideIndexURL: () => "/pyodide/",
      resolvePyodideScriptUrl: () => "/pyodide/pyodide.js",
    });

    const ready = await manager.prewarmCFPWorker({
      workerInitTimeoutMs: 1000,
    });

    assert.equal(ready, true);
    assert.equal(created, 1);
    assert.ok(manager.getPrewarmedWorker());

    const readyAgain = await manager.prewarmCFPWorker({
      workerInitTimeoutMs: 1000,
    });

    assert.equal(readyAgain, true);
    assert.equal(created, 1);
  } finally {
    globalThis.Worker = previousWorker;
  }
});
