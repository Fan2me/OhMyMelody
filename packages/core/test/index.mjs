import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { IndexedDBStore } from "../dist/cache/base.js";
import { CFPIndexedDBCache } from "../dist/cache/cfp.js";
import { createCFPWorkerManager } from "../dist/cfp/worker-manager.js";
import { CORE_INFERENCE_WORKER_MODULE_URL } from "../dist/inference/inference.js";

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

test("core cfp package entrypoint defines the worker module URL", async () => {
  const source = await readFile(new URL("../dist/cfp/index.js", import.meta.url), "utf8");
  assert.match(source, /CORE_CFP_WORKER_MODULE_URL/);
  assert.match(source, /\.\/worker\.js\?worker/);
});

test("core cfp package entrypoint defines the cfp script URL", async () => {
  const source = await readFile(new URL("../dist/cfp/index.js", import.meta.url), "utf8");
  assert.match(source, /CORE_CFP_SCRIPT_URL/);
  assert.match(source, /\.\.\/\.\.\/cfp\.py/);
});

test("core inference package entrypoint defines the worker module URL", async () => {
  assert.match(CORE_INFERENCE_WORKER_MODULE_URL, /\/inference\/worker\.js$/);
});

test("CFPIndexedDBCache keeps CFP key helpers private", () => {
  const cache = new CFPIndexedDBCache({
    normalizeCFPBatches: () => [],
  });

  assert.equal("getCFPCacheChunkKey" in cache, false);
  assert.equal("getCFPCacheMetaKey" in cache, false);
  assert.equal("isCFPCacheKeyMatch" in cache, false);
});

test("IndexedDBStore skips opening the database for empty write/delete batches", async () => {
  const store = new IndexedDBStore({
    dbName: "TestCache",
    storeNames: ["entries"],
  });
  let opened = 0;
  store.open = async () => {
    opened += 1;
    throw new Error("open should not be called");
  };

  const putResult = await store.putMany("entries", []);
  const deleteResult = await store.deleteMany("entries", []);

  assert.equal(putResult, true);
  assert.equal(deleteResult, 0);
  assert.equal(opened, 0);
});

test("CFPIndexedDBCache deletes by cacheBaseKey index", async () => {
  const cache = new CFPIndexedDBCache({
    normalizeCFPBatches: () => [],
  });
  cache.cfpStore = {
    async deleteByIndex(storeName, indexName, indexValue) {
      assert.equal(storeName, "cfp");
      assert.equal(indexName, "cacheBaseKey");
      assert.equal(indexValue, "demo::runtime-v2");
      return 3;
    },
  };

  const deleted = await cache.deleteCFPCacheEntriesForKey("demo::runtime-v2");

  assert.equal(deleted, 3);
});

test("CFPIndexedDBCache enumerates meta entries through the kind index", async () => {
  const cache = new CFPIndexedDBCache({
    normalizeCFPBatches: () => [],
  });
  let listKeysCalled = false;
  cache.cfpStore = {
    async getAllByIndex(storeName, indexName, indexValue) {
      assert.equal(storeName, "cfp");
      assert.equal(indexName, "kind");
      assert.equal(indexValue, "chunked");
      return [
        {
          cacheBaseKey: "demo::runtime-v2",
          version: 3,
          kind: "chunked",
          chunkCount: 0,
          complete: true,
          updatedAt: 1,
          lastAccessed: 1,
          byteSize: 64,
        },
      ];
    },
    async listKeys() {
      listKeysCalled = true;
      return [];
    },
  };

  const entries = await cache.listCFPCacheMetaEntries();

  assert.equal(entries.length, 1);
  assert.equal(entries[0]?.key, "demo::runtime-v2");
  assert.equal(listKeysCalled, false);
});

test("CFPIndexedDBCache reads grouped cache records through the cacheBaseKey index", async () => {
  const cache = new CFPIndexedDBCache({
    normalizeCFPBatches: (rawBatches) => rawBatches,
  });
  cache.cfpStore = {
    async getAllByIndex(storeName, indexName, indexValue) {
      assert.equal(storeName, "cfp");
      assert.equal(indexName, "cacheBaseKey");
      assert.equal(indexValue, "demo::runtime-v2");
      return [
        {
          cacheBaseKey: "demo::runtime-v2",
          version: 3,
          kind: "chunked",
          chunkCount: 1,
          complete: true,
          updatedAt: 1,
          lastAccessed: 1,
          byteSize: 64,
        },
        {
          cacheBaseKey: "demo::runtime-v2",
          version: 2,
          kind: "chunk",
          index: 0,
          start: 0,
          end: 128,
          data: new Float32Array([1, 2]),
          shape: new Int32Array([1, 1, 2]),
        },
      ];
    },
  };

  const batches = await cache.getCFPCache("demo::runtime-v2");

  assert.equal(batches?.length, 1);
  assert.equal(batches?.[0]?.index, 0);
});
