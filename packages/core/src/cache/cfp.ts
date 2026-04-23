import { IndexedDBStore } from "./base.js";

export interface CFPBatch {
  data: Float32Array;
  shape: ArrayLike<number>;
  start?: number | null;
  end?: number | null;
  index?: number;
}

function toPositiveFinite(value: unknown, fallback: number): number {
  const next = Number(value);
  return Number.isFinite(next) && next > 0 ? next : fallback;
}

function toNonNegativeFinite(value: unknown, fallback: number): number {
  const next = Number(value);
  return Number.isFinite(next) && next >= 0 ? next : fallback;
}

export function countCFPBatchFrames(
  npyBatches: readonly CFPBatch[] = [],
): number {
  let totalFrames = 0;
  for (const batch of Array.isArray(npyBatches) ? npyBatches : []) {
    const shape = batch?.shape;
    const frameCount = shape ? Number(shape[2]) : 0;
    if (Number.isFinite(frameCount) && frameCount > 0) {
      totalFrames += frameCount;
    }
  }
  return totalFrames;
}

export function estimateDurationSecFromCFPBatches(
  npyBatches: readonly CFPBatch[] = [],
  frameSec = 0.01,
  totalFrames = 0,
): number {
  const safeFrameSec = Math.max(0, Number(frameSec) || 0.01);
  if (Number.isFinite(totalFrames) && totalFrames > 0) {
    return totalFrames * safeFrameSec;
  }
  return countCFPBatchFrames(npyBatches) * safeFrameSec;
}

interface CFPCacheMeta {
  cacheBaseKey: string;
  version: number;
  kind: "chunked";
  chunkCount: number;
  complete: boolean;
  updatedAt: number;
  lastAccessed: number;
  byteSize: number;
}

interface CFPCacheChunkRecord {
  cacheBaseKey: string;
  version: number;
  kind: "chunk";
  index: number;
  start: number | null;
  end: number | null;
  data: Float32Array;
  shape: ArrayLike<number>;
}

type CFPCacheStoreValue = CFPCacheChunkRecord | CFPCacheMeta;

function estimateCFPCacheChunkBytes(chunk: CFPCacheChunkRecord): number {
  const dataBytes =
    chunk?.data instanceof Float32Array ? chunk.data.byteLength : 0;
  const shapeLength = Number(
    chunk?.shape && typeof chunk.shape.length === "number"
      ? chunk.shape.length
      : 0,
  );
  const shapeBytes =
    Number.isFinite(shapeLength) && shapeLength > 0
      ? Math.floor(shapeLength) * 8
      : 0;
  return dataBytes + shapeBytes + 64;
}

function getCFPCacheMetaKey(key: string): string {
  return `${key}::meta`;
}

function getCFPCacheChunkKey(key: string, index: number): string {
  return `${key}::chunk::${String(index).padStart(6, "0")}`;
}

function normalizeCacheBaseKey(key: string): string {
  return String(key || "").trim();
}

function normalizeCFPCacheChunkRecord(
  normalizeCFPBatches: (rawBatches: unknown) => CFPBatch[],
  cacheBaseKey: string,
  rawChunk: unknown,
  fallbackIndex = 0,
): CFPCacheChunkRecord | null {
  const normalized = normalizeCFPBatches([rawChunk]);
  if (!normalized.length) {
    return null;
  }

  const chunk = normalized[0];
  if (!chunk) {
    return null;
  }

  const rawChunkRecord =
    rawChunk && typeof rawChunk === "object"
      ? (rawChunk as { index?: unknown; start?: unknown; end?: unknown })
      : {};
  const safeIndex =
    Number.isInteger(rawChunkRecord.index) && Number(rawChunkRecord.index) >= 0
      ? Number(rawChunkRecord.index)
      : Math.max(0, Math.floor(toPositiveFinite(fallbackIndex, 0)));
  const start = Number.isFinite(rawChunkRecord.start)
    ? Math.floor(Number(rawChunkRecord.start))
    : null;
  const end = Number.isFinite(rawChunkRecord.end)
    ? Math.floor(Number(rawChunkRecord.end))
    : null;

  return {
    cacheBaseKey,
    version: 2,
    kind: "chunk",
    index: safeIndex,
    start,
    end,
    data: chunk.data,
    shape: chunk.shape,
  };
}

function isCFPCacheMeta(record: unknown): record is CFPCacheMeta {
  return !!record && typeof record === "object" && (record as { kind?: unknown }).kind === "chunked";
}

function isCFPCacheChunkRecord(record: unknown): record is CFPCacheChunkRecord {
  return !!record && typeof record === "object" && (record as { kind?: unknown }).kind === "chunk";
}

interface CFPCacheRecordGroup {
  cacheBaseKey: string;
  chunks: CFPCacheChunkRecord[];
  meta: CFPCacheMeta | null;
}

export class CFPIndexedDBCache {
  private static readonly MAX_CFP_BYTES = 1 * 1024 * 1024 * 1024;
  private readonly normalizeCFPBatches: (rawBatches: unknown) => CFPBatch[];
  private readonly cfpStore: IndexedDBStore;

  constructor({
    normalizeCFPBatches,
  }: {
    normalizeCFPBatches: (rawBatches: unknown) => CFPBatch[];
  }) {
    if (typeof normalizeCFPBatches !== "function") {
      throw new Error(
        "CFPIndexedDBCache requires a normalizeCFPBatches function",
      );
    }
    this.normalizeCFPBatches = normalizeCFPBatches;
    this.cfpStore = new IndexedDBStore({
      dbName: "CFPFeatureCache",
      version: 3,
      storeNames: [
        {
          name: "cfp",
          indexes: [
            {
              name: "cacheBaseKey",
              keyPath: "cacheBaseKey",
            },
            {
              name: "kind",
              keyPath: "kind",
            },
          ],
          onUpgrade(store, context) {
            if (context.oldVersion < 2) {
              store.clear();
            }
          },
        },
      ],
    });
  }

  buildCFPCacheMeta({
    cacheBaseKey = "",
    chunkCount = 0,
    complete = false,
    updatedAt = Date.now(),
    lastAccessed = updatedAt,
    byteSize = 0,
  }: {
    cacheBaseKey?: string;
    chunkCount?: number;
    complete?: boolean;
    updatedAt?: number;
    lastAccessed?: number;
    byteSize?: number;
  } = {}): CFPCacheMeta {
    const safeCacheBaseKey = normalizeCacheBaseKey(cacheBaseKey);
    return {
      cacheBaseKey: safeCacheBaseKey,
      version: 3,
      kind: "chunked",
      chunkCount: Math.max(0, Math.floor(toNonNegativeFinite(chunkCount, 0))),
      complete: complete === true,
      updatedAt: Math.floor(toNonNegativeFinite(updatedAt, Date.now())),
      lastAccessed: Math.floor(toNonNegativeFinite(lastAccessed, Date.now())),
      byteSize: Math.max(0, Math.floor(toNonNegativeFinite(byteSize, 0))),
    };
  }

  private normalizeCFPCacheMeta(rawMeta: unknown): CFPCacheMeta | null {
    if (!rawMeta || typeof rawMeta !== "object") {
      return null;
    }

    const typed = rawMeta as {
      cacheBaseKey?: unknown;
      chunkCount?: unknown;
      complete?: unknown;
      updatedAt?: unknown;
      lastAccessed?: unknown;
      byteSize?: unknown;
    };

    const updatedAt = Math.floor(
      toNonNegativeFinite(typed.updatedAt, Date.now()),
    );
    const lastAccessed = Math.floor(
      toNonNegativeFinite(typed.lastAccessed, updatedAt),
    );
    const byteSize = Math.max(
      0,
      Math.floor(toNonNegativeFinite(typed.byteSize, 0)),
    );
    return this.buildCFPCacheMeta({
      cacheBaseKey: normalizeCacheBaseKey(String(typed.cacheBaseKey || "")),
      chunkCount: Math.max(
        0,
        Math.floor(toNonNegativeFinite(typed.chunkCount, 0)),
      ),
      complete: typed.complete === true,
      updatedAt,
      lastAccessed,
      byteSize,
    });
  }

  private normalizeCFPCacheChunks(
    cacheBaseKey: string,
    rawEntries: readonly unknown[],
  ): CFPCacheChunkRecord[] {
    const normalizedChunks: CFPCacheChunkRecord[] = [];
    for (let i = 0; i < rawEntries.length; i += 1) {
      const rawEntry = rawEntries[i];
      if (!isCFPCacheChunkRecord(rawEntry)) {
        continue;
      }
      const normalizedChunk = normalizeCFPCacheChunkRecord(
        this.normalizeCFPBatches,
        cacheBaseKey,
        rawEntry,
        rawEntry.index,
      );
      if (!normalizedChunk) {
        continue;
      }
      normalizedChunks.push(normalizedChunk);
    }
    normalizedChunks.sort((a, b) => a.index - b.index);
    return normalizedChunks;
  }

  private async loadCFPCacheRecordGroup(
    key: string,
  ): Promise<CFPCacheRecordGroup> {
    const cacheBaseKey = normalizeCacheBaseKey(key);
    if (!cacheBaseKey) {
      return {
        cacheBaseKey,
        meta: null,
        chunks: [],
      };
    }
    const rawEntries = await this.cfpStore.getAllByIndex<unknown>(
      "cfp",
      "cacheBaseKey",
      cacheBaseKey,
    );
    return {
      cacheBaseKey,
      meta: this.normalizeCFPCacheMeta(
        rawEntries.find((entry) => isCFPCacheMeta(entry)) ?? null,
      ),
      chunks: this.normalizeCFPCacheChunks(cacheBaseKey, rawEntries),
    };
  }

  private async putCFPCacheMeta(
    key: string,
    metaInput: {
      chunkCount: number;
      complete: boolean;
      updatedAt?: number;
      lastAccessed?: number;
      byteSize: number;
    },
  ): Promise<CFPCacheMeta> {
    const nextMeta = this.buildCFPCacheMeta({
      cacheBaseKey: key,
      chunkCount: metaInput.chunkCount,
      complete: metaInput.complete,
      updatedAt: metaInput.updatedAt ?? Date.now(),
      lastAccessed: metaInput.lastAccessed ?? Date.now(),
      byteSize: metaInput.byteSize,
    });
    await this.cfpStore.put("cfp", getCFPCacheMetaKey(key), nextMeta);
    return nextMeta;
  }

  async deleteCFPCacheEntriesForKey(key: string): Promise<number> {
    const cacheBaseKey = normalizeCacheBaseKey(key);
    if (!cacheBaseKey) {
      return 0;
    }
    return await this.cfpStore.deleteByIndex(
      "cfp",
      "cacheBaseKey",
      cacheBaseKey,
    );
  }

  private async listCFPCacheMetaEntries(): Promise<
    Array<{ key: string; meta: CFPCacheMeta }>
  > {
    const rawMetas = await this.cfpStore.getAllByIndex<unknown>(
      "cfp",
      "kind",
      "chunked",
    );
    if (!rawMetas.length) {
      return [];
    }
    const entries: Array<{ key: string; meta: CFPCacheMeta }> = [];
    for (let i = 0; i < rawMetas.length; i += 1) {
      const meta = this.normalizeCFPCacheMeta(rawMetas[i]);
      if (!meta) {
        continue;
      }
      const cacheBaseKey = normalizeCacheBaseKey(meta.cacheBaseKey);
      if (!cacheBaseKey) {
        continue;
      }
      const byteSize =
        meta.byteSize > 0
          ? meta.byteSize
          : await this.estimateCFPCacheByteSizeForKey(
              cacheBaseKey,
              meta.chunkCount,
            );
      entries.push({
        key: cacheBaseKey,
        meta: {
          ...meta,
          cacheBaseKey,
          byteSize,
        },
      });
    }
    return entries;
  }

  private async estimateCFPCacheByteSizeForKey(
    key: string,
    chunkCount: number,
  ): Promise<number> {
    const safeChunkCount = Math.max(
      0,
      Math.floor(toNonNegativeFinite(chunkCount, 0)),
    );
    if (safeChunkCount === 0) {
      return 0;
    }
    const { chunks } = await this.loadCFPCacheRecordGroup(key);
    let byteSize = 0;
    for (let i = 0; i < chunks.length; i += 1) {
      const chunk = chunks[i];
      if (!chunk) {
        continue;
      }
      byteSize += estimateCFPCacheChunkBytes(chunk);
    }
    return byteSize;
  }

  private async touchCFPCacheMeta(key: string): Promise<void> {
    try {
      const rawMeta = await this.cfpStore.get<unknown>(
        "cfp",
        getCFPCacheMetaKey(key),
      );
      const meta = this.normalizeCFPCacheMeta(rawMeta);
      if (!meta) {
        return;
      }
      await this.putCFPCacheMeta(key, {
        chunkCount: meta.chunkCount,
        complete: meta.complete,
        updatedAt: meta.updatedAt,
        lastAccessed: Date.now(),
        byteSize: meta.byteSize,
      });
    } catch {}
  }

  private async refreshCFPCacheMeta(
    key: string,
    {
      complete = null,
    }: {
      complete?: boolean | null;
    } = {},
  ): Promise<CFPCacheMeta | null> {
    const { meta, chunks } = await this.loadCFPCacheRecordGroup(key);
    const expectedChunkCount = Math.max(
      0,
      Math.floor(toNonNegativeFinite(meta?.chunkCount ?? 0, 0)),
    );
    let byteSize = 0;
    for (let i = 0; i < chunks.length; i += 1) {
      const chunk = chunks[i];
      if (!chunk) {
        continue;
      }
      byteSize += estimateCFPCacheChunkBytes(chunk);
    }

    return await this.putCFPCacheMeta(key, {
      chunkCount: Math.max(expectedChunkCount, chunks.length),
      complete: complete ?? meta?.complete ?? false,
      byteSize,
    });
  }

  private async enforceCFPCacheBudget(
    currentKey: string | null = null,
  ): Promise<void> {
    const entries = await this.listCFPCacheMetaEntries();
    if (!entries.length) {
      return;
    }

    let totalBytes = 0;
    for (const entry of entries) {
      totalBytes += Math.max(0, Math.floor(entry.meta.byteSize || 0));
    }
    if (totalBytes <= CFPIndexedDBCache.MAX_CFP_BYTES) {
      return;
    }

    const purgeCandidates = entries
      .filter((entry) => entry.key !== currentKey)
      .sort((a, b) => {
        const accessDelta = a.meta.lastAccessed - b.meta.lastAccessed;
        if (accessDelta !== 0) return accessDelta;
        const updateDelta = a.meta.updatedAt - b.meta.updatedAt;
        if (updateDelta !== 0) return updateDelta;
        return a.key.localeCompare(b.key);
      });

    for (const entry of purgeCandidates) {
      if (totalBytes <= CFPIndexedDBCache.MAX_CFP_BYTES) {
        break;
      }
      await this.deleteCFPCacheEntriesForKey(entry.key);
      totalBytes -= Math.max(0, Math.floor(entry.meta.byteSize || 0));
    }
  }

  async getCFPCache(
    key: string,
    { allowPartial = false }: { allowPartial?: boolean } = {},
  ): Promise<CFPBatch[] | null> {
    const { meta, chunks } = await this.loadCFPCacheRecordGroup(key);
    if (!meta && !chunks.length) {
      return null;
    }
    if (!meta) {
      return null;
    }

    const chunkCount = Math.max(
      0,
      Math.floor(toPositiveFinite(meta.chunkCount, 0)),
    );
    if (!allowPartial && meta.complete !== true) {
      return null;
    }
    if (chunkCount === 0) {
      return [];
    }

    if (!chunks.length) {
      return null;
    }
    if (!allowPartial && chunks.length !== chunkCount) {
      return null;
    }

    await this.touchCFPCacheMeta(key);
    return chunks.map(({ data, shape, start, end, index }) => ({
      data,
      shape,
      start,
      end,
      index,
    }));
  }

  async setCFPCache(
    key: string,
    value: unknown,
  ): Promise<{ chunkCount: number; complete: boolean }> {
    const normalizedBatches = this.normalizeCFPBatches(value);
    await this.deleteCFPCacheEntriesForKey(key);
    if (!normalizedBatches.length) {
      return { chunkCount: 0, complete: false };
    }

    const entries: Array<{ key: string; value: CFPCacheStoreValue }> =
      normalizedBatches
        .map((batch, index) => ({
          key: getCFPCacheChunkKey(key, index),
          value: normalizeCFPCacheChunkRecord(
            this.normalizeCFPBatches,
            key,
            batch,
            index,
          ),
        }))
        .filter(
          (entry): entry is { key: string; value: CFPCacheChunkRecord } =>
            entry.value !== null,
        );

    entries.push({
      key: getCFPCacheMetaKey(key),
      value: this.buildCFPCacheMeta({
        cacheBaseKey: key,
        chunkCount: normalizedBatches.length,
        complete: true,
      }),
    });

    await this.cfpStore.putMany("cfp", entries);
    await this.refreshCFPCacheMeta(key, { complete: true });
    await this.enforceCFPCacheBudget(key);
    return {
      chunkCount: normalizedBatches.length,
      complete: true,
    };
  }

  async appendCFPCacheChunk(
    key: string,
    chunk: unknown,
    {
      index,
      reset = false,
      complete = false,
      expectedChunkCount = null,
    }: {
      index?: number;
      reset?: boolean;
      complete?: boolean;
      expectedChunkCount?: number | null;
    } = {},
  ): Promise<{ index: number; chunkCount: number; complete: boolean }> {
    const nextIndex = Number(index);
    const safeIndex =
      Number.isInteger(nextIndex) && nextIndex >= 0 ? nextIndex : null;
    if (safeIndex === null) {
      throw new Error(
        "appendCFPCacheChunk requires a non-negative integer index",
      );
    }

    const normalizedChunk = normalizeCFPCacheChunkRecord(
      this.normalizeCFPBatches,
      key,
      chunk,
      safeIndex,
    );
    if (!normalizedChunk) {
      throw new Error(`invalid CFP chunk at index=${safeIndex}`);
    }
    if (reset) {
      await this.deleteCFPCacheEntriesForKey(key);
    }

    const nextExpectedChunkCount = Number(expectedChunkCount);
    const chunkCount = Math.max(
      safeIndex + 1,
      Number.isInteger(nextExpectedChunkCount) && nextExpectedChunkCount >= 0
        ? nextExpectedChunkCount
        : 0,
    );
    await this.cfpStore.putMany<CFPCacheStoreValue>("cfp", [
      {
        key: getCFPCacheChunkKey(key, safeIndex),
        value: normalizedChunk,
      },
      {
        key: getCFPCacheMetaKey(key),
        value: this.buildCFPCacheMeta({
          cacheBaseKey: key,
          chunkCount,
          complete,
        }),
      },
    ]);
    if (complete === true) {
      await this.refreshCFPCacheMeta(key, { complete: true });
      await this.enforceCFPCacheBudget(key);
    }
    return {
      index: safeIndex,
      chunkCount,
      complete: complete === true,
    };
  }

  async finalizeCFPCache(
    key: string,
    { chunkCount = null }: { chunkCount?: number | null } = {},
  ): Promise<{ chunkCount: number; complete: boolean }> {
    const existingMeta = await this.cfpStore.get<CFPCacheMeta>(
      "cfp",
      getCFPCacheMetaKey(key),
    );
    const nextChunkCount = Number(chunkCount);
    const safeChunkCount = Math.max(
      0,
      Number.isInteger(nextChunkCount) && nextChunkCount >= 0
        ? nextChunkCount
        : Math.floor(toPositiveFinite(existingMeta?.chunkCount, 0)),
    );
    if (safeChunkCount === 0 && !existingMeta) {
      return {
        chunkCount: 0,
        complete: false,
      };
    }
    await this.refreshCFPCacheMeta(key, { complete: true });
    await this.enforceCFPCacheBudget(key);
    return {
      chunkCount: safeChunkCount,
      complete: true,
    };
  }

  async clearCFPCache() {
    await this.cfpStore.clear("cfp", "CFP cache");
    return { dbName: "CFPFeatureCache", storeName: "cfp", cleared: true };
  }
}

export async function clearAllIndexedDBCaches(): Promise<{
  cfpCleared: boolean;
  predictionCleared: boolean;
}> {
  const cfp = new CFPIndexedDBCache({
    normalizeCFPBatches: () => [],
  });
  const prediction = new IndexedDBStore({
    dbName: "PredictionCache",
    version: 1,
    storeNames: ["pred"],
  });
  const [cfpResult, predictionCleared] = await Promise.all([
    cfp.clearCFPCache(),
    prediction.clear("pred", "prediction cache"),
  ]);
  return {
    cfpCleared: !!cfpResult?.cleared,
    predictionCleared: !!predictionCleared,
  };
}
