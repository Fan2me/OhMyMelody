export interface CFPBatch {
  data: Float32Array;
  shape: ArrayLike<number>;
  start?: number | null;
  end?: number | null;
  index?: number;
}

export interface PredictionCacheEntry {
  totalArgmax: Int32Array;
  visibleArgmax: Int32Array;
  totalConfidence: Float32Array;
  visibleConfidence: Float32Array;
  totalExpectedFrames?: number;
  totalBatchCount?: number;
  complete?: boolean;
}

export interface IndexedDBStoreOptions {
  dbName: string;
  version?: number;
  storeNames?: readonly string[];
}

export interface IndexedDBEntry<TValue = unknown> {
  key: IDBValidKey;
  value: TValue;
}

function toPositiveFinite(value: unknown, fallback: number): number {
  const next = Number(value);
  return Number.isFinite(next) && next > 0 ? next : fallback;
}

function toNonNegativeFinite(value: unknown, fallback: number): number {
  const next = Number(value);
  return Number.isFinite(next) && next >= 0 ? next : fallback;
}

function estimateCFPCacheChunkBytes(chunk: CFPCacheChunkRecord): number {
  const dataBytes = chunk?.data instanceof Float32Array ? chunk.data.byteLength : 0;
  const shapeLength = Number(chunk?.shape && typeof chunk.shape.length === 'number' ? chunk.shape.length : 0);
  const shapeBytes = Number.isFinite(shapeLength) && shapeLength > 0 ? Math.floor(shapeLength) * 8 : 0;
  return dataBytes + shapeBytes + 64;
}

export class IndexedDBStore {
  private readonly dbName: string;
  private readonly version: number;
  private readonly storeNames: string[];

  constructor({
    dbName,
    version = 1,
    storeNames = [],
  }: IndexedDBStoreOptions) {
    this.dbName = String(dbName || '').trim();
    this.version = Math.max(1, Math.floor(Number(version) || 1));
    this.storeNames = Array.from(
      new Set(
        (Array.isArray(storeNames) ? storeNames : [])
          .map((name) => String(name || '').trim())
          .filter(Boolean),
      ),
    );
  }

  ensureIndexedDBAvailable(): void {
    if (typeof indexedDB === 'undefined' || !indexedDB) {
      throw new Error('IndexedDB is not supported in current environment');
    }
  }

  async open(): Promise<IDBDatabase> {
    this.ensureIndexedDBAvailable();
    if (!this.dbName) {
      throw new Error('IndexedDBStore requires dbName');
    }

    return await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db) {
          return;
        }
        for (const storeName of this.storeNames) {
          if (!db.objectStoreNames.contains(storeName)) {
            db.createObjectStore(storeName);
          }
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  close(db: IDBDatabase | null): void {
    try {
      if (db && typeof db.close === 'function') {
        db.close();
      }
    } catch {}
  }

  async withDb<TResult>(fn: (db: IDBDatabase) => Promise<TResult>): Promise<TResult> {
    const db = await this.open();
    try {
      return await fn(db);
    } finally {
      this.close(db);
    }
  }

  async get<TValue = unknown>(
    storeName: string,
    key: IDBValidKey,
  ): Promise<TValue | null> {
    const values = await this.getMany<TValue>(storeName, [key]);
    return values.length ? (values[0] ?? null) : null;
  }

  async getMany<TValue = unknown>(
    storeName: string,
    keys: readonly IDBValidKey[],
  ): Promise<Array<TValue | null>> {
    if (!Array.isArray(keys) || !keys.length) {
      return [];
    }

    return await this.withDb(async (db) =>
      await new Promise<Array<TValue | null>>((resolve) => {
        const values: Array<TValue | null> = new Array(keys.length).fill(null);
        let settled = false;
        const finish = (nextValues: Array<TValue | null>) => {
          if (settled) {
            return;
          }
          settled = true;
          resolve(nextValues);
        };

        try {
          const tx = db.transaction(storeName, 'readonly');
          const store = tx.objectStore(storeName);
          for (let i = 0; i < keys.length; i += 1) {
            const key = keys[i];
            if (typeof key === 'undefined') {
              continue;
            }
            const req = store.get(key);
            req.onsuccess = () => {
              values[i] = (req.result ?? null) as TValue | null;
            };
            req.onerror = () => {
              values[i] = null;
            };
          }
          tx.oncomplete = () => finish(values);
          tx.onabort = () => finish(values);
          tx.onerror = () => finish(values);
        } catch {
          finish(values);
        }
      }),
    );
  }

  async put<TValue = unknown>(
    storeName: string,
    key: IDBValidKey,
    value: TValue,
  ): Promise<true> {
    await this.putMany(storeName, [{ key, value }]);
    return true;
  }

  async putMany<TValue = unknown>(
    storeName: string,
    entries: readonly IndexedDBEntry<TValue>[],
  ): Promise<true> {
    return await this.withDb(async (db) =>
      await new Promise<true>((resolve, reject) => {
        let settled = false;
        const finish = (err: unknown) => {
          if (settled) {
            return;
          }
          settled = true;
          if (err) {
            reject(err);
            return;
          }
          resolve(true);
        };

        try {
          const tx = db.transaction(storeName, 'readwrite');
          const store = tx.objectStore(storeName);
          for (const entry of entries || []) {
            if (!entry || typeof entry.key === 'undefined') {
              continue;
            }
            store.put(entry.value, entry.key);
          }
          tx.oncomplete = () => finish(null);
          tx.onabort = () => finish(tx.error || new Error(`failed to write ${storeName}`));
          tx.onerror = () => finish(tx.error || new Error(`failed to write ${storeName}`));
        } catch (error) {
          finish(error);
        }
      }),
    );
  }

  async deleteMany(storeName: string, keys: readonly IDBValidKey[]): Promise<number> {
    if (!Array.isArray(keys) || !keys.length) {
      return 0;
    }

    return await this.withDb(async (db) =>
      await new Promise<number>((resolve, reject) => {
        let settled = false;
        const finish = (err: unknown) => {
          if (settled) {
            return;
          }
          settled = true;
          if (err) {
            reject(err);
            return;
          }
          resolve(keys.length);
        };

        try {
          const tx = db.transaction(storeName, 'readwrite');
          const store = tx.objectStore(storeName);
          for (const key of keys) {
            store.delete(key);
          }
          tx.oncomplete = () => finish(null);
          tx.onabort = () => finish(tx.error || new Error(`failed to delete ${storeName}`));
          tx.onerror = () => finish(tx.error || new Error(`failed to delete ${storeName}`));
        } catch (error) {
          finish(error);
        }
      }),
    );
  }

  async listKeys(storeName: string): Promise<IDBValidKey[]> {
    return await this.withDb(async (db) =>
      await new Promise<IDBValidKey[]>((resolve, reject) => {
        let settled = false;
        const finish = (err: unknown, value: IDBValidKey[] = []) => {
          if (settled) {
            return;
          }
          settled = true;
          if (err) {
            reject(err);
            return;
          }
          resolve(value);
        };

        try {
          const tx = db.transaction(storeName, 'readonly');
          const store = tx.objectStore(storeName);
          if (typeof store.getAllKeys === 'function') {
            const req = store.getAllKeys();
            req.onsuccess = () => finish(null, Array.isArray(req.result) ? req.result : []);
            req.onerror = () => finish(req.error || new Error(`failed to list ${storeName} keys`));
            return;
          }

          const keys: IDBValidKey[] = [];
          const req = store.openCursor();
          req.onsuccess = () => {
            const cursor = req.result;
            if (cursor) {
              keys.push(cursor.key);
              cursor.continue();
              return;
            }
            finish(null, keys);
          };
          req.onerror = () => finish(req.error || new Error(`failed to iterate ${storeName} keys`));
        } catch (error) {
          finish(error);
        }
      }),
    );
  }

  async clear(storeName: string, label = storeName): Promise<boolean> {
    return await this.withDb(async (db) =>
      await new Promise<boolean>((resolve, reject) => {
        let settled = false;
        const finish = (err: unknown, value = false) => {
          if (settled) {
            return;
          }
          settled = true;
          if (err) {
            reject(err);
            return;
          }
          resolve(value);
        };

        try {
          const tx = db.transaction(storeName, 'readwrite');
          const store = tx.objectStore(storeName);
          const req = store.clear();
          tx.oncomplete = () => finish(null, true);
          tx.onabort = () => finish(tx.error || new Error(`failed to clear ${label}`));
          tx.onerror = () => finish(tx.error || new Error(`failed to clear ${label}`));
          req.onerror = () => finish(req.error || new Error(`failed to clear ${label}`));
        } catch (error) {
          finish(error);
        }
      }),
    );
  }
}

export function buildPredictionCacheKey({
  fileKey = '',
  modelName = '',
  backend = 'webgpu',
} = {}): string {
  return [String(fileKey || ''), String(modelName || ''), String(backend || '')].join('::');
}

export function normalizePredictionCacheEntry(
  entry: unknown,
): PredictionCacheEntry | null {
  if (!entry || typeof entry !== 'object') {
    return null;
  }

  const typed = entry as {
    totalArgmax?: unknown;
    visibleArgmax?: unknown;
    totalConfidence?: unknown;
    visibleConfidence?: unknown;
    totalExpectedFrames?: unknown;
    totalBatchCount?: unknown;
    complete?: unknown;
  };

  const totalArgmax =
    typed.totalArgmax instanceof Int32Array
      ? typed.totalArgmax
      : Int32Array.from(Array.isArray(typed.totalArgmax) ? typed.totalArgmax : []);
  const visibleArgmax =
    typed.visibleArgmax instanceof Int32Array
      ? typed.visibleArgmax
      : Int32Array.from(Array.isArray(typed.visibleArgmax) ? typed.visibleArgmax : []);
  const totalConfidence =
    typed.totalConfidence instanceof Float32Array
      ? typed.totalConfidence
      : Float32Array.from(Array.isArray(typed.totalConfidence) ? typed.totalConfidence : []);
  const visibleConfidence =
    typed.visibleConfidence instanceof Float32Array
      ? typed.visibleConfidence
      : Float32Array.from(Array.isArray(typed.visibleConfidence) ? typed.visibleConfidence : []);
  const totalExpectedFrames = Number(typed.totalExpectedFrames);
  const totalBatchCount = Number(typed.totalBatchCount);
  const complete = typed.complete === true;

  const normalized: PredictionCacheEntry = {
    totalArgmax,
    visibleArgmax,
    totalConfidence,
    visibleConfidence,
    complete,
  };

  if (Number.isFinite(totalExpectedFrames) && totalExpectedFrames > 0) {
    normalized.totalExpectedFrames = Math.floor(totalExpectedFrames);
  }
  if (Number.isFinite(totalBatchCount) && totalBatchCount > 0) {
    normalized.totalBatchCount = Math.floor(totalBatchCount);
  }

  return normalized;
}

export function buildCFPAnalysisCacheKey({
  namespace = '',
  fileKey = '',
  modelType = 'melody',
  backend = 'runtime-auto',
}: {
  namespace?: string;
  fileKey?: string;
  modelType?: string;
  backend?: string;
} = {}): string {
  // CFP features are model-agnostic; keep modelType in signature only for backward compatibility.
  void modelType;
  return [
    String(namespace || '').trim(),
    String(fileKey || '').trim(),
    String(backend || 'runtime-auto').trim(),
  ]
    .filter(Boolean)
    .join('::');
}

export function countCFPBatchFrames(npyBatches: readonly CFPBatch[] = []): number {
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
  version: number;
  kind: 'chunked';
  chunkCount: number;
  complete: boolean;
  updatedAt: number;
  lastAccessed: number;
  byteSize: number;
}

interface CFPCacheChunkRecord {
  version: number;
  kind: 'chunk';
  index: number;
  start: number | null;
  end: number | null;
  data: Float32Array;
  shape: ArrayLike<number>;
}

type CFPCacheStoreValue = CFPCacheChunkRecord | CFPCacheMeta;

export class CFPIndexedDBCache {
  private static readonly MAX_CFP_BYTES = 1024 * 1024 * 1024;
  private readonly normalizeCFPBatches: (rawBatches: unknown) => CFPBatch[];
  private readonly cfpStore: IndexedDBStore;
  private readonly predictionStore: IndexedDBStore;

  constructor({
    normalizeCFPBatches,
  }: {
    normalizeCFPBatches: (rawBatches: unknown) => CFPBatch[];
  }) {
    if (typeof normalizeCFPBatches !== 'function') {
      throw new Error('CFPIndexedDBCache requires a normalizeCFPBatches function');
    }
    this.normalizeCFPBatches = normalizeCFPBatches;
    this.cfpStore = new IndexedDBStore({
      dbName: 'CFPFeatureCache',
      version: 1,
      storeNames: ['cfp'],
    });
    this.predictionStore = new IndexedDBStore({
      dbName: 'PredictionCache',
      version: 1,
      storeNames: ['pred'],
    });
  }

  getCFPCacheMetaKey(key: string): string {
    return `${key}::meta`;
  }

  getCFPCacheChunkKey(key: string, index: number): string {
    return `${key}::chunk::${String(index).padStart(6, '0')}`;
  }

  isCFPCacheKeyMatch(baseKey: string, entryKey: IDBValidKey): boolean {
    if (entryKey === baseKey) {
      return true;
    }
    if (entryKey === this.getCFPCacheMetaKey(baseKey)) {
      return true;
    }
    return typeof entryKey === 'string' && entryKey.startsWith(`${baseKey}::chunk::`);
  }

  buildCFPCacheMeta({
    chunkCount = 0,
    complete = false,
    updatedAt = Date.now(),
    lastAccessed = updatedAt,
    byteSize = 0,
  }: {
    chunkCount?: number;
    complete?: boolean;
    updatedAt?: number;
    lastAccessed?: number;
    byteSize?: number;
  } = {}): CFPCacheMeta {
    return {
      version: 3,
      kind: 'chunked',
      chunkCount: Math.max(0, Math.floor(toNonNegativeFinite(chunkCount, 0))),
      complete: complete === true,
      updatedAt: Math.floor(toNonNegativeFinite(updatedAt, Date.now())),
      lastAccessed: Math.floor(toNonNegativeFinite(lastAccessed, Date.now())),
      byteSize: Math.max(0, Math.floor(toNonNegativeFinite(byteSize, 0))),
    };
  }

  private normalizeCFPCacheMeta(rawMeta: unknown): CFPCacheMeta | null {
    if (!rawMeta || typeof rawMeta !== 'object') {
      return null;
    }

    const typed = rawMeta as {
      chunkCount?: unknown;
      complete?: unknown;
      updatedAt?: unknown;
      lastAccessed?: unknown;
      byteSize?: unknown;
    };

    const updatedAt = Math.floor(toNonNegativeFinite(typed.updatedAt, Date.now()));
    const lastAccessed = Math.floor(toNonNegativeFinite(typed.lastAccessed, updatedAt));
    const byteSize = Math.max(0, Math.floor(toNonNegativeFinite(typed.byteSize, 0)));
    return this.buildCFPCacheMeta({
      chunkCount: Math.max(0, Math.floor(toNonNegativeFinite(typed.chunkCount, 0))),
      complete: typed.complete === true,
      updatedAt,
      lastAccessed,
      byteSize,
    });
  }

  private getCFPCacheBaseKeyFromMetaKey(metaKey: IDBValidKey): string {
    return typeof metaKey === 'string' && metaKey.endsWith('::meta')
      ? metaKey.slice(0, -6)
      : String(metaKey);
  }

  normalizeCFPCacheChunkRecord(
    rawChunk: unknown,
    fallbackIndex = 0,
  ): CFPCacheChunkRecord | null {
    const normalized = this.normalizeCFPBatches([rawChunk]);
    if (!normalized.length) {
      return null;
    }

    const chunk = normalized[0];
    if (!chunk) {
      return null;
    }

    const rawChunkRecord =
      rawChunk && typeof rawChunk === 'object'
        ? (rawChunk as { index?: unknown; start?: unknown; end?: unknown })
        : {};
    const safeIndex =
      Number.isInteger(rawChunkRecord.index) && Number(rawChunkRecord.index) >= 0
        ? Number(rawChunkRecord.index)
        : Math.max(0, Math.floor(toPositiveFinite(fallbackIndex, 0)));
    const start = Number.isFinite(rawChunkRecord.start) ? Math.floor(Number(rawChunkRecord.start)) : null;
    const end = Number.isFinite(rawChunkRecord.end) ? Math.floor(Number(rawChunkRecord.end)) : null;

    return {
      version: 2,
      kind: 'chunk',
      index: safeIndex,
      start,
      end,
      data: chunk.data,
      shape: chunk.shape,
    };
  }

  async deleteCFPCacheEntriesForKey(key: string): Promise<number> {
    const allKeys = await this.cfpStore.listKeys('cfp');
    const matchedKeys = allKeys.filter((entryKey) => this.isCFPCacheKeyMatch(key, entryKey));
    if (!matchedKeys.length) {
      return 0;
    }
    return await this.cfpStore.deleteMany('cfp', matchedKeys);
  }

  private async listCFPCacheMetaEntries(): Promise<Array<{ key: string; meta: CFPCacheMeta }>> {
    const allKeys = await this.cfpStore.listKeys('cfp');
    const metaKeys = allKeys.filter(
      (entryKey): entryKey is string => typeof entryKey === 'string' && entryKey.endsWith('::meta'),
    );
    if (!metaKeys.length) {
      return [];
    }

    const rawMetas = await this.cfpStore.getMany<unknown>('cfp', metaKeys);
    const entries: Array<{ key: string; meta: CFPCacheMeta }> = [];
    for (let i = 0; i < metaKeys.length; i += 1) {
      const metaKey = metaKeys[i];
      if (typeof metaKey !== 'string') {
        continue;
      }
      const meta = this.normalizeCFPCacheMeta(rawMetas[i]);
      if (!meta) {
        continue;
      }
      const byteSize =
        meta.byteSize > 0
          ? meta.byteSize
          : await this.estimateCFPCacheByteSizeForKey(
            this.getCFPCacheBaseKeyFromMetaKey(metaKey),
            meta.chunkCount,
          );
      entries.push({
        key: this.getCFPCacheBaseKeyFromMetaKey(metaKey),
        meta: {
          ...meta,
          byteSize,
        },
      });
    }
    return entries;
  }

  private async estimateCFPCacheByteSizeForKey(key: string, chunkCount: number): Promise<number> {
    const safeChunkCount = Math.max(0, Math.floor(toNonNegativeFinite(chunkCount, 0)));
    if (safeChunkCount === 0) {
      return 0;
    }
    const chunkKeys: IDBValidKey[] = [];
    for (let i = 0; i < safeChunkCount; i += 1) {
      chunkKeys.push(this.getCFPCacheChunkKey(key, i));
    }
    const rawChunks = await this.cfpStore.getMany<unknown>('cfp', chunkKeys);
    let byteSize = 0;
    for (let i = 0; i < rawChunks.length; i += 1) {
      const rawChunk = rawChunks[i];
      if (typeof rawChunk === 'undefined' || rawChunk === null) {
        continue;
      }
      const normalizedChunk = this.normalizeCFPCacheChunkRecord(rawChunk, i);
      if (!normalizedChunk) {
        continue;
      }
      byteSize += estimateCFPCacheChunkBytes(normalizedChunk);
    }
    return byteSize;
  }

  private async touchCFPCacheMeta(key: string): Promise<void> {
    try {
      const rawMeta = await this.cfpStore.get<unknown>('cfp', this.getCFPCacheMetaKey(key));
      const meta = this.normalizeCFPCacheMeta(rawMeta);
      if (!meta) {
        return;
      }
      await this.cfpStore.put('cfp', this.getCFPCacheMetaKey(key), this.buildCFPCacheMeta({
        chunkCount: meta.chunkCount,
        complete: meta.complete,
        updatedAt: meta.updatedAt,
        lastAccessed: Date.now(),
        byteSize: meta.byteSize,
      }));
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
    const existingMeta = this.normalizeCFPCacheMeta(
      await this.cfpStore.get<unknown>('cfp', this.getCFPCacheMetaKey(key)),
    );

    const expectedChunkCount = Math.max(
      0,
      Math.floor(
        toNonNegativeFinite(
          existingMeta?.chunkCount ?? 0,
          0,
        ),
      ),
    );
    const chunkKeys: IDBValidKey[] = [];
    for (let i = 0; i < expectedChunkCount; i += 1) {
      chunkKeys.push(this.getCFPCacheChunkKey(key, i));
    }
    const rawChunks = chunkKeys.length ? await this.cfpStore.getMany<unknown>('cfp', chunkKeys) : [];
    let byteSize = 0;
    let actualChunkCount = 0;
    for (let i = 0; i < rawChunks.length; i += 1) {
      const rawChunk = rawChunks[i];
      if (typeof rawChunk === 'undefined' || rawChunk === null) {
        continue;
      }
      const normalizedChunk = this.normalizeCFPCacheChunkRecord(rawChunk, i);
      if (!normalizedChunk) {
        continue;
      }
      byteSize += estimateCFPCacheChunkBytes(normalizedChunk);
      actualChunkCount += 1;
    }

    const nextMeta = this.buildCFPCacheMeta({
      chunkCount: Math.max(expectedChunkCount, actualChunkCount),
      complete: complete ?? existingMeta?.complete ?? false,
      updatedAt: Date.now(),
      lastAccessed: Date.now(),
      byteSize,
    });
    await this.cfpStore.put('cfp', this.getCFPCacheMetaKey(key), nextMeta);
    return nextMeta;
  }

  private async enforceCFPCacheBudget(currentKey: string | null = null): Promise<void> {
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
    const directValue = await this.cfpStore.get<CFPBatch[]>('cfp', key);
    if (Array.isArray(directValue)) {
      void this.touchCFPCacheMeta(key);
      return directValue;
    }

    const meta = await this.cfpStore.get<CFPCacheMeta>('cfp', this.getCFPCacheMetaKey(key));
    if (!meta || typeof meta !== 'object') {
      return directValue || null;
    }

    const chunkCount = Math.max(0, Math.floor(toPositiveFinite(meta.chunkCount, 0)));
    if (!allowPartial && meta.complete !== true) {
      return null;
    }
    if (chunkCount === 0) {
      return [];
    }

    const chunkKeys: IDBValidKey[] = [];
    for (let i = 0; i < chunkCount; i += 1) {
      chunkKeys.push(this.getCFPCacheChunkKey(key, i));
    }
    const rawChunks = await this.cfpStore.getMany<unknown>('cfp', chunkKeys);
    const normalizedChunks: CFPCacheChunkRecord[] = [];
    for (let i = 0; i < rawChunks.length; i += 1) {
      const rawChunk = rawChunks[i];
      if (typeof rawChunk === 'undefined' || rawChunk === null) {
        continue;
      }
      const normalizedChunk = this.normalizeCFPCacheChunkRecord(rawChunk, i);
      if (!normalizedChunk) {
        continue;
      }
      normalizedChunks.push(normalizedChunk);
    }
    if (!normalizedChunks.length) {
      return null;
    }
    if (!allowPartial && normalizedChunks.length !== chunkCount) {
      return null;
    }

    normalizedChunks.sort((a, b) => a.index - b.index);
    await this.touchCFPCacheMeta(key);
    return normalizedChunks.map(({ data, shape, start, end, index }) => ({
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

    const entries: Array<{ key: string; value: CFPCacheStoreValue }> = normalizedBatches
      .map((batch, index) => ({
        key: this.getCFPCacheChunkKey(key, index),
        value: this.normalizeCFPCacheChunkRecord(batch, index),
      }))
      .filter((entry): entry is { key: string; value: CFPCacheChunkRecord } => entry.value !== null);

    entries.push({
      key: this.getCFPCacheMetaKey(key),
      value: this.buildCFPCacheMeta({
        chunkCount: normalizedBatches.length,
        complete: true,
      }),
    });

    await this.cfpStore.putMany('cfp', entries);
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
    const safeIndex = Number.isInteger(nextIndex) && nextIndex >= 0 ? nextIndex : null;
    if (safeIndex === null) {
      throw new Error('appendCFPCacheChunk requires a non-negative integer index');
    }

    const normalizedChunk = this.normalizeCFPCacheChunkRecord(chunk, safeIndex);
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
    await this.cfpStore.putMany<CFPCacheStoreValue>('cfp', [
      {
        key: this.getCFPCacheChunkKey(key, safeIndex),
        value: normalizedChunk,
      },
      {
        key: this.getCFPCacheMetaKey(key),
        value: this.buildCFPCacheMeta({
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
    const existingMeta = await this.cfpStore.get<CFPCacheMeta>('cfp', this.getCFPCacheMetaKey(key));
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
    await this.cfpStore.clear('cfp', 'CFP cache');
    return { dbName: 'CFPFeatureCache', storeName: 'cfp', cleared: true };
  }

  async getPredictionCache(key: string): Promise<PredictionCacheEntry | null> {
    return await this.predictionStore.get<PredictionCacheEntry>('pred', key);
  }

  async setPredictionCache(
    key: string,
    value: PredictionCacheEntry,
  ): Promise<void> {
    await this.predictionStore.put('pred', key, value);
  }

  async clearPredictionCache() {
    await this.predictionStore.clear('pred', 'prediction cache');
    return { dbName: 'PredictionCache', storeName: 'pred', cleared: true };
  }

  async clearAllIndexedDBCaches() {
    const [cfp, prediction] = await Promise.all([
      this.clearCFPCache(),
      this.clearPredictionCache(),
    ]);
    return {
      cfpCleared: !!cfp?.cleared,
      predictionCleared: !!prediction?.cleared,
    };
  }
}
