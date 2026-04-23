export interface IndexedDBStoreOptions {
  dbName: string;
  version?: number;
  storeNames?: readonly (string | IndexedDBStoreDefinition)[];
}

export interface IndexedDBEntry<TValue = unknown> {
  key: IDBValidKey;
  value: TValue;
}

export interface IndexedDBStoreIndexDefinition {
  keyPath: string | string[];
  name: string;
  options?: IDBIndexParameters;
}

export interface IndexedDBStoreDefinition {
  indexes?: readonly IndexedDBStoreIndexDefinition[];
  name: string;
  onUpgrade?: (
    store: IDBObjectStore,
    context: { db: IDBDatabase; oldVersion: number; newVersion: number | null },
  ) => void;
}

function normalizeStoreDefinitions(
  storeNames: readonly (string | IndexedDBStoreDefinition)[] = [],
): IndexedDBStoreDefinition[] {
  const normalized: IndexedDBStoreDefinition[] = [];
  const seen = new Set<string>();
  for (const entry of Array.isArray(storeNames) ? storeNames : []) {
    const definition =
      typeof entry === "string"
        ? { name: entry }
        : entry && typeof entry === "object"
          ? entry
          : null;
    const name = String(definition?.name || "").trim();
    if (!name || seen.has(name)) {
      continue;
    }
    seen.add(name);
    normalized.push({
      name,
      indexes: Array.isArray(definition?.indexes) ? definition.indexes : [],
      onUpgrade:
        typeof definition?.onUpgrade === "function"
          ? definition.onUpgrade
          : undefined,
    });
  }
  return normalized;
}

interface IndexedDBTransactionContext {
  store: IDBObjectStore;
  tx: IDBTransaction;
}

export class IndexedDBStore {
  private readonly dbName: string;
  private readonly version: number;
  private readonly storeDefinitions: IndexedDBStoreDefinition[];

  constructor({ dbName, version = 1, storeNames = [] }: IndexedDBStoreOptions) {
    this.dbName = String(dbName || "").trim();
    this.version = Math.max(1, Math.floor(Number(version) || 1));
    this.storeDefinitions = normalizeStoreDefinitions(storeNames);
  }

  ensureIndexedDBAvailable(): void {
    if (typeof indexedDB === "undefined" || !indexedDB) {
      throw new Error("IndexedDB is not supported in current environment");
    }
  }

  async open(): Promise<IDBDatabase> {
    this.ensureIndexedDBAvailable();
    if (!this.dbName) {
      throw new Error("IndexedDBStore requires dbName");
    }

    return await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);
      request.onupgradeneeded = (event) => {
        const db = request.result;
        if (!db) {
          return;
        }
        const oldVersion =
          event && typeof event.oldVersion === "number" ? event.oldVersion : 0;
        const newVersion =
          event && typeof event.newVersion === "number"
            ? event.newVersion
            : null;
        for (const definition of this.storeDefinitions) {
          let store: IDBObjectStore;
          if (!db.objectStoreNames.contains(definition.name)) {
            store = db.createObjectStore(definition.name);
          } else {
            store = request.transaction!.objectStore(definition.name);
          }
          for (const indexDefinition of definition.indexes || []) {
            if (!store.indexNames.contains(indexDefinition.name)) {
              store.createIndex(
                indexDefinition.name,
                indexDefinition.keyPath,
                indexDefinition.options,
              );
            }
          }
          definition.onUpgrade?.(store, {
            db,
            oldVersion,
            newVersion,
          });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  close(db: IDBDatabase | null): void {
    try {
      if (db && typeof db.close === "function") {
        db.close();
      }
    } catch {}
  }

  async withDb<TResult>(
    fn: (db: IDBDatabase) => Promise<TResult>,
  ): Promise<TResult> {
    const db = await this.open();
    let result: TResult;

    try {
      result = await fn(db);
    } finally {
      this.close(db);
    }

    return result;
  }

  private async withTransaction<TResult>(
    storeName: string,
    mode: IDBTransactionMode,
    onError: string,
    run: (
      context: IndexedDBTransactionContext,
      finish: (error?: unknown, value?: TResult) => void,
    ) => void,
  ): Promise<TResult> {
    return await this.withDb(
      async (db) =>
        await new Promise<TResult>((resolve, reject) => {
          let settled = false;
          const finish = (error?: unknown, value?: TResult) => {
            if (settled) {
              return;
            }
            settled = true;
            if (error) {
              reject(error);
              return;
            }
            resolve(value as TResult);
          };

          try {
            const tx = db.transaction(storeName, mode);
            const store = tx.objectStore(storeName);
            tx.onabort = () =>
              finish(tx.error || new Error(`failed to ${onError}`));
            tx.onerror = () =>
              finish(tx.error || new Error(`failed to ${onError}`));
            run(
              {
                store,
                tx,
              },
              finish,
            );
          } catch (error) {
            finish(error);
          }
        }),
    );
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

    return await this.withDb(
      async (db) =>
        await new Promise<Array<TValue | null>>((resolve) => {
          const values: Array<TValue | null> = Array.from(
            { length: keys.length },
            () => null,
          );
          let settled = false;
          const finish = (nextValues: Array<TValue | null>) => {
            if (settled) {
              return;
            }
            settled = true;
            resolve(nextValues);
          };

          try {
            const tx = db.transaction(storeName, "readonly");
            const store = tx.objectStore(storeName);
            for (let i = 0; i < keys.length; i += 1) {
              const key = keys[i];
              if (typeof key === "undefined") {
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
    if (!Array.isArray(entries) || !entries.length) {
      return true;
    }

    return await this.withTransaction<true>(
      storeName,
      "readwrite",
      `write ${storeName}`,
      ({ store, tx }, finish) => {
        for (const entry of entries) {
          if (!entry || typeof entry.key === "undefined") {
            continue;
          }
          store.put(entry.value, entry.key);
        }
        tx.oncomplete = () => finish(undefined, true);
      },
    );
  }

  async deleteMany(
    storeName: string,
    keys: readonly IDBValidKey[],
  ): Promise<number> {
    if (!Array.isArray(keys) || !keys.length) {
      return 0;
    }

    return await this.withDb(
      async (db) =>
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
            const tx = db.transaction(storeName, "readwrite");
            const store = tx.objectStore(storeName);
            for (const key of keys) {
              store.delete(key);
            }
            tx.oncomplete = () => finish(null);
            tx.onabort = () =>
              finish(tx.error || new Error(`failed to delete ${storeName}`));
            tx.onerror = () =>
              finish(tx.error || new Error(`failed to delete ${storeName}`));
          } catch (error) {
            finish(error);
          }
        }),
    );
  }

  async deleteByIndex(
    storeName: string,
    indexName: string,
    indexValue: IDBValidKey,
  ): Promise<number> {
    return await this.withTransaction<number>(
      storeName,
      "readwrite",
      `delete ${storeName} by index ${indexName}`,
      ({ store, tx }, finish) => {
        let deletedCount = 0;
        const index = store.index(indexName);
        const request = index.openKeyCursor(IDBKeyRange.only(indexValue));
        request.onsuccess = () => {
          const cursor = request.result;
          if (!cursor) {
            return;
          }
          store.delete(cursor.primaryKey);
          deletedCount += 1;
          cursor.continue();
        };
        request.onerror = () =>
          finish(
            request.error ||
              new Error(`failed to iterate ${storeName} index ${indexName}`),
          );
        tx.oncomplete = () => finish(undefined, deletedCount);
      },
    );
  }

  async getAllByIndex<TValue = unknown>(
    storeName: string,
    indexName: string,
    indexValue: IDBValidKey,
  ): Promise<TValue[]> {
    return await this.withTransaction<TValue[]>(
      storeName,
      "readonly",
      `read ${storeName} by index ${indexName}`,
      ({ store, tx }, finish) => {
        const values: TValue[] = [];
        const index = store.index(indexName);
        if (typeof index.getAll === "function") {
          const req = index.getAll(IDBKeyRange.only(indexValue));
          req.onsuccess = () =>
            finish(
              undefined,
              (Array.isArray(req.result) ? req.result : []) as TValue[],
            );
          req.onerror = () =>
            finish(
              req.error ||
                new Error(`failed to read ${storeName} by index ${indexName}`),
            );
          return;
        }

        const request = index.openCursor(IDBKeyRange.only(indexValue));
        request.onsuccess = () => {
          const cursor = request.result;
          if (!cursor) {
            return;
          }
          values.push(cursor.value as TValue);
          cursor.continue();
        };
        request.onerror = () =>
          finish(
            request.error ||
              new Error(`failed to iterate ${storeName} index ${indexName}`),
          );
        tx.oncomplete = () => finish(undefined, values);
      },
    );
  }

  async listKeys(storeName: string): Promise<IDBValidKey[]> {
    return await this.withTransaction<IDBValidKey[]>(
      storeName,
      "readonly",
      `list ${storeName} keys`,
      ({ store }, finish) => {
        if (typeof store.getAllKeys === "function") {
          const req = store.getAllKeys();
          req.onsuccess = () =>
            finish(undefined, Array.isArray(req.result) ? req.result : []);
          req.onerror = () =>
            finish(req.error || new Error(`failed to list ${storeName} keys`));
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
          finish(undefined, keys);
        };
        req.onerror = () =>
          finish(req.error || new Error(`failed to iterate ${storeName} keys`));
      },
    );
  }

  async clear(storeName: string, label = storeName): Promise<boolean> {
    return await this.withTransaction<boolean>(
      storeName,
      "readwrite",
      `clear ${label}`,
      ({ store, tx }, finish) => {
        const req = store.clear();
        tx.oncomplete = () => finish(undefined, true);
        req.onerror = () =>
          finish(req.error || new Error(`failed to clear ${label}`));
      },
    );
  }
}
