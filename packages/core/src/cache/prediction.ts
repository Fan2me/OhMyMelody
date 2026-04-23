import { IndexedDBStore } from "./base.js";

export interface PredictionCacheEntry {
  totalArgmax: Int32Array;
  totalConfidence: Float32Array;
}

export function normalizePredictionCacheEntry(
  entry: unknown,
): PredictionCacheEntry | null {
  if (!entry || typeof entry !== 'object') {
    return null;
  }

  const typed = entry as {
    totalArgmax?: unknown;
    totalConfidence?: unknown;
  };

  const totalArgmax =
    typed.totalArgmax instanceof Int32Array
      ? typed.totalArgmax
      : Int32Array.from(Array.isArray(typed.totalArgmax) ? typed.totalArgmax : []);
  const totalConfidence =
    typed.totalConfidence instanceof Float32Array
      ? typed.totalConfidence
      : Float32Array.from(Array.isArray(typed.totalConfidence) ? typed.totalConfidence : []);

  return {
    totalArgmax,
    totalConfidence,
  };
}

export class PredictionIndexedDBCache {
  private readonly predictionStore: IndexedDBStore;

  constructor() {
    this.predictionStore = new IndexedDBStore({
      dbName: 'PredictionCache',
      version: 1,
      storeNames: ['pred'],
    });
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
}
