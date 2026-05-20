import {
  type PredictionCacheEntry,
} from "@ohm/core/cache/prediction.js";
import { normalizeCoreModelName } from "@ohm/core/model-catalog.js";
import type { InferenceResult } from "../analysis.js";

export function buildInferenceCacheKey(
  fileKey: string,
  modelName: string,
  backend = "runtime-inference-v3",
): string {
  return [
    String(fileKey || "").trim(),
    normalizeCoreModelName(modelName),
    String(backend || "").trim(),
  ].join("::");
}

export function toInferenceResultFromCache(
  entry: PredictionCacheEntry,
): InferenceResult {
  const totalArgmax = Array.from(entry.totalArgmax || []);
  const totalConfidence = Array.from(entry.totalConfidence || []);
  return {
    totalArgmax,
    totalConfidence,
  };
}

export function isPredictionCacheEntryUsable(
  entry: PredictionCacheEntry | null | undefined,
): entry is PredictionCacheEntry {
  if (!entry) {
    return false;
  }
  if (entry.totalArgmax.length <= 0) {
    return false;
  }
  if (entry.totalConfidence.length < entry.totalArgmax.length) {
    return false;
  }
  return true;
}

export function toPredictionCacheEntry(
  result: InferenceResult,
): PredictionCacheEntry {
  return {
    totalArgmax: Int32Array.from(
      Array.isArray(result.totalArgmax) ? result.totalArgmax : [],
    ),
    totalConfidence: Float32Array.from(
      Array.isArray(result.totalConfidence) ? result.totalConfidence : [],
    ),
  };
}
