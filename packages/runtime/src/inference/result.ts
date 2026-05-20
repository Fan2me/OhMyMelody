import type { InferenceResult } from "../analysis.js";

export function buildEmptyInferenceResult(): InferenceResult {
  return {
    totalArgmax: [],
    totalConfidence: [],
  };
}

export function mergeInferenceResults(
  previous: InferenceResult,
  next: InferenceResult,
): InferenceResult {
  return {
    totalArgmax: [...(previous.totalArgmax || []), ...(next.totalArgmax || [])],
    totalConfidence: [
      ...(previous.totalConfidence || []),
      ...(next.totalConfidence || []),
    ],
  };
}
