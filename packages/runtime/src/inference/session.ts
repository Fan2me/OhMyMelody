import type { InferenceResult } from "../analysis.js";
import { buildEmptyInferenceResult } from "./result.js";

export type InferenceSessionState = {
  modelName: string;
  id: number;
  useCachedResult: boolean;
  complete: boolean;
  result: InferenceResult;
};

export function createInferenceSessionState(): InferenceSessionState {
  return {
    modelName: "",
    id: 0,
    useCachedResult: false,
    complete: false,
    result: buildEmptyInferenceResult(),
  };
}

export function resetInferenceSessionState(
  state: InferenceSessionState,
  nextModelName = "",
): void {
  state.modelName = nextModelName;
  state.id += 1;
  state.useCachedResult = false;
  state.complete = false;
  state.result = buildEmptyInferenceResult();
}
