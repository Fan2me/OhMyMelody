import type { CFPBatch, CFPIndexedDBCache } from "@ohm/core/cache/cfp.js";
import { getModuleLogger } from "@ohm/core/logging/logger.js";

const cfpLogger = getModuleLogger("core.runtime.cfp");

export const CFP_CACHE_BACKEND = "runtime-v2";

export function buildCFPCacheKey({
  namespace,
  fileKey,
  backend = CFP_CACHE_BACKEND,
}: {
  namespace: string;
  fileKey: string;
  backend?: string;
}): string {
  return [namespace, fileKey, backend]
    .map((part) => String(part || "").trim())
    .filter(Boolean)
    .join("::");
}

export function normalizeCFPBatches(rawBatches: unknown): CFPBatch[] {
  if (!Array.isArray(rawBatches)) {
    return [];
  }
  const normalized: CFPBatch[] = [];
  for (const item of rawBatches) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const typed = item as { data?: unknown; shape?: unknown };
    if (
      !(typed.data instanceof Float32Array) ||
      !(typed.shape instanceof Int32Array)
    ) {
      continue;
    }
    normalized.push({
      data: typed.data,
      shape: typed.shape,
    });
  }
  return normalized;
}

export async function commitCFPCache({
  cache,
  cacheKey,
  batches,
  startIndex,
  complete,
}: {
  cache: CFPIndexedDBCache;
  cacheKey: string;
  batches: readonly CFPBatch[];
  startIndex: number;
  complete: boolean;
}): Promise<void> {
  if (!cacheKey || !batches.length) {
    return;
  }
  try {
    for (let localIndex = 0; localIndex < batches.length; localIndex += 1) {
      const batch = batches[localIndex];
      if (!batch) {
        continue;
      }
      const index = startIndex + localIndex;
      await cache.appendCFPCacheChunk(cacheKey, batch, {
        index,
        reset: index === 0,
        complete: complete === true && localIndex === batches.length - 1,
        expectedChunkCount: startIndex + batches.length,
      });
    }
    if (complete) {
      await cache.finalizeCFPCache(cacheKey, {
        chunkCount: startIndex + batches.length,
      });
    }
    cfpLogger.info(
      `runtime cfp cache written: ${cacheKey}, batches=${batches.length}, complete=${complete ? "true" : "false"}`,
    );
  } catch (error) {
    cfpLogger.warn(
      `runtime cfp cache write skipped: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}
