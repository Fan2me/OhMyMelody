import { getCFPChunkCleanupPython } from "../script/script.js";

export function toErrorMessage(err: unknown): string {
  return err && typeof err === "object" && "toString" in err
    ? String(err)
    : String(err);
}

export function noop(..._args: unknown[]): void {}

export interface CFPChunkCleanupLike {
  runPython?: (code: string) => unknown;
  FS?: {
    unlink?: (path: string) => void;
  } | null;
}

export function cleanupPyodideChunkArtifacts(
  pyodide: CFPChunkCleanupLike | null,
): void {
  if (!pyodide) {
    return;
  }

  try {
    if (typeof pyodide.runPython === "function") {
      pyodide.runPython(getCFPChunkCleanupPython());
    }
  } catch {}

  try {
    const fs = pyodide.FS;
    if (fs && typeof fs.unlink === "function") {
      try {
        fs.unlink("cfp_out_shape.bin");
      } catch {}
      try {
        fs.unlink("cfp_out.bin");
      } catch {}
    }
  } catch {}
}

type GlobalWithIdleCallback = typeof globalThis & {
  requestIdleCallback?: (
    cb: () => void,
    options?: { timeout?: number },
  ) => number;
};

function getRequestIdleCallback():
  | ((cb: () => void, options?: { timeout?: number }) => number)
  | null {
  if (typeof globalThis === "undefined") {
    return null;
  }
  const globalRef = globalThis as GlobalWithIdleCallback;
  return typeof globalRef.requestIdleCallback === "function"
    ? globalRef.requestIdleCallback.bind(globalRef)
    : null;
}

export function pushToNextIdle(
  fn: () => void,
  { timeout }: { timeout: number } = { timeout: 1000 },
): number | ReturnType<typeof setTimeout> {
  const requestIdleCallback = getRequestIdleCallback();
  if (requestIdleCallback) {
    return requestIdleCallback(fn, {
      timeout: timeout,
    });
  }
  return setTimeout(fn, 0);
}

export function sortByStart<T extends { start?: number | null }>(
  results: readonly T[],
): T[] {
  return [...results].sort(
    (a, b) => Number(a.start ?? 0) - Number(b.start ?? 0),
  );
}

export function toPositiveFinite(
  value: unknown,
  fallback: number | null,
): number | null {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}
