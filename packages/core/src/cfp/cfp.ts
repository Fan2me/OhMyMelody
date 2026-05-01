export interface Range {
  start: number;
  end: number;
}

interface CFPProfileLike {
  [key: string]: unknown;
}

interface CFPWorkerTimingLike {
  [key: string]: unknown;
  phase?: unknown;
  start?: unknown;
  end?: unknown;
  t0?: unknown;
  t1?: unknown;
  tToPyStart?: unknown;
  tToPyEnd?: unknown;
  tPyStart?: unknown;
  tPyEnd?: unknown;
}

export function isPyodideOOMError(err: unknown): boolean {
  const msg =
    err && typeof err === "object" && "toString" in err ? String(err) : "";
  return /ArrayMemoryError|Unable to allocate|out of memory|MemoryError/i.test(
    msg,
  );
}

export function formatCFPProfileLine(
  profile: CFPProfileLike | null | undefined,
): string {
  if (!profile || typeof profile !== "object") {
    return "";
  }
  const pick = (key: string): string | null => {
    const value = profile[key];
    return Number.isFinite(value) ? Number(value).toFixed(1) : null;
  };
  const parts = [
    ["total", pick("total_ms")],
    ["pre", pick("preprocess_ms")],
    ["feature", pick("feature_total_ms")],
    ["stft", pick("stft_total_ms")],
    ["fft", pick("stft_fft_ms")],
    ["layers", pick("cfp_layers_ms")],
    ["freqMap", pick("freq_total_ms")],
    ["freqBuild", pick("freq_map_build_ms")],
    ["freqDot", pick("freq_dot_ms")],
    ["quefMap", pick("quef_total_ms")],
    ["quefBuild", pick("quef_map_build_ms")],
    ["quefDot", pick("quef_dot_ms")],
    ["post", pick("post_norm_total_ms")],
    ["export", pick("export_ms")],
  ]
    .filter(([, value]) => value !== null)
    .map(([label, value]) => `${label}=${value}ms`);
  return parts.join(" ");
}

export function formatCFPWorkerTimingLine(
  message: CFPWorkerTimingLike | null | undefined,
): string {
  if (!message || typeof message !== "object") {
    return "";
  }
  const pickDelta = (startKey: string, endKey: string): string | null => {
    const start = message[startKey];
    const end = message[endKey];
    return Number.isFinite(start) && Number.isFinite(end)
      ? (Number(end) - Number(start)).toFixed(1)
      : null;
  };
  const phase = typeof message.phase === "string" ? message.phase : "timing";
  const toPyMs = pickDelta("tToPyStart", "tToPyEnd");
  const pyMs = pickDelta("tPyStart", "tPyEnd");
  const totalMs =
    Number.isFinite(message.t0) && Number.isFinite(message.t1)
      ? (Number(message.t1) - Number(message.t0)).toFixed(1)
      : null;
  const rangeLabel =
    typeof message.start === "number" && typeof message.end === "number"
      ? `[${message.start},${message.end}) `
      : "";
  const parts = [
    totalMs !== null ? `total=${totalMs}ms` : null,
    toPyMs !== null ? `toPy=${toPyMs}ms` : null,
    pyMs !== null ? `pyExec=${pyMs}ms` : null,
  ].filter(Boolean);
  if (!parts.length) {
    return "";
  }
  return `${phase} ${rangeLabel}${parts.join(" ")}`;
}

export function splitCFPRangeOnOOM(
  start: number,
  end: number,
  minChunkSamples: number,
): {
  left: Range;
  right: Range;
  mid: number;
} | null {
  const safeStart = Number(start);
  const safeEnd = Number(end);
  const safeMin = Math.max(1, Math.floor(Number(minChunkSamples) || 0));
  if (
    !Number.isFinite(safeStart) ||
    !Number.isFinite(safeEnd) ||
    safeEnd - safeStart <= safeMin * 2
  ) {
    return null;
  }
  const mid = Math.floor((safeStart + safeEnd) / 2);
  return {
    left: { start: safeStart, end: mid },
    right: { start: mid, end: safeEnd },
    mid,
  };
}
