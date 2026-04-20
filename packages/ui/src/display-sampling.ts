const DEFAULT_MIN_UNITS_PER_SECOND = 5;
const DEFAULT_MAX_UNITS_PER_SECOND = 100;
const VALID_REPRESENTATIVE_MODES = new Set<RepresentativeMode>([
  "first",
  "last",
  "middle",
  "best",
  "first-valid",
  "last-valid",
  "highest-confidence",
]);

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function toFiniteNumber(value: unknown, fallback: number): number {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

export const DEFAULT_DISPLAY_SAMPLING_CONFIG = Object.freeze({
  minUnitsPerSecond: DEFAULT_MIN_UNITS_PER_SECOND,
  maxUnitsPerSecond: DEFAULT_MAX_UNITS_PER_SECOND,
  representativeMode: "highest-confidence" as RepresentativeMode,
});

export type DisplaySamplingConfig = Readonly<{
  minUnitsPerSecond: number;
  maxUnitsPerSecond: number;
  representativeMode: RepresentativeMode;
}>;

export type RepresentativeMode =
  | "first"
  | "last"
  | "middle"
  | "best"
  | "first-valid"
  | "last-valid"
  | "highest-confidence";

function normalizeRepresentativeMode(
  value: unknown,
  fallback: RepresentativeMode = DEFAULT_DISPLAY_SAMPLING_CONFIG.representativeMode,
): RepresentativeMode {
  const normalized = String(value || "").trim().toLowerCase();
  return VALID_REPRESENTATIVE_MODES.has(normalized as RepresentativeMode)
    ? (normalized as RepresentativeMode)
    : fallback;
}

export function normalizeDisplaySamplingConfig(
  config: Partial<DisplaySamplingConfig> | null | undefined = {},
): DisplaySamplingConfig {
  const safeConfig = config ?? {};
  const minUnitsPerSecond = clampNumber(
    Math.floor(toFiniteNumber(safeConfig.minUnitsPerSecond, DEFAULT_MIN_UNITS_PER_SECOND)),
    1,
    100,
  );
  const maxUnitsPerSecond = clampNumber(
    Math.floor(toFiniteNumber(safeConfig.maxUnitsPerSecond, DEFAULT_MAX_UNITS_PER_SECOND)),
    1,
    100,
  );
  return Object.freeze({
    minUnitsPerSecond: Math.min(minUnitsPerSecond, maxUnitsPerSecond),
    maxUnitsPerSecond: Math.max(minUnitsPerSecond, maxUnitsPerSecond),
    representativeMode: normalizeRepresentativeMode(
      safeConfig.representativeMode,
      DEFAULT_DISPLAY_SAMPLING_CONFIG.representativeMode,
    ),
  });
}

export function getDisplayUnitsPerSecondForZoom({
  zoom,
  minZoom = 1,
  maxZoom = 20,
  minUnitsPerSecond = DEFAULT_MIN_UNITS_PER_SECOND,
  maxUnitsPerSecond = DEFAULT_MAX_UNITS_PER_SECOND,
}: {
  zoom?: number | undefined;
  minZoom?: number | undefined;
  maxZoom?: number | undefined;
  minUnitsPerSecond?: number | undefined;
  maxUnitsPerSecond?: number | undefined;
} = {}): number {
  const loZoom = Math.max(1, toFiniteNumber(minZoom, 1));
  const hiZoom = Math.max(loZoom, toFiniteNumber(maxZoom, loZoom));
  const safeZoom = clampNumber(toFiniteNumber(zoom, loZoom), loZoom, hiZoom);
  const t = hiZoom === loZoom ? 1 : (safeZoom - loZoom) / (hiZoom - loZoom);
  const loUnits = Math.max(1, toFiniteNumber(minUnitsPerSecond, DEFAULT_MIN_UNITS_PER_SECOND));
  const hiUnits = Math.max(loUnits, toFiniteNumber(maxUnitsPerSecond, DEFAULT_MAX_UNITS_PER_SECOND));
  return loUnits + (hiUnits - loUnits) * clampNumber(t, 0, 1);
}

export function getDisplayMaxZoomForTotalFrames({
  totalFrames = 0,
  maxUnitsPerSecond = DEFAULT_MAX_UNITS_PER_SECOND,
  frameRateHz = 100,
}: {
  totalFrames?: number | undefined;
  maxUnitsPerSecond?: number | undefined;
  frameRateHz?: number | undefined;
} = {}): number {
  const safeTotalFrames = Math.max(0, Math.floor(toFiniteNumber(totalFrames, 0)));
  const safeFrameRate = Math.max(0.0001, toFiniteNumber(frameRateHz, 100));
  const safeMaxUnitsPerSecond = Math.max(
    1,
    toFiniteNumber(maxUnitsPerSecond, DEFAULT_MAX_UNITS_PER_SECOND),
  );
  const minVisibleFrames = Math.max(
    1,
    Math.round((safeFrameRate * DEFAULT_MAX_UNITS_PER_SECOND) / safeMaxUnitsPerSecond),
  );
  if (safeTotalFrames <= 0) {
    return 1;
  }
  return Math.max(1, safeTotalFrames / minVisibleFrames);
}

export function getDisplayFrameRateHz(totalUnits: number, durationSec: number): number {
  const safeUnits = Math.max(1, Math.floor(toFiniteNumber(totalUnits, 1)));
  const safeDuration = Math.max(0, toFiniteNumber(durationSec, 0));
  if (safeDuration > 0) {
    return Math.max(0.0001, safeUnits / safeDuration);
  }
  return Math.max(1, safeUnits);
}

export function getDisplayVisibleFrameCount({
  zoom,
  totalFrames,
  durationSec: _durationSec,
  minUnitsPerSecond: _minUnitsPerSecond = DEFAULT_MIN_UNITS_PER_SECOND,
  maxUnitsPerSecond: _maxUnitsPerSecond = DEFAULT_MAX_UNITS_PER_SECOND,
}: {
  zoom?: number | undefined;
  totalFrames: number;
  durationSec: number;
  minUnitsPerSecond?: number | undefined;
  maxUnitsPerSecond?: number | undefined;
}): number {
  const safeTotalFrames = Math.max(1, Math.floor(toFiniteNumber(totalFrames, 1)));
  const safeZoom = Math.max(1, toFiniteNumber(zoom, 1));
  return Math.max(10, Math.min(safeTotalFrames, Math.floor(safeTotalFrames / safeZoom)));
}

export function getDisplayStrideFramesForZoom({
  zoom,
  minZoom = 1,
  maxZoom = 20,
  minUnitsPerSecond = DEFAULT_MIN_UNITS_PER_SECOND,
  maxUnitsPerSecond = DEFAULT_MAX_UNITS_PER_SECOND,
  frameRateHz = 100,
}: {
  zoom?: number | undefined;
  minZoom?: number | undefined;
  maxZoom?: number | undefined;
  minUnitsPerSecond?: number | undefined;
  maxUnitsPerSecond?: number | undefined;
  frameRateHz?: number | undefined;
} = {}): number {
  const unitsPerSecond = getDisplayUnitsPerSecondForZoom({
    zoom,
    minZoom,
    maxZoom,
    minUnitsPerSecond,
    maxUnitsPerSecond,
  });
  const safeFrameRate = Math.max(1, toFiniteNumber(frameRateHz, 100));
  return Math.max(1, Math.round(safeFrameRate / Math.max(1e-6, unitsPerSecond)));
}

export function getAlignedSampleStart(offset: number, step: number): number {
  const safeStep = Math.max(1, Math.floor(toFiniteNumber(step, 1)));
  const safeOffset = Math.max(0, Math.floor(toFiniteNumber(offset, 0)));
  return safeOffset - (safeOffset % safeStep);
}

export function pickRepresentativeIndex({
  start,
  end,
  mode = DEFAULT_DISPLAY_SAMPLING_CONFIG.representativeMode,
  isValidAt = null,
  scoreAt = null,
}: {
  start?: number;
  end?: number;
  mode?: RepresentativeMode | string;
  isValidAt?: ((index: number) => boolean) | null;
  scoreAt?: ((index: number) => number) | null;
} = {}): number {
  const safeStart = Math.max(0, Math.floor(toFiniteNumber(start, 0)));
  const safeEnd = Math.max(safeStart + 1, Math.floor(toFiniteNumber(end, safeStart + 1)));
  const first = safeStart;
  const last = safeEnd - 1;
  const middle = Math.max(safeStart, Math.min(last, Math.floor((safeStart + last) / 2)));
  const hasPredicate = typeof isValidAt === "function";
  const hasScore = typeof scoreAt === "function";
  const normalizedMode = normalizeRepresentativeMode(mode);

  if (normalizedMode === "highest-confidence") {
    let bestIdx = -1;
    let bestScore = -Infinity;
    for (let i = safeStart; i < safeEnd; i += 1) {
      if (hasPredicate && !isValidAt(i)) {
        continue;
      }
      const score = hasScore ? Number(scoreAt(i)) : NaN;
      if (!Number.isFinite(score)) {
        continue;
      }
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) {
      return bestIdx;
    }
    if (hasPredicate) {
      for (let i = safeStart; i < safeEnd; i += 1) {
        if (isValidAt(i)) {
          return i;
        }
      }
    }
    return first;
  }

  if (normalizedMode === "best") {
    return pickRepresentativeIndex({
      start: safeStart,
      end: safeEnd,
      mode: "highest-confidence",
      isValidAt,
      scoreAt,
    });
  }

  if (normalizedMode === "last") {
    return last;
  }
  if (normalizedMode === "middle") {
    if (hasPredicate) {
      const middleCandidates = [middle, middle - 1, middle + 1];
      for (const candidate of middleCandidates) {
        if (candidate >= safeStart && candidate < safeEnd && isValidAt(candidate)) {
          return candidate;
        }
      }
    }
    return middle;
  }
  if (normalizedMode === "last-valid") {
    if (hasPredicate) {
      for (let i = last; i >= safeStart; i -= 1) {
        if (isValidAt(i)) {
          return i;
        }
      }
    }
    return last;
  }
  if (normalizedMode === "first") {
    return first;
  }
  if (hasPredicate) {
    for (let i = safeStart; i < safeEnd; i += 1) {
      if (isValidAt(i)) {
        return i;
      }
    }
  }
  return first;
}

export function getDisplayZoomBounds({
  totalFrames,
  durationSec,
  maxUnitsPerSecond = DEFAULT_MAX_UNITS_PER_SECOND,
}: {
  totalFrames: number;
  durationSec: number;
  maxUnitsPerSecond?: number | undefined;
}): { minZoom: number; maxZoom: number } {
  const safeTotalFrames = Math.max(1, Math.floor(toFiniteNumber(totalFrames, 1)));
  const frameRateHz = getDisplayFrameRateHz(safeTotalFrames, durationSec);
  return {
    minZoom: 1,
    maxZoom: getDisplayMaxZoomForTotalFrames({
      totalFrames: safeTotalFrames,
      maxUnitsPerSecond,
      frameRateHz,
    }),
  };
}
