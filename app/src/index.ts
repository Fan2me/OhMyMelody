import {
  CORE_MODEL_DEFAULT_NAME,
  CORE_MODEL_NAMES,
} from "@ohm/core";
import { CFPManager, createAnalyzer } from "@ohm/runtime";
import { createSpectrumUi } from "@ohm/ui";
import type { RepresentativeMode } from "@ohm/ui";
import {
  applyLocalizedText,
  getAppLanguage,
  initAppI18n,
  setAppLanguage,
  t,
  SUPPORTED_LANGUAGES,
} from "./i18n.js";

await initAppI18n();

const modelNameSelect = document.getElementById("modelName") as HTMLSelectElement | null;
const languageSelect = document.getElementById("languageSelect") as HTMLSelectElement | null;
const forceRefreshCheckbox = document.getElementById("forceRefreshCheckbox") as HTMLInputElement | null;
const audioFileInput = document.getElementById("audioFile") as HTMLInputElement | null;
const startMicRecordBtn = document.getElementById("startMicRecordBtn") as HTMLButtonElement | null;
const stopMicRecordBtn = document.getElementById("stopMicRecordBtn") as HTMLButtonElement | null;
const exportMicRecordBtn = document.getElementById("exportMicRecordBtn") as HTMLButtonElement | null;
const micRecordStatus = document.getElementById("micRecordStatus") as HTMLElement | null;
const mainEnabled = document.getElementById("mainEnabled") as HTMLInputElement | null;
const mainOverlayFps = document.getElementById("mainOverlayFps") as HTMLSelectElement | null;
const overviewEnabled = document.getElementById("overviewEnabled") as HTMLInputElement | null;
const mainFps = document.getElementById("mainFps") as HTMLSelectElement | null;
const overviewFps = document.getElementById("overviewFps") as HTMLSelectElement | null;
const overviewOverlayFps = document.getElementById("overviewOverlayFps") as HTMLSelectElement | null;
const representativeMode = document.getElementById("representativeMode") as HTMLSelectElement | null;
const minUnitsPerSecond = document.getElementById("minUnitsPerSecond") as HTMLSelectElement | null;
const maxUnitsPerSecond = document.getElementById("maxUnitsPerSecond") as HTMLSelectElement | null;
const pitchRangeMinNote = document.getElementById("pitchRangeMinNote") as HTMLSelectElement | null;
const pitchRangeMaxNote = document.getElementById("pitchRangeMaxNote") as HTMLSelectElement | null;
const playbackRateSlider = document.getElementById("playbackRateSlider") as HTMLInputElement | null;
const playbackRateValue = document.getElementById("playbackRateValue") as HTMLElement | null;
const abLoopEnabled = document.getElementById("abLoopEnabled") as HTMLInputElement | null;
const abLoopStart = document.getElementById("abLoopStart") as HTMLInputElement | null;
const abLoopEnd = document.getElementById("abLoopEnd") as HTMLInputElement | null;
const setLoopStartBtn = document.getElementById("setLoopStartBtn") as HTMLButtonElement | null;
const setLoopEndBtn = document.getElementById("setLoopEndBtn") as HTMLButtonElement | null;
const clearLoopBtn = document.getElementById("clearLoopBtn") as HTMLButtonElement | null;
const debugPanelEnabled = document.getElementById("debugPanelEnabled") as HTMLInputElement | null;
const debugInfo = document.getElementById("debugInfo") as HTMLElement | null;
const audioPlayer = document.getElementById("audioPlayer") as HTMLAudioElement | null;
const spectrumCanvasWrapper = document.getElementById("spectrumCanvasWrapper") as HTMLElement | null;
const uiStatus = document.getElementById("uiStatus") as HTMLElement | null;
const fileDropHint = document.getElementById("fileDropHint") as HTMLElement | null;

const STATUS_LABEL_KEYS: Record<string, string> = {
  idle: "status.idle",
  running: "status.running",
  succeeded: "status.succeeded",
  failed: "status.failed",
  cancelled: "status.cancelled",
  disposed: "status.disposed",
};

const analyzer = createAnalyzer(
  {
    cfpManager: new CFPManager({
      // Keep worker creation in the app entry so Vite can transform the worker
      // URL correctly in both dev server and production build.
      createWorkerInstance: () =>
        new Worker(
          new URL("../../packages/core/src/cfp/worker.ts", import.meta.url),
          { type: "module" },
        ),
      cfpScriptUrl: new URL("../../packages/core/cfp.py", import.meta.url).toString(),
    }),
  },
);
const ui = createSpectrumUi({
  analyzer,
  mount: spectrumCanvasWrapper,
  audioElement: audioPlayer,
  sections: {
    main: { enabled: true, overlay: true, fps: 60, overlayFps: 60 },
    overview: { enabled: true, overlay: true, fps: 24, overlayFps: 24 },
  },
});

let currentAudioObjectUrl = "";
let previousStatus = ui.getState().status;
let previousCfpLen = 0;
let previousInferenceCount = 0;
let modelEntries: string[] = Array.from(CORE_MODEL_NAMES);
const SETTINGS_KEY = "ohmymelody.app.settings.v1";

type AppSettings = {
  modelName?: string;
  forceRefresh?: boolean;
  audioPlaybackRate?: number;
  abLoop?: {
    enabled?: boolean;
    start?: number | null;
    end?: number | null;
  };
  debugPanelEnabled?: boolean;
  sections?: {
    main?: { enabled?: boolean; fps?: number; overlayFps?: number };
    overview?: { enabled?: boolean; fps?: number; overlayFps?: number };
  };
  displaySampling?: {
    representativeMode?: RepresentativeMode;
    minUnitsPerSecond?: number;
    maxUnitsPerSecond?: number;
  };
  pitchRange?: {
    minNote?: string;
    maxNote?: string;
  };
};

let mediaStream: MediaStream | null = null;
let mediaRecorder: MediaRecorder | null = null;
let micAnalysisAbortController: AbortController | null = null;
let micAnalysisStopRequested = false;
let micChunks: BlobPart[] = [];
let recordedMicFile: File | null = null;
let lastAnalyzedFile: File | null = null;
let lockedModelName = "";
let recordedMicObjectUrl = "";
let debugRafId: number | null = null;
let playbackRate = 1;
let abLoopStartTime: number | null = null;
let abLoopEndTime: number | null = null;
let abLoopEnabledState = false;
type MicAnalysisState = "idle" | "running" | "stopping";
let micLiveAnalysisState: MicAnalysisState = "idle";
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const BINS_PER_SEMITONE = 5;
const MIN_PITCH_NOTE = "C1";
const MAX_PITCH_NOTE = "C7";
const DEFAULT_PITCH_MIN_NOTE = "C2";
const DEFAULT_PITCH_MAX_NOTE = "C6";
const MIN_PLAYBACK_RATE = 0.5;
const MAX_PLAYBACK_RATE = 3;

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function formatAudioTime(seconds: number | null | undefined): string {
  if (!Number.isFinite(Number(seconds)) || Number(seconds) < 0) {
    return t("playback.notSet");
  }
  const safe = Math.max(0, Number(seconds));
  const totalMillis = Math.round(safe * 1000);
  const wholeSeconds = Math.floor(totalMillis / 1000);
  const millis = totalMillis % 1000;
  const minutes = Math.floor(wholeSeconds / 60);
  const remainingSeconds = wholeSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function formatPlaybackRate(rate: number): string {
  return `${clampNumber(Number(rate) || 1, MIN_PLAYBACK_RATE, MAX_PLAYBACK_RATE).toFixed(2)}x`;
}

function setMicLiveAnalysisState(nextState: MicAnalysisState): void {
  micLiveAnalysisState = nextState;
  syncMicButtons();
}

function beginMicLiveAnalysis(): void {
  setMicLiveAnalysisState("running");
}

function resetMicLiveAnalysisState(): void {
  setMicLiveAnalysisState("idle");
}

function getSpectrumStrideForZoom({
  zoom,
  minZoom = 1,
  maxZoom = 20,
  minUnitsPerSecond = 5,
  maxUnitsPerSecond = 100,
  frameRateHz = 100,
}: {
  zoom: number;
  minZoom?: number;
  maxZoom?: number;
  minUnitsPerSecond?: number;
  maxUnitsPerSecond?: number;
  frameRateHz?: number;
}): number {
  const loZoom = Math.max(1, Number.isFinite(minZoom) ? minZoom : 1);
  const hiZoom = Math.max(loZoom, Number.isFinite(maxZoom) ? maxZoom : loZoom);
  const safeZoom = clampNumber(Number.isFinite(zoom) ? zoom : loZoom, loZoom, hiZoom);
  const t = hiZoom === loZoom ? 1 : (safeZoom - loZoom) / (hiZoom - loZoom);
  const loUps = Math.max(1, Number.isFinite(minUnitsPerSecond) ? minUnitsPerSecond : 5);
  const hiUps = Math.max(loUps, Number.isFinite(maxUnitsPerSecond) ? maxUnitsPerSecond : 100);
  const unitsPerSecond = loUps + (hiUps - loUps) * clampNumber(t, 0, 1);
  const safeFrameRate = Math.max(1, Number.isFinite(frameRateHz) ? frameRateHz : 100);
  return Math.max(1, Math.round(safeFrameRate / Math.max(1e-6, unitsPerSecond)));
}

function noteToBin(note: string): number {
  const match = /^([A-G]#?)(-?\d+)$/.exec(String(note || "").trim());
  if (!match) return 0;
  const noteName = match[1] || "C";
  const octave = Number(match[2]);
  const idx = NOTE_NAMES.indexOf(noteName);
  if (idx < 0 || !Number.isFinite(octave)) return 0;
  const semitone = (octave - 1) * 12 + idx;
  return Math.max(0, semitone * BINS_PER_SEMITONE);
}

function binToNote(bin: number): string {
  const clamped = Math.max(0, Math.floor(Number(bin) || 0));
  const semitone = Math.floor(clamped / BINS_PER_SEMITONE);
  const noteName = NOTE_NAMES[semitone % 12] || "C";
  const octave = 1 + Math.floor(semitone / 12);
  return `${noteName}${octave}`;
}

function clampPitchBin(bin: number): number {
  const minBin = noteToBin(MIN_PITCH_NOTE);
  const maxBin = noteToBin(MAX_PITCH_NOTE);
  const safe = Math.floor(Number(bin) || 0);
  return Math.max(minBin, Math.min(maxBin, safe));
}

function buildPitchNoteOptions(minOctave = 0, maxOctave = 8): string[] {
  const out: string[] = [];
  for (let octave = minOctave; octave <= maxOctave; octave += 1) {
    for (const note of NOTE_NAMES) {
      out.push(`${note}${octave}`);
    }
  }
  return out;
}

function formatDebugNumber(value: number | null | undefined, digits = 2): string {
  if (!Number.isFinite(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function formatDebugStateBlock(): string {
  const debugState = ui.getDebugState();
  const uiState = ui.getState();
  const displayRows = Math.max(1, debugState.spectral.displayMaxBin - debugState.spectral.displayMinBin + 1);
  const viewCols = Math.max(1, Math.floor(debugState.viewport.viewFrames));
  const totalCols = Math.max(1, Math.floor(debugState.viewport.totalFrames));
  const render = debugState.render;
  const deltaX =
    Number.isFinite(Number(debugState.hover.canvasX)) && Number.isFinite(Number(debugState.hover.frameX))
      ? Number(debugState.hover.canvasX) - Number(debugState.hover.frameX)
      : NaN;
  const deltaY =
    Number.isFinite(Number(debugState.hover.canvasY)) && Number.isFinite(Number(debugState.hover.frameY))
      ? Number(debugState.hover.canvasY) - Number(debugState.hover.frameY)
      : NaN;
  const spectrumStride = Math.max(
    1,
    getSpectrumStrideForZoom({
      zoom: debugState.viewport.zoom,
      minZoom: 1,
      maxZoom: 20,
      minUnitsPerSecond: uiState.displaySampling.minUnitsPerSecond,
      maxUnitsPerSecond: uiState.displaySampling.maxUnitsPerSecond,
      frameRateHz: 100,
    }),
  );
  const spectrumUps = Math.max(1, Math.round(100 / spectrumStride));
  const sampledColsByUps = Math.max(1, Math.ceil(viewCols / spectrumStride));
  const formatTaskFps = (stats: { actualFps: number; targetFps: number }) =>
    `${formatDebugNumber(stats.actualFps, 1)}/${formatDebugNumber(stats.targetFps, 0)}fps`;
  const dirty = debugState.render.dirty;
  const format01 = (value: number) => (value ? "1" : "0");

  return [
    `viewport`,
    `  totalFrames=${debugState.viewport.totalFrames} viewFrames=${debugState.viewport.viewFrames} offset=${formatDebugNumber(debugState.viewport.offsetFrames, 2)} zoom=${formatDebugNumber(debugState.viewport.zoom, 3)}`,
    `spectral`,
    `  bins=${debugState.spectral.totalBins} display=${debugState.spectral.displayMinBin}-${debugState.spectral.displayMaxBin} xPerFrame=${formatDebugNumber(debugState.spectral.xPerFrame, 2)} yPerBin=${formatDebugNumber(debugState.spectral.yPerBin, 2)}`,
    `render`,
    `  mainBase      fps=${formatTaskFps(render.mainBase)} rows=${render.mainBase.rows} cols=${render.mainBase.cols} stride=${render.mainBase.sampleStrideFrames} px=${render.mainBase.canvasWidth}x${render.mainBase.canvasHeight} dpr=${formatDebugNumber(render.mainBase.dpr, 2)}`,
    `  mainOverlay   fps=${formatTaskFps(render.mainOverlay)} target=${formatDebugNumber(render.mainOverlay.targetFps, 0)}`,
    `  overviewBase  fps=${formatTaskFps(render.overviewBase)} rows=${render.overviewBase.rows} cols=${render.overviewBase.cols} stride=${render.overviewBase.sampleStrideFrames} px=${render.overviewBase.canvasWidth}x${render.overviewBase.canvasHeight} dpr=${formatDebugNumber(render.overviewBase.dpr, 2)}`,
    `  overviewOverlay fps=${formatTaskFps(render.overviewOverlay)} target=${formatDebugNumber(render.overviewOverlay.targetFps, 0)}`,
    `dirty`,
    `  mask=${dirty.currentMask} force=${dirty.currentMask !== 0 ? "yes" : "no"} playing=${dirty.playing ? "yes" : "no"} scheduler=${dirty.schedulerRunning ? "on" : "off"}`,
    `  bits mainBase=${format01(dirty.mainBase)} mainOverlay=${format01(dirty.mainOverlay)} overviewBase=${format01(dirty.overviewBase)} overviewOverlay=${format01(dirty.overviewOverlay)}`,
    `sampling`,
    `  mode=${uiState.displaySampling.representativeMode} minUPS=${uiState.displaySampling.minUnitsPerSecond} maxUPS=${uiState.displaySampling.maxUnitsPerSecond} stride=${spectrumStride} (~${spectrumUps} UPS @100Hz)`,
    `draw-space`,
    `  pitchBins=${displayRows} sampledCols=${sampledColsByUps} sourceViewFrames=${viewCols} totalFrames=${totalCols}`,
    `prediction`,
    `  frameCount=${debugState.prediction.frameCount} confidence=${debugState.prediction.confidenceCount} inferenceTotal=${debugState.prediction.inferenceTotalCount}`,
    `hover`,
    `  frame=${debugState.hover.frame} bin=${formatDebugNumber(debugState.hover.frameBin, 2)} dx=${formatDebugNumber(deltaX, 2)} dy=${formatDebugNumber(deltaY, 2)}`,
  ].join("\n");
}

function renderDebugPanel(): void {
  if (!debugInfo) return;
  const enabled = debugPanelEnabled?.checked !== false;
  debugInfo.style.display = enabled ? "block" : "none";
  if (!enabled) {
    debugInfo.textContent = t("debug.panelOff");
    return;
  }
  const block = formatDebugStateBlock();
  const items = buildDebugItemsFromBlock(block);
  debugInfo.innerHTML = items
    .map(
      (it) =>
        `<div class="debug-item"><div class="debug-title">${escapeHtml(it.title)}</div><div class="debug-content">${escapeHtml(
          it.content,
        )}</div></div>`,
    )
    .join("");
}

function escapeHtml(value: string): string {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function buildDebugItemsFromBlock(block: string): Array<{ title: string; content: string }> {
  const lines = block.split(/\r?\n/);
  const items: Array<{ title: string; content: string }> = [];
  let curTitle: string | null = null;
  let curContentLines: string[] = [];
  for (const rawLine of lines) {
    const line = rawLine.replace(/\s+$/g, "");
    if (/^\S/.test(line)) {
      if (curTitle !== null) {
        items.push({ title: curTitle, content: curContentLines.join("\n").trim() });
      }
      curTitle = line.trim();
      curContentLines = [];
    } else {
      curContentLines.push(line.replace(/^\s+/, ""));
    }
  }
  if (curTitle !== null) {
    items.push({ title: curTitle, content: curContentLines.join("\n").trim() });
  }
  return items;
}

function ensureDebugLoop(): void {
  if (debugRafId !== null || typeof window === "undefined") return;
  const tick = () => {
    debugRafId = null;
    renderDebugPanel();
    if (debugPanelEnabled?.checked !== false) {
      debugRafId = window.requestAnimationFrame(tick);
    }
  };
  debugRafId = window.requestAnimationFrame(tick);
}

function stopDebugLoop(): void {
  if (debugRafId === null || typeof window === "undefined") return;
  window.cancelAnimationFrame(debugRafId);
  debugRafId = null;
}

function log(message: string): void {
  const time = new Date().toLocaleTimeString();
  const next = `[${time}] ${message}`;
  console.log(next);
}

function translateStatus(status: string): string {
  const key = STATUS_LABEL_KEYS[status] ?? `status.${status}`;
  return t(key);
}

function translateRuntimeMessage(message: string | null | undefined): string {
  switch (message) {
    case "Analyzing audio...":
      return t("status.analyzing");
    case "Cancelled":
      return t("status.cancelledMessage");
    case "Analysis failed":
      return t("status.failedMessage");
    case null:
    case undefined:
      return "";
    default:
      return message;
  }
}

function syncLocalizedUi(): void {
  applyLocalizedText(document);
  document.title = t("app.title");
  syncUiStatus(ui.getState().status);
  syncAbLoopUi();
  syncMicButtons();
}

function populateLanguageOptions(): void {
  if (!languageSelect) return;
  const current = getAppLanguage();
  languageSelect.replaceChildren();
  for (const language of SUPPORTED_LANGUAGES) {
    const option = document.createElement("option");
    option.value = language;
    option.textContent = t(`language.options.${language}`);
    languageSelect.appendChild(option);
  }
  languageSelect.value = current;
}

function loadAppSettings(): AppSettings {
  if (typeof localStorage === "undefined") {
    return {};
  }
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as AppSettings;
    if (!parsed || typeof parsed !== "object") {
      return {};
    }
    return parsed;
  } catch {
    return {};
  }
}

function saveAppSettings(next: Partial<AppSettings>): void {
  if (typeof localStorage === "undefined") {
    return;
  }
  try {
    const current = loadAppSettings();
    const merged: AppSettings = {
      ...current,
      ...next,
    sections: {
      ...current.sections,
      ...next.sections,
      main: {
        ...current.sections?.main,
        ...next.sections?.main,
      },
      overview: {
        ...current.sections?.overview,
        ...next.sections?.overview,
      },
    },
    abLoop: {
      ...current.abLoop,
      ...next.abLoop,
    },
  };
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(merged));
  } catch {}
}

function registerAppServiceWorker(): void {
  if (!("serviceWorker" in navigator)) {
    return;
  }
  if (import.meta.env.DEV) {
  void navigator.serviceWorker.getRegistrations().then(async (registrations) => {
      await Promise.all(registrations.map((registration) => registration.unregister()));
      if ("caches" in window) {
        const keys = await caches.keys();
        await Promise.all(
          keys
            .filter((key) => key.startsWith("ohm-app-"))
            .map((key) => caches.delete(key)),
        );
      }
    });
    return;
  }
  void navigator.serviceWorker.register("./sw.js").then(async (registration) => {
    try {
      await navigator.serviceWorker.ready;
      const controller = navigator.serviceWorker.controller ?? registration.active ?? registration.waiting ?? registration.installing;
      controller?.postMessage({ type: "warm-default-assets" });
    } catch (error) {
      log(t("logs.pwaWarmFailed", { message: error instanceof Error ? error.message : String(error) }));
    }
  }).catch((error) => {
    log(t("logs.pwaRegisterFailed", { message: error instanceof Error ? error.message : String(error) }));
  });
}

function syncUiStatus(status: string): void {
  if (!uiStatus) return;
  uiStatus.textContent = translateStatus(status);
  uiStatus.dataset.status = status;
}

function populateModels(): void {
  if (!modelNameSelect) return;
  const current = modelNameSelect.value || modelEntries[0] || CORE_MODEL_DEFAULT_NAME;
  modelNameSelect.replaceChildren();
  for (const modelName of modelEntries) {
    const option = document.createElement("option");
    option.value = modelName;
    option.textContent = modelName;
    modelNameSelect.appendChild(option);
  }
  if (modelEntries.some((name) => name === current)) {
    modelNameSelect.value = current;
  }
}

function populatePitchRangeOptions(): void {
  const options = buildPitchNoteOptions(1, 7);
  if (pitchRangeMinNote) {
    const current = pitchRangeMinNote.value;
    pitchRangeMinNote.replaceChildren();
    for (const note of options) {
      const option = document.createElement("option");
      option.value = note;
      option.textContent = note;
      pitchRangeMinNote.appendChild(option);
    }
    pitchRangeMinNote.value = options.includes(current) ? current : DEFAULT_PITCH_MIN_NOTE;
  }
  if (pitchRangeMaxNote) {
    const current = pitchRangeMaxNote.value;
    pitchRangeMaxNote.replaceChildren();
    for (const note of options) {
      const option = document.createElement("option");
      option.value = note;
      option.textContent = note;
      pitchRangeMaxNote.appendChild(option);
    }
    pitchRangeMaxNote.value = options.includes(current) ? current : DEFAULT_PITCH_MAX_NOTE;
  }
}

function syncPitchRangeConfig(): void {
  const minNote = pitchRangeMinNote?.value || MIN_PITCH_NOTE;
  const maxNote = pitchRangeMaxNote?.value || MAX_PITCH_NOTE;
  const minBin = clampPitchBin(noteToBin(minNote));
  const maxBin = clampPitchBin(noteToBin(maxNote));
  const resolvedMin = Math.min(minBin, maxBin);
  const resolvedMax = Math.max(minBin, maxBin);
  const normalizedMinNote = binToNote(resolvedMin);
  const normalizedMaxNote = binToNote(resolvedMax);
  if (pitchRangeMinNote) {
    pitchRangeMinNote.value = normalizedMinNote;
  }
  if (pitchRangeMaxNote) {
    pitchRangeMaxNote.value = normalizedMaxNote;
  }
  ui.setPitchRange({
    minBin: resolvedMin,
    maxBin: resolvedMax,
  });
  saveAppSettings({
    pitchRange: {
      minNote: normalizedMinNote,
      maxNote: normalizedMaxNote,
    },
  });
}

function readCurrentAudioTime(): number {
  if (!audioPlayer) {
    return 0;
  }
  const currentTime = Number(audioPlayer.currentTime);
  return Number.isFinite(currentTime) && currentTime >= 0 ? currentTime : 0;
}

function normalizeAbLoopPoints(): void {
  if (abLoopStartTime !== null && abLoopEndTime !== null && abLoopEndTime < abLoopStartTime) {
    [abLoopStartTime, abLoopEndTime] = [abLoopEndTime, abLoopStartTime];
  }
  if (!audioPlayer || !Number.isFinite(Number(audioPlayer.duration)) || Number(audioPlayer.duration) <= 0) {
    return;
  }
  const duration = Number(audioPlayer.duration);
  if (abLoopStartTime !== null) {
    abLoopStartTime = clampNumber(abLoopStartTime, 0, duration);
  }
  if (abLoopEndTime !== null) {
    abLoopEndTime = clampNumber(abLoopEndTime, 0, duration);
  }
  if (abLoopStartTime !== null && abLoopEndTime !== null && abLoopEndTime <= abLoopStartTime) {
    abLoopEndTime = Math.min(duration, abLoopStartTime + 0.1);
  }
}

function syncAbLoopUi(): void {
  normalizeAbLoopPoints();
  if (abLoopEnabled) {
    abLoopEnabled.checked = abLoopEnabledState;
  }
  if (abLoopStart) {
    abLoopStart.value = formatAudioTime(abLoopStartTime);
  }
  if (abLoopEnd) {
    abLoopEnd.value = formatAudioTime(abLoopEndTime);
  }
  if (audioPlayer) {
    audioPlayer.loop = false;
  }
}

function setAbLoopEnabled(enabled: boolean, persist = true): void {
  abLoopEnabledState = enabled;
  if (abLoopEnabled) {
    abLoopEnabled.checked = enabled;
  }
  syncAbLoopUi();
  if (persist) {
    saveAppSettings({
      abLoop: {
        enabled: abLoopEnabledState,
        start: abLoopStartTime,
        end: abLoopEndTime,
      },
    });
  }
}

function setAbLoopPoint(which: "start" | "end", seconds: number | null, persist = true): void {
  const safeSeconds = Number.isFinite(Number(seconds)) && Number(seconds) >= 0 ? Number(seconds) : null;
  if (which === "start") {
    abLoopStartTime = safeSeconds;
  } else {
    abLoopEndTime = safeSeconds;
  }
  normalizeAbLoopPoints();
  syncAbLoopUi();
  if (persist) {
    saveAppSettings({
      abLoop: {
        enabled: abLoopEnabledState,
        start: abLoopStartTime,
        end: abLoopEndTime,
      },
    });
  }
}

function clearAbLoop(persist = true): void {
  abLoopStartTime = null;
  abLoopEndTime = null;
  abLoopEnabledState = false;
  syncAbLoopUi();
  if (persist) {
    saveAppSettings({
      abLoop: {
        enabled: false,
        start: null,
        end: null,
      },
    });
  }
}

function applyPlaybackRate(nextRate: number, persist = true): void {
  const safeRate = clampNumber(Number(nextRate) || 1, MIN_PLAYBACK_RATE, MAX_PLAYBACK_RATE);
  playbackRate = safeRate;
  if (audioPlayer && Math.abs(audioPlayer.playbackRate - safeRate) > 1e-6) {
    audioPlayer.playbackRate = safeRate;
  }
  if (playbackRateSlider) {
    playbackRateSlider.value = String(safeRate);
  }
  if (playbackRateValue) {
    playbackRateValue.textContent = formatPlaybackRate(safeRate);
  }
  if (persist) {
    saveAppSettings({ audioPlaybackRate: safeRate });
  }
}

function snapAudioToAbLoopStart(): void {
  if (!audioPlayer || abLoopStartTime === null || abLoopEndTime === null || abLoopEndTime <= abLoopStartTime) {
    return;
  }
  if (audioPlayer.currentTime >= abLoopEndTime) {
    audioPlayer.currentTime = abLoopStartTime;
    void audioPlayer.play().catch(() => {});
  }
}

function setDropHint(active: boolean): void {
  if (!fileDropHint) return;
  fileDropHint.dataset.active = active ? "true" : "false";
  fileDropHint.setAttribute("aria-hidden", active ? "false" : "true");
}

function revokeCurrentAudioUrl(): void {
  if (!currentAudioObjectUrl) return;
  try {
    URL.revokeObjectURL(currentAudioObjectUrl);
  } catch {}
  currentAudioObjectUrl = "";
}

function revokeRecordedMicUrl(): void {
  if (!recordedMicObjectUrl) return;
  try {
    URL.revokeObjectURL(recordedMicObjectUrl);
  } catch {}
  recordedMicObjectUrl = "";
}

function bindAudioFile(file: File | null): void {
  if (!file || !audioPlayer) {
    return;
  }

  revokeCurrentAudioUrl();
  currentAudioObjectUrl = URL.createObjectURL(file);
  audioPlayer.src = currentAudioObjectUrl;
  audioPlayer.load();
  applyPlaybackRate(playbackRate, false);
  syncAbLoopUi();
  log(t("logs.audioLoaded", { name: file.name }));
}

function handleAudioTimeUpdate(): void {
  snapAudioToAbLoopStart();
}

function handleAudioLoadedMetadata(): void {
  normalizeAbLoopPoints();
  syncAbLoopUi();
  applyPlaybackRate(playbackRate, false);
}

function handlePlaybackRateChange(): void {
  if (!audioPlayer) {
    return;
  }
  applyPlaybackRate(audioPlayer.playbackRate, true);
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tagName = target.tagName.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select" || target.isContentEditable;
}

function handleGlobalKeyDown(event: KeyboardEvent): void {
  if (!event.altKey) {
    return;
  }
  const key = String(event.key || "").toLowerCase();
  if (!key) {
    return;
  }
  if (event.ctrlKey || event.metaKey) {
    return;
  }

  const shouldIgnoreEditable = isEditableTarget(event.target);
  if (shouldIgnoreEditable && key !== "l" && key !== "a" && key !== "b" && key !== "c") {
    return;
  }

  if (key === "a" && !event.shiftKey) {
    event.preventDefault();
    setAbLoopPoint("start", readCurrentAudioTime());
    setAbLoopEnabled(true);
    log(t("logs.aSet", { time: formatAudioTime(abLoopStartTime) }));
    return;
  }
  if (key === "b" && !event.shiftKey) {
    event.preventDefault();
    setAbLoopPoint("end", readCurrentAudioTime());
    setAbLoopEnabled(true);
    log(t("logs.bSet", { time: formatAudioTime(abLoopEndTime) }));
    return;
  }
  if (key === "l") {
    event.preventDefault();
    setAbLoopEnabled(!abLoopEnabledState);
    log(abLoopEnabledState ? t("logs.loopOn") : t("logs.loopOff"));
    return;
  }
  if (key === "c" && event.shiftKey) {
    event.preventDefault();
    clearAbLoop();
    log(t("logs.loopCleared"));
  }
}

async function analyzeCurrentAudio(sourceFile: File): Promise<void> {
  lastAnalyzedFile = sourceFile;
  bindAudioFile(sourceFile);
  ui.setAudioElement(audioPlayer);

  try {
    await ui.analyze({
      input: buildAnalyzeInput(sourceFile),
      execution: {
        forceRefresh: forceRefreshCheckbox?.checked === true,
      },
    });
    log(t("logs.analysisCompleted", { name: sourceFile.name }));
  } catch (error) {
    log(t("logs.analysisFailed", { message: error instanceof Error ? error.message : String(error) }));
  } finally {
    updateStateFromUi();
  }
}

function getCurrentAnalyzableFile(): File | null {
  const uploadFile = audioFileInput?.files && audioFileInput.files[0] ? audioFileInput.files[0] : null;
  return uploadFile || recordedMicFile || lastAnalyzedFile;
}

function syncSectionConfig(): void {
  const minUps = Math.min(100, Math.max(1, Number(minUnitsPerSecond?.value || 5)));
  const maxUps = Math.min(100, Math.max(minUps, Number(maxUnitsPerSecond?.value || 100)));
  ui.setSections({
    main: {
      enabled: mainEnabled?.checked !== false,
      overlay: mainEnabled?.checked !== false,
      fps: Number(mainFps?.value || 60),
      overlayFps: Number(mainOverlayFps?.value || 60),
    },
    overview: {
      enabled: overviewEnabled?.checked !== false,
      overlay: overviewEnabled?.checked !== false,
      fps: Number(overviewFps?.value || 24),
      overlayFps: Number(overviewOverlayFps?.value || 24),
    },
  });
  ui.setRefreshRate({
    main: Number(mainFps?.value || 60),
    mainOverlay: Number(mainOverlayFps?.value || 60),
    overview: Number(overviewFps?.value || 0),
    overviewOverlay: Number(overviewOverlayFps?.value || 0),
  });
  ui.setDisplaySampling({
    representativeMode: (representativeMode?.value || "highest-confidence") as RepresentativeMode,
    minUnitsPerSecond: minUps,
    maxUnitsPerSecond: maxUps,
  });
  saveAppSettings({
    sections: {
      main: {
        enabled: mainEnabled?.checked !== false,
        fps: Number(mainFps?.value || 60),
        overlayFps: Number(mainOverlayFps?.value || 60),
      },
      overview: {
        enabled: overviewEnabled?.checked !== false,
        fps: Number(overviewFps?.value || 0),
        overlayFps: Number(overviewOverlayFps?.value || 0),
      },
    },
    displaySampling: {
      representativeMode: (representativeMode?.value || "highest-confidence") as RepresentativeMode,
      minUnitsPerSecond: minUps,
      maxUnitsPerSecond: maxUps,
    },
  });
}

function readSelectedModelName(): string {
  return modelNameSelect?.value || CORE_MODEL_DEFAULT_NAME;
}

function buildAnalyzeInput(file: File): {
  source: { kind: "file"; file: File; label: string };
  model: { name: string };
  fileKey: string;
} {
  return {
    source: {
      kind: "file",
      file,
      label: file.name,
    },
    model: {
      name: readSelectedModelName(),
    },
    fileKey: file.name,
  };
}

function buildAnalyzeStreamInput(stream: MediaStream): {
  source: { kind: "stream"; stream: MediaStream; label: string };
  model: { name: string };
  fileKey: string;
} {
  return {
    source: {
      kind: "stream",
      stream,
      label: "mic-live",
    },
    model: {
      name: readSelectedModelName(),
    },
    fileKey: "mic-live",
  };
}

function updateStateFromUi(): void {
  if (uiStatus) {
    syncUiStatus(ui.getState().status);
  }
}

function syncMicButtons(): void {
  const recording = !!mediaRecorder && mediaRecorder.state === "recording";
  if (startMicRecordBtn) {
    startMicRecordBtn.disabled = recording;
  }
  if (stopMicRecordBtn) {
    stopMicRecordBtn.disabled = !recording || micLiveAnalysisState === "idle";
  }
  if (exportMicRecordBtn) {
    exportMicRecordBtn.disabled = recording || !recordedMicFile;
  }
  if (micRecordStatus) {
    micRecordStatus.textContent = recording
      ? t("controls.recording")
      : recordedMicFile
        ? t("controls.recorded", { name: recordedMicFile.name })
        : t("controls.notRecorded");
  }
}

function setRecordedMicFile(file: File | null): void {
  recordedMicFile = file;
  revokeRecordedMicUrl();
  if (file && audioPlayer) {
    recordedMicObjectUrl = URL.createObjectURL(file);
    audioPlayer.src = recordedMicObjectUrl;
    audioPlayer.load();
    applyPlaybackRate(playbackRate, false);
    syncAbLoopUi();
  }
  syncMicButtons();
  if (file && micLiveAnalysisState === "idle") {
    void analyzeCurrentAudio(file);
  }
}

function analyzeCurrentStream(stream: MediaStream): void {
  if (micAnalysisStopRequested) {
    log(t("logs.micAnalysisStopped"));
    return;
  }
  micAnalysisAbortController?.abort(t("logs.micAnalysisRestarted"));
  micAnalysisAbortController = new AbortController();
  beginMicLiveAnalysis();
  void ui
    .analyze({
      input: buildAnalyzeStreamInput(stream),
      execution: {
        forceRefresh: true,
        signal: micAnalysisAbortController.signal,
      },
    })
    .then(() => {
      log(t("logs.streamAnalysisCompleted"));
    })
    .catch((error) => {
      log(t("logs.streamAnalysisFailed", { message: error instanceof Error ? error.message : String(error) }));
    })
    .finally(() => {
      micAnalysisAbortController = null;
      resetMicLiveAnalysisState();
    });
}

async function startMicRecording(): Promise<void> {
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
    log(t("logs.micUnsupported"));
    return;
  }
  if (mediaRecorder && mediaRecorder.state === "recording") {
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micAnalysisStopRequested = false;
    micChunks = [];
    const recorder = new MediaRecorder(mediaStream);
    mediaRecorder = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        micChunks.push(event.data);
      }
    };
    recorder.onstop = () => {
      const mimeType = recorder.mimeType || "audio/webm";
      const blob = new Blob(micChunks, { type: mimeType });
      const suffix = mimeType.includes("ogg") ? "ogg" : mimeType.includes("mp4") ? "m4a" : "webm";
      const file = new File([blob], `mic-record-${Date.now()}.${suffix}`, { type: mimeType });
      setRecordedMicFile(file);
      log(t("logs.micRecorded", { name: file.name }));
      micChunks = [];
      syncMicButtons();
    };
    recorder.start();
    analyzeCurrentStream(mediaStream);
    syncMicButtons();
    log(t("logs.micRecordingStarted"));
  } catch (error) {
    log(t("logs.micRecordingFailed", { message: error instanceof Error ? error.message : String(error) }));
    stopMicRecording();
  }
}

function stopMicRecording(): void {
  micAnalysisStopRequested = true;
  ui.cancel(t("logs.micRecordStoppedReason"));
  micAnalysisAbortController?.abort(t("logs.micRecordStoppedReason"));
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
  if (mediaStream) {
    for (const track of mediaStream.getTracks()) {
      try {
        track.stop();
      } catch {}
    }
    mediaStream = null;
  }
  if (micLiveAnalysisState === "running") {
    setMicLiveAnalysisState("stopping");
  } else {
    resetMicLiveAnalysisState();
  }
  syncMicButtons();
}

function exportMicRecording(): void {
  if (!recordedMicFile) {
    log(t("logs.noMicRecording"));
    return;
  }
  const url = URL.createObjectURL(recordedMicFile);
  const link = document.createElement("a");
  link.href = url;
  link.download = recordedMicFile.name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => {
    try {
      URL.revokeObjectURL(url);
    } catch {}
  }, 0);
  log(t("logs.micExported", { name: recordedMicFile.name }));
}

ui.subscribe((state) => {
  syncUiStatus(state.status);

  if (modelNameSelect) {
    const running = state.status === "running";
    modelNameSelect.disabled = running;
    if (running) {
      lockedModelName = modelNameSelect.value || lockedModelName || CORE_MODEL_DEFAULT_NAME;
    }
  }

  if (state.cfp.length !== previousCfpLen) {
    previousCfpLen = state.cfp.length;
    log(t("logs.cfpUpdated", { count: state.cfp.length }));
  }

  const inferenceCount = state.inference ? state.inference.totalArgmax.length : 0;
  if (inferenceCount !== previousInferenceCount) {
    previousInferenceCount = inferenceCount;
    if (state.inference) {
      log(t("logs.inferenceUpdated", { count: state.inference.totalArgmax.length }));
    }
  }

  if (state.status !== previousStatus) {
    previousStatus = state.status;
    const translatedMessage = translateRuntimeMessage(state.message);
    log(t("logs.uiStatus", {
      status: translateStatus(state.status),
      message: translatedMessage ? ` - ${translatedMessage}` : "",
    }));
  }

  updateStateFromUi();
  renderDebugPanel();
});

const savedSettings = loadAppSettings();
populateModels();
populatePitchRangeOptions();
populateLanguageOptions();
if (modelNameSelect && savedSettings.modelName) {
  modelNameSelect.value = savedSettings.modelName;
}
if (forceRefreshCheckbox && typeof savedSettings.forceRefresh === "boolean") {
  forceRefreshCheckbox.checked = savedSettings.forceRefresh;
}
if (debugPanelEnabled && typeof savedSettings.debugPanelEnabled === "boolean") {
  debugPanelEnabled.checked = savedSettings.debugPanelEnabled;
}
if (mainEnabled && typeof savedSettings.sections?.main?.enabled === "boolean") {
  mainEnabled.checked = savedSettings.sections.main.enabled;
}
if (mainFps && Number.isFinite(savedSettings.sections?.main?.fps)) {
  mainFps.value = String(savedSettings.sections!.main!.fps);
}
if (mainOverlayFps && Number.isFinite(savedSettings.sections?.main?.overlayFps)) {
  mainOverlayFps.value = String(savedSettings.sections!.main!.overlayFps);
}
if (overviewEnabled && typeof savedSettings.sections?.overview?.enabled === "boolean") {
  overviewEnabled.checked = savedSettings.sections.overview.enabled;
}
if (overviewFps && Number.isFinite(savedSettings.sections?.overview?.fps)) {
  overviewFps.value = String(savedSettings.sections!.overview!.fps);
}
if (overviewOverlayFps && Number.isFinite(savedSettings.sections?.overview?.overlayFps)) {
  overviewOverlayFps.value = String(savedSettings.sections!.overview!.overlayFps);
}
if (representativeMode && savedSettings.displaySampling?.representativeMode) {
  representativeMode.value = savedSettings.displaySampling.representativeMode;
}
if (minUnitsPerSecond && Number.isFinite(savedSettings.displaySampling?.minUnitsPerSecond)) {
  minUnitsPerSecond.value = String(savedSettings.displaySampling!.minUnitsPerSecond);
}
if (maxUnitsPerSecond && Number.isFinite(savedSettings.displaySampling?.maxUnitsPerSecond)) {
  maxUnitsPerSecond.value = String(savedSettings.displaySampling!.maxUnitsPerSecond);
}
if (pitchRangeMinNote && savedSettings.pitchRange?.minNote) {
  pitchRangeMinNote.value = savedSettings.pitchRange.minNote;
}
if (pitchRangeMaxNote && savedSettings.pitchRange?.maxNote) {
  pitchRangeMaxNote.value = savedSettings.pitchRange.maxNote;
}
const savedAbLoop = savedSettings.abLoop ?? {};
if (Number.isFinite(savedSettings.audioPlaybackRate)) {
  playbackRate = clampNumber(Number(savedSettings.audioPlaybackRate), MIN_PLAYBACK_RATE, MAX_PLAYBACK_RATE);
}
if (typeof savedAbLoop.enabled === "boolean") {
  abLoopEnabledState = savedAbLoop.enabled;
}
if (Number.isFinite(savedAbLoop.start)) {
  abLoopStartTime = Number(savedAbLoop.start);
}
if (Number.isFinite(savedAbLoop.end)) {
  abLoopEndTime = Number(savedAbLoop.end);
}
if (modelNameSelect) {
  saveAppSettings({ modelName: readSelectedModelName() });
}
if (forceRefreshCheckbox) {
  saveAppSettings({ forceRefresh: forceRefreshCheckbox.checked });
}
if (debugPanelEnabled) {
  saveAppSettings({ debugPanelEnabled: debugPanelEnabled.checked });
}
applyPlaybackRate(playbackRate, false);
if (abLoopEnabled) {
  abLoopEnabled.checked = abLoopEnabledState;
}
syncSectionConfig();
syncPitchRangeConfig();
ui.setDisplaySampling({
  representativeMode: (representativeMode?.value || "highest-confidence") as RepresentativeMode,
  minUnitsPerSecond: Math.max(1, Number(minUnitsPerSecond?.value || 5)),
    maxUnitsPerSecond: Math.min(
      100,
      Math.max(
        Math.max(1, Number(minUnitsPerSecond?.value || 5)),
        Number(maxUnitsPerSecond?.value || 100),
      ),
    ),
});
syncLocalizedUi();
ui.mount(spectrumCanvasWrapper);
ui.setAudioElement(audioPlayer);
renderDebugPanel();
if (debugPanelEnabled?.checked !== false) {
  ensureDebugLoop();
}
log(t("logs.pageStarted"));
registerAppServiceWorker();

if (mainEnabled) {
  mainEnabled.addEventListener("change", syncSectionConfig);
}
if (overviewEnabled) {
  overviewEnabled.addEventListener("change", syncSectionConfig);
}
if (mainFps) {
  mainFps.addEventListener("change", syncSectionConfig);
}
if (mainOverlayFps) {
  mainOverlayFps.addEventListener("change", syncSectionConfig);
}
if (overviewFps) {
  overviewFps.addEventListener("change", syncSectionConfig);
}
if (overviewOverlayFps) {
  overviewOverlayFps.addEventListener("change", syncSectionConfig);
}
if (representativeMode) {
  representativeMode.addEventListener("change", syncSectionConfig);
}
if (minUnitsPerSecond) {
  minUnitsPerSecond.addEventListener("change", syncSectionConfig);
}
if (maxUnitsPerSecond) {
  maxUnitsPerSecond.addEventListener("change", syncSectionConfig);
}
if (pitchRangeMinNote) {
  pitchRangeMinNote.addEventListener("change", () => {
    syncPitchRangeConfig();
  });
}
if (pitchRangeMaxNote) {
  pitchRangeMaxNote.addEventListener("change", () => {
    syncPitchRangeConfig();
  });
}
if (modelNameSelect) {
  modelNameSelect.addEventListener("change", () => {
    if (ui.getState().status === "running") {
      if (lockedModelName) {
        modelNameSelect.value = lockedModelName;
      }
      log(t("logs.currentRunningIgnoreSwitch"));
      return;
    }
    saveAppSettings({ modelName: readSelectedModelName() });
    const file = getCurrentAnalyzableFile();
    if (!file) {
      log(t("logs.modelSwitchedNoFile"));
      return;
    }
    log(t("logs.modelSwitchReanalyze", { name: readSelectedModelName() }));
    void analyzeCurrentAudio(file);
  });
}
if (forceRefreshCheckbox) {
  forceRefreshCheckbox.addEventListener("change", () => {
    saveAppSettings({ forceRefresh: forceRefreshCheckbox.checked });
  });
}
if (debugPanelEnabled) {
  debugPanelEnabled.addEventListener("change", () => {
    saveAppSettings({ debugPanelEnabled: debugPanelEnabled.checked });
    renderDebugPanel();
    if (debugPanelEnabled.checked) {
      ensureDebugLoop();
    } else {
      stopDebugLoop();
    }
  });
}
if (languageSelect) {
  languageSelect.addEventListener("change", () => {
    void setAppLanguage(languageSelect.value).then(() => {
      populateLanguageOptions();
      syncLocalizedUi();
      renderDebugPanel();
    });
  });
}
if (playbackRateSlider) {
  playbackRateSlider.addEventListener("input", () => {
    applyPlaybackRate(Number(playbackRateSlider.value), true);
  });
}
  if (abLoopEnabled) {
    abLoopEnabled.addEventListener("change", () => {
      setAbLoopEnabled(abLoopEnabled.checked);
      log(abLoopEnabled.checked ? t("logs.loopOn") : t("logs.loopOff"));
    });
  }
  if (setLoopStartBtn) {
  setLoopStartBtn.addEventListener("click", () => {
    setAbLoopPoint("start", readCurrentAudioTime());
    setAbLoopEnabled(true);
    log(t("logs.aSet", { time: formatAudioTime(abLoopStartTime) }));
  });
}
if (setLoopEndBtn) {
  setLoopEndBtn.addEventListener("click", () => {
    setAbLoopPoint("end", readCurrentAudioTime());
    setAbLoopEnabled(true);
    log(t("logs.bSet", { time: formatAudioTime(abLoopEndTime) }));
  });
}
  if (clearLoopBtn) {
  clearLoopBtn.addEventListener("click", () => {
    clearAbLoop();
    log(t("logs.loopCleared"));
  });
}

if (audioFileInput) {
  audioFileInput.addEventListener("change", () => {
    const file = audioFileInput.files && audioFileInput.files[0] ? audioFileInput.files[0] : null;
    if (!file) {
      return;
    }
    void analyzeCurrentAudio(file);
  });
}

if (startMicRecordBtn) {
  startMicRecordBtn.addEventListener("click", () => {
    void startMicRecording();
  });
}

if (stopMicRecordBtn) {
  stopMicRecordBtn.addEventListener("click", () => {
    stopMicRecording();
  });
}

if (exportMicRecordBtn) {
  exportMicRecordBtn.addEventListener("click", () => {
    exportMicRecording();
  });
}

if (audioPlayer) {
  audioPlayer.addEventListener("timeupdate", handleAudioTimeUpdate);
  audioPlayer.addEventListener("loadedmetadata", handleAudioLoadedMetadata);
  audioPlayer.addEventListener("ratechange", handlePlaybackRateChange);
}

document.addEventListener("keydown", handleGlobalKeyDown);

function handleDrop(files: File[]): void {
  const file = files[0] || null;
  if (!file) return;
  if (audioFileInput) {
    const dt = new DataTransfer();
    dt.items.add(file);
    audioFileInput.files = dt.files;
  }
  log(t("logs.dragImport", { name: file.name }));
  void analyzeCurrentAudio(file);
}

let dragDepth = 0;
document.addEventListener("dragenter", (event) => {
  if (!event.dataTransfer) return;
  dragDepth += 1;
  setDropHint(true);
});
document.addEventListener("dragover", (event) => {
  if (!event.dataTransfer) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
  setDropHint(true);
});
document.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) {
    setDropHint(false);
  }
});
document.addEventListener("drop", (event) => {
  if (!event.dataTransfer) return;
  event.preventDefault();
  dragDepth = 0;
  setDropHint(false);
  const files = Array.from(event.dataTransfer.files || []).filter(Boolean);
  if (!files.length) return;
  handleDrop(files);
});

window.addEventListener("beforeunload", () => {
  stopDebugLoop();
  stopMicRecording();
  revokeCurrentAudioUrl();
  revokeRecordedMicUrl();
  ui.destroy();
});
