import {
  CORE_MODEL_DEFAULT_NAME,
  CORE_MODEL_NAMES,
} from "@ohm/core";
import { createAnalyzer } from "@ohm/runtime";
import { createSpectrumUi } from "@ohm/ui";
import type { RepresentativeMode } from "@ohm/ui";

const modelNameSelect = document.getElementById("modelName") as HTMLSelectElement | null;
const audioSourceSelect = document.getElementById("audioSourceSelect") as HTMLSelectElement | null;
const usePredictionCacheCheckbox = document.getElementById("usePredictionCacheCheckbox") as HTMLInputElement | null;
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
const debugPanelEnabled = document.getElementById("debugPanelEnabled") as HTMLInputElement | null;
const debugInfo = document.getElementById("debugInfo") as HTMLElement | null;
const audioPlayer = document.getElementById("audioPlayer") as HTMLAudioElement | null;
const spectrumCanvasWrapper = document.getElementById("spectrumCanvasWrapper") as HTMLElement | null;
const uiStatus = document.getElementById("uiStatus") as HTMLElement | null;
const fileDropHint = document.getElementById("fileDropHint") as HTMLElement | null;

const analyzer = createAnalyzer();
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
  audioSource?: string;
  usePredictionCache?: boolean;
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
let micChunks: BlobPart[] = [];
let recordedMicFile: File | null = null;
let lastAnalyzedFile: File | null = null;
let lockedModelName = "";
let recordedMicObjectUrl = "";
let debugRafId: number | null = null;
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const BINS_PER_SEMITONE = 5;
const MIN_PITCH_NOTE = "C1";
const MAX_PITCH_NOTE = "C7";

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
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

  return [
    `refresh: main=${uiState.sections.main.fps} mainOverlay=${uiState.sections.main.overlayFps} overview=${uiState.sections.overview.fps} overviewOverlay=${uiState.sections.overview.overlayFps}`,
    `sampling: mode=${uiState.displaySampling.representativeMode} minUPS=${uiState.displaySampling.minUnitsPerSecond} maxUPS=${uiState.displaySampling.maxUnitsPerSecond}`,
    `sampling-spectrum: stride=${spectrumStride} frames (~${spectrumUps} UPS @100Hz)`,
    `viewport: total=${debugState.viewport.totalFrames} view=${debugState.viewport.viewFrames} offset=${formatDebugNumber(debugState.viewport.offsetFrames, 2)} zoom=${formatDebugNumber(debugState.viewport.zoom, 3)}`,
    `draw-matrix: rows(pitchBins)=${displayRows} cols(sampledFrames)=${sampledColsByUps} sourceViewFrames=${viewCols} totalFrames=${totalCols}`,
    `prediction: inferenceVisible=${debugState.prediction.inferenceVisibleCount} inferenceTotal=${debugState.prediction.inferenceTotalCount}`,
    `hover: frame=${debugState.hover.frame} bin=${formatDebugNumber(debugState.hover.frameBin, 2)} dx=${formatDebugNumber(deltaX, 2)} dy=${formatDebugNumber(deltaY, 2)}`,
  ].join("\n");
}

function renderDebugPanel(): void {
  if (!debugInfo) return;
  const enabled = debugPanelEnabled?.checked !== false;
  debugInfo.style.display = enabled ? "block" : "none";
  if (!enabled) {
    debugInfo.textContent = "调试面板已关闭";
    return;
  }
  debugInfo.textContent = formatDebugStateBlock();
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

function loadAppSettings(): AppSettings {
  if (typeof localStorage === "undefined") {
    return {};
  }
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as AppSettings;
    return parsed && typeof parsed === "object" ? parsed : {};
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
    };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(merged));
  } catch {}
}

function registerAppServiceWorker(): void {
  if (!("serviceWorker" in navigator)) {
    return;
  }
  void navigator.serviceWorker.register("./sw.js").then(async (registration) => {
    try {
      await navigator.serviceWorker.ready;
      const controller = navigator.serviceWorker.controller ?? registration.active ?? registration.waiting ?? registration.installing;
      controller?.postMessage({ type: "warm-default-assets" });
    } catch (error) {
      log(`PWA 预热失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  }).catch((error) => {
    log(`PWA 缓存注册失败: ${error instanceof Error ? error.message : String(error)}`);
  });
}

function syncUiStatus(status: string): void {
  if (!uiStatus) return;
  uiStatus.textContent = status;
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
    pitchRangeMinNote.value = options.includes(current) ? current : MIN_PITCH_NOTE;
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
    pitchRangeMaxNote.value = options.includes(current) ? current : MAX_PITCH_NOTE;
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

function setDropHint(active: boolean): void {
  if (!fileDropHint) return;
  fileDropHint.dataset.active = active ? "true" : "false";
  fileDropHint.setAttribute("aria-hidden", active ? "false" : "true");
}

function ensureUploadAudioSource(): void {
  if (!audioSourceSelect) return;
  if (audioSourceSelect.value !== "upload") {
    audioSourceSelect.value = "upload";
    audioSourceSelect.dispatchEvent(new Event("change", { bubbles: true }));
  }
}

function ensureMicAudioSource(): void {
  if (!audioSourceSelect) return;
  if (audioSourceSelect.value !== "mic") {
    audioSourceSelect.value = "mic";
    audioSourceSelect.dispatchEvent(new Event("change", { bubbles: true }));
  }
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
  log(`音频文件已装载: ${file.name}`);
}

async function analyzeCurrentAudio(sourceFile: File): Promise<void> {
  lastAnalyzedFile = sourceFile;
  bindAudioFile(sourceFile);
  ui.setAudioElement(audioPlayer);

  try {
    await ui.analyze({
      input: buildAnalyzeInput(sourceFile),
      execution: {
        allowCache: usePredictionCacheCheckbox?.checked !== false,
        forceRefresh: false,
      },
    });
    log(`分析完成: ${sourceFile.name}`);
  } catch (error) {
    log(`分析失败: ${error instanceof Error ? error.message : String(error)}`);
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
    stopMicRecordBtn.disabled = !recording;
  }
  if (exportMicRecordBtn) {
    exportMicRecordBtn.disabled = !recordedMicFile;
  }
  if (micRecordStatus) {
    micRecordStatus.textContent = recording
      ? "录音中..."
      : recordedMicFile
        ? `已录制: ${recordedMicFile.name}`
        : "未录制";
  }
}

function setRecordedMicFile(file: File | null): void {
  recordedMicFile = file;
  revokeRecordedMicUrl();
  if (file && audioPlayer) {
    recordedMicObjectUrl = URL.createObjectURL(file);
    audioPlayer.src = recordedMicObjectUrl;
    audioPlayer.load();
  }
  syncMicButtons();
  if (file) {
    void analyzeCurrentAudio(file);
  }
}

async function startMicRecording(): Promise<void> {
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
    log("当前环境不支持麦克风录制");
    return;
  }
  if (mediaRecorder && mediaRecorder.state === "recording") {
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
      log(`麦克风录音完成: ${file.name}`);
      micChunks = [];
      syncMicButtons();
    };
    recorder.start();
    ensureMicAudioSource();
    syncMicButtons();
    log("开始麦克风录制");
  } catch (error) {
    log(`麦克风录制失败: ${error instanceof Error ? error.message : String(error)}`);
    stopMicRecording();
  }
}

function stopMicRecording(): void {
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
  syncMicButtons();
}

function exportMicRecording(): void {
  if (!recordedMicFile) {
    log("没有可导出的麦克风录音");
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
  log(`麦克风录音已导出: ${recordedMicFile.name}`);
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
    log(`CFP 已更新: batches=${state.cfp.length}`);
  }

  const inferenceCount = state.inference ? state.inference.totalBatchCount : 0;
  if (inferenceCount !== previousInferenceCount) {
    previousInferenceCount = inferenceCount;
    if (state.inference) {
      log(`推理已更新: batchCount=${state.inference.totalBatchCount}`);
    }
  }

  if (state.status !== previousStatus) {
    previousStatus = state.status;
    log(`UI 状态: ${state.status}${state.message ? ` - ${state.message}` : ""}`);
  }

  updateStateFromUi();
  renderDebugPanel();
});

const savedSettings = loadAppSettings();
populateModels();
populatePitchRangeOptions();
if (modelNameSelect && savedSettings.modelName) {
  modelNameSelect.value = savedSettings.modelName;
}
if (audioSourceSelect && savedSettings.audioSource) {
  audioSourceSelect.value = savedSettings.audioSource;
}
if (usePredictionCacheCheckbox && typeof savedSettings.usePredictionCache === "boolean") {
  usePredictionCacheCheckbox.checked = savedSettings.usePredictionCache;
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
if (modelNameSelect) {
  saveAppSettings({ modelName: readSelectedModelName() });
}
if (audioSourceSelect) {
  saveAppSettings({ audioSource: audioSourceSelect.value });
}
if (usePredictionCacheCheckbox) {
  saveAppSettings({ usePredictionCache: usePredictionCacheCheckbox.checked });
}
if (debugPanelEnabled) {
  saveAppSettings({ debugPanelEnabled: debugPanelEnabled.checked });
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
syncUiStatus(ui.getState().status);
ui.mount(spectrumCanvasWrapper);
ui.setAudioElement(audioPlayer);
renderDebugPanel();
if (debugPanelEnabled?.checked !== false) {
  ensureDebugLoop();
}
log("页面已启动");
syncMicButtons();
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
      log("当前正在分析中，模型切换已忽略");
      return;
    }
    saveAppSettings({ modelName: readSelectedModelName() });
    const file = getCurrentAnalyzableFile();
    if (!file) {
      log("模型已切换，当前没有可重推的音频");
      return;
    }
    log(`模型已切换，开始重新推理: ${readSelectedModelName()}`);
    void analyzeCurrentAudio(file);
  });
}
if (usePredictionCacheCheckbox) {
  usePredictionCacheCheckbox.addEventListener("change", () => {
    saveAppSettings({ usePredictionCache: usePredictionCacheCheckbox.checked });
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
if (audioSourceSelect) {
  audioSourceSelect.addEventListener("change", () => {
    saveAppSettings({ audioSource: audioSourceSelect.value });
    if (audioSourceSelect.value === "mic") {
      log("已切换到麦克风模式");
    } else {
      log("已切换到上传文件模式");
    }
  });
}

if (audioFileInput) {
  audioFileInput.addEventListener("change", () => {
    const file = audioFileInput.files && audioFileInput.files[0] ? audioFileInput.files[0] : null;
    if (!file) {
      return;
    }
    ensureUploadAudioSource();
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

function handleDrop(files: File[]): void {
  const file = files[0] || null;
  if (!file) return;
  ensureUploadAudioSource();
  if (audioFileInput) {
    const dt = new DataTransfer();
    dt.items.add(file);
    audioFileInput.files = dt.files;
  }
  log(`拖拽导入: ${file.name}`);
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
