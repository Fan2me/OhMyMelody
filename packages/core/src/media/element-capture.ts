import { createCaptureSession, type CaptureSession } from "../audio/capture-session.js";
import {
  captureMediaSource,
  type MediaElementCaptureOptions,
  type MediaCaptureSource,
} from "./capture.js";

interface MediaElementCaptureState {
  documentRef: Document;
  urlApi: typeof URL;
  audioCtx: AudioContext;
  playbackRate: number;
  audioEl: HTMLAudioElement;
  objectUrl: string;
  session: CaptureSession | null;
  timeoutId: ReturnType<typeof setTimeout> | null;
  playbackStarted: boolean;
  onChunk: ((chunk: readonly Float32Array[], sampleRate: number) => void) | null;
}

type HTMLAudioElementWithPlaybackFlags = HTMLAudioElement & {
  playsInline?: boolean;
  preservesPitch?: boolean;
  webkitPreservesPitch?: boolean;
};
function toPositiveFinite(value: unknown, fallback: number): number {
  const next = Number(value);
  return Number.isFinite(next) && next > 0 ? next : fallback;
}

export function normalizePlaybackRate(
  value: unknown,
  fallback = 4,
  min = 1,
  max = 16,
): number {
  const rate = toPositiveFinite(value, fallback);
  return Math.max(min, Math.min(max, rate));
}

export function buildMediaDecodeTimeoutMs(
  durationSec: number,
  playbackRate: number,
  minimumMs = 15000,
  tailMs = 15000,
): number {
  const safeDuration = Math.max(0, Number(durationSec) || 0);
  const safeRate = normalizePlaybackRate(playbackRate);
  return Math.max(
    minimumMs,
    Math.ceil((safeDuration / safeRate) * 1000) + tailMs,
  );
}

export function getMediaElementErrorMessage(
  mediaError: MediaError | null | undefined,
): string {
  if (!mediaError) {
    return "unknown media error";
  }

  const code = Number(mediaError.code || 0);
  if (code === 1) return "media playback aborted";
  if (code === 2) return "network error while loading media";
  if (code === 3) return "media decode error";
  if (code === 4) return "media source not supported";
  return mediaError.message || "unknown media error";
}

function resolveMediaElementCaptureEnvironment(
  file: Blob,
  audioCtx: AudioContext,
  options: MediaElementCaptureOptions = {},
): MediaElementCaptureState {
  const documentRef = typeof document !== "undefined" ? document : null;
  const urlApi = typeof URL !== "undefined" ? URL : null;
  if (
    !documentRef ||
    typeof documentRef.createElement !== "function" ||
    !documentRef.body ||
    !urlApi ||
    typeof urlApi.createObjectURL !== "function"
  ) {
    throw new Error("media element capture is unavailable");
  }
  if (!audioCtx || typeof audioCtx.createMediaElementSource !== "function") {
    throw new Error("media element capture is unsupported in this browser");
  }

  const playbackRate = normalizePlaybackRate(options.playbackRate);
  const audioEl = documentRef.createElement("audio");
  const objectUrl = urlApi.createObjectURL(file);
  return {
    documentRef,
    urlApi,
    audioCtx,
    playbackRate,
    audioEl,
    objectUrl,
    session: null,
    timeoutId: null,
    playbackStarted: false,
    onChunk: typeof options.onChunk === "function" ? options.onChunk : null,
  };
}

function cleanupMediaElementCaptureResources(
  state: MediaElementCaptureState | null,
): void {
  if (!state) {
    return;
  }

  if (state.timeoutId) {
    clearTimeout(state.timeoutId);
    state.timeoutId = null;
  }

  try {
    state.session?.dispose();
  } catch {}

  state.session = null;

  if (state.audioEl) {
    state.audioEl.oncanplay = null;
    state.audioEl.onended = null;
    state.audioEl.onerror = null;
    state.audioEl.onstalled = null;
    state.audioEl.onabort = null;
    try {
      state.audioEl.pause();
    } catch {}
    try {
      state.audioEl.remove();
    } catch {}
    try {
      state.audioEl.removeAttribute("src");
      state.audioEl.load();
    } catch {}
  }

  try {
    if (typeof state.urlApi.revokeObjectURL === "function") {
      state.urlApi.revokeObjectURL(state.objectUrl);
    }
  } catch {}
}

async function startMediaElementCapturePlayback(
  state: MediaElementCaptureState,
  file: Blob,
  finish: (err: Error | null, result?: AudioBuffer) => void,
): Promise<void> {
  if (state.playbackStarted) {
    return;
  }
  state.playbackStarted = true;
  try {
    if (!state.session) {
      state.session = await createCaptureSession(
        state.audioCtx,
        state.audioCtx.createMediaElementSource(state.audioEl),
        state.onChunk,
      );
    }
    if (
      state.audioCtx.state === "suspended" &&
      typeof state.audioCtx.resume === "function"
    ) {
      await state.audioCtx.resume();
    }
    const playPromise = state.audioEl.play();
    if (playPromise && typeof playPromise.then === "function") {
      await playPromise;
    }
    const durationSec =
      Number.isFinite(state.audioEl.duration) && state.audioEl.duration > 0
        ? state.audioEl.duration
        : Math.max(30, Math.ceil(file.size / 16000));
    state.timeoutId = setTimeout(
      () => {
        finish(
          new Error(
            `media element capture timed out after ${Math.ceil(durationSec / state.playbackRate) + 15}s`,
          ),
        );
      },
      buildMediaDecodeTimeoutMs(durationSec, state.playbackRate),
    );
  } catch (error) {
    finish(error instanceof Error ? error : new Error(String(error)));
  }
}

function createMediaElementCaptureSource(
  file: Blob,
  audioCtx: AudioContext,
  options: MediaElementCaptureOptions = {},
): MediaCaptureSource<AudioBuffer, MediaElementCaptureOptions> {
  const state = resolveMediaElementCaptureEnvironment(file, audioCtx, options);

  return {
    async start(
      finish: (err: Error | null, result?: AudioBuffer) => void,
    ): Promise<void> {
      try {
        const audioEl = state.audioEl as HTMLAudioElementWithPlaybackFlags;
        audioEl.crossOrigin = "anonymous";
        audioEl.preload = "auto";
        audioEl.controls = false;
        audioEl.playsInline = true;
        audioEl.style.display = "none";
        audioEl.playbackRate = state.playbackRate;
        if ("preservesPitch" in audioEl) {
          audioEl.preservesPitch = true;
        }
        if ("webkitPreservesPitch" in audioEl) {
          audioEl.webkitPreservesPitch = true;
        }
        audioEl.src = state.objectUrl;
        const body = state.documentRef.body;
        if (!body) {
          throw new Error("media element capture is unavailable");
        }
        body.appendChild(audioEl);

        audioEl.oncanplay = () => {
          void startMediaElementCapturePlayback(state, file, finish);
        };
        audioEl.onended = () => {
          try {
            if (!state.session) {
              throw new Error("capture session is unavailable");
            }
            finish(
              null,
              state.session.finalize(state.audioCtx.sampleRate),
            );
          } catch (error) {
            finish(error instanceof Error ? error : new Error(String(error)));
          }
        };
        audioEl.onerror = () => {
          finish(
            new Error(
              `audio element failed: ${getMediaElementErrorMessage(audioEl.error)}`,
            ),
          );
        };
        audioEl.onstalled = () => {};
        audioEl.onabort = () => {
          finish(new Error("audio element capture aborted"));
        };

        audioEl.load();
        if (audioEl.readyState >= 2) {
          void startMediaElementCapturePlayback(state, file, finish);
        }
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)));
      }
    },
    dispose(): void {
      cleanupMediaElementCaptureResources(state);
    },
  };
}

export async function captureAudioFromMediaElement(
  file: Blob,
  options: MediaElementCaptureOptions = {},
): Promise<AudioBuffer> {
  return await captureMediaSource(
    (audioCtx, captureOptions) =>
      createMediaElementCaptureSource(file, audioCtx, captureOptions),
    options,
  );
}
