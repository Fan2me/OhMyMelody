import { createCaptureSession, type CaptureSession } from '../audio/capture-session.js';
import {
  captureMediaSource,
  type MediaCaptureContextOptions,
  type MediaCaptureSource,
} from './capture.js';

interface MediaStreamCaptureState {
  audioCtx: AudioContext;
  stream: MediaStream;
  session: CaptureSession | null;
  onChunk: ((chunk: readonly Float32Array[], sampleRate: number) => void) | null;
  finalizeTimer: ReturnType<typeof setTimeout> | null;
}

function cleanupMediaStreamCaptureResources(state: MediaStreamCaptureState | null): void {
  if (!state) {
    return;
  }

  if (state.finalizeTimer !== null) {
    clearTimeout(state.finalizeTimer);
    state.finalizeTimer = null;
  }

  try {
    state.session?.dispose();
  } catch {}

  state.session = null;
}

function isStreamActive(stream: MediaStream): boolean {
  const tracks = typeof stream.getTracks === 'function' ? stream.getTracks() : [];
  if (!tracks.length) {
    return false;
  }
  return tracks.some((track) => track && track.readyState !== 'ended');
}

function createMediaStreamCaptureSource(
  stream: MediaStream,
  audioCtx: AudioContext,
  options: MediaCaptureContextOptions = {},
): MediaCaptureSource<AudioBuffer, MediaCaptureContextOptions> {
  if (!stream || typeof stream.getTracks !== 'function') {
    throw new Error('media stream is required');
  }

  const state: MediaStreamCaptureState = {
    audioCtx,
    stream,
    session: null,
    onChunk: typeof options.onChunk === 'function' ? options.onChunk : null,
    finalizeTimer: null,
  };

  return {
    async start(finish: (err: Error | null, result?: AudioBuffer) => void): Promise<void> {
      try {
        if (typeof audioCtx.createMediaStreamSource !== 'function') {
          throw new Error('media stream capture is unsupported in this browser');
        }

        const tracks = typeof stream.getTracks === 'function' ? stream.getTracks() : [];
        if (!tracks.length) {
          throw new Error('media stream has no tracks');
        }

        state.session = await createCaptureSession(
          audioCtx,
          audioCtx.createMediaStreamSource(stream),
          state.onChunk,
        );

        for (const track of tracks) {
          if (!track || track.readyState === 'ended') {
            continue;
          }
          track.addEventListener(
            'ended',
            () => {
              if (isStreamActive(stream)) {
                return;
              }
              if (state.finalizeTimer !== null) {
                clearTimeout(state.finalizeTimer);
              }
              state.finalizeTimer = setTimeout(() => {
                state.finalizeTimer = null;
                if (isStreamActive(stream)) {
                  return;
                }
                try {
                  if (!state.session) {
                    throw new Error('capture session is unavailable');
                  }
                  finish(null, state.session.finalize(audioCtx.sampleRate));
                } catch (error) {
                  finish(error instanceof Error ? error : new Error(String(error)));
                }
              }, 80);
            },
            { once: true },
          );
        }

        if (!isStreamActive(stream)) {
          throw new Error('media stream is not active');
        }
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)));
      }
    },
    dispose(): void {
      cleanupMediaStreamCaptureResources(state);
    },
  };
}

export async function captureAudioFromMediaStream(
  stream: MediaStream,
  options: MediaCaptureContextOptions = {},
): Promise<AudioBuffer> {
  return await captureMediaSource(
    (audioCtx, captureOptions) => createMediaStreamCaptureSource(stream, audioCtx, captureOptions),
    options,
  );
}
