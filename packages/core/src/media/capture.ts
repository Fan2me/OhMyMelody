import { throwIfAborted } from "../abort/abort.js";
import {
  createAudioContextInstance,
  closeAudioContext,
} from "../audio/decoder.js";

export interface MediaCaptureContextOptions {
  audioContext?: AudioContext | null;
  signal?: AbortSignal | null;
  onChunk?: ((chunk: readonly Float32Array[], sampleRate: number) => void) | null;
}

export interface MediaElementCaptureOptions extends MediaCaptureContextOptions {
  playbackRate?: number;
}

export interface ResolvedMediaCaptureAudioContext {
  audioCtx: AudioContext;
  dispose: () => Promise<void>;
}

export interface MediaCaptureSource<
  TResult = unknown,
  TOptions extends MediaCaptureContextOptions = MediaCaptureContextOptions,
> {
  start: (
    finish: (error: Error | null, result?: TResult) => void,
  ) => Promise<void>;
  dispose: () => void;
}

export type MediaCaptureSourceFactory<
  TResult = unknown,
  TOptions extends MediaCaptureContextOptions = MediaCaptureContextOptions,
> = (
  audioCtx: AudioContext,
  options: TOptions,
) => Promise<MediaCaptureSource<TResult, TOptions>> | MediaCaptureSource<TResult, TOptions>;

function createCaptureFinalizer<TResult>({
  signal = null,
  cleanup,
  onResolve,
  onReject,
  abortReason,
}: {
  signal?: AbortSignal | null;
  cleanup: () => void;
  onResolve: (result: TResult) => void;
  onReject: (error: Error) => void;
  abortReason: string;
}) {
  let settled = false;

  const finalize = (error: Error | null, result?: TResult) => {
    if (settled) {
      return;
    }
    settled = true;
    try {
      cleanup();
    } finally {
      if (signal) {
        signal.removeEventListener("abort", handleAbort);
      }
    }

    if (error) {
      onReject(error);
      return;
    }
    if (typeof result === "undefined") {
      onReject(new Error("capture produced no result"));
      return;
    }
    onResolve(result);
  };

  const handleAbort = () => {
    const reason = signal?.reason ?? abortReason;
    const abortError =
      typeof DOMException !== "undefined"
        ? new DOMException(reason, "AbortError")
        : Object.assign(new Error(reason), { name: "AbortError" });
    finalize(abortError);
  };

  return {
    finalize,
    handleAbort,
    isSettled: () => settled,
  };
}

export async function resolveMediaCaptureAudioContext(
  options: MediaCaptureContextOptions = {},
): Promise<ResolvedMediaCaptureAudioContext> {
  const audioCtx = await createAudioContextInstance(options.audioContext);
  const providedAudioContext = !!options.audioContext;
  return {
    audioCtx,
    dispose: async () => {
      if (!providedAudioContext) {
        await closeAudioContext(audioCtx);
      }
    },
  };
}

export async function captureMediaSource<
  TResult,
  TOptions extends MediaCaptureContextOptions = MediaCaptureContextOptions,
>(
  createSource: MediaCaptureSourceFactory<TResult, TOptions>,
  options: TOptions = {} as TOptions,
): Promise<TResult> {
  throwIfAborted(options.signal, "media capture aborted");

  const audioContextHandle = await resolveMediaCaptureAudioContext(options);
  let source: MediaCaptureSource<TResult, TOptions> | null = null;

  return await new Promise<TResult>((resolve, reject) => {
    const lifecycle = createCaptureFinalizer<TResult>({
      signal: options.signal ?? null,
      abortReason: "media capture aborted",
      cleanup: () => {
        try {
          source?.dispose();
        } finally {
          void audioContextHandle.dispose();
        }
      },
      onResolve: resolve,
      onReject: reject,
    });

    const finish = lifecycle.finalize;
    const handleAbort = lifecycle.handleAbort;

    if (options.signal) {
      if (options.signal.aborted) {
        handleAbort();
        return;
      }
      options.signal.addEventListener("abort", handleAbort, { once: true });
    }

    void (async () => {
      try {
        source = await createSource(audioContextHandle.audioCtx, options);
        if (lifecycle.isSettled()) {
          source.dispose();
          return;
        }
        await source.start(finish);
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)));
      }
    })();
  });
}
