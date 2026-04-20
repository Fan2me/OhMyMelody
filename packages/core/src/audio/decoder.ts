import {
  postProcessDecodedAudio,
  type DecodedAudioResult,
  readFileAsArrayBuffer,
} from "./pcm.js";
import { captureAudioFromMediaElement } from "../media/element-capture.js";
import { getModuleLogger } from "../logging/logger.js";

const logger = getModuleLogger("core.audio.decoder");

export type AudioContextCtor = {
  new (): AudioContext;
};

function getAudioContextCtor(): AudioContextCtor | null {
  if (typeof window === "undefined") {
    return null;
  }
  const windowRef = window as Window & {
    AudioContext?: AudioContextCtor;
    webkitAudioContext?: AudioContextCtor;
  };
  return (windowRef.AudioContext ||
    windowRef.webkitAudioContext ||
    null) as AudioContextCtor | null;
}

export async function createAudioContextInstance(
  audioContext?: AudioContext | null | undefined,
): Promise<AudioContext> {
  if (audioContext) {
    return audioContext;
  }

  const AudioContextCtor = getAudioContextCtor();
  if (typeof AudioContextCtor === "function") {
    return new AudioContextCtor();
  }

  throw new Error("Web Audio API is not supported in current environment");
}

export async function closeAudioContext(
  audioCtx: BaseAudioContext | null,
): Promise<void> {
  if (!audioCtx) {
    return;
  }

  try {
    const closable = audioCtx as BaseAudioContext & {
      close?: () => Promise<void> | void;
    };
    if (audioCtx.state !== "closed" && typeof closable.close === "function") {
      await closable.close();
    }
  } catch {}
}

async function decodeAudioBufferNative(
  audioCtx: BaseAudioContext,
  arrayBuffer: ArrayBuffer,
): Promise<AudioBuffer> {
  if (!audioCtx || typeof audioCtx.decodeAudioData !== "function") {
    throw new Error("decodeAudioData is not available");
  }

  let lastError: unknown = null;

  try {
    const maybePromise = audioCtx.decodeAudioData(arrayBuffer.slice(0));
    if (maybePromise && typeof maybePromise.then === "function") {
      return maybePromise;
    }
  } catch (error) {
    lastError = error;
  }

  return new Promise<AudioBuffer>((resolve, reject) => {
    try {
      audioCtx.decodeAudioData(arrayBuffer.slice(0), resolve, reject);
    } catch (error) {
      lastError = error;
      reject(error);
    }
  }).catch((error) => {
    lastError = error;
    throw lastError || new Error("Unable to decode audio data");
  });
}

export async function decodeAudioRaw(
  file: File | Blob,
): Promise<DecodedAudioResult> {
  const audioCtx = await createAudioContextInstance();
  try {
    const arrayBuffer = await readFileAsArrayBuffer(file);
    let audioBuffer: AudioBuffer | null = null;
    let decodedPcm: Float32Array | null = null;
    let sampleRate = 0;

    try {
      audioBuffer = await decodeAudioBufferNative(audioCtx, arrayBuffer);
    } catch (nativeError) {
      logger.warn(
        `原生音频解码失败，回退到媒体元素采集: ${nativeError instanceof Error ? nativeError.message : String(nativeError)}`,
      );
      const fallbackAudioBuffer = await captureAudioFromMediaElement(file, {
        audioContext: audioCtx,
        playbackRate: 16,
      });
      if (fallbackAudioBuffer) {
        audioBuffer = fallbackAudioBuffer;
        sampleRate = fallbackAudioBuffer.sampleRate || 0;
        logger.info(
          `媒体元素采集完成，采样率: ${sampleRate || 0}，通道: ${fallbackAudioBuffer.numberOfChannels}`,
        );
      }
    }

    if (audioBuffer) {
      return postProcessDecodedAudio({
        audioBuffer,
        sampleRate: sampleRate || audioBuffer.sampleRate || 0,
      });
    }

    if (decodedPcm) {
      return postProcessDecodedAudio({
        decodedPcm,
        sampleRate,
      });
    }

    throw new Error("audio decode produced no usable data");
  } finally {
    await closeAudioContext(audioCtx);
  }
}
