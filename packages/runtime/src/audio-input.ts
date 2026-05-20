import { decodeAudioRaw } from "@ohm/core/audio/decoder.js";
import { postProcessDecodedAudio } from "@ohm/core/audio/pcm.js";
import { captureAudioFromMediaStream } from "@ohm/core/media/stream-capture.js";
import type { AnalyzeInput } from "./types.js";

export type DecodedRuntimeAudio = {
  pcm: Float32Array;
  fs: number;
  mode?: string;
};

export function sourceToBlob(source: AnalyzeInput["source"]): Blob {
  if (source.kind === "file") {
    return source.file;
  }
  if (source.kind === "blob") {
    return source.blob;
  }
  if (source.kind === "buffer") {
    return new Blob([source.buffer as unknown as BlobPart]);
  }
  throw new Error(`Unsupported analysis source kind: ${source.kind}`);
}

export async function decodeInputAudio(
  input: AnalyzeInput,
): Promise<DecodedRuntimeAudio> {
  if (input.source.kind === "stream") {
    const audioBuffer = await captureAudioFromMediaStream(input.source.stream);
    return postProcessDecodedAudio({
      audioBuffer,
      sampleRate: audioBuffer.sampleRate,
    });
  }

  const blob = sourceToBlob(input.source);
  return decodeAudioRaw(blob);
}
