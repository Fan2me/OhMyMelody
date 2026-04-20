export interface DecodedAudioResult {
  pcm: Float32Array;
  fs: number;
}

export interface ChannelPcmChunkSink {
  channelChunks: Float32Array[][];
  channelCount: number;
  totalFrames: number;
}

const TARGET_SAMPLE_RATE = 8000;

function resampleMonoPcm(
  pcm: Float32Array,
  sourceRate: number,
  targetRate = TARGET_SAMPLE_RATE,
): Float32Array {
  const srcRate = Math.max(1, Math.floor(sourceRate || 1));
  const dstRate = Math.max(1, Math.floor(targetRate || TARGET_SAMPLE_RATE));
  if (!pcm.length) {
    return new Float32Array(0);
  }
  if (srcRate === dstRate) {
    return pcm.slice();
  }

  const dstLength = Math.max(1, Math.round((pcm.length * dstRate) / srcRate));
  const dst = new Float32Array(dstLength);
  const ratio = srcRate / dstRate;

  for (let i = 0; i < dstLength; i += 1) {
    const sourcePos = i * ratio;
    const idx = Math.floor(sourcePos);
    const frac = sourcePos - idx;
    const s0 = pcm[Math.min(pcm.length - 1, idx)] ?? 0;
    const s1 = pcm[Math.min(pcm.length - 1, idx + 1)] ?? 0;
    dst[i] = s0 + (s1 - s0) * frac;
  }

  return dst;
}

function mixDownToMono(audioBuffer: AudioBuffer): Float32Array {
  const channelCount = Math.max(1, audioBuffer.numberOfChannels || 1);
  const length = audioBuffer.length || 0;
  const mono = new Float32Array(length);

  if (channelCount === 1) {
    mono.set(audioBuffer.getChannelData(0));
    return mono;
  }

  for (let ch = 0; ch < channelCount; ch += 1) {
    const data =
      audioBuffer.numberOfChannels > ch ? audioBuffer.getChannelData(ch) : null;
    if (!data) {
      continue;
    }
    for (let i = 0; i < length; i += 1) {
      const sample = data[i] ?? 0;
      mono[i] = (mono[i] ?? 0) + sample / channelCount;
    }
  }

  return mono;
}

export function appendChannelPcmChunkToSink(
  sink: ChannelPcmChunkSink,
  chunk: readonly Float32Array[],
): void {
  const channelCount = chunk.length;
  if (!channelCount) {
    return;
  }

  if (!sink.channelCount) {
    sink.channelCount = channelCount;
    sink.channelChunks.length = channelCount;
    for (let ch = 0; ch < channelCount; ch += 1) {
      sink.channelChunks[ch] = [];
    }
  } else if (sink.channelCount !== channelCount) {
    throw new Error("capture channel count changed during capture");
  }

  const frameCount = chunk[0]?.length ?? 0;
  if (!frameCount) {
    return;
  }

  for (let ch = 0; ch < channelCount; ch += 1) {
    const data = chunk[ch];
    if (!data) {
      throw new Error("capture channel data is unavailable");
    }
    if (data.length !== frameCount) {
      throw new Error("capture channel frame count mismatch");
    }
    const channelChunks = sink.channelChunks[ch];
    if (!channelChunks) {
      throw new Error("capture channel sink is unavailable");
    }
    channelChunks.push(data.slice());
  }

  sink.totalFrames += frameCount;
}

export function concatFloat32Chunks(
  chunks: readonly Float32Array[],
  totalFrames: number,
): Float32Array {
  if (!chunks.length || !totalFrames) {
    return new Float32Array(0);
  }

  const pcm = new Float32Array(totalFrames);
  let offset = 0;
  for (const chunk of chunks) {
    if (!chunk.length) {
      continue;
    }
    pcm.set(chunk, offset);
    offset += chunk.length;
  }
  return offset === totalFrames ? pcm : pcm.slice(0, offset);
}

export function createAudioBufferFromChannelChunks(
  audioCtx: BaseAudioContext,
  channelChunks: readonly Float32Array[][],
  totalFrames: number,
  sampleRate: number,
): AudioBuffer {
  if (!audioCtx) {
    throw new Error("audio context is required");
  }

  const channelCount = channelChunks.length;
  if (!channelCount) {
    throw new Error("capture produced no channels");
  }
  if (!totalFrames) {
    throw new Error("capture produced no PCM frames");
  }

  const buffer = audioCtx.createBuffer(
    channelCount,
    totalFrames,
    Math.max(1, Math.floor(sampleRate || 44100)),
  );

  for (let ch = 0; ch < channelCount; ch += 1) {
    const pcm = concatFloat32Chunks(channelChunks[ch] || [], totalFrames);
    buffer.copyToChannel(pcm as Float32Array<ArrayBuffer>, ch);
  }

  return buffer;
}

export function postProcessDecodedAudio({
  audioBuffer = null,
  decodedPcm = null,
  sampleRate = 0,
  targetSampleRate = TARGET_SAMPLE_RATE,
}: {
  audioBuffer?: AudioBuffer | null;
  decodedPcm?: Float32Array | null;
  sampleRate?: number;
  targetSampleRate?: number;
}): DecodedAudioResult {
  const pcm = audioBuffer ? mixDownToMono(audioBuffer) : decodedPcm;
  if (!pcm) {
    throw new Error("no PCM data to post-process");
  }
  const srcRate = audioBuffer
    ? audioBuffer.sampleRate || targetSampleRate
    : sampleRate;
  return {
    pcm: resampleMonoPcm(pcm, srcRate, targetSampleRate),
    fs: Math.max(1, Math.floor(targetSampleRate || TARGET_SAMPLE_RATE)),
  };
}

export async function readFileAsArrayBuffer(file: Blob): Promise<ArrayBuffer> {
  if (file && typeof file.arrayBuffer === "function") {
    return await file.arrayBuffer();
  }
  return await new Promise<ArrayBuffer>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (result instanceof ArrayBuffer) {
        resolve(result);
        return;
      }
      reject(new Error("failed to read audio file"));
    };
    reader.onerror = () =>
      reject(reader.error || new Error("failed to read audio file"));
    reader.readAsArrayBuffer(file);
  });
}
