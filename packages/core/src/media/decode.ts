import {
  ALL_FORMATS,
  AudioBufferSink,
  BlobSource,
  Input,
} from "mediabunny";
import {
  appendChannelPcmChunkToSink,
  concatFloat32Chunks,
  mixDownChannelsToMono,
  postProcessDecodedAudio,
  type ChannelPcmChunkSink,
  type DecodedAudioResult,
} from "../audio/pcm.js";

export async function decodeMediaAudioTrack(
  file: Blob,
): Promise<DecodedAudioResult> {
  if (!file) {
    throw new Error("media file is required");
  }

  const input = new Input({
    source: new BlobSource(file),
    formats: ALL_FORMATS,
  });

  try {
    const audioTrack = await input.getPrimaryAudioTrack();
    if (!audioTrack) {
      throw new Error("media file has no audio track");
    }

    const decoderConfig = await audioTrack.getDecoderConfig();
    if (!decoderConfig) {
      throw new Error("media audio track has no WebCodecs decoder config");
    }

    const canDecode = await audioTrack.canDecode();
    if (!canDecode) {
      throw new Error(
        `media audio codec is not supported by WebCodecs: ${decoderConfig.codec}`,
      );
    }

    const sampleRate = Math.max(1, Math.floor(await audioTrack.getSampleRate()));
    const sink = new AudioBufferSink(audioTrack);
    const pcmSink: ChannelPcmChunkSink = {
      channelChunks: [],
      channelCount: 0,
      totalFrames: 0,
    };

    for await (const { buffer } of sink.buffers()) {
      const channels = Array.from(
        { length: Math.max(1, buffer.numberOfChannels || 1) },
        (_, ch) => buffer.getChannelData(ch),
      );
      appendChannelPcmChunkToSink(pcmSink, channels);
    }

    if (!pcmSink.totalFrames) {
      throw new Error("media audio decode produced no PCM frames");
    }

    const channels = pcmSink.channelChunks.map((chunks) =>
      concatFloat32Chunks(chunks, pcmSink.totalFrames),
    );
    const decodedPcm = mixDownChannelsToMono(channels);

    return postProcessDecodedAudio({
      decodedPcm,
      sampleRate,
    });
  } finally {
    input.dispose();
  }
}
