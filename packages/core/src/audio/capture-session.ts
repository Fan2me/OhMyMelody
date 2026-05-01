import {
  appendChannelPcmChunkToSink,
  createAudioBufferFromChannelChunks,
  type ChannelPcmChunkSink,
} from './pcm.js';
import {
  attachChannelCaptureWorklet,
  type ChannelCaptureWorkletHandle,
} from './worklet.js';

export class CaptureSession implements ChannelPcmChunkSink {
  public readonly channelChunks: Float32Array[][] = [];
  public channelCount = 0;
  public totalFrames = 0;

  private sourceNode: AudioNode | null = null;
  private workletHandle: ChannelCaptureWorkletHandle | null = null;
  private readonly onChunk: ((chunk: readonly Float32Array[], sampleRate: number) => void) | null;

  public constructor(
    private readonly audioCtx: AudioContext,
    onChunk: ((chunk: readonly Float32Array[], sampleRate: number) => void) | null = null,
  ) {
    if (!audioCtx) {
      throw new Error('audio context is required');
    }
    this.onChunk = typeof onChunk === 'function' ? onChunk : null;
  }

  public async attach(sourceNode: AudioNode): Promise<void> {
    if (!sourceNode || typeof sourceNode.connect !== 'function') {
      throw new Error('source node is required');
    }
    if (this.workletHandle) {
      throw new Error('capture source is already attached');
    }

    this.sourceNode = sourceNode;
    this.workletHandle = await attachChannelCaptureWorklet({
      audioCtx: this.audioCtx,
      sourceNode,
      onChunk: (chunk) => {
        this.onChunk?.(chunk, this.audioCtx.sampleRate);
        appendChannelPcmChunkToSink(this, chunk);
      },
    });
  }

  public finalize(sampleRate: number): AudioBuffer {
    if (!this.totalFrames) {
      throw new Error('capture produced no PCM frames');
    }

    return createAudioBufferFromChannelChunks(
      this.audioCtx,
      this.channelChunks,
      this.totalFrames,
      Math.max(1, Math.floor(sampleRate)),
    );
  }

  public dispose(): void {
    try {
      if (this.workletHandle) {
        this.workletHandle.dispose();
      }
      if (this.sourceNode) {
        this.sourceNode.disconnect();
      }
    } catch {}

    this.sourceNode = null;
    this.workletHandle = null;
  }
}

export async function createCaptureSession(
  audioCtx: AudioContext,
  sourceNode: AudioNode,
  onChunk: ((chunk: readonly Float32Array[], sampleRate: number) => void) | null = null,
): Promise<CaptureSession> {
  const session = new CaptureSession(audioCtx, onChunk);
  await session.attach(sourceNode);
  return session;
}
