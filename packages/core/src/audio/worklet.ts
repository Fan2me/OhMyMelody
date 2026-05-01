export interface ChannelCaptureWorkletHandle {
  node: AudioWorkletNode;
  silentGain: GainNode;
  dispose(): void;
}

export interface ChannelCaptureWorkletOptions {
  audioCtx: AudioContext;
  sourceNode: AudioNode;
  onChunk: (chunk: readonly Float32Array[]) => void;
}

const CHANNEL_CAPTURE_WORKLET_NAME = 'ohm-channel-capture-processor';

let channelCaptureWorkletModuleUrl: string | null = null;
const channelCaptureWorkletModuleReadyByWorklet = new WeakMap<
  AudioWorklet,
  Promise<void>
>();

function buildChannelCaptureWorkletSource(): string {
  return `
class ChannelCaptureProcessor extends AudioWorkletProcessor {
  process(inputs, outputs) {
    const input = inputs[0] || [];
    const output = outputs[0] || [];
    const channelCount = input.length;
    const frameCount = channelCount > 0 && input[0] ? input[0].length : 0;

    if (!frameCount) {
      for (let i = 0; i < output.length; i += 1) {
        const channel = output[i];
        if (channel) {
          channel.fill(0);
        }
      }
      return true;
    }

    const channels = new Array(channelCount);
    const transfer = new Array(channelCount);
    for (let ch = 0; ch < channelCount; ch += 1) {
      const channel = input[ch];
      if (!channel) {
        channels[ch] = new Float32Array(0);
        transfer[ch] = channels[ch].buffer;
        continue;
      }
      const copy = new Float32Array(channel.length);
      copy.set(channel);
      channels[ch] = copy;
      transfer[ch] = copy.buffer;
    }

    for (let i = 0; i < output.length; i += 1) {
      const channel = output[i];
      if (channel) {
        channel.fill(0);
      }
    }

    this.port.postMessage(channels, transfer);
    return true;
  }
}

registerProcessor(${JSON.stringify(CHANNEL_CAPTURE_WORKLET_NAME)}, ChannelCaptureProcessor);
`;
}

function getChannelCaptureWorkletModuleUrl(): string {
  if (channelCaptureWorkletModuleUrl) {
    return channelCaptureWorkletModuleUrl;
  }

  const source = buildChannelCaptureWorkletSource();
  const blob = new Blob([source], { type: 'text/javascript' });
  channelCaptureWorkletModuleUrl = URL.createObjectURL(blob);
  return channelCaptureWorkletModuleUrl;
}

async function ensureChannelCaptureWorkletModuleLoaded(audioCtx: AudioContext): Promise<void> {
  if (!audioCtx.audioWorklet || typeof audioCtx.audioWorklet.addModule !== 'function') {
    throw new Error('AudioWorklet is not supported in current environment');
  }

  const worklet = audioCtx.audioWorklet;
  let ready = channelCaptureWorkletModuleReadyByWorklet.get(worklet);
  if (!ready) {
    ready = worklet
      .addModule(getChannelCaptureWorkletModuleUrl())
      .then(() => undefined)
      .catch((error) => {
        channelCaptureWorkletModuleReadyByWorklet.delete(worklet);
        throw error;
      });
    channelCaptureWorkletModuleReadyByWorklet.set(worklet, ready);
  }

  await ready;
}

export async function attachChannelCaptureWorklet({
  audioCtx,
  sourceNode,
  onChunk,
}: ChannelCaptureWorkletOptions): Promise<ChannelCaptureWorkletHandle> {
  if (!audioCtx) {
    throw new Error('audio context is required');
  }
  if (!sourceNode || typeof sourceNode.connect !== 'function') {
    throw new Error('source node is required');
  }
  if (typeof onChunk !== 'function') {
    throw new Error('onChunk handler is required');
  }

  await ensureChannelCaptureWorkletModuleLoaded(audioCtx);

  const node = new AudioWorkletNode(audioCtx, CHANNEL_CAPTURE_WORKLET_NAME, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    channelCountMode: 'max',
    channelInterpretation: 'discrete',
  });
  const silentGain = audioCtx.createGain();
  silentGain.gain.value = 0;

  node.port.onmessage = (event: MessageEvent) => {
    const payload: Float32Array[] = event.data;
    if (Array.isArray(payload) && payload.every((channel) => channel instanceof Float32Array)) {
      onChunk(payload);
    }
  };

  sourceNode.connect(node);
  node.connect(silentGain);
  silentGain.connect(audioCtx.destination);

  return {
    node,
    silentGain,
    dispose() {
      try {
        node.port.onmessage = null;
      } catch {}
      try {
        sourceNode.disconnect(node);
      } catch {}
      try {
        node.disconnect();
      } catch {}
      try {
        silentGain.disconnect();
      } catch {}
    },
  };
}
