export type SampleQueue = {
  chunks: Float32Array[];
  totalSamples: number;
};

export function createSampleQueue(): SampleQueue {
  return {
    chunks: [],
    totalSamples: 0,
  };
}

export function enqueueSamples(queue: SampleQueue, samples: Float32Array): void {
  if (!samples.length) {
    return;
  }
  queue.chunks.push(samples);
  queue.totalSamples += samples.length;
}

export function takeSamples(
  queue: SampleQueue,
  sampleCount: number,
): Float32Array | null {
  const take = Math.max(0, Math.floor(sampleCount));
  if (!take || queue.totalSamples < take) {
    return null;
  }

  const out = new Float32Array(take);
  let offset = 0;
  while (offset < take && queue.chunks.length > 0) {
    const head = queue.chunks[0];
    if (!head) {
      queue.chunks.shift();
      continue;
    }
    const available = head.length;
    const need = take - offset;
    const copyCount = Math.min(available, need);
    out.set(head.subarray(0, copyCount), offset);
    offset += copyCount;
    queue.totalSamples -= copyCount;
    if (copyCount === available) {
      queue.chunks.shift();
    } else {
      queue.chunks[0] = head.subarray(copyCount);
    }
  }

  return offset === take ? out : out.slice(0, offset);
}

export function drainSamples(queue: SampleQueue): Float32Array {
  if (!queue.totalSamples || !queue.chunks.length) {
    return new Float32Array(0);
  }
  return takeSamples(queue, queue.totalSamples) ?? new Float32Array(0);
}

export function padSamplesToLength(
  samples: Float32Array,
  targetLength: number,
): Float32Array {
  const target = Math.max(0, Math.floor(targetLength));
  if (target <= 0) {
    return new Float32Array(0);
  }
  if (samples.length >= target) {
    return samples;
  }
  const padded = new Float32Array(target);
  padded.set(samples);
  return padded;
}
