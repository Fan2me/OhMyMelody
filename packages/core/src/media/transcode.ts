type FFmpegInstance = {
  load(options: {
    coreURL: string;
    wasmURL: string;
    classWorkerURL: string;
  }): Promise<boolean>;
  on(event: 'progress', handler: (payload: { progress?: number; time?: number }) => void): void;
  off(event: 'progress', handler: (payload: { progress?: number; time?: number }) => void): void;
  writeFile(path: string, data: Uint8Array): Promise<void>;
  readFile(path: string): Promise<Uint8Array | ArrayBufferLike | string>;
  deleteFile(path: string): Promise<void>;
  exec(args: string[]): Promise<number>;
};

export interface TranscodeProgressPayload {
  progress: number;
  time: number;
  inputName: string;
  outputName: string;
}

export interface MediaPreparationResult<TFile extends Blob & { name?: string } = Blob & { name?: string }> {
  originalFile: TFile;
  analysisFile: TFile;
  converted: boolean;
}

const FFMPEG_CORE_VERSION = '0.12.10';
const FFMPEG_RUNTIME_VERSION = '0.12.15';
const FFMPEG_VENDOR_ROOT = '/vendor/ffmpeg';
const FFMPEG_CLASS_WORKER_URL = `${FFMPEG_VENDOR_ROOT}/@ffmpeg/ffmpeg@${FFMPEG_RUNTIME_VERSION}/dist/esm/worker.js`;
const FFMPEG_CORE_JS_URL = `${FFMPEG_VENDOR_ROOT}/@ffmpeg/core@${FFMPEG_CORE_VERSION}/dist/esm/ffmpeg-core.js`;
const FFMPEG_CORE_WASM_URL = `${FFMPEG_VENDOR_ROOT}/@ffmpeg/core@${FFMPEG_CORE_VERSION}/dist/esm/ffmpeg-core.wasm`;
const VIDEO_MIME_PREFIX = 'video/';
const VIDEO_EXTENSIONS = new Set([
  'mp4',
  'm4v',
  'mov',
  'webm',
  'mkv',
  'avi',
  'flv',
  'ts',
  'm2ts',
  'mts',
  '3gp',
  '3gpp',
]);

let ffmpegInstance: FFmpegInstance | null = null;
let ffmpegLoadPromise: Promise<FFmpegInstance> | null = null;
let ffmpegQueue = Promise.resolve();

function getFileStem(fileName = ''): string {
  const rawName = String(fileName || '').trim();
  if (!rawName) {
    return 'media';
  }
  const withoutQuery = (rawName.split('?')[0] ?? rawName).split('#')[0] ?? rawName;
  const lastSlash = withoutQuery.lastIndexOf('/');
  const baseName = lastSlash >= 0 ? withoutQuery.slice(lastSlash + 1) : withoutQuery;
  const dotIndex = baseName.lastIndexOf('.');
  const stem = dotIndex > 0 ? baseName.slice(0, dotIndex) : baseName;
  const normalized = stem
    .normalize('NFKC')
    .replace(/[\s]+/g, '_')
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^[_.-]+/, '')
    .replace(/[_.-]+$/, '');
  return normalized || 'media';
}

function getMediaExtension(file: Blob & { name?: string }): string {
  const name = String(file && file.name ? file.name : '').trim();
  const dotIndex = name.lastIndexOf('.');
  if (dotIndex >= 0 && dotIndex < name.length - 1) {
    return name.slice(dotIndex + 1).toLowerCase();
  }
  return '';
}

export function isVideoLikeMediaFile(file: File | Blob | null | undefined): boolean {
  if (!file) {
    return false;
  }
  const mime = String(file.type || '').toLowerCase();
  if (mime.startsWith(VIDEO_MIME_PREFIX)) {
    return true;
  }
  const ext = getMediaExtension(file);
  return VIDEO_EXTENSIONS.has(ext);
}

function buildTempInputName(file: Blob & { name?: string }): string {
  const ext = getMediaExtension(file);
  return `${getFileStem(file && file.name)}-${Date.now().toString(36)}.${ext || 'bin'}`;
}

function buildOutputName(file: Blob & { name?: string }, extension = 'mp3'): string {
  return `${getFileStem(file && file.name)}.${String(extension || 'mp3').replace(/^\./, '')}`;
}

async function loadFFmpegRuntime(): Promise<FFmpegInstance> {
  if (ffmpegInstance) {
    return ffmpegInstance;
  }
  if (ffmpegLoadPromise) {
    return await ffmpegLoadPromise;
  }

  ffmpegLoadPromise = (async () => {
    const { FFmpeg } = await import('@ffmpeg/ffmpeg');
    const ffmpeg = new FFmpeg() as unknown as FFmpegInstance;
    await ffmpeg.load({
      coreURL: FFMPEG_CORE_JS_URL,
      wasmURL: FFMPEG_CORE_WASM_URL,
      classWorkerURL: FFMPEG_CLASS_WORKER_URL,
    });
    ffmpegInstance = ffmpeg;
    return ffmpeg;
  })().finally(() => {
    ffmpegLoadPromise = null;
  });

  return await ffmpegLoadPromise;
}

async function runSerializedFFmpegTask<T>(task: () => Promise<T>): Promise<T> {
  const previous = ffmpegQueue;
  let release = () => {};
  ffmpegQueue = new Promise((resolve) => {
    release = resolve;
  });
  await previous.catch(() => {});
  try {
    return await task();
  } finally {
    release();
  }
}

export async function transcodeMediaFileToMp3(
  file: Blob & { name?: string },
  {
    outputName = '',
    onProgress = null,
  }: {
    outputName?: string;
    onProgress?: ((payload: TranscodeProgressPayload) => void) | null;
  } = {},
): Promise<File> {
  if (!file) {
    throw new Error('media file is required');
  }

  return await runSerializedFFmpegTask(async () => {
      const [ffmpeg, { fetchFile }] = await Promise.all([
        loadFFmpegRuntime(),
        import('@ffmpeg/util'),
      ]);
    const inputName = buildTempInputName(file);
    const finalOutputName = outputName || buildOutputName(file, 'mp3');
    const progressHandler =
      typeof onProgress === 'function'
        ? ({ progress = 0, time = 0 }: { progress?: number; time?: number } = {}) => {
            onProgress({
              progress,
              time,
              inputName,
              outputName: finalOutputName,
            });
          }
        : null;

    if (progressHandler) {
      ffmpeg.on('progress', progressHandler);
    }

    try {
      await ffmpeg.writeFile(inputName, await fetchFile(file));
      const exitCode = await ffmpeg.exec([
        '-i',
        inputName,
        '-vn',
        '-map',
        '0:a:0',
        '-c:a',
        'libmp3lame',
        '-q:a',
        '2',
        finalOutputName,
      ]);
      if (exitCode !== 0) {
        throw new Error(`ffmpeg exited with code ${exitCode}`);
      }

      const outputBytes = await ffmpeg.readFile(finalOutputName);
      if (typeof outputBytes === 'string') {
        throw new Error('ffmpeg returned text instead of binary audio data');
      }

      const outputData =
        outputBytes instanceof Uint8Array
          ? outputBytes
          : new Uint8Array(outputBytes as unknown as ArrayBufferLike);
      return new File([outputData.slice()], finalOutputName, {
        type: 'audio/mpeg',
      });
    } finally {
      if (progressHandler) {
        try {
          ffmpeg.off('progress', progressHandler);
        } catch {}
      }
      try {
        await ffmpeg.deleteFile(inputName);
      } catch {}
      try {
        await ffmpeg.deleteFile(finalOutputName);
      } catch {}
    }
  });
}

export async function prepareMediaFileForAnalysis(
  file: Blob & { name?: string },
): Promise<MediaPreparationResult<Blob & { name?: string }>> {
  if (!file) {
    throw new Error('media file is required');
  }

  if (!isVideoLikeMediaFile(file)) {
    return {
      originalFile: file,
      analysisFile: file,
      converted: false,
    };
  }

  const analysisFile = await transcodeMediaFileToMp3(file);
  return {
    originalFile: file,
    analysisFile,
    converted: true,
  };
}
