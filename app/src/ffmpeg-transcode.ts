import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile, toBlobURL } from "@ffmpeg/util";

const OUTPUT_SAMPLE_RATE = 44100;
const OUTPUT_CHANNELS = 2;
const OUTPUT_MIME_TYPE = "audio/wav";
const OUTPUT_EXTENSION = "wav";
const FFMPEG_CORE_BASE_URL = "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/umd";

let ffmpegInstance: FFmpeg | null = null;
let ffmpegLoadPromise: Promise<FFmpeg> | null = null;
let ffmpegExecutionQueue: Promise<void> = Promise.resolve();
let ffmpegLogBound = false;

function buildTempFileStem(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).slice(2, 10);
  return `ohm-${timestamp}-${random}`;
}

function getFileExtension(name: string): string {
  const trimmed = String(name || "").trim();
  const dotIndex = trimmed.lastIndexOf(".");
  if (dotIndex < 0 || dotIndex >= trimmed.length - 1) {
    return "";
  }
  return trimmed.slice(dotIndex + 1).toLowerCase();
}

function buildOutputFileName(name: string): string {
  const trimmed = String(name || "").trim();
  const baseName = trimmed.replace(/\.[^.]+$/, "") || "audio";
  return `${baseName}.ffmpeg-fallback.${OUTPUT_EXTENSION}`;
}

async function getFfmpeg(): Promise<FFmpeg> {
  if (ffmpegInstance?.loaded) {
    return ffmpegInstance;
  }
  if (ffmpegLoadPromise) {
    return ffmpegLoadPromise;
  }

  const ffmpeg = ffmpegInstance ?? new FFmpeg();
  ffmpegInstance = ffmpeg;

  if (!ffmpegLogBound) {
    ffmpeg.on("log", ({ type, message }) => {
      console.debug(`[ffmpeg:${type}] ${message}`);
    });
    ffmpegLogBound = true;
  }

  ffmpegLoadPromise = ffmpeg
    .load({
      coreURL: await toBlobURL(`${FFMPEG_CORE_BASE_URL}/ffmpeg-core.js`, "text/javascript"),
      wasmURL: await toBlobURL(`${FFMPEG_CORE_BASE_URL}/ffmpeg-core.wasm`, "application/wasm"),
    })
    .then(() => ffmpeg)
    .finally(() => {
      ffmpegLoadPromise = null;
    });

  return ffmpegLoadPromise;
}

async function cleanupTranscodeFiles(ffmpeg: FFmpeg, ...paths: string[]): Promise<void> {
  for (const path of paths) {
    if (!path) {
      continue;
    }
    try {
      await ffmpeg.deleteFile(path);
    } catch {}
  }
}

async function enqueueFfmpegTask<T>(task: () => Promise<T>): Promise<T> {
  const previous = ffmpegExecutionQueue;
  let release!: () => void;
  ffmpegExecutionQueue = new Promise<void>((resolve) => {
    release = resolve;
  });
  await previous;
  try {
    return await task();
  } finally {
    release();
  }
}

export async function transcodeMediaToPlayableAudio(file: File): Promise<File> {
  if (!file) {
    throw new Error("media file is required");
  }

  return enqueueFfmpegTask(async () => {
    const ffmpeg = await getFfmpeg();
    const inputExtension = getFileExtension(file.name) || "bin";
    const inputPath = `${buildTempFileStem()}.${inputExtension}`;
    const outputPath = `${buildTempFileStem()}.${OUTPUT_EXTENSION}`;

    try {
      await ffmpeg.writeFile(inputPath, await fetchFile(file));
      const exitCode = await ffmpeg.exec([
        "-i",
        inputPath,
        "-vn",
        "-map_metadata",
        "-1",
        "-ac",
        String(OUTPUT_CHANNELS),
        "-ar",
        String(OUTPUT_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        outputPath,
      ]);
      if (exitCode !== 0) {
        throw new Error(`ffmpeg exited with code ${exitCode}`);
      }

      const outputData = await ffmpeg.readFile(outputPath);
      if (!(outputData instanceof Uint8Array)) {
        throw new Error("ffmpeg output is not binary audio data");
      }
      const fileBytes = new Uint8Array(outputData.byteLength);
      fileBytes.set(outputData);

      return new File(
        [fileBytes],
        buildOutputFileName(file.name),
        {
          type: OUTPUT_MIME_TYPE,
          lastModified: Date.now(),
        },
      );
    } finally {
      await cleanupTranscodeFiles(ffmpeg, inputPath, outputPath);
    }
  });
}
