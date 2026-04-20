import { getModuleLogger } from "@ohm/core/logging/logger.js";
import { decodeInputAudio } from "../analysis.js";
import type { AnalyzeInput } from "../types.js";

const audioLogger = getModuleLogger("core.runtime.audio");

export interface AudioResult {
  pcm: Float32Array;
  fs: number;
  mode?: string;
}

export class AudioManager
{
  private decodedAudio: AudioResult | null = null;

  async setAudio(input: AnalyzeInput): Promise<AudioResult> {
    this.decodedAudio = await decodeInputAudio(input);
    audioLogger.info("runtime audio decoded: PCM ready");
    return this.decodedAudio;
  }

  getAudio(): AudioResult | null {
    return this.decodedAudio;
  }

  getPcmChunk(start: number, end: number): Float32Array {
    if (!this.decodedAudio) {
      throw new Error("audio is unavailable");
    }
    return this.decodedAudio.pcm.slice(start, end);
  }
}
