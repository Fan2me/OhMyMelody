export interface MediaPreparationResult<TFile extends Blob & { name?: string } = Blob & { name?: string }> {
  originalFile: TFile;
  analysisFile: TFile;
  converted: boolean;
}

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

export async function prepareMediaFileForAnalysis(
  file: Blob & { name?: string },
): Promise<MediaPreparationResult<Blob & { name?: string }>> {
  if (!file) {
    throw new Error('media file is required');
  }

  return {
    originalFile: file,
    analysisFile: file,
    converted: false,
  };
}
