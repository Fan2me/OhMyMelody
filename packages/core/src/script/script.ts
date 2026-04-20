export function isLikelyHtmlText(text: unknown): boolean {
  const value = String(text || '').trim().slice(0, 200).toLowerCase();
  return value.startsWith('<!doctype') || value.startsWith('<html') || value.includes('<!doctype html');
}

export function getPyodidePathAppendPython(): string {
  return 'import sys; sys.path.append(".")';
}

export function getCFPChunkExecutionPython(): string {
  return `
import cfp
W = cfp.cfp_process_from_array(x_pcm, fs_pcm, model_type="melody")
`;
}

export function getCFPProfileReadPython(): string {
  return 'import cfp\ncfp.get_last_cfp_profile_json()';
}

export function getCFPChunkCleanupPython(): string {
  return `
import gc
for _name in ("x_pcm", "fs_pcm", "W"):
    try:
        del globals()[_name]
    except KeyError:
        pass
gc.collect()
`;
}

export async function loadExternalScript(
  documentRef: Document | null,
  scriptUrl: string,
): Promise<boolean> {
  if (
    !documentRef ||
    !documentRef.createElement ||
    !documentRef.head ||
    typeof scriptUrl !== 'string' ||
    !scriptUrl.trim()
  ) {
    throw new Error('script loading is unavailable');
  }

  return await new Promise<boolean>((resolve, reject) => {
    try {
      const script = documentRef.createElement('script');
      script.src = scriptUrl;
      script.onload = () => resolve(true);
      script.onerror = () => reject(new Error(`Failed to load script from ${scriptUrl}`));
      documentRef.head.appendChild(script);
    } catch (error) {
      reject(error);
    }
  });
}
