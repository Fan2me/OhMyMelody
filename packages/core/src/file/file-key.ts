function toHex(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let hex = '';
  for (let i = 0; i < bytes.length; i += 1) {
    hex += (bytes[i] ?? 0).toString(16).padStart(2, '0');
  }
  return hex;
}

function sumBytes(arr: Uint8Array): number {
  let total = 0;
  for (let i = 0; i < arr.length; i += 1) {
    total += arr[i] ?? 0;
  }
  return total;
}

export async function getFileCFPKeyLegacy(
  file: Blob & { name?: string },
): Promise<string> {
  const name = file.name || 'unknown';
  const size = file.size;
  const head = await file.slice(0, 65536).arrayBuffer();
  const tail = await file.slice(Math.max(0, size - 65536), size).arrayBuffer();

  const headSum = sumBytes(new Uint8Array(head));
  const tailSum = sumBytes(new Uint8Array(tail));
  return `${name}_${size}_${headSum}_${tailSum}`;
}

export async function getFileCFPKey(
  file: Blob & { name?: string },
): Promise<string> {
  const name = file.name || 'unknown';
  const size = file.size;

  try {
    if (
      typeof crypto !== 'undefined' &&
      crypto &&
      crypto.subtle &&
      typeof crypto.subtle.digest === 'function'
    ) {
      const buffer = await file.arrayBuffer();
      const slimBuffer = buffer.slice(0, Math.min(buffer.byteLength, 1024 * 1024));
      const digest = await crypto.subtle.digest('SHA-256', slimBuffer);
      return `${name}_${size}_${toHex(digest)}`;
    }
  } catch {
    // Fallback keeps compatibility if subtle digest is unavailable.
  }

  return await getFileCFPKeyLegacy(file);
}
