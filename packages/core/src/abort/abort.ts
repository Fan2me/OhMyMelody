const DEFAULT_ABORT_REASON = 'Operation aborted';

type AbortLikeError = Error & {
  code?: string | number;
  reason?: unknown;
  type?: string;
};

function reasonToMessage(reason: unknown, fallback = DEFAULT_ABORT_REASON): string {
  if (reason instanceof Error && reason.message) {
    return reason.message;
  }
  if (typeof reason === 'string' && reason.trim()) {
    return reason.trim();
  }
  return fallback;
}

export function createAbortError(reason: unknown = DEFAULT_ABORT_REASON): Error {
  const message = reasonToMessage(reason);
  if (typeof DOMException !== 'undefined') {
    return new DOMException(message, 'AbortError');
  }

  const error = new Error(message) as AbortLikeError;
  error.name = 'AbortError';
  error.code = 'ABORT_ERR';
  error.reason = reason;
  return error;
}

export function isAbortError(error: unknown): boolean {
  const err = error as AbortLikeError | null | undefined;
  if (!err) {
    return false;
  }
  return err.name === 'AbortError' || err.code === 'ABORT_ERR' || err.code === 20 || err.type === 'aborted';
}

export function throwIfAborted(
  signal: AbortSignal | null | undefined,
  reason: unknown = DEFAULT_ABORT_REASON,
): void {
  if (signal?.aborted) {
    throw createAbortError(signal.reason ?? reason);
  }
}
