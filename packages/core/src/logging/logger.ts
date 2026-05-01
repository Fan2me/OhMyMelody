export type LogLevel = "debug" | "info" | "warn" | "error";

declare global {
  var __OHM_ENABLE_CONSOLE_LOGS: boolean | undefined;
}

export interface LogEntry {
  id: string;
  module: string;
  level: LogLevel;
  message: string;
  data?: unknown;
  error: Error | null;
  timestampMs: number;
}

export interface LogSink {
  handle(entry: LogEntry): void;
}

export interface StructuredLogger {
  module: string;
  emit(level: LogLevel, message: unknown, data?: unknown): LogEntry | null;
  debug(message: unknown, data?: unknown): LogEntry | null;
  info(message: unknown, data?: unknown): LogEntry | null;
  warn(message: unknown, data?: unknown): LogEntry | null;
  error(message: unknown, data?: unknown): LogEntry | null;
  log(message: unknown, data?: unknown): LogEntry | null;
  child(childModule?: string): StructuredLogger;
  addSink(sink: LogSink | ((entry: LogEntry) => void)): boolean;
  setMinLevel(nextLevel: LogLevel): LogLevel;
  getMinLevel(): LogLevel;
}

export type LoggerInput =
  | StructuredLogger
  | Partial<Pick<Console, "log" | "info" | "warn" | "error" | "debug">>
  | ((line: string, entry: LogEntry) => void)
  | null
  | undefined;

const LOG_LEVEL_PRIORITY: Readonly<Record<LogLevel, number>> = Object.freeze({
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
});

const LOG_LEVELS: readonly LogLevel[] = Object.freeze([
  "debug",
  "info",
  "warn",
  "error",
]);

const defaultConsoleLogger = createConsoleGate(console);

let rootLoggerInstance: StructuredLogger = getLogger(defaultConsoleLogger, {
  module: "root",
});
const moduleLoggerCache = new Map<string, StructuredLogger>();

function normalizeLevel(level: unknown, fallback: LogLevel = "info"): LogLevel {
  const value = String(level || "").toLowerCase();
  return (LOG_LEVELS as readonly string[]).includes(value)
    ? (value as LogLevel)
    : fallback;
}

function shouldLog(level: LogLevel, minLevel: LogLevel): boolean {
  return LOG_LEVEL_PRIORITY[level] >= LOG_LEVEL_PRIORITY[minLevel];
}

function stringifyValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (value instanceof Error) return value.message || String(value);
  if (value === null || value === undefined) return String(value);
  if (
    typeof value === "number" ||
    typeof value === "boolean" ||
    typeof value === "bigint"
  ) {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return Object.prototype.toString.call(value);
  }
}

function resolveMessage(
  message: unknown,
  data: unknown,
): { message: string; data: unknown; error: Error | null } {
  const error =
    message instanceof Error ? message : data instanceof Error ? data : null;
  const payload = data instanceof Error ? undefined : data;
  if (error) {
    const errorMessage = error.message || String(error);
    return {
      message:
        typeof message === "string"
          ? `${message}: ${errorMessage}`
          : errorMessage,
      data: payload,
      error,
    };
  }
  return {
    message: typeof message === "string" ? message : stringifyValue(message),
    data: payload,
    error: null,
  };
}

function toConsoleMethod(
  logger: LoggerInput,
  method: "log" | "info" | "warn" | "error" | "debug",
): ((...args: unknown[]) => void) | null {
  if (!logger || typeof logger !== "object") {
    return null;
  }
  const candidate = logger as Partial<Record<typeof method, unknown>>;
  return typeof candidate[method] === "function"
    ? (candidate[method] as (...args: unknown[]) => void)
    : null;
}

function isDefaultConsoleLoggingEnabled(): boolean {
  return globalThis.__OHM_ENABLE_CONSOLE_LOGS === true;
}

function createConsoleGate(
  target: Pick<Console, "log" | "info" | "warn" | "error" | "debug">,
): Pick<Console, "log" | "info" | "warn" | "error" | "debug"> {
  const forward =
    (method: "log" | "info" | "warn" | "error" | "debug") =>
    (...args: unknown[]) => {
      if (!isDefaultConsoleLoggingEnabled()) {
        return;
      }
      target[method](...args);
    };
  return {
    log: forward("log"),
    info: forward("info"),
    warn: forward("warn"),
    error: forward("error"),
    debug: forward("debug"),
  };
}

function createEntry(
  moduleName: string,
  level: LogLevel,
  message: string,
  data: unknown,
  error: Error | null,
  id: number,
): LogEntry {
  return {
    id: String(id),
    module: moduleName,
    level,
    message,
    data,
    error,
    timestampMs: Date.now(),
  };
}

function createLoggerApi({
  module = "",
  sinks = [],
  minLevel = "info",
}: {
  module?: string;
  sinks?: Array<LogSink | ((entry: LogEntry) => void)>;
  minLevel?: LogLevel;
}): StructuredLogger {
  let nextEntryId = 1;
  let currentMinLevel = normalizeLevel(minLevel);
  const resolvedSinks: LogSink[] = sinks
    .map((sink) => (typeof sink === "function" ? { handle: sink } : sink))
    .filter(
      (sink): sink is LogSink => !!sink && typeof sink.handle === "function",
    );

  function emit(
    level: LogLevel,
    message: unknown,
    data?: unknown,
  ): LogEntry | null {
    const normalizedLevel = normalizeLevel(level);
    if (!shouldLog(normalizedLevel, currentMinLevel)) {
      return null;
    }
    const payload = resolveMessage(message, data);
    const entry = createEntry(
      module,
      normalizedLevel,
      payload.message,
      payload.data,
      payload.error,
      nextEntryId++,
    );
    for (const sink of resolvedSinks) {
      try {
        sink.handle(entry);
      } catch {}
    }
    return entry;
  }

  return {
    module,
    emit,
    debug: (message, data) => emit("debug", message, data),
    info: (message, data) => emit("info", message, data),
    warn: (message, data) => emit("warn", message, data),
    error: (message, data) => emit("error", message, data),
    log: (message, data) => emit("info", message, data),
    child(childModule = "") {
      return createLoggerApi({
        module: [module, childModule].filter(Boolean).join("."),
        sinks: resolvedSinks,
        minLevel: currentMinLevel,
      });
    },
    addSink(sink) {
      const normalized = typeof sink === "function" ? { handle: sink } : sink;
      if (!normalized || typeof normalized.handle !== "function") {
        return false;
      }
      resolvedSinks.push(normalized);
      return true;
    },
    setMinLevel(nextLevel: LogLevel) {
      currentMinLevel = normalizeLevel(nextLevel, currentMinLevel);
      return currentMinLevel;
    },
    getMinLevel() {
      return currentMinLevel;
    },
  };
}

function isStructuredLogger(value: unknown): value is StructuredLogger {
  return (
    !!value &&
    typeof value === "object" &&
    typeof (value as Partial<StructuredLogger>).emit === "function" &&
    typeof (value as Partial<StructuredLogger>).info === "function" &&
    typeof (value as Partial<StructuredLogger>).warn === "function" &&
    typeof (value as Partial<StructuredLogger>).error === "function"
  );
}

function createForwardingSink(target: LoggerInput): LogSink | null {
  if (!target) {
    return null;
  }
  if (typeof target === "function") {
    return {
      handle(entry) {
        target(formatLogEntry(entry), entry);
      },
    };
  }
  if (isStructuredLogger(target)) {
    return {
      handle(entry) {
        target.emit(entry.level, entry.message, entry.data);
      },
    };
  }
  const log = toConsoleMethod(target, "log");
  const info = toConsoleMethod(target, "info");
  const warn = toConsoleMethod(target, "warn");
  const error = toConsoleMethod(target, "error");
  const debug = toConsoleMethod(target, "debug");
  if (!log && !info && !warn && !error && !debug) {
    return null;
  }
  return {
    handle(entry) {
      const line = formatLogEntry(entry);
      switch (entry.level) {
        case "debug":
          (debug ?? info ?? log)?.(line);
          break;
        case "info":
          (info ?? log)?.(line);
          break;
        case "warn":
          (warn ?? info ?? log)?.(line);
          break;
        case "error":
          (error ?? warn ?? info ?? log)?.(line);
          break;
      }
    },
  };
}

function formatLogEntry(entry: LogEntry): string {
  const level = entry.level.toUpperCase();
  const moduleName = entry.module ? `[${entry.module}]` : "";
  const dataText =
    entry.data === undefined ? "" : ` ${stringifyValue(entry.data)}`;
  const errorText = entry.error
    ? ` ${entry.error.message || String(entry.error)}`
    : "";
  return `${level} ${moduleName} ${entry.message}${dataText}${errorText}`.trim();
}

export function createLogger({
  module = "root",
  sinks = [],
  minLevel = "info",
}: {
  module?: string;
  sinks?: Array<LogSink | ((entry: LogEntry) => void)>;
  minLevel?: LogLevel | string;
} = {}): StructuredLogger {
  const normalizedSinks = sinks
    .map(
      (sink) =>
        createForwardingSink(sink as LoggerInput) ??
        (typeof sink === "function" ? { handle: sink } : sink),
    )
    .filter(
      (sink): sink is LogSink => !!sink && typeof sink.handle === "function",
    );
  return createLoggerApi({
    module,
    sinks: normalizedSinks,
    minLevel: normalizeLevel(minLevel),
  });
}

export function getLogger(
  logger: LoggerInput,
  options: { module?: string; minLevel?: LogLevel | string } = {},
): StructuredLogger {
  if (isStructuredLogger(logger)) {
    return options.module ? logger.child(options.module) : logger;
  }
  const sink = createForwardingSink(logger);
  return createLogger({
    module: options.module ?? "root",
    ...(options.minLevel !== undefined ? { minLevel: options.minLevel } : {}),
    sinks: sink ? [sink] : [],
  });
}

export function setRootLogger(logger: LoggerInput): void {
  rootLoggerInstance = getLogger(logger, { module: "root" });
}

export function getRootLogger(): StructuredLogger {
  return rootLoggerInstance;
}

export function getModuleLogger(moduleName: string): StructuredLogger {
  const key = String(moduleName || "").trim() || "root";
  const cached = moduleLoggerCache.get(key);
  if (cached) {
    return cached;
  }

  const logger: StructuredLogger = {
    get module() {
      return key;
    },
    emit(level: LogLevel, message: unknown, data?: unknown) {
      return getRootLogger().child(key).emit(level, message, data);
    },
    debug(message: unknown, data?: unknown) {
      return getRootLogger().child(key).debug(message, data);
    },
    info(message: unknown, data?: unknown) {
      return getRootLogger().child(key).info(message, data);
    },
    warn(message: unknown, data?: unknown) {
      return getRootLogger().child(key).warn(message, data);
    },
    error(message: unknown, data?: unknown) {
      return getRootLogger().child(key).error(message, data);
    },
    log(message: unknown, data?: unknown) {
      return getRootLogger().child(key).log(message, data);
    },
    child(childModule = "") {
      return getModuleLogger([key, childModule].filter(Boolean).join("."));
    },
    addSink(sink) {
      return getRootLogger().child(key).addSink(sink);
    },
    setMinLevel(nextLevel: LogLevel) {
      return getRootLogger().child(key).setMinLevel(nextLevel);
    },
    getMinLevel() {
      return getRootLogger().child(key).getMinLevel();
    },
  };
  moduleLoggerCache.set(key, logger);
  return logger;
}
