// Not ctx.logger directly: the core logic must be unit-testable without cordis, and the
// per-step DEBUG diagnostics on the hot path have to be skipped before the string is built.
// The plugin entry adapts ctx.logger into a LogSink (see createCordisSink).

import type { ContextGuardLogLevel } from '../config.ts';

export interface LogSink {
  debug(message: string): void;
  info(message: string): void;
  warn(message: string): void;
  error(message: string): void;
}

/** Structural subset of the cordis logger, so we don't import the runtime package. */
export interface CordisLikeLogger {
  debug?(message: string): void;
  info?(message: string): void;
  warn?(message: string): void;
  error?(message: string): void;
}

const LEVEL_ORDER: Record<ContextGuardLogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40
};

/** Lets the plugin's lines be filtered out of the DSH log. */
export const LOG_PREFIX = '[context-guard]';

// ctx.logger may not be ready yet (construction can happen before `ready`), so a missing
// method falls back to console instead of throwing.
export function createCordisSink(logger: CordisLikeLogger | undefined): LogSink {
  const call = (level: keyof LogSink, message: string): void => {
    const fn = logger?.[level];
    if (typeof fn === 'function') {
      try {
        fn.call(logger, message);
        return;
      } catch {
        // logging must never break the main path; fall back to console.
      }
    }
    /* eslint-disable no-console */
    if (level === 'error') console.error(message);
    else if (level === 'warn') console.warn(message);
    else console.log(message);
    /* eslint-enable no-console */
  };
  return {
    debug: (message) => call('debug', message),
    info: (message) => call('info', message),
    warn: (message) => call('warn', message),
    error: (message) => call('error', message)
  };
}

/** Collecting sink for tests. */
export class MemorySink implements LogSink {
  readonly lines: { level: ContextGuardLogLevel; message: string }[] = [];
  debug(message: string): void {
    this.lines.push({ level: 'debug', message });
  }
  info(message: string): void {
    this.lines.push({ level: 'info', message });
  }
  warn(message: string): void {
    this.lines.push({ level: 'warn', message });
  }
  error(message: string): void {
    this.lines.push({ level: 'error', message });
  }
  at(level: ContextGuardLogLevel): string[] {
    return this.lines.filter((line) => line.level === level).map((line) => line.message);
  }
}

/** Level-filtered logger. Model-visible text picks its language elsewhere. */
export class GuardLogger {
  private readonly sink: LogSink;
  private readonly threshold: number;

  constructor(level: ContextGuardLogLevel, sink: LogSink) {
    this.sink = sink;
    this.threshold = LEVEL_ORDER[level];
  }

  /** Lets callers skip building an expensive message. */
  enabled(level: ContextGuardLogLevel): boolean {
    return LEVEL_ORDER[level] >= this.threshold;
  }

  debug(message: string): void {
    if (this.enabled('debug')) this.sink.debug(`${LOG_PREFIX} ${message}`);
  }
  info(message: string): void {
    if (this.enabled('info')) this.sink.info(`${LOG_PREFIX} ${message}`);
  }
  warn(message: string): void {
    if (this.enabled('warn')) this.sink.warn(`${LOG_PREFIX} ${message}`);
  }
  error(message: string): void {
    if (this.enabled('error')) this.sink.error(`${LOG_PREFIX} ${message}`);
  }

  /** Single-line message, cause chain included. */
  static describe(error: unknown): string {
    if (error instanceof Error) {
      const cause = (error as { cause?: unknown }).cause;
      const suffix = cause === undefined ? '' : ` <- ${GuardLogger.describe(cause)}`;
      return `${error.message}${suffix}`;
    }
    return String(error);
  }
}
