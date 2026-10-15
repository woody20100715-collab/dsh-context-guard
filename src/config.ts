// No runtime @deepseek-ai/* imports on purpose: the unit tests must run without DSH installed.

/** `auto` follows the session's main language. */
export type SummaryLanguagePreference = 'auto' | 'zh' | 'en';

export type ContextGuardLogLevel = 'debug' | 'info' | 'warn' | 'error';

/** User-writable config (the `config` field in `cordis.patch.yml`); defaults in {@link DEFAULTS}. */
export interface ContextGuardConfig {
  /** false keeps the plugin mounted (it still provides ctx.compaction) but compresses nothing. */
  enabled?: boolean;
  /** Registers agent/pre-step pressure compaction plus context-overflow recovery. */
  auto?: boolean;
  /** Share of the model context window that triggers compaction. */
  thresholdRatio?: number;
  /** Share of the tail kept verbatim. */
  retainRatio?: number;
  /** When > 0, overrides retainRatio with an absolute token count. */
  retainTokens?: number;
  maxSummaryTokens?: number;
  compactionRetries?: number;
  maxOverflowRetries?: number;
  /** Keep exact values (numbers, dates, paths, URLs, commands, code) instead of paraphrasing. */
  faithful?: boolean;
  summaryLanguage?: SummaryLanguagePreference;
  /** LRU over summary inputs; 0 disables the cache. */
  summaryCacheSize?: number;
  /** Empty provider/model = follow the session's latest route. */
  summarizationProvider?: string;
  summarizationModel?: string;

  enableToolCompression?: boolean;
  maxToolOutputTokens?: number;
  /** Below this, don't even try: the overhead can outweigh the saving. */
  minToolOutputTokens?: number;
  /** What the compressor aims for; still over budget means tighter truncation. */
  targetToolOutputTokens?: number;

  jsonMaxStringChars?: number;
  jsonHeadItems?: number;
  jsonTailItems?: number;

  logHeadLines?: number;
  logTailLines?: number;
  /** Cap on kept ERROR/WARN lines. */
  logMaxIssues?: number;
  /** Lines kept after each issue line, so stack frames and "Caused by" survive. */
  logContextLines?: number;

  tableHeadRows?: number;
  tableTailRows?: number;

  textHeadChars?: number;
  textTailChars?: number;
  textExtractSignals?: boolean;

  enableCCR?: boolean;
  ccrMaxEntries?: number;
  ccrMaxBytes?: number;
  ccrTtlMs?: number;
  ccrPersist?: boolean;
  /** Leading `~` is expanded. */
  ccrStorePath?: string;
  /** Larger entries stay in memory only, so one huge output cannot fill the disk. */
  ccrMaxChars?: number;
  // Retrieval is never re-compressed, so this is the only guard against blowing the
  // window in one go: hand back one large page and let the model page with offset.
  retrieveMaxChars?: number;

  assemblyEnabled?: boolean;
  /** Inject the "part of the history was compacted, use context_retrieve" notice. */
  noticeEnabled?: boolean;
  /** Reserved for the model's answer; deducted from the budget before compacting. */
  reservedOutputTokens?: number;
  /** Conservative fallback when the system prompt cannot be measured. */
  systemPromptTokenFallback?: number;

  logLevel?: ContextGuardLogLevel;
}

/** Validated config: every field present, cross-field relations already checked. */
export interface ResolvedConfig extends Required<ContextGuardConfig> {
  /** Extra keys seen during validation (the Loader may pass them through); warning only. */
  readonly unknownKeys: readonly string[];
}

export const DEFAULTS = {
  enabled: true,
  auto: true,
  thresholdRatio: 0.8,
  retainRatio: 0.16,
  retainTokens: 0,
  maxSummaryTokens: 2048,
  compactionRetries: 2,
  maxOverflowRetries: 2,
  faithful: true,
  summaryLanguage: 'auto',
  summaryCacheSize: 32,
  summarizationProvider: '',
  summarizationModel: '',

  enableToolCompression: true,
  maxToolOutputTokens: 2000,
  minToolOutputTokens: 400,
  // ~55% of the trigger threshold: landing right on it would re-trigger on the next step.
  targetToolOutputTokens: 1100,

  jsonMaxStringChars: 400,
  jsonHeadItems: 20,
  jsonTailItems: 5,

  logHeadLines: 15,
  logTailLines: 20,
  logMaxIssues: 40,
  logContextLines: 3,

  tableHeadRows: 15,
  tableTailRows: 10,

  textHeadChars: 1200,
  textTailChars: 800,
  textExtractSignals: true,

  enableCCR: true,
  ccrMaxEntries: 100,
  ccrMaxBytes: 10 * 1024 * 1024,
  ccrTtlMs: 60 * 60 * 1000,
  ccrPersist: true,
  ccrStorePath: '~/.dsh/context-guard/ccr',
  ccrMaxChars: 2 * 1024 * 1024,
  retrieveMaxChars: 60_000,

  assemblyEnabled: true,
  noticeEnabled: true,
  reservedOutputTokens: 8192,
  systemPromptTokenFallback: 4000,

  logLevel: 'info'
} as const satisfies Required<ContextGuardConfig>;

/** Messages reach the user as-is, so they should name the fix. */
export class ContextGuardConfigError extends Error {
  /** Empty key = cross-field error. */
  readonly key: string;
  constructor(key: string, message: string) {
    super(key === '' ? message : `dsh-context-guard config: ${key} — ${message}`);
    this.name = 'ContextGuardConfigError';
    this.key = key;
  }
}

const KNOWN_KEYS: ReadonlySet<string> = new Set(Object.keys(DEFAULTS));
const LOG_LEVELS: ReadonlySet<string> = new Set(['debug', 'info', 'warn', 'error']);
const LANGUAGES: ReadonlySet<string> = new Set(['auto', 'zh', 'en']);

// Throws rather than silently fixing: a value that is set but ignored is undiagnosable,
// and the Cordis Loader shows construction-time errors to the user anyway.
export function resolveConfig(raw?: ContextGuardConfig | null): ResolvedConfig {
  const input = (raw ?? {}) as Record<string, unknown>;
  const unknownKeys = Object.keys(input).filter((key) => !KNOWN_KEYS.has(key));
  const out: Record<string, unknown> = { ...DEFAULTS };

  for (const key of KNOWN_KEYS) {
    const value = input[key];
    if (value !== undefined && value !== null) out[key] = value;
  }

  const num = (key: string, min: number, max: number): number => {
    const value = out[key];
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new ContextGuardConfigError(key, `expected a finite number, received ${JSON.stringify(value)}`);
    }
    if (value < min || value > max) {
      throw new ContextGuardConfigError(key, `expected a number in [${min}, ${max}], received ${value}`);
    }
    return value;
  };
  const bool = (key: string): boolean => {
    const value = out[key];
    if (typeof value !== 'boolean') {
      throw new ContextGuardConfigError(key, `expected a boolean, received ${JSON.stringify(value)}`);
    }
    return value;
  };
  const str = (key: string): string => {
    const value = out[key];
    if (typeof value !== 'string') {
      throw new ContextGuardConfigError(key, `expected a string, received ${JSON.stringify(value)}`);
    }
    return value;
  };
  const enumOf = <T extends string>(key: string, allowed: ReadonlySet<string>): T => {
    const value = out[key];
    if (typeof value !== 'string' || !allowed.has(value)) {
      throw new ContextGuardConfigError(key, `expected one of ${[...allowed].join(' | ')}, received ${JSON.stringify(value)}`);
    }
    return value as T;
  };

  bool('enabled');
  bool('auto');
  num('thresholdRatio', 0.05, 1);
  num('retainRatio', 0, 0.9);
  num('retainTokens', 0, Number.MAX_SAFE_INTEGER);
  num('maxSummaryTokens', 128, 1_000_000);
  num('compactionRetries', 0, 10);
  num('maxOverflowRetries', 0, 10);
  bool('faithful');
  enumOf<SummaryLanguagePreference>('summaryLanguage', LANGUAGES);
  num('summaryCacheSize', 0, 4096);
  str('summarizationProvider');
  str('summarizationModel');

  bool('enableToolCompression');
  num('maxToolOutputTokens', 64, 10_000_000);
  num('minToolOutputTokens', 0, 10_000_000);
  num('targetToolOutputTokens', 32, 10_000_000);
  num('jsonMaxStringChars', 16, 1_000_000);
  num('jsonHeadItems', 0, 10_000);
  num('jsonTailItems', 0, 10_000);
  num('logHeadLines', 0, 10_000);
  num('logTailLines', 0, 10_000);
  num('logMaxIssues', 0, 10_000);
  num('logContextLines', 0, 100);
  num('tableHeadRows', 0, 10_000);
  num('tableTailRows', 0, 10_000);
  num('textHeadChars', 0, 10_000_000);
  num('textTailChars', 0, 10_000_000);
  bool('textExtractSignals');

  bool('enableCCR');
  num('ccrMaxEntries', 0, 1_000_000);
  num('ccrMaxBytes', 0, Number.MAX_SAFE_INTEGER);
  num('ccrTtlMs', 0, Number.MAX_SAFE_INTEGER);
  bool('ccrPersist');
  str('ccrStorePath');
  num('ccrMaxChars', 1024, Number.MAX_SAFE_INTEGER);
  num('retrieveMaxChars', 1024, 10_000_000);

  bool('assemblyEnabled');
  bool('noticeEnabled');
  num('reservedOutputTokens', 0, 10_000_000);
  num('systemPromptTokenFallback', 0, 10_000_000);
  enumOf<ContextGuardLogLevel>('logLevel', LOG_LEVELS);

  // cross-field relations: fail at load time instead of silently at runtime.
  const thresholdRatio = out.thresholdRatio as number;
  const retainRatio = out.retainRatio as number;
  if (retainRatio >= thresholdRatio) {
    throw new ContextGuardConfigError(
      'retainRatio',
      `must be smaller than thresholdRatio (${retainRatio} >= ${thresholdRatio}); otherwise no compactable range can ever be selected`
    );
  }
  const maxToolOutputTokens = out.maxToolOutputTokens as number;
  const minToolOutputTokens = out.minToolOutputTokens as number;
  if (minToolOutputTokens > maxToolOutputTokens) {
    throw new ContextGuardConfigError(
      'minToolOutputTokens',
      `must not exceed maxToolOutputTokens (${minToolOutputTokens} > ${maxToolOutputTokens})`
    );
  }
  const targetToolOutputTokens = out.targetToolOutputTokens as number;
  if (targetToolOutputTokens > maxToolOutputTokens) {
    throw new ContextGuardConfigError(
      'targetToolOutputTokens',
      `must not exceed maxToolOutputTokens (${targetToolOutputTokens} > ${maxToolOutputTokens}); the compressor could never reach its target`
    );
  }
  const summarizationProvider = out.summarizationProvider as string;
  const summarizationModel = out.summarizationModel as string;
  if ((summarizationProvider === '') !== (summarizationModel === '')) {
    throw new ContextGuardConfigError(
      'summarizationModel',
      'set both summarizationProvider and summarizationModel, or neither (neither = follow the session route)'
    );
  }
  if ((out.ccrPersist as boolean) && (out.ccrStorePath as string).trim() === '') {
    throw new ContextGuardConfigError('ccrStorePath', 'must be non-empty when ccrPersist is true');
  }

  return Object.freeze({
    ...(out as Required<ContextGuardConfig>),
    unknownKeys: Object.freeze([...unknownKeys])
  });
}

/** Expands a leading `~`, `~/` or `~\` against `home`. */
export function expandHome(input: string, home: string | undefined): string {
  if (home === undefined || home === '') return input;
  if (input === '~') return home;
  if (input.startsWith('~/') || input.startsWith('~\\')) return `${home}${input.slice(1)}`;
  return input;
}
