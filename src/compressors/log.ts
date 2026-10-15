import type { ResolvedConfig } from '../config.ts';
import type { TokenEstimator } from '../utils/tokenizer.ts';
import type { CompressionResult } from './types.ts';

export type LogLevel = 'error' | 'warn' | 'info' | 'debug' | 'other';

export interface LogLine {
  /** Original line, indentation included. */
  readonly raw: string;
  readonly level: LogLevel;
  /** Form used for repeat-folding: timestamps, numbers and hex addresses stripped. */
  readonly normalized: string;
}

/** First match wins, so the order is the priority. */
const LEVEL_PATTERNS: readonly (readonly [LogLevel, RegExp])[] = [
  ['error', /(?:^|[^a-z])(?:ERROR|ERR|FATAL|CRITICAL|PANIC|SEVERE|FAIL(?:ED|URE)?|Exception|Traceback|Unhandled|rejected)(?:[^a-z]|$)/i],
  ['warn', /(?:^|[^a-z])(?:WARN(?:ING)?|DEPRECATED)(?:[^a-z]|$)/i],
  ['info', /(?:^|[^a-z])(?:INFO|NOTICE)(?:[^a-z]|$)/i],
  ['debug', /(?:^|[^a-z])(?:DEBUG|TRACE|VERBOSE|FINE)(?:[^a-z]|$)/i]
];

/** Timestamps, durations, bare numbers, hex addresses, UUIDs. */
const VOLATILE_PATTERNS: readonly RegExp[] = [
  /\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?/g,
  /\d{2}:\d{2}:\d{2}(?:\.\d+)?/g,
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi,
  /\b0x[0-9a-f]+\b/gi,
  /\b\d+(?:\.\d+)?(?:ms|s|us|ns|MB|KB|GB)?\b/g
];

export function classifyLogLine(line: string): LogLevel {
  for (const [level, pattern] of LEVEL_PATTERNS) {
    if (pattern.test(line)) return level;
  }
  return 'other';
}

export function normalizeLogLine(line: string): string {
  let out = line.trim();
  for (const pattern of VOLATILE_PATTERNS) out = out.replace(pattern, '<n>');
  return out;
}

export function parseLogLines(text: string): LogLine[] {
  const lines = text.split(/\r?\n/);
  return lines.map((raw) => ({ raw, level: classifyLogLine(raw), normalized: normalizeLogLine(raw) }));
}

function looksLikeContinuation(line: LogLine): boolean {
  const trimmed = line.raw.trim();
  if (trimmed.length === 0) return false;
  if (/^\s+\S/.test(line.raw)) return true; // indented
  return /^(?:at\s|Caused by:|\s*\.\.\.\s*\d+ more|\.{3}|Suppressed:|caused by)/i.test(trimmed);
}

interface FoldedLine {
  readonly line: LogLine;
  readonly count: number;
  readonly firstIndex: number;
}

function foldRepeats(lines: readonly LogLine[]): FoldedLine[] {
  const out: FoldedLine[] = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line === undefined) continue;
    const previous = out[out.length - 1];
    // only non-empty identical neighbours fold; blank lines carry the paragraph structure.
    if (previous !== undefined && line.normalized.length > 0 && previous.line.normalized === line.normalized) {
      out[out.length - 1] = { line: previous.line, count: previous.count + 1, firstIndex: previous.firstIndex };
      continue;
    }
    out.push({ line, count: 1, firstIndex: index });
  }
  return out;
}

/** Returns null for anything too short to be worth the log strategy. */
export function compressLog(text: string, config: ResolvedConfig, estimator: TokenEstimator): CompressionResult | null {
  const lines = parseLogLines(text);
  // fewer than MIN_LOG_LINES: the structural overhead can outweigh the saving.
  if (lines.length < 8) return null;

  const folded = foldRepeats(lines);
  const counts: Record<LogLevel, number> = { error: 0, warn: 0, info: 0, debug: 0, other: 0 };
  for (const line of lines) counts[line.level] += 1;

  // issue lines are kept whole, timestamps included: they anchor the ordering.
  const issueIndexes: number[] = [];
  for (let index = 0; index < folded.length && issueIndexes.length < config.logMaxIssues; index += 1) {
    const level = folded[index]?.line.level;
    if (level === 'error' || level === 'warn') issueIndexes.push(index);
  }
  const issueSet = new Set<number>();
  for (const index of issueIndexes) {
    for (let offset = 0; offset <= config.logContextLines; offset += 1) {
      const target = folded[index + offset];
      if (target === undefined) break;
      if (offset > 0 && !looksLikeContinuation(target.line)) break;
      issueSet.add(index + offset);
    }
  }

  // head is the command and environment, tail the summary - both always survive.
  const headCount = Math.min(config.logHeadLines, folded.length);
  const tailStart = Math.max(headCount, folded.length - config.logTailLines);
  const selected = new Set<number>();
  for (let index = 0; index < headCount; index += 1) selected.add(index);
  for (const index of issueSet) selected.add(index);
  for (let index = tailStart; index < folded.length; index += 1) selected.add(index);

  const out: string[] = [];
  const header =
    `[context-guard/log] ${lines.length} lines -> ${selected.size} shown · ${counts.error} error · ${counts.warn} warn · ${counts.info} info · ${counts.debug} debug`;
  out.push(header);
  let previousIndex = -1;
  for (let index = 0; index < folded.length; index += 1) {
    if (!selected.has(index)) continue;
    const entry = folded[index];
    if (entry === undefined) continue;
    if (previousIndex !== -1 && index > previousIndex + 1) {
      const omitted = index - previousIndex - 1;
      out.push(`  ... (${omitted} lines omitted) ...`);
    }
    const suffix = entry.count > 1 ? `  (x${entry.count})` : '';
    out.push(`${entry.line.raw}${suffix}`);
    previousIndex = index;
  }

  const foldedRepeats = lines.length - folded.length;
  const notes: string[] = [];
  if (foldedRepeats > 0) notes.push(`folded ${foldedRepeats} consecutive repeats (marked xN)`);
  if (counts.error + counts.warn > config.logMaxIssues) {
    notes.push(`${counts.error + counts.warn} issue lines, only the first ${config.logMaxIssues} kept (the full log stays retrievable)`);
  }
  const rendered = out.join('\n');
  notes.push(`token estimate ${estimator.estimate(text)} -> ${estimator.estimate(rendered)}`);

  return { text: rendered, strategy: 'log:fold+prioritize', notes };
}
