import type { ResolvedConfig } from '../config.ts';
import type { TokenEstimator } from '../utils/tokenizer.ts';
import type { CompressionResult } from './types.ts';

/** Placeholder key for omitted items; the prefix keeps it from colliding with real fields. */
export const OMITTED_MARKER = '__context_guard_omitted_items';

export const TRUNCATED_MARKER = '...[truncated]';

export interface JsonCompressionStats {
  truncatedStrings: number;
  omittedItems: number;
  depthLimited: boolean;
}

/** Per recursion level. */
interface JsonLimits {
  readonly stringChars: number;
  readonly headItems: number;
  readonly tailItems: number;
  readonly maxDepth: number;
}

/** Returns undefined when the text is not JSON; not null, since null is a valid JSON value. */
export function tryParseJson(text: string): unknown {
  const trimmed = text.trim();
  if (trimmed.length === 0) return undefined;
  const first = trimmed[0];
  // cheap reject: a JSON document starts with one of these, so logs and tables never
  // pay for a full JSON.parse.
  if (first !== undefined && '{["-0123456789tfn'.includes(first) === false) return undefined;
  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return undefined;
  }
}

/** Keeps both ends: the tail usually holds file names, error codes and units. */
function truncateString(value: string, maxChars: number, stats: JsonCompressionStats): string {
  if (value.length <= maxChars) return value;
  const headChars = Math.max(1, Math.floor(maxChars * 0.6));
  const tailChars = Math.max(0, maxChars - headChars);
  stats.truncatedStrings += 1;
  const removed = value.length - headChars - tailChars;
  const tail = tailChars > 0 ? value.slice(value.length - tailChars) : '';
  return `${value.slice(0, headChars)}${TRUNCATED_MARKER}${removed} chars, kept ${headChars}+${tailChars}${tail}`;
}

function compressValue(value: unknown, limits: JsonLimits, stats: JsonCompressionStats, depth: number): unknown {
  if (typeof value === 'string') return truncateString(value, limits.stringChars, stats);
  if (value === null || typeof value !== 'object') return value;
  if (depth >= limits.maxDepth) {
    stats.depthLimited = true;
    // keep the type and size of the collapsed subtree.
    if (Array.isArray(value)) return `[Array(${value.length}) depth-limited]`;
    return { [OMITTED_MARKER]: `object depth-limited (${Object.keys(value).length} keys)` };
  }
  if (Array.isArray(value)) {
    const head = limits.headItems;
    const tail = limits.tailItems;
    if (value.length <= head + tail || value.length <= head) {
      return value.map((item) => compressValue(item, limits, stats, depth + 1));
    }
    const kept = [
      ...value.slice(0, head).map((item) => compressValue(item, limits, stats, depth + 1)),
      { [OMITTED_MARKER]: value.length - head - tail },
      ...value.slice(value.length - tail).map((item) => compressValue(item, limits, stats, depth + 1))
    ];
    stats.omittedItems += value.length - head - tail;
    return kept;
  }
  const record = value as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(record)) {
    const child = record[key];
    if (child === undefined) continue; // stringify would drop it too; explicit here so the stats see it.
    out[key] = compressValue(child, limits, stats, depth + 1);
  }
  return out;
}

function limitsForLevel(config: ResolvedConfig, level: number): JsonLimits {
  const shrink = 2 ** level;
  return {
    stringChars: Math.max(48, Math.floor(config.jsonMaxStringChars / shrink)),
    headItems: Math.max(2, Math.floor(config.jsonHeadItems / shrink)),
    tailItems: Math.max(0, Math.floor(config.jsonTailItems / shrink)),
    // deeper levels cap depth too, so deep nesting cannot eat the whole budget.
    maxDepth: Math.max(3, 12 - level * 2)
  };
}

/** Last resort: the JSON is no longer valid, but the budget holds and the cut is marked. */
function hardTruncate(text: string, budgetTokens: number, estimator: TokenEstimator): string {
  if (estimator.estimate(text) <= budgetTokens) return text;
  // binary search for the largest prefix that fits: the estimator is non-linear for CJK.
  let low = 0;
  let high = text.length;
  while (low < high) {
    const mid = Math.floor((low + high + 1) / 2);
    if (estimator.estimate(text.slice(0, mid)) <= budgetTokens) low = mid;
    else high = mid - 1;
  }
  const kept = text.slice(0, low);
  const dropped = text.length - low;
  return `${kept}${TRUNCATED_MARKER}${dropped} chars of JSON omitted to fit the token budget`;
}

/** Returns null when the input is not JSON, so the caller can try another compressor. */
export function compressJson(text: string, config: ResolvedConfig, estimator: TokenEstimator): CompressionResult | null {
  const parsed = tryParseJson(text);
  if (parsed === undefined) return null;

  const budget = config.targetToolOutputTokens;
  const notes: string[] = [];
  let stats: JsonCompressionStats = { truncatedStrings: 0, omittedItems: 0, depthLimited: false };
  let rendered = text;
  let strategy = 'json:minify';
  let level = 0;

  // 6 levels take us from plain minify down to a bare skeleton; past that, hard truncate.
  for (; level <= 6; level += 1) {
    stats = { truncatedStrings: 0, omittedItems: 0, depthLimited: false };
    const value = compressValue(parsed, limitsForLevel(config, level), stats, 0);
    rendered = JSON.stringify(value) ?? '';
    if (estimator.estimate(rendered) <= budget) break;
  }

  const before = estimator.estimate(text);
  const after = estimator.estimate(rendered);
  if (after > budget) {
    rendered = hardTruncate(rendered, budget, estimator);
    strategy = 'json:hard-truncate';
    notes.push('still over budget after structural compression; hard-truncated (the JSON is no longer valid)');
  } else if (stats.omittedItems > 0 || stats.truncatedStrings > 0 || stats.depthLimited) {
    strategy = 'json:sample+truncate';
  }
  if (stats.truncatedStrings > 0) notes.push(`truncated ${stats.truncatedStrings} long string values (head and tail kept)`);
  if (stats.omittedItems > 0) notes.push(`sampled out ${stats.omittedItems} array items, total recorded under ${OMITTED_MARKER}`);
  if (stats.depthLimited) notes.push('nested values past the depth limit were collapsed to a summary');
  notes.push(`token estimate ${before} -> ${estimator.estimate(rendered)}`);

  return { text: rendered, strategy, notes };
}
