// Compaction decisions cannot call a model, and ctx.tokenMeter only measures session events
// that already happened, so string estimates come from here.
//
// Calibrated for DeepSeek: ~0.6 token per CJK char, ~4 chars per token for ASCII source and
// logs. Overestimating compacts early; underestimating runs into the context window.

/** The narrow interface every compressor depends on, so the implementation stays swappable. */
export interface TokenEstimator {
  /** Implementation name; used in logs and test assertions. */
  readonly name: string;
  estimate(text: string): number;
}

/** CJK ideographs, kana, hangul, full-width punctuation. */
const CJK_PATTERN = /[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef\uac00-\ud7af]/gu;

/** Everything that is not CJK, matched in runs. */
const ASCII_RUN_PATTERN = /[^\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef\uac00-\ud7af]+/gu;

export class HeuristicTokenEstimator implements TokenEstimator {
  readonly name = 'heuristic';
  private readonly cjkPerChar: number;
  private readonly asciiPerChar: number;

  constructor(cjkPerChar = 0.6, asciiPerChar = 0.25) {
    this.cjkPerChar = cjkPerChar;
    this.asciiPerChar = asciiPerChar;
  }

  estimate(text: string): number {
    if (text.length === 0) return 0;
    let tokens = 0;
    for (const match of text.matchAll(CJK_PATTERN)) tokens += match.length * this.cjkPerChar;
    for (const match of text.matchAll(ASCII_RUN_PATTERN)) tokens += match[0].length * this.asciiPerChar;
    // non-empty text always occupies at least one token.
    return Math.max(1, Math.ceil(tokens));
  }
}

/** Small LRU around an estimator: the same text gets re-estimated at several thresholds. */
export class CachedTokenEstimator implements TokenEstimator {
  readonly name: string;
  private readonly cache = new Map<string, number>();
  private readonly inner: TokenEstimator;
  private readonly maxEntries: number;

  constructor(inner: TokenEstimator, maxEntries = 512) {
    this.inner = inner;
    this.maxEntries = maxEntries;
    this.name = `${inner.name}+cache`;
  }

  estimate(text: string): number {
    if (this.maxEntries <= 0) return this.inner.estimate(text);
    const hit = this.cache.get(text);
    if (hit !== undefined) {
      // Map keeps insertion order: re-setting a hit moves it to the end.
      this.cache.delete(text);
      this.cache.set(text, hit);
      return hit;
    }
    const value = this.inner.estimate(text);
    this.cache.set(text, value);
    if (this.cache.size > this.maxEntries) {
      const oldest = this.cache.keys().next();
      if (!oldest.done) this.cache.delete(oldest.value);
    }
    return value;
  }

  /** Called on config hot reload. */
  clear(): void {
    this.cache.clear();
  }
}

/** Wraps `gpt-tokenizer`'s synchronous `encode`. */
class GptTokenEstimator implements TokenEstimator {
  readonly name = 'gpt-tokenizer';
  private readonly encode: (text: string) => number[];

  constructor(encode: (text: string) => number[]) {
    this.encode = encode;
  }

  estimate(text: string): number {
    return this.encode(text).length;
  }
}

export interface TokenEstimatorResult {
  /** Already wrapped in the cache. */
  readonly estimator: TokenEstimator;
  /** false means the heuristic was used; `note` says why. */
  readonly exact: boolean;
  /** For logs. */
  readonly note: string;
}

/** Shared, stateless default. */
export const defaultTokenEstimator: TokenEstimator = new CachedTokenEstimator(new HeuristicTokenEstimator());

// Dynamic import on purpose: gpt-tokenizer is an optional dependency, and the offline
// profile (nothing extra installed) still has to work.
export async function createTokenEstimator(exact = true): Promise<TokenEstimatorResult> {
  if (!exact) {
    return { estimator: defaultTokenEstimator, exact: false, note: 'heuristic estimator (exact tokenizer disabled by config)' };
  }
  try {
    const mod = (await import('gpt-tokenizer')) as { encode?: (text: string) => number[] };
    if (typeof mod.encode !== 'function') {
      return { estimator: defaultTokenEstimator, exact: false, note: 'gpt-tokenizer has no encode(); using heuristic estimator' };
    }
    return {
      estimator: new CachedTokenEstimator(new GptTokenEstimator(mod.encode), 256),
      exact: true,
      note: 'gpt-tokenizer encode() (exact for GPT-style BPE vocabularies)'
    };
  } catch {
    return {
      estimator: defaultTokenEstimator,
      exact: false,
      note: 'gpt-tokenizer not installed; using calibrated heuristic estimator (CJK 0.6 tok/char, ASCII 4 chars/tok)'
    };
  }
}

/** Falls back to 0 when the value cannot be serialized. */
export function estimateJsonTokens(value: unknown, estimator: TokenEstimator): number {
  try {
    const text = typeof value === 'string' ? value : JSON.stringify(value);
    return text === undefined ? 0 : estimator.estimate(text);
  } catch {
    return 0;
  }
}

/** Structurally compatible with DSH's TextBlock. */
export interface TextBlockLike {
  readonly type: string;
  readonly text?: string;
}

/** Only text and reasoning blocks are counted. */
export function estimateBlocks(blocks: readonly TextBlockLike[], estimator: TokenEstimator): number {
  let total = 0;
  for (const block of blocks) {
    if (typeof block.text === 'string' && (block.type === 'text' || block.type === 'reasoning')) {
      total += estimator.estimate(block.text);
    }
  }
  return total;
}
