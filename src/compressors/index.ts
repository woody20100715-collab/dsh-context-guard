import type { ResolvedConfig } from '../config.ts';
import type { TokenEstimator } from '../utils/tokenizer.ts';
import { compressJson } from './json.ts';
import { compressLog } from './log.ts';
import { compressTable } from './table.ts';
import { compressText } from './text.ts';
import type { CompressionResult, ContentCategory } from './types.ts';

export type { CompressionResult, ContentCategory } from './types.ts';
export { compressJson, tryParseJson, OMITTED_MARKER, TRUNCATED_MARKER } from './json.ts';
export { compressLog, parseLogLines, classifyLogLine, normalizeLogLine } from './log.ts';
export { compressTable, detectTable } from './table.ts';
export { compressText, extractSignalSentences, signalScore, sliceByTokens, splitParagraphs, splitSentences } from './text.ts';

// Below this saving ratio we keep the original: making the model pay a retrieval
// call to save 3% of tokens is a net loss.
export const MIN_SAVING_RATIO = 0.15;

/** Minimal shape compatible with DSH's TextBlock, so we don't import DSH at runtime. */
export interface GuardContentBlock {
  readonly type: string;
  readonly text?: string;
}

export interface CompressionMeta {
  /** Always true; lets callers and tests recognise a compressed result. */
  readonly compressed: true;
  /** CCR id of the original, when CCR is on. */
  readonly originalId?: string;
  readonly originalTokens: number;
  readonly compressedTokens: number;
  readonly category: ContentCategory;
  /** The strategy that actually ran. */
  readonly strategy: string;
  readonly notes: readonly string[];
  /** Number of blocks rewritten. */
  readonly rewrittenBlocks: number;
}

export interface BlockCompressionOutcome {
  readonly blocks: GuardContentBlock[];
  readonly meta: CompressionMeta;
}

/** Tool name -> category hint; a fallback only, content sniffing wins. */
const TOOL_HINTS: readonly (readonly [RegExp, ContentCategory])[] = [
  [/^(?:grep|glob|list_|search)/i, 'text'],
  [/^(?:pwsh|bash|terminal|shell|run)/i, 'log'],
  [/^(?:read|read_file)/i, 'text'],
  [/^(?:web_search|web_fetch|fetch)/i, 'text']
];

/** Fewer lines than this are not worth the log strategy. */
const MIN_LOG_LINES = 8;

// Order matters: json, table, log, then text. The tool name is only a fallback - the
// same tool can return JSON or a build log, so content beats the name.
export function detectCategory(text: string, toolName?: string): ContentCategory {
  const trimmed = text.trimStart();
  if ((trimmed.startsWith('{') || trimmed.startsWith('[')) && isParsableJson(text)) return 'json';
  if (looksLikeTable(text)) return 'table';
  if (looksLikeLog(text)) return 'log';
  if (toolName !== undefined) {
    for (const [pattern, category] of TOOL_HINTS) {
      if (pattern.test(toolName)) return category === 'log' ? 'log' : category;
    }
  }
  return 'text';
}

function isParsableJson(text: string): boolean {
  try {
    JSON.parse(text.trim());
    return true;
  } catch {
    return false;
  }
}

function looksLikeTable(text: string): boolean {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 3) return false;
  const first = lines[0] ?? '';
  const second = lines[1] ?? '';
  if (first.includes('|') && second.includes('-') && /^\s*\|?\s*:?-{2,}/.test(second)) return true;
  for (const delimiter of [',', '\t', ';']) {
    const counts = lines.slice(0, 12).map((line) => line.split(delimiter).length - 1);
    const expected = counts[0] ?? 0;
    if (expected >= 1 && counts.filter((count) => count === expected).length >= 3) return true;
  }
  return false;
}

function looksLikeLog(text: string): boolean {
  const lines = text.split(/\r?\n/);
  if (lines.length < MIN_LOG_LINES) return false;
  let classified = 0;
  for (const line of lines.slice(0, 200)) {
    if (/(?:^|[^a-z])(?:ERROR|ERR|WARN(?:ING)?|INFO|DEBUG|TRACE|FATAL|CRITICAL|SEVERE)(?:[^a-z]|$)/i.test(line)) {
      classified += 1;
      if (classified >= 2) return true;
    }
  }
  return false;
}

// Sniffing can be wrong, so a strategy that declines falls through to the others
// instead of giving up - that matters most for output from tools we don't know.
export function compressByCategory(
  text: string,
  category: ContentCategory,
  config: ResolvedConfig,
  estimator: TokenEstimator
): CompressionResult | null {
  const order: readonly ContentCategory[] =
    category === 'json' ? ['json', 'table', 'log', 'text']
      : category === 'table' ? ['table', 'json', 'log', 'text']
        : category === 'log' ? ['log', 'table', 'json', 'text']
          : ['text', 'log', 'table', 'json'];
  for (const candidate of order) {
    const result =
      candidate === 'json' ? compressJson(text, config, estimator)
        : candidate === 'table' ? compressTable(text, config, estimator)
          : candidate === 'log' ? compressLog(text, config, estimator)
            : compressText(text, config, estimator);
    if (result === null) continue;
    const notes = candidate === category ? [] : [`detected ${category} but used the ${candidate} strategy`];
    return { text: result.text, strategy: result.strategy, notes: [...notes, ...result.notes] };
  }
  return null;
}

// Per block, never concatenated: images and nested blocks must keep their order, and
// merging would make a nested tool-result look like ordinary text.
export function compressContentBlocks(
  blocks: readonly GuardContentBlock[],
  config: ResolvedConfig,
  estimator: TokenEstimator,
  options?: { toolName?: string }
): BlockCompressionOutcome | null {
  if (!config.enableToolCompression) return null;

  const textBlocks = blocks.filter(
    (block) => typeof block.text === 'string' && (block.type === 'text' || block.type === 'reasoning')
  );
  if (textBlocks.length === 0) return null;

  const originalTokens = textBlocks.reduce((total, block) => total + estimator.estimate(block.text ?? ''), 0);
  if (originalTokens < config.minToolOutputTokens) return null;

  let category: ContentCategory = 'text';
  let strategy = '';
  let rewrittenBlocks = 0;
  let compressedTokens = 0;
  const notes: string[] = [];
  const out: GuardContentBlock[] = [];

  for (const block of blocks) {
    const text = block.text;
    if (typeof text !== 'string' || (block.type !== 'text' && block.type !== 'reasoning')) {
      out.push(block);
      continue;
    }
    const ownTokens = estimator.estimate(text);
    // a block must be over the cap on its own; partially rewritten blocks are hard to line up.
    if (ownTokens <= config.maxToolOutputTokens) {
      out.push(block);
      compressedTokens += ownTokens;
      continue;
    }
    category = detectCategory(text, options?.toolName);
    const result = compressByCategory(text, category, config, estimator);
    if (result === null) {
      out.push(block);
      compressedTokens += ownTokens;
      continue;
    }
    out.push({ type: block.type, text: result.text });
    compressedTokens += estimator.estimate(result.text);
    rewrittenBlocks += 1;
    strategy = result.strategy;
    notes.push(...result.notes);
  }

  if (rewrittenBlocks === 0) return null;
  const savings = originalTokens === 0 ? 0 : (originalTokens - compressedTokens) / originalTokens;
  if (savings < MIN_SAVING_RATIO) {
    return null;
  }

  return {
    blocks: out,
    meta: {
      compressed: true,
      originalTokens,
      compressedTokens,
      category,
      strategy,
      notes,
      rewrittenBlocks
    }
  };
}

// ContentBlock has nowhere to put custom metadata, so the id has to go into the text
// itself - otherwise the model can never call context_retrieve.
export function appendRetrievalFooter(
  blocks: readonly GuardContentBlock[],
  meta: CompressionMeta,
  enableCCR: boolean
): GuardContentBlock[] {
  const saved = meta.originalTokens - meta.compressedTokens;
  const percent = meta.originalTokens === 0 ? 0 : Math.round((saved / meta.originalTokens) * 100);
  const head = `[context-guard] compressed ${meta.category} output: ~${meta.originalTokens} -> ~${meta.compressedTokens} tokens (-${percent}%), strategy=${meta.strategy}.`;
  const recoverable = meta.originalId !== undefined;
  const tail = enableCCR && recoverable
    ? `The full original is available: call context_retrieve with id="${meta.originalId}" if you need details that were omitted.`
    : 'Omitted details are not recoverable; re-run the tool if you need them.';
  const footer = `\n\n${head} ${tail}`;

  const out = [...blocks];
  for (let index = out.length - 1; index >= 0; index -= 1) {
    const block = out[index];
    if (block !== undefined && block.type === 'text' && typeof block.text === 'string') {
      out[index] = { type: 'text', text: `${block.text}${footer}` };
      return out;
    }
  }
  out.push({ type: 'text', text: footer.trimStart() });
  return out;
}
