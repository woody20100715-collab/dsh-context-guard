import type { ResolvedConfig } from '../config.ts';
import type { TokenEstimator } from '../utils/tokenizer.ts';
import type { CompressionResult } from './types.ts';

export const OMIT_MARKER = '[... omitted ...]';

// A sentence scores one point per pattern it matches.
const SIGNAL_PATTERNS: readonly RegExp[] = [
  /(?:[A-Za-z]:\\|\/(?:[\w.-]+\/)+)[\w.-]+/, // file path
  /\b(?:npm|pnpm|yarn|node|npx|git|docker|kubectl|cargo|go|python|pip|pytest|tsc)\s+[\w@./-]+/, // command
  /\b\d+(?:\.\d+)?(?:%|ms|s|MB|GB|KB|tokens?)?\b/, // numbers and metrics
  /\b(?:must|must not|required|shall|should|never|always|deprecated|breaking|注意|必须|禁止|不要|记得)\b/i, // constraints
  /\b(?:error|failed|failure|warning|todo|fixme)\b/i, // risk words
  /[{};()=><]{2,}|=>|::|\bfunction\b|\bclass\b|\bdef\b/, // code-ish
  /https?:\/\/[^\s)]+/ // url
];

export function splitParagraphs(text: string): string[] {
  return text.split(/\n\s*\n/).filter((paragraph) => paragraph.trim().length > 0);
}

export function splitSentences(paragraph: string): string[] {
  return paragraph
    .split(/(?<=[。！？；.!?;])\s*/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length > 0);
}

/** One point per matched pattern; long sentences get a small bonus. */
export function signalScore(sentence: string): number {
  let score = 0;
  for (const pattern of SIGNAL_PATTERNS) if (pattern.test(sentence)) score += 1;
  if (sentence.length > 40) score += 0.5;
  return score;
}

/** Top-scoring sentences, re-sorted into their original order. */
export function extractSignalSentences(middle: string, limit: number): string[] {
  const candidates: { sentence: string; score: number; index: number }[] = [];
  const paragraphs = splitParagraphs(middle);
  let index = 0;
  for (const paragraph of paragraphs) {
    for (const sentence of splitSentences(paragraph)) {
      const score = signalScore(sentence);
      if (score > 0) candidates.push({ sentence, score, index });
      index += 1;
    }
  }
  candidates.sort((a, b) => (b.score === a.score ? a.index - b.index : b.score - a.score));
  return candidates
    .slice(0, limit)
    .sort((a, b) => a.index - b.index)
    .map((candidate) => candidate.sentence);
}

/** `fromEnd` takes a suffix. Binary search, not a ratio: the estimator is non-linear. */
export function sliceByTokens(text: string, budget: number, fromEnd: boolean, estimator: TokenEstimator): string {
  if (estimator.estimate(text) <= budget) return text;
  let low = 0;
  let high = text.length;
  while (low < high) {
    const mid = Math.floor((low + high + 1) / 2);
    const slice = fromEnd ? text.slice(text.length - mid) : text.slice(0, mid);
    if (estimator.estimate(slice) <= budget) low = mid;
    else high = mid - 1;
  }
  return fromEnd ? text.slice(text.length - low) : text.slice(0, low);
}

export function compressText(text: string, config: ResolvedConfig, estimator: TokenEstimator): CompressionResult | null {
  const total = estimator.estimate(text);
  if (total <= config.targetToolOutputTokens) return null;

  // 45% head + 45% tail, leaving ~10% for the sampled middle and the markers.
  const headBudget = Math.max(32, Math.floor(config.targetToolOutputTokens * 0.45));
  const tailBudget = Math.max(32, Math.floor(config.targetToolOutputTokens * 0.45));
  const headChars = Math.min(config.textHeadChars, text.length);
  const tailChars = Math.min(config.textTailChars, text.length);
  if (headChars + tailChars >= text.length) return null; // nothing left in the middle to omit.

  const headRaw = text.slice(0, headChars);
  const tailRaw = text.slice(text.length - tailChars);
  const head = sliceByTokens(headRaw, headBudget, false, estimator);
  const tail = sliceByTokens(tailRaw, tailBudget, true, estimator);
  const middleStart = head.length;
  const middleEnd = text.length - tail.length;
  const middle = middleEnd > middleStart ? text.slice(middleStart, middleEnd) : '';

  const notes: string[] = [];
  const parts: string[] = [head];
  if (middle.length > 0) {
    const omittedTokens = estimator.estimate(middle);
    const signalLimit = Math.max(3, Math.floor(config.targetToolOutputTokens * 0.08));
    const signals = config.textExtractSignals ? extractSignalSentences(middle, signalLimit) : [];
    if (signals.length > 0) {
      parts.push(
        `\n\n${OMIT_MARKER} ${middle.length} chars (~${omittedTokens} tokens); high-signal lines from the omitted middle follow, in original order:\n` +
          signals.map((sentence) => `- ${sentence}`).join('\n')
      );
      notes.push(`extracted ${signals.length} high-signal sentences from the middle`);
    } else {
      parts.push(`\n\n${OMIT_MARKER} ${middle.length} chars (~${omittedTokens} tokens)`);
    }
    notes.push(`omitted ${middle.length} chars from the middle`);
  }
  parts.push(tail);
  const rendered = parts.join('');

  notes.push(`token estimate ${total} -> ${estimator.estimate(rendered)}`);
  return { text: rendered, strategy: config.textExtractSignals ? 'text:head+tail+signals' : 'text:head+tail', notes };
}
