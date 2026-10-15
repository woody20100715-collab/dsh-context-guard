import type { ResolvedConfig } from '../config.ts';
import type { TokenEstimator } from '../utils/tokenizer.ts';
import type { CompressionResult } from './types.ts';

export type TableFlavor = 'markdown' | 'delimited';

export interface TableDetection {
  readonly flavor: TableFlavor;
  /** markdown: header row + separator; delimited: 1. */
  readonly headerLines: number;
  /** Only set for delimited tables. */
  readonly delimiter?: string;
  readonly columns: number;
  readonly dataRows: number;
}

const MARKDOWN_SEPARATOR = /^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$/;

function countDelimiter(line: string, delimiter: string): number {
  let count = 0;
  let quoted = false;
  for (const char of line) {
    if (char === '"') quoted = !quoted;
    else if (char === delimiter && !quoted) count += 1;
  }
  return count;
}

export function detectTable(text: string): TableDetection | null {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 3) return null;

  const first = lines[0] ?? '';
  const second = lines[1] ?? '';
  if (first.includes('|') && MARKDOWN_SEPARATOR.test(second) && second.includes('-')) {
    const columns = first.split('|').filter((cell) => cell.trim().length > 0).length;
    if (columns >= 2) return { flavor: 'markdown', headerLines: 2, columns, dataRows: lines.length - 2 };
  }

  for (const delimiter of [',', '\t', ';']) {
    const probe = lines.slice(0, 12);
    const counts = probe.map((line) => countDelimiter(line, delimiter));
    const expected = counts[0] ?? 0;
    if (expected < 1) continue;
    const consistent = counts.filter((count) => count === expected).length;
    if (consistent >= Math.max(3, Math.ceil(probe.length * 0.8))) {
      return { flavor: 'delimited', headerLines: 1, delimiter, columns: expected + 1, dataRows: lines.length - 1 };
    }
  }
  return null;
}

// Tail rows beat a random sample: table output (test results, dependency lists, file
// listings) ends with summaries, and sampling would break the implicit row order.
export function compressTable(text: string, config: ResolvedConfig, estimator: TokenEstimator): CompressionResult | null {
  const detection = detectTable(text);
  if (detection === null) return null;

  const lines = text.split(/\r?\n/);
  // indexes of the non-empty lines, in original order.
  const indexes: number[] = [];
  for (let index = 0; index < lines.length; index += 1) {
    if ((lines[index] ?? '').trim().length > 0) indexes.push(index);
  }
  const headerIndexes = indexes.slice(0, detection.headerLines);
  const dataIndexes = indexes.slice(detection.headerLines);
  const head = dataIndexes.slice(0, config.tableHeadRows);
  const tail = dataIndexes.slice(Math.max(config.tableHeadRows, dataIndexes.length - config.tableTailRows));

  const omitted = dataIndexes.length - head.length - tail.length;
  if (omitted <= 0) return null; // the table already fits; nothing to drop.

  const out: string[] = [];
  for (const index of headerIndexes) out.push(lines[index] ?? '');
  for (const index of head) out.push(lines[index] ?? '');
  const marker = detection.flavor === 'markdown'
    ? `| ... (${omitted} rows omitted) ... |`
    : `... (${omitted} rows omitted) ...`;
  out.push(marker);
  for (const index of tail) out.push(lines[index] ?? '');

  const rendered = out.join('\n');
  const notes = [
    `${detection.flavor} table: ${detection.columns} columns / ${detection.dataRows} data rows, ${omitted} middle rows omitted`,
    `token estimate ${estimator.estimate(text)} -> ${estimator.estimate(rendered)}`
  ];
  return { text: rendered, strategy: `table:head+tail(${detection.flavor})`, notes };
}
