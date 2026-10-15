import test from 'node:test';
import assert from 'node:assert/strict';

import { resolveConfig } from '../src/config.ts';
import { defaultTokenEstimator } from '../src/utils/tokenizer.ts';
import {
  appendRetrievalFooter,
  compressContentBlocks,
  compressJson,
  compressLog,
  compressTable,
  compressText,
  detectCategory,
  OMITTED_MARKER,
  tryParseJson
} from '../src/compressors/index.ts';
import { detectTable } from '../src/compressors/table.ts';
import { parseLogLines, classifyLogLine } from '../src/compressors/log.ts';
import { extractSignalSentences, signalScore, sliceByTokens } from '../src/compressors/text.ts';

const estimator = defaultTokenEstimator;
const config = resolveConfig({ targetToolOutputTokens: 300, maxToolOutputTokens: 400, minToolOutputTokens: 50 });

test('json: samples large arrays, keeps full key names, reports the omitted count', () => {
  const rows = Array.from({ length: 300 }, (_value, index) => ({
    id: index,
    name: `service-${index}`,
    status: 'ok',
    endpoint: `https://api.example.com/v1/items/${index}`
  }));
  const text = JSON.stringify(rows, null, 2);
  const result = compressJson(text, config, estimator);
  assert.ok(result, 'should be detected as JSON and compressed');

  const before = estimator.estimate(text);
  const after = estimator.estimate(result.text);
  assert.ok(after < before / 2, `should cut tokens substantially: ${before} -> ${after}`);

  const parsed = JSON.parse(result.text) as unknown[];
  // how many entries survive depends on which tightening layer fits, but
  // kept + omitted must always equal the total or the model cannot tell 20 from 2000
  const markerEntry = parsed.find(
    (entry): entry is Record<string, number> => typeof entry === 'object' && entry !== null && OMITTED_MARKER in entry
  );
  assert.ok(markerEntry, 'sampling must leave an omission marker');
  const omitted = markerEntry[OMITTED_MARKER] ?? 0;
  assert.ok(omitted > 0);
  assert.equal(parsed.length - 1 + omitted, 300, 'kept + omitted = original total');
  assert.ok(parsed.length < 300);
  // full key names are kept; abbreviated keys lose semantics
  assert.deepEqual(Object.keys(parsed[0] as object), ['id', 'name', 'status', 'endpoint']);
  assert.equal((parsed[0] as { endpoint: string }).endpoint, 'https://api.example.com/v1/items/0');
  assert.equal((parsed[parsed.length - 1] as { id: number }).id, 299);
});

test('json: truncates long strings to head + tail and keeps the removed length', () => {
  const long = 'A'.repeat(3000) + 'TAIL_MARKER';
  const text = JSON.stringify({ path: long, keep: 42 });
  const result = compressJson(text, config, estimator);
  assert.ok(result);
  const parsed = JSON.parse(result.text) as { path: string; keep: number };
  assert.equal(parsed.keep, 42, 'short values are untouched');
  assert.ok(parsed.path.includes('...[truncated]'), 'must carry a truncation marker');
  assert.ok(parsed.path.includes('TAIL_MARKER'), 'tail is kept; it usually carries units or file names');
  assert.ok(parsed.path.length < 600);
});

test('json: non-JSON text returns null and falls through to other compressors', () => {
  assert.equal(compressJson('plain text, not json', config, estimator), null);
  assert.equal(tryParseJson('not json'), undefined);
  assert.deepEqual(tryParseJson('{"a":1}'), { a: 1 });
});

test('log: key lines, head/tail and duplicate folding all apply', () => {
  const lines: string[] = ['$ npm test', 'node v24.18.0'];
  for (let index = 0; index < 120; index += 1) lines.push(`2026-10-15T10:00:00.000Z INFO polling queue (attempt ${index})`);
  for (let index = 0; index < 30; index += 1) lines.push('2026-10-15T10:00:01.000Z INFO heartbeat: unchanged');
  lines.push('2026-10-15T10:00:02.000Z ERROR Failed to connect to 10.0.0.7:5432');
  lines.push('    at Socket.connect (/app/src/db.ts:42:11)');
  lines.push('    at process.processTicksAndRejections (node:internal/process/task_queues:95:5)');
  lines.push('2026-10-15T10:00:03.000Z WARN retrying in 500ms');
  lines.push('Tests: 3 failed, 120 passed, 123 total');

  const text = lines.join('\n');
  const parsed = parseLogLines(text);
  assert.equal(parsed.length, lines.length);
  assert.equal(classifyLogLine('2026-01-01 ERROR boom'), 'error');
  assert.equal(classifyLogLine('2026-01-01 WARN careful'), 'warn');
  assert.equal(classifyLogLine('just some output'), 'other');

  const result = compressLog(text, config, estimator);
  assert.ok(result);
  assert.ok(result.text.includes('ERROR Failed to connect to 10.0.0.7:5432'), 'error lines stay whole, timestamp and address included');
  assert.ok(result.text.includes('at Socket.connect (/app/src/db.ts:42:11)'), 'stack frames must follow the error line');
  assert.ok(result.text.includes('WARN retrying in 500ms'));
  assert.ok(result.text.includes('Lines') === false, 'header is the English summary line, not a stray word');
  assert.match(result.text, /^\[context-guard\/log\] \d+ lines -> \d+ shown/);
  assert.ok(result.text.includes('Tests: 3 failed, 120 passed'), 'trailing summary must be kept');
  assert.ok(result.text.includes('$ npm test'), 'leading command must be kept');
  assert.ok(/heartbeat: unchanged\s+\(x30\)/.test(result.text), 'repeated lines fold with a count');
  assert.ok(estimator.estimate(result.text) < estimator.estimate(text) / 2);
});

test('log: too few lines disables the log strategy', () => {
  assert.equal(compressLog('one\ntwo\nthree', config, estimator), null);
});

test('table: markdown keeps the header, head/tail rows and an omission marker', () => {
  const rows = Array.from({ length: 80 }, (_value, index) => `| ${index} | file-${index}.ts | ${index * 10} |`);
  const text = ['| id | name | size |', '| --- | --- | --- |', ...rows].join('\n');
  const detection = detectTable(text);
  assert.ok(detection);
  assert.equal(detection.flavor, 'markdown');
  assert.equal(detection.columns, 3);
  assert.equal(detection.dataRows, 80);

  const result = compressTable(text, config, estimator);
  assert.ok(result);
  assert.ok(result.text.startsWith('| id | name | size |'), 'header must stay on the first line');
  assert.ok(result.text.includes('| --- | --- | --- |'), 'separator row must stay or it is no longer a valid table');
  assert.ok(result.text.includes('| file-0.ts |'));
  assert.ok(result.text.includes('| file-79.ts |'), 'tail rows must be kept');
  assert.ok(result.text.includes('rows omitted'));
  assert.ok(estimator.estimate(result.text) < estimator.estimate(text) / 2);
});

test('table: CSV is detected and compressed too', () => {
  const rows = Array.from({ length: 60 }, (_value, index) => `${index},alpha,${index * 3}`);
  const text = ['id,name,value', ...rows].join('\n');
  const detection = detectTable(text);
  assert.equal(detection?.flavor, 'delimited');
  const result = compressTable(text, config, estimator);
  assert.ok(result);
  assert.ok(result.text.startsWith('id,name,value'));
  assert.ok(result.text.includes('rows omitted'));
});

test('text: keeps head/tail and extracts high-signal sentences from the middle', () => {
  const head = 'Project overview. '.repeat(80);
  const middle = [
    'Some prose that carries no signal at all.',
    'Deploy with pnpm run build:web before publishing.',
    'More filler content here to push the budget.',
    'The timeout is 30 seconds and must not be exceeded.',
    'C:/projects/demo-app/package.json'
  ].join('\n\n');
  const tail = 'Conclusion: all checks passed. '.repeat(60);
  const text = `${head}\n\n${middle}\n\n${tail}`;

  assert.ok(signalScore('run pnpm run build:web now') > 0);
  assert.ok(signalScore('a plain sentence') === 0);
  const signals = extractSignalSentences(middle, 5);
  assert.ok(signals.some((sentence) => sentence.includes('build:web')));
  assert.ok(signals.some((sentence) => sentence.includes('must not be exceeded')));

  const result = compressText(text, config, estimator);
  assert.ok(result);
  assert.ok(result.text.includes('Project overview.'));
  assert.ok(result.text.includes('Conclusion: all checks passed.'));
  assert.ok(result.text.includes('omitted'));
  assert.ok(result.text.includes('build:web'), 'high-signal middle sentence must be kept');
  assert.ok(estimator.estimate(result.text) < estimator.estimate(text));
});

test('text: sliceByTokens respects the token budget and keeps head/tail ordering intact', () => {
  const text = '中文内容测试。'.repeat(500);
  const slice = sliceByTokens(text, 50, false, estimator);
  assert.ok(estimator.estimate(slice) <= 50);
  assert.ok(slice.startsWith('中文内容测试。'));
  const tail = sliceByTokens(text, 50, true, estimator);
  assert.ok(estimator.estimate(tail) <= 50);
  assert.ok(tail.endsWith('中文内容测试。'));
});

test('detectCategory: detection order is json > table > log > text', () => {
  assert.equal(detectCategory('{"a":1}'), 'json');
  assert.equal(detectCategory('[1,2,3]'), 'json');
  assert.equal(detectCategory('| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |'), 'table');
  const log = Array.from({ length: 12 }, (_value, index) => `2026-01-01 INFO line ${index}`).join('\n');
  assert.equal(detectCategory(log), 'log');
  assert.equal(detectCategory('just a paragraph of prose with no structure'), 'text');
  // tool name is only a fallback: pwsh returning JSON is still detected by content
  assert.equal(detectCategory('{"a":1}', 'pwsh'), 'json');
});

test('compressContentBlocks: below the threshold nothing moves, above it compresses with metadata', () => {
  const small = [{ type: 'text', text: 'short output' }];
  assert.equal(compressContentBlocks(small, config, estimator), null);

  const big = [{ type: 'text', text: JSON.stringify(Array.from({ length: 200 }, (_v, index) => ({ index, value: 'x'.repeat(20) }))) }];
  const outcome = compressContentBlocks(big, config, estimator, { toolName: 'big_tool' });
  assert.ok(outcome);
  assert.equal(outcome.meta.compressed, true);
  assert.equal(outcome.meta.category, 'json');
  assert.equal(outcome.meta.rewrittenBlocks, 1);
  assert.ok(outcome.meta.compressedTokens < outcome.meta.originalTokens);
  assert.equal(outcome.blocks.length, 1);
});

test('compressContentBlocks: non-text blocks such as images pass through in order', () => {
  const blocks = [
    { type: 'text', text: JSON.stringify(Array.from({ length: 200 }, (_v, index) => ({ index, pad: 'y'.repeat(20) }))) },
    { type: 'image', text: undefined }
  ];
  const outcome = compressContentBlocks(blocks, config, estimator);
  assert.ok(outcome);
  assert.equal(outcome.blocks.length, 2);
  assert.equal(outcome.blocks[1]?.type, 'image');
});

test('compressContentBlocks: disabled means pass-through', () => {
  const disabled = resolveConfig({ enableToolCompression: false });
  const big = [{ type: 'text', text: JSON.stringify(Array.from({ length: 500 }, (_v, index) => ({ index }))) }];
  assert.equal(compressContentBlocks(big, disabled, estimator), null);
});

test('appendRetrievalFooter: CCR enabled gives an id, disabled says not recoverable', () => {
  const blocks = [{ type: 'text', text: 'compressed body' }];
  const withId = appendRetrievalFooter(
    blocks,
    { compressed: true, originalId: 'ccr_0123456789abcdef', originalTokens: 1000, compressedTokens: 100, category: 'json', strategy: 'json:sample', notes: [], rewrittenBlocks: 1 },
    true
  );
  const text = withId[0]?.text ?? '';
  assert.ok(text.startsWith('compressed body'));
  assert.ok(text.includes('context_retrieve'), 'must tell the model how to retrieve');
  assert.ok(text.includes('ccr_0123456789abcdef'));

  const withoutId = appendRetrievalFooter(
    blocks,
    { compressed: true, originalTokens: 1000, compressedTokens: 100, category: 'json', strategy: 'json:sample', notes: [], rewrittenBlocks: 1 },
    false
  );
  assert.ok((withoutId[0]?.text ?? '').includes('not recoverable'));
});
