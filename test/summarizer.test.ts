import test from 'node:test';
import assert from 'node:assert/strict';

import { resolveConfig } from '../src/config.ts';
import { GuardLogger, MemorySink } from '../src/utils/logger.ts';
import { defaultTokenEstimator } from '../src/utils/tokenizer.ts';
import {
  buildSummaryInstruction,
  detectLanguage,
  emptySummary,
  extractPreciseValues,
  heuristicExtract,
  HighFidelitySummarizer,
  mergeSummaries,
  parseSummaryJson,
  renderStructuredSummary,
  type CompletionRequest,
  type SimpleMessage,
  type SummarizationInput
} from '../src/summarizer.ts';

const sink = new MemorySink();
const logger = new GuardLogger('debug', sink);

const MESSAGES: SimpleMessage[] = [
  { role: 'user', text: '请把服务部署到 C:\\work\\demo-app，端口 8080，必须使用 pnpm run build:web 构建。' },
  { role: 'assistant', text: '好的，我先检查目录结构。' },
  { role: 'tool', text: 'build ok in 42s', toolName: 'pwsh' },
  { role: 'tool', text: 'connection refused', toolName: 'pwsh', isError: true },
  { role: 'user', text: '记得不要改动 package.json 的 version 字段。' }
];

const INPUT: SummarizationInput = { messages: MESSAGES, raw: [] };

test('language: auto decides by CJK ratio, explicit config wins', () => {
  assert.equal(detectLanguage(['你好，请帮我部署服务'], 'auto'), 'zh');
  assert.equal(detectLanguage(['please deploy the service now'], 'auto'), 'en');
  // technical Chinese is full of English identifiers, so the threshold must be lenient
  assert.equal(detectLanguage(['把 buildContext() 改成 async 的，pnpm run test 要过'], 'auto'), 'zh');
  assert.equal(detectLanguage(['你好'], 'en'), 'en');
  assert.equal(detectLanguage(['hello'], 'zh'), 'zh');
  assert.equal(detectLanguage([], 'auto'), 'en');
});

test('instruction: forces JSON output and demands literal values', () => {
  const zh = buildSummaryInstruction('zh', true);
  assert.ok(zh.includes('只输出一个 JSON 对象'));
  assert.ok(zh.includes('goal'));
  assert.ok(zh.includes('precise'));
  const en = buildSummaryInstruction('en', true);
  assert.ok(en.includes('ONE JSON object'));
  assert.ok(en.includes('verbatim'));
});

test('parse: tolerates code fences and surrounding prose, rejects non-JSON', () => {
  const parsed = parseSummaryJson('\u0060\u0060\u0060json\n{"goal":["部署服务"],"precise":["8080"]}\n\u0060\u0060\u0060');
  assert.deepEqual(parsed?.goal, ['部署服务']);
  assert.deepEqual(parsed?.precise, ['8080']);
  const embedded = parseSummaryJson('Sure! Here it is:\n{"actions":["pwsh -> ok"]}\nHope that helps.');
  assert.deepEqual(embedded?.actions, ['pwsh -> ok']);
  assert.equal(parseSummaryJson('no json at all'), null);
  assert.equal(parseSummaryJson('{"goal":[]}'), null, 'an all-empty summary is invalid');
});

test('precise: the deterministic pass extracts numbers, paths, commands, constraints', () => {
  const values = extractPreciseValues(MESSAGES.map((message) => message.text).join('\n'));
  assert.ok(values.numbers.includes('8080'));
  assert.ok(values.paths.some((path) => path.startsWith('C:\\work\\demo-app')));
  assert.ok(values.commands.some((command) => command.includes('pnpm run build:web')));
  assert.ok(values.constraints.some((constraint) => constraint.includes('必须')));
});

test('heuristic: rebuilds goal, tool status and constraints without a model', () => {
  const summary = heuristicExtract(MESSAGES, 'zh');
  assert.ok(summary.goal[0]?.includes('部署'));
  assert.ok(summary.actions.includes('pwsh -> 成功'));
  assert.ok(summary.actions.includes('pwsh -> 失败'));
  assert.ok(summary.data.includes('8080'));
  assert.ok(summary.precise.some((item) => item.includes('build:web')));
  assert.ok(summary.constraints.some((item) => item.includes('必须')));
});

test('merge/render: merges, dedupes and renders headings by language', () => {
  const merged = mergeSummaries(
    { ...emptySummary(), goal: ['A'] },
    { ...emptySummary(), goal: ['A', 'B'], todos: ['C'] }
  );
  assert.deepEqual(merged.goal, ['A', 'B']);
  assert.deepEqual(merged.todos, ['C']);
  const zh = renderStructuredSummary(merged, 'zh');
  assert.ok(zh.startsWith('【已压缩历史摘要】'));
  assert.ok(zh.includes('用户目标'));
  const en = renderStructuredSummary(merged, 'en');
  assert.ok(en.includes('[Compacted history summary]'));
  assert.ok(en.includes('Open tasks'));
});

test('summarize: valid model JSON uses the llm strategy and merges the deterministic ledger', async () => {
  let calls = 0;
  const completion = async (request: CompletionRequest) => {
    calls += 1;
    assert.equal(request.maxTokens, 2048);
    assert.ok(request.instruction.includes('JSON'));
    return {
      text: JSON.stringify({ goal: ['部署服务到生产环境'], actions: ['检查目录'], data: [], precise: [], constraints: [], todos: ['重启服务'], openQuestions: [] }),
      provider: 'test-provider',
      model: 'test-model',
      rawOutput: [{ type: 'text', text: 'raw' }]
    };
  };
  const summarizer = new HighFidelitySummarizer(resolveConfig({}), logger, defaultTokenEstimator, completion);
  const outcome = await summarizer.summarize(INPUT);
  assert.equal(outcome.strategy, 'llm');
  assert.equal(outcome.provider, 'test-provider');
  assert.equal(outcome.language, 'zh');
  const text = outcome.blocks.map((block) => block.text).join('\n');
  assert.ok(text.includes('<compacted-summary>'));
  assert.ok(text.includes('【已压缩历史摘要】'));
  assert.ok(text.includes('部署服务到生产环境'), 'structured model output must be present');
  assert.ok(text.includes('8080'), 'deterministic numbers must be merged in');
  assert.ok(text.includes('build:web'), 'deterministic commands must be merged in');
  assert.ok(text.includes('"__compacted":true'), 'metadata must be parseable by consumers');
  assert.equal(outcome.originalMessageCount, MESSAGES.length);
  assert.equal(calls, 1);

  // the second call with the same input hits the cache, no model call
  const again = await summarizer.summarize(INPUT);
  assert.equal(again.strategy, 'llm');
  assert.equal(calls, 1, 'identical history blocks must not call the model twice');
  assert.equal(summarizer.cacheSize, 1);
});

test('summarize: model failure degrades to the deterministic summary, precise values survive', async () => {
  const failing = async () => {
    throw new Error('provider exploded');
  };
  const summarizer = new HighFidelitySummarizer(resolveConfig({}), logger, defaultTokenEstimator, failing);
  const outcome = await summarizer.summarize(INPUT);
  assert.equal(outcome.strategy, 'heuristic');
  assert.equal(outcome.provider, '');
  const text = outcome.blocks.map((block) => block.text).join('\n');
  assert.ok(text.includes('8080'));
  assert.ok(text.includes('pnpm run build:web'));
  assert.ok(text.includes('pwsh -> 成功'));
  assert.ok(text.includes('pwsh -> 失败'));
  assert.ok(outcome.notes.some((note) => note.includes('deterministic channel')));
});

test('summarize: non-JSON model output falls back to llm+heuristic without losing the ledger', async () => {
  const chatty = async () => ({ text: 'The conversation was about deploying a service.', provider: 'p', model: 'm' });
  const summarizer = new HighFidelitySummarizer(resolveConfig({}), logger, defaultTokenEstimator, chatty);
  const outcome = await summarizer.summarize(INPUT);
  assert.equal(outcome.strategy, 'llm+heuristic');
  const text = outcome.blocks.map((block) => block.text).join('\n');
  assert.ok(text.includes('8080'));
  assert.ok(text.includes('deploying a service'), 'raw model text is kept as an appendix');
});

test('summarize: with no LLM channel it is purely deterministic and never fails', async () => {
  const summarizer = new HighFidelitySummarizer(resolveConfig({}), logger, defaultTokenEstimator);
  const outcome = await summarizer.summarize(INPUT);
  assert.equal(outcome.strategy, 'heuristic');
  assert.ok(outcome.notes.some((note) => note.includes('no LLM channel injected')));
});

test('summarize: summaryLanguage=zh renders Chinese even for English input', async () => {
  const summarizer = new HighFidelitySummarizer(resolveConfig({ summaryLanguage: 'zh' }), logger, defaultTokenEstimator);
  const outcome = await summarizer.summarize({ messages: [{ role: 'user', text: 'deploy now' }] });
  assert.equal(outcome.language, 'zh');
  assert.ok(outcome.blocks[0]?.text.includes('自动生成的检查点摘要'));
});
