import test from 'node:test';
import assert from 'node:assert/strict';

import { resolveConfig } from '../src/config.ts';
import { defaultTokenEstimator } from '../src/utils/tokenizer.ts';
import {
  ContextAssembler,
  computeBudget,
  emptyCompactionState,
  extractOriginalId,
  isCompressedBlocks,
  planAssembly,
  renderCapabilityNotice,
  renderStatusNotice,
  type AssemblyMessage
} from '../src/assembler.ts';

const config = resolveConfig({});
const assembler = new ContextAssembler(config, defaultTokenEstimator);

function message(overrides: Partial<AssemblyMessage> & { tokens: number }): AssemblyMessage {
  return {
    role: 'user',
    priority: 2,
    compressed: false,
    droppable: false,
    label: 'test',
    ...overrides
  };
}

test('budget: a normal window yields a remaining budget', () => {
  const budget = computeBudget({ contextWindow: 100_000, usedTokens: 50_000, systemTokens: 5_000, reservedOutputTokens: 8_000, fallbackContextWindow: 64_000 }, 0.8);
  assert.equal(budget.contextWindow, 100_000);
  assert.equal(budget.degraded, false);
  assert.equal(budget.available, 37_000);
  assert.equal(budget.overBy, 0);
  assert.equal(budget.overThreshold, false);
});

test('budget: detects over-threshold and over-window, degrades when the window is unknown', () => {
  const over = computeBudget({ contextWindow: 10_000, usedTokens: 9_000, systemTokens: 1_000, reservedOutputTokens: 2_000, fallbackContextWindow: 64_000 }, 0.8);
  assert.equal(over.overThreshold, true);
  assert.ok(over.available < 0);
  // available = 10000 - 1000 - 2000 - 9000 = -2000
  assert.equal(over.overBy, 2_000);

  const degraded = computeBudget({ usedTokens: 100, systemTokens: 10, reservedOutputTokens: 10, fallbackContextWindow: 8_000 }, 0.8);
  assert.equal(degraded.degraded, true);
  assert.equal(degraded.contextWindow, 8_000);
});

test('plan: a comfortable budget keeps everything', () => {
  const messages = [message({ tokens: 100 }), message({ tokens: 100 })];
  const plan = planAssembly(messages, 1_000);
  assert.equal(plan.keep.length, 2);
  assert.equal(plan.dropped.length, 0);
  assert.equal(plan.freedTokens, 0);
});

test('plan: only droppable entries are dropped; user messages and tool results never are', () => {
  const messages = [
    message({ tokens: 500, priority: 0, droppable: false, label: 'user' }),
    message({ tokens: 900, priority: 1, droppable: false, label: 'tool-error' }),
    message({ tokens: 400, priority: 2, droppable: false, label: 'tool-ok' }),
    message({ tokens: 600, priority: 3, droppable: true, label: 'plugin-low-priority' }),
    message({ tokens: 300, priority: 2, droppable: true, label: 'plugin-high-priority' })
  ];
  // non-droppable entries total 1800, so a 2100 budget fits exactly one 300-token droppable
  const plan = planAssembly(messages, 2_100);
  assert.deepEqual(plan.dropped.map((entry) => entry.label), ['plugin-low-priority'], 'lowest-priority plugin context goes first');
  assert.ok(plan.keep.some((entry) => entry.label === 'plugin-high-priority'), 'higher-priority droppable entries stay');
  for (const label of ['user', 'tool-error', 'tool-ok']) {
    assert.ok(plan.keep.some((entry) => entry.label === label), `${label} must be kept (dropping tool results breaks tool-call pairing)`);
  }
  assert.equal(plan.stillOverBudget, false);
});

test('plan: reports stillOverBudget when non-droppable entries alone exceed it', () => {
  const messages = [message({ tokens: 5_000, priority: 0 })];
  const plan = planAssembly(messages, 100);
  assert.equal(plan.stillOverBudget, true);
  assert.equal(plan.keep.length, 1);
});

test('notice: the capability text follows the language and says whether CCR is on', () => {
  const zh = renderCapabilityNotice('zh', true);
  assert.ok(zh.includes('context_retrieve'));
  assert.ok(zh.includes('精确值'));
  const en = renderCapabilityNotice('en', true);
  assert.ok(en.startsWith('## Context guard'));
  const noCcr = renderCapabilityNotice('zh', false);
  assert.ok(noCcr.includes('无法恢复'));
});

test('notice: the status notice only appears with content and carries compression stats', () => {
  const budget = computeBudget({ contextWindow: 100_000, usedTokens: 10, systemTokens: 10, reservedOutputTokens: 10, fallbackContextWindow: 64_000 }, 0.8);
  assert.equal(renderStatusNotice(emptyCompactionState(), budget, 'zh'), null);

  const state = { compactions: 2, lastShadowedMessages: 40, lastOriginalTokens: 90_000, lastSummaryTokens: 1_200, retrievableEntries: 3 };
  const notice = renderStatusNotice(state, budget, 'zh');
  assert.ok(notice);
  assert.ok(notice.includes('40'));
  assert.ok(notice.includes('90000'));
  assert.ok(notice.includes('context_retrieve'));
});

test('assembler: capabilityNotice and statusNotice respect noticeEnabled', () => {
  const silent = new ContextAssembler(resolveConfig({ noticeEnabled: false }), defaultTokenEstimator);
  const budget = computeBudget({ contextWindow: 1_000, usedTokens: 2_000, systemTokens: 0, reservedOutputTokens: 0, fallbackContextWindow: 1_000 }, 0.8);
  assert.equal(silent.statusNotice(emptyCompactionState(), budget, 'zh'), null);
  assert.ok(silent.capabilityNotice('zh').length > 0, 'the static capability text ignores noticeEnabled');
});

test('assembler: extracts originalId from the footer and detects compressed blocks', () => {
  const text = 'body\n\n[context-guard] compressed json output ... id="ccr_0123456789abcdef" ...';
  assert.equal(extractOriginalId(text), 'ccr_0123456789abcdef');
  assert.equal(isCompressedBlocks([{ type: 'text', text }]), true);
  assert.equal(isCompressedBlocks([{ type: 'text', text: 'plain' }]), false);
  assert.equal(extractOriginalId('no id here'), undefined);
});

test('assembler: the saving decision matches the configured threshold', () => {
  assert.equal(assembler.isWorthNotice(1_000, 400), true);
  assert.equal(assembler.isWorthNotice(1_000, 950), false);
  assert.equal(assembler.isWorthNotice(0, 0), false);
});
