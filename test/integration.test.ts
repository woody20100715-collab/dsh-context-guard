import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

let plugin: typeof import('../src/index.ts') | undefined;
let unavailable: string | undefined;
try {
  plugin = await import('../src/index.ts');
} catch (error) {
  unavailable = `DSH runtime dependencies unavailable: ${String(error)}`;
}

const skip = unavailable === undefined ? false : unavailable;

// root ctx with a fake llm / tokenMeter / sessions
async function boot(
  options: { dir?: string; meterTokensPerNode?: number; estimateMessage?: number; config?: Record<string, unknown> } = {}
) {
  const { Context } = await import('@deepseek-ai/cordis');
  const { default: SystemPrompt } = await import('@deepseek-ai/dsh-system-prompt');
  const { default: ToolRuntime } = await import('@deepseek-ai/dsh-tools');
  const dir = options.dir ?? (await mkdtemp(join(tmpdir(), 'cg-int-')));
  const ctx = new Context();
  const state = { estimateMessage: options.estimateMessage ?? 50 };

  ctx.provide('llm', {
    // the stream is made to fail on purpose: the summarizer must degrade to the deterministic pass, not throw
    stream: async function* stream() {
      throw new Error('no llm available in this test');
    },
    resolveModelInfo: async () => ({ provider: 'test', id: 'test-model', name: 'Test model', context: { contextWindow: 10_000 } })
  } as never);
  ctx.provide('tokenMeter', {
    measure: (session: { surface: { nodes: readonly unknown[] } }) => {
      const perNode = options.meterTokensPerNode ?? 1_000;
      return {
        logRevision: 0,
        baseline: { kind: 'none', tokens: 0 },
        surfaceDeltaTokens: 0,
        totalTokens: session.surface.nodes.length * perNode,
        surfaceTokens: session.surface.nodes.length * perNode,
        nodes: session.surface.nodes.map((seq) => ({ seq, tokens: perNode, heuristicTokens: perNode }))
      };
    },
    estimateMessage: () => state.estimateMessage
  } as never);
  ctx.provide('sessions', { flush: async () => true } as never);

  await ctx.plugin(SystemPrompt as never, {} as never);
  await ctx.plugin(ToolRuntime as never, {} as never);
  await ctx.plugin(plugin!.default as never, {
    ccrStorePath: dir,
    ccrPersist: true,
    enableCCR: true,
    maxToolOutputTokens: 200,
    minToolOutputTokens: 50,
    targetToolOutputTokens: 120,
    summaryCacheSize: 0,
    compactionRetries: 0,
    ...(options.config ?? {})
  } as never);

  return { ctx, dir, state, ownsDir: options.dir === undefined };
}

test('integration: plugin exports match the DSH contract (name + default Service class)', { skip }, () => {
  assert.equal(plugin?.name, 'dsh-context-guard');
  assert.equal(typeof plugin?.default, 'function');
  assert.deepEqual((plugin?.default as { inject?: string[] }).inject, ['llm', 'tokenMeter', 'sessions']);
});

test('integration: mounting provides ctx.compaction; tool output is compressed and retrievable via context_retrieve', { skip }, async () => {
  const { ctx, dir, ownsDir } = await boot();
  try {
    const { ToolCallId } = await import('@deepseek-ai/dsh-llm');
    const { defineTool } = await import('@deepseek-ai/dsh-tools');

    assert.ok(ctx.get('compaction'), 'ctx.compaction must come from this plugin');

    const original = JSON.stringify(
      Array.from({ length: 200 }, (_value, index) => ({ index, name: `item-${index}`, payload: 'p'.repeat(30) }))
    );
    ctx.tools.register(
      defineTool({
        name: 'big_tool',
        description: 'Returns a large JSON payload for compression testing.',
        parameters: {},
        output: {
          schema: { type: 'string' },
          render: (_args, value) => [{ type: 'text', text: value }]
        },
        execute: async () => original
      })
    );

    const signal = new AbortController().signal;
    const result = await ctx.tools.execute({ callId: ToolCallId('call-big'), name: 'big_tool', arguments: {}, signal });
    assert.equal(result.isError, false);
    const compressed = plugin!.textOf(result.content);
    assert.ok(compressed.length < original.length / 2, `compression should shorten the output substantially: ${original.length} -> ${compressed.length}`);
    assert.ok(compressed.includes('[context-guard] compressed'), 'must carry the compression footer');

    const id = /id="(ccr_[0-9a-f]{16})"/.exec(compressed)?.[1];
    assert.ok(id, 'CCR enabled must yield a retrieval id');

    const restored = await ctx.tools.execute({
      callId: ToolCallId('call-retrieve'),
      name: plugin!.RETRIEVE_TOOL_NAME,
      arguments: { id },
      signal
    });
    assert.equal(restored.isError, false);
    const restoredText = plugin!.textOf(restored.content);
    assert.ok(restoredText.startsWith(original), 'retrieved content must be the full original');
    assert.ok(restoredText.includes('item-199'), 'the tail element must not be lost');

    const missingResult = await ctx.tools.execute({
      callId: ToolCallId('call-missing'),
      name: plugin!.RETRIEVE_TOOL_NAME,
      arguments: { id: 'ccr_ffffffffffffffff' },
      signal
    });
    assert.equal(missingResult.isError, false);
    assert.ok(plugin!.textOf(missingResult.content).includes('No original content'));
  } finally {
    if (ownsDir) await rm(dir, { recursive: true, force: true });
  }
});

test('integration: the compaction transaction follows the durable protocol (start -> summary -> replacement -> end)', { skip }, async () => {
  const { ctx, dir, ownsDir } = await boot();
  try {
    const { Session, SessionId } = await import('@deepseek-ai/dsh-session');
    const { createUserMessage } = await import('@deepseek-ai/dsh-llm');
    const engine = ctx.get('compaction') as import('@deepseek-ai/dsh-compaction').CompactionEngine;

    const session = Session.create(SessionId('session-integration'));
    session.append('turn/start', { turn: 1 });
    const texts = [
      '请把服务部署到 C:\\work\\demo-app，端口 8080，必须使用 pnpm run build:web 构建。',
      '第一条：先检查依赖。',
      '第二条：注意磁盘剩余 3.5 GB。',
      '第三条：日志目录是 /var/log/app。',
      '第四条：超时时间 30 秒。',
      '最后：记得不要修改 version 字段。'
    ];
    for (const text of texts) {
      session.append('user/message', createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } }), {
        surfaceOp: 'append'
      });
    }
    // session.surface.nodes is a live view; a replacement mutates it in place, so
    // snapshot it before using it across an await
    const before = [...session.surface.nodes];
    assert.equal(before.length, texts.length);

    const agent = { session, options: { provider: 'test', model: 'test-model' } };
    const result = await engine.compactRegion(before[0]!, before[before.length - 1]!, agent as never);

    assert.equal(result.shadowedSeqs.length, texts.length);

    const events = session.ownEvents();
    // provider/model are not on CompactionResult; they live on the compaction/summary event
    // the LLM channel fails here on purpose, so provider/model must be empty (deterministic)
    const summaryData = (events.find((event) => event.type === 'compaction/summary') as { data: { provider: string; model: string } }).data;
    assert.equal(summaryData.provider, '', 'a failing LLM channel must fall back to the deterministic summary');
    assert.equal(summaryData.model, '');
    const types = events.map((event) => event.type);
    const startIdx = types.indexOf('compaction/start');
    const summaryIdx = types.indexOf('compaction/summary');
    const endIdx = types.indexOf('compaction/end');
    assert.ok(startIdx >= 0 && summaryIdx > startIdx && endIdx > summaryIdx, `event order must be start -> summary -> end: ${types.join(',')}`);
    assert.equal(types[summaryIdx + 1], 'user/message', 'the replacement user/message must immediately follow summary');

    const replacement = events[summaryIdx + 1] as { surfaceOp?: unknown; sourceEventSeqs?: unknown[] };
    // the replace boundary fields are named startSeq/endSeq as of 0.2.0
    assert.deepEqual(replacement.surfaceOp, { op: 'replace', startSeq: before[0], endSeq: before[before.length - 1] });
    assert.deepEqual(replacement.sourceEventSeqs?.slice(2), [...before]);

    assert.equal(session.surface.nodes.length, 1, 'the whole surface is replaced by one summary');
    const derived = session.deriveMessages().map((message) => plugin!.textOf(message.content)).join('\n');
    assert.ok(derived.includes('8080'), 'precise value: the port must survive');
    assert.ok(derived.includes('C:\\work\\demo-app'), 'precise value: the path must survive');
    assert.ok(derived.includes('pnpm run build:web'), 'precise value: the command must survive');
    assert.ok(derived.includes('3.5 GB'), 'precise value: the number must survive');
    assert.ok(derived.includes('不要修改 version'), 'precise value: the user constraint must survive');
    assert.ok(derived.includes('__compacted'), 'the summary must carry parseable metadata');
  } finally {
    if (ownsDir) await rm(dir, { recursive: true, force: true });
  }
});

test('integration: a summary that is not smaller must fail with exactly one compaction/end carrying error', { skip }, async () => {
  const { ctx, dir, ownsDir } = await boot({ estimateMessage: 100_000 });
  try {
    const { Session, SessionId } = await import('@deepseek-ai/dsh-session');
    const { createUserMessage } = await import('@deepseek-ai/dsh-llm');
    const engine = ctx.get('compaction') as import('@deepseek-ai/dsh-compaction').CompactionEngine;

    const session = Session.create(SessionId('session-shrink-gate'));
    session.append('turn/start', { turn: 1 });
    for (const text of ['一', '二', '三']) {
      session.append('user/message', createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } }), {
        surfaceOp: 'append'
      });
    }
    const nodes = [...session.surface.nodes];
    const agent = { session, options: { provider: 'test', model: 'test-model' } };

    await assert.rejects(
      () => engine.compactRegion(nodes[0]!, nodes[nodes.length - 1]!, agent as never),
      /summary is not smaller than the shadowed content/
    );

    const ends = session.ownEvents().filter((event) => event.type === 'compaction/end');
    assert.equal(ends.length, 1, 'the failure path must emit compaction/end exactly once too');
    assert.ok(typeof (ends[0] as { data: { error?: string } }).data.error === 'string');
    assert.equal(session.surface.nodes.length, 3, 'surface must be untouched on failure');
  } finally {
    if (ownsDir) await rm(dir, { recursive: true, force: true });
  }
});

test('integration: agent/pre-step injects a status notice after compaction, once per turn', { skip }, async () => {
  const { ctx, dir, ownsDir } = await boot();
  try {
    const { Session, SessionId } = await import('@deepseek-ai/dsh-session');
    const { createUserMessage } = await import('@deepseek-ai/dsh-llm');
    const engine = ctx.get('compaction') as import('@deepseek-ai/dsh-compaction').CompactionEngine;

    const session = Session.create(SessionId('session-pre-step'));
    session.append('turn/start', { turn: 1 });
    for (const text of ['目标：部署服务', '细节一', '细节二', '细节三']) {
      session.append('user/message', createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } }), {
        surfaceOp: 'append'
      });
    }
    const nodes = [...session.surface.nodes];
    const agent = { session, options: { provider: 'test', model: 'test-model' }, status: 'running' };
    await engine.compactRegion(nodes[0]!, nodes[nodes.length - 1]!, agent as never);

    const waterfall = (ctx as unknown as { waterfall: (name: string, ...rest: unknown[]) => Promise<unknown> }).waterfall.bind(ctx);
    const payload = { agent, messages: [], turn: 7, step: 1, signal: new AbortController().signal };
    const run = () => waterfall('agent/pre-step', payload, async () => ({ kind: 'enter', messages: [] }));

    const first = (await run()) as { messages: { content: { type: string; text?: string }[] }[] };
    assert.equal(first.messages.length, 1, 'one status notice should be injected after compaction');
    const notice = plugin!.textOf(first.messages[0]!.content);
    assert.ok(notice.includes('[context-guard]'));
    assert.ok(notice.includes('context_retrieve') || notice.includes('已压缩'));

    const second = (await run()) as { messages: { content: { type: string; text?: string }[] }[] };
    assert.equal(second.messages.length, 0, 'no repeat injection within one turn: it would break request prefix caching');
  } finally {
    if (ownsDir) await rm(dir, { recursive: true, force: true });
  }
});
