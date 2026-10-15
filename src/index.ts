// DSH has no `ctx.compaction.registerBackend`: `ctx.compaction` is the
// CompactionEngine singleton, so this plugin subclasses it (same shape as the
// built-in dsh-compaction-basic) and registers the rest from the constructor.

import type { Context } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
import {
  CompactionEngine,
  CompactionId,
  ManualCompactionError,
  compactCheckpointSource,
  toolPairingBalancedAfter,
  toolPairingBalancedBefore
} from '@deepseek-ai/dsh-compaction';
import type { CompactionResult, CompactionTrigger } from '@deepseek-ai/dsh-compaction';
import { BlockAssembler, CONTEXT_WINDOW_EXCEEDED_CODE, createUserMessage } from '@deepseek-ai/dsh-llm';
import type { ContentBlock, GenerateOptions, Message, StreamChunk, TextBlock, ToolSchema, UserMessage } from '@deepseek-ai/dsh-llm';
import { SessionSeq } from '@deepseek-ai/dsh-session';
import type { Session } from '@deepseek-ai/dsh-session';
import type { Agent } from '@deepseek-ai/dsh-agent';
import type { TokenMeter } from '@deepseek-ai/dsh-token-meter';
import { defineTool } from '@deepseek-ai/dsh-tools';
import type { PostToolDecision, ToolExecution, ToolExecutionResult } from '@deepseek-ai/dsh-tools';
import { randomUUID } from 'node:crypto';

import { resolveConfig, type ContextGuardConfig, type ResolvedConfig } from './config.ts';
import { CCRStore, sliceForRetrieval } from './ccr.ts';
import {
  appendRetrievalFooter,
  compressContentBlocks,
  type GuardContentBlock
} from './compressors/index.ts';
import {
  ContextAssembler,
  emptyCompactionState,
  isCompressedBlocks,
  type AssemblyMessage,
  type CompactionState,
  type MessagePriority
} from './assembler.ts';
import {
  HighFidelitySummarizer,
  detectLanguage,
  type CompletionRequest,
  type CompletionResult,
  type SimpleMessage,
  type SummarizationInput,
  type SummaryLanguage,
  type TextCompletion
} from './summarizer.ts';
import { createCordisSink, GuardLogger } from './utils/logger.ts';
import { defaultTokenEstimator, type TokenEstimator } from './utils/tokenizer.ts';

// Message source kind contributed by this plugin. DSH 0.2 dropped the generic
// `plugin` kind for a per-producer union, so compression notices keep an
// auditable origin in the durable log.
declare module '@deepseek-ai/dsh-llm' {
  interface MessageSourceMap {
    'context-guard': {
      kind: 'context-guard';
    };
  }
}

/** Plugin name; also the prompt-section name and message source kind. */
export const name = 'dsh-context-guard';

export const RETRIEVE_TOOL_NAME = 'context_retrieve';

export const COMPRESSION_MARKER = '[context-guard] compressed';

/** Prompt section name and its stable order (after the tool docs, before the generative-SDK sections). */
const PROMPT_SECTION = 'context-guard:context-policy';
const PROMPT_SECTION_ORDER = 3050;

/** Same type as compactNow's third argument; `Parameters` avoids depending on dsh-commands. */
type SourceCommandId = Parameters<CompactionEngine['compactNow']>[2];

/** Only the fields we use; avoids depending on dsh-token-meter's export surface. */
interface MeasurementLike {
  readonly totalTokens: number;
  readonly nodes: readonly { readonly seq: SessionSeq; readonly tokens: number; readonly heuristicTokens: number }[];
}

interface TransactionOptions {
  /** `current-turn` = automatic compaction inside the open turn; null = standalone manual transaction. */
  readonly owner: 'current-turn' | null;
  /** Surface relation that must hold while the summary is generated asynchronously. */
  readonly stability: 'whole-surface' | 'selected-span';
  readonly flush?: () => Promise<void>;
  readonly sourceCommandId?: SourceCommandId;
}

/** Distinguishes 'history changed' from 'summary failed' when the transaction aborts. */
class SurfaceChangedError extends Error {}

/** Shape only; the Loader renders the config UI, cross-field constraints live in `resolveConfig`. */
const ConfigSchema = z.object({
  enabled: z.boolean().description('总开关'),
  auto: z.boolean().description('注册自动压力压缩与上下文溢出恢复'),
  thresholdRatio: z.number().min(0.05).max(1).description('占用达到该比例即压缩历史'),
  retainRatio: z.number().min(0).max(0.9).description('保留为逐字原文的最近尾部比例'),
  retainTokens: z.number().step(1).min(0).description('>0 时覆盖 retainRatio'),
  maxSummaryTokens: z.number().step(1).min(128).description('摘要生成上限'),
  compactionRetries: z.number().step(1).min(0).description('压力压缩重试次数'),
  maxOverflowRetries: z.number().step(1).min(0).description('溢出恢复重试次数'),
  faithful: z.boolean().description('高保真摘要（保留精确值）'),
  summaryLanguage: z.union([z.const('auto'), z.const('zh'), z.const('en')]).description('摘要语言'),
  summaryCacheSize: z.number().step(1).min(0).description('摘要缓存条数'),
  summarizationProvider: z.string().description('摘要 provider（空 = 跟随会话路由）'),
  summarizationModel: z.string().description('摘要 model（空 = 跟随会话路由）'),

  enableToolCompression: z.boolean().description('工具输出压缩开关'),
  maxToolOutputTokens: z.number().step(1).min(64).description('工具输出压缩阈值'),
  minToolOutputTokens: z.number().step(1).min(0).description('低于该值不压缩'),
  targetToolOutputTokens: z.number().step(1).min(32).description('压缩目标 token'),
  jsonMaxStringChars: z.number().step(1).min(16).description('JSON 字符串截断长度'),
  jsonHeadItems: z.number().step(1).min(0).description('数组保留头部元素数'),
  jsonTailItems: z.number().step(1).min(0).description('数组保留尾部元素数'),
  logHeadLines: z.number().step(1).min(0).description('日志保留头部行数'),
  logTailLines: z.number().step(1).min(0).description('日志保留尾部行数'),
  logMaxIssues: z.number().step(1).min(0).description('日志保留关键行上限'),
  logContextLines: z.number().step(1).min(0).description('关键行上下文行数'),
  tableHeadRows: z.number().step(1).min(0).description('表格保留前 N 行'),
  tableTailRows: z.number().step(1).min(0).description('表格保留后 N 行'),
  textHeadChars: z.number().step(1).min(0).description('长文本保留头部字符'),
  textTailChars: z.number().step(1).min(0).description('长文本保留尾部字符'),
  textExtractSignals: z.boolean().description('长文本抽取中部高信号句'),

  enableCCR: z.boolean().description('可逆压缩开关'),
  ccrMaxEntries: z.number().step(1).min(0).description('CCR 条目上限'),
  ccrMaxBytes: z.number().step(1).min(0).description('CCR 字节上限'),
  ccrTtlMs: z.number().step(1).min(0).description('CCR 存活时间'),
  ccrPersist: z.boolean().description('CCR 落盘'),
  ccrStorePath: z.string().description('CCR 存储路径'),
  ccrMaxChars: z.number().step(1).min(1024).description('单条 CCR 内容上限'),

  assemblyEnabled: z.boolean().description('动态组装开关'),
  noticeEnabled: z.boolean().description('注入上下文状态提示'),
  reservedOutputTokens: z.number().step(1).min(0).description('为输出预留的 token'),
  systemPromptTokenFallback: z.number().step(1).min(0).description('系统提示 token 兜底估计'),
  logLevel: z.union([z.const('debug'), z.const('info'), z.const('warn'), z.const('error')]).description('日志级别')
});

  /**
   * Provides `ctx.compaction` and registers the extension points. A broken guard
   * must never fail a tool call, so every subsystem contains its own failures.
   */
export class ContextGuardEngine extends CompactionEngine {
  /** All required: partial activation would leave a half-working backend. */
  static inject = ['llm', 'tokenMeter', 'sessions'];

  static Config = ConfigSchema;

  readonly config: ResolvedConfig;

  private readonly logger: GuardLogger;
  private readonly estimator: TokenEstimator;
  private readonly ccr: CCRStore;
  private readonly summarizer: HighFidelitySummarizer;
  private readonly assembler: ContextAssembler;
  private readonly states = new WeakMap<Session, CompactionState>();
  /** One status notice per turn: re-injecting every step keeps changing the request prefix. */
  private readonly noticeTurns = new WeakMap<Session, number>();
  private readonly overflowRetries = new WeakMap<Agent, number>();
  private language: SummaryLanguage = 'en';

  constructor(ctx: Context, config: ContextGuardConfig = {}) {
    super(ctx);
    this.config = resolveConfig(config);
    this.logger = new GuardLogger(this.config.logLevel, createCordisSink(ctx.logger));
    this.estimator = defaultTokenEstimator;
    this.ccr = new CCRStore(this.config, this.logger);
    this.summarizer = new HighFidelitySummarizer(this.config, this.logger, this.estimator, this.createCompletion());

    if (this.config.unknownKeys.length > 0) {
      this.logger.warn(`ignoring unknown config keys: ${this.config.unknownKeys.join(', ')}`);
    }
    if (!this.config.enabled) {
      this.logger.warn('enabled=false: the plugin still owns ctx.compaction and compresses nothing. If mounted alongside compaction-basic, keep only one.');
    }

    this.assembler = new ContextAssembler(this.config, this.estimator);
    this.registerToolOutputCompression();
    this.registerRetrieveTool();
    this.registerPromptSection();
    if (this.config.auto) this.registerAutomaticHooks();

    this.logger.info(
      `loaded: threshold ${this.config.thresholdRatio}, tool output ${this.config.maxToolOutputTokens} tokens, CCR ${this.config.enableCCR ? 'on' : 'off'}, faithful ${this.config.faithful ? 'on' : 'off'}`
    );
  }

  // --- tool output compression ---

  /**
   * `tools/post-execute` waterfall: the only point where the real output is known
   * but not yet in the request. Must call `next()`; only `accept` decisions are
   * replaced.
   */
  private registerToolOutputCompression(): void {
    if (!this.config.enableToolCompression || !this.config.enabled) return;
    this.ctx.on('tools/post-execute', async (exec: ToolExecution, result: Readonly<ToolExecutionResult>, next: () => Promise<PostToolDecision>) => {
      let decision: PostToolDecision;
      try {
        decision = await next();
      } catch (error) {
        // next() failed in an upstream policy; not ours to fix, rethrow as-is
        this.logger.warn(`post-execute upstream failed: ${GuardLogger.describe(error)}`);
        throw error;
      }
      if (decision.kind !== 'accept') return decision;
      // Retrieval results are never re-compressed: the model asked for the verbatim
      // original, and compressing it again traps it in a retrieve -> compressed ->
      // retrieve loop. The tool's own offset/limit paging bounds the window instead.
      if (exec.name === RETRIEVE_TOOL_NAME) return decision;
      const content = decision.content ?? result.content;
      try {
        const outcome = compressContentBlocks(content as readonly GuardContentBlock[], this.config, this.estimator, { toolName: exec.name });
        if (outcome === null) return decision;
        const meta = { ...outcome.meta };
        let blocks: readonly GuardContentBlock[] = outcome.blocks;
        if (this.config.enableCCR) {
          const originalId = this.ccr.store(textOf(content), { toolName: exec.name, tokens: outcome.meta.originalTokens });
          if (originalId !== undefined) blocks = appendRetrievalFooter(blocks, { ...meta, originalId }, true);
          else blocks = appendRetrievalFooter(blocks, meta, false);
        } else {
          blocks = appendRetrievalFooter(blocks, meta, false);
        }
        this.logger.debug(
          `tool ${exec.name} output compressed: ${outcome.meta.category}/${outcome.meta.strategy} ${outcome.meta.originalTokens}->${outcome.meta.compressedTokens} tokens`
        );
        return { kind: 'accept', content: blocks as ContentBlock[] };
      } catch (error) {
        // fall back to the original: a bloated output beats a corrupt one
        this.logger.warn(`tool ${exec.name} output compression failed, original kept: ${GuardLogger.describe(error)}`);
        return decision;
      }
    });
  }

  /**
   * `tools` is optional: `ctx.inject` keeps the backend mountable without a tools
   * service (headless compaction).
   */
  private registerRetrieveTool(): void {
    if (!this.config.enabled || !this.config.enableCCR) return;
    this.ctx.inject(['tools'], (ctx) => {
      ctx.tools.register(
        defineTool({
          name: RETRIEVE_TOOL_NAME,
          description:
            'Restore the original content behind a compressed tool output or compacted history entry. ' +
            'Compressed results carry a footer with an id such as id="ccr_0123456789abcdef"; pass that id here. ' +
            'Use it whenever you need details that were omitted (full logs, full JSON, full file content) instead of guessing. ' +
            `The restored text is returned VERBATIM and is never compressed again; when it is longer than one page, the footer tells you the total length and the next offset to continue from. ` +
            'Returns { found: false } when the entry expired or was evicted.',
          parameters: {
            id: {
              type: 'string',
              required: true,
              description: 'The id from the compression footer, e.g. "ccr_0123456789abcdef".'
            },
            offset: {
              type: 'integer',
              description: 'Character offset to start from (default 0). Use the offset printed in the previous footer to page through a long original.'
            },
            limit: {
              type: 'integer',
              description: `Maximum characters to return in this call (default and hard cap: ${this.config.retrieveMaxChars}).`
            }
          },
          output: {
            schema: {
              type: 'object',
              additionalProperties: false,
              properties: {
                found: { type: 'boolean', required: true, description: 'Whether the original content was found.' },
                id: { type: 'string', description: 'The requested id.' },
                content: { type: 'string', description: 'The verbatim content of this page.' },
                note: { type: 'string', description: 'Human-readable explanation when not found.' },
                tokens: { type: 'integer', description: 'Token estimate of this page.' },
                toolName: { type: 'string', description: 'The tool that produced the original output.' },
                offset: { type: 'integer', description: 'Offset this page starts at.' },
                totalChars: { type: 'integer', description: 'Total length of the original text in characters.' },
                truncated: { type: 'boolean', description: 'Whether more content follows this page.' }
              }
            },
            render: (_args, value) => {
              if (value.found !== true) return [{ type: 'text', text: `[context-guard] ${value.note ?? 'original content not found'}` }];
              const start = value.offset ?? 0;
              const end = start + (value.content?.length ?? 0);
              const total = value.totalChars ?? end;
              const paging = value.truncated === true
                ? ` — page ${start}-${end} of ${total} chars; continue with offset=${end}`
                : ` — complete (${start}-${end} of ${total} chars)`;
              return [
                {
                  type: 'text',
                  text: `${value.content ?? ''}\n\n[context-guard] restored verbatim (never re-compressed) ~${value.tokens ?? 0} tokens from ${value.toolName ?? 'unknown tool'}${paging}`
                }
              ];
            }
          },
          isConcurrencySafe: () => true,
          execute: async (args, exec) => {
            if (exec.signal.aborted) throw new Error('context_retrieve aborted');
            const record = await this.ccr.retrieveFromDisk(args.id);
            if (record === null) {
              return {
                found: false,
                id: args.id,
                note: 'No original content for this id. It may have expired (ccrTtlMs), been evicted (ccrMaxEntries/ccrMaxBytes), or CCR persistence may be disabled. Re-run the original tool if you still need the data.'
              };
            }
            const page = sliceForRetrieval(record.text, args.offset ?? 0, Math.min(args.limit ?? this.config.retrieveMaxChars, this.config.retrieveMaxChars));
            return {
              found: true,
              id: record.id,
              content: page.content,
              tokens: this.estimator.estimate(page.content),
              offset: page.offset,
              totalChars: page.totalChars,
              truncated: page.truncated,
              ...(record.toolName === undefined ? {} : { toolName: record.toolName })
            };
          }
        })
      );
    });
  }

  // A registered section is idempotent, dies with the fiber and keeps a stable
  // position (KV cache); the assemble waterfall is only for rewriting others' output.
  private registerPromptSection(): void {
    if (!this.config.enabled) return;
    this.ctx.inject(['systemPrompt'], (ctx) => {
      ctx.systemPrompt.section({
        name: PROMPT_SECTION,
        order: PROMPT_SECTION_ORDER,
        text: () => this.assembler.capabilityNotice(this.language)
      });
    });
  }

  // --- layers 2/3: automatic triggers ---

  private registerAutomaticHooks(): void {
    this.ctx.on('agent/pre-step', async (payload, next) => {
      const { agent, signal } = payload;
      if (this.config.enabled) {
        try {
          const result = await this.compactIfNeeded(agent, 'pressure', signal);
          if (result !== null) {
            this.logger.info(
              `pressure compaction: replaced ${result.shadowedSeqs.length} surface nodes (seq ${result.shadowedRange.start}-${result.shadowedRange.end}, ~${result.shadowedTokenCount} tokens)`
            );
          }
        } catch (error) {
          // a failed compaction must never break the turn: log it and continue
          this.logger.warn(`pressure compaction failed, continuing the turn: ${GuardLogger.describe(error)}`);
        }
      }
      const decision = await next();
      if (decision.kind !== 'enter') return decision;
      return this.applyAssemblyLayer(agent, payload.turn, decision);
    });

    this.ctx.on('agent/request-error', async (payload, next) => {
      const { agent, failure, signal } = payload;
      if (!this.config.enabled || failure.code !== CONTEXT_WINDOW_EXCEEDED_CODE || signal.aborted) return next();
      const retries = this.overflowRetries.get(agent) ?? 0;
      if (retries >= this.config.maxOverflowRetries) return next();
      const generation = agent.session.surface.replaceGeneration;
      try {
        const result = await this.compactIfNeeded(agent, 'context-overflow', signal);
        if (result !== null) this.logger.info(`context overflow recovery: compacted ${result.shadowedSeqs.length} nodes`);
      } catch (error) {
        this.logger.warn(`context overflow recovery failed: ${GuardLogger.describe(error)}`);
        if (!signal.aborted && agent.session.surface.replaceGeneration > generation) {
          this.overflowRetries.set(agent, retries + 1);
          return { kind: 'retry' };
        }
        return next();
      }
      if (signal.aborted || agent.session.surface.replaceGeneration <= generation) return next();
      this.overflowRetries.set(agent, retries + 1);
      return { kind: 'retry' };
    });

    this.ctx.on('agent/status', ({ agent, status }) => {
      if (status === 'idle') this.overflowRetries.delete(agent);
    });
  }

  /**
   * Assembly layer: trims what is about to enter the request and injects a status
   * notice. History is not ours to drop here; that is the compactor's job.
   */
  private applyAssemblyLayer(
    agent: Agent,
    turn: number,
    decision: { kind: 'enter'; messages: UserMessage[]; startsRequestSeries?: true }
  ): PostStepDecision {
    if (!this.config.enabled || !this.config.assemblyEnabled) return decision as PostStepDecision;
    try {
      const session = agent.session;
      const measurement = this.measure(session);
      const contextWindow = session.requestContext()?.contextWindow;
      const budget = this.assembler.budget({
        ...(contextWindow === undefined ? {} : { contextWindow }),
        usedTokens: measurement.totalTokens,
        systemTokens: this.config.systemPromptTokenFallback,
        reservedOutputTokens: this.config.reservedOutputTokens,
        fallbackContextWindow: this.config.systemPromptTokenFallback * 16
      });

      // pre-step payload.messages only carries user-role messages (the loop appends
      // tool results as their own surface nodes), so rank by source: never drop real
      // human input (source.kind === 'user'), injected context is droppable.
      const candidates: AssemblyMessage[] = decision.messages.map((message, index) => {
        const isFromUser = message.source.kind === 'user';
        const compressed = isCompressedBlocks(message.content as readonly GuardContentBlock[]);
        // priority 3 = already-compressed injected context, dropped first
        const priority: MessagePriority = isFromUser ? 0 : compressed ? 3 : 2;
        return {
          id: String(message.id),
          role: 'user' as const,
          tokens: this.meter.estimateMessage(message),
          priority,
          compressed,
          droppable: !isFromUser,
          label: `${message.source.kind}#${index}`
        };
      });
      const plan = this.assembler.plan(candidates, Math.max(0, budget.available));
      let admitted = decision.messages;
      if (plan.dropped.length > 0) {
        const dropped = new Set(plan.dropped.map((message) => message.id));
        admitted = decision.messages.filter((message) => !dropped.has(String(message.id)));
        this.logger.warn(
          `budget trim: dropped ${plan.dropped.length} injected messages (~${plan.freedTokens} tokens freed); user messages are always kept`
        );
      }
      if (plan.stillOverBudget) {
        this.logger.warn('budget exhausted and the compaction layer freed nothing; continuing with this step');
      }

      const language = this.resolveSessionLanguage(agent.session, decision.messages);
      this.language = language;
      const state = this.states.get(session) ?? emptyCompactionState();
      const notice = this.assembler.statusNotice(state, budget, language);
      if (notice === null) return { ...decision, messages: admitted } as PostStepDecision;
      // once per turn: injecting every step keeps changing the request prefix
      const lastTurn = this.noticeTurns.get(session);
      if (lastTurn === turn) return { ...decision, messages: admitted } as PostStepDecision;
      this.noticeTurns.set(session, turn);
      const noticeMessage = createUserMessage({
        content: [{ type: 'text', text: notice }],
        source: { kind: 'context-guard' }
      });
      return { ...decision, messages: [...admitted, noticeMessage] } as PostStepDecision;
    } catch (error) {
      this.logger.warn(`assembly failed, continuing unchanged: ${GuardLogger.describe(error)}`);
      return decision as PostStepDecision;
    }
  }

  // --- layer 2: CompactionEngine implementation ---

  async compactIfNeeded(agent: Agent, trigger: CompactionTrigger, signal: AbortSignal): Promise<CompactionResult | null> {
    const session = agent.session;
    if (!this.config.enabled) return null;
    const target = routedTarget(session) ?? agentRoute(agent);
    if (target === undefined) return null;

    let measurement = this.measure(session);

    if (trigger === 'context-overflow') {
      // on overflow drop the keep-recent-tail policy: shrink the window as fast as possible
      const range = selectCompactableRange(session, measurement, 0);
      if (range === null) return null;
      return this.compactRegion(range.start, range.end, agent, signal);
    }

    const contextWindow = await this.resolveContextWindow(target, signal);
    if (contextWindow === undefined) {
      this.logger.warn(
        `no context window for ${target.provider}/${target.model} (adapter declares no contextWindow), skipping pressure compaction`
      );
      return null;
    }
    const thresholdTokens = Math.floor(contextWindow * this.config.thresholdRatio);
    if (measurement.totalTokens < thresholdTokens) return null;

    assertNoActiveCompaction(session, 'automatic pressure compaction');
    const retainTokens =
      this.config.retainTokens > 0 ? this.config.retainTokens : Math.floor(contextWindow * this.config.retainRatio);

    let result: CompactionResult | null = null;
    for (let attempt = 0; attempt <= this.config.compactionRetries; attempt += 1) {
      const range = selectCompactableRange(session, measurement, retainTokens);
      if (range === null) break;
      result = await this.compactRegion(range.start, range.end, agent, signal);
      measurement = this.measure(session);
      if (measurement.totalTokens < thresholdTokens) return result;
    }
    if (result !== null) {
      this.logger.warn(
        `still above threshold after ${this.config.compactionRetries + 1} compaction attempts (~${measurement.totalTokens} >= ${thresholdTokens} tokens)`
      );
    }
    return result;
  }

  async compactRegion(start: SessionSeq, end: SessionSeq, agent: Agent, signal?: AbortSignal): Promise<CompactionResult> {
    return this.runTransaction(agent.session, start, end, agent, { owner: 'current-turn', stability: 'whole-surface' }, signal);
  }

  // /compact path: force one useful compaction on an idle session
  async compactNow(agent: Agent, signal: AbortSignal, sourceCommandId?: SourceCommandId): Promise<CompactionResult | null> {
    signal.throwIfAborted();
    try {
      return await agent.runMaintenance(async (agentSignal) => {
        const operationSignal = AbortSignal.any([agentSignal, signal]);
        operationSignal.throwIfAborted();
        const range = selectCompactableRange(agent.session, this.measure(agent.session), 0);
        if (range === null) return null;
        return await this.runTransaction(
          agent.session,
          range.start,
          range.end,
          agent,
          {
            owner: null,
            stability: 'selected-span',
            ...(sourceCommandId === undefined ? {} : { sourceCommandId }),
            flush: async () => {
              await this.ctx.sessions.flush(agent.session);
            }
          },
          operationSignal
        );
      });
    } catch (error) {
      if (error instanceof ManualCompactionError) throw error;
      if (signal.aborted) throw new ManualCompactionError('cancelled', 'manual compaction cancelled', { cause: error });
      throw new ManualCompactionError('busy', 'manual compaction needs an idle agent with no pending work', { cause: error });
    }
  }

  // --- compaction transaction (DSH durable protocol) ---

  /**
   * One compaction transaction. `compaction/start` is the lock, the summary must be
   * produced inside it, `compaction/summary` must be immediately followed by the
   * replacement `user/message` (it records that message's shadow price), and
   * `compaction/end` is attempted exactly once, success or failure.
   */
  private async runTransaction(
    session: Session,
    start: SessionSeq,
    end: SessionSeq,
    agent: Agent,
    options: TransactionOptions,
    signal?: AbortSignal
  ): Promise<CompactionResult> {
    if (options.owner === null) signal?.throwIfAborted();
    // The summary call replays the session's system/tools to hit the KV cache, so
    // the current session and agent are registered here (agent = routing fallback).
    this.activeSession = session;
    this.activeAgent = agent;
    const selection = validateSurfaceRegion(session, start, end);
    const entryState = inspectEntryState(session);
    assertCompactionInactive(entryState.unmatchedCompactionStart, entryState.latestEndSeedSeq, 'context-guard compaction');

    let owner: number | null;
    if (options.owner === null) {
      if (entryState.openTurn !== null) throw new ManualCompactionError('busy', 'manual compaction: session still has an open turn');
      owner = null;
    } else {
      if (entryState.openTurn === null) throw new Error('compactRegion: no open turn - automatic compaction events must be wrapped in a turn');
      owner = entryState.openTurn;
    }

    const compactionId = CompactionId(randomUUID());
    const lifecycle = {
      compactionId,
      ...(options.sourceCommandId === undefined ? {} : { sourceCommandId: options.sourceCommandId }),
      turn: owner
    };
    const startEvent = session.append('compaction/start', lifecycle);

    let failure: { error: unknown; stage: 'summary' | 'commit' } | undefined;
    let flushFailure: unknown;
    let result: CompactionResult | undefined;
    let closing = false;
    let stage: 'summary' | 'commit' = 'summary';

    try {
      const measurement = this.measure(session);
      const selectedNodes = measurement.nodes.slice(selection.startIdx, selection.endIdx + 1);
      if (
        selectedNodes.length !== selection.shadowedSeqs.length ||
        selectedNodes.some((node, index) => node.seq !== selection.shadowedSeqs[index])
      ) {
        throw new SurfaceChangedError('context-guard: surface changed before the summary started');
      }
      const shadowedTokenCount = selectedNodes.reduce((total, node) => total + node.heuristicTokens, 0);
      const shadowedRouteTokenCount = selectedNodes.reduce((total, node) => total + node.tokens, 0);
      const input = this.buildSummarizationInput(session, selection.shadowedSeqs);
      const outcome = await this.summarizer.summarize(input, signal);
      const checkpointMessage = createUserMessage({
        content: outcome.blocks as ContentBlock[],
        source: compactCheckpointSource(compactionId, options.sourceCommandId)
      });
      const framedTokens = this.meter.estimateMessage(checkpointMessage);
      if (framedTokens >= shadowedRouteTokenCount) {
        // a summary that is not smaller makes compaction pointless, or worse
        throw new Error(
          `summary is not smaller than the shadowed content (${framedTokens} estimated tokens >= ${shadowedRouteTokenCount})`
        );
      }

      if (options.owner === null) signal?.throwIfAborted();
      if (options.stability === 'whole-surface') assertWholeSurfaceUnchanged(this.measure(session), measurement);
      else assertSelectedSpanStable(session, selection);

      stage = 'commit';
      const summaryEvent = session.append('compaction/summary', {
        compactionId,
        ...(options.sourceCommandId === undefined ? {} : { sourceCommandId: options.sourceCommandId }),
        summary: outcome.blocks as ContentBlock[],
        shadowedRange: { start, end },
        shadowedSeqs: [...selection.shadowedSeqs],
        shadowedTokenCount,
        provider: outcome.provider,
        model: outcome.model,
        maxTokens: this.config.maxSummaryTokens,
        ...(outcome.rawOutput === undefined ? {} : { rawOutput: outcome.rawOutput as ContentBlock[] }),
        ...(outcome.usage === undefined ? {} : { usage: outcome.usage as never })
      });
      // the adjacent event: the replacement message; no append may go in between
      session.append('user/message', checkpointMessage, {
        // 0.2.0 renamed the replace bounds to startSeq/endSeq (same semantics: inclusive, by surface position)
        surfaceOp: { op: 'replace', startSeq: start, endSeq: end },
        sourceEventSeqs: [startEvent.seq, summaryEvent.seq, ...selection.shadowedSeqs]
      });
      closing = true;
      const endEvent = session.append('compaction/end', lifecycle);
      result = {
        compactionId,
        ...(options.sourceCommandId === undefined ? {} : { sourceCommandId: options.sourceCommandId }),
        startSeq: startEvent.seq,
        summarySeq: summaryEvent.seq,
        endSeq: endEvent.seq,
        summary: outcome.blocks as ContentBlock[],
        shadowedRange: { start, end },
        shadowedSeqs: [...selection.shadowedSeqs],
        shadowedTokenCount
      };
      const previous = this.states.get(session) ?? emptyCompactionState();
      this.states.set(session, {
        compactions: previous.compactions + 1,
        lastShadowedMessages: selection.shadowedSeqs.length,
        lastOriginalTokens: shadowedRouteTokenCount,
        lastSummaryTokens: framedTokens,
        retrievableEntries: this.ccr.stats().entries
      });
    } catch (error) {
      failure = { error, stage };
      if (!closing) {
        closing = true;
        try {
          session.append('compaction/end', { ...lifecycle, error: GuardLogger.describe(error) });
        } catch (closeError) {
          failure = { error: closeError, stage: 'commit' };
        }
      }
    }

    if (options.flush !== undefined && result !== undefined) {
      try {
        await options.flush();
      } catch (error) {
        flushFailure = error;
      }
    }
    if (options.owner === null) signal?.throwIfAborted();
    if (failure !== undefined) {
      if (options.owner === null) {
        // manual path: classify the failure so the command layer can give actionable feedback
        if (failure.stage === 'commit') throw new ManualCompactionError('commit', 'manual compaction commit did not finish cleanly', { cause: failure.error });
        if (failure.error instanceof SurfaceChangedError) {
          throw new ManualCompactionError('changed', 'history changed during manual compaction', { cause: failure.error });
        }
        throw new ManualCompactionError('summary', 'manual compaction produced no smaller summary', { cause: failure.error });
      }
      throw failure.error;
    }
    if (flushFailure !== undefined) {
      throw new ManualCompactionError('persistence', 'manual compaction persistence checkpoint failed', { cause: flushFailure });
    }
    if (result === undefined) throw new Error('compaction committed without a result');
    return result;
  }

  // --- helpers ---

  private get meter(): TokenMeter {
    return this.ctx.tokenMeter;
  }

  private measure(session: Session): MeasurementLike {
    return this.meter.measure(session) as unknown as MeasurementLike;
  }

  /** Adapter-declared window first, then the session's last request. */
  private async resolveContextWindow(target: { provider: string; model: string }, signal: AbortSignal): Promise<number | undefined> {
    try {
      const info = await this.ctx.llm.resolveModelInfo(target.provider, target.model, signal);
      const declared = info.context?.contextWindow;
      if (typeof declared === 'number' && declared > 0) return declared;
    } catch (error) {
      this.logger.debug(`resolveModelInfo failed: ${GuardLogger.describe(error)}`);
    }
    return undefined;
  }

  /** Reuses the session's own system/tools to hit the KV cache, then projects a plain-text view. */
  private buildSummarizationInput(session: Session, seqs: readonly SessionSeq[]): SummarizationInput {
    const header = session.requestHeader();
    const raw: Message[] = [];
    const simple: SimpleMessage[] = [];
    const toolNames = collectToolNames(session, seqs);
    for (const seq of seqs) {
      const event = session.eventAt(seq);
      if (event === undefined) continue;
      const message = session.deriveEventMessage(event);
      if (message === null) continue;
      raw.push(message);
      simple.push(toSimpleMessage(message, toolNames));
    }
    return {
      ...(header?.system === undefined ? {} : { system: header.system }),
      ...(header?.tools === undefined ? {} : { tools: header.tools }),
      messages: simple,
      raw
    };
  }

  /** From the most recent user messages; falls back to the current step's messages. */
  private resolveSessionLanguage(session: Session, extra: readonly Message[]): SummaryLanguage {
    const samples: string[] = [];
    for (const message of extra) {
      if (message.role === 'user') samples.push(textOf(message.content));
    }
    for (const seq of session.surface.nodes.slice(-8)) {
      const event = session.eventAt(seq);
      if (event === undefined || event.type !== 'user/message') continue;
      const message = session.deriveEventMessage(event);
      if (message !== null && message.role === 'user') samples.push(textOf(message.content));
    }
    if (samples.length === 0) return this.language;
    return detectLanguage(samples, this.config.summaryLanguage);
  }

  /**
   * Sends the raw messages instead of the text projection and reuses the session's
   * own system/tools, so this call is a true prefix of the last request and the
   * provider's KV cache applies.
   */
  private createCompletion(): TextCompletion {
    return async (request: CompletionRequest): Promise<CompletionResult> => {
      const target = this.completionTarget();
      const raw = Array.isArray(request.raw) ? (request.raw as Message[]) : [];
      const instruction = createUserMessage({
        content: [{ type: 'text', text: request.instruction }],
        source: { kind: 'context-guard' }
      });
      const options = {
        provider: target.provider,
        model: target.model,
        messages: [...raw, instruction],
        ...(request.system === undefined ? {} : { system: request.system }),
        ...(request.tools === undefined ? {} : { tools: [...(request.tools as ToolSchema[])] }),
        maxTokens: request.maxTokens,
        sessionId: this.completionSessionId(),
        purpose: 'compaction' as const,
        ...(request.signal === undefined ? {} : { signal: request.signal })
      } satisfies GenerateOptions;

      const assembler = new BlockAssembler();
      for await (const chunk of this.ctx.llm.stream(options) as AsyncIterable<StreamChunk>) assembler.push(chunk);
      const finish = assembler.finish;
      if (finish.kind === 'error' || finish.kind === 'aborted') {
        throw new Error(`summary request failed: ${finish.failure.message}`);
      }
      if (finish.kind === 'max-tokens') {
        // truncated JSON can never parse: fail early so the caller takes the deterministic path
        throw new Error('MAX_TOKENS: summary output was truncated by the generation limit');
      }
      const rawOutput = assembler.blocks();
      const text = rawOutput
        .filter((block): block is TextBlock => block.type === 'text')
        .map((block) => block.text)
        .join('\n');
      return {
        text,
        provider: options.provider,
        model: options.model,
        rawOutput: rawOutput.map((block) => ({ type: block.type, ...('text' in block ? { text: block.text } : {}) })),
        ...(assembler.usage === undefined ? {} : { usage: assembler.usage })
      };
    };
  }

  /** Explicit config > last session route > agent route (same order as the built-in backend). */
  private completionTarget(): { provider: string; model: string } {
    if (this.config.summarizationProvider !== '' && this.config.summarizationModel !== '') {
      return { provider: this.config.summarizationProvider, model: this.config.summarizationModel };
    }
    const session = this.activeSession;
    const routed = session === undefined ? undefined : routedTarget(session);
    if (routed !== undefined) return routed;
    const fromAgent = this.activeAgent === undefined ? undefined : agentRoute(this.activeAgent);
    if (fromAgent !== undefined) return fromAgent;
    throw new Error(
      'no summarization provider/model: configure summarizationProvider and summarizationModel, or let the session route one request first'
    );
  }

  private completionSessionId(): Session['id'] | undefined {
    return this.activeSession?.id;
  }

  private activeSession: Session | undefined;

  private activeAgent: Agent | undefined;
}

/** Same shape as dsh-agent's pre-step decision, declared here to keep its type surface out of this module. */
type PostStepDecision =
  | { kind: 'reject' }
  | { kind: 'enter'; messages: UserMessage[]; startsRequestSeries?: true };

export function textOf(blocks: readonly { readonly type: string; readonly text?: string }[]): string {
  return blocks
    .filter((block) => typeof block.text === 'string' && (block.type === 'text' || block.type === 'reasoning'))
    .map((block) => block.text ?? '')
    .join('\n');
}

function routedTarget(session: Session): { provider: string; model: string } | undefined {
  const config = session.requestHeader()?.config;
  if (config === undefined || config.provider.length === 0 || config.model.length === 0) return undefined;
  return { provider: config.provider, model: config.model };
}

/** Route from agent options; only valid when both provider and model are set. */
function agentRoute(agent: Agent): { provider: string; model: string } | undefined {
  const { provider, model } = agent.options;
  if (provider === undefined || provider.length === 0 || model === undefined || model.length === 0) return undefined;
  return { provider, model };
}

/** tool/result events carry no tool name, so map toolCallId -> name over the span. */
function collectToolNames(session: Session, seqs: readonly SessionSeq[]): Map<string, string> {
  const names = new Map<string, string>();
  for (const seq of seqs) {
    const event = session.eventAt(seq);
    if (event === undefined || event.type !== 'assistant/message') continue;
    for (const block of event.data.message.content) {
      if (block.type === 'tool-call') names.set(String(block.id), block.name);
    }
  }
  return names;
}

// 0.2.0 message model: tool results are first-class `role: 'tool'` messages (toolCallId
// + message-level isError), there is no `tool-result` block, and `developer` joined the
// role union. The projection must follow, or the tool -> ok/failed ledger goes wrong.
function toSimpleMessage(message: Message, toolNames: Map<string, string>): SimpleMessage {
  const text = textOf(message.content);
  if (message.role === 'assistant') return { role: 'assistant', text };
  if (message.role === 'tool') {
    const toolName = toolNames.get(String(message.toolCallId));
    return {
      role: 'tool',
      text,
      ...(toolName === undefined ? {} : { toolName }),
      ...(message.isError === true ? { isError: true } : {})
    };
  }
  if (message.role === 'system' || message.role === 'developer') return { role: 'system', text };
  return { role: 'user', text };
}

interface SurfaceSelection {
  readonly start: SessionSeq;
  readonly end: SessionSeq;
  readonly startIdx: number;
  readonly endIdx: number;
  readonly shadowedSeqs: readonly SessionSeq[];
}

/** Both ends must exist, be ordered, and not cut a tool call/result pair. */
function validateSurfaceRegion(session: Session, start: SessionSeq, end: SessionSeq): SurfaceSelection {
  const nodes = session.surface.nodes;
  const startIdx = nodes.indexOf(start);
  const endIdx = nodes.indexOf(end);
  if (startIdx === -1) throw new Error(`compactRegion: start seq ${start} is not on the current surface`);
  if (endIdx === -1) throw new Error(`compactRegion: end seq ${end} is not on the current surface`);
  if (startIdx > endIdx) throw new Error(`compactRegion: start seq ${start} comes after end seq ${end}`);
  if (!toolPairingBalancedBefore(session, nodes[startIdx] as SessionSeq)) {
    throw new Error(`compactRegion: start seq ${start} is not a balanced boundary (would cut a tool call/result pair)`);
  }
  if (!toolPairingBalancedAfter(session, nodes[endIdx] as SessionSeq)) {
    throw new Error(`compactRegion: end seq ${end} is not a balanced boundary (would cut a step, or the step is still open)`);
  }
  return { start, end, startIdx, endIdx, shadowedSeqs: nodes.slice(startIdx, endIdx + 1) };
}

/** The whole surface must be unchanged across the summary call. */
function assertWholeSurfaceUnchanged(current: MeasurementLike, prepared: MeasurementLike): void {
  const same =
    current.nodes.length === prepared.nodes.length &&
    current.nodes.every((node, index) => node.seq === prepared.nodes[index]?.seq && node.tokens === prepared.nodes[index]?.tokens);
  if (!same) throw new SurfaceChangedError('context-guard: session surface changed during summarization');
}

/** The selected span must be unchanged; the manual path tolerates new events outside it. */
function assertSelectedSpanStable(session: Session, prepared: SurfaceSelection): void {
  let current: SurfaceSelection;
  try {
    current = validateSurfaceRegion(session, prepared.start, prepared.end);
  } catch (error) {
    throw new SurfaceChangedError('context-guard: the selected span is no longer a valid replacement target', { cause: error });
  }
  const same =
    current.shadowedSeqs.length === prepared.shadowedSeqs.length &&
    current.shadowedSeqs.every((seq, index) => seq === prepared.shadowedSeqs[index]);
  if (!same) throw new SurfaceChangedError('context-guard: the selected span changed during summarization');
}

/** Session entry state: open turn, unmatched compaction/start, latest seed boundary. */
interface EntryState {
  readonly openTurn: number | null;
  readonly unmatchedCompactionStart: { readonly seq: SessionSeq } | undefined;
  readonly latestEndSeedSeq: SessionSeq | undefined;
}

/** Scans the session log backwards for the state a compaction entry needs. */
function inspectEntryState(session: Session): EntryState {
  let openTurn: number | null = null;
  let openTurnKnown = false;
  let unmatchedCompactionStart: { seq: SessionSeq } | undefined;
  let compactionKnown = false;
  let latestEndSeedSeq: SessionSeq | undefined;
  for (let seq = session.seq - 1; seq >= 0; seq -= 1) {
    const event = session.eventAt(SessionSeq(seq));
    if (event === undefined) continue;
    if (latestEndSeedSeq === undefined && event.type === 'session/end-seed') latestEndSeedSeq = event.seq;
    if (!compactionKnown) {
      if (event.type === 'compaction/start') {
        unmatchedCompactionStart = { seq: event.seq };
        compactionKnown = true;
      } else if (event.type === 'compaction/end') {
        compactionKnown = true;
      }
    }
    if (!openTurnKnown) {
      if (event.type === 'turn/start') {
        openTurn = event.data.turn;
        openTurnKnown = true;
      } else if (event.type === 'turn/end') {
        openTurnKnown = true;
      }
    }
    if (openTurnKnown && compactionKnown && latestEndSeedSeq !== undefined) break;
  }
  return { openTurn, unmatchedCompactionStart, latestEndSeedSeq };
}

/** An unmatched `compaction/start` means the session lock is held. */
function assertCompactionInactive(unmatchedCompactionStart: { seq: SessionSeq } | undefined, latestEndSeedSeq: SessionSeq | undefined, stage: string): void {
  if (unmatchedCompactionStart === undefined) return;
  if (latestEndSeedSeq !== undefined && latestEndSeedSeq > unmatchedCompactionStart.seq) return;
  throw new ManualCompactionError('busy', `${stage}: a compaction is already in progress (session compaction lock is held)`);
}

/** Re-checks the lock after the async window before summarization. */
function assertNoActiveCompaction(session: Session, stage: string): void {
  const state = inspectEntryState(session);
  assertCompactionInactive(state.unmatchedCompactionStart, state.latestEndSeedSeq, stage);
}

/**
 * Picks a head-anchored span that keeps the recent tail: accumulate tokens from the
 * end until the retain budget is reached, then move the cut point back to a
 * tool-pairing-balanced boundary. Returns null when there is no safe span.
 */
function selectCompactableRange(
  session: Session,
  measurement: MeasurementLike,
  retainTokens: number
): { start: SessionSeq; end: SessionSeq } | null {
  const pricedNodes = measurement.nodes;
  if (pricedNodes.length === 0) return null;
  const surfaceNodes = session.surface.nodes;
  if (surfaceNodes.length !== pricedNodes.length || surfaceNodes.some((seq, index) => seq !== pricedNodes[index]?.seq)) {
    throw new Error('context-guard: token measurement does not match the current surface');
  }
  let accumulated = 0;
  let keepFromIdx = pricedNodes.length;
  for (let index = pricedNodes.length - 1; index >= 0; index -= 1) {
    accumulated += pricedNodes[index]?.tokens ?? 0;
    keepFromIdx = index;
    if (accumulated >= retainTokens) break;
  }
  if (keepFromIdx === 0) return null;
  while (keepFromIdx > 0) {
    const boundary = surfaceNodes[keepFromIdx];
    if (boundary !== undefined && toolPairingBalancedBefore(session, boundary)) break;
    keepFromIdx -= 1;
  }
  if (keepFromIdx === 0) return null;
  const start = surfaceNodes[0];
  const end = surfaceNodes[keepFromIdx - 1];
  if (start === undefined || end === undefined) return null;
  return { start, end };
}

export default ContextGuardEngine;
