import type { ResolvedConfig } from './config.ts';
import { MIN_SAVING_RATIO } from './compressors/index.ts';
import type { GuardContentBlock } from './compressors/index.ts';
import type { SummaryLanguage } from './summarizer.ts';
import type { TokenEstimator } from './utils/tokenizer.ts';

// 0 = current user message and non-plugin context, never dropped; 1 = failed tool results;
// 2 = ordinary tool results and assistant messages; 3 = already-compressed output and
// plugin-injected bulk, trimmed first
export type MessagePriority = 0 | 1 | 2 | 3;

export interface AssemblyMessage {
  readonly id?: string;
  readonly role: 'system' | 'user' | 'assistant' | 'tool';
  readonly tokens: number;
  readonly priority: MessagePriority;
  readonly compressed: boolean;
  /**
   * Only plugin-injected context may be dropped. Dropping a tool result breaks the assistant's
   * tool-call/tool-result pairing and the provider rejects the whole request, which is a worse
   * failure than an oversized context.
   */
  readonly droppable: boolean;
  readonly label: string;
}

export interface BudgetInput {
  readonly contextWindow?: number;
  readonly usedTokens: number;
  readonly systemTokens: number;
  readonly reservedOutputTokens: number;
  readonly fallbackContextWindow: number;
}

export interface Budget {
  readonly contextWindow: number;
  /** true when the window came from the fallback instead of the session */
  readonly degraded: boolean;
  readonly available: number;
  readonly ratio: number;
  readonly overBy: number;
  readonly overThreshold: boolean;
}

export function computeBudget(input: BudgetInput, thresholdRatio: number): Budget {
  const degraded = input.contextWindow === undefined || input.contextWindow <= 0;
  const contextWindow = degraded ? input.fallbackContextWindow : (input.contextWindow ?? input.fallbackContextWindow);
  const overhead = input.systemTokens + input.reservedOutputTokens;
  const available = contextWindow - overhead - input.usedTokens;
  const ratio = contextWindow <= 0 ? 1 : input.usedTokens / contextWindow;
  return {
    contextWindow,
    degraded,
    available,
    ratio,
    overBy: available < 0 ? -available : 0,
    overThreshold: ratio >= thresholdRatio
  };
}

export interface AssemblyPlan {
  readonly keep: AssemblyMessage[];
  readonly dropped: AssemblyMessage[];
  readonly trimmed: string[];
  readonly freedTokens: number;
  /** even after dropping every droppable message */
  readonly stillOverBudget: boolean;
}

// Evict by priority, not by age: a 50k-token plugin injection can matter less than the newest
// tool output, which is usually what the current step is reasoning about.
export function planAssembly(messages: readonly AssemblyMessage[], availableTokens: number): AssemblyPlan {
  const total = messages.reduce((sum, message) => sum + message.tokens, 0);
  if (total <= availableTokens) {
    return { keep: [...messages], dropped: [], trimmed: [], freedTokens: 0, stillOverBudget: false };
  }
  const keep = new Set<number>();
  const droppableIndexes: number[] = [];
  let used = 0;
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (message === undefined) continue;
    if (!message.droppable) {
      keep.add(index);
      used += message.tokens;
      continue;
    }
    droppableIndexes.push(index);
  }
  // droppable plugin context goes lowest-priority-first, i.e. priority 3 out first
  const ordered = [...droppableIndexes].sort((a, b) => {
    const left = messages[a];
    const right = messages[b];
    const priorityDiff = (right?.priority ?? 0) - (left?.priority ?? 0);
    return priorityDiff !== 0 ? priorityDiff : b - a;
  });
  const droppedIndexes: number[] = [];
  for (const index of ordered) {
    const message = messages[index];
    if (message === undefined) continue;
    if (used + message.tokens <= availableTokens) {
      keep.add(index);
      used += message.tokens;
    } else {
      droppedIndexes.push(index);
    }
  }
  const kept = messages.filter((_message, index) => keep.has(index));
  const dropped = messages.filter((_message, index) => droppedIndexes.includes(index));
  return {
    keep: kept,
    dropped,
    trimmed: [],
    freedTokens: total - used,
    stillOverBudget: used > availableTokens
  };
}

export interface CompactionState {
  readonly compactions: number;
  readonly lastShadowedMessages: number;
  readonly lastOriginalTokens: number;
  readonly lastSummaryTokens: number;
  /** entries currently retrievable through CCR */
  readonly retrievableEntries: number;
}

export function emptyCompactionState(): CompactionState {
  return { compactions: 0, lastShadowedMessages: 0, lastOriginalTokens: 0, lastSummaryTokens: 0, retrievableEntries: 0 };
}

// Telling the model that output was compressed and can be retrieved is what makes compression
// reversible. Static, so it stays in the system prompt and keeps the request prefix (KV cache).
export function renderCapabilityNotice(language: SummaryLanguage, enableCCR: boolean): string {
  if (language === 'zh') {
    const lines = [
      '## 上下文守卫（dsh-context-guard）',
      '',
      '本会话启用了分层上下文压缩：过长的工具输出会被按内容类型压缩，过长的历史会被替换为高保真摘要。',
      '压缩后的工具输出末尾会带有形如 `[context-guard] compressed ... id="ccr_xxxx"` 的脚注。'
    ];
    if (enableCCR) {
      lines.push(
        '',
        '当你需要被省略的原始细节（完整日志、完整 JSON、完整文件内容）时，调用 `context_retrieve` 并传入脚注里的 id 取回原文。',
        '不要凭压缩后的片段猜测被省略的内容；不确定时先取回。',
        '压缩摘要中的数字、路径、命令等精确值均已原样保留，可直接引用。'
      );
    } else {
      lines.push('', '注意：可逆取回（CCR）未启用，被省略的内容无法恢复；需要细节时请重新执行相应工具。');
    }
    return lines.join('\n');
  }
  const lines = [
    '## Context guard (dsh-context-guard)',
    '',
    'This session runs layered context compression: oversized tool output is compressed by content type, and long history is replaced by a high-fidelity summary.',
    'Compressed tool output ends with a footer such as `[context-guard] compressed ... id="ccr_xxxx"`.'
  ];
  if (enableCCR) {
    lines.push(
      '',
      'When you need omitted detail (full log, full JSON, full file content), call `context_retrieve` with the id from that footer to restore the original.',
      'Do not guess omitted content from the compressed excerpt; retrieve it first when unsure.',
      'Numbers, paths and commands in compacted summaries are preserved verbatim and can be quoted directly.'
    );
  } else {
    lines.push('', 'Note: reversible retrieval (CCR) is disabled, so omitted content cannot be restored; re-run the tool when you need detail.');
  }
  return lines.join('\n');
}

// Unlike the capability notice this is injected only when something changed (a compaction just
// happened, or the budget is spent); injecting every step would pollute the request prefix.
export function renderStatusNotice(state: CompactionState, budget: Budget, language: SummaryLanguage): string | null {
  if (state.compactions === 0 && budget.available > 0) return null;
  if (language === 'zh') {
    const parts: string[] = ['[context-guard] 上下文状态：'];
    if (state.compactions > 0) {
      parts.push(
        `已压缩 ${state.compactions} 次历史（最近一次将 ${state.lastShadowedMessages} 条消息、约 ${state.lastOriginalTokens} tokens 压缩为约 ${state.lastSummaryTokens} tokens）。`
      );
      if (state.retrievableEntries > 0) {
        parts.push(`当前有 ${state.retrievableEntries} 份原始工具输出可通过 context_retrieve 取回。`);
      }
    }
    if (budget.available <= 0) {
      parts.push(`注意：剩余预算已耗尽（窗口 ${budget.contextWindow}，已用约 ${budget.ratio * 100}%），请优先完成当前步骤，避免继续产出大段输出。`);
    }
    return parts.join('');
  }
  const parts: string[] = ['[context-guard] context status:'];
  if (state.compactions > 0) {
    parts.push(
      `compacted history ${state.compactions}x (latest replaced ${state.lastShadowedMessages} messages, ~${state.lastOriginalTokens} tokens -> ~${state.lastSummaryTokens} tokens).`
    );
    if (state.retrievableEntries > 0) {
      parts.push(`${state.retrievableEntries} original tool outputs are retrievable via context_retrieve.`);
    }
  }
  if (budget.available <= 0) {
    parts.push(`Budget exhausted (window ${budget.contextWindow}, ~${budget.ratio * 100}% used); finish the current step instead of producing more large output.`);
  }
  return parts.join(' ');
}

export function isCompressedBlocks(blocks: readonly GuardContentBlock[]): boolean {
  return blocks.some((block) => typeof block.text === 'string' && block.text.includes('[context-guard] compressed'));
}

export function extractOriginalId(text: string): string | undefined {
  const match = /id="(ccr_[0-9a-f]{16})"/.exec(text);
  return match?.[1];
}

/**
 * Pure computation - DSH interaction (tokenMeter reads, UserMessage construction, the
 * PreStepDecision) stays in the plugin entry. Note `agent/pre-step` payload.messages holds only
 * the inbox messages pulled for this step, not the full history: the session surface owns that,
 * and the compactor has already run there.
 */
export class ContextAssembler {
  private readonly config: ResolvedConfig;
  private readonly estimator: TokenEstimator;

  constructor(config: ResolvedConfig, estimator: TokenEstimator) {
    this.config = config;
    this.estimator = estimator;
  }

  budget(input: Omit<BudgetInput, 'reservedOutputTokens' | 'fallbackContextWindow'> & Partial<BudgetInput>): Budget {
    return computeBudget(
      {
        ...input,
        reservedOutputTokens: input.reservedOutputTokens ?? this.config.reservedOutputTokens,
        fallbackContextWindow: input.fallbackContextWindow ?? this.config.systemPromptTokenFallback * 16
      },
      this.config.thresholdRatio
    );
  }

  plan(messages: readonly AssemblyMessage[], availableTokens: number): AssemblyPlan {
    return planAssembly(messages, availableTokens);
  }

  capabilityNotice(language: SummaryLanguage): string {
    return renderCapabilityNotice(language, this.config.enableCCR);
  }

  statusNotice(state: CompactionState, budget: Budget, language: SummaryLanguage): string | null {
    if (!this.config.noticeEnabled) return null;
    return renderStatusNotice(state, budget, language);
  }

  estimate(text: string): number {
    return this.estimator.estimate(text);
  }

  /** don't bother the model with a notice for a marginal saving */
  isWorthNotice(originalTokens: number, compressedTokens: number): boolean {
    if (originalTokens <= 0) return false;
    return (originalTokens - compressedTokens) / originalTokens >= MIN_SAVING_RATIO;
  }
}
