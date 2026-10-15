import type { ResolvedConfig, SummaryLanguagePreference } from './config.ts';
import type { GuardLogger } from './utils/logger.ts';
import { GuardLogger as Logger } from './utils/logger.ts';
import type { TokenEstimator } from './utils/tokenizer.ts';

export type SummaryLanguage = 'zh' | 'en';

/** projection of a DSH message, kept free of DSH types so the summarizer stays a pure function */
export interface SimpleMessage {
  /** `developer` (tool add/remove notices) carries no human intent and is projected to `system` */
  readonly role: 'system' | 'user' | 'assistant' | 'tool';
  readonly text: string;
  readonly toolName?: string;
  readonly isError?: boolean;
}

/**
 * `raw` is opaque passthrough (the original DSH Message[]) so the LLM channel can reuse the
 * session prefix and hit the KV cache. Pure logic never reads it, only forwards it.
 */
export interface SummarizationInput {
  readonly system?: string;
  readonly tools?: readonly unknown[];
  /** the compressed span, in surface order */
  readonly messages: readonly SimpleMessage[];
  readonly raw?: unknown;
}

export interface CompletionRequest {
  readonly system?: string;
  readonly tools?: readonly unknown[];
  readonly raw?: unknown;
  readonly instruction: string;
  readonly maxTokens: number;
  readonly signal?: AbortSignal;
}

export interface CompletionResult {
  readonly text: string;
  readonly provider: string;
  readonly model: string;
  /** full output blocks, kept so the compaction/summary event can be rebuilt */
  readonly rawOutput?: readonly { readonly type: string; readonly text?: string }[];
  readonly usage?: unknown;
}

/** injected by the plugin entry from `ctx.llm.stream`; tests pass a stub */
export type TextCompletion = (request: CompletionRequest) => Promise<CompletionResult>;

/** every field holds individually listable facts, never prose */
export interface StructuredSummary {
  readonly goal: string[];
  readonly decisions: string[];
  readonly actions: string[];
  readonly data: string[];
  /** exact-value ledger: paths, URLs, commands, identifiers */
  readonly precise: string[];
  readonly constraints: string[];
  readonly todos: string[];
  readonly openQuestions: string[];
}

/** raw output of the deterministic extractor */
export interface PreciseValues {
  readonly numbers: string[];
  readonly dates: string[];
  readonly paths: string[];
  readonly urls: string[];
  readonly commands: string[];
  /** camelCase / snake_case / dotted.name */
  readonly identifiers: string[];
  readonly constraints: string[];
}

export interface SummaryOutcome {
  readonly blocks: readonly { readonly type: 'text'; readonly text: string }[];
  readonly structured: StructuredSummary;
  readonly language: SummaryLanguage;
  readonly strategy: 'llm' | 'llm+heuristic' | 'heuristic';
  /** empty unless the LLM channel ran (same for `model`) */
  readonly provider: string;
  readonly model: string;
  readonly rawOutput?: readonly { readonly type: string; readonly text?: string }[];
  readonly usage?: unknown;
  readonly tokens: number;
  readonly originalMessageCount: number;
  /** developer-facing log lines */
  readonly notes: readonly string[];
}

// cap per category: the summary must not turn into a second transcript
const PRECISE_LIMIT = 24;

// deterministic extraction rules (the model-free channel)
const EXTRACTORS = {
  url: /https?:\/\/[^\s"'<>)\]}]+/g,
  winPath: /[A-Za-z]:\\(?:[^\\\/:*?"<>|\r\n\s\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]+\\)*[^\\\/:*?"<>|\r\n\s\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]*/g,
  posixPath: /(?<![\w:])\/(?:[\w.@+-]+\/)+[\w.@+-]+/g,
  date: /\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?\b|\b\d{1,2}:\d{2}(?::\d{2})?\b/g,
  number: /(?<![\w.])(?:[$¥€£]\s*)?\d[\d,_]*(?:\.\d+)?\s*(?:%|ms|s|m|h|KB|MB|GB|TB|tokens?|条|行|个|次|元|秒|分钟)?/g,
  command: /(?:^|\n)\s*(?:\$|>|PS [^>]*>)\s*([^\n]+)|\b(?:npm|pnpm|yarn|npx|node|git|docker|kubectl|cargo|go|python|python3|pip|pytest|tsc|make|cmake|gradle|mvn)\s+[\w@./:=+-]+(?:\s+[\w@./:=+-]+)*/g,
  identifier: /\b[a-z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)+\b|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b|\b[a-z][a-zA-Z0-9]*(?:\.[a-zA-Z][a-zA-Z0-9]*){2,}\b/g,
  constraint: /[^\n。！？.!?]*(?:必须|禁止|不要|不得|记得|务必|只能|一定要|must\s+not|must|never|always|should|required|do not|don't)[^\n。！？.!?]*[。！？.!?]?/gi
} as const;

function uniqueLimit(values: Iterable<string>, limit = PRECISE_LIMIT): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of values) {
    const value = raw.trim();
    if (value.length === 0 || value.length > 200) continue;
    if (seen.has(value)) continue;
    seen.add(value);
    out.push(value);
    if (out.length >= limit) break;
  }
  return out;
}

/** last line of defence: these values are merged into the summary even when the model fails */
export function extractPreciseValues(text: string): PreciseValues {
  const matchAll = (pattern: RegExp): string[] => [...text.matchAll(pattern)].map((match) => match[0] ?? '');
  const commands = [...text.matchAll(EXTRACTORS.command)].map((match) => (match[1] ?? match[0] ?? '').trim());
  return {
    numbers: uniqueLimit(matchAll(EXTRACTORS.number).filter((value) => /\d/.test(value))),
    dates: uniqueLimit(matchAll(EXTRACTORS.date)),
    paths: uniqueLimit([...matchAll(EXTRACTORS.winPath), ...matchAll(EXTRACTORS.posixPath)]),
    urls: uniqueLimit(matchAll(EXTRACTORS.url)),
    commands: uniqueLimit(commands),
    identifiers: uniqueLimit(matchAll(EXTRACTORS.identifier)),
    constraints: uniqueLimit(matchAll(EXTRACTORS.constraint).map((value) => value.trim()))
  };
}

export function emptySummary(): StructuredSummary {
  return { goal: [], decisions: [], actions: [], data: [], precise: [], constraints: [], todos: [], openQuestions: [] };
}

/**
 * Rule-based summary, no model call. Lower quality than the LLM channel, but it cannot fail -
 * and a summary that fails leaves the context unable to converge, so the session keeps
 * hitting the window.
 */
export function heuristicExtract(messages: readonly SimpleMessage[], language: SummaryLanguage): StructuredSummary {
  const zh = language === 'zh';
  const goal: string[] = [];
  const actions: string[] = [];
  const constraints: string[] = [];
  const textParts: string[] = [];

  for (const message of messages) {
    textParts.push(message.text);
    if (message.role === 'user') {
      // the first line of a user message is usually the goal; 200 chars keeps the whole request out
      const firstLine = message.text.split(/\r?\n/).find((line) => line.trim().length > 0) ?? '';
      if (firstLine.trim().length > 0) goal.push(firstLine.trim().slice(0, 200));
      constraints.push(...[...message.text.matchAll(EXTRACTORS.constraint)].map((match) => match[0].trim()));
    } else if (message.role === 'tool') {
      const name = message.toolName ?? (zh ? '未知工具' : 'unknown tool');
      const status = message.isError === true ? (zh ? '失败' : 'failed') : (zh ? '成功' : 'ok');
      actions.push(`${name} -> ${status}`);
    } else if (message.role === 'assistant') {
      const line = message.text.split(/\r?\n/).find((candidate) => candidate.trim().length > 0) ?? '';
      if (line.trim().length > 0) actions.push(line.trim().slice(0, 160));
    }
  }

  const precise = extractPreciseValues(textParts.join('\n'));
  return {
    goal: uniqueLimit(goal, 8),
    decisions: [],
    actions: uniqueLimit(actions, 20),
    data: uniqueLimit([...precise.numbers, ...precise.dates], PRECISE_LIMIT),
    precise: uniqueLimit([...precise.paths, ...precise.urls, ...precise.commands, ...precise.identifiers], PRECISE_LIMIT * 2),
    constraints: uniqueLimit([...constraints, ...precise.constraints], 12),
    todos: [],
    openQuestions: []
  };
}

export function detectLanguage(texts: readonly string[], preference: SummaryLanguagePreference): SummaryLanguage {
  if (preference === 'zh' || preference === 'en') return preference;
  let cjk = 0;
  let latin = 0;
  for (const text of texts) {
    for (const char of text) {
      const code = char.codePointAt(0) ?? 0;
      if ((code >= 0x3000 && code <= 0x303f) || (code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3040 && code <= 0x30ff) || (code >= 0xff00 && code <= 0xffef)) cjk += 1;
      else if ((code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a)) latin += 1;
    }
  }
  if (cjk === 0 && latin === 0) return 'en';
  // 0.15, not 0.5: Chinese technical sessions are full of English identifiers
  return cjk / (cjk + latin) >= 0.15 ? 'zh' : 'en';
}

/** strict JSON so every field can be validated and a parse failure can fall back to the heuristic channel */
export function buildSummaryInstruction(language: SummaryLanguage, faithful: boolean): string {
  const schema = '{"goal":[],"decisions":[],"actions":[],"data":[],"precise":[],"constraints":[],"todos":[],"openQuestions":[]}';
  if (language === 'zh') {
    return [
      '你是上下文压缩器。请把上面的对话历史压缩成结构化摘要，供后续推理继续使用。',
      '',
      '硬性要求：',
      '1. 只输出一个 JSON 对象，不要代码块、不要解释、不要在 JSON 前后添加任何文字。',
      '2. 所有字面量必须原样保留，禁止改写或概括：数字、金额、百分比、日期、时间、版本号、',
      '   文件路径、URL、命令、函数/变量/类型名、错误码、配置键。',
      '3. 把"做了什么"写成祈使/陈述短句，不要写"用户询问了……助手回答了……"这类过程描述。',
      '4. 每项不超过 120 字；没有内容的字段用空数组。',
      '5. 保留用户提出的所有约束与偏好（"必须/不要/记得"等），逐条列入 constraints。',
      '',
      faithful ? '6. 精确值优先：宁可条目多，也不要把具体值概括掉。' : '6. 在保证要点不丢失的前提下尽量精简。',
      '',
      'JSON 结构（键名固定）：',
      schema,
      '',
      '字段含义：goal=用户目标与当前任务；decisions=已确定的关键决策及其理由；',
      'actions=已执行的工具调用与结果状态（成功/失败）；data=重要数据与指标；',
      'precise=路径/URL/命令/代码标识符等精确值台账；constraints=用户约束与偏好；',
      'todos=待办事项；openQuestions=尚未解决的问题。'
    ].join('\n');
  }
  return [
    'You are a context compressor. Condense the conversation history above into a structured summary that later reasoning can build on.',
    '',
    'Hard requirements:',
    '1. Output ONE JSON object only. No code fences, no prose before or after it.',
    '2. Preserve every literal verbatim - never paraphrase or generalize: numbers, amounts, percentages,',
    '   dates, times, version numbers, file paths, URLs, commands, function/variable/type names, error codes, config keys.',
    '3. Write what was DONE as terse statements, not "the user asked ... the assistant replied ...".',
    '4. Keep each item under 120 characters; use an empty array when a field has no content.',
    '5. Keep every user constraint and preference (must / never / remember / do not) as its own constraints entry.',
    '',
    faithful ? '6. Precision over brevity: prefer more items to collapsing a concrete value.' : '6. Stay concise while losing no key point.',
    '',
    'JSON shape (fixed keys):',
    schema,
    '',
    'Field meaning: goal=user objective and current task; decisions=settled decisions with reasons;',
    'actions=executed tool calls and their outcome (ok/failed); data=important numbers and metrics;',
    'precise=paths/URLs/commands/identifiers ledger; constraints=user constraints and preferences;',
    'todos=outstanding work; openQuestions=unresolved questions.'
  ].join('\n');
}

function toStringArray(value: unknown, limit = PRECISE_LIMIT): string[] {
  if (typeof value === 'string') return value.trim().length === 0 ? [] : [value.trim()];
  if (!Array.isArray(value)) return [];
  return uniqueLimit(value.filter((item): item is string => typeof item === 'string'), limit);
}

/** tolerant parse: strip fences, then take first `{` to last `}` - models like to add a sentence */
export function parseSummaryJson(raw: string): StructuredSummary | null {
  const fenced = raw.replace(/^\s*```[a-zA-Z]*\s*/gm, '').replace(/```\s*$/gm, '');
  const start = fenced.indexOf('{');
  const end = fenced.lastIndexOf('}');
  if (start === -1 || end <= start) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(fenced.slice(start, end + 1));
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
  const record = parsed as Record<string, unknown>;
  const summary: StructuredSummary = {
    goal: toStringArray(record.goal, 8),
    decisions: toStringArray(record.decisions, 12),
    actions: toStringArray(record.actions, 24),
    data: toStringArray(record.data, PRECISE_LIMIT),
    precise: toStringArray(record.precise, PRECISE_LIMIT * 2),
    constraints: toStringArray(record.constraints, 12),
    todos: toStringArray(record.todos, 12),
    openQuestions: toStringArray(record.openQuestions, 8)
  };
  const total = Object.values(summary).reduce((count, items) => count + items.length, 0);
  return total === 0 ? null : summary;
}

export function mergeSummaries(base: StructuredSummary, extra: StructuredSummary): StructuredSummary {
  return {
    goal: uniqueLimit([...base.goal, ...extra.goal], 10),
    decisions: uniqueLimit([...base.decisions, ...extra.decisions], 16),
    actions: uniqueLimit([...base.actions, ...extra.actions], 32),
    data: uniqueLimit([...base.data, ...extra.data], PRECISE_LIMIT * 2),
    precise: uniqueLimit([...base.precise, ...extra.precise], PRECISE_LIMIT * 4),
    constraints: uniqueLimit([...base.constraints, ...extra.constraints], 16),
    todos: uniqueLimit([...base.todos, ...extra.todos], 16),
    openQuestions: uniqueLimit([...base.openQuestions, ...extra.openQuestions], 10)
  };
}

const HEADINGS: Record<SummaryLanguage, Record<keyof StructuredSummary, string>> = {
  zh: {
    goal: '用户目标',
    decisions: '关键决策',
    actions: '已执行操作',
    data: '重要数据',
    precise: '精确值（路径/URL/命令/标识符）',
    constraints: '约束与偏好',
    todos: '待办事项',
    openQuestions: '未决问题'
  },
  en: {
    goal: 'Goal',
    decisions: 'Key decisions',
    actions: 'Actions taken',
    data: 'Important data',
    precise: 'Exact values (paths/URLs/commands/identifiers)',
    constraints: 'Constraints & preferences',
    todos: 'Open tasks',
    openQuestions: 'Open questions'
  }
};

export function renderStructuredSummary(summary: StructuredSummary, language: SummaryLanguage): string {
  const headings = HEADINGS[language];
  const title = language === 'zh' ? '【已压缩历史摘要】' : '[Compacted history summary]';
  const lines: string[] = [title];
  const order: readonly (keyof StructuredSummary)[] = ['goal', 'decisions', 'actions', 'data', 'precise', 'constraints', 'todos', 'openQuestions'];
  for (const key of order) {
    const items = summary[key];
    if (items.length === 0) continue;
    lines.push(`- ${headings[key]}：`);
    for (const item of items) lines.push(`  - ${item}`);
  }
  if (lines.length === 1) {
    lines.push(language === 'zh' ? '- （无可用内容）' : '- (no recoverable content)');
  }
  return lines.join('\n');
}

/** appended to the summary block as a trailing comment, for consumers and assertions */
export interface SummaryMetadata {
  readonly __compacted: true;
  readonly __originalMessageCount: number;
  readonly __summaryTokens: number;
  readonly __strategy: SummaryOutcome['strategy'];
  readonly __language: SummaryLanguage;
}

// tag follows DSH's built-in compactor so existing consumers keep working; the preamble
// is written in the session language
export function frameSummaryBlocks(
  body: string,
  language: SummaryLanguage,
  metadata: SummaryMetadata
): { type: 'text'; text: string }[] {
  const preamble = language === 'zh'
    ? '这是一段自动生成的检查点摘要，用于压缩更早的对话以释放上下文。请把其中的内容当作已确立的背景直接使用，不要复述，也不要确认这份摘要本身；从后续消息继续任务即可。'
    : 'This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured background as established and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.';
  const metaLine = `<!-- context-guard: ${JSON.stringify(metadata)} -->`;
  return [
    { type: 'text', text: `${preamble}\n\n<compacted-summary>\n${body}\n</compacted-summary>\n${metaLine}` }
  ];
}

// FNV-1a in two independent lanes to reduce collisions; only ever used for the cache key
function hashText(text: string): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    h1 = Math.imul(h1 ^ code, 0x01000193) >>> 0;
    h2 = Math.imul(h2 + code, 0x85ebca6b) >>> 0;
  }
  return `${h1.toString(16)}${h2.toString(16)}`;
}

interface CacheEntry {
  readonly outcome: SummaryOutcome;
}

/**
 * Cache key = language + faithful flag + hash of the message projection. Replays of the same
 * span (session resumed, trigger fired twice) are byte-identical, so a hit skips a full request.
 */
export class HighFidelitySummarizer {
  private readonly config: ResolvedConfig;
  private readonly logger: GuardLogger;
  private readonly completion: TextCompletion | undefined;
  private readonly estimator: TokenEstimator;
  private readonly cache = new Map<string, CacheEntry>();

  constructor(config: ResolvedConfig, logger: GuardLogger, estimator: TokenEstimator, completion?: TextCompletion) {
    this.config = config;
    this.logger = logger;
    this.estimator = estimator;
    this.completion = completion;
  }

  resolveLanguage(input: SummarizationInput): SummaryLanguage {
    const samples: string[] = [];
    for (let index = input.messages.length - 1; index >= 0 && samples.length < 6; index -= 1) {
      const message = input.messages[index];
      if (message !== undefined && message.text.trim().length > 0) samples.push(message.text);
    }
    return detectLanguage(samples, this.config.summaryLanguage);
  }

  private cacheKey(input: SummarizationInput, language: SummaryLanguage): string {
    const payload = input.messages.map((message) => `${message.role}:${message.toolName ?? ''}:${message.isError === true ? 'E' : 'O'}:${message.text}`).join('\u0000');
    return `${language}|${this.config.faithful ? 'f' : 'q'}|${input.messages.length}|${hashText(payload)}`;
  }

  async summarize(input: SummarizationInput, signal?: AbortSignal): Promise<SummaryOutcome> {
    const language = this.resolveLanguage(input);
    const key = this.cacheKey(input, language);
    const cached = this.cache.get(key);
    if (cached !== undefined) {
      // delete + re-set moves the entry to the end (LRU)
      this.cache.delete(key);
      this.cache.set(key, cached);
      this.logger.debug(`摘要缓存命中（${input.messages.length} 条消息）`);
      return cached.outcome;
    }

    const notes: string[] = [];
    const heuristic = heuristicExtract(input.messages, language);
    let structured = heuristic;
    let strategy: SummaryOutcome['strategy'] = 'heuristic';
    let provider = '';
    let model = '';
    let rawOutput: SummaryOutcome['rawOutput'];
    let usage: SummaryOutcome['usage'];

    if (this.completion !== undefined && !(signal?.aborted ?? false)) {
      try {
        const instruction = buildSummaryInstruction(language, this.config.faithful);
        const result = await this.completion({
          ...(input.system === undefined ? {} : { system: input.system }),
          ...(input.tools === undefined ? {} : { tools: input.tools }),
          ...(input.raw === undefined ? {} : { raw: input.raw }),
          instruction,
          maxTokens: this.config.maxSummaryTokens,
          ...(signal === undefined ? {} : { signal })
        });
        provider = result.provider;
        model = result.model;
        rawOutput = result.rawOutput;
        usage = result.usage;
        const parsed = parseSummaryJson(result.text);
        if (parsed !== null) {
          structured = mergeSummaries(parsed, heuristic);
          strategy = 'llm';
        } else {
          // not JSON: keep the heuristic ledger and append the model text as a note instead
          structured = mergeSummaries(heuristic, {
            ...emptySummary(),
            decisions: this.config.faithful ? [result.text.trim().slice(0, 1200)] : []
          });
          strategy = 'llm+heuristic';
          notes.push('model output was not valid JSON, kept the deterministic ledger plus the raw text');
        }
      } catch (error) {
        strategy = 'heuristic';
        notes.push(`model summarization failed, fell back to the deterministic channel: ${Logger.describe(error)}`);
        this.logger.warn(notes[notes.length - 1] ?? 'model summarization failed');
      }
    } else if (this.completion === undefined) {
      notes.push('no LLM channel injected, using the deterministic channel');
    }

    const body = renderStructuredSummary(structured, language);
    const metadata: SummaryMetadata = {
      __compacted: true,
      __originalMessageCount: input.messages.length,
      __summaryTokens: this.estimator.estimate(body),
      __strategy: strategy,
      __language: language
    };
    const outcome: SummaryOutcome = {
      blocks: frameSummaryBlocks(body, language, metadata),
      structured,
      language,
      strategy,
      provider,
      model,
      ...(rawOutput === undefined ? {} : { rawOutput }),
      ...(usage === undefined ? {} : { usage }),
      tokens: metadata.__summaryTokens,
      originalMessageCount: input.messages.length,
      notes
    };

    if (this.config.summaryCacheSize > 0) {
      this.cache.set(key, { outcome });
      while (this.cache.size > this.config.summaryCacheSize) {
        const oldest = this.cache.keys().next();
        if (oldest.done) break;
        this.cache.delete(oldest.value);
      }
    }
    return outcome;
  }

  get cacheSize(): number {
    return this.cache.size;
  }

  clearCache(): void {
    this.cache.clear();
  }
}
