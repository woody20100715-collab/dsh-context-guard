# dsh-context-guard

给 DSH（DeepSeek Harness）用的上下文守卫：**过长的工具输出按内容类型压掉，过长的历史换成高保真摘要，被压掉的内容随时能原样取回。**

上下文一长，模型就开始犯迷糊：注意力被稀释，关键数字被噪声淹没，一次目录列表或构建日志几秒钟就能吃掉几万 token。这个插件在三个位置分别下手：

| 时机 | 做什么 |
| --- | --- |
| 工具输出进历史之前 | 判断内容类型（JSON / 日志 / 表格 / 长文本）分别压缩；压缩率不到 15% 就放弃，不值得为几个 token 让模型多跑一次取回 |
| 历史快撑满窗口时 | 换成结构化摘要，数字、路径、命令、用户约束逐字保留 |
| 每次模型调用之前 | 算 token 预算，只裁插件自己注入的上下文；用户消息和工具结果一律不动 |

被压掉的原文不会丢。原文存进本地的 CCR，压缩结果末尾带一行脚注：

```
[context-guard] compressed log output: ~16036 -> ~620 tokens (-96%), strategy=log:fold+prioritize.
The full original is available: call context_retrieve with id="ccr_3b25d97745ae4930" ...
```

模型需要细节时按 id 取回，**取回的是逐字原文，不会再被压一次**（超长的话按 `offset` 分页）。

## 安装

```powershell
git clone https://github.com/woody20100715-collab/dsh-context-guard.git
cd dsh-context-guard
npm install
npm run build

# 装进目标 profile（web / headless / 你自己建的都行）
dsh plugin --profile web add .
```

`dsh plugin add` 会做三件事：把包装进 profile、发现本包声明了 `dsh.bundle.patch` 于是自动登记成 bundle 层、再对运行时版本做一次兼容性校验（版本对不上会直接拒绝安装）。

插件自带的补丁会关掉内置的 `compaction-basic` 并插入自己——`ctx.compaction` 是单例服务，两个压缩后端不能同时挂。装完可以确认一下：

```powershell
dsh --profile web --dump-config | Select-String 'compaction-basic|context-guard' -Context 1
```

> GUI 用的 `desktop` profile 由应用独占，独立 CLI 不允许改。要在 GUI 里用，请走应用内的插件管理界面，或者**完全退出应用之后**再执行 `dsh plugin --profile desktop add .`。

## 配置

全部写在 `cordis.patch.yml` 的 `config` 里，所有键都可以省略。

### 触发

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 总开关 |
| `auto` | `true` | 自动压力压缩 + 上下文溢出恢复 |
| `thresholdRatio` | `0.8` | 已用 token 占窗口比例达到多少就压历史 |
| `retainRatio` | `0.16` | 最近这段保留为逐字原文的比例，必须小于 `thresholdRatio` |
| `retainTokens` | `0` | >0 时用绝对 token 数覆盖 `retainRatio` |
| `maxSummaryTokens` | `2048` | 摘要生成上限 |
| `compactionRetries` | `2` | 压一次没降到阈值以下时的重试次数 |
| `maxOverflowRetries` | `2` | 撞到上下文窗口上限后的重试次数 |

### 摘要

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `faithful` | `true` | 高保真：数字、路径、命令、约束逐字保留 |
| `summaryLanguage` | `auto` | `auto` 时按会话里中文占比决定摘要语言 |
| `summaryCacheSize` | `32` | 相同历史块的摘要缓存条数 |
| `summarizationProvider` / `summarizationModel` | `''` | 指定摘要用的模型；留空则跟随会话当前路由 |

### 工具输出压缩

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `enableToolCompression` | `true` | 开关 |
| `maxToolOutputTokens` | `2000` | 超过这个估算值才压 |
| `minToolOutputTokens` | `400` | 低于这个值不折腾 |
| `targetToolOutputTokens` | `1100` | 压缩目标，压不到就继续加严 |
| `jsonMaxStringChars` | `400` | JSON 里单个字符串值的截断长度 |
| `jsonHeadItems` / `jsonTailItems` | `20` / `5` | 数组保留的头尾元素数 |
| `logHeadLines` / `logTailLines` | `15` / `20` | 日志固定保留的首尾行数 |
| `logMaxIssues` | `40` | 最多保留多少条 ERROR / WARN |
| `logContextLines` | `3` | 错误行后面跟几行堆栈 |
| `tableHeadRows` / `tableTailRows` | `15` / `10` | 表格保留的首尾行数 |
| `textHeadChars` / `textTailChars` | `1200` / `800` | 长文本保留的首尾字符数 |
| `textExtractSignals` | `true` | 从被省略的中段里挑出含路径 / 命令 / 数字的句子 |

### 可逆取回

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `enableCCR` | `true` | 关掉就不可取回了，脚注会写明 |
| `ccrMaxEntries` | `100` | 条目上限，超出按 LRU 淘汰 |
| `ccrMaxBytes` | `10485760` | 总字节上限 |
| `ccrTtlMs` | `3600000` | 存活时间 |
| `ccrPersist` | `true` | 是否落盘（默认写到 `~/.dsh/context-guard/ccr`） |
| `ccrStorePath` | `~/.dsh/context-guard/ccr` | 落盘目录 |
| `retrieveMaxChars` | `60000` | 单次取回最多返回多少字符，更长的用 `offset` 分页 |

### 其它

`assemblyEnabled`、`noticeEnabled`、`reservedOutputTokens`、`systemPromptTokenFallback`、`logLevel`。

配置写错会在加载时直接报错，比如：

```
dsh-context-guard config: retainRatio — must be smaller than thresholdRatio (0.8 >= 0.8)
```

## 压缩策略

**JSON**：先去掉多余空白；还超预算就截断长字符串（留首尾）；再超就抽样数组，但会留下 `__context_guard_omitted_items` 说明省略了多少条。键名一律不缩写——缩写省不了几个 token，却会让模型猜错字段含义。

**日志**：连续重复的行折叠成 `… (x214)`，ERROR / WARN 连同后面的堆栈整行保留（含时间戳），首尾固定保留。

**表格**：Markdown 表格和 CSV / TSV 都认，保留表头 + 首尾行，中间写 `... (N rows omitted) ...`。

**长文本**：留首尾，中段按"路径 / 命令 / 数字 / 约束词 / 代码特征"挑几句高信号的留下。

**历史摘要**走双通道：模型按固定 JSON 结构产出摘要；同时另一条正则通道把数字、金额、日期、路径、URL、命令、标识符、约束句提取出来，无条件并进最终摘要。就算模型那次调用完全失败，这条通道也保证关键事实不丢——这就是 `faithful: true` 的实际含义。

摘要与替换消息严格遵守 DSH 的 durable 协议（`compaction/start` → `compaction/summary` → 紧邻的替换 `user/message` → `compaction/end`），并且有一道收缩门限：摘要没比被替换内容更小就直接失败，避免越压越长。

## 开发

```powershell
npm test          # 52 个测试
npm run typecheck # tsc --noEmit
npm run build     # 产出 dist/
npm run dev       # dsh --patch ./cordis.patch.yml
```

测试里有 5 个跑在**真实 DSH 运行时**上（真 cordis、真 ToolRuntime、真 Session、真 CompactionEngine）：插件挂载、工具输出压缩与取回往返、durable 事件顺序、收缩门限失败路径、pre-step 提示注入。`@deepseek-ai/*` 装不上时会自动跳过这几个用例并说明原因，所以离线环境 `npm test` 仍然可用。

源码用 `.ts` 后缀互相 import，`node --test` 可以直接跑源码；构建时由 `rewriteRelativeImportExtensions` 改写成 `.js`。因此需要 Node 22.6+（类型剥离）和 TypeScript 5.8+。

版本要对齐：本项目对的是 `@deepseek-ai/dsh-*@0.2.0-rc.2` 与 `cordis@4.0.4`。peer 版本写错的话，安装时会被 DSH 的兼容性闸门拦下来。

## 已知限制

1. 动态组装层不能重排历史——`agent/pre-step` 只看得到本步骤的 inbox 消息，完整历史归压缩层管。想更省窗口就调 `thresholdRatio` / `retainRatio`。
2. 摘要质量取决于模型。成本敏感的话用 `summarizationProvider` / `summarizationModel` 指一个便宜的路由。
3. CCR 是进程内缓存 + 可选落盘，不是分布式存储。
4. 取回上限是 `retrieveMaxChars`，更长的原文需要模型自己分页。
5. 把 `faithful` 关掉会明显更省 token，但模型更容易在被省略的细节上编。

## License

MIT
