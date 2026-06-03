# 设计文档审查报告

> 审查日期: 2026-04-16
> 审查范围: docs/ 目录下所有设计文档、计划文档、规格文档、bug 报告
> 文档总数: ~110 份

> **2026-04-17 复核说明:** 这份报告记录的是 2026-04-16 的一次性快照，不再代表当前完整状态。2026-04-17 已再次对照代码复核，并已批量修正文档中的 skill 名称、backend/runner 路径、`orchestration`/`run_loop` 命名漂移，以及若干硬编码本机路径。下面保留原始审查内容作为历史记录；凡仍保留旧名称、`--continue`、`.triton-agent/roles/*` 或 `skills/optimize-supervisor/` 的文档，优先把它们理解为历史设计/计划快照，而不是当前实现契约。

## 一、总体评估

项目文档体系完整，覆盖了从早期 CLI 设计到最新 skill 重命名的完整演进过程。但正因为经历了多次重构（后端包重构、模块重命名、skill 统一命名等），大量早期文档中的路径、类名、模块名引用已与当前代码不一致。

**文档分类统计：**

| 类别 | 数量 | 状态 |
|---|---|---|
| 已实现且与代码一致 | ~25 | 无需更新 |
| 已实现但引用过时 | ~55 | 需更新引用 |
| 部分实现/未实现 | ~5 | 需关注 |
| 已被后续文档取代 | ~20 | 可归档 |
| bug 报告 | 2 | 已修复，可归档 |

---

## 二、核心不一致问题（跨文档共性问题）

### 1. Skill 目录名称已全部重命名

2026-04-16 的 skill 重命名计划将所有 skill 统一为 `triton-npu-*` 前缀。大量早期文档仍使用旧名称：

| 旧名称 | 新名称 | 涉及文档数 |
|---|---|---|
| `test-gen` | `triton-npu-gen-test` | ~10 |
| `bench-gen` | `triton-npu-gen-bench` | ~10 |
| `eval-gen` | `triton-npu-gen-eval-suite` | ~8 |
| `operator-eval` | `triton-npu-run-eval` | ~12 |
| `optimize` | `triton-npu-optimize` | ~15 |
| `optimize-check` | `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` | ~4 |
| `ascend-npu-operator-profiler` | `triton-npu-profile-operator` | ~5 |
| `ascend-operator-ir-analyzer` | `triton-npu-analyze-ir` | ~4 |
| `triton-repair-experience` | `triton-npu-repair-guide` | ~2 |
| `optimize-supervisor` | (已删除) | ~4 |

### 2. 后端 Runner 模块已迁移到 `backends/` 包

早期文档引用的顶层 runner 模块已全部移入 `src/triton_agent/backends/` 包：

| 旧路径 | 新路径 | 涉及文档数 |
|---|---|---|
| `src/triton_agent/agent.py` | (已删除) | ~5 |
| `src/triton_agent/runner_factory.py` | `backends/factory.py` | ~3 |
| `src/triton_agent/codex_runner.py` | `backends/codex.py` | ~8 |
| `src/triton_agent/opencode_runner.py` | `backends/opencode.py` | ~5 |
| `src/triton_agent/pi_runner.py` | `backends/pi.py` | ~5 |
| `src/triton_agent/claude_runner.py` | `backends/claude.py` | ~4 |

### 3. Optimize 模块重命名

| 旧名称 | 新名称 | 涉及文档数 |
|---|---|---|
| `OptimizeSupervisor` (类) | `OptimizeRunLoop` | ~10 |
| `optimize/supervisor.py` | `optimize/run_loop.py` | ~8 |
| `optimize/runtime.py` | `optimize/orchestration.py` | ~10 |
| `generation/runtime.py` | `generation/orchestration.py` | ~3 |
| `OptimizeController` (类) | `OptimizeRunLoop` | ~3 |
| `SupervisedRoundRunner` | `SupervisedOptimizeAdapter` | ~3 |
| `RunnerWithStreams` | `RecoveryRunnerAdapter` | ~3 |
| `optimize/gate.py` | (未创建，逻辑在 `execution.py` 和 `contract.py`) | ~5 |

### 4. `--continue` 标志已被 `--resume` 取代

早期文档（2026-04-02 ~ 2026-04-07）引用 `--continue` 标志，已被 `--resume {auto,continue,fresh}` 取代。涉及约 5 份文档。

### 5. `.triton-agent/roles/` 角色文件已不再使用

supervisor round-gate 相关文档引用 `.triton-agent/roles/optimize-worker.md` 和 `.triton-agent/roles/optimize-supervisor.md`，这些文件已不再创建，supervisor 行为已整合到 prompts 中。涉及约 5 份文档。

### 6. `optimize-supervisor` skill 已删除

多份文档引用 `skills/optimize-supervisor/` 目录，该 skill 已在 2026-04-16 的重命名计划中删除。涉及约 4 份文档。

---

## 三、逐文档详细审查

### docs/ 目录（顶层设计文档）

| 文档 | 主题 | 状态 | 主要问题 |
|---|---|---|---|
| `2026-03-31-triton-agent-cli.md` | 原始 CLI 综合设计 | **过时** | 仅列出 5 个子命令（当前 11 个）；仅提及 codex 后端（当前 5 个）；描述 symlink 模式（已改为 copy）；引用旧 skill 名称 |
| `2026-03-31-opencode-backend.md` | 添加 opencode 后端 | **部分过时** | 称为"第二个后端"（当前 5 个）；核心实现描述仍准确 |
| `2026-03-31-skill-command-script.md` | 添加 run-command.py 辅助脚本 | **路径过时** | 引用 `skills/run-validation/`（已改名为 `skills/triton-npu-run-eval/`） |
| `2026-03-31-skill-link-idempotent-symlinks.md` | Skill symlink 幂等性 | **已取代** | 被 `skill-copy-staging.md` 取代，symlink 方案已废弃 |
| `2026-03-31-verbose-output-formatting.md` | 详细输出格式 | **部分过时** | 提及 "skill links" 的 `a -> b` 显示，但当前使用 copy 而非 symlink |
| `2026-03-31-generation-overwrite-control.md` | gen-test/gen-bench --force-overwrite | **不完整** | 未提及 `gen-eval` 也支持 `--force-overwrite` |
| `2026-03-31-optimize-skill-redesign.md` | Optimize skill 重设计 | **路径过时** | 引用 `skills/optimize`（已改名为 `skills/triton-npu-optimize`）；引用 `quick_validate.py` 的硬编码用户路径 |
| `2026-03-31-optimize-workspace-agents.md` | 工作区 AGENTS.md 写入 | **部分过时** | 未提及 openhands 后端 |
| `2026-03-31-subcommand-snake-case-aliases.md` | 子命令 snake_case 别名 | **不完整** | 仅提及 4 个别名，当前 CLI 有更多子命令需要别名 |
| `2026-04-02-skill-copy-staging.md` | Skill copy 暂存 | **部分过时** | 仅提及 codex 和 opencode 后端，未提及 claude/pi/openhands |
| `2026-04-02-unified-run-skill.md` | 统一运行 skill | **路径过时** | 引用 `skills/run-validation/`（已改名为 `skills/triton-npu-run-eval/`） |
| `2026-04-02-optimize-continue-mode.md` | --continue 模式 | **已取代** | `--continue` 已被 `--resume {auto,continue,fresh}` 取代 |
| `2026-04-02-optimize-min-rounds.md` | --min-rounds | **类名过时** | 引用 `OptimizeSupervisor`（已改名为 `OptimizeRunLoop`） |
| `2026-04-02-ascend-npu-operator-profiler-skill.md` | Profiler skill | **路径过时** | 引用 `quick_validate.py` 硬编码路径；引用 `benchmark_analyzer.py`（已删除） |
| `2026-04-03-claude-backend.md` | Claude 后端 | **部分过时** | 称为"第四个后端"（当前 5 个）；未提及 openhands |
| `2026-04-03-pi-backend.md` | Pi 后端 | **部分过时** | 称为"第三个后端"（当前 5 个） |
| `2026-04-03-optimize-no-agent-session.md` | --no-agent-session | **不完整** | 未提及 Claude 的 `--no-session-persistence` |
| `2026-04-07-cli-execution-comparison-refactor.md` | 执行/比较命令重构 | **部分过时** | 提议创建 `src/triton_agent/comparison.py`（从未创建）；比较逻辑后来被扁平化到 `commands/comparison.py` |
| `2026-04-07-cli-optimize-refactor-layering.md` | Optimize 分层重构 | **多处过时** | 提议 `optimize/runtime.py`（已改名为 `orchestration.py`）；提议 `cli_parser.py`/`cli_dispatch.py`（未创建）；引用 `OptimizeSupervisor`（已改名） |
| `2026-04-07-optimize-graceful-interrupt.md` | 优雅中断处理 | **类名/路径过时** | 引用 `OptimizeSupervisor`；引用顶层 runner 模块（已迁移到 `backends/`） |
| `2026-04-07-optimize-status-subcommand.md` | optimize-status 子命令 | **部分过时** | Markdown 表格示例使用中文表头"名称"，与英文优先约定不一致 |
| `2026-04-08-inspect-ir-script.md` | IR 检查脚本 | **路径过时** | 引用 `skills/ascend-operator-ir-analyzer/`（已改名为 `skills/triton-npu-analyze-ir/`）；引用 `quick_validate.py` 硬编码路径 |
| `2026-04-08-ir-dir-flag-alignment.md` | --ir-dir 标志对齐 | **路径过时** | 引用 `skills/ascend-operator-ir-analyzer/`（已改名）；引用 `quick_validate.py` 硬编码路径 |
| `2026-04-08-inspect-ir-ranking-and-change-scan.md` | IR 排名和变更扫描 | **路径过时** | 引用 `skills/ascend-operator-ir-analyzer/`（已改名） |
| `2026-04-08-opt-note-overall-summary.md` | opt-note 总体摘要 | **路径过时** | 引用 `skills/optimize/references/`（已改名为 `skills/triton-npu-optimize/references/`） |
| `2026-04-08-optimize-artifact-directories.md` | 优化产物目录 | **可能过时** | 称 `optimize-status` 仅依赖 `perf.txt`，可能已扩展 |
| `2026-04-08-optimize-round-failure-handling-name.md` | 重命名 contracts.md | **已实现** | 无明显问题 |
| `2026-04-09-gen-eval-force-overwrite.md` | gen-eval --force-overwrite | **与代码一致** | 无明显问题 |
| `2026-03-31-bench-mode-selection.md` | --bench-mode 选择 | **与代码一致** | 无明显问题 |
| `2026-03-31-codex-ephemeral-git-check-flags.md` | Codex ephemeral 标志 | **与代码一致** | 无明显问题 |
| `2026-03-31-default-mode-values.md` | 默认模式值 | **与代码一致** | 无明显问题 |
| `2026-03-31-friendly-cli-errors.md` | 友好 CLI 错误 | **与代码一致** | 无明显问题 |
| `2026-03-31-optimize-pattern-directory.md` | Pattern 目录结构 | **与代码一致** | 无明显问题 |
| `2026-03-31-optimize-pattern-expansion.md` | Pattern 扩展 | **与代码一致** | 无明显问题 |
| `2026-03-31-optimize-pattern-index.md` | Pattern 索引 | **与代码一致** | 无明显问题 |
| `2026-03-31-optimize-round-attempt-log.md` | attempts.md 日志 | **与代码一致** | 无明显问题 |
| `2026-03-31-process-runner-extraction.md` | 进程运行器提取 | **与代码一致** | 无明显问题 |
| `2026-03-31-show-output-streaming.md` | --show-output 流式 | **与代码一致** | 无明显问题 |
| `2026-03-31-streaming-pty-exit-cleanup.md` | PTY 退出清理 | **与代码一致** | 无明显问题 |
| `2026-03-31-test-mode-selection.md` | --test-mode 选择 | **与代码一致** | 无明显问题 |
| `2026-03-31-skill-natural-language-interfaces.md` | Skill 自然语言接口 | **与代码一致** | 风格指南，无代码引用 |
| `2026-03-31-unified-process-runner-modes.md` | 统一进程运行模式 | **与代码一致** | 无明显问题 |
| `2026-04-01-compare-perf-subcommand.md` | compare-perf 子命令 | **与代码一致** | 无明显问题 |
| `2026-04-01-generated-harness-metadata.md` | 生成 harness 元数据 | **不完整** | 后续文档扩展了 `# kernel:` 等元数据，此文档不完整但不矛盾 |
| `2026-04-01-local-run-bench.md` | 本地 run-bench | **与代码一致** | 无明显问题 |
| `2026-04-01-local-run-test-and-compare-result.md` | 本地 run-test/compare-result | **与代码一致** | 无明显问题 |
| `2026-04-01-remote-agent-command-context.md` | 远程 agent 命令 | **不完整** | 未提及 `gen-eval`/`gen-eval-batch` 也支持远程；未提及 openhands 后端 |
| `2026-04-01-remote-run-commands.md` | 远程运行命令 | **与代码一致** | 无明显问题 |
| `2026-04-01-remove-run-skills.md` | 删除运行 skill | **已实现** | `skills/test-run/` 和 `skills/bench-run/` 已删除 |
| `2026-04-02-bugfix-regressions.md` | Bug 修复回归 | **已修复** | 引用的 bug 已修复 |
| `2026-04-02-generated-harness-pytorch-entrypoints.md` | PyTorch 入口点 | **与代码一致** | 无明显问题 |
| `2026-04-02-msprof-parse-bin-no-tabulate.md` | 移除 tabulate 依赖 | **与代码一致** | 无明显问题 |
| `2026-04-02-workspace-placeholder-exclusion.md` | workspace 排除规则 | **与代码一致** | 配置指南 |
| `2026-04-03-batch-optimize-subcommand.md` | optimize-batch 子命令 | **与代码一致** | 无明显问题 |
| `2026-04-03-pyright-strict-src-scope.md` | Pyright 严格模式 | **与代码一致** | 配置指南 |
| `2026-04-03-remote-profiler-support.md` | 远程 profiler | **与代码一致** | 无明显问题 |
| `2026-04-07-cli-generation-refactor.md` | 生成命令重构 | **部分过时** | 提议 `generation.py` 扁平模块（已重构为 `generation/` 包） |
| `2026-04-07-batch-optimize-prefixed-show-output.md` | 批量优化前缀输出 | **与代码一致** | 无明显问题 |
| `2026-04-07-ascend-operator-ir-analyzer-skill.md` | IR 分析 skill | **路径过时** | 引用 `quick_validate.py` 硬编码路径 |

### docs/plans/ 目录（实施计划）

| 文档 | 主题 | 状态 | 主要问题 |
|---|---|---|---|
| `2026-04-02-optimize-continue-mode.md` | --continue 模式 | **已取代** | `--continue` 已被 `--resume` 取代 |
| `2026-04-03-pi-backend.md` | Pi 后端 | **路径过时** | 引用 `src/triton_agent/pi_runner.py`（已迁移到 `backends/pi.py`） |
| `2026-04-03-remote-profiler-support.md` | 远程 profiler | **路径过时** | 引用 `skills/run-validation/`、`skills/ascend-npu-operator-profiler/`、`skills/optimize/`（均已改名） |
| `2026-04-07-ascend-operator-ir-analyzer-skill.md` | IR 分析 skill | **路径过时** | 引用 `skills/ascend-operator-ir-analyzer/`（已改名）；引用 `quick_validate.py` 硬编码路径 |
| `2026-04-07-cli-optimize-refactor-layering.md` | Optimize 分层 | **多处过时** | 引用 `optimize/runtime.py`（已改名 `orchestration.py`）；引用 `OptimizeSupervisor`（已改名）；引用 `--continue`（已取代）；提议 `gate.py`（未创建） |
| `2026-04-07-optimize-graceful-interrupt.md` | 优雅中断 | **路径过时** | 引用顶层 runner 模块（已迁移到 `backends/`）；引用 `OptimizeSupervisor`（已改名） |
| `2026-04-08-inspect-ir-script.md` | IR 检查脚本 | **路径过时** | 引用 `skills/ascend-operator-ir-analyzer/`（已改名）；引用 `quick_validate.py` |
| `2026-04-08-ir-dir-flag-alignment.md` | --ir-dir 对齐 | **路径过时** | 同上 |
| `2026-04-08-inspect-ir-ranking-and-change-scan.md` | IR 排名扫描 | **路径过时** | 引用 `skills/ascend-operator-ir-analyzer/`（已改名） |
| `2026-04-08-opt-note-overall-summary.md` | opt-note 摘要 | **路径过时** | 引用 `skills/optimize/references/`（已改名） |
| `2026-04-09-gen-eval.md` | gen-eval 子命令 | **多处过时** | 引用 `src/triton_agent/agent.py`（已删除）；引用旧 skill 名称；引用扁平 runner 模块 |
| `2026-04-09-gen-eval-batch.md` | gen-eval-batch | **路径过时** | 引用 `generation_batch.py` 扁平模块（已重构为 `generation/batch.py`） |
| `2026-04-09-generation-package-refactor.md` | Generation 包重构 | **部分过时** | 引用 `generation/runtime.py`（已改名为 `orchestration.py`） |
| `2026-04-09-optimize-analysis-driven.md` | 分析驱动优化 | **路径过时** | 引用 `optimize/supervisor.py`（已改名）；引用旧 skill 名称；引用扁平 runner 模块 |
| `2026-04-09-optimize-resume-mode.md` | --resume 模式 | **路径过时** | 引用 `optimize/runtime.py`（已改名）；引用 `agent.py`（已删除） |
| `2026-04-09-optimize-status-speedup.md` | 状态加速指标 | **路径过时** | 引用 `skills/optimize/references/`（已改名） |
| `2026-04-09-optimize-status-perf-compat.md` | 状态 perf 兼容 | **路径过时** | 引用 `skills/operator-eval/`（已改名） |
| `2026-04-09-compare-perf-speedup.md` | compare-perf 加速 | **路径过时** | 引用 `skills/operator-eval/`（已改名） |
| `2026-04-09-backends-package-refactor.md` | 后端包重构 | **已实现** | 引用 `generation/runtime.py` 和 `optimize/runtime.py`（均已改名为 `orchestration.py`）；未提及 openhands |
| `2026-04-10-optimize-supervisor-round-gate.md` | Supervisor round gate | **多处过时** | 引用 `optimize/gate.py`（未创建）；引用 `optimize/supervisor.py`（已改名）；引用 `skills/optimize-supervisor/`（已删除）；引用 `.triton-agent/roles/`（已废弃）；引用 `tests/test_optimize/guidance.py`（实际为 `tests/test_optimize_guidance.py`） |
| `2026-04-13-optimize-baseline-prep.md` | Baseline 准备 | **路径过时** | 引用 `optimize/gate.py`（未创建）；引用 `skills/optimize/`（已改名）；引用 `tests/test_optimize_gate.py`（不存在） |
| `2026-04-13-optimize-compare-perf-authority.md` | compare-perf 权威性 | **路径过时** | 引用 `optimize/gate.py`（未创建）；引用 `tests/test_optimize/guidance.py`（路径错误）；引用 `tests/test_optimize_gate.py`（不存在） |
| `2026-04-13-optimize-supervise-mode.md` | --supervise 模式 | **多处过时** | 引用 `optimize/runtime.py`（已改名）；引用 `optimize/supervisor.py`（已改名）；引用 `OptimizeSupervisor`（已改名）；引用 `.triton-agent/roles/`（已废弃） |
| `2026-04-13-optimize-supervised-log-archive.md` | 监督日志归档 | **路径过时** | 引用 `optimize/runtime.py`（已改名）；引用 `tests/test_optimize/guidance.py`（路径错误） |
| `2026-04-14-optimize-check-loop.md` | optimize-check 循环 | **路径过时** | 引用 `optimize/runtime.py`（已改名）；引用 `optimize/supervisor.py`（已改名）；引用旧 skill 名称；引用 `skills/optimize-supervisor/`（已删除） |
| `2026-04-14-optimize-user-prompt-plan.md` | --prompt 标志 | **路径过时** | 引用 `optimize/runtime.py`（已改名） |
| `2026-04-15-comparison-skill-wrapper-flattening.md` | 比较 skill 扁平化 | **路径过时** | 引用 `skills/operator-eval/`（已改名） |
| `2026-04-15-optimize-loop-naming.md` | 循环命名重命名 | **已实现** | 无明显问题 |
| `2026-04-15-optimize-runtime-split.md` | 运行时拆分 | **已实现** | 无明显问题 |
| `2026-04-15-orchestration-module-rename.md` | orchestration 重命名 | **已实现** | 无明显问题 |
| `2026-04-15-runner-wrapper-flattening.md` | Runner 包装扁平化 | **已实现** | 无明显问题 |
| `2026-04-15-optimize-triton-kernel-continuity-prompt.md` | Triton kernel 连续性 | **与代码一致** | 无明显问题 |
| `2026-04-16-skill-renaming-and-supervisor-prompt-consolidation.md` | Skill 重命名 | **已实现** | 最新计划，与代码一致 |
| `2026-03-31-run-command-explicit-paths.md` | 显式路径标志 | **已实现** | 无明显问题 |
| `2026-04-02-generated-harness-pytorch-entrypoints.md` | PyTorch 入口点 | **路径过时** | 引用旧 skill 名称 `test-gen`/`bench-gen` |
| `2026-04-03-batch-optimize-subcommand.md` | optimize-batch | **已实现** | 无明显问题 |
| `2026-04-07-cli-execution-comparison-refactor.md` | 执行/比较重构 | **部分过时** | 提议创建 `comparison.py`（未创建） |
| `2026-04-07-cli-generation-refactor.md` | 生成命令重构 | **部分过时** | 提议扁平 `generation.py`（已重构为包） |
| `2026-04-07-optimize-status-subcommand.md` | optimize-status | **已实现** | 无明显问题 |
| `2026-04-09-docs-layout.md` | 文档布局迁移 | **已实现** | 无明显问题 |
| `2026-04-09-optimize-status-baseline-selection.md` | Baseline 选择 | **已实现** | 无明显问题 |
| `2026-04-09-optimize-status-markdown-table.md` | Markdown 表格 | **已实现** | 无明显问题 |
| `2026-04-09-optimize-status-render.md` | 状态渲染 | **已实现** | 无明显问题 |
| `2026-04-13-baseline-contract-plan.md` | Baseline 契约 | **已实现** | 无明显问题 |
| `2026-04-13-cli-backend-dedup-refactor.md` | 后端去重重构 | **部分过时** | 未提及 openhands 后端 |
| `2026-04-13-optimize-status-single-workspace-input.md` | 单工作区输入 | **已实现** | 无明显问题 |
| `2026-04-14-docs-superpowers-archive.md` | 文档归档 | **已实现** | 无明显问题 |
| `2026-04-14-openhands-backend.md` | OpenHands 后端 | **已实现** | 无明显问题 |
| `2026-04-14-optimize-batch-default-max-concurrency.md` | 批量并发默认值 | **已实现** | 无明显问题 |
| `2026-04-14-optimize-supervisor-alias.md` | --supervisor 别名 | **已实现** | 无明显问题 |

### docs/specs/ 目录（设计规格）

| 文档 | 主题 | 状态 | 主要问题 |
|---|---|---|---|
| `2026-04-09-gen-eval-design.md` | gen-eval 设计 | **路径过时** | 引用旧 skill 名称 `eval-gen`/`test-gen`/`bench-gen`/`operator-eval` |
| `2026-04-09-gen-eval-batch-design.md` | gen-eval-batch 设计 | **路径过时** | 引用旧 skill 名称 |
| `2026-04-09-generation-package-refactor-design.md` | Generation 包重构 | **部分过时** | 引用 `generation/runtime.py`（已改名为 `orchestration.py`） |
| `2026-04-09-compare-perf-speedup-design.md` | compare-perf 加速 | **路径过时** | 引用 `operator-eval` skill（已改名） |
| `2026-04-09-optimize-status-perf-compat-design.md` | Perf 兼容设计 | **路径过时** | 引用 `skills/operator-eval/`（已改名） |
| `2026-04-09-optimize-analysis-driven-design.md` | 分析驱动设计 | **需验证** | `--require-analysis` 是否已实现需确认 |
| `2026-04-10-optimize-supervisor-round-gate-design.md` | Supervisor round gate | **多处过时** | 有"Superseded"标注；引用 `.triton-agent/roles/*`（已废弃）；引用 `optimize-supervisor` skill（已删除）；引用 `optimize/supervisor.py`（已改名） |
| `2026-04-13-optimize-supervise-mode-design.md` | --supervise 设计 | **部分过时** | 有"Superseded"标注；引用 `.triton-agent/roles/*`（已废弃）；函数名与实际实现不同 |
| `2026-04-13-optimize-supervised-log-archive-design.md` | 日志归档设计 | **部分过时** | 有"Superseded"标注；引用 `.triton-agent/roles/*`（已废弃） |
| `2026-04-13-optimize-compare-perf-authority-design.md` | compare-perf 权威设计 | **未实现** | `perf_summary_source` 字段在当前代码中不存在 |
| `2026-04-13-capture-ir-local-python-design.md` | IR 捕获本地 Python | **路径过时** | 引用 `skills/ascend-operator-ir-analyzer/`（已改名） |
| `2026-04-13-profile-bench-local-python-design.md` | Profile bench 本地 Python | **路径过时** | 引用 `skills/operator-eval/`（已改名） |
| `2026-04-13-cli-backend-dedup-refactor-design.md` | 后端去重设计 | **部分过时** | 引用 `triton_agent.comparison`（已删除）；引用 `triton_agent.test_runner`/`bench_runner`（已删除） |
| `2026-04-14-optimize-check-loop-design.md` | optimize-check 设计 | **路径过时** | 引用旧 skill 名称 `optimize-check`（已改名 `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`） |
| `2026-04-14-optimize-reset-design.md` | --reset-optimize 设计 | **需验证** | `--reset-optimize` 是否已实现需确认 |
| `2026-04-14-optimize-user-prompt-design.md` | --prompt 设计 | **需验证** | `--prompt` 是否已实现需确认 |
| `2026-04-15-comparison-skill-wrapper-flattening-design.md` | 比较 skill 扁平化 | **路径过时** | 引用 `skills/operator-eval/`（已改名）；引用 `src/triton_agent/test_runner.py`/`bench_runner.py` 作为"不重构"目标（已删除） |
| `2026-04-15-optimize-runtime-split-design.md` | 运行时拆分设计 | **类名过时** | 引用 `OptimizeController`/`SupervisedRoundRunner`/`RunnerWithStreams`（均已改名） |
| `2026-04-16-optimize-round-performance-analysis-skill-design.md` | 性能分析 skill | **未实现** | `triton-npu-analyze-round-performance` skill 目录不存在；`perf_analysis_path` 字段不存在 |
| `2026-04-16-repair-skill-merge-design.md` | Repair skill 合并 | **已实现** | 无明显问题 |
| `2026-04-16-skill-renaming-and-supervisor-prompt-consolidation-design.md` | Skill 重命名设计 | **已实现** | 与代码一致 |
| `2026-04-16-skill-first-agent-prompts-design.md` | Skill-first prompt | **已实现** | 与代码一致 |
| `2026-04-16-optimize-unsupervised-min-rounds-prompt-design.md` | 无监督 min-rounds | **已实现** | 与代码一致 |
| `2026-04-09-backends-package-refactor-design.md` | 后端包重构 | **已实现** | 无明显问题 |
| `2026-04-09-batch-workspace-discovery-sharing-design.md` | 批量工作区发现 | **已实现** | 无明显问题 |
| `2026-04-09-docs-layout-design.md` | 文档布局 | **已实现** | 无明显问题 |
| `2026-04-09-optimize-resume-mode-design.md` | --resume 设计 | **已实现** | 无明显问题 |
| `2026-04-09-optimize-batch-root-workspace-design.md` | 批量根工作区 | **已实现** | 无明显问题 |
| `2026-04-09-optimize-status-baseline-selection-design.md` | Baseline 选择 | **已实现** | 无明显问题 |
| `2026-04-09-optimize-status-markdown-table-design.md` | Markdown 表格 | **已实现** | 无明显问题 |
| `2026-04-09-optimize-status-render-design.md` | 状态渲染 | **已实现** | 无明显问题 |
| `2026-04-09-optimize-status-speedup-design.md` | 加速指标 | **已实现** | 无明显问题 |
| `2026-04-13-optimize-baseline-contract-design.md` | Baseline 契约 | **已实现** | 无明显问题 |
| `2026-04-13-optimize-baseline-prep-design.md` | Baseline 准备 | **已实现** | 无明显问题 |
| `2026-04-13-optimize-status-single-workspace-input-design.md` | 单工作区输入 | **已实现** | 无明显问题 |
| `2026-04-14-openhands-backend-design.md` | OpenHands 后端 | **已实现** | 无明显问题 |
| `2026-04-14-optimize-baseline-state-contract-design.md` | Baseline 状态契约 | **已实现** | 无明显问题 |
| `2026-04-14-optimize-supervisor-alias-design.md` | --supervisor 别名 | **已实现** | 无明显问题 |
| `2026-04-15-agents-doc-boundary-design.md` | AGENTS.md 边界 | **已实现** | 无明显问题 |
| `2026-04-15-optimize-loop-naming-design.md` | 循环命名 | **已实现** | 无明显问题 |
| `2026-04-15-orchestration-module-rename-design.md` | orchestration 重命名 | **已实现** | 无明显问题 |
| `2026-04-15-runner-wrapper-flattening-design.md` | Runner 扁平化 | **已实现** | 无明显问题 |
| `2026-04-15-optimize-triton-kernel-continuity-prompt-design.md` | Kernel 连续性 | **已实现** | 无明显问题 |

### docs/bug-reports/ 目录

| 文档 | 主题 | 状态 | 主要问题 |
|---|---|---|---|
| `2026-04-02-bug-review-status.md` | Bug 审查状态 | **已修复/可归档** | 所有 bug 已修复；引用 `codex_runner.py` 和 `supervisor.py` 旧路径 |
| `2026-04-02-differential-test-comparison.md` | 差分测试比较 bug | **已修复/可归档** | 所有 bug 已修复 |

---

## 四、未实现的设计规格

以下规格文档描述的功能在当前代码中未找到对应实现：

1. **`2026-04-13-optimize-compare-perf-authority-design.md`**: `perf_summary_source` 字段和 `compare-perf` 权威性检查未实现
2. **`2026-04-16-optimize-round-performance-analysis-skill-design.md`**: `triton-npu-analyze-round-performance` skill 和 `perf_analysis_path` 字段未实现

---

## 五、建议

### 高优先级

1. **统一旧 skill 名称引用**: 对所有仍引用旧 skill 名称的文档进行批量替换，这是涉及面最广的过时问题
2. **更新模块路径引用**: 将所有引用旧扁平 runner 模块和旧 optimize 模块名的文档更新为当前路径
3. **标注已取代文档**: 对已被后续文档取代的早期文档（如 `skill-link-idempotent-symlinks.md`、`optimize-continue-mode.md`）在文件头部添加 "Superseded" 标注

### 中优先级

4. **归档 bug 报告**: 两个 bug 报告中的问题已全部修复，可移至归档目录或添加 "Resolved" 标注
5. **验证未实现规格**: 确认 `perf_summary_source` 和 `triton-npu-analyze-round-performance` skill 是否仍在计划中，如已放弃则标注
6. **移除硬编码路径**: 多份文档引用 `/Users/cdj/.codex/skills/.system/skill-creator/scripts/quick_validate.py`，这是环境特定路径，应移除

### 低优先级

7. **补充缺失后端**: 部分文档仅提及早期后端（codex/opencode），未提及后来添加的 pi/claude/openhands
8. **统一 "Link" 命名**: `SkillLinkManager`/`SkillLinkSet` 类名仍使用 "Link" 但实际实现为 copy，可考虑重命名或添加注释
