# triton-agent 系统知识库

> 本文件是对 `triton-agent` 新引擎系统的完整架构分析与使用指南。
> 供后续 AI 或新开 chat 窗口快速学习。
> 最后更新: 2026-04-08
> 文档存储位置: `triton-agent/docs_cc/triton-agent-knowledge-base.md`

---

## 〇、架构总结（速读版）

### 核心设计：薄 CLI + 外部 Agent + 技能驱动

triton-agent **本身不包含 LLM 调用逻辑**，而是作为编排层，把任务分发给外部 Agent 后端（codex / opencode / pi / claude）。Agent 通过读取 `skills/` 目录下的自然语言工作流指南来自主完成任务。

### 9 个子命令

| 类别 | 命令 | 说明 |
|------|------|------|
| 生成 | `gen-test` / `gen-bench` | 启动 Agent 生成测试/基准文件 |
| 执行 | `run-test` / `run-bench` | 直接执行脚本（不启动 Agent） |
| 比较 | `compare-result` / `compare-perf` | 精度/性能对比 |
| 优化 | `optimize` / `optimize-batch` / `optimize-status` | 多轮优化循环 |

### 4 个关键架构组件

1. **AgentRunner**（抽象基类）→ CodexRunner / PiRunner / ClaudeRunner / OpenCodeRunner
2. **SkillLinkManager** — 将 `skills/` 拷贝到 Agent 的工作空间（`.codex/skills/`, `.pi/skills/` 等）
3. **OptimizeSupervisor** — 管理多轮优化、min_rounds 检查、stall 自动恢复
4. **OptimizeGuidanceManager** — 在工作目录写入临时 `AGENTS.md`/`CLAUDE.md` 引导文件

### 6 个技能

| 技能 | 职责 |
|------|------|
| `test-gen` | 生成正确性测试（standalone / differential） |
| `bench-gen` | 生成性能基准（standalone / msprof） |
| `operator-eval` | 执行测试/基准/比较（包含实际脚本） |
| `optimize` | 多轮优化工作流 + 13 种优化模式 |
| `ascend-npu-operator-profiler` | NPU 性能分析 |
| `ascend-operator-ir-analyzer` | Triton/Bisheng IR 捕获与分析 |

### 主要修改文件（按修改频率排序）

1. **`skills/`** — 最频繁，改工作流指南或添加优化模式
2. **`src/triton_agent/prompts.py`** — 修改 Prompt 模板
3. **`src/triton_agent/optimize_guidance.py`** — 修改优化引导内容
4. **`src/triton_agent/supervisor.py`** — 修改优化循环策略
5. **`src/triton_agent/runner_factory.py`** + 新 Runner — 添加新 Agent 后端
6. **`src/triton_agent/cli.py`** + `models.py` — 添加新子命令

### 与 triton-dataset 集成示例

```bash
DATASET=/path/to/triton-dataset/DLBlas/kernelagent
REMOTE=user@npu-host:2222

# 生成差分测试
uv run triton-agent gen-test -i $DATASET/level1/op_1_*/op_1_*.py --test-mode differential --remote $REMOTE

# 运行测试
uv run triton-agent run-test --test-file differential_test_op_1.py --operator-file op_1.py --remote $REMOTE

# 批量优化
uv run triton-agent optimize-batch -i $DATASET/level1/ --max-concurrency 4 --remote $REMOTE
```

---

## 一、系统定位与新旧对比

### 旧系统：`triton-ascend-optimizer`

```
triton-ascend-optimizer/
├── operator-isolator   # Magic CLI 实现，隔离算子代码
├── operator-tester     # Magic CLI 实现，生成测试代码
├── new-optimizer/      # Python 实现，核心优化引擎
│   └── optimizer/
│       ├── script/     # gen_test, run_test, run_bench 等命令
│       ├── agents/     # LLM Agent 循环 (test_fix_loop 等)
│       └── ...
└── operator-optimizer  # (旧版) Magic CLI 优化
```

- 命令入口：`python -m optimizer.script gen_test/run_test/run_bench`
- 测试模式：`differential`（差分测试）
- Agent 后端：内置 OpenAI API 调用
- 功能测试依赖 `OLD_<operator>.py` 作为 oracle

### 新系统：`triton-agent`

```
triton-agent/
├── src/triton_agent/   # 核心 Python 包
├── skills/             # 自然语言技能定义（Agent 可读）
├── docs/               # 设计文档
├── tests/              # 单元测试
├── workspace/          # 本地实验占位
├── pyproject.toml      # uv 管理
├── AGENTS.md           # 项目规则
└── README.md           # 使用说明
```

- 命令入口：`uv run triton-agent <subcommand>`
- Agent 后端：**多后端**（codex / opencode / pi / claude）
- 技能驱动：skills/ 目录为 Agent 提供自然语言工作流指导
- 支持本地 + **远程 SSH 执行**
- 支持 **msprof 性能分析** + **IR 分析**

### 核心差异

| 特性 | 旧系统 (optimizer) | 新系统 (triton-agent) |
|------|-------------------|---------------------|
| 架构 | 单体 Python + Magic CLI | 薄 CLI + 外部 Agent 后端 |
| Agent 后端 | 内置 OpenAI API | codex / opencode / pi / claude |
| 技能系统 | 无 | skills/ 自然语言工作流 |
| 测试模式 | differential | standalone + differential |
| 性能分析 | 无 | standalone + msprof |
| 远程执行 | Docker 内执行 | SSH (`--remote user@host:port`) |
| 优化循环 | test_fix_loop | OptimizeSupervisor + opt-round-N |
| 产物管理 | TEST_RESULT.pt | opt-round-N/ + opt-note.md |
| 精度比较 | 内置 _compare_result_files | compare-result (strict/balanced/relaxed) |

---

## 二、架构详解

### 2.1 入口与命令

CLI 入口：`triton_agent.cli:main`（pyproject.toml 注册为 `triton-agent`）

9 个子命令（`CommandKind` 枚举）：

| 命令 | 功能 | 是否启动 Agent |
|------|------|--------------|
| `gen-test` | 生成正确性测试文件 | 是 |
| `gen-bench` | 生成性能基准测试文件 | 是 |
| `run-test` | 执行测试文件 | 否（直接运行脚本） |
| `run-bench` | 执行基准测试文件 | 否（直接运行脚本） |
| `compare-result` | 比较差分测试结果 | 否 |
| `compare-perf` | 比较性能数据 | 否 |
| `optimize` | 单算子优化（多轮） | 是 |
| `optimize-batch` | 批量算子优化 | 是 |
| `optimize-status` | 查看优化状态 | 否（只读扫描） |

### 2.2 核心模块

```
src/triton_agent/
├── cli.py                 # CLI 解析与入口
├── models.py              # CommandKind, AgentRequest, AgentResult
├── agent.py               # AgentRunner 抽象基类
├── runner_factory.py      # create_runner() 工厂
├── codex_runner.py        # Codex 后端
├── opencode_runner.py     # OpenCode 后端
├── pi_runner.py           # Pi 后端
├── claude_runner.py       # Claude 后端
├── process_runner.py      # 进程管理（buffered/streaming/interactive）
├── supervisor.py          # OptimizeSupervisor（多轮 + stall recovery）
├── prompts.py             # Prompt 构建
├── skills.py              # SkillLinkManager（技能拷贝到工作目录）
├── optimize_guidance.py   # OptimizeGuidanceManager（写 AGENTS.md/CLAUDE.md）
├── generation.py          # gen-test/gen-bench 请求构建与执行
├── execution.py           # run-test/run-bench 执行（本地/远程）
├── comparison.py          # compare-result/compare-perf
├── paths.py               # 输出路径默认命名规则
├── output.py              # render_result() 输出渲染
├── verbose.py             # 诊断日志
├── run_skill.py           # 动态加载 skills/ 脚本
├── test_runner.py         # 测试执行模块（代理到 skill 脚本）
├── bench_runner.py        # 性能执行模块
├── result_normalization.py # Agent 结果标准化
├── commands/              # 子命令处理器
│   ├── generation.py      # handle_gen_test, handle_gen_bench
│   ├── execution.py       # handle_run_test, handle_run_bench
│   ├── comparison.py      # handle_compare_result, handle_compare_perf
│   └── optimize.py        # handle_optimize, handle_optimize_batch, handle_optimize_status
└── optimize/              # 优化子系统
    ├── models.py          # OptimizeRunOptions, BatchOptimizeResult
    ├── runtime.py         # build_optimize_request, run_optimize_request
    ├── batch.py           # run_optimize_batch（线程池并发）
    ├── validation.py      # validate_optimize_options
    ├── status.py          # scan_optimize_status_workspaces
    └── render.py          # render_optimize_status_results
```

### 2.3 数据流

```
用户命令 → cli.py 解析 → commands/ 处理器
  ├─ gen-test/gen-bench:
  │   → generation.py: build_generation_request()
  │   → SkillLinkManager: 拷贝 skills/ 到工作目录
  │   → runner_factory: create_runner(agent_name)
  │   → AgentRunner.run(request) → 启动外部 Agent 进程
  │   → Agent 读取 skills/ → 生成测试/基准文件
  │   → SkillLinkManager.cleanup()
  │
  ├─ run-test/run-bench:
  │   → execution.py → run_skill.py 动态加载 skills/operator-eval/scripts/
  │   → 本地执行 python3 test_*.py --operator-file ...
  │   → 或 SSH 远程执行
  │
  ├─ optimize:
  │   → optimize/runtime.py: build_optimize_request()
  │   → SkillLinkManager: 拷贝 skills/
  │   → OptimizeGuidanceManager: 写 AGENTS.md/CLAUDE.md（临时指导文件）
  │   → OptimizeSupervisor.run():
  │       → runner.run() → Agent 执行优化
  │       → 检查 min_rounds 要求 → 不足则 resume()
  │       → 检查 stall → 自动恢复（最多2次）
  │   → Guidance cleanup（恢复/删除临时文件）
  │   → SkillLinkManager.cleanup()
  │
  └─ optimize-batch:
      → ThreadPoolExecutor(max_concurrency)
      → 每个 workspace 独立执行 optimize 流程
```

### 2.4 技能系统 (skills/)

```
skills/
├── test-gen/           # 测试生成技能
│   ├── SKILL.md        # Agent 读取的工作流指南
│   └── references/     # 规范文档
├── bench-gen/          # 基准生成技能
├── operator-eval/      # 执行与评估技能
│   ├── SKILL.md
│   └── scripts/        # 实际执行脚本（run-command.py 等）
├── optimize/           # 优化技能
│   ├── SKILL.md
│   └── references/     # 优化模式库 (patterns/)
├── ascend-npu-operator-profiler/  # NPU 性能分析技能
│   ├── SKILL.md
│   └── scripts/        # parse_bin.py, profile_summary.py
└── ascend-operator-ir-analyzer/   # IR 分析技能
    ├── SKILL.md
    └── scripts/        # capture_ir.py, inspect_ir.py
```

**关键设计**：CLI 是薄壳，实际工作流逻辑在 skills/ 中用自然语言描述，由 Agent 读取并执行。

### 2.5 Agent 后端

| 后端 | 命令 | 技能目录 | 特殊参数 |
|------|------|---------|---------|
| codex | `codex exec --cd <dir> --sandbox danger-full-access` | `.codex/skills/` | `--ephemeral`, `--skip-git-repo-check` |
| opencode | opencode 进程 | `.opencode/skills/` | — |
| pi | pi 进程 | `.pi/skills/` | `--thinking high`, `--no-extensions`, `--no-skills` |
| claude | claude 进程 | `.claude/skills/` | `--print`, `--dangerously-skip-permissions` |

### 2.6 优化循环 (OptimizeSupervisor)

```
OptimizeSupervisor.run():
  1. runner.run(request)          # 首次运行
  2. while result.succeeded and needs_more_rounds:
       runner.resume(request)     # 自动继续直到 min_rounds 满足
  3. if !succeeded and stalled:
       for attempt in range(max_recovery_attempts):
         runner.resume(request)   # stall 恢复（最多2次）
  4. return result
```

优化产物结构：
```
operator_workspace/
├── operator.py              # 原始算子
├── test_operator.py         # 正确性测试
├── bench_operator.py        # 性能基准
├── opt-note.md              # 优化全局笔记
├── learned_lessons.md       # 可复用经验
├── opt-round-1/
│   ├── opt_operator.py      # 本轮优化算子
│   ├── attempts.md          # 尝试记录
│   ├── summary.md           # 轮次总结
│   ├── perf.txt             # 性能数据
│   ├── profile/             # msprof 分析结果
│   └── ir/                  # IR 存档
├── opt-round-2/
│   └── ...
└── opt-round-N/
```

---

## 三、完整命令合集

### 3.0 安装

```bash
# 使用 uv 安装（推荐）
cd triton-agent
uv sync

# 或 pip 安装
pip install -e .
```

### 3.1 gen-test — 生成正确性测试

```bash
# 基本用法（standalone 模式，默认使用 codex 后端）
uv run triton-agent gen-test -i operator.py

# 指定测试模式
uv run triton-agent gen-test -i operator.py --test-mode standalone
uv run triton-agent gen-test -i operator.py --test-mode differential

# 指定输出路径
uv run triton-agent gen-test -i operator.py -o test_operator.py

# 切换 Agent 后端
uv run triton-agent gen-test -i operator.py --agent codex
uv run triton-agent gen-test -i operator.py --agent opencode
uv run triton-agent gen-test -i operator.py --agent pi
uv run triton-agent gen-test -i operator.py --agent claude

# 覆盖已有输出
uv run triton-agent gen-test -i operator.py --force-overwrite

# 远程执行
uv run triton-agent gen-test -i operator.py --remote user@host:2222

# 交互模式 / 显示输出 / 详细日志
uv run triton-agent gen-test -i operator.py --interact
uv run triton-agent gen-test -i operator.py --show-output
uv run triton-agent gen-test -i operator.py --verbose
```

### 3.2 gen-bench — 生成性能基准

```bash
# 基本用法（standalone 模式）
uv run triton-agent gen-bench -i operator.py

# 指定模式
uv run triton-agent gen-bench -i operator.py --bench-mode standalone
uv run triton-agent gen-bench -i operator.py --bench-mode msprof

# 指定输出 / 后端 / 远程
uv run triton-agent gen-bench -i operator.py -o bench_operator.py
uv run triton-agent gen-bench -i operator.py --agent codex
uv run triton-agent gen-bench -i operator.py --remote user@host
```

### 3.3 run-test — 执行测试

```bash
# 本地执行
uv run triton-agent run-test --test-file test_operator.py --operator-file operator.py

# 指定测试模式（不指定则从文件 metadata 读取）
uv run triton-agent run-test --test-file test_operator.py --operator-file operator.py --test-mode standalone
uv run triton-agent run-test --test-file differential_test_operator.py --operator-file opt_operator.py --test-mode differential

# 远程执行
uv run triton-agent run-test --test-file test_operator.py --operator-file operator.py --remote user@host:2222
uv run triton-agent run-test --test-file test_operator.py --operator-file operator.py --remote user@host --remote-workdir /tmp/triton-agent

# 保留远程目录（调试用）
uv run triton-agent run-test --test-file test_operator.py --operator-file operator.py --remote user@host --keep-remote-workdir
```

### 3.4 run-bench — 执行基准测试

```bash
# 本地执行
uv run triton-agent run-bench --bench-file bench_operator.py --operator-file operator.py

# 指定模式
uv run triton-agent run-bench --bench-file bench_operator.py --operator-file operator.py --bench-mode standalone
uv run triton-agent run-bench --bench-file bench_operator.py --operator-file operator.py --bench-mode msprof

# 远程执行
uv run triton-agent run-bench --bench-file bench_operator.py --operator-file operator.py --remote user@host:2222
```

### 3.5 compare-result — 比较差分测试结果

```bash
# 本地比较
uv run triton-agent compare-result --oracle-result oracle_result.pt --new-result opt_result.pt

# 指定比较级别
uv run triton-agent compare-result --oracle-result oracle_result.pt --new-result opt_result.pt --compare-level strict
uv run triton-agent compare-result --oracle-result oracle_result.pt --new-result opt_result.pt --compare-level balanced
uv run triton-agent compare-result --oracle-result oracle_result.pt --new-result opt_result.pt --compare-level relaxed

# 远程比较
uv run triton-agent compare-result --oracle-result oracle_result.pt --new-result opt_result.pt --remote user@host
```

### 3.6 compare-perf — 比较性能数据

```bash
uv run triton-agent compare-perf --baseline baseline_perf.txt --compare opt_perf.txt
```

### 3.7 optimize — 单算子优化

```bash
# 基本优化（默认 differential + standalone，codex 后端）
uv run triton-agent optimize -i operator.py

# 指定模式
uv run triton-agent optimize -i operator.py --test-mode differential --bench-mode standalone

# 指定最少轮次
uv run triton-agent optimize -i operator.py --min-rounds 3

# 继续已有优化会话
uv run triton-agent optimize -i operator.py --continue

# 禁用 Agent 持久会话
uv run triton-agent optimize -i operator.py --no-agent-session

# 切换 Agent 后端
uv run triton-agent optimize -i operator.py --agent pi
uv run triton-agent optimize -i operator.py --agent claude

# 远程执行
uv run triton-agent optimize -i operator.py --remote user@host:2222 --remote-workdir /tmp/triton-agent

# 交互 / 显示输出
uv run triton-agent optimize -i operator.py --interact
uv run triton-agent optimize -i operator.py --show-output
```

### 3.8 optimize-batch — 批量优化

```bash
# 基本批量优化（扫描目录下所有子目录）
uv run triton-agent optimize-batch -i operators_root/

# 控制并发数
uv run triton-agent optimize-batch -i operators_root/ --max-concurrency 4

# 指定 Agent 和模式
uv run triton-agent optimize-batch -i operators_root/ --agent pi --test-mode differential --bench-mode standalone

# 继续已有会话
uv run triton-agent optimize-batch -i operators_root/ --continue

# 显示实时输出（带 workspace 前缀）
uv run triton-agent optimize-batch -i operators_root/ --show-output
```

### 3.9 optimize-status — 查看优化状态

```bash
uv run triton-agent optimize-status -i operators_root/
uv run triton-agent optimize-status -i operators_root/ --verbose
```

### 3.10 开发检查

```bash
# 代码风格检查
uv run --group dev ruff check

# 静态类型检查
uv run pyright

# 单元测试
uv run python -m unittest discover -s tests -v
```

### 3.11 Skill 脚本直接调用（在 Agent 工作空间内）

```bash
# 运行测试
python3 skills/operator-eval/scripts/run-command.py run-test --test-file test_op.py --operator-file op.py

# 运行基准
python3 skills/operator-eval/scripts/run-command.py run-bench --bench-file bench_op.py --operator-file op.py

# 性能分析
python3 skills/operator-eval/scripts/run-command.py profile-bench --bench-file bench_op.py --operator-file op.py

# 比较结果
python3 skills/operator-eval/scripts/run-command.py compare-result --oracle-result a.pt --new-result b.pt

# 捕获 IR
python3 skills/ascend-operator-ir-analyzer/scripts/capture_ir.py --ir-dir ir_output/ --bench-file bench_op.py --operator-file op.py

# 查看 IR 阶段
python3 skills/ascend-operator-ir-analyzer/scripts/inspect_ir.py list-stages --ir-dir ir_output/
python3 skills/ascend-operator-ir-analyzer/scripts/inspect_ir.py stage-summary --ir-dir ir_output/
python3 skills/ascend-operator-ir-analyzer/scripts/inspect_ir.py diff-stages --ir-dir ir_output/
python3 skills/ascend-operator-ir-analyzer/scripts/inspect_ir.py find-changes --ir-dir ir_output/
```

---

## 四、主要修改文件指南

### 4.1 日常扩展场景

| 修改目标 | 文件 | 说明 |
|---------|------|------|
| **添加新 Agent 后端** | `src/triton_agent/runner_factory.py` | 注册新 Runner |
| | `src/triton_agent/<new>_runner.py` | 实现 AgentRunner.run() |
| | `src/triton_agent/skills.py` | 添加 prepare_<new>_skills() |
| | `src/triton_agent/cli.py` | 添加 `--agent` 选项 |
| **添加新子命令** | `src/triton_agent/models.py` | 添加 CommandKind 枚举 |
| | `src/triton_agent/cli.py` | build_parser() + main() |
| | `src/triton_agent/commands/<new>.py` | 实现处理器 |
| **修改 Prompt** | `src/triton_agent/prompts.py` | build_prompt() |
| **修改优化引导** | `src/triton_agent/optimize_guidance.py` | _render_guidance() |
| **修改优化循环** | `src/triton_agent/supervisor.py` | OptimizeSupervisor |
| **修改测试执行** | `skills/operator-eval/scripts/test_runner.py` | 实际测试执行逻辑 |
| **修改基准执行** | `skills/operator-eval/scripts/bench_runner.py` | 实际基准执行逻辑 |
| **修改精度比较** | `skills/operator-eval/scripts/compare_result.py` | 比较逻辑 |
| **添加优化模式** | `skills/optimize/references/patterns/<new>.md` | 模式文档 |
| | `skills/optimize/references/patterns/index.md` | 模式索引 |
| **修改测试生成规范** | `skills/test-gen/references/test-standalone-spec.md` | standalone 规范 |
| | `skills/test-gen/references/test-differential-spec.md` | differential 规范 |
| **修改基准生成规范** | `skills/bench-gen/references/bench-standalone-spec.md` | standalone 规范 |
| | `skills/bench-gen/references/bench-msprof-spec.md` | msprof 规范 |
| **修改 IR 分析** | `skills/ascend-operator-ir-analyzer/scripts/` | capture/inspect 脚本 |
| **修改性能分析** | `skills/ascend-npu-operator-profiler/scripts/` | parse/summary 脚本 |

### 4.2 核心分层

```
修改优先级（从高到低）：
1. skills/          ← Agent 直接读取的工作流指南和脚本，最频繁修改
2. src/triton_agent/optimize/  ← 优化子系统
3. src/triton_agent/prompts.py ← Prompt 工程
4. src/triton_agent/cli.py     ← CLI 扩展
5. src/triton_agent/agent.py   ← Runner 抽象
```

### 4.3 与 triton-dataset 集成

要将新系统用于 triton-dataset 中的算子：

1. **算子文件**：`triton-dataset/DLBlas/kernelagent/level*/op_*/op_*.py`
2. **生成测试**：`uv run triton-agent gen-test -i <operator.py> --test-mode differential --remote user@host:port`
3. **运行测试**：`uv run triton-agent run-test --test-file <test.py> --operator-file <op.py> --remote user@host:port`
4. **批量优化**：`uv run triton-agent optimize-batch -i <level_root> --max-concurrency 4 --remote user@host:port`

---

## 五、重要设计原则

1. **CLI 是薄壳**：编排在 CLI，工作流逻辑在 skills 中
2. **技能即知识**：skills/ 用自然语言描述，Agent 读取并自主执行
3. **不替换 Triton**：optimize guidance 明确禁止用 PyTorch 替换 Triton 算子路径
4. **产物保护**：默认不覆盖已有文件，需 `--force-overwrite`
5. **多 Agent 支持**：Runner 抽象解耦了 Agent 后端差异
6. **远程执行**：通过 SSH 支持在 NPU 服务器上远程执行
7. **可恢复优化**：`--continue` 支持断点续跑，OptimizeSupervisor 自动处理 stall

---

## 六、关键概念速查

| 概念 | 说明 |
|------|------|
| `test-mode: standalone` | 自包含断言测试，不需 oracle |
| `test-mode: differential` | 差分测试，需 oracle 结果对比 |
| `bench-mode: standalone` | 本地计时测试（do_bench_npu） |
| `bench-mode: msprof` | msprof 性能分析模式 |
| `compare-level: strict` | 严格精度比较 |
| `compare-level: balanced` | 平衡精度比较（默认） |
| `compare-level: relaxed` | 宽松精度比较 |
| `opt-round-N/` | 第 N 轮优化产物目录 |
| `opt-note.md` | 优化全局笔记 |
| `learned_lessons.md` | 可复用优化经验 |
| `api-kind: triton-wrapper` | 包装 Triton kernel 的函数入口 |
| `api-kind: torch-function` | PyTorch 函数入口 |
| `api-kind: torch-module` | nn.Module 入口 |
| `SkillLinkManager` | 将 skills/ 拷贝到 Agent 工作空间 |
| `OptimizeSupervisor` | 管理多轮优化 + stall 恢复 |
| `OptimizeGuidanceManager` | 写入临时 AGENTS.md/CLAUDE.md |
