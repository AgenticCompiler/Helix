# operator-report Command Implementation Plan (Agent-Driven)

## Motive

Add a `triton-agent report -i <operator_dir>` command that launches an AI agent
to read the workspace artifacts and generate `report.md` in the operator workspace
directory. The report is a human-readable Chinese document suitable for external review.

Unlike `report-batch` (which is pure data-parsing), the `report` command uses the
agent's model capabilities to synthesize information from multiple artifacts
(`opt-note.md`, `summary.md`, `env-info.json`, `round-state.json`, etc.)
into a coherent report.

## Reference Template

The report follows the format shown in `/mnt/data01/pandaoxin/fhq/docs/report.md`:

1. 优化概览 (latency table across rounds)
2. 优化历程 (round overview table with round directory paths)
3. 各轮优化详解 (per-round detail, full summary.md content)
4. 最终代码结构 (best-round operator source code)
5. 性能提升分解 (key optimization points breakdown, from opt-note.md Overall Summary)
6. 验证信息 (verification metadata: chip, correctness, bench)
7. 优化文件 (path references)

---

## Work Items

### W1: 修改 opt-note-format.md 标准模板

**File:** `skills/triton/triton-npu-optimize/references/opt-note-format.md`

在 `## Overall Summary` 模板中增加 `Key optimization points` 节，使 agent 在完成 optimize
时自动写入关键优化点清单，供 report agent 解析使用。

```md
## Overall Summary
Final best round: round-N
...
Key optimization points:
  1. <optimization point>: <improvement> (round N)
  2. ...
```

同步更新 Example 和 Writing Guidance。

---

### W2: 优化时记录硬件环境信息

#### W2.1 新增硬件信息采集模块

**File:** `src/triton_agent/hardware_info.py` (new)

函数 `capture_hardware_info()`:
- 调用 `npu-smi info -t board -i 0 -c 0` 获取 Chip Name / Chip Version
- 读取 `/usr/local/Ascend/driver/version.info` 获取 Driver Version
- 读取环境变量 `ASCEND_TOOLKIT_HOME` 获取 CANN 路径/版本
- 容错 fallback 到 `npu-smi info -m`

#### W2.2 optimize 会话启动时写入 env-info.json

**File:** `src/triton_agent/optimize/orchestration.py`

在 `build_optimize_request` 中，workspace 建立后首次运行时（`env-info.json` 不存在时）:
- 调用 `capture_hardware_info()`
- 写入 `workspace/env-info.json`:
  ```json
  {
    "target_chip": "A3",
    "hardware": { "chip_name": "Ascend 910B2", "cann_version": "cann-9.0.0-beta.2", ... },
    "started_at": "2026-05-28T12:00:00"
  }
  ```

`env-info.json` 是 CLI 基础设施层文件：由 optimize 启动时写入，report agent 读取。
AI agent（优化会话中的 LLM）不直接读写此文件。

---

### W3: 创建 triton-npu-report 技能

**Directory:** `skills/triton-npu-report/` (new)

```text
skills/triton-npu-report/
├── SKILL.md
└── references/
    └── report-format.md
```

#### W3.1 SKILL.md

工作流说明书，指导 agent 如何生成报告：

- 步骤1: 读取 `env-info.json` 获取硬件环境信息
- 步骤2: 读取 `opt-note.md` 解析所有轮次条目和 Overall Summary
- 步骤3: 读取每个 `opt-round-N/summary.md` 获取各轮详解
- 步骤4: 读取最优轮次的 `opt_<operator>.py` 获取最终源码
- 步骤5: 读取 `opt-note.md` Overall Summary 的 `Key optimization points` 获取优化点分解
- 步骤6: 生成 `report.md`，格式参考 `references/report-format.md`

#### W3.2 report-format.md

中文报告模板，与 `/mnt/data01/pandaoxin/fhq/docs/report.md` 同结构：

```markdown
# 优化报告
**生成时间**：...
**workspace**：...

## 1、优化概览
| 指标 | 初始版本 | Round N | ... |
...

## 2、优化历程
| Round | Parent | Round目录 | Theme | Analysis Level | Latency | Speedup | Des |
...

## 3、各轮优化详解
### 3.1 Round N (opt-round-N/)
{summary.md 全文}

## 4、最终代码结构
```python
{最优轮算子源码}
```

## 5、性能提升分解
| 优化点 | 提升 | Round | 说明 |
...

## 6、验证信息
**目标芯片**：...
**精度验证**：...
**验证模式**：性能优化模式

## 7、优化文件
- 优化后源码：opt-round-N/opt_*.py
- 最终性能报告：opt-round-N/*_perf.txt
- 优化笔记：opt-note.md
- 优化知识：learned_lessons.md
```

---

### W4: 注册 report CLI 命令 + Agent 启动

#### W4.1 命令注册

**File:** `src/triton_agent/models.py` — 新增 `REPORT = "report"` + `COMMAND_TO_SKILL` 映射为空

**File:** `src/triton_agent/cli.py` — 注册 `_CommandSpec`:
```python
CommandKind.REPORT: _CommandSpec(
    handler=handle_report,
    help_group="Reporting",
    help_summary="Generate operator-level optimization report.",
    description="Launch an agent to read workspace artifacts and render report.md.",
    has_output=False,
    has_agent=True,
    has_interact=True,
    has_show_output=True,
    has_prompt=True,
)
```

**File:** `src/triton_agent/skill_staging.py` — 新增 StageRule:
```python
CommandKind.REPORT: StageRule(
    directives=(
        "+triton-npu-report",
    ),
),
```

#### W4.2 Prompt 构建

**File:** `src/triton_agent/prompts.py` — 新增 prompt intro:
```python
PROMPT_INTROS = {
    ...
    CommandKind.REPORT: "Read the optimize workspace and generate a Chinese optimization report.",
}
```

#### W4.3 Handler

**File:** `src/triton_agent/commands/report.py` (new)

```python
def handle_report(parser, args) -> int:
    workspace = Path(args.input).expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"Not a directory: {workspace}")
    
    agent_name = getattr(args, "agent", "codex")
    prompt = build_prompt(CommandKind.REPORT, ...)
    
    # Stage skills
    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.REPORT,
    )
    
    request = AgentRequest(
        command_kind=CommandKind.REPORT,
        input_path=workspace,
        workdir=workspace,
        operator_path=None,
        output_path=workspace / "report.md",
        agent_name=agent_name,
        skill_name="triton-npu-report",
        prompt=prompt,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
        interact=getattr(args, "interact", False),
        show_output=getattr(args, "show_output", False),
        ...
    )
    
    manager = SkillLinkManager(skills_root())
    links = manager.prepare_skills(...)
    try:
        runner = create_runner(agent_name)
        result = runner.run(request)
    finally:
        manager.cleanup(links)
    
    return result.return_code
```

---

### W5: env-info.json 与 reset 整合

**File:** `src/triton_agent/optimize/resume.py`

`--reset-optimize` 时删除 `env-info.json`（无需特殊保护，下次 optimize 重新采集即可）。

---

### W6: 在真实数据上集成验证

在工作区 `/mnt/data01/pandaoxin/bzhan/triton-dataset/.../13_InterleaveRope` 上运行：
```bash
uv run triton-agent report -i <operator_dir> --agent <agent>
```
检查生成的 `report.md` 是否完整、数据是否正确。

---

### W7: report-batch 命令增强 — 并行生成各算子 report.md

**目标**：`report-batch` 命令除了在批跑根目录生成 `report-batch-state.json` + `report-batch.md`，
还需在每个算子 workspace 下生成 `report.md`（复用 agent 驱动的 report 能力）。

#### W7.1 report-batch _CommandSpec 更新

**File:** `src/triton_agent/cli.py`

REPORT_BATCH 的 `_CommandSpec` 增加 `has_agent=True`, `has_show_output=True`，
以便用户指定 `--agent` 和 `--show-output`。

#### W7.2 handle_batch_report 增强

**File:** `src/triton_agent/commands/batch_report.py`

在现有两步（`write_batch_report_state` + `render_batch_report_file`）之后，
新增第三步：

1. 发现所有算子 workspace 目录
2. 对每个 workspace，执行 agent 生成 `report.md`：
   - 构建 AgentRequest：`command_kind=REPORT`, `workdir=ws_dir`, `agent_name=...`
   - 通过 `SkillLinkManager` 将 `triton-npu-report` 技能 staging 到对应 workspace
   - 通过 `create_runner(agent_name)` 创建 runner 并 `run(request)`
3. 每个 report 生成结果打印到控制台

#### W7.3 执行策略

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--agent` | `opencode` | 控制 per-workspace report 使用的 agent 后端 |
| `--show-output` | `false` | 是否实时输出 agent 的 stdout |
| `--concurrency` / `-c` | `1` | 并发生成 report 的最大并行 worker 数 |

默认并行 1 个：使用 `concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)`，
每个 worker 线程独立创建自己的 `SkillLinkManager` + `AgentRunner`，
对不同 workspace 并行启动 agent 生成 `report.md`。

并行安全考虑：
- 每个 workspace 独立隔离，互不干扰
- `SkillLinkManager` prepare/cleanup 在 worker 线程内完成
- `AgentRunner.run()` 是进程启动操作（非 shared state），线程安全

---

## File Change Summary

| File | Change | Purpose |
|------|--------|---------|
| `skills/triton/triton-npu-optimize/references/opt-note-format.md` | 修改 | W1: Overall Summary 模板增加 Key optimization points |
| `src/triton_agent/hardware_info.py` | 新增 | W2.1: 硬件信息采集 |
| `src/triton_agent/optimize/orchestration.py` | 修改 | W2.2: optimize 启动时写入 env-info.json |
| `skills/triton-npu-report/SKILL.md` | 新增 | W3.1: report 工作流说明书 |
| `skills/triton-npu-report/references/report-format.md` | 新增 | W3.2: 中文报告模板 |
| `src/triton_agent/models.py` | 修改 | W4.1: REPORT command kind |
| `src/triton_agent/cli.py` | 修改 | W4.1: 注册 report 命令 |
| `src/triton_agent/skill_staging.py` | 修改 | W4.1: 新增 StageRule |
| `src/triton_agent/prompts.py` | 修改 | W4.2: 新增 prompt intro |
| `src/triton_agent/commands/report.py` | 新增 | W4.3: handler |
| `src/triton_agent/optimize/resume.py` | 修改 | W5: reset 时删除 env-info.json |
| `src/triton_agent/cli.py` | 修改 | W7.1: REPORT_BATCH 增加 has_agent |
| `src/triton_agent/commands/batch_report.py` | 修改 | W7.2: 增强 handle_batch_report，per-workspace agent 启动 |

---

## Agent 流程对比

| | optimize | report | report-batch (per-workspace) |
|---|---|---|---|
| **驱动方式** | Agent 分轮次迭代优化算子 | Agent 一次性读取 workspace 生成报告 | 批量：每个 workspace 启动一次 report agent |
| **核心技能** | `triton-npu-optimize` | `triton-npu-report` | `triton-npu-report` |
| **Agent 职责** | 维护 opt-note.md, round-state.json, 修改算子 | 阅读已有数据，生成 report.md | 同 report |
| **是否修改算子** | 是 | 否（只读） | 否（只读） |
| **是否分轮** | 是（多轮 loop） | 否（单次任务） | 否（单次任务） |
| **--interact** | 支持 | 支持 | 不支持（非交互） |
| **--show-output** | 支持 | 支持 | 支持 |
| **输出** | 多轮优化产物 | report.md | 每个 workspace 一个 report.md + batch 级别 report-batch.md |

---

## Decisions

| # | Question | Decision |
|---|----------|----------|
| A | Key optimization points 是否加入 opt-note-format.md？ | **是**，加入 Overall Summary 模板 |
| B | 硬件信息精度 | **精细型号**，通过 `npu-smi` + driver version 采集，存入 `env-info.json` |
| C | 延迟表粒度 | **平均值（geomean）** |
| D | 精度验证数据源 | `round-state.json` 的 `correctness_status` |
| E | 报告语言 | **中文** |
| F | report.md 位置 | **算子目录下**，与 `opt-note.md` 同级 |
| G | 旧 session 没有 Key optimization points | **暂不处理**，第 5 节展示「未记录」 |
| H | 硬件元数据文件命名 | **`env-info.json`** |
| I | `--reset-optimize` 时的策略 | **直接删除** `env-info.json`，下次 optimize 重新采集 |
| J | report 生成方式 | **Agent 驱动**，通过 `triton-npu-report` 技能 + agent 模型能力生成 |
