# Optimize 编排层(Stage Gate)设计规格

> 状态:**已实现**(2026-07-06)。本文档归档当前系统的形态与决策依据。
> 配套:`docs/plans/2026-04-09-optimize-analysis-driven-design.md`(分层分析)、`docs/plans/2026-06-05-optimize-batched-round-mode.md`(批处理 round 控制器)。
> 上游分析:`/data1/luchang/triton-npu-pattern-classification.md`(40 pattern 六维分类)、`/data1/luchang/triton-npu-pattern-restructure-design.md`(重构提案)、Xe-Forge 论文+源码对照。

## 一、目标与背景

### 1.1 要解决的差距

对照 Xe-Forge(多 stage 分离 + 硬依赖 + issue 驱动 skip + 再分析),`triton-agent` 的核心差距是 **#0 编排层缺位**:40 个 pattern 平铺成 `pattern_index.md`,Priority 1-4 是咨询性散文,agent 可自由跳层(例如 round 1 就试 autotune 而跳过结构融合)。结果是"问题平铺、agent 随便试、跳层"。

### 1.2 设计决策(已拍板)

经讨论确认三条关键决策:

1. **Gate 模型,不是 Plan 模型**。`triton-npu-optimize` skill 有一条硬规则(SKILL.md ~L254):"禁止前瞻性优化计划——下一方向必须由新鲜 profiling/benchmark 证据驱动"。Xe-Forge 的" upfront 多 stage 计划"与之冲突。因此编排层**不提交多轮计划**,只对**当前轮**的允许集合做依赖门控。
2. **硬阻断(选项 B),不是软提示**。事后驳回,不是事前阻止——这是架构决定的硬上限:CLI 把 agent 当子进程跑,子进程启动后 CLI 既看不见也拦不住它做什么,所以事前阻止做不到(除非走 DSPy"pipeline 自己跑 stage",我们没选)。但**事后驳回**是真硬杠杆:在 `check_batch_round` 加 per-round gate 合规检查,违规 round(声明了未解锁的下游 stage)即使过了 correctness+benchmark 也**驳回**——不晋升为 baseline、不解锁下游、下轮 prompt 告知"上轮因 gate 被驳,先做前置"。配 pre-run 指派式 prompt(`build_round_plan_lines` 直接指派 `selected_hint` 为本轮 stage)降低违规频率。
3. **Minimal 检测范围**。v1 只做:代码结构 AST 扫描(机械反模式)+ agent 写的 `issues.json`(闭枚举,脚本校验+coerce)。`report.txt` Cat1-5 解析器、IR performance-signals JSON 通道作为后续。硬阻断要求检测不失真(否则误杀合法 round),所以扫描器扫**最新 accepted round 的算子**而非输入算子,保持会话内失真不发生。

### 1.3 不做的事(显式 out of scope)

- stage 内迭代控制(best-of / 2% 门 / 换思路 / 有界契约)——CoVeR 差距,不抄
- 验证 cascade、KB 注入(`format_for_stage`)、DSPy、硬件查询、best-of-k

## 二、架构:硬 Gate(选项 B)= pre-run 指派 + post-run 驳回

| 环节 | 机制 | 软/硬 |
|---|---|---|
| CLI 算 stage | `planner.plan_round` Python 逻辑 | 硬(算出来是啥就是啥) |
| 把 stage 告诉 agent | `build_round_plan_lines` 拼指派式 prompt 注入 | 软(文字,agent 可无视) |
| agent 实际做什么 | agent 自由 | 不受 CLI 控制 |
| 判 round 是否违规 | `check_batch_round` 里 `_enforce_round_gate` Python 逻辑 | 硬(代码判) |
| 违规 → 驳回 | 复用既有 reject 机制(不晋升/不解锁/下轮告知) | 硬(代码执行) |

**Pre-run(每 batch 起始,`_build_worker_batch_request`)**:
1. `scan_kernel_issues` skill-script 扫**最新 accepted round 的算子**(无则输入算子)→ 机械 issue
2. 读上一轮 `opt-round-N/issues.json`(agent 语义 issue)
3. 合并 → 按 `stages.json` 路由 → A5 硬件门控过滤
4. `plan_round` 算出 actionable/blocked/skipped + `selected_hint`
5. `build_round_plan_lines(plan)` 指**派** `selected_hint` 为本轮 stage,注入 worker prompt
6. plan 存到 `self._current_batch_plan` 供 post-run 用

**Post-run(每 batch 结束,`check_batch_round`)**:对 batch 内每个**过了 correctness+benchmark** 的 round,按序:
1. 读 `round-state.json:stage`
2. `decide_round_gate(stage, addressed_base, skipped, intra_batch_addressed)` 判是否合规
3. 合规 → stage 加入 `intra_batch_addressed`(batch 内累加,下个 round 的 gate 能看到),并写 `opt-round-N/stage-addressed.json` marker(下个 batch 的 gate 扫这些 marker 重建 `stages_addressed`,**不依赖 workflow state 文件**)
4. 违规 → round status 翻成 `fail` + 加 `gate violation` issue("round declared stage X but requires prereq Y first")——不晋升、不解锁、`has_failures=True`
5. round 未声明 `stage` → 不驳回(过了 c+b),但不记入 `intra_batch_addressed`(下游保守不解锁)——这是要求 agent 填 `stage` 的硬激励

**Re-analysis**:每 batch 重扫最新 accepted 算子 → 一次结构重写若引入新 memory_access issue,下轮 `memory_access` 离开 skipped、进入 actionable;若修掉了某 stage 的 issue,该 stage 进 skipped、解锁下游。

**与"无前瞻计划"规则的兼容性**:gate 只约束当前 batch 的允许集合(由新鲜证据算出),不承诺未来轮次。指派是 per-batch 起始快照;严格 per-round 指派需 `--round-batch-size 1`(SKILL.md 已说明)。post-run 的 `intra_batch_addressed` 累加使 batch 内进度仍被 gate 跟踪。

## 三、Stage 分类与依赖图

### 3.1 8 个 stage(采纳用户分析的 3 个 + 补 5 个,每个 stage 挂 `patterns` 字段)

| stage | P | 来源 | patterns |
|---|---|---|---|
| `algorithmic` | 1 | 补(纯算法/内核结构重写,最深,先做) | classic-matmul, algebraic-optimization, shift-2d-mask-to-1d-index-stream |
| `boundary` | 1 | **用户 stage1** 算子边界重构(compute fusion + layout copy elim + aux-op-fusion;主手段见 SKILL.md 结构优先级门) | auxiliary-op-fusion, layout-materialization-elision, reduce-avoid-transpose-copy |
| `parallel_decomposition` | 2 | **用户 stage2** 并行分解 | program-multiple-rows, parallel, grid-flatten-and-ub-buffering, atomic-contention-owner-computes-store |
| `memory_access` | 2 | **用户 stage3** 内存访问重构(4 手段:碎片事务融合 / load 顺序 / 批量搬运+片上拣选 / 地址计算重构 + footprint + locality) | merge-adjacent-stores, accumulator-layout-alignment, remove-implicit-transpose, reorder-load, discrete_memory_access, sliding-window-inner-w-slab-gather, stencil-resize-gm-to-ub-staging, slice_coalesce, flat-index-decode-tiling, padded_row_col_copy, block-pointer-dimensionality, diagonal, tiling, effective-extent-tiling, slice_intermediate |
| `scalar_control` | 2 | 补(纯标量/控制开销,非地址类) | vec-cmp, loop-invariant-hoisting, scalar-latency-traps, exact-tile-no-boundary-fast-path |
| `pipeline` | 2 | 补(传输-计算重叠;reorder-load 按用户放 memory_access) | software-pipeline, software-pipeline-dependency-profiling, static-range-to-range |
| `micro_tuning` | 3 | 参数搜索 + A5 + attention | autotune, tile-selection-heuristic, a5-simt-sliding-window-tuning, a5-force-simt-only-discrete-access, simt-clip-window-closed-reduction, attention-cv-pipeline |
| `compile_hints` | 4 | 降级提示 | compile_hints |

39 个 pattern 归位(第 40 个 `scalar-vector-simulation-signal` 是诊断卡,非动作,不挂 stage)。`stages.json` schema v2,每个 stage 带 `patterns: [...]`;`build_round_plan_lines` 在指派 stage 时一并注入该 stage 的候选 pattern 列表(补上"stage→pattern"那一跳,agent 被指派到 stage 后直接拿到候选 pattern,不用自己翻索引)。

**关键分类决策(严格按用户分析)**:`remove-implicit-transpose`/`accumulator-layout-alignment` → memory_access(碎片事务融合),而 `layout-materialization-elision`/`reduce-avoid-transpose-copy` → boundary(layout copy elim,wrapper 级物化消除);`reorder-load` → memory_access(load 顺序,用户定的);`flat-index-decode`/`padded_row_col_copy`/`block-pointer-dimensionality` → memory_access(地址计算重构)。A5-SIMT pattern 不打成独立 stage,以 `hardware_scope` 标签挂在所属 stage 上,由硬件门控过滤。

### 3.2 硬依赖边(17 条,"递减语义范围")

```
algorithmic → {boundary, parallel_decomposition, memory_access, scalar_control, pipeline, micro_tuning}
boundary    → {parallel_decomposition, memory_access, scalar_control, pipeline, micro_tuning}
parallel_decomposition → micro_tuning
memory_access → pipeline, micro_tuning
scalar_control → micro_tuning
pipeline → micro_tuning
micro_tuning → compile_hints
```

链路:`algorithmic → boundary → {parallel_decomposition, memory_access, scalar_control} → pipeline(需 memory_access) → micro_tuning(需 6 个前置) → compile_hints`。`algorithmic` 是唯一无前置的 stage(永远可跑);`compile_hints` 只直接依赖 `micro_tuning`(后者传递带入 algorithmic),已修掉冗余的 `algorithmic→compile_hints` 边使 redirect 目标更精准。

**门控语义**:stage `s` 可跑 ⟺ `prereqs(s) ⊆ (stages_addressed ∪ exhausted ∪ stages_with_no_issues_this_round)`。即一个前置要么在先前 round 被处理过、要么被耗尽(exhausted,见下)、要么本轮无 issue(自动视为已解决)。只有 `algorithmic` 无前置,永远可跑。

### 3.3 Spinning guard(防同一 stage 空转)

实测暴露:gate 只"阻跳层"不"强制推进",agent 可能在同一 stage 连续多轮无进展(实测 4_Abs 在 micro_tuning 上转了 13 轮,round 5 之后全是 ≤1.01x 的噪声级 gain)。为此加 **exhaustion**:

- **per-round 进展判定**:gate 批准一个 round 时,调 `local_optimum_check.round_geomean_speedup(round_dir, baseline_perf_path)`(复用既有 perf 对比,不重新实现)拿本轮 baseline-relative geomean speedup;读上一轮 marker 里存的 speedup;adjacent gain = 本轮 − 上轮。`gain <= 0.02`(2% 噪声门)→ "no progress"。speedup 不可测时**保守判 progress=True**(测量失败绝不误触发 exhaustion)。
- **marker 记 progress + speedup**:`opt-round-N/stage-addressed.json` 现存 `{"stage", "progress": bool, "speedup": float|null}`。progress 默认 True(老 marker 无此字段时保守)。
- **exhaustion 判定**:`_get_exhausted_stages(workdir)` 按 round 号顺序扫 marker,跟踪"同 stage 连续 no-progress"计数;**progress round 或 stage 切换都重置计数**;计数 ≥ `_STAGE_EXHAUSTION_THRESHOLD`(默认 5)→ 该 stage 标 exhausted(**sticky**,一旦标了本 session 不解)。
- **exhausted 的语义**:`plan_round` 把 exhausted stage **从 actionable/selected_hint 排除**(不再指派),但**算 resolved**(下游依赖解锁)。即"别再指派它,但别因为它卡住下游"。`build_round_plan_lines` 注入 "Exhausted ..." 行告诉 agent。

**在 4_Abs 实测上的反事实推演**:micro_tuning round 5 是 1.04x(progress,重置计数),round 6-13 全 ≤1.01x(no progress)。threshold=5 → round 10(第 5 个连续 no-progress,即 round 6,7,8,9,10)标 exhausted,round 11 起不再指派 micro_tuning、改指 compile_hints(其前置 micro_tuning 已 exhausted=resolved)。即把 13 轮 spinning 砍到 10 轮、省 3 轮。要更早停把 `_STAGE_EXHAUSTION_THRESHOLD` 调到 3(round 8 停)。

## 四、结构化 issue 检测(Minimal)

### 4.1 闭枚举 `IssueType`

34 个 issue 类型,定义在 `src/triton_agent/optimize/issue_detection.py` 的 `IssueType` enum,且必须与 `stages.json` 的 `issue_routing` 键 1:1 同步(由 `tests/test_optimize_issue_detection.py::IssueTypeContractTests` 强制,防漂移)。

### 4.2 两条检测通道

**通道 A:代码结构扫描器**(`skills/triton/triton-npu-optimize/scripts/scan_kernel_issues.py`,skill-script,不 import triton_agent)。纯 stdlib `ast`+`re`,高精度机械反模式:

- `permute_contiguous_materialization`(regex 精确)
- `implicit_transpose_in_dot`(`tl.trans`+`tl.dot` 共现)
- `static_range_unroll`(regex 精确)
- `flat_1d_index_decode`(AST: jit 核内同时有 `//` 和 `%` + arange + program_id;**要求 div 与 mod 同时出现**,边界 mask 的 lone `%` 不误报)
- `invalid_num_warps`(regex,非 2 幂)
- `missing_autotune`(有 `@triton.jit` 无 `@triton.autotune`)
- `missing_max_contiguous` / `missing_multiple_of`(weak signal,低 severity)
- `wrapper_loop_per_launch`(AST:`for` 体里有 Subscript-call `kernel[grid](...)`)
- `manual_k_reduction`(AST:jit 核内 `for` 体有 `+=` 累乘)

扫描器对 SyntaxError 退化到 regex-only,单 detector 异常不中止整体扫描,按 `(issue_type, location)` 去重。

**通道 B:agent 语义 issue**(`opt-round-N/issues.json`)。agent 在 triage 后写 JSON 数组 `{issue_type, severity, location, description, suggested_fix}`,`issue_type` 必须用闭枚举值。`validate_and_coerce` 仿 Xe-Forge `_coerce_issue`:精确 enum → 规范化(小写、`-`/空格→`_`)→ 未知丢弃+warning。`open_ended` 允许带 `suggested_stage` 覆盖默认路由。

### 4.3 合并与路由

`merge_issues(scanner, agent)` → `route_issues(graph)` 按 `stages.json` 分组到 stage。无 issue 的 stage 自动进 skipped(依赖视为已解决)。

## 五、per-batch 编排流程

```
# ---- pre-run (_build_worker_batch_request) ----
addressed = _get_stages_addressed_from_rounds(workdir)             # 扫 opt-round-*/stage-addressed.json marker
kernel    = _latest_accepted_round_dir(workdir) → inspect_round_artifacts().operator_path  # 无则输入
scanner_issues = scan_kernel_issues_module().scan(kernel)            # 经 skill_contract 桥
agent_issues   = load_agent_issues(most_recent_round_dir)
issues = filter_hardware_gated(merge_issues(scanner_issues, agent_issues), target_chip)
routed = route_issues(issues, graph)
plan   = plan_round(routed, addressed=addressed, graph=graph)       # 含 selected_hint
self._current_batch_plan = plan
prompt = build_optimize_round_prompt(...) + build_round_plan_lines(plan)  # 指派式
worker_result = self._run_request(replace(request, prompt=prompt))

# ---- post-run (check_batch_round, 对每个过 c+b 的 round 按序) ----
stage = _read_round_stage(round_dir)
decision = decide_round_gate(stage, addressed_base=plan.addressed, skipped=plan.skipped, intra_batch_addressed)
if decision.allowed:
    intra_batch_addressed.add(stage)                                # batch 内累加
    progress, speedup = _round_progress(round_dir)                 # 复用 local_optimum_check 算 adjacent gain
    _write_stage_addressed_marker(round_dir, stage, progress, speedup)  # marker 现含 progress+speedup
else:
    round.status = "fail"; round.issues = ["gate violation: ..."]   # 硬驳回
    _clear_stage_addressed_marker(round_dir)                       # 清旧 marker(防修复轮残留)
# 无单独"批次后记录"步骤——marker 在 _enforce_round_gate 内就写了
```

pre-run 还多一步:`exhausted = _get_exhausted_stages(workdir)`(扫 marker 的 progress,按 §3.3 算),传给 `plan_round(exhausted=...)`。`selected_hint` 现在是**指派**(最低 priority_level + 最高 severity,且排除 exhausted),`build_round_plan_lines` 直接告诉 agent "Assigned stage: X" + Exhausted 列表。`decide_round_gate` 是 post-run 硬判(基于 `graph.gate` + intra-batch 累加)。**addressed 的来源是 `opt-round-N/stage-addressed.json` marker**(只有 gate 判 pass 的 round 才有),不是 workflow state——这修掉了"state.json 被 session cleanup 删 → stages_addressed 丢失"的 bug(实测暴露过:round-1 声明了 micro_tuning 但 round-2 仍被判"micro_tuning 未 addressed")。marker 与 round-state.json:stage(声明)区别:marker 反映 **gate 裁决 + 进展**,round-state 反映 **agent 声称**;一个 gate-rejected round 即使 c+b passed 也没有 marker,不算 addressed。

**硬阻断的边界**:只能"驳回违规 round",不能"阻止 agent 尝试"。对非对抗 agent(我们的情况),违规 = 白干一轮 + 下轮被纠正,几轮后学会遵守。对抗性规避(声明假 stage)未做交叉校验——`summary.md` 的 pattern→stage 反查是后续 hardening。

## 六、状态追踪

- `state.json`(经 `optimize_workflow_state.py` skill-script)曾新增 `stages_addressed: list[str]`,但**实测暴露 bug**:该文件在 session 末被 `OptimizeSessionArtifactsManager` cleanup 删除,导致 `stages_addressed` 丢失、round-1 声明的 stage 不被后续 round 的 gate 看到(整个 run 的"已处理"解锁路径死的,只靠 auto-skip 侥幸跑通)。**已改:gated 的 addressed 现以 `opt-round-N/stage-addressed.json` marker 为源**(持久、不被清理),`state.json:stages_addressed` 字段保留作只读展示但 gate 不再依赖。`_record_addressed_stages` 已删,marker 由 `_enforce_round_gate` 在判 pass 时直接写。
- `record_stage_addressed` 幂等去重、保序。
- `round-state.json` 契约(`contract.json` + `RoundState` dataclass)新增 optional `stage` 字段;`artifacts.md` 已用 `update-artifacts.py` 重生成同步。

## 七、文件清单

### 新增(CLI 编排逻辑)
- `src/triton_agent/optimize/stages.py` — `Stage` enum(8:algorithmic/boundary/parallel_decomposition/memory_access/scalar_control/pipeline/micro_tuning/compile_hints)、`StageDescriptor`(含 `patterns`)、`StageGraph`(`prereqs`/`allowable_stages`/`blocked_stages`/`gate`/`has_cycle`)、`load_stage_graph`/`default_stage_graph`
- `src/triton_agent/optimize/issue_detection.py` — `IssueType`/`Issue`/`validate_and_coerce`/`route_issues`/`filter_hardware_gated`/`load_agent_issues`/`merge_issues`
- `src/triton_agent/optimize/planner.py` — `RoundPlan`/`StageIssueSummary`/`plan_round`/`_pick_hint`/`GateDecision`/`decide_round_gate`(post-run 硬判)

### 新增(skill-script,不 import triton_agent)
- `skills/triton/triton-npu-optimize/scripts/scan_kernel_issues.py` — AST/regex 扫描器
- `skills/triton/triton-npu-optimize/references/stages.json` — 机器可读契约(schema v2:stage 定义+`patterns`+依赖边+issue 路由+硬件门控)

### 新增(测试)
- `tests/test_optimize_stages.py`(21,含 every-stage-has-patterns)、`tests/test_optimize_issue_detection.py`(21)、`tests/test_scan_kernel_issues.py`(12)、`tests/test_optimize_planner.py`(18,含 `decide_round_gate` + patterns 注入)、`tests/test_optimize_stage_state.py`(13)、`tests/test_optimize_round_gate_helpers.py`(9,`_latest_accepted_round_dir`/`_read_round_stage`)

### 修改
- `src/triton_agent/optimize/execution.py` — imports + `_compute_round_plan`(addressed 从 marker 扫)/`_scan_current_kernel`/`_current_kernel_path`(扫最新 accepted 算子)/`_latest_accepted_round_dir`/`_load_prior_round_issues`/`_most_recent_round_dir`/`_read_round_stage`/`_enforce_round_gate`(post-run 硬驳回 + pass 写 marker / reject 清 marker)/`_write_stage_addressed_marker`/`_clear_stage_addressed_marker`/`_get_stages_addressed_from_rounds`;`_build_worker_batch_request` 注入指派式 plan + 存 `self._current_batch_plan`;`check_batch_round` 每 round 调 `_enforce_round_gate`。`_record_addressed_stages` 已删(marker 在 gate 内直接写,不再批次后记)。
- `src/triton_agent/optimize/prompts.py` — `build_round_plan_lines(plan, graph)`(指派式 + enforcement 警告 + 注入 assigned stage 的 patterns 列表)
- `src/triton_agent/optimize/workflow_state.py` — `record_stage_addressed_in_workflow_state`/`get_stages_addressed_from_state` 桥(**gate 不再依赖**;保留作只读展示)
- `src/triton_agent/optimize/skill_contract.py` — `scan_kernel_issues_module()` 桥声明
- `skills/triton/triton-npu-optimize/scripts/optimize_workflow_state.py` — `stages_addressed` schema + `record_stage_addressed`/`get_stages_addressed` + 校验
- `skills/common/ascend-npu-optimize-submit-round/references/contract.json` + `scripts/optimize_submit_round_contract.py` — optional `stage` 字段
- `skills/triton/triton-npu-optimize/references/artifacts.md` — 重生成(含 `stage`)
- `skills/triton/triton-npu-optimize/SKILL.md` — "Stage gate (programmatic, hard-enforced)" 节(agent 角色 picker→executor;stage 序更新为 algorithmic→boundary→...;说明 patterns 注入)

## 八、遵循的约束

- **skill-script 不 import triton_agent**;runtime 经 `skill_loader.load_skill_script_module` 桥接(AGENTS.md L23-25)
- **机器可读契约单一源**(`stages.json`,AGENTS.md L46);`IssueType` enum 与之 1:1 契约测试
- **无前瞻计划**(gate 模型尊重 SKILL.md ~L254)
- **复用既有原语**:`AgentRunner.run`/`check_batch_round`/`_run_request`/`compute_range_progress`/`OptimizeCheckResult`,无新 agent-launch 路径
- **防御式**:所有编排 hook 包 try/except,任何检测/路由异常返回 None/空,绝不阻塞既有 optimize 流程
- **修改 contract.json 后重跑 `update-artifacts.py`**(AGENTS.md 要求,已执行)

## 九、测试与验证状态

- ✅ Ruff:所有改动文件 clean
- ✅ 交付测试:110 passed(stages 21 + issue_detection 21 + scanner 12 + planner 19 + stage_state 13 + round_gate_helpers 21 + contract 3)
- ✅ 运行时导入:所有集成模块 `import` 成功,无循环依赖;`default_stage_graph()` 正确加载 8 stages/17 deps/34 routings/39 patterns 归位
- ⚠️ Pyright:**无法运行**——沙箱 nodeenv/node.js 装不上(`RuntimeError: nodeenv failed`),环境限制非代码问题。运行时导入作部分替代
- ⚠️ 全量 `pytest tests/`:有**预先存在**的环境性失败(`torch` 未装、`tomllib` 需 3.11+ 而沙箱 3.10),与本次改动无关;optimize 相关测试全绿
- ⚠️ **无端到端验证**:逻辑层全绿,但"指派式 prompt + post-run 驳回"在真实 `triton optimize` 会话里是否真能改变 agent 行为,未观测过(无真实跑)。这是最大未验证点

## 十、后续(显式 deferred)

1. **端到端验证(最该先做)**——跑一个真实 `triton optimize`(建议 `--round-batch-size 1`),看 worker 输出是否引用"Assigned stage"、违规 round 是否被驳回、stage 序列是否按依赖图走。在观测前,硬 gate 的实际效果是黑盒
2. **`report.txt` Cat1-5 解析器**——`triton-npu-pattern-signal/references/report-txt-format.md` 是字段规格,`scalar-vector-simulation-signal.md` 的 Cat1-5 是检测规则。落地后作为第三条检测通道(仿真信号→域→issue),是当前最有价值的检测扩展
3. **IR performance-signals JSON 通道**——复用 `inspect_ir.py performance-signals` 的 `_PerformanceSignalsPayload`(vector/transfer/sync-heavy stages)
4. **对抗性 hardening**——`summary.md` 的 pattern→stage 反查,防止 agent 声明假 stage 规避 gate(目前依赖 agent 诚实声明,非对抗场景够用)
5. **验证 cascade / KB 注入 / stage 内 best-of**——独立差距,不在编排层范围

## 十一、与 Xe-Forge 的映射

| Xe-Forge | 本实现 | 差异 |
|---|---|---|
| `IssueType` enum + `get_stage_for_issue` 三层路由 | `IssueType` enum + `stages.json` issue_routing + `route_issues` | 我们多一层契约测试防漂移;无 keyword inference(只精确+规范化) |
| `_HARD_DEPENDENCIES` + `_enforce_dependencies` 拓扑校正 | `StageGraph.dependencies` + `gate` + post-run `_enforce_round_gate` 驳回 | Xe-Forge 校正 LLM 多 stage 计划;我们无前瞻计划可校正,改为**事后驳回违规 round**(pre-run 指派 + post-run reject) |
| pipeline 自己跑 stage(事前阻止) | 无法事前阻止(agent 子进程自由) | 架构差异:我们没走 DSPy;硬杠杆是 post-run reject,不是 pre-run prevention |
| `AnalyzerAgent`(DSPy Predict)语义检测 | agent 写 `issues.json` + 扫描器 | 我们无 DSPy/LLM SDK;agent 在闭枚举内自检,脚本 coerce |
| 每 stage 后再分析 | 每轮重扫当前核 | 粒度不同(我们 per-round,非 per-stage),但自动再分析语义一致 |
| issue-driven skip | stages_with_no_issues 自动 skipped | 语义一致 |
