# Pattern 验证 Batch 操作说明

在 commit 分析 → skills 更新之后，用本流程验证 **Agent 能否复现人工优化点**。

设计细节见 [docs/specs/2026-05-30-pattern-validation-batch-design.md](../specs/2026-05-30-pattern-validation-batch-design.md)。

**核心原则：** synthesis 报告格式不固定；由 **agent 阅读并决策**，skill 只规定流程与产物契约。`generate_manifest.py` / `scaffold_batch.py` 仅为可选辅助。

---

## 推荐：一键全链路（agent 自动执行）

前提：目标仓库已有 `PERF_PATTERN_SYNTHESIS.md`（通常来自 `analyze-commit-perf`）。

```bash
cd /path/to/target-repo

uv run triton-agent pattern-validation-loop \
  -i . \
  --synthesis PERF_PATTERN_SYNTHESIS.md \
  --batch-dir pattern-validation-batch \
  --base origin/main \
  --min-rounds 10 \
  --max-iterations 5 \
  --show-output \
  --agent opencode
```

Agent 按 [skills/triton-npu-pattern-validation-loop/SKILL.md](../../skills/triton-npu-pattern-validation-loop/SKILL.md) 执行：

```text
init → 读 synthesis 更新 skills → agent 搭建 workspace → optimize-batch → audit → iterate
```

必读契约：

- [skill-update-contract.md](../../skills/triton-npu-pattern-validation-loop/references/skill-update-contract.md)
- [workspace-scaffold-contract.md](../../skills/triton-npu-pattern-validation-loop/references/workspace-scaffold-contract.md)

状态文件：`.triton-agent/pattern-validation-loop-state.json`

---

## Agent 在各阶段做什么

| 阶段 | Agent 职责 | 脚本辅助 |
|------|------------|----------|
| 读 synthesis | 通读报告（不要求 G1-I1 表格）；必要时读 `PERF_KNOWLEDGE_BASE.md` | — |
| 更新 skills | 决定 extend / 新建 / 跳过；改 staged pattern cards | `build_pattern_index.py` |
| 搭建 batch | 选算子、Git 取优化前快照、找 test、写 `validation-meta.json` | 可选 `scaffold_batch.py` |
| optimize | 调用 `triton-agent optimize-batch` | — |
| audit | 跑 `audit_batch.py --archive-passed`；通过的移入 `_completed/` | `audit_batch.py` |
| 迭代 | 改 skills → 仅 reset **active** workspace → 重跑 | `reset_workspace_rounds.py` |

---

## 手动 fallback（同一流程）

### ① 整合 skills

Agent（或人）读 synthesis 后编辑 staged pattern cards，然后：

```bash
uv run python skills/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton-npu-optimize-knowledge/references/pattern_index.md
```

### ② 搭建 batch 目录（agent 主导）

每个 workspace 需包含：

```text
chunk_o/
  chunk_o.py              # base..HEAD 内首次改动前的快照（agent 用 git show 提取）
  test_*.py               # agent 在仓库中搜索并复制
  bench_*.py              # 可选
  validation-meta.json    # agent 填写 expected_patterns + synthesis_refs
```

`validation-meta.json` 示例见 [workspace-scaffold-contract.md](../../skills/triton-npu-pattern-validation-loop/references/workspace-scaffold-contract.md)。

手写的 batch 索引示例：`docs/fixtures/q2tritonkernel-pattern-validation.manifest.json`

若已写好 manifest，可选用：

```bash
python3 skills/triton-npu-pattern-validation-loop/scripts/scaffold_batch.py \
  --manifest pattern-validation-batch/manifest.json \
  --output pattern-validation-batch
```

### ③ 跑 optimize-batch

```bash
triton-agent optimize-batch \
  -i pattern-validation-batch \
  --resume fresh \
  --reset-optimize \
  --min-rounds 10 \
  --max-concurrency 1 \
  --show-output \
  --use-repo-staged-knowledge \
  --agent opencode
```

### ④ 审计 + 归档 + 迭代

```bash
python3 skills/triton-npu-pattern-validation-loop/scripts/audit_batch.py \
  --batch-root pattern-validation-batch \
  --archive-passed \
  --json > pattern-validation-batch/audit-report.json
```

- 审计 **active** 目录（batch 根下带 `validation-meta.json` 的子目录）
- 全部 pattern 命中的 workspace 自动移到 `pattern-validation-batch/_completed/<name>/`
- 下次 `optimize-batch -i pattern-validation-batch` **不会**再调度 `_completed/` 里的算子
- 整个 loop 完成条件：`audit-report.json` 里 `active_remaining` 为空

skills 未生效时（只处理 active；`_completed/` 不动）：

```bash
python3 skills/triton-npu-pattern-validation-loop/scripts/reset_workspace_rounds.py \
  --batch-root pattern-validation-batch
```

---

## 成功标准（建议）

| 项 | 标准 |
|----|------|
| Pattern 引用 | audit 脚本 `missing_patterns` 为空 |
| 归档 | 通过后位于 `_completed/<workspace>/`，不再参与后续 optimize-batch |
| 机制 | summary 里能看到 synthesis 对应改动 |
| 性能 | 有 NPU 时相对 baseline 有提升（可选） |

---

## 注意

- 不要把 repo-local-only 教训写进 `expected_patterns`。
- pre-opt 快照应对齐 analyze-commit-perf 使用的 `base_revision`。
- 复杂 kernel 可能需 agent 额外复制同目录依赖 `.py`。
