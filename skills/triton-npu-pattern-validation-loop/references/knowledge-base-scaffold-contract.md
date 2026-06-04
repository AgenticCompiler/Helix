# Knowledge-Base-Driven Workspace Planning

Use this contract when `PERF_KNOWLEDGE_BASE.md` exists in the target repo (typical output of
`analyze-commit-perf`). It refines [workspace-scaffold-contract.md](workspace-scaffold-contract.md)
Step 1 for repos where **one source file contains many `@triton.jit` kernels** and perf commits
in the knowledge base are **kernel-scoped**, not file-scoped.

## Goal

Before copying files into `pattern-validation-batch/`:

1. Read `PERF_KNOWLEDGE_BASE.md` and **split perf lessons per kernel** (not per source file).
2. For each kernel, find the **host launch function(s)** in repo source that call it.
3. Build **one optimize workspace per launch function**, named after the **launch function** (e.g., `chunk_bwd_dv_local`).
4. If **one launch function** calls **multiple kernels** (branches / pipeline), **merge** those
   kernels into **one** operator file under that launch-function-named workspace.

## Naming rules (required)

| Artifact | Name |
|----------|------|
| Workspace directory | `<launch_function_name>/` (for example `chunk_bwd_dv_local/`) |
| Operator file | `<launch_function_name>.py` (same stem as directory) |
| `validation-meta.json` → `kernel_name` | Same string as directory (the launch function name) |
| `validation-meta.json` → `launch_functions` | Host entrypoint(s) tests call (for example `chunk_bwd_dv_local`) |

Do **not** name workspaces after the source file (`chunk_o.py`) when that file validates multiple
launch functions independently.

## Pre-optimization Baseline Rule (Critical)

Always use the **pre-optimization** kernel and launch function as the optimization starting point. 
Do **not** scaffold or create separate workspaces for post-optimization specialized kernels (such as `chunk_gated_delta_rule_bwd_kernel_dhu_k128_blockdim128` or `chunk_gated_delta_rule_fwd_kernel_h_k128_blockdim128` which were added in later performance commits). 
- The workspace and operator file must be based on the **original, pre-optimization** kernel (e.g., `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` or `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`).
- Any specialized variants (like K=128 or K=64 variants) added during the commit history are the *results* of optimization. The optimize agent should be starting from the unoptimized baseline and re-applying or re-discovering those optimizations, not starting with the already-specialized kernels.
- When generating the workspace plan, the CLI automatically scans the **pre-optimization snapshot** (the base revision or the state before the first in-range commit touching that file) to identify the original kernels and launch functions. Always respect this plan.

## Step 0 — Generate a workspace plan

From repo root:

```bash
triton-agent pattern-validation-plan -i "$REPO" \
  --knowledge PERF_KNOWLEDGE_BASE.md \
  --batch-dir pattern-validation-batch \
  --base "$BASE"
```

`pattern-validation-loop` runs this automatically when `PERF_KNOWLEDGE_BASE.md` exists.
The prepare agent may re-run the same command if the plan is missing or stale.

The script:

- Parses `## File Analyses` sections in the knowledge base and groups commit lessons by
  **kernel symbol** mentioned in each commit block.
- Scans each referenced `source_path` for `@triton.jit` definitions and host functions that
  launch them.
- Emits one planned workspace per **launch function**, with `workspace` / `kernel_name` set to
  the **launch function name** (for example `chunk_bwd_dv_local`).

Review the JSON plan before scaffolding. Fix missing launch mappings manually in the plan or
in repo source/tests.

## Step 1 — Map kernels from the knowledge base

In each `### <source_path>` section under `## File Analyses`:

- Read every `##### <commit> …` block.
- Extract kernel symbols from **What changed** (backticks and `*_kernel` identifiers).
- Bucket lessons under each **kernel name**, not under the file path.

Example from `chunk_o.py` analysis:

| Kernel (lesson bucket) | Typical launch function |
|------------------------|-------------------------|
| `chunk_bwd_kernel_dv_local` | `chunk_bwd_dv_local` |
| `chunk_bwd_kernel_dqkwg` | `chunk_bwd_dqkwg` |
| `chunk_fwd_kernel_o` | `chunk_fwd_o` |
| `chunk_bwd_kernel_dv` | `chunk_bwd_dv` |

Pattern links and reusable rules attached to a commit apply to every kernel named in that
commit's **What changed** section.

## Step 2 — Resolve launch functions in source

For each planned workspace:

1. Open the pre-opt snapshot of `source_path` (Git procedure in workspace-scaffold-contract Step 2).
2. Confirm `launch_functions` from the plan (or discover them by searching `kernel_name[` inside
   host `def` bodies).
3. Trace the **full call chain** for that launch:
   - every `@triton.jit` kernel the launch calls (including both branches of an `if` / `else`)
   - host helpers and constants only that launch needs
4. If two kernels appear only because one launch branches, keep **both** in the **same**
   operator extract — still **one** workspace directory named after the **launch function**.

**Do not** create separate workspaces for inner kernels that share a single test entrypoint.

## Step 3 — Build `$BATCH/<launch_function_name>/`

For each entry in `workspace-plan.json`:

```text
pattern-validation-batch/chunk_bwd_dv_local/
  chunk_bwd_dv_local.py    # minimal extract: launch + its kernel(s) + helpers
  test_*.py.txt          # reference only (dtype/shapes); not runnable pytest
  validation-meta.json
  deps/                           # literal directory name deps (not {deps}); often empty when repo-path sync works
    fla/                          # optional: copied only when import smoke fails or --copy-deps
    ...
```

`validation-meta.json` must include:

```json
{
  "workspace": "chunk_bwd_dv_local",
  "kernel_name": "chunk_bwd_dv_local",
  "launch_functions": ["chunk_bwd_dv_local"],
  "kernels_in_operator": ["chunk_bwd_kernel_dv_local"],
  "source_path": "src/kernels/fla/ops/common/chunk_o.py",
  "operator_filename": "chunk_bwd_dv_local.py",
  "knowledge_lessons": ["85171374766b", "a8cb0ffb2f7d"],
  "expected_patterns": ["layout-materialization-elision"],
  "dependency_dir": "deps",
  "dependency_strategy": "repo_path",
  "repo_path_injected": true,
  "import_smoke_passed": true,
  "copied_dependencies": []
}
```

- `knowledge_lessons`: commit title prefixes or SHAs from the knowledge base that apply to this
  kernel (from the plan).
- `expected_patterns`: from synthesis / pattern promotion, filtered to this kernel's lessons.

## Step 4 — Cross-check with synthesis

`PERF_PATTERN_SYNTHESIS.md` still drives **which patterns** to put in `expected_patterns`.
`PERF_KNOWLEDGE_BASE.md` drives **how many workspaces** and **which kernel/launch mapping**.

When synthesis mentions a file (for example `chunk_o.py`) but the knowledge base shows multiple
kernels, **split** into multiple kernel-named workspaces — do not copy the whole file once.

## Merge rule (one launch, multiple kernels)

When a single launch function contains:

```python
if cond:
    kernel_a[grid](...)
else:
    kernel_b[grid](...)
```

Then:

- **One** workspace directory: named after the kernel you treat as primary (plan script picks
  the first launch in source order; override in the plan JSON if tests imply otherwise).
- **One** operator file containing `kernel_a`, `kernel_b`, the launch function, and shared helpers.
- `kernels_in_operator`: `["kernel_a", "kernel_b"]`
- `launch_functions`: `["the_launch_fn"]`

## After planning

Continue with [workspace-scaffold-contract.md](workspace-scaffold-contract.md) Step 2–6 (pre-opt
snapshot, tests, `deps/`, verify CLI, record_iteration).
