# Workspace Scaffold Contract (Agent-Driven)

The orchestrating agent **plans and builds** validation workspaces. Do not assume
`PERF_PATTERN_SYNTHESIS.md` uses a fixed schema (for example `G1-I1` tables). Read the
report as natural language plus any structure it actually contains.

Optional helpers under `scripts/` (`scaffold_batch.py`, `generate_manifest.py`) exist
only when the agent already authored a manifest JSON. They are **not** the primary path.

## Inputs to read

Before choosing operators, read as needed:

| Source | Use |
|--------|-----|
| `PERF_PATTERN_SYNTHESIS.md` | Primary: which patterns to validate in skills |
| `PERF_KNOWLEDGE_BASE.md` | **Kernel-scoped** perf lessons, commits, and source paths — drives workspace count and naming |
| [knowledge-base-scaffold-contract.md](knowledge-base-scaffold-contract.md) | Required when `PERF_KNOWLEDGE_BASE.md` exists: split by kernel, map launch functions |
| Staged `pattern_index.md` | Pattern IDs to cite in `expected_patterns` |
| Git history `base_revision..HEAD` | Confirm which files changed and when |

Also use loop state (`base_revision`, `batch_dir`) from init.

## Step 0 — Plan from PERF_KNOWLEDGE_BASE (when present)

If `PERF_KNOWLEDGE_BASE.md` exists, follow
[knowledge-base-scaffold-contract.md](knowledge-base-scaffold-contract.md) **before** Step 1:

```bash
python3 "$SKILL/scripts/plan_workspaces_from_knowledge.py" \
  --knowledge PERF_KNOWLEDGE_BASE.md \
  --repo "$REPO" \
  --base "$BASE" \
  --output "$BATCH/workspace-plan.json"
```

Rules from that contract:

- Split knowledge-base lessons **per kernel**, not per source file.
- **One launch function → one workspace**; directory and operator file named after the
  **primary kernel** (for example `chunk_bwd_kernel_dv_local/chunk_bwd_kernel_dv_local.py`).
- If one launch function calls **multiple kernels** (branches), merge them into that single
  operator file.

## Step 1 — Select validation targets

When no knowledge base is available, decide **one optimize workspace per validation target**
as below. When `workspace-plan.json` exists, **each planned entry is one target**.

A validation target is usually a **public launch entrypoint** (PyTorch wrapper, host launcher,
exported API) together with the `@triton.jit` kernels and host helpers its call chain needs.
With knowledge-base planning, the workspace is named after the **primary kernel**, while
`launch_functions` in `validation-meta.json` records the host entrypoint tests call. Prefer:

- Kernels named explicitly in synthesis as perf-relevant or skill-promotion targets
- Source files with clear pre-optimization vs post-optimization story in the analyzed range
- Entries where updated skills should change optimize-agent behavior

When several unrelated launch entrypoints or independently validated kernels live in the
**same** repo `source_path`, create **separate** workspaces — one per synthesis target —
instead of one workspace for the whole file. When **one** launch entrypoint necessarily calls
**several** cooperating kernels, keep them in **one** workspace (see Step 2b). See
[Step 2b — Multi-kernel single source file](#step-2b--multi-kernel-single-source-file-manual-split).

Skip or do not require in audit:

- Repo-local-only lessons (not promoted to shared skills)
- Rejected or weak-evidence items
- Pure infra / test-only / doc-only changes with no operator optimize story

Record your operator selection in loop state when scaffold completes (Phase C Step 6).
Optionally keep a working note in `$BATCH/manifest.json` as you plan.

## Step 2 — Resolve pre-optimization snapshot

For each target `source_path` (path in repo, e.g. `src/kernels/foo/bar.py`):

**Goal:** the operator file content **before** branch perf work began, within
`base_revision..HEAD`.

**Critical Rule:** Always use the **original, pre-optimization** kernels and launch functions as the starting point. Do **not** include post-optimization specialized kernels (such as `chunk_gated_delta_rule_bwd_kernel_dhu_k128_blockdim128` or `chunk_gated_delta_rule_fwd_kernel_h_k128_blockdim128` which were added in later performance commits). The operator file content must represent the baseline before those optimizations were introduced.

Recommended Git procedure:

```bash
cd "$REPO"
git log "$BASE..HEAD" --reverse --format=%H -- "$source_path"
# Let FIRST = first commit hash in that list (if any)

# Preferred snapshot: parent of first in-range commit
git show "${FIRST}^:$source_path"

# If file did not exist at parent, try first in-range blob:
git show "${FIRST}:$source_path"

# If no in-range commits touch the file, try base revision:
git show "${BASE}:$source_path"
```

Write the snapshot to:

```text
$BATCH/<workspace>/<operator_filename>
```

`<workspace>` is the planned launch function name when using knowledge-base planning (for example
`chunk_bwd_dv_local`). Otherwise it is usually the operator stem (e.g. `chunk_o`).
`<operator_filename>` must be `<launch_function_name>.py` in the knowledge-base-driven layout.

You may call `scaffold_batch.py --manifest ...` **only after** you authored manifest JSON
yourself; never treat `generate_manifest.py` output as authoritative without review.

**Note:** `scaffold_batch.py` copies the **entire** pre-opt file into one workspace. Use it
only when `source_path` already maps to a single validation target. For multi-kernel source
files, follow Step 2b and build workspaces manually.

## Step 2b — Multi-kernel single source file (manual split)

Some repos place several `@triton.jit` kernels, launch wrappers, and shared helpers in one
`.py` file — or spread one launch path across a few sibling modules. `optimize-batch` still
requires **exactly one operator `.py` per workspace** (excluding `test_*`, `bench_*`,
generated artifacts). There is **no** supported automatic split helper in this skill — do
**not** rely on blind AST/script extraction. You split by reading the pre-opt snapshot(s) and
synthesis, then composing minimal runnable operator files yourself.

**Workspace boundary rule:** group by **what the test/bench actually launches**, not by
counting `@triton.jit` decorators. One workspace may legitimately contain **multiple**
kernels when a single validation target's call chain needs them all.

### When this section applies

Use manual split when **any** of the following is true:

- Synthesis names multiple independent kernels or mechanisms inside the same `source_path`
  that should be validated separately
- The pre-opt snapshot defines more than one `@triton.jit` kernel and tests/benches exercise
  them through **different** entrypoints
- Copying the whole file would let the optimize agent change the wrong kernel or unrelated helpers
- One launch entrypoint calls several kernels and you must extract the **full call chain**
  into one runnable operator file (see below)

When each kernel already lives in its own file with its own test (for example `chunk_o.py`,
`wy_fast.py`), skip this section and use Step 2 as written.

### How to choose workspace boundaries

Use synthesis plus the repo test/bench call graph. Do **not** split purely because a file
contains multiple `@triton.jit` definitions.

| Situation | Workspace strategy |
|-----------|-------------------|
| One test/bench calls entrypoint `forward_foo`, which launches kernels A → B → C in sequence | **One workspace**: entrypoint + kernels A, B, C + shared helpers in the call chain |
| Synthesis validates kernel A and kernel B as **independent** optimization stories; tests call them separately | **Separate workspaces**, even if A and B live in the same file |
| Entrypoint X uses kernel C; entrypoint Y uses kernel D in the same file | **Two workspaces**: (X + C + helpers) and (Y + D + helpers) |
| Small `@triton.jit` helper is only used inside kernel A's launch path | Keep it in **A's workspace**; do not give it its own workspace |
| Same helper kernel is reused by two unrelated entrypoints | Duplicate or inline the minimal helper in **each** workspace; do not merge unrelated entrypoints |
| Launch path imports kernels from **other repo files** | One workspace still, but extract/import the **full cross-file closure** (see below) |

When unsure, prefer the boundary that matches **how the copied test exercises the code**.

### One launch entrypoint, multiple cooperating kernels

This is common: a host wrapper or PyTorch `forward` orchestrates several `@triton.jit`
kernels (for example prepass → main compute → epilogue, or multi-stage reduction).

For this case:

1. Set `validation_target` to the **launch entrypoint** (for example `forward_gated_delta`,
   `run_chunk_scan`), not to only one inner kernel name.
2. Trace the **dynamic call chain** from that entrypoint:
   - every `@triton.jit` kernel it launches directly or indirectly
   - host-side grid setup, temporaries, and helpers those launches require
   - constexpr tables and small utils used only along this path
3. Put **all of that** into the single workspace operator file (or operator file + copied
   dependency modules listed in `copied_dependencies`).
4. Set `expected_patterns` to patterns synthesis expects anywhere in **this pipeline** (prep,
   main, epilogue, or cross-kernel tiling). Audit still passes if any round cites the
   expected IDs.
5. In `included_symbols`, list **every** kernel and the entrypoint, for example
   `["forward_gated_delta", "prep_kernel", "main_kernel", "epilogue_kernel"]`.
6. In `excluded_targets`, list entrypoints/kernels from the same source file that belong to
   **other** validation targets, not inner kernels that your entrypoint legitimately calls.

**Cross-file closures:** when the entrypoint imports kernels from sibling repo modules, fetch
each file's **pre-opt snapshot** (Step 2 Git procedure) and copy/extract only the symbols in
the call chain. Record every contributing repo path in `closure_source_paths` (Step 4).

**Do not** split cooperating kernels into separate workspaces when the repo test only exercises
them through one entrypoint — the optimize agent needs the runnable pipeline, and partial
extracts may not compile or may optimize the wrong stage.

### Planning (before writing files)

For each synthesis validation target in a shared `source_path` (or cross-file call chain):

1. **Name the target** — prefer the **launch entrypoint** (for example `forward_chunk_o`,
   `run_scan`); if synthesis only names an inner kernel, identify which entrypoint test uses
   to reach it.
2. **Map patterns** — list only the `expected_patterns` that apply to **this target's pipeline**,
   not every pattern mentioned for the whole file.
3. **Pick the launch surface** — the function tests/benches must call after the split (PyTorch
   wrapper, host launch helper, or exported API).
4. **Trace the dependency closure** — from that launch surface, list:
   - **every** `@triton.jit` kernel the entrypoint launches (there may be several)
   - host-side helpers, constexpr tables, small utils, and types the call chain needs
   - imports from other repo modules (sibling `.py`, package `__init__`, shared utils), with
     their repo paths when the closure spans multiple files
5. **Exclude** entrypoints and kernels belonging to **other** synthesis targets in the same
   source file (or unrelated pipelines), even if they sit nearby in the file.

Record the plan in each workspace's `validation-meta.json` (see Step 4).

### Composing the workspace operator file

Work from the **pre-opt snapshot** content (Step 2). Do not copy current HEAD unless you
intentionally want post-opt code.

For workspace `$BATCH/<target_name>/`:

1. Create **one** operator file, usually named after the target stem (for example
   `chunk_o.py`), even when the repo `source_path` basename differs.
2. Include, in dependency order:
   - imports required by this target (keep Ascend/Triton imports faithful to the snapshot)
   - constants, `@triton.jit` kernels, and host helpers in the **minimal closure** for this
     target's launch entrypoint (often **more than one** `@triton.jit` kernel)
   - the public launch entrypoint that tests will call
3. Prefer **minimal extraction** over copying the whole original file:
   - include every kernel the entrypoint launches; omit kernels/wrappers for **other**
     validation targets
   - include shared helpers only when this call chain needs them
4. Keep semantics identical to the pre-opt snapshot for the included code — do not
   "pre-optimize" or refactor while splitting.

**Do not:**

- drop the full multi-kernel source file into the workspace when synthesis validates only one
  entrypoint or pipeline inside it
- split cooperating kernels that one test entrypoint always launches together into separate
  workspaces
- put multiple **unrelated** validation targets into one workspace
- run opaque auto-split scripts and accept the output without reading it

### Resolving repo-local imports

Optimize workspaces are **flat** directories. Repo package layout (`src.kernels.foo.bar`)
will not work unchanged. Resolve imports deliberately, in this order:

| Strategy | Use when |
|----------|----------|
| **Copy sibling modules under `deps/`** | A small repo `.py` file is imported by the target (for example `utils.py`, `common.py`). Copy to `$BATCH/<workspace>/deps/` and import as `from deps.utils import ...` or add `deps/` to `sys.path` in the operator/tests. **Never** place helper `.py` files at the workspace root — `optimize-batch` fails with `multiple candidate operator files`. |
| **Copy minimal subset inline** | A helper is a few lines and copying the whole module would pull unrelated kernels. |
| **Adjust test/bench imports** | The harness imported the repo package path; change it to import the workspace operator module or entry function directly. |
| **Add a thin adapter** | Tests expect a specific function name; provide a small wrapper in the operator file that calls the extracted launch path. |

Rules:

- Copy **only** modules in the dependency closure for this target. Do not copy entire package
  trees "just in case".
- After copying, run a quick import sanity check from the workspace directory before
  `optimize-batch` (for example `python3 -c "import <operator_stem>"` or run the copied test
  once).
- List every copied repo file in `validation-meta.json` → `copied_dependencies` using paths
  under `deps/` (for example `deps/tiling_utils.py`). Set `dependency_dir` to `deps` when using
  the default layout.
- Explain non-obvious import fixes in `notes`.

If optimize later fails on a missing import, add the smallest additional file or inline the
missing helper, then update `validation-meta.json`.

### Tests and benches for split targets

- Prefer a test/bench that exercises **only** this workspace's launch entrypoint (which may
  still run several cooperating kernels internally).
- If the repo test imports the whole module and runs multiple **unrelated** entrypoints,
  **copy and trim** it, or author a focused test that calls only this workspace's entrypoint.
- Do not attach a whole-file integration test to a single-target workspace unless that test
  truly depends on only the extracted closure.
- When one repo test file covers multiple unrelated entrypoints, split it the same way you
  split the operator — one workspace gets one tailored test file.

### `validation-meta.json` for split workspaces

In addition to the usual fields, set:

| Field | Required | Meaning |
|-------|----------|---------|
| `validation_target` | yes | Launch entrypoint this workspace validates (for example `forward_chunk_scan`) |
| `source_path` | yes | Primary repo file path (same for several workspaces is OK) |
| `split_from` | recommended | Primary `source_path` when manually extracted from a larger file |
| `closure_source_paths` | recommended when cross-file | All repo files whose pre-opt code was pulled into this workspace |
| `included_symbols` | recommended | Entrypoint, **all** cooperating `@triton.jit` kernels, and key helpers in the operator file |
| `dependency_dir` | recommended | Subdirectory for helper modules (default: `deps`) |
| `copied_dependencies` | recommended | Helper `.py` paths under `dependency_dir/` (for example `deps/utils.py`) |
| `excluded_targets` | optional | Other entrypoints/kernels/pipelines left out (audit trail) |

Example — **independent kernel** in a shared file:

```json
{
  "workspace": "chunk_o",
  "source_path": "src/kernels/fla/ops/common/fused_ops.py",
  "operator_filename": "chunk_o.py",
  "validation_target": "forward_chunk_o",
  "split_from": "src/kernels/fla/ops/common/fused_ops.py",
  "base_revision": "origin/main",
  "head_revision": "HEAD",
  "expected_patterns": ["grid-flatten-and-ub-buffering"],
  "synthesis_refs": ["chunk_o grid flatten in fused_ops.py"],
  "included_symbols": ["forward_chunk_o", "chunk_o_kernel", "CHUNK_BLOCK"],
  "dependency_dir": "deps",
  "copied_dependencies": ["deps/tiling_utils.py"],
  "excluded_targets": ["forward_wy_fast", "wy_fast_kernel", "scaled_dot_kkt_kernel"],
  "copied_tests": ["test_chunk_o.py.txt"],
  "copied_benches": [],
  "notes": "Manual extract; wy_fast validated in separate workspace"
}
```

Example — **one entrypoint, multiple cooperating kernels**:

```json
{
  "workspace": "gated_delta_scan",
  "source_path": "src/kernels/fla/ops/gated_delta_rule/fused_scan.py",
  "operator_filename": "gated_delta_scan.py",
  "validation_target": "forward_gated_delta",
  "split_from": "src/kernels/fla/ops/gated_delta_rule/fused_scan.py",
  "closure_source_paths": [
    "src/kernels/fla/ops/gated_delta_rule/fused_scan.py",
    "src/kernels/fla/ops/gated_delta_rule/wy_fast.py"
  ],
  "base_revision": "origin/main",
  "head_revision": "HEAD",
  "expected_patterns": [
    "layout-materialization-elision",
    "grid-flatten-and-ub-buffering"
  ],
  "synthesis_refs": ["gated delta scan pipeline: prep + wy_fast + epilogue"],
  "included_symbols": [
    "forward_gated_delta",
    "prep_indices_kernel",
    "wy_fast_kernel",
    "epilogue_kernel"
  ],
  "copied_dependencies": [],
  "excluded_targets": ["forward_standalone_wy_fast"],
  "copied_tests": ["test_gated_delta_scan.py.txt"],
  "copied_benches": [],
  "notes": "Single entrypoint launches 3 kernels; wy_fast.py prep-opt inlined from sibling file"
}
```

### Pre-flight checklist (split workspaces)

Before moving to Step 3 / optimize-batch, verify:

- [ ] Workspace contains exactly **one** operator `.py` at the root
- [ ] Operator file contains the intended `validation_target` entrypoint and **all** kernels
  that entrypoint launches — not the whole multi-target source file
- [ ] Unrelated entrypoints/kernels from the same `source_path` are absent
- [ ] Copied tests call only this workspace's launch entrypoint (which may run multiple inner
  kernels)
- [ ] Import sanity check passes from the workspace directory
- [ ] `validation-meta.json` documents `included_symbols`, `copied_dependencies`, and
  `closure_source_paths` when applicable

## Step 3 — Copy harness and dependencies

Each workspace must satisfy `optimize-batch` layout rules:

- Exactly **one** operator `.py` at workspace root (excluding `test_*`, `bench_*`, `conftest.py`, etc.)
- At least one reference test file (`test_*.py.txt`) when the repo has matching tests
- **No other** `.py` files at workspace root — put copied helpers under `deps/` (or another
  single subdirectory recorded in `dependency_dir`)

`conftest.py` is ignored by `optimize-batch` operator discovery and may be used for pytest path
setup. Helper modules belong in `deps/`, not beside the operator file.

### Tests (reference only)

Search the repo and choose files that actually exercise the operator:

- `test_<stem>.py`, `test_*<stem>*.py`
- `differential_test_*`
- Imports or pytest markers referencing the module path

Copy chosen test files into the workspace as **`test_<name>.py.txt`** (append `.txt` to the
original `.py` name). These are **reference artifacts for the optimize agent** — they document
dtype, tensor shapes, and how the operator is called. They are **not** runnable pytest files.

Record the destination names in `validation-meta.json` → `copied_tests` (for example
`test_chunk_gated_delta_rule_fwd_h.py.txt`).

The pattern-validation loop injects optimize-batch instructions explaining how to use these
reference files when generating real `test_*.py` for optimize rounds.

Prefer minimal copies; if reference tests import sibling modules, copy those helpers under
`deps/` or adjust imports deliberately.

### Benchmarks (optional)

Copy `bench_*` files when present and relevant.

### Import failures

If optimize later fails on missing local imports, copy required sibling modules into the
workspace and note that in `validation-meta.json` → `notes`.

## Step 4 — Write `validation-meta.json`

Create in each workspace:

```json
{
  "workspace": "chunk_o",
  "source_path": "src/kernels/.../chunk_o.py",
  "operator_filename": "chunk_o.py",
  "base_revision": "origin/main",
  "head_revision": "HEAD",
  "expected_patterns": ["layout-materialization-elision", "grid-flatten-and-ub-buffering"],
  "synthesis_refs": ["host-side transpose before HSTU", "grid flatten for chunk_o"],
  "copied_tests": ["test_chunk_o.py.txt"],
  "copied_benches": [],
  "notes": "Why this workspace exists and what success looks like"
}
```

Field guidance:

| Field | Required | Meaning |
|-------|----------|---------|
| `expected_patterns` | yes | Pattern IDs from staged index that optimize agent should apply |
| `validation_target` | yes when split from a multi-kernel file | Launch entrypoint this workspace validates |
| `split_from` | recommended when manually extracted | Primary repo `source_path` before split |
| `closure_source_paths` | recommended when cross-file | All repo files contributing to the extracted call chain |
| `included_symbols` | recommended when manually extracted | Entrypoint, cooperating kernels, and key helpers |
| `copied_dependencies` | recommended | Repo `.py` files copied into the workspace for imports |
| `synthesis_refs` | recommended | Human-readable synthesis anchors (no fixed ID scheme) |
| `copied_tests` / `copied_benches` | recommended | Audit trail of what you copied |

`audit_batch.py` matches `expected_patterns` against `opt-round-*/attempts.md` and
`summary.md`. Choose IDs that match real card filenames under `references/patterns/`.

## Step 5 — Optional batch manifest

If helpful, write `$BATCH/manifest.json` summarizing workspaces for humans and reruns:

```json
{
  "repo": "/abs/path/to/repo",
  "base_revision": "origin/main",
  "synthesis_report": "PERF_PATTERN_SYNTHESIS.md",
  "operators": [ { "...": "mirror validation-meta fields" } ]
}
```

Example fixture (hand-authored): `docs/fixtures/q2tritonkernel-pattern-validation.manifest.json`

## Step 6 — Verify before optimize-batch

Checklist per workspace:

- [ ] Operator file is pre-optimization snapshot, not current HEAD (unless intentional)
- [ ] For multi-kernel source files: operator is a **manual extract** for one launch
  `validation_target` and its full kernel call chain — not the whole source file (see Step 2b)
- [ ] Test file(s) present and reference the operator
- [ ] `validation-meta.json` lists realistic `expected_patterns`
- [ ] No `local-only` synthesis goals listed as required patterns
- [ ] Workspace directory name is stable across iterations

Record scaffold completion:

```bash
triton-agent pattern-validation-verify -i "$BATCH"
python3 "$SKILL/scripts/record_iteration.py" \
  --state "$STATE" --phase scaffold \
  --note "N workspaces: ..."
```

Do not run optimize-batch until `triton-agent pattern-validation-verify -i "$BATCH"` exits 0.

## When to rebuild vs reuse workspaces

| Situation | Action |
|-----------|--------|
| First loop iteration | Build workspaces under `$BATCH` |
| Skill iteration only | Keep active workspaces; `reset_workspace_rounds.py` skips `_completed/` |
| Workspace passed audit | Archive to `$BATCH/_completed/<name>/`; exclude from next optimize-batch |
| New operator added to validation scope | Add new active workspace under `$BATCH` |
| Wrong pre-opt snapshot or wrong tests | Fix files in place or recreate that workspace only |
