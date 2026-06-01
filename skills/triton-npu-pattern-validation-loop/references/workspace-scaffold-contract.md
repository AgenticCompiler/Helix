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
| `PERF_PATTERN_SYNTHESIS.md` | Primary: which kernels/mechanisms to validate in skills |
| `PERF_KNOWLEDGE_BASE.md` | File paths, commits, diffs when synthesis is high-level |
| Staged `pattern_index.md` | Pattern IDs to cite in `expected_patterns` |
| Git history `base_revision..HEAD` | Confirm which files changed and when |

Also use loop state (`base_revision`, `batch_dir`) from init.

## Step 1 — Select validation targets

Decide **one optimize workspace per operator candidate**. Prefer:

- Kernels named explicitly in synthesis as perf-relevant or skill-promotion targets
- Source files with clear pre-optimization vs post-optimization story in the analyzed range
- Entries where updated skills should change optimize-agent behavior

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

`<workspace>` is usually the operator stem (e.g. `chunk_o`). `<operator_filename>` is
typically the basename of `source_path`.

You may call `scaffold_batch.py --manifest ...` **only after** you authored manifest JSON
yourself; never treat `generate_manifest.py` output as authoritative without review.

## Step 3 — Copy harness and dependencies

Each workspace must satisfy `optimize-batch` layout rules:

- Exactly **one** operator `.py` at workspace root (excluding `test_*`, `bench_*`, etc.)
- At least one runnable test file

### Tests

Search the repo and choose files that actually exercise the operator:

- `test_<stem>.py`, `test_*<stem>*.py`
- `differential_test_*`
- Imports or pytest markers referencing the module path

Copy chosen files into the workspace. Prefer minimal copies; if tests import sibling
modules, copy those `.py` files too or adjust imports deliberately.

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
  "copied_tests": ["test_chunk_o.py"],
  "copied_benches": [],
  "notes": "Why this workspace exists and what success looks like"
}
```

Field guidance:

| Field | Required | Meaning |
|-------|----------|---------|
| `expected_patterns` | yes | Pattern IDs from staged index that optimize agent should apply |
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
- [ ] Test file(s) present and reference the operator
- [ ] `validation-meta.json` lists realistic `expected_patterns`
- [ ] No `local-only` synthesis goals listed as required patterns
- [ ] Workspace directory name is stable across iterations

Record scaffold completion:

```bash
python3 "$SKILL/scripts/record_iteration.py" \
  --state "$STATE" --phase scaffold \
  --note "N workspaces: ..."
```

## When to rebuild vs reuse workspaces

| Situation | Action |
|-----------|--------|
| First loop iteration | Build workspaces under `$BATCH` |
| Skill iteration only | Keep active workspaces; `reset_workspace_rounds.py` skips `_completed/` |
| Workspace passed audit | Archive to `$BATCH/_completed/<name>/`; exclude from next optimize-batch |
| New operator added to validation scope | Add new active workspace under `$BATCH` |
| Wrong pre-opt snapshot or wrong tests | Fix files in place or recreate that workspace only |
