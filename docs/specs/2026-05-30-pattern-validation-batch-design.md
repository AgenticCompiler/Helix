# Pattern Validation Batch Design

## Summary

Close the loop between `analyze-commit-perf` and `optimize-batch`: after promoting
commit-derived lessons into `triton-npu-optimize-knowledge`, build a dedicated batch
root of pre-optimization operator workspaces, run fixed-round optimize sessions, and
iterate on pattern cards until agents rediscover the human optimizations.

**Orchestration model (2026-06 update):** `triton-agent pattern-validation-loop` is
**CLI-orchestrated**: prepare and analyze agents handle synthesis/skills/scaffold and
evidence review; the CLI runs `pattern-validation-verify`, `optimize-batch`, evidence
collection, and `reset_workspace_rounds.py`. Helper scripts materialize agent-authored
manifests when useful; they do not parse a fixed synthesis schema such as `G1-I1` tables
as the source of truth.

## Problem

Commit analysis produces `PERF_KNOWLEDGE_BASE.md` and `PERF_PATTERN_SYNTHESIS.md`, but
those documents do not prove the updated skills help live optimize agents. Users need a
repeatable validation harness with:

- one workspace per kernel under test
- the **pre-optimization** operator snapshot as optimize input
- current repo tests for shape and reference semantics
- explicit expected pattern IDs from synthesis
- a reset/re-run loop when skills still fail to apply

## Goals

- Document a four-phase loop: integrate skills → scaffold batch → run optimize → adjust
  skills and reset until success criteria are met.
- Provide optional scaffold helpers when an agent (or human) has already authored
  `manifest.json`; document Git snapshot and test-selection rules for agents in
  `workspace-scaffold-contract.md`.
- Keep optimize artifact semantics unchanged (`baseline/`, `opt-round-N/`, `opt-note.md`).
- Store per-workspace expected pattern IDs for post-run auditing.

## Non-Goals

- Do not add a new CLI subcommand in the first version; reuse `optimize-batch`.
- Do not auto-edit pattern cards from failed optimize runs.
- Do not require remote NPU access inside the scaffold helper.
- Do not replace manual review of `attempts.md` / `summary.md`.

## Workspace Layout

Batch root (example: `pattern-validation-batch/`):

```text
pattern-validation-batch/
  manifest.json                 # batch metadata + operator list
  chunk_o/
    chunk_o.py                  # sole operator candidate (pre-opt snapshot)
    test_chunk_o.py             # copied from target repo (required)
    bench_chunk_o.py            # optional if present in repo
    validation-meta.json        # expected_patterns, source commits, notes
  wy_fast/
    ...
  _completed/                   # reserved; skipped by optimize-batch discovery
    chunk_delta_h/              # archived after audit pass
      validation-meta.json      # validation_status=completed
      ...
```

Rules aligned with `optimize-batch`:

- Exactly one operator `.py` per **active** child directory after excluding reserved
  `_completed/`, `test_*`, `bench_*`, etc.
- Child directory name becomes the workspace label in batch logs.
- After audit pass, move workspace to `_completed/<name>/` so later batch runs skip it.
- Do not place optimize artifacts in the batch root itself unless running a single
  workspace with `--input <workspace-dir>`.

## Pre-Optimization Snapshot Selection

For each `source_path` in the manifest:

1. List commits in `<base_revision>..HEAD` touching the path (`git log --reverse`).
2. If non-empty, extract `git show <first_commit>^:<source_path>` as the pre-opt
   snapshot (last version before the first in-range change).
3. If the first in-range commit creates the file, fall back to the blob at
   `<base_revision>:<source_path>` when available.
4. Write the snapshot to `<workspace>/<operator_filename>`.

This matches the user intent: optimize starts from the code **before** branch perf work,
not from current HEAD.

## Test And Bench Files

- Copy explicit `test_paths` from the manifest when provided.
- Otherwise search the repo for `test_*.py` / `differential_test_*.py` whose text
  references the operator stem or source path basename.
- Copy matching `bench_*.py` when found; optimize may still regenerate harnesses, but
  existing bench files reduce setup friction.

## Validation Loop

### Phase 1 — Integrate skills

- Apply `PERF_PATTERN_SYNTHESIS.md` recommendations to
  `skills/triton-npu-optimize-knowledge/references/patterns/`.
- Regenerate `pattern_index.md`.
- Record promoted pattern IDs and workspace mapping in the batch manifest.

### Phase 2 — Scaffold batch root

```bash
python3 scripts/scaffold_pattern_validation_batch.py \
  --manifest docs/fixtures/q2tritonkernel-pattern-validation.manifest.json \
  --output /path/to/pattern-validation-batch
```

### Phase 3 — Run optimize batch

```bash
cd /path/to/pattern-validation-batch/../  # or repo containing batch root
triton-agent optimize-batch \
  -i pattern-validation-batch \
  --resume fresh \
  --reset-optimize \
  --min-rounds 10 \
  --optimize-knowledge v1 \
  --max-concurrency 1 \
  --show-output
```

Use `--optimize-knowledge v1` unless validating a staged v2/v3 fork.

### Phase 4 — Audit and iterate

For each workspace:

1. Run `python3 scripts/audit_pattern_validation_batch.py --workspace <dir>`.
2. Read `opt-round-*/attempts.md` and `summary.md` for cited pattern IDs.
3. Compare against `validation-meta.json` → `expected_patterns`.

If patterns were not applied or benchmarks did not improve:

1. Extend or fix pattern cards.
2. Regenerate `pattern_index.md`.
3. Reset optimize artifacts **keeping baseline** is not supported by `--reset-optimize`
   (it removes baseline too). For skill-only reruns after an initial baseline pass,
   delete manually:

   ```bash
   rm -rf opt-round-* opt-note.md learned_lessons.md
   # keep baseline/, test_*, bench_*, operator py
   ```

   Or rerun from scratch with `--resume fresh --reset-optimize` when baseline must be
   re-established.

4. Re-run `optimize-batch` with `--min-rounds 10`.

Repeat until every workspace reports matched expected patterns in audit output and
round summaries show plausible application of the commit-derived optimizations.

## Success Criteria

Per workspace:

- `audit_pattern_validation_batch.py` reports all `expected_patterns` found in at least
  one round's `attempts.md` or `summary.md`.
- At least one round records a pattern-backed hypothesis aligned with synthesis (not
  only generic tiling guesses).
- Optional: benchmark improvement vs `baseline/` when NPU bench is available.

## Future Work

- CLI subcommand `scaffold-pattern-validation-batch` wrapping the helper script.
- Automatic manifest generation from `PERF_PATTERN_SYNTHESIS.md` group items.
- Structured JSON audit report aggregating all workspaces.
