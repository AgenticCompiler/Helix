# Pattern Validation Iteration Contract

## Loop state file

Path: `.triton-agent/pattern-validation-loop-state.json`

Schema version: `1`

Fields:

| Field | Meaning |
|-------|---------|
| `status` | `running` \| `complete` \| `failed` |
| `iteration` | Current iteration number (starts at 1) |
| `max_iterations` | Stop skill loop after this many audit failures |
| `repo` | Absolute Git repo path |
| `base_revision` | Git base for scaffold |
| `batch_dir` | Batch root relative or absolute |
| `synthesis_path` | Path to `PERF_PATTERN_SYNTHESIS.md` |
| `min_rounds` | `--min-rounds` for each optimize-batch run |
| `history` | List of `{phase, iteration, note, audit}` events |

## Batch root layout

Agent creates workspaces under the batch root. Optional `manifest.json` is **agent-authored**
metadata, not output of a required parser.

```text
pattern-validation-batch/
  manifest.json              # optional: agent-written index of workspaces
  audit-report.json          # latest audit JSON
  VALIDATION_SUMMARY.md      # written when all targets archived
  chunk_o/                   # active — still scheduled by optimize-batch
    <operator>.py
    test_*.py
    validation-meta.json
    baseline/
    opt-round-N/
  wy_fast/                   # active
    ...
  _completed/                # reserved — NOT scheduled by optimize-batch
    chunk_delta_h/           # passed audit; kept for evidence
      validation-meta.json   # validation_status=completed, archived_at=...
      baseline/
      opt-round-N/
```

Rules:

- Only **active** workspaces (batch root children with `validation-meta.json`) run in the next
  `optimize-batch -i "$BATCH"`.
- After audit pass, move workspace to `_completed/` with `audit_batch.py --archive-passed`.
- Do not manually delete `_completed/` unless intentionally re-validating that operator.

## Success criteria

Per workspace (`validation-meta.json` → `expected_patterns`):

1. `audit_batch.py` reports `passed: true`
2. At least one `opt-round-*/summary.md` cites a **mechanism** aligned with synthesis (not only pattern ID string match)
3. Optional: benchmark improvement vs `baseline/` when NPU available

Whole loop success: `audit-report.json` → `active_remaining` is `[]` and all targets live under `_completed/`.

## Reset between iterations

Use `reset_workspace_rounds.py` on **active** workspaces only. It skips `_completed/`.

Removes:

- `opt-round-*`
- `opt-note.md`
- `learned_lessons.md`

Keep:

- operator `.py`
- `test_*`, `bench_*`
- `validation-meta.json`
- `baseline/`

## Subprocess commands

The orchestrating agent runs `triton-agent optimize-batch` as a **shell subprocess** from `REPO`. Do not simulate optimize rounds inside this skill without invoking the CLI.

Audit and archive:

```bash
python3 "$SKILL/scripts/audit_batch.py" \
  --batch-root "$BATCH" --archive-passed --json > "$BATCH/audit-report.json"
```
