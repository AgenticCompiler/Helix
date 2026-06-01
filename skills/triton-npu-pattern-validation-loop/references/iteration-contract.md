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
| `skills_dir` | Persistent loop skills workdir (default: `pattern-validation-skills`) |
| `synthesis_path` | Path to `PERF_PATTERN_SYNTHESIS.md` |
| `min_rounds` | `--min-rounds` for each optimize-batch run |
| `history` | List of `{phase, iteration, note, audit}` events |

## Batch root layout

Agent creates workspaces under the batch root. Optional `manifest.json` is **agent-authored**
metadata, not output of a required parser.

```text
pattern-validation-batch/
  ...
  _completed/
    ...

pattern-validation-skills/          # persistent; never deleted by loop CLI
  triton-npu-optimize-knowledge/
    references/patterns/
    references/pattern_index.md
```

Optimize staging: `optimize-batch --skills-source-dir pattern-validation-skills` copies from
`pattern-validation-skills/<skill-name>/` into each workspace backend skills dir before optimize.

Active workspace example:

```text
pattern-validation-batch/chunk_o/
  chunk_o.py
  test_*.py
  validation-meta.json
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

Every `optimize-batch` invocation must include **`--show-output`**. This streams nested optimize agent output to the terminal with a `[workspace]` prefix. Long optimize runs that produce no stdout are more likely to be killed by CI/job timeouts or idle watchdogs.

Every `optimize-batch` shell command must prefix **`TRITON_AGENT_STALL_TIMEOUT_SECONDS=0`** so nested optimize agents are not killed by triton-agent idle stall detection.

Optional optimize passthrough flags from the loop start command (`pattern-validation-loop` CLI): when set at launch, include the same flags on every optimize-batch run; when unset at launch, omit them and let optimize-batch use its own defaults.

Initial run:

```bash
TRITON_AGENT_STALL_TIMEOUT_SECONDS=0 triton-agent optimize-batch \
  -i "$BATCH" \
  --resume fresh \
  --reset-optimize \
  --min-rounds "$MIN_ROUNDS" \
  --concurrency 1 \
  --show-output \
  --skills-source-dir "$SKILLS" \
  --agent <backend>
  # optional when set on pattern-validation-loop start:
  # --target-chip A5 --test-mode differential --bench-mode standalone
```

Later iteration:

```bash
TRITON_AGENT_STALL_TIMEOUT_SECONDS=0 triton-agent optimize-batch \
  -i "$BATCH" \
  --resume continue \
  --min-rounds "$MIN_ROUNDS" \
  --concurrency 1 \
  --show-output \
  --skills-source-dir "$SKILLS" \
  --agent <backend>
  # same optional passthrough flags as initial run when provided at loop start
```

Audit and archive:

```bash
python3 "$SKILL/scripts/audit_batch.py" \
  --batch-root "$BATCH" --archive-passed --json > "$BATCH/audit-report.json"
```
