# Pattern Validation Iteration Contract

## Orchestration model

`triton-agent pattern-validation-loop` is **CLI-orchestrated**:

| Step | Runner | Responsibility |
|------|--------|----------------|
| Seed skills | CLI | Copy install-bundle knowledge into `pattern-validation-skills/` when missing |
| Prepare | Code agent | Read synthesis, edit skills, scaffold workspaces, run `pattern-validation-verify` |
| Verify gate | CLI | `triton-agent pattern-validation-verify -i "$BATCH"` (must exit 0) |
| Optimize | CLI | `optimize-batch` with `--skills-source-dir` (direct Python API, streamed output) |
| Evidence | CLI | `audit_batch.py --output "$BATCH/audit-report.json"` |
| Analyze | Code agent | Review evidence, update skills, archive passes, mark complete or request another iteration |
| Reset | CLI | `reset_workspace_rounds.py` on active workspaces before the next optimize |

Prepare and analyze agents **must not** shell out to `triton-agent optimize-batch`.

## Loop state file

Path: `.triton-agent/pattern-validation-loop-state.json`

Schema version: `1`

Fields:

| Field | Meaning |
|-------|---------|
| `status` | `running` \| `complete` \| `failed` |
| `iteration` | Current iteration number (starts at 1) |
| `max_iterations` | Stop after this many optimize/analyze cycles without completion |
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
  audit-report.json              # CLI-written evidence bundle
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
- After analyze agent confirms pass, archive to `_completed/` with `audit_batch.py --archive-passed`.
- Do not manually delete `_completed/` unless intentionally re-validating that operator.

## Success criteria

Per workspace (`validation-meta.json` → `expected_patterns`):

1. Analyze agent confirms optimize rounds applied synthesis-backed mechanisms.
2. Evidence report lists round artifacts under `opt-round-*/` for review.
3. Optional: benchmark improvement vs `baseline/` when NPU available.

Heuristic `heuristic_suggested_pass` in `audit-report.json` (substring match on pattern IDs) is a
**hint only**.

Whole loop success: analyze agent sets loop state `status=complete` and `active_remaining` is
empty after archiving passed workspaces.

## Reset between iterations

The CLI runs `reset_workspace_rounds.py` on **active** workspaces only. It skips `_completed/`.

Removes:

- `opt-round-*`
- `opt-note.md`
- `learned_lessons.md`

Keep:

- operator `.py`
- `test_*`, `bench_*`
- `validation-meta.json`
- `baseline/`

## CLI optimize-batch settings

Environment:

```bash
export TRITON_AGENT_STALL_TIMEOUT_SECONDS=0
```

Initial iteration (CLI):

```bash
triton-agent optimize-batch -i "$BATCH" \
  --resume fresh --reset-optimize \
  --min-rounds "$MIN_ROUNDS" --concurrency 1 --show-output \
  --skills-source-dir "$SKILLS" --agent <backend>
```

Later iteration (CLI):

```bash
triton-agent optimize-batch -i "$BATCH" \
  --resume continue \
  --min-rounds "$MIN_ROUNDS" --concurrency 1 --show-output \
  --skills-source-dir "$SKILLS" --agent <backend>
```

Optional passthrough flags from loop start (`pattern-validation-loop` CLI): when set at launch,
the CLI includes the same flags on every optimize-batch run.

## Evidence and archive helpers

Collect evidence (CLI or manual):

```bash
python3 "$SKILL/scripts/audit_batch.py" \
  --batch-root "$BATCH" --output "$BATCH/audit-report.json"
```

Archive after **agent review** (not heuristics alone):

```bash
python3 "$SKILL/scripts/audit_batch.py" \
  --batch-root "$BATCH" --archive-passed
```

Scaffold verify (prepare agent and CLI gate):

```bash
triton-agent pattern-validation-verify -i "$BATCH"
```
