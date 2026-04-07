# Optimize Status Subcommand

## Summary

- Add a new read-only `optimize-status` subcommand for scanning batch optimize workspaces and summarizing current optimization progress.
- Reuse the `optimize-batch` root-directory scan model: each immediate child directory under `--input` is one operator workspace candidate.
- Report optimization status primarily through benchmark numbers instead of round narrative text.
- Use mean improvement across matched per-case latency deltas as the primary ranking metric for the best optimization round.

## User-Visible Behavior

- Add:
  - `uv run triton-agent optimize-status --input <root-dir>`
- The command scans immediate child directories only.
- The command does not launch a code agent, stage skills, or execute remote commands.
- The command keeps scanning even when some workspaces have missing or malformed optimize artifacts.
- The default output is a compact per-workspace numeric summary plus final totals.

Example shape:

```text
[OK] matmul
  Baseline mean: 1.82
  Best mean: 1.49
  Avg improvement: +14.6%
  Best round: round-3
  Logged best: round-3

[WARN] layernorm
  Baseline mean: 2.31
  Best mean: unknown
  Avg improvement: unknown
  Best round: unknown
  Warning: missing comparable round perf data

Summary: 1 ok, 1 warning, 0 no-session
```

## CLI Contract

- `optimize-status` accepts `--input/-i` as the batch root directory.
- `optimize-status` accepts `--verbose` for parsing diagnostics and artifact-source hints.
- Do not add agent-selection, remote, output-generation, or interactive flags in this change.
- JSON output is out of scope for the first version.

## Workspace Discovery

- Reuse the batch root shape from `optimize-batch`:
  - resolve `--input`
  - require it to exist and be a directory
  - scan immediate child directories only
- Unlike `optimize-batch`, do not require discovery of one unambiguous operator source file.
- Treat a child directory as an optimization workspace when any optimization artifact exists, such as:
  - `opt-note.md`
  - one or more `opt-round-*` directories
  - generated benchmark perf files associated with the original operator or a round operator
- If a child directory has no optimize artifacts, report it as `no-session` instead of failing the whole command.

## Numeric Model

### Baseline Perf

- The baseline perf source is the original operator benchmark result saved beside the original operator file using the existing `<operator-file-stem>_perf.txt` format.
- Parse perf files using the same `latency-<id>: <float>` contract already used by `compare-perf`.
- The baseline mean shown in output is the arithmetic mean of baseline latency values.

### Round Perf

- Each `opt-round-N/` directory may provide benchmark evidence through:
  - a copied or summarized `perf.txt`
  - or a round-local `<operator-file-stem>_perf.txt`
- Prefer round-local normalized perf data over free-form text in `summary.md`.
- Only use `summary.md` for warnings or provenance, not as the primary numeric source.

### Improvement Metric

- For each latency id shared by the baseline perf and one round perf, compute per-case relative improvement:

```text
improvement(id) = (baseline(id) - round(id)) / baseline(id)
```

- The round's primary optimization score is:

```text
mean(improvement(id) for all matched ids)
```

- Display that value as `Avg improvement: +X%`.
- Also display:
  - `Baseline mean`, computed from the baseline perf values
  - `Best mean`, computed from the selected round perf values
- This keeps the main ranking metric aligned with average improvement rate while still showing absolute latency numbers.

## Comparability Rules

- A round is comparable only when:
  - both baseline and round perf files parse successfully
  - both contain at least one latency entry
  - latency ids match exactly
- If latency ids differ, do not attempt partial comparison.
- If any baseline latency value is `<= 0`, skip that id from improvement-rate calculation and emit a warning.
- If all ids are skipped or no comparable round remains, report the workspace numeric status as `unknown`.

## Best Round Resolution

- Compute one numeric score per comparable round from the mean of per-case improvement rates.
- Select the round with the highest numeric score as `Best round`.
- Independently parse logged best status from `opt-note.md` when available by reading the latest round marked `Best status: current best`.
- Display both:
  - `Best round`: numeric best round from perf comparison
  - `Logged best`: workflow-reported best round from `opt-note.md`
- If those values differ, emit a warning instead of forcing one to override the other.

## Artifact Parsing Priority

- Numeric comparison should prefer concrete perf files first.
- Recommended source order:
  1. baseline `<operator>_perf.txt`
  2. round-local `perf.txt`
  3. round-local `*_perf.txt`
- `opt-note.md` is used for:
  - logged best round discovery
  - light status context
  - warnings about incomplete session structure
- `summary.md` and `attempts.md` are not part of the default output body for this command.

## Workspace Status Categories

- `ok`
  - at least one comparable round exists and numeric best data is available
- `warning`
  - optimization artifacts exist, but best-round comparison is incomplete or partially unavailable
- `no-session`
  - the child directory does not yet contain optimize artifacts

The command should keep final totals in those categories instead of treating artifact gaps as hard failures.

## Implementation Notes

- Keep this feature in the CLI orchestration layer; it is a local repository scan, not a skill behavior change.
- Reuse existing perf-file parsing logic instead of inventing a second parser.
- Add small helpers for:
  - discovering workspace status artifacts
  - locating baseline and round perf files
  - parsing `opt-note.md` best markers
  - computing aggregate improvement metrics
- Keep free-form summary parsing minimal and defensive.

## Documentation Updates

- Update `README.md` to document `optimize-status`.
- Update `AGENTS.md` to list the new subcommand and its numeric-summary purpose.
- Keep this design document as the source of truth for the command's comparison semantics.

## Verification

- Parser tests for `optimize-status --input`.
- CLI tests for:
  - invalid input path handling
  - empty batch root handling
  - `no-session` workspace reporting
  - comparable round selection and best-round ranking
  - mismatch between numeric best and logged best
  - id-mismatch warnings
  - missing perf artifact warnings
- Repo verification after implementation:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m unittest discover -s tests -v`
