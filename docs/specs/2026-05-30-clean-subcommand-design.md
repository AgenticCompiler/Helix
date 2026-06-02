# Clean Subcommand Design

## Summary

Add a `clean` subcommand that removes known generated artifacts from one operator workspace or from every immediate child workspace under a batch root.

By default the command preserves the original operator source file plus reusable generated test and benchmark harnesses. When `--deep` is set, it also removes the generated test and benchmark harnesses.

## Goals

- Add a first-class CLI subcommand for resetting a workspace without manually deleting files.
- Support both one workspace input and batch-root input with the same discovery semantics as `status`.
- Delete only repository-known generated artifacts by default.
- Preserve the original operator source file by default.
- Preserve reusable test and benchmark harnesses by default.
- Remove generated operator files such as `opt_<op>.py` and `triton_<op>.py` by default.
- Remove profiling leftovers such as `PROF_*` directories and `extra-info.json`.
- Provide a predictable summary of what was removed, skipped, or not found.

## Non-Goals

- Do not delete arbitrary user-authored files that are not part of the repository's known artifact set.
- Do not recurse through nested batch roots beyond the immediate child workspace level.
- Do not launch agents.
- Do not add a dry-run mode in the first iteration.
- Do not infer or preserve custom harness names outside the existing workspace conventions and baseline metadata.

## User-Visible Behavior

Run:

```bash
uv run triton-agent clean --input .
uv run triton-agent clean --input operators_root
uv run triton-agent clean --input . --deep
uv run triton-agent clean --input operators_root --deep
```

`--input` accepts:

- one operator workspace directory
- one batch root whose immediate child directories are operator workspaces

`--deep` extends cleanup to generated case files:

- `test_<op>.py`
- `differential_test_<op>.py`
- `bench_<op>.py`

Default cleanup removes known generated artifacts while keeping:

- the original operator source file resolved by the existing optimize workspace rules
- reusable generated test and benchmark harnesses
- unrelated user files such as notes, scripts, or auxiliary data that are not on the known-artifact list

The command exits successfully when targeted workspaces were scanned even if some known artifact paths were absent. Missing artifact paths are treated as no-op cleanup, not errors.

## Discovery

Use `status`-style input discovery:

- If `--input` itself looks like a single operator workspace, clean only that workspace.
- Otherwise, treat `--input` as a batch root and scan its immediate child directories.
- If a batch root has no child directories, fail with the same shape as `status`: `No operator workspaces found under <root>`.

For single-workspace resolution, reuse the existing optimize operator discovery rules so the command can distinguish the original operator file from generated operator files:

- exclude `test_`
- exclude `differential_test_`
- exclude `bench_`
- exclude `opt_`
- exclude `__init__.py`

`triton_<op>.py` is not a valid original operator candidate under the intended workspace model and is always treated as a generated artifact.

## Known Artifact Set

For each workspace, default cleanup removes these known generated artifacts when present:

- `triton_<op>.py`
- `opt_<op>.py`
- `*_result.pt`
- `*_perf.txt`
- `baseline/`
- `opt-round-*`
- `opt-verify/`
- `.triton-agent/`
- `triton-agent-logs/`
- `opt-note.md`
- `learned_lessons.md`
- `report.md`
- `log_check_result.json`
- `log_check_result.md`
- `pattern_analysis.json`
- `pattern_analysis.md`
- `extra-info.json`
- `PROF_*`

When `--deep` is set, also remove:

- `test_<op>.py`
- `differential_test_<op>.py`
- `bench_<op>.py`

For batch-root input, also clean batch-root-level generated artifacts:

- `optimize-batch-status.json`
- `log_check_summary.md`
- `report-batch-state.json`
- `report-batch.md`

## Safety Rules

- Delete only paths on the known-artifact list.
- Never delete the resolved original operator source file.
- Never delete unrelated user-authored files that do not match the known-artifact list.
- Support both files and directories for known artifact paths.
- If a known artifact path is a symlink, unlink only the symlink itself.
- Missing paths are ignored.

## Output And Exit Codes

For one workspace input, print a concise summary that includes:

- workspace path
- removed path count
- skipped or absent path count

For batch-root input, print one line per workspace plus one batch summary line.

Return codes:

- `0`: cleanup completed for the requested scope
- `1`: no operator workspaces found under a batch root
- `2`: argument or input validation failure reported through `argparse`

## Implementation Notes

- Add `CommandKind.CLEAN`.
- Register `clean` in the top-level CLI parser with `--input`, `--verbose`, and `--deep`.
- Implement the command handler in a new `src/triton_agent/commands/clean.py`.
- Add a focused cleanup module to hold artifact discovery and deletion logic so the command handler stays thin.
- Reuse existing optimize workspace candidate resolution rather than duplicating operator-file rules.
- Reuse `status`-style workspace-or-batch-root detection semantics, but base single-workspace detection on cleanable workspace signals instead of optimize-only signals so a directory that only contains `triton_<op>.py` or `PROF_*` can still be cleaned directly.

## Testing

- Parser coverage for `clean` and `--deep`.
- Single-workspace handler coverage proving default cleanup preserves the original operator file and case files.
- Single-workspace handler coverage proving `--deep` also removes harness files.
- Coverage proving generated operator files `opt_<op>.py` and `triton_<op>.py` are deleted.
- Coverage proving `PROF_*` directories and `extra-info.json` are deleted.
- Batch-root coverage proving discovery matches `status` semantics.
- Coverage proving batch-root cleanup removes `optimize-batch-status.json`, `log_check_summary.md`, `report-batch-state.json`, and `report-batch.md`.
- Coverage proving unrelated files are preserved.
- Coverage proving symlink artifact paths are unlinked rather than recursively traversed.
