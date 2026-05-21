# Compare Perf All Skill Parser Alignment

## Goal

- Keep the skill-side `skills/triton-npu-run-eval/scripts/run-command.py compare-perf` parser aligned with the repository CLI and `README.md` by accepting `--metric-source all`.

## User-Visible Behavior

- `python3 ./scripts/run-command.py compare-perf --metric-source all ...` must parse successfully.
- The skill script must continue to forward `metric_source="all"` unchanged to `perf_artifacts.compare_perf_files(...)`.
- No comparison logic changes are introduced by this alignment. The existing `all` behavior in `skills/triton-npu-run-eval/scripts/perf_artifacts.py` remains the source of truth.

## Error Handling

- Invalid `--metric-source` values other than `auto`, `kernel`, `total-op`, and `all` must still be rejected by argument parsing.

## Verification

- Add a skill-script parser regression test that exercises `compare-perf --metric-source all`.
- Run the targeted unittest coverage for the skill command script.
- Run the required strict skill-script pyright wrapper for `skills/triton-npu-run-eval/scripts/run-command.py`.
