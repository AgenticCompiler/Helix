# Refactor `msprof-analyze` Into A Generic Ascend NPU Operator Profiler Skill

## Goal

Replace the narrow `msprof-analyze` benchmark comparison skill with a more general Ascend NPU operator profiler skill centered on getting and analyzing operator performance data:

- run generated benchmark profiling through the unified `run-command.py profile-bench` helper, with direct `msprof <command>` kept as a fallback
- locate the generated `PROF_*` output directory
- read `op_statistic_*.csv` and `op_summary_*.csv`
- summarize the current operator's performance data in a concise report

Keep `parse_bin.py` as an optional helper for future binary analysis work, but remove the unused benchmark comparison script.

## User-Visible Semantics

- The skill should trigger for Ascend NPU operator profiling and performance-analysis requests, especially when the user wants operator-level timing data, hotspot identification, bottleneck diagnosis, or profiler-backed comparison.
- The default workflow should prefer running generated benchmark harnesses through the run-validation helper, for example:
  ```bash
  python3 ../triton-npu-run-eval/scripts/run-command.py profile-bench --bench-file bench_matmul.py --operator-file matmul.py
  ```
- `standalone` benchmark mode should profile the plain `--operator-file` benchmark invocation and should not use `--bench`.
- `msprof` benchmark mode should first query `--num-bench`, then profile one selected `--bench <N>` case, and should require benchmark `# kernel:` metadata.
- Remote-aware profiling should reuse the same `--remote` and `--remote-workdir` semantics as the existing triton-npu-run-eval helpers.
- After profiling, the skill should inspect the generated `PROF_*/mindstudio_profiler_output/` directory and summarize operator timing data from:
  - `op_statistic_<timestamp>.csv`
  - `op_summary_<timestamp>.csv`
- The skill should treat `op_summary` as potentially large and avoid eager whole-file loading when a streaming pass is enough.
- If the user does not identify the target operator explicitly, the skill may infer it from the hottest operator in `op_statistic`, but it should say that this is an inference.
- The old multi-version benchmark comparison flow is no longer part of this skill.
- The `optimize` skill should use this profiler skill when benchmark results need deeper operator-level explanation.

## Design

- Rename the skill directory and frontmatter name to `triton-npu-profile-operator` so the skill's identity matches its broader scope.
- Rewrite `SKILL.md` around one primary flow:
  1. run `python3 ../triton-npu-run-eval/scripts/run-command.py profile-bench ...`
  2. find the relevant `PROF_*` directory
  3. summarize the operator timing data
  4. present the result in the conversation
- Add a lightweight standard-library script that:
  - resolves the latest or explicitly provided `PROF_*` directory
  - finds the newest `op_statistic_*.csv` and `op_summary_*.csv`
  - reads `op_statistic` normally
  - streams `op_summary` row by row to aggregate matching operator timings
  - renders a concise Markdown report
- Keep only the minimum validation and failure-handling guidance inline in `SKILL.md` instead of maintaining a separate troubleshooting reference file.
- Keep `parse_bin.py` in place and document it as a secondary tool for raw profiler binary inspection.
- Delete `benchmark_analyzer.py` and remove references to its benchmark-comparison workflow from the skill and references.

## Verification

- Update the parser unit test to load `parse_bin.py` from the renamed skill directory.
- Add a unit test for the new profile summary script that verifies it can summarize a sample `PROF_*` directory and report the selected operator timing data.
- Run the targeted unit tests plus the standard repository verification commands after the refactor.
