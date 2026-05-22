# Run-Bench JSONL Perf Artifact Implementation Plan

## Summary

Replace current ad hoc line-plus-comment perf artifact format with JSONL records. Spec: `docs/specs/2026-05-21-run-bench-jsonl-perf-artifact-design.md`.

## Files Changed

### Core library: `skills/triton-npu-run-eval/scripts/perf_artifacts.py`

1. **Rename field**: `PerfCaseRecord.elapsed_seconds` → `case_wall_clock_seconds`
2. **Add JSONL render**: `render_perf_case_records_jsonl()` → list of JSON strings
3. **Add JSONL render single**: `render_perf_case_record_jsonl(record)` → one JSON string
4. **Add JSONL parse**: `_parse_perf_entries_from_jsonl()` → PerfParseOutcome from JSONL
5. **Auto-detect format**: Update `_parse_perf_entries_impl()` and `_parse_required_perf_entries_impl()` to check first non-empty line for `{` prefix → JSONL, else legacy text

### Producer: `skills/triton-npu-run-eval/scripts/bench_runner_msprof.py`

1. Rename all `elapsed_seconds=` → `case_wall_clock_seconds=` in PerfCaseRecord constructions (12 sites)
2. `_write_msprof_perf()` → use `render_perf_case_records_jsonl()` instead of `render_perf_case_records()`

### Producer: `skills/triton-npu-run-eval/scripts/bench_runner_standalone.py`

1. Rename all `elapsed_seconds=` → `case_wall_clock_seconds=` in PerfCaseRecord constructions (4 sites)
2. `_build_standalone_run_one_case_script()` → rename `'elapsed_seconds'` → `'case_wall_clock_seconds'` in JSON payload
3. `_parse_standalone_case_result_payload()` → rename `parsed["elapsed_seconds"]` → `parsed["case_wall_clock_seconds"]`
4. `_write_standalone_perf()` → use `render_perf_case_records_jsonl()` instead of `render_perf_case_records()`

### Producer: `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`

1. Rename `elapsed_seconds=` → `case_wall_clock_seconds=` in `_run_standalone_case()` (2 sites)
2. `run_local_standalone_bench()` → use `render_perf_case_records_jsonl()` instead of `render_perf_case_records()`

### Test files updated

- `tests/test_bench_runner.py` (elapsed_seconds references, perf content assertions)
- `tests/test_standalone_bench_runtime.py` (elapsed_seconds references, perf content assertions)
- `tests/test_comparison_commands.py` (JSONL fixture data)
- `tests/test_remote_execution.py` (elapsed_seconds in expected subprocess output)
- `tests/test_verify.py` (perf artifact content assertions)
- `tests/test_skill_command_script.py` (perf artifact fixtures)
- Add JSONL-specific parser tests

## Implementation Order

1. Rename field → 2. Add JSONL functions → 3. Update producers → 4. Update parsers → 5. Update tests → 6. Verify
