# Perf-Counter Bench Mode Design

## Summary

Add a third bench mode `perf-counter` that measures case execution time using Python's `time.perf_counter()` instead of NPU profiling infrastructure (`msprof` or `torch_npu.profiler`). The tradeoff: significantly faster than NPU profiling but less accurate — no per-kernel or per-operator breakdown, just end-to-end case iteration timing.

As part of this change, bench mode stops being a property of the bench file and becomes purely a runtime concern: bench files no longer declare `# bench-mode`, the mode is chosen at execution time (defaulting to `torch-npu-profiler`), and the mode used is recorded in the perf JSONL output.

## Goals

- Run a benchmark case, time each measurement iteration with `time.perf_counter()`, and report the average iteration time.
- Produce valid `PerfCaseRecord` / JSONL output consumable by existing `compare-perf` and optimize pipelines (via `kernel_avg_time_us`).
- Record which bench mode produced each JSONL record, so consumers can detect cross-mode comparisons.
- Support all execution modes: local, local-parallel, remote, remote-parallel.
- Eliminate profiling overhead — no `torch_npu.profiler` API calls, no `msprof` wrapper process, no CSV parsing.
- Set `TRITON_ALWAYS_COMPILE=1` before running cases so the first iteration does not include JIT compilation time.
- Keep torch synchronization (`torch.npu.synchronize()`) after each iteration so timing reflects real device completion.

## Non-Goals

- No per-kernel or per-operator time breakdown. Without a profiler, individual kernel times cannot be attributed, so `ops` stays empty.
- `profile-bench` does not support `perf-counter`. It always uses `torch-npu-profiler` to capture profiler traces. No `--bench-mode` flag on `profile-bench`.

## Design Decisions

### Bench mode moves from bench file to runtime

Before this change, bench files declared their mode via a `# bench-mode: ...` header comment, and `run-bench` / `profile-bench` would read it via metadata parsing (`_resolve_bench_mode_from_metadata()`). This created several problems:

- Adding a new mode required updating allowlists in 5+ places (cli.py, execution.py, resume.py, run-command.py, capture_ir.py).
- `profile-bench` and `run-bench` shared the same metadata resolution, forcing `profile-bench` to handle modes it didn't support.
- Optimize resume had to re-parse bench mode from harness files.

After this change:

- Bench files no longer carry a `# bench-mode` header. Mode is purely a runtime decision.
- `run-bench --bench-mode` defaults to `torch-npu-profiler`. The only other valid value is `msprof` or `perf-counter`.
- `profile-bench` always uses `torch_npu.profiler`. It no longer accepts `--bench-mode`.
- The bench mode used is recorded in each JSONL record (`"bench_mode"` field).

### Bench mode recorded in JSONL output

Add a `bench_mode` field to `PerfCaseRecord` and include it in JSONL output. This enables `compare-perf` to reject cross-mode comparisons — the metrics have fundamentally different meanings across modes, so comparing them is not meaningful.

The field is `None` for records produced before this change (backward compatible).

## Mode Name

`perf-counter` — descriptive of the mechanism (`time.perf_counter()`).

- `torch-npu-profiler` — uses `torch_npu.profiler` API (default)
- `msprof` — uses the `msprof` CLI wrapper
- **`perf-counter`** — uses `time.perf_counter()` directly, no profiler

## Execution Flow

### Per-case timing loop

```
# Set before looping over cases (mirrors profile_all_bench_cases)
os.environ["TRITON_ALWAYS_COMPILE"] = "1"

for case in cases:
    t_total_start = time.monotonic()

    # warmup (not timed)
    for _ in range(case.warmup):
        case.fn()
        torch.npu.synchronize()

    # measurement (each iteration timed with perf_counter)
    iteration_times: list[float] = []
    for _ in range(case.repeats):
        t0 = time.perf_counter()
        case.fn()
        torch.npu.synchronize()
        iteration_times.append(time.perf_counter() - t0)

    t_total_elapsed = time.monotonic() - t_total_start

    avg_iteration_seconds = statistics.mean(iteration_times)
    avg_iteration_us = avg_iteration_seconds * 1_000_000

    # build PerfCaseRecord
    PerfCaseRecord(
        case_label=case.case_id,
        kernel_names=resolution.kernel_names,
        kernel_source=resolution.kernel_source,
        metrics={
            "kernel_avg_time_us": avg_iteration_us,
            "ops": [],
        },
        case_wall_clock_seconds=t_total_elapsed,
        bench_mode="perf-counter",
    )
```

### Why two different clocks?

- `case_wall_clock_seconds` uses `time.monotonic()` — consistent with the existing `_run_bench_case()` in `bench_runtime.py:363-370`. It measures total wall-clock duration of the whole case (warmup + repeats), and monotonic is immune to system clock adjustments over longer spans.
- `kernel_avg_time_us` uses `time.perf_counter()` — higher resolution for short per-iteration intervals. `perf_counter` is also monotonic but offers better precision for sub-second measurements.

### Comparison with existing modes

| Aspect | torch-npu-profiler | msprof | perf-counter |
|--------|-------------------|--------|--------------|
| Profiling tool | `torch_npu.profiler` API | `msprof` CLI wrapper | none |
| Per-iteration timing | From profiler CSV | From profiler CSV | `time.perf_counter()` |
| Per-operator breakdown | yes | yes | no |
| CSV parsing | operator/kernel/op_statistic | op_statistic | none |
| Overhead | profiler overhead | msprof + profiler overhead | near-zero |
| Output `ops` | real op rows | real op rows | `[]` (no per-kernel attribution) |
| TRITON_ALWAYS_COMPILE | yes (line 235) | N/A (subprocess) | yes |

Note on parallel mode: `_run_local_bench_torch_npu_profiler_parallel` uses `ThreadPoolExecutor` where all threads share the parent process environment. `TRITON_ALWAYS_COMPILE=1` set in the main process before dispatching worker threads is visible to all workers — no per-subprocess env setup is needed. The same applies to the perf-counter parallel path.

### Where it fits in the dispatch matrix

```
                         devices=None              devices=not None
                         ─────────────────────────────────────────────────
bench_mode="msprof"      _run_local_bench_msprof   _run_local_bench_msprof_parallel
bench_mode="torch-npu"   _run_local_bench_torch_   _run_local_bench_torch_npu_
                         npu_profiler              profiler_parallel
bench_mode="perf-counter" _run_local_bench_perf_   _run_local_bench_perf_
                          counter                  counter_parallel
```

## Schema Changes

### `PerfCaseRecord` — new `bench_mode` field

```python
@dataclass(frozen=True)
class PerfCaseRecord:
    case_label: str
    kernel_names: list[str]
    kernel_source: str
    metrics: PerfMetrics | None = None
    error_message: str | None = None
    case_wall_clock_seconds: float | None = None
    bench_mode: str | None = None  # new
```

Default `None` for backward compatibility with existing records.

### JSONL output — new `"bench_mode"` key

`render_perf_case_record_jsonl()` includes `"bench_mode": record.bench_mode` in the payload. For existing records where `bench_mode` is `None`, the key is included as `"bench_mode": null` — `json.dumps` serializes `None` as `null` by default. If omitting the key for old records is desired, the renderer must explicitly filter it.

When `bench_mode == "perf-counter"` and `ops` is empty, `total_op_avg_time_us` is set to `kernel_avg_time_us` (not `0`):

```python
if metrics is not None:
    kernel_avg_time_us = metrics["kernel_avg_time_us"]
    ops = metrics["ops"]
    if ops:
        total_op_avg_time_us = sum(op["avg_time_us"] for op in ops)
    elif record.bench_mode == "perf-counter":
        total_op_avg_time_us = kernel_avg_time_us
    else:
        total_op_avg_time_us = 0.0
```

This ensures `--metric-source total-op` and `--metric-source all` produce meaningful comparisons for `perf-counter` results — the per-case timing is treated as the total operator time. Optimize/operator workflows that rely on `total-op` or `all` continue to work without special-casing.

Example output for a perf-counter record:
```json
{"case_label":"case1","kernel_names":["my_kernel"],"kernel_source":"metadata","kernel_avg_time_us":12.5,"ops":[],"total_op_avg_time_us":12.5,"error_message":null,"case_wall_clock_seconds":1.23,"bench_mode":"perf-counter"}
```

## Metrics Mapping

The `PerfCaseRecord` produced by this mode:

| Field | Value |
|-------|-------|
| `case_label` | case id (unchanged) |
| `kernel_names` | resolved kernel names (unchanged) |
| `kernel_source` | `"metadata"`, `"operator"`, or `"metadata+operator"` (unchanged) |
| `metrics.kernel_avg_time_us` | average per-iteration time in microseconds (from `perf_counter`) |
| `metrics.ops` | `[]` (empty — no profiler means no per-kernel attribution) |
| `error_message` | error string or `None` (unchanged) |
| `case_wall_clock_seconds` | total wall clock for the case including warmup (from `time.monotonic()`) |
| `bench_mode` | `"perf-counter"` |

Key decision: populate `kernel_avg_time_us` with the per-iteration average so `compare-perf --metric-source kernel` works without changes. In this mode the value represents "average case iteration wall-clock time" rather than "NPU kernel execution time."

## Files to Change

### 1. `skills/triton-npu-run-eval/scripts/perf_artifacts.py`

- Add `bench_mode: str | None = None` to `PerfCaseRecord` (after line 28).
- Update `render_perf_case_record_jsonl()` to include `"bench_mode": record.bench_mode` in the output payload.

### 2. `skills/triton-npu-run-eval/scripts/bench_runtime.py`

- Add `time_all_bench_cases()` — public entry point, mirrors `profile_all_bench_cases()` but uses perf_counter loop. Must set `TRITON_ALWAYS_COMPILE=1`. Passes `bench_mode="perf-counter"` to every `PerfCaseRecord`.
- Add `_time_bench_case()` — per-case timing, mirrors `_run_bench_case()` but calls `_time_case_iterations()` instead of `_profile_case_with_profiler()`. Sets `bench_mode="perf-counter"` on the record.
- Add `_time_case_iterations()` — the core perf_counter loop (warmup + timed repeats + sync). Returns `PerfMetrics`. Unlike `_profile_case_with_profiler()` which returns `tuple[PerfMetrics | None, str | None]` (with the error string handling CSV parse failures), this function has no profiler to fail — but `case.fn()` can still raise. That exception is handled by the caller `_time_bench_case()`, which must wrap the call in try/except, catch the exception, and produce a `PerfCaseRecord` with `error_message` set and `case_wall_clock_seconds` recorded (mirroring how `_run_bench_case` at line 375-382 handles errors from `_profile_case_with_profiler`).
- Update existing call sites (`profile_all_bench_cases`, `_run_bench_case`, msprof paths) to set `bench_mode` on their records (`"torch-npu-profiler"`, `"msprof"` respectively).

### 3. `skills/triton-npu-run-eval/scripts/bench_runner.py`

- Add `_run_local_bench_perf_counter()` — delegates to `runtime.time_all_bench_cases()`
- Add `_run_local_bench_perf_counter_parallel()` — parallel variant with NPU device pool
- Add `_run_remote_bench_perf_counter()` — remote variant
- Add `_run_remote_bench_perf_counter_parallel()` — remote parallel variant
- Update `run_local_bench()` dispatch (line 96) to branch on `bench_mode == "perf-counter"`
- Update `run_remote_bench()` dispatch (line 172) similarly
- Update existing msprof and torch-npu-profiler paths to pass `bench_mode` through to record construction. This includes the serialization bridge in parallel mode: `_build_torch_npu_profiler_run_one_case_script()` (line 1239) constructs an inline Python script that prints JSON; its payload dict must include `"bench_mode"`. `_run_local_torch_npu_profiler_case_in_subprocess()` (line 1264) and `_run_remote_torch_npu_profiler_case_in_subprocess()` (line 1402) parse that JSON and construct `PerfCaseRecord` — both must read the new field. Without this, `bench_mode` is silently dropped in parallel profiler paths.

### 4. `skills/triton-npu-run-eval/scripts/run-command.py`

- **`run_bench` path** (line 413-416): Remove the `_resolve_bench_mode_from_metadata()` fallback. Replace `resolved_bench_mode = args.bench_mode or _resolve_bench_mode_from_metadata(bench_file)` with `resolved_bench_mode = args.bench_mode or "torch-npu-profiler"`.
- **`run_bench` subparser** (line 201): Add `"perf-counter"` to `--bench-mode` choices, default `"torch-npu-profiler"`.
- **`profile-bench` path** (line 362-366): Remove `--bench-mode` argument entirely. Remove the `_resolve_bench_mode_from_metadata()` call. Hardcode `resolved_bench_mode = "torch-npu-profiler"`. `profile-bench` always uses the profiler.
- **`profile_bench` subparser** (line 207): Remove `--bench-mode` argument.
- **`_resolve_bench_mode_from_metadata()` (line 629)**: Remove the function. It is no longer called by any code path.

### 5. `skills/triton-npu-run-eval/scripts/profile_runner.py`

- Remove `_normalize_bench_mode()` (line 29-30) — no longer needed when bench mode is hardcoded.
- `run_local_profile_bench()` and `run_remote_profile_bench()`: Remove `bench_mode` parameter, always use `torch-npu-profiler`.

### 6. `skills/triton-npu-run-eval/scripts/bench_contract.py`

- Remove `bench-mode` parsing from `parse_bench_metadata()`. The function still parses `# kernel:` / `# kernels:` and other metadata, just not `# bench-mode`.

### 7. `src/helix/cli.py`

- **`run-bench` command**: `_BENCH_MODE_CHOICES` updated to `("torch-npu-profiler", "msprof", "perf-counter")`. Default `"torch-npu-profiler"`.
- **`profile-bench` path** (if represented in CLI — check `src/helix/cli.py`): Remove `--bench-mode` if present.
- **Other commands** (`gen-eval`, `gen-bench`, `verify`, `optimize`, etc.): `--bench-mode` still informs the agent prompt about the target execution mode, but the generated bench file no longer writes a `# bench-mode` header — the generated file is mode-agnostic. `perf-counter` is accepted on generation commands too (the shared `_BENCH_MODE_CHOICES` covers all commands with `has_bench_mode=True`); on the generation path it simply tells the agent the user intends to run with `perf-counter`.
- **`_BENCH_MODE_CHOICES`**: Add `"perf-counter"`. All commands with `has_bench_mode=True` share the same choices tuple.

### 8. `src/helix/execution.py`

- **`resolve_bench_mode_from_metadata()` (line 225-232)**: Remove the function. Bench mode is no longer resolved from file metadata.
- **`run_local_bench()` / `run_remote_bench()`**: Accept `bench_mode` parameter as before. Callers now pass the CLI value or the default `"torch-npu-profiler"`.
- **`handle_run_bench()` (in `commands/execution.py`)**: Replace `args.bench_mode or resolve_bench_mode_from_metadata(bench_file)` with `args.bench_mode or "torch-npu-profiler"`.

### 9. `src/helix/optimize/resume.py`

Remove `_parse_bench_mode()` (line 334) — bench mode is no longer parsed from harness file headers. The optimize session's canonical bench mode is stored in `baseline/state.json`, which already has `bench_mode` as a required field (`BaselineState.bench_mode`).

**`_classify_optimize_workspace()` (lines 113-120)**: Replace `bench_mode = _parse_bench_mode(bench_harness)` with:

```python
bench_mode = _resolve_bench_mode_from_baseline(workdir)
```

Where `_resolve_bench_mode_from_baseline()` loads `BaselineState` via `load_baseline_state(workdir)` and returns `state.bench_mode`. If `baseline/state.json` is missing or unreadable, classify as `"partial-session"`.

**`_require_resumable_session()` (lines 219-232)**: The bench harness existence check at line 219-223 stays (the file must exist), but lines 229-232 change from checking `inspection.bench_mode is None` (which was derived from harness metadata) to checking the baseline-derived value. The error message changes from `"unreadable bench-mode metadata"` to `"unable to resolve bench mode from baseline/state.json"`.

This is actually cleaner than the current design: `baseline/state.json` is already the canonical source of truth for the optimize session's configuration, and it already stores `bench_mode`. The harness file metadata was a redundant (and now removed) copy.

### 10. `skills/triton-npu-analyze-ir/scripts/capture_ir.py`

- `_resolve_bench_mode()` (line 548-561): This function reads `# bench-mode` from bench file headers. Since bench files no longer carry this header, default to `"torch-npu-profiler"` (the function already defaults to `"torch-npu-profiler"` at line 561 when no header is found). No code change required — the existing fallback handles the new behavior correctly.

### 11. `skills/triton-npu-run-eval/scripts/perf_artifacts.py` — compare-perf cross-mode check

The `bench_mode` field added to JSONL (section 1) enables `compare-perf` to detect cross-mode comparisons. Add a check in `_parse_perf_entries_from_jsonl()` or the compare entry point:

- Parse `bench_mode` from each JSONL line.
- If baseline and candidate have different `bench_mode` values (both non-None), **reject with an error**: `"cannot compare results from different bench modes: baseline=<A>, candidate=<B>."` The metrics have fundamentally different meanings across modes (profiler-derived kernel latency vs. case wall-clock); cross-mode comparison is not meaningful.
- If `bench_mode` is `None` on **both** sides (both records from before this change), skip the cross-mode check — neither side carries mode info.
- If `bench_mode` is `None` on only one side and the other side has a known mode, **reject**: `"cannot compare results: one input has bench_mode=<A> and the other has no bench_mode (pre-perf-counter record)."` A pre-change record cannot be `perf-counter`, so the comparison is not valid.

Note: `perf-counter` results set `total_op_avg_time_us = kernel_avg_time_us` (see section 1), so `--metric-source total-op` and `--metric-source all` work without special handling. No per-mode `metric_source` guard is needed.

### 12. `src/helix/run_eval_mcp_server.py`

- **Line 168** (`run_bench` tool): Update `bench_mode` field description to include `"perf-counter"`.
- **Line 206** (`profile_bench` tool): Remove `bench_mode` field from the tool schema.

### 13. `skills/triton-npu-gen-bench/references/bench-spec.md`

This is the generation-side spec for bench file format. Since bench mode is no longer a property of the bench file:

- **Line 3**: Remove `# bench-mode: <torch-npu-profiler|msprof>` from the mode-marker description.
- **Lines 16-21**: Remove the `# bench-mode:` line from the metadata header block. Keep `# api-name:`, `# api-kind:`, and `# kernels:`.
- **Lines 23-26**: Remove the `# bench-mode:` field description. Replace with a note that bench mode is a runtime concern owned by execution tooling.
- **Line 132**: Remove `# bench-mode: torch-npu-profiler` from the example header.
- **Line 163**: Remove the sentence describing how to replace bench-mode in generated files.

### 14. `skills/triton-npu-gen-bench/SKILL.md`

The generation skill's instructions reference `# bench-mode` as a required output of the generation process (lines 27, 44). Update:
- Remove the instruction to write `# bench-mode: ...` into the generated bench file header.
- Replace with a note that bench mode is a runtime concern. The generated bench file is mode-agnostic and works with any bench mode (`torch-npu-profiler`, `msprof`, or `perf-counter`).

### 15. `skills/triton-npu-run-eval/references/run-bench.md`

- **Line 14**: Replace "reads `# bench-mode: ...` from the benchmark file" with "defaults to `torch-npu-profiler`."
- **Line 15**: Add `perf-counter` to the override description: `--bench-mode torch-npu-profiler`, `--bench-mode msprof`, or `--bench-mode perf-counter`.
- Add a mode note for `perf-counter`: "In `perf-counter` mode, the runner measures per-iteration wall-clock time with `time.perf_counter()` instead of using NPU profiling. Results include `kernel_avg_time_us` (per-iteration average) but `ops` is empty. Results are only comparable within the same mode."

### 16. `skills/triton-npu-run-eval/references/profile-bench.md`

- **Line 13**: Remove "If `--bench-mode` is omitted, the command reads `# bench-mode: ...` from the benchmark file." `profile-bench` no longer accepts `--bench-mode`.
- **Lines 15-20**: Replace mode-specific notes with a single statement: "`profile-bench` always uses `torch_npu.profiler`. It does not support `perf-counter` (which produces no profiler traces)."

### 17. `skills/triton-npu-run-eval-mcp/references/run-bench.md`

- Same changes as section 15 (run-eval/references/run-bench.md): update default to `torch-npu-profiler`, add `perf-counter` to valid values, add mode note.

### 18. `skills/triton-npu-run-eval-mcp/references/profile-bench.md`

- Same changes as section 16 (run-eval/references/profile-bench.md): remove `bench_mode`, state profile-bench always uses profiler.

### 19. `skills/triton-npu-run-eval/references/compare-perf.md`

- **Line 31**: The `total-op` description says "requires raw op statistics for total-op aggregation." For `perf-counter` results, `total_op_avg_time_us` equals `kernel_avg_time_us` — no raw op statistics are needed. Update to: "`total-op` requires total-op aggregation (from raw op statistics in profiler modes, or derived from `kernel_avg_time_us` in `perf-counter` mode)."
- Add a note that `perf-counter` results support all `--metric-source` values: `kernel`, `total-op`, and `all` produce the same value (since `total_op_avg_time_us == kernel_avg_time_us`); `auto` also works and resolves to the same result via the existing kernel-first fallback.

### 20. `skills/triton-npu-run-eval-mcp/references/compare-perf.md`

- Same changes as section 19 (run-eval/references/compare-perf.md): update `total-op` description, add `perf-counter` note.

## Implementation Order

1. Add `bench_mode` field to `PerfCaseRecord` and JSONL renderer in `perf_artifacts.py`
2. Add `_time_case_iterations()`, `_time_bench_case()`, `time_all_bench_cases()` in `bench_runtime.py`
3. Add perf-counter runner functions in `bench_runner.py` + update dispatch
4. Update existing record construction sites to set `bench_mode`
5. Update `run-command.py`: remove `_resolve_bench_mode_from_metadata()`, update run-bench default, hardcode profile-bench
6. Update `profile_runner.py`: remove `bench_mode` parameter
7. Update `bench_contract.py`: drop `bench-mode` metadata parsing
8. Update `cli.py`: add `perf-counter` to choices, update defaults
9. Update `execution.py` / `commands/execution.py`: delete `resolve_bench_mode_from_metadata()`, use plain default `"torch-npu-profiler"`
10. Update `resume.py`: replace `_parse_bench_mode()` with `_resolve_bench_mode_from_baseline()`, switching source of truth to `baseline/state.json`
11. Update `perf_artifacts.py`: add cross-mode error rejection in compare-perf path
12. Update `run_eval_mcp_server.py` docstrings
13. Update `bench-spec.md`: remove `# bench-mode` header requirement
14. Update `SKILL.md` (gen-bench): remove `# bench-mode` from generation instructions
15. Update `run-bench.md` (both copies): update default, add `perf-counter`
16. Update `profile-bench.md` (both copies): remove `bench_mode`, hardcode profiler
17. Update `compare-perf.md` (both copies): update `total-op` description for `perf-counter`
18. Run `pyright` strict check on modified skill scripts

## Test Plan

| Test | What it verifies |
|------|-----------------|
| `test_time_case_iterations_returns_valid_metrics` | `_time_case_iterations()` returns `PerfMetrics` with `kernel_avg_time_us > 0` and `ops=[]` |
| `test_time_all_bench_cases_produces_jsonl` | `time_all_bench_cases()` writes valid JSONL with `ops=[]`, non-None `kernel_avg_time_us`, `case_wall_clock_seconds`, and `bench_mode="perf-counter"` |
| `test_time_all_bench_cases_sets_triton_always_compile` | `TRITON_ALWAYS_COMPILE=1` is set during execution and restored afterwards |
| `test_run_local_bench_dispatches_perf_counter` | `run_local_bench(bench_mode="perf-counter")` routes to `_run_local_bench_perf_counter` |
| `test_run_local_bench_defaults_to_torch_npu_profiler` | `run_bench` with no `--bench-mode` defaults to `torch-npu-profiler` |
| `test_perf_counter_case_wall_clock_seconds_on_failure` | A case that raises still gets `case_wall_clock_seconds` and `bench_mode="perf-counter"` recorded |
| `test_perf_case_record_jsonl_includes_bench_mode` | JSONL output contains `"bench_mode":"perf-counter"` for perf-counter records, `"bench_mode":"torch-npu-profiler"` for torch-npu-profiler records |
| `test_profile_bench_has_no_bench_mode_arg` | `profile_bench` subparser does not accept `--bench-mode` |
| `test_bench_contract_no_longer_parses_bench_mode` | `parse_bench_metadata()` on a bench file with `# bench-mode: perf-counter` does not include `"bench-mode"` in the returned dict |
| `test_capture_ir_defaults_bench_mode` | `_resolve_bench_mode()` returns `"torch-npu-profiler"` for a bench file without `# bench-mode` header |
| `test_compare_perf_rejects_mismatched_bench_mode` | `compare-perf` with baseline `bench_mode="msprof"` and candidate `bench_mode="perf-counter"` raises an error |
| `test_compare_perf_rejects_known_vs_none_bench_mode` | `compare-perf` with one side `bench_mode="perf-counter"` and the other `bench_mode=null` raises an error |
| `test_compare_perf_skips_check_when_both_none` | `compare-perf` with both sides `bench_mode=null` proceeds without cross-mode error |
| `test_perf_counter_total_op_equals_kernel_avg` | JSONL output for `perf-counter` has `total_op_avg_time_us == kernel_avg_time_us` |
| `test_compare_perf_all_metric_source_works_with_perf_counter` | `--metric-source all` on two `perf-counter` JSONL files produces valid speedup (uses `total_op_avg_time_us` which equals `kernel_avg_time_us`) |
