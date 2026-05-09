# Profile Parser Split Design

## Summary

Split profiling data parsing into two mode-native parsers — one for `msprof` output, one for `torch_npu.profiler` (standalone) output — that read their native artifacts directly and produce a unified result model. Eliminate the current fake-CSV synthesis path (`_materialize_msprof_view`) that forces standalone data through the msprof pipeline and loses rich pipeline ratio data.

## Motivation

### Current Architecture

The current profiling pipeline forces both bench modes through a single parsing path:

```
msprof run       → PROF_*/mindstudio_profiler_output/  → profile_summary.py (reads op_statistic + op_summary + ...)
standalone run   → operator_details.csv                → _materialize_msprof_view() → FAKE op_statistic.csv (Core Type: UNKNOWN)
                  feeds into the same profile_summary.py path → distorted results
```

Problems:
- **Data loss**: standalone's `kernel_details.csv` contains the same pipeline ratios as msprof's `op_summary.csv` (`aic_mac_ratio`, `aiv_vec_ratio`, `cube_utilization`), but it is never read.
- **Fake data**: `_materialize_msprof_view` synthesizes `op_statistic` with `Core Type: UNKNOWN` and `Count: 1`, losing real aggregation.
- **Monolithic parser**: `profile_summary.py` is ~950 lines mixing CSV parsing, classification, Markdown rendering, and JSON output.
- **Implicit contracts**: Data flows through untyped dicts and TypedDicts spread across multiple files.

### Goal

Two native parsers that each understand their mode's artifacts, with no synthesis or faking:

```
msprof run       → PROF_*/mindstudio_profiler_output/  → MsprofProfileParser  ↘
standalone run   → ASCEND_PROFILER_OUTPUT/             → StandaloneProfileParser → ParsedProfile (typed, with | None for unavailable fields)
```

## Benchmark Modes

### msprof mode

- Bench file: runnable CLI (`--num-bench`, `--bench N`, `--operator-file`)
- Profiler: `msprof python3 bench_<op>.py --bench <N>` (CANN-level profiler)
- Output: `PROF_XXXX/` directory

### standalone mode

- Bench file: import-only module (`build_operator_api()`, `build_standalone_bench_cases()`)
- Profiler: `torch_npu.profiler.profile()` (PyTorch profiler API, Level1)
- Output: `*_ascend_pt/ASCEND_PROFILER_OUTPUT/` directory

## Artifact Catalog

### msprof Mode: `PROF_*/mindstudio_profiler_output/`

| File | Rows | Key Data |
|---|---|---|
| `op_statistic_*.csv` | Per-operator-type | OP Type, **Core Type**, Count, Total/Min/Avg/Max Time(us), Ratio(%) |
| `op_summary_*.csv` | Per-kernel invocation | Op Name, OP Type, Task Type, Task Duration(us), Task Wait Time(us), Block Dim, **aic_mac_ratio**, aic_scalar_ratio, aic_mte1/2/3_ratio, **aiv_vec_ratio**, aiv_scalar_ratio, aiv_mte2/3_ratio, **cube_utilization(%)** |
| `task_time_*.csv` | Per-task | kernel_name, kernel_type, task_time(us), task_start(us), task_stop(us) |
| `api_statistic_*.csv` | Per-host-API | Level, API Name, Time(us), Count, Avg(us) |
| `msprof_*.json` | Timeline events | name, pid, tid, ts, dur, ph, cat, args |

Unique to msprof: `task_time_*.csv`, `msprof_*.db`, `device_0/sqlite/*.db`, `.bin` files.

### standalone Mode: `ASCEND_PROFILER_OUTPUT/`

| File | Rows | Key Data |
|---|---|---|
| `op_statistic.csv` | Per-operator-type | OP Type, Core Type, Count, Total/Min/Avg/Max Time(us), Ratio(%) |
| `kernel_details.csv` | Per-kernel invocation | Step Id, Name, Type, Accelerator Core, Duration(us), Wait Time(us), Block Dim, **aic_mac_ratio**, aic_scalar_ratio, aic_mte1/2/3_ratio, **aiv_vec_ratio**, aiv_scalar_ratio, aiv_mte2/3_ratio, **cube_utilization(%)** |
| `operator_details.csv` | Per-torch-op | Name, Host Self Duration(us), Host Total Duration(us), Device Self Duration(us), Device Total Duration(us) |
| `api_statistic.csv` | Per-host-API | Level, API Name, Time(us), Count, Avg(us) |
| `step_trace_time.csv` | Per-step | Step, Computing(us), Communication(Not Overlapped)(us), Free(us), Stage(us) |
| `trace_view.json` | Timeline events | name, cat (cpu_op, async_npu, enqueue, dequeue), dur, ph |

Unique to standalone: `operator_details.csv`, `step_trace_time.csv`, `ascend_pytorch_profiler.db`.

### Cross-Mode Equivalence

| msprof file | standalone file | Equivalence |
|---|---|---|
| `op_statistic_*.csv` | `op_statistic.csv` | **Identical** schema |
| `op_summary_*.csv` | `kernel_details.csv` | **Same pipeline ratios**, different column naming |
| `api_statistic_*.csv` | `api_statistic.csv` | **Identical** schema |
| `msprof_*.json` | `trace_view.json` | Both timeline traces, different event categories |
| `task_time_*.csv` | — | msprof only |
| — | `operator_details.csv` | standalone only |
| — | `step_trace_time.csv` | standalone only |

### Column Name Mapping: op_summary ↔ kernel_details

Both files share the same pipeline ratio columns. Column names differ as follows:

| msprof `op_summary` | standalone `kernel_details` |
|---|---|
| Op Name | Name |
| OP Type | Type |
| Task Type | Accelerator Core |
| Task Duration(us) | Duration(us) |
| Task Wait Time(us) | Wait Time(us) |
| Task Start Time(us) | Start Time(us) |
| (missing) | Step Id |

All pipeline columns (`aic_mac_ratio`, `aiv_vec_ratio`, `cube_utilization(%)`, etc.) are **identical**.

## Data Model

```python
@dataclass
class OperatorStats:
    """Aggregated per-operator timing. Available in BOTH modes."""
    op_type: str
    core_type: str                    # msprof: Core Type; standalone: Core Type (from op_statistic)
    count: int
    total_time_us: float
    min_time_us: float
    avg_time_us: float
    max_time_us: float
    ratio_percent: float

@dataclass
class PipelineStage:
    """Per-pipeline-stage ratios from one kernel invocation."""
    aic_mac_ratio: float
    aic_scalar_ratio: float
    aic_mte1_ratio: float
    aic_mte2_ratio: float
    aic_mte3_ratio: float
    aiv_vec_ratio: float
    aiv_scalar_ratio: float
    aiv_mte2_ratio: float
    aiv_mte3_ratio: float
    cube_utilization: float
    block_dim: int

@dataclass
class KernelInvocation:
    """Per-invocation kernel data. Available in BOTH modes."""
    op_name: str
    duration_us: float
    wait_time_us: float
    block_dim: int
    pipeline: PipelineStage | None    # None for non-compute ops (Cast, RandomNormal, etc.)

@dataclass
class TaskRecord:
    """Task scheduler record. msprof ONLY."""
    kernel_name: str
    kernel_type: str
    task_time_us: float
    task_start_us: float
    task_stop_us: float

@dataclass
class HostApiCall:
    """Host API call timing. Available in BOTH modes."""
    api_name: str
    level: str
    time_us: float
    count: int
    avg_us: float

@dataclass
class TorchOpTiming:
    """PyTorch-level operator timing. standalone ONLY."""
    name: str
    host_self_us: float
    host_total_us: float
    device_self_us: float
    device_total_us: float

@dataclass
class StepTrace:
    """Per-step compute/communication breakdown. standalone ONLY."""
    step: int
    computing_us: float
    communication_not_overlapped_us: float
    overlapped_us: float
    free_us: float

@dataclass
class ParsedProfile:
    """Complete parsed profile. None where unavailable in source mode."""
    bench_mode: Literal["msprof", "standalone"]

    # Always available
    operators: list[OperatorStats]
    invocations: list[KernelInvocation]
    host_api_calls: list[HostApiCall]

    # msprof only
    task_timeline: list[TaskRecord] | None

    # standalone only
    torch_op_timing: list[TorchOpTiming] | None
    step_trace: list[StepTrace] | None
```

## Parser Architecture

### File Layout

```
skills/triton-npu-profile-operator/scripts/
├── profile_summary.py              # CLI entry point (thin, ≈100 lines)
├── models.py                       # All dataclasses (ParsedProfile, OperatorStats, etc.)
├── parsers/
│   ├── __init__.py
│   ├── base.py                     # ProfileParser ABC
│   ├── msprof_parser.py            # Parses mindstudio_profiler_output/
│   │   ├── parse_op_statistic(csv_path)       → list[OperatorStats]
│   │   ├── parse_op_summary(csv_path)         → list[KernelInvocation]
│   │   ├── parse_task_time(csv_path)          → list[TaskRecord]
│   │   ├── parse_api_statistic(csv_path)      → list[HostApiCall]
│   │   └── parse_msprof_json(json_path)       → dict (timeline events)
│   └── standalone_parser.py        # Parses ASCEND_PROFILER_OUTPUT/
│       ├── parse_op_statistic(csv_path)       → list[OperatorStats]
│       ├── parse_kernel_details(csv_path)     → list[KernelInvocation]
│       ├── parse_operator_details(csv_path)   → list[TorchOpTiming]
│       ├── parse_step_trace(csv_path)         → list[StepTrace]
│       ├── parse_api_statistic(csv_path)      → list[HostApiCall]
│       └── parse_trace_view(json_path)        → dict (timeline events)
├── reporter.py                     # ParsedProfile → Markdown / JSON
└── parse_bin.py                    # Existing: binary .bin extraction
```

### Parser Auto-Detection

Given a profile directory, detect the mode by checking for the mode-specific subdirectory:

```python
def detect_mode_and_resolve(profile_path: Path) -> tuple[Literal["msprof", "standalone"], Path]:
    """Returns (mode, output_dir) where output_dir is the artifacts directory."""
    # msprof: <dir>/mindstudio_profiler_output/
    msprof = profile_path / "mindstudio_profiler_output"
    if msprof.is_dir():
        return "msprof", msprof

    # standalone: <dir>/*/ASCEND_PROFILER_OUTPUT/
    standalone = list(profile_path.rglob("ASCEND_PROFILER_OUTPUT"))
    if standalone:
        return "standalone", max(standalone, key=lambda p: p.stat().st_mtime_ns)

    raise ValueError(f"Cannot determine profile mode for {profile_path}")
```

### Shared Parsers

`parse_op_statistic()` and `parse_api_statistic()` are shared between both parsers since the CSV schemas are identical. They can live in `parsers/base.py` or as standalone functions.

### Items to Remove

- `standalone_bench_runtime._materialize_msprof_view()` — the fake CSV synthesis
- `profile_summary.py.build_profile_payload()` — replaced by per-mode parsers
- The monolithic rendering in `profile_summary.py` — replaced by `reporter.py`
- `profile_summary._read_profiler_metrics()` — standalone parser reads `operator_details.csv` + `kernel_details.csv` directly

## Classification Logic

The existing classification logic (operator type: cube/vector/mix, bound: compute-bound/memory-bound/scalar-overhead/mixed) moves into `reporter.py` as a `ProfileAnalyzer` class that operates on `ParsedProfile`:

```python
class ProfileAnalyzer:
    def classify_operator_type(self, profile: ParsedProfile) -> Literal["cube", "vector", "mix", "other", "unknown"]
    def classify_bound(self, profile: ParsedProfile) -> BoundClassification | None
    def select_target_operator(self, profile: ParsedProfile, target_op: str | None) -> OperatorStats
```

The analyzer only fires when the required data is available (e.g., bound classification requires `KernelInvocation.pipeline` with valid ratios).

## CLI Changes

### New: `profile-report` subcommand

```bash
python3 run-command.py profile-report \
    --profile-dir PROF_000001_.../ \
    [--target-op matmul_kernel] \
    [--format json|markdown]
```

This command:
1. Auto-detects profile mode from the directory structure
2. Loads the appropriate parser
3. Parses all available artifacts
4. Runs analysis (classification, target selection)
5. Renders the report

### Existing: `profile-bench` keeps executing + reporting

No behavioral change to `profile-bench`. It continues to run profiling then show a summary. Internally, it may optionally delegate to `profile-report` logic for the summary step.

## Migration Path

1. **Phase 1**: Create `models.py`, `parsers/msprof_parser.py`, `parsers/standalone_parser.py`, `reporter.py` alongside existing `profile_summary.py`
2. **Phase 2**: Add `profile-report` CLI subcommand using new parsers
3. **Phase 3**: Switch `profile-bench`'s inline summary to use new `reporter.py`
4. **Phase 4**: Remove `_materialize_msprof_view()`, `_read_profiler_metrics()`, and dead code from `profile_summary.py`
5. **Phase 5**: Delete old `profile_summary.py` (renamed to `profile_summary.py` acting as thin CLI entry)

No backward compatibility concern — the new parsers produce richer data than the old pipeline, and the old pipeline's fake data (`UNKNOWN` core types) has no downstream consumers that depend on it.

## NPU Validation Notes

Validated on Ascend NPU (`cdj@192.168.9.225 /home/cdj/tmp`, torch_npu 2.7.1, CANN-8.5.0):

- **msprof run**: `msprof python3 bench_matmul.py --bench 1` produces `PROF_*/mindstudio_profiler_output/` with 5 CSVs + 1 JSON + 1 DB
- **standalone run**: `torch_npu.profiler.profile(level=Level1)` produces `*_ascend_pt/ASCEND_PROFILER_OUTPUT/` with 5 CSVs + 1 JSON + 2 DBs
- **Pipeline ratios confirmed**: both `op_summary.csv` (msprof) and `kernel_details.csv` (standalone) contain identical `aic_mac_ratio`, `aiv_vec_ratio`, `cube_utilization(%)` columns for AI_CORE kernel invocations
- **Non-compute ops**: `Cast`, `DSARandomNormal` rows have pipeline ratios at zero or N/A in both modes
