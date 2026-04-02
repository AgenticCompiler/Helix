---
name: ascend-npu-operator-profiler
description: Get and analyze Ascend NPU operator performance data. Use for profiling Ascend operators, identifying hot operators and timing bottlenecks, summarizing performance evidence, comparing profiling results across runs, or inspecting profiler artifacts such as msprof outputs, op_statistic/op_summary CSV files, and Ascend profiler .bin data.
---

# Ascend NPU Operator Profiler

Profile Ascend NPU operators with `msprof` and summarize the resulting timing data.

## Default workflow

1. Run the target command by putting `msprof` directly in front of it.

   ```bash
   msprof python3 bench_matmul.py --operator-file matmul.py
   ```

2. Find the generated `PROF_*` directory near the command's working directory.

3. Summarize the profile output with the bundled script.

   ```bash
   python3 ./scripts/profile_summary.py <path-to-PROF-dir-or-parent> [--target-op <op-name>]
   ```

4. Present the summary in the conversation. Call out:
   - which operator was analyzed
   - whether that operator was explicit or inferred
   - total, average, min, and max time
   - runtime ratio and top hotspots

## Working rules

- Prefer the direct `msprof <command>` form unless the user explicitly needs another invocation style.
- Treat `op_summary_*.csv` as potentially large. Do not dump it into the conversation or read it naively line-by-line into memory if a streaming pass is enough.
- Prefer the bundled summary script over ad hoc shell parsing so the report stays consistent.
- If the user names the operator, pass it through `--target-op`.
- If the user does not name the operator, let the script infer the hottest operator from `op_statistic` and state that this is an inference.
- Always show the final Markdown summary in the conversation instead of only mentioning generated files.

## Validation checks

- Accept any of these input shapes for CSV summarization: a `PROF_*` directory, a `mindstudio_profiler_output/` directory, or a parent directory that contains one of them.
- Before summarizing CSV output, confirm `mindstudio_profiler_output/` exists and contains at least one `op_statistic_*.csv`.
- When `op_summary_*.csv` exists, use it as a cross-check for the target operator but keep the main summary anchored on `op_statistic`.
- Before running binary parsing, confirm the `.bin` file exists and contains `ZZ{` markers.

## Failure handling

- If no profile directory can be resolved, re-run the profiling command and check the working directory.
- If `mindstudio_profiler_output/` or `op_statistic_*.csv` is missing, stop and report that the profiler output is incomplete.
- If the requested operator is not present in `op_statistic`, list the available operators and ask the user to confirm the target.
- If `op_summary` uses unrecognized columns, fall back to `op_statistic` and mention that the `op_summary` aggregation was unavailable.
- If `.bin` parsing finds no `ZZ{` markers, treat the file as unsupported profiler data instead of guessing.

## Bundled resources

- `scripts/profile_summary.py`
  Use this to locate the relevant `PROF_*` directory, read `op_statistic` and `op_summary`, and render a concise Markdown report.

- `scripts/parse_bin.py`
  Keep this for raw profiler binary inspection when the user provides files such as `visualize_data.bin`.

- `references/profiler-csv-spec.md`
  Read this when you need the expected CSV layout or column names for `op_statistic` and `op_summary`.

- `references/binary-format-spec.md`
  Read this only when the user needs `.bin` parsing details.

## Examples

Profile a benchmark and summarize the hottest operator:

```bash
msprof python3 bench_matmul.py --operator-file matmul.py
python3 <skill-path>/scripts/profile_summary.py .
```

Profile a benchmark and summarize a known operator:

```bash
msprof python3 bench_matmul.py --operator-file matmul.py
python3 <skill-path>/scripts/profile_summary.py . --target-op MatMul
```

Inspect a raw profiler binary block:

```bash
python3 <skill-path>/scripts/parse_bin.py visualize_data.bin --block-id 0
```
