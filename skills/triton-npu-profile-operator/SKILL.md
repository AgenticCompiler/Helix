---
name: triton-npu-profile-operator
description: Get and analyze Ascend NPU operator performance data. Use for profiling Ascend operators, identifying hot operators and timing bottlenecks, summarizing performance evidence, comparing profiling results across runs, or inspecting profiler artifacts such as msprof outputs, op_statistic/op_summary CSV files, and Ascend profiler .bin data.
---

# Ascend NPU Operator Profiler

Profile Ascend NPU operators with `msprof` and summarize the resulting timing data.

## Default workflow

1. Profile benchmark harnesses through the unified triton-npu-run-eval helper.

   ```bash
   python3 ../triton-npu-run-eval/scripts/run-command.py profile-bench --bench-file bench_matmul.py --operator-file matmul.py
   ```

2. Let the helper print or copy back the generated local `PROF_*` directory.

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

- Prefer `python3 ../triton-npu-run-eval/scripts/run-command.py profile-bench ...` for benchmark profiling, especially when the workflow is remote-aware.
- If the benchmark metadata says `# bench-mode: standalone`, profile `python3 bench_<op>.py --operator-file <operator-file>` and do not pass `--bench`; standalone mode must not receive `--bench` or `--num-bench`.
- If the benchmark metadata says `# bench-mode: msprof`, first query `--num-bench`, then profile one selected `--bench <N>` case; this mode requires resolvable `# kernel:` metadata in the benchmark header.
- When the outer task is remote-aware, pass the same `--remote` and `--remote-workdir` settings through `profile-bench` so profiling runs on the remote machine while the resulting `PROF_*` directory is copied back locally.
- Keep direct `msprof <command>` only as a local fallback when there is no generated benchmark harness or when the user explicitly wants a manual invocation.
- Treat `op_summary_*.csv` as potentially large. Do not dump it into the conversation or read it naively line-by-line into memory if a streaming pass is enough.
- Prefer the bundled summary script over ad hoc shell parsing so the report stays consistent.
- If the user names the operator, pass it through `--target-op`.
- If the user does not name the operator, let the script infer the hottest operator from `op_statistic` and state that this is an inference.
- Always show the final Markdown summary in the conversation instead of only mentioning generated files.

## Bench mode contract

- `standalone`
  - Use:
    ```bash
    python3 ../triton-npu-run-eval/scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file <operator>.py
    ```
  - Runtime command shape inside the helper:
    ```bash
    msprof python3 bench_<operator>.py --operator-file <operator>.py
    ```
  - Do not pass `--bench` or `--num-bench`.

- `msprof`
  - Use:
    ```bash
    python3 ../triton-npu-run-eval/scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file <operator>.py --bench 1
    ```
  - The helper will first query `python3 bench_<operator>.py --num-bench`.
  - The helper then profiles one selected `--bench <N>` case and defaults to case `1` when `--bench` is omitted.
  - This mode requires benchmark metadata with `# kernel: <resolved_kernel_name>`.

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
python3 ../triton-npu-run-eval/scripts/run-command.py profile-bench --bench-file bench_matmul.py --operator-file matmul.py
```

Profile a benchmark and summarize a known operator:

```bash
python3 ../triton-npu-run-eval/scripts/run-command.py profile-bench --bench-file bench_matmul.py --operator-file matmul.py --target-op MatMul
```

Profile one `msprof` benchmark case on a remote machine and keep the remote workspace:

```bash
python3 ../triton-npu-run-eval/scripts/run-command.py profile-bench --bench-file bench_matmul.py --operator-file opt_matmul.py --bench 2 --remote user@host:2222 --remote-workdir /tmp/triton-agent --keep-remote-workdir
```

Fallback manual profiling when no benchmark harness exists:

```bash
msprof python3 bench_matmul.py --operator-file matmul.py
python3 <skill-path>/scripts/profile_summary.py . --target-op MatMul
```

Inspect a raw profiler binary block:

```bash
python3 <skill-path>/scripts/parse_bin.py visualize_data.bin --block-id 0
```
