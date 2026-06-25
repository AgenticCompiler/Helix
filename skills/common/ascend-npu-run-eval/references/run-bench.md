# `run-bench`

Run a generated benchmark with:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py
python3 ./scripts/run-command.py run-bench --bench-file bench_a.py --operator-file a.py --output ./artifacts/a_perf.txt
```

Rules:

- Always pass both `--bench-file` and `--operator-file`.
- If `--bench-mode` is omitted, defaults to `torch-npu-profiler`.
- Use `--bench-mode torch-npu-profiler`, `--bench-mode msprof`, or `--bench-mode perf-counter` only when you need to override the default.
- Use `--output <path>` when you need the perf artifact at a specific location.
- On success, `run-bench` prints `Perf file: <path>` and a short hint to use `compare-perf` instead of reading perf files directly.
- On failure, `run-bench` prints the captured benchmark output so the error remains diagnosable.

Mode notes:

- In both modes, the benchmark file is import-only. `run-bench` imports the module, calls `build_operator_api(operator_module)`, reads the declared cases from `build_bench_cases()`, and constructs each executable case via `build_bench_case_fn(operator_api, case)`.
- In `torch-npu-profiler` mode, the runner profiles each declared case with `torch_npu.profiler` and writes `latency-<case-id>` perf entries.
- In `msprof` mode, `run-bench` aggregates the stable-order union of benchmark metadata kernels and `@triton.jit` / `@tilelang.jit` kernels discovered from the runtime `--operator-file`.
- In `msprof` mode, a failed benchmark case does not stop later cases from running; the generated perf file keeps successful cases and records `# latency-error-case-*` comments for failed ones.
- In `msprof` mode, kernel-miss cases still write `latency-<case-id>: NA`, but also include raw op statistics plus a `# latency-error-case-*` explanation.
- In `perf-counter` mode, the runner measures per-iteration wall-clock time with `time.perf_counter()` instead of using NPU profiling. Results include `kernel_avg_time_us` (per-iteration average) but `ops` is empty. `total_op_avg_time_us` equals `kernel_avg_time_us`. Results are only comparable within the same mode; `compare-perf` rejects cross-mode comparisons.

Examples:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py --bench-mode torch-npu-profiler
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode msprof
```

Remote examples:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py --remote user@host:2222
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode msprof --remote user@host:2222 --remote-workdir /tmp/triton-agent
```
