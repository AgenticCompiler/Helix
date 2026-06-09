# `run-bench`

Use the `run-bench` MCP tool to execute a generated benchmark.

Rules:

- Always pass both `bench_file` and `operator_file`.
- If `bench_mode` is omitted, the tool reads `# bench-mode: ...` from the benchmark file.
- Use `bench_mode="standalone"` or `bench_mode="msprof"` only when you need to override the embedded metadata.
- On success, `run-bench` prints `Perf file: <path>` and a short hint to use `compare-perf` instead of reading perf files directly.
- On failure, `run-bench` prints the captured benchmark output so the error remains diagnosable.

Mode notes:

- In `standalone` mode, the benchmark file is import-only. `run-bench` imports the module, calls `build_operator_api(operator_module)`, then calls `build_standalone_bench_cases(operator_api)`.
- In `standalone` mode, the runner profiles each declared case with `torch_npu.profiler` and writes `latency-<case-id>` perf entries.
- In `msprof` mode, `run-bench` aggregates the stable-order union of benchmark metadata kernels and `@triton.jit` kernels discovered from the runtime `operator_file`.
- In `msprof` mode, a failed benchmark case does not stop later cases from running; the generated perf file keeps successful cases and records `# latency-error-case-*` comments for failed ones.
- In `msprof` mode, kernel-miss cases still write `latency-case-*: NA`, but also include raw op statistics plus a `# latency-error-case-*` explanation.

Argument examples:

- `run-bench(bench_file="bench_<operator>.py", operator_file="<operator>.py")`
- `run-bench(bench_file="bench_<operator>.py", operator_file="opt_<operator>.py", bench_mode="msprof")`
- `run-bench(bench_file="bench_<operator>.py", operator_file="<operator>.py", remote="user@host:2222")`
- `run-bench(bench_file="bench_<operator>.py", operator_file="opt_<operator>.py", bench_mode="msprof", remote="user@host:2222", remote_workdir="/tmp/triton-agent")`
