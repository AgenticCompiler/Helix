# `run-bench`

Run a generated benchmark with:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py
```

Rules:

- Always pass both `--bench-file` and `--operator-file`.
- If `--bench-mode` is omitted, the command reads `# bench-mode: ...` from the benchmark file.
- Use `--bench-mode standalone` or `--bench-mode msprof` only when you need to override the embedded metadata.
- Use `--npu-devices 0,1,4-7` when you want `run-bench` to execute benchmark cases concurrently across multiple Ascend devices. The list supports inclusive numeric ranges, rejects empty or duplicate entries, and leaves behavior unchanged when omitted.
- On success, `run-bench` prints `Perf file: <path>` and a short hint to use `compare-perf` instead of reading perf files directly.
- On failure, `run-bench` prints the captured benchmark output so the error remains diagnosable.

Mode notes:

- In `standalone` mode, the benchmark file is import-only. `run-bench` imports the module, calls `build_operator_api(operator_module)`, then calls `build_standalone_bench_cases(operator_api)`.
- In `standalone` mode, the runner profiles each declared case with `torch_npu.profiler` and writes `latency-<case-id>` perf entries.
- In `standalone` mode, `--npu-devices` runs declared standalone cases in parallel through isolated case workers and assigns one visible device per case.
- In `msprof` mode, `run-bench` aggregates the stable-order union of benchmark metadata kernels and `@triton.jit` kernels discovered from the runtime `--operator-file`.
- In `msprof` mode, a failed benchmark case does not stop later cases from running; the generated perf file keeps successful cases and records `# latency-error-case-*` comments for failed ones.
- In `msprof` mode, kernel-miss cases still write `latency-case-*: NA`, but also include raw op statistics plus a `# latency-error-case-*` explanation.
- In `msprof` mode, `--npu-devices` runs benchmark cases in parallel through isolated case workspaces and assigns one visible device per case.

Examples:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py --bench-mode standalone
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode msprof
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode msprof --npu-devices 0,1,2,3
```

Remote examples:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py --remote user@host:2222
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode msprof --remote user@host:2222 --remote-workdir /tmp/triton-agent
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode standalone --remote user@host:2222 --remote-workdir /tmp/triton-agent --npu-devices 0-3
```

Remote note:

- When `--remote` and `--npu-devices` are combined, the device list applies to the one remote target host named by `--remote`.
