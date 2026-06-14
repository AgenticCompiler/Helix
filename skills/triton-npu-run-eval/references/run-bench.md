# `run-bench`

Run a generated benchmark with:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py
```

Rules:

- Always pass both `--bench-file` and `--operator-file`.
- If `--bench-mode` is omitted, the command reads `# bench-mode: ...` from the benchmark file.
- Use `--bench-mode standalone`, `--bench-mode msprof`, or `--bench-mode msprof-simulator` only when you need to override the embedded metadata.
- On success, `run-bench` prints `Perf file: <path>` and a short hint to use `compare-perf` instead of reading perf files directly.
- On failure, `run-bench` prints the captured benchmark output so the error remains diagnosable.

Mode notes:

- In `standalone` mode, the benchmark file is import-only. `run-bench` imports the module, calls `build_operator_api(operator_module)`, then calls `build_standalone_bench_cases(operator_api)`.
- In `standalone` mode, the runner profiles each declared case with `torch_npu.profiler` and writes `latency-<case-id>` perf entries.
- In `msprof` mode, `run-bench` aggregates the stable-order union of benchmark metadata kernels and `@triton.jit` kernels discovered from the runtime `--operator-file`.
- In `msprof` mode, a failed benchmark case does not stop later cases from running; the generated perf file keeps successful cases and records `# latency-error-case-*` comments for failed ones.
- In `msprof` mode, kernel-miss cases still write `latency-case-*: NA`, but also include raw op statistics plus a `# latency-error-case-*` explanation.
In `msprof-simulator` and `standalone` modes, the extracted simulation data (`extracted_bin_data/`) is copied to the directory specified by `--extract-dest-dir` if provided; otherwise it defaults to the parent directory of `--bench-file`. Pass `--extract-dest-dir baseline/` during baseline preparation, and pass `--extract-dest-dir opt-round-N/` to keep each round's data separate.
- In `msprof-simulator` mode, `--simulator-case-idx <N>` specifies which benchmark case to run for simulation. Defaults to 1. Use this when a prior optimization round has identified a specific case as the focus.

Examples:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py --bench-mode standalone --extract-dest-dir opt-round-N/
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode msprof
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file baseline/<operator>.py --bench-mode msprof-simulator --extract-dest-dir baseline
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode msprof-simulator --extract-dest-dir opt-round-N/
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode msprof-simulator --simulator-case-idx 3 --extract-dest-dir opt-round-N/
```

Remote examples:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py --remote user@host:2222
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode msprof --remote user@host:2222 --remote-workdir /tmp/triton-agent
```
