# `run-bench`

Run a generated benchmark with:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py
python3 ./scripts/run-command.py run-bench --bench-file bench_a.py --operator-file a.py --output ./artifacts/a_perf.txt
```

Rules:

- Always pass both `--bench-file` and `--operator-file`.
- If `--bench-mode` is omitted, the command reads `# bench-mode: ...` from the benchmark file.
- Use `--bench-mode torch-npu-profiler` or `--bench-mode msprof` only when you need to override the embedded metadata.
- Use `--output <path>` when you need the perf artifact at a specific location.
- On success, `run-bench` prints `Perf file: <path>` and a short hint to use `compare-perf` instead of reading perf files directly.
- On failure, `run-bench` prints the captured benchmark output so the error remains diagnosable.

Mode notes:

- In both modes, the benchmark file is import-only. `run-bench` imports the module, calls `build_operator_api(operator_module)`, reads the declared cases from `build_bench_cases()`, and constructs each executable case via `build_bench_case_fn(operator_api, case)`.
- In `torch-npu-profiler` mode, the runner profiles each declared case with `torch_npu.profiler` and writes `latency-<case-id>` perf entries.
- In `msprof` mode, `run-bench` aggregates the stable-order union of benchmark metadata kernels and `@triton.jit` kernels discovered from the runtime `--operator-file`.
- In `msprof` mode, a failed benchmark case does not stop later cases from running; the generated perf file keeps successful cases and records `# latency-error-case-*` comments for failed ones.

In `msprof` mode, kernel-miss cases still write `latency-<case-id>: NA`, but also include raw op statistics plus a `# latency-error-case-*` explanation.
In `msprof-simulator` and `torch-npu-profiler` modes, the extracted simulation data (`extracted_bin_data/`) is copied to the directory specified by `--extract-dest-dir` if provided; otherwise it defaults to the parent directory of `--bench-file`. Pass `--extract-dest-dir baseline/` during baseline preparation, and pass `--extract-dest-dir opt-round-N/` to keep each round's data separate.
- In `msprof-simulator` and `torch-npu-profiler` mode, `--simulator-case-idx <N>` specifies which benchmark case to run for simulation. Use this when a prior optimization round has identified a specific case as the focus.

Examples:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py --bench-mode torch-npu-profiler
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file baseline/<operator>.py --bench-mode torch-npu-profiler --extract-dest-dir baseline
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode torch-npu-profiler --extract-dest-dir opt-round-N/
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode torch-npu-profiler --simulator-case-idx 3 --extract-dest-dir opt-round-N/
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
