---
name: operator-eval
description: Execute and evaluate generated operator artifacts. Use when you need to run generated test cases, run generated benchmark cases, profile benchmark harnesses, or compare result and performance artifacts, including during optimization workflows.
---

# Run Test And Bench

Use the bundled helper script in this skill:

```bash
python3 ./scripts/run-command.py <subcommand> ...
```

Use the operator-eval skill to execute generated test files, benchmark files, profiling runs, and comparison flows.
Do not reread the Python files under `./scripts/` unless you need to debug, patch, or verify command behavior. In normal use, call the helper script directly and avoid spending context on code that is not needed for the current run.

## Run Test

Run a generated test with:

```bash
python3 ./scripts/run-command.py run-test --test-file test_<operator>.py --operator-file <operator>.py
python3 ./scripts/run-command.py run-test --test-file differential_test_<operator>.py --operator-file opt_<operator>.py
```

Notes:
- Always pass both `--test-file` and `--operator-file`.
- If `--test-mode` is omitted, the command reads `# test-mode: ...` from the test file.
- Use `--test-mode standalone` or `--test-mode differential` only when you need to override the embedded metadata.

Examples:

```bash
python3 ./scripts/run-command.py run-test --test-file test_<operator>.py --operator-file <operator>.py --test-mode standalone
python3 ./scripts/run-command.py run-test --test-file differential_test_<operator>.py --operator-file opt_<operator>.py --test-mode differential
```

Remote examples:

```bash
python3 ./scripts/run-command.py run-test --test-file test_<operator>.py --operator-file <operator>.py --remote user@host:2222
python3 ./scripts/run-command.py run-test --test-file differential_test_<operator>.py --operator-file opt_<operator>.py --remote user@host:2222 --remote-workdir /tmp/triton-agent
```

## Compare Differential Results

If the test mode is `differential`, compare the archived result payloads after `run-test` succeeds:

```bash
python3 ./scripts/run-command.py compare-result --oracle-result <oracle_result.pt> --new-result <new_result.pt>
python3 ./scripts/run-command.py compare-result --oracle-result <oracle_result.pt> --new-result <new_result.pt> --compare-level balanced
```

Remote example:

```bash
python3 ./scripts/run-command.py compare-result --oracle-result <oracle_result.pt> --new-result <new_result.pt> --remote user@host:2222 --remote-workdir /tmp/triton-agent
```

## Run Bench

Run a generated benchmark with:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py
```

Notes:
- Always pass both `--bench-file` and `--operator-file`.
- If `--bench-mode` is omitted, the command reads `# bench-mode: ...` from the benchmark file.
- Use `--bench-mode standalone` or `--bench-mode msprof` only when you need to override the embedded metadata.

Examples:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py --bench-mode standalone
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode msprof
```

Remote examples:

```bash
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py --remote user@host:2222
python3 ./scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode msprof --remote user@host:2222 --remote-workdir /tmp/triton-agent
```

## Profile Bench

Profile a generated benchmark and summarize the copied-back `PROF_*` output with:

```bash
python3 ./scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file <operator>.py
python3 ./scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --target-op MatMul
```

Notes:
- Always pass both `--bench-file` and `--operator-file`.
- If `--bench-mode` is omitted, the command reads `# bench-mode: ...` from the benchmark file.
- In `standalone` mode, do not pass `--bench`; the helper profiles the plain `--operator-file` benchmark run.
- In `msprof` mode, the helper first queries `--num-bench`, then profiles one selected `--bench <N>` case, defaulting to case `1` when you omit `--bench`.

Remote examples:

```bash
python3 ./scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file <operator>.py --remote user@host:2222
python3 ./scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench 2 --remote user@host:2222 --remote-workdir /tmp/triton-agent --keep-remote-workdir
```
