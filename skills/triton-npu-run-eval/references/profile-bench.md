# `profile-bench`

Profile a generated benchmark and summarize the copied-back `PROF_*` output with:

```bash
python3 ./scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file <operator>.py
python3 ./scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --target-op MatMul
```

Rules:

- Always pass both `--bench-file` and `--operator-file`.
- If `--bench-mode` is omitted, the command reads `# bench-mode: ...` from the benchmark file.

Mode notes:

- In both modes, pass `--case-id <id>` and do not pass numeric case selectors.
- In `torch-npu-profiler` mode, the helper profiles one declared benchmark case through the shared benchmark runtime helper.
- In `msprof` mode, the helper resolves the selected benchmark case by `case_id`, wraps shared runtime-helper case execution in `msprof`, and defaults to the sole declared case only when the benchmark declares exactly one case.
- In `msprof` mode, do not pass kernel filter arguments; `msprof` profiles the selected benchmark case and the report can be filtered later with `--target-op`.

Remote examples:

```bash
python3 ./scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file <operator>.py --case-id <id> --remote user@host:2222
python3 ./scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --case-id <id> --remote user@host:2222 --remote-workdir /tmp/triton-agent --keep-remote-workdir
```

Use `profile-report` to re-summarize an existing `PROF_*` directory without re-running the benchmark:

```bash
python3 ./scripts/run-command.py profile-report --profile-dir PROF_000001_.../ --target-op MatMul
python3 ./scripts/run-command.py profile-report --profile-dir PROF_000001_.../ --target-op MatMul --format json
```

If the inline `profile-bench` summary is not enough, rerun `profile-report` on the copied-back directory or inspect the raw files inside that `PROF_*` directory directly.
