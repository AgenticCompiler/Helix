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

- In `standalone` mode, pass `--case-id <id>` and do not pass `--bench`; the helper profiles one declared standalone case through the standalone runtime helper.
- In `msprof` mode, the helper first queries `--num-bench`, then profiles one selected `--bench <N>` case, defaulting to case `1` when you omit `--bench`.
- In `msprof` mode, do not pass kernel filter arguments; `msprof` profiles the selected benchmark case and the report can be filtered later with `--target-op`.

Remote examples:

```bash
python3 ./scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file <operator>.py --case-id <id> --remote user@host:2222
python3 ./scripts/run-command.py profile-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench 2 --remote user@host:2222 --remote-workdir /tmp/triton-agent --keep-remote-workdir
```

Use `profile-report` to re-summarize an existing `PROF_*` directory without re-running the benchmark:

```bash
python3 ./scripts/run-command.py profile-report --profile-dir PROF_000001_.../ --target-op MatMul
python3 ./scripts/run-command.py profile-report --profile-dir PROF_000001_.../ --target-op MatMul --format json
```
