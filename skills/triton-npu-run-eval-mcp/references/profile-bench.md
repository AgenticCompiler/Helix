# `profile-bench`

Use the `profile-bench` MCP tool to profile a generated benchmark and summarize the copied-back `PROF_*` output.

Rules:

- Always pass both `bench_file` and `operator_file`.
- If `bench_mode` is omitted, the tool reads `# bench-mode: ...` from the benchmark file.

Mode notes:

- In `standalone` mode, pass `case_id` and do not pass `bench`; the helper profiles one declared standalone case through the standalone runtime helper.
- In `msprof` mode, the helper first queries the available benchmark count, then profiles one selected `bench` case, defaulting to case `1` when you omit it.
- In `msprof` mode, do not pass `kernel_name`; `msprof` profiles the selected benchmark case and the report can be filtered later with `target_op`.

Argument examples:

- `profile-bench(bench_file="bench_<operator>.py", operator_file="<operator>.py")`
- `profile-bench(bench_file="bench_<operator>.py", operator_file="<operator>.py", case_id="<id>", remote="user@host:2222")`
- `profile-bench(bench_file="bench_<operator>.py", operator_file="opt_<operator>.py", bench=2, remote="user@host:2222", remote_workdir="/tmp/triton-agent", keep_remote_workdir=True)`

Use the `profile-report` MCP tool to re-summarize an existing `PROF_*` directory without re-running the benchmark.
