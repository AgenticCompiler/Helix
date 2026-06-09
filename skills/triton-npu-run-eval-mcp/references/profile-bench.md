# `profile-bench`

Use the `profile-bench` MCP tool to profile a generated benchmark and summarize the copied-back `PROF_*` output.

Rules:

- Always pass both `bench_file` and `operator_file`.
- If `bench_mode` is omitted, the tool reads `# bench-mode: ...` from the benchmark file.

Mode notes:

- In both modes, pass `case_id` and do not pass numeric case selectors.
- In `standalone` mode, the helper profiles one declared benchmark case through the shared benchmark runtime helper.
- In `msprof` mode, the helper resolves the selected benchmark case by `case_id`, wraps shared runtime-helper case execution in `msprof`, and defaults to the sole declared case only when the benchmark declares exactly one case.
- In `msprof` mode, do not pass `kernel_name`; `msprof` profiles the selected benchmark case and the report can be filtered later with `target_op`.

Argument examples:

- `profile-bench(bench_file="bench_<operator>.py", operator_file="<operator>.py")`
- `profile-bench(bench_file="bench_<operator>.py", operator_file="<operator>.py", case_id="<id>", remote="user@host:2222")`
- `profile-bench(bench_file="bench_<operator>.py", operator_file="opt_<operator>.py", case_id="<id>", remote="user@host:2222", remote_workdir="/tmp/triton-agent", keep_remote_workdir=True)`

Use the `profile-report` MCP tool to re-summarize an existing `PROF_*` directory without re-running the benchmark.
