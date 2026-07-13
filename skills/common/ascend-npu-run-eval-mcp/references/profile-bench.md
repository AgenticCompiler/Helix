# `profile-bench`

Use the `profile-bench` MCP tool to profile a generated benchmark and summarize the copied-back `PROF_*` output.

Rules:

- Always pass both `bench_file` and `operator_file`.
- `profile-bench` always uses `torch_npu.profiler`. It does not accept `bench_mode` and does not support `perf-counter` (which produces no profiler traces).

Argument examples:

- `profile-bench(bench_file="bench_<operator>.py", operator_file="<operator>.py")`
- `profile-bench(bench_file="bench_<operator>.py", operator_file="<operator>.py", case_id="<id>", remote="user@host:2222")`
- `profile-bench(bench_file="bench_<operator>.py", operator_file="opt_<operator>.py", case_id="<id>", remote="user@host:2222", remote_workdir="/tmp/helix", keep_remote_workdir=True)`

Use the `profile-report` MCP tool to re-summarize an existing `PROF_*` directory without re-running the benchmark.
