# Remote Profiler Support

## Summary

- Add remote-aware benchmark profiling through the unified `skills/operator-eval/scripts/run-command.py` helper layer.
- Keep the code agent local for `optimize --remote ...`; profiler execution itself should reuse the same remote workspace runtime used by `run-test` and `run-bench`.
- Make `ascend-npu-operator-profiler` describe different argument rules for `standalone` and `msprof` benchmark modes.

## User-visible behavior

- Generated benchmark profiling should prefer:
  - `python3 skills/operator-eval/scripts/run-command.py profile-bench --bench-file <bench> --operator-file <operator>`
- `profile-bench` accepts:
  - `--bench-file`
  - `--operator-file`
  - optional `--bench-mode`
  - optional `--bench <N>`
  - optional `--target-op`
  - optional `--remote user@host[:port]`
  - optional `--remote-workdir <path>`
  - optional `--keep-remote-workdir`
  - optional `--verbose`
- If `--bench-mode` is omitted, the helper should read `# bench-mode: ...` from the benchmark metadata header.
- In `standalone` mode, profiling wraps:
  - `msprof python3 bench_<op>.py --operator-file <operator-file>`
- In `standalone` mode, `--bench` is invalid because the benchmark contract does not use benchmark-case selection.
- In `msprof` mode, profiling first queries:
  - `python3 bench_<op>.py --num-bench`
- In `msprof` mode, profiling then runs one selected case:
  - `msprof op --kernel-name=<kernel> python3 bench_<op>.py --operator-file <operator-file> --bench <N>`
- In `msprof` mode, omitting `--bench` should default to case `1`.
- Remote profiling should copy the resulting `PROF_*` directory back beside the local operator file before rendering the summary.
- If `--keep-remote-workdir` is set, the helper should also print the retained remote workspace path.

## Design notes

- Keep remote SSH, `scp`, temporary workspace creation, and cleanup in the shared operator-eval runtime so profiler execution follows the same semantics as existing remote test and benchmark helpers.
- Add a dedicated `profile_runner.py` module instead of folding profiler behavior into `bench_runner.py`; `run-bench` remains a perf-oriented flow, while `profile-bench` is for evidence collection and summary.
- Reuse existing benchmark metadata parsing so benchmark mode and `# kernel:` stay sourced from the generated harness header.
- Summaries should still be rendered by `skills/ascend-npu-operator-profiler/scripts/profile_summary.py` so profiler reporting stays centralized.

## Documentation updates

- Update `skills/ascend-npu-operator-profiler/SKILL.md` to prefer `profile-bench` and explain mode-specific argument rules.
- Update `skills/operator-eval/SKILL.md` so the helper is discoverable beside `run-test` and `run-bench`.
- Update `skills/optimize/SKILL.md` so remote-aware optimize runs keep passing profiler execution through the same remote settings.
- Update `README.md` and `AGENTS.md` so the repository-level contract matches the helper behavior.
