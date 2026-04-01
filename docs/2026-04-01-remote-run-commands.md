# Remote Execution For Run Commands

## Summary

- Add optional SSH-backed execution to `run-test`, `run-bench`, and `compare-result`.
- Use `--remote user@host[:port]` to select a remote machine.
- Use `--remote-workdir <path>` to place each run under a created subdirectory in a fixed remote root; otherwise create a temporary remote directory.

## Behavior

- `run-test` copies the operator and generated test harness to the remote workspace, runs the harness there, and streams stdout/stderr locally.
- `run-test` and `run-bench` no longer accept `--agent`; they execute directly either locally or through SSH.
- `run-test` reads `# test-mode: ...` from harness metadata when `--test-mode` is not passed.
- Differential `run-test` copies the remote `TEST_RESULT.pt` payload back to the local machine and archives it with the existing `<operator-stem>_result.pt` naming rule.
- `run-bench` copies the operator and benchmark harness to the remote workspace, runs the benchmark remotely, and keeps perf parsing and local artifact writing on the local machine.
- `run-bench` reads `# bench-mode: ...` from harness metadata when `--bench-mode` is not passed.
- `compare-result` copies both result payloads and a standalone compare helper script to the remote workspace, then runs the comparison there.
- `--verbose` prints the concrete `ssh` and `scp` commands used for remote execution so SSH issues are easier to debug.

## Scope

- Keep local execution as the default when `--remote` is absent.
- Do not change `compare-perf` in this iteration.
