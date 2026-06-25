# Remote Context For Agent-Driven Commands

## Summary

- Add `--remote user@host[:port]` and optional `--remote-workdir <path>` to the agent-driven commands `gen-test`, `gen-bench`, and `optimize`.
- Keep the code agent running locally.
- Pass the remote requirement through prompt context so the selected skill knows it must use remote-aware repository commands during validation.

## User-Visible Behavior

- `gen-test`, `gen-bench`, and `optimize` now accept the same remote target syntax already used by `run-test`, `run-bench`, and `compare-result`.
- These commands do not execute the code agent on the remote machine.
- Instead, the prompt explicitly tells the agent:
  - which remote target to use
  - whether a fixed remote root was requested
  - that validation commands must include the same remote flags

## Skill Contract Updates

- `skills/triton-npu-gen-test/SKILL.md` should show both local and remote `run-test` validation examples.
- `skills/triton-npu-gen-bench/SKILL.md` should show both local and remote `run-bench` validation examples.
- `skills/triton/triton-npu-optimize/SKILL.md` should state that every generated and validation command must carry the same remote flags when the outer request is remote.

## Scope

- Do not move remote execution of tests or benchmarks back into the agent-backed CLI commands.
- Do not change the existing local default behavior.
- Do not add remote support to `compare-perf` in this change.
