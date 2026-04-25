## Summary

- During `optimize`, write a temporary workspace `AGENTS.md` that keeps the optimization workflow visible to the code agent throughout a long run.
- If the workspace already has an `AGENTS.md`, back it up before writing the temporary optimize guidance and restore it after the run.
- When the selected backend is Claude, write the temporary guidance into `CLAUDE.md` instead of `AGENTS.md` so the backend sees its native top-level instruction file.
- The temporary optimize guidance must explicitly forbid replacing the Triton NPU operator path with a direct PyTorch operator or module implementation.
- The temporary optimize guidance must also require multiple generated correctness-test cases and benchmark cases when the optimize flow needs to regenerate harnesses.
- The temporary optimize guidance should also steer the code agent toward the staged optimize, profiler, and IR-analysis skills so it can reuse existing optimization guidance and investigation tools.

## User-Visible Behavior

- `optimize` injects a short run-specific `AGENTS.md` into the operator workspace before launching the agent.
- `optimize --agent claude` injects the same run-specific guidance into `CLAUDE.md` instead.
- The injected guidance should include the actual test mode and benchmark mode chosen for the current optimize run.
- The injected guidance should tell the code agent to keep the task scoped to optimizing the existing NPU Triton operator implementation, not to delete or bypass the Triton operator call path and replace it with direct PyTorch operators or modules.
- The injected guidance should also tell the code agent that regenerated test and benchmark harnesses must contain multiple cases rather than a single case.
- The injected guidance should recommend consulting the staged optimize skill first, then using the staged profiler and IR-analysis skills when performance evidence or compiler-lowering evidence is needed.
- If no workspace `AGENTS.md` exists, the optimize-specific file is removed after the run.
- If a workspace `AGENTS.md` already exists, it is restored after the run and the temporary optimize file is removed.
- The same backup and restore semantics apply to `CLAUDE.md` for Claude optimize runs.
- Verbose mode should show the backup, temporary write, removal, and restore steps.

## Implementation Notes

- Keep the temporary guidance concise and focused on optimization invariants.
- Limit this behavior to the `optimize` command.
- Select the temporary guidance filename from the backend name at preparation time instead of hard-coding `AGENTS.md`.
- Use a dedicated manager so preparation and cleanup remain symmetric and testable.
