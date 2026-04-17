# 2026-04-02 Bugfix Regressions

## Summary

This change bundle fixes confirmed regressions and robustness gaps reported in `docs/bug-reports/2026-04-02-bug-review-status.md` without widening the CLI surface area.

## User-visible behavior

- Streaming process runners treat `stall_timeout_seconds <= 0` as "no stall timeout" instead of an immediate stall condition.
- Optimize supervision keeps using recovery resumes after repeated stalls instead of falling back to a fresh run path.
- Codex diff filtering stops dropping normal indented output that appears after a diff block.
- Remote triton-npu-run-eval commands quote generated shell arguments for filenames and metadata that may contain spaces or shell metacharacters.
- Invalid run-skill result payloads fail with a short actionable error instead of a raw `KeyError`.

## Implementation approach

- Add regression tests first for the confirmed failures in process running, optimize recovery, diff filtering, remote command quoting, and CLI result normalization.
- Keep the fixes local to existing orchestration helpers instead of moving skill behavior into the CLI.
- Update `docs/bug-reports/2026-04-02-bug-review-status.md` to reflect the repaired issues and the items that were triaged as lower-risk or not reproducible bugs.
