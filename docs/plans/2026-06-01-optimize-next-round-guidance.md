# Optimize Next-Round Guidance Implementation Plan

**Goal:** Strengthen post-`check-round` continue guidance so optimize workers know the next round name and must reflect on the next bottleneck and evidence level before editing code.

**Architecture:** Keep the behavior prompt-driven. Update the optimize-check success summary for immediate in-session guidance, the CLI follow-up summary for cross-invocation handoff, and the shared continue prompt for durable next-round discipline. Cover the new behavior with targeted unit tests.

## Task 1: Document the behavior change

- Files:
  - Create: `docs/specs/2026-06-01-optimize-next-round-guidance-design.md`
  - Create: `docs/plans/2026-06-01-optimize-next-round-guidance.md`

- [ ] Record the approved behavior scope and touched files.

## Task 2: Write failing tests for the new guidance

- Files:
  - Modify: `tests/test_optimize_checks.py`
  - Modify: `tests/test_cli.py`
  - Modify: `tests/test_optimize_runtime.py`

- [ ] Add a test that asserts `check-round` pass summaries with unsatisfied `min_rounds` include the next round name and pre-round reflection language.
- [ ] Extend the continue-guidance tests so they also forbid parallel round optimization and parameter-only tuning sweeps.
- [ ] Add a test that asserts CLI follow-up summaries include `Next round: opt-round-N+1`.
- [ ] Add a test that asserts continue prompts mention pre-round reflection and profiling / IR / compiler-source choice points.
- [ ] Add a test that asserts the structured optimize-check result carries `next_option` directly.
- [ ] Run the targeted tests and confirm they fail for the expected missing text.

## Task 3: Implement the prompt-chain updates

- Files:
  - Modify: `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round_contract.py`
  - Modify: `src/helix/optimize/execution.py`
  - Modify: `src/helix/optimize/prompts.py`

- [ ] Update optimize-check pass summaries for the “must continue” path.
- [ ] Update CLI follow-up summaries to expose `Next round`.
- [ ] Update shared continue guidance to require a deliberate pre-round reflection before editing code.

## Task 4: Verify the finished behavior

- Files:
  - Modify: `tests/test_optimize_checks.py`
  - Modify: `tests/test_cli.py`
  - Modify: `tests/test_optimize_runtime.py`
  - Modify: `tests/test_skill_command_script.py`

- [ ] Run the targeted tests again and confirm they pass.
- [ ] If prompt text changes require assertion adjustments elsewhere, update only the tests that intentionally cover these prompts.
