# OpenCode General Subagent And Continuous Round Serialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clarify that continuous optimize rounds must run sequentially and stage an OpenCode config that disables the built-in `general` task subagent.

**Architecture:** Update optimize prompt builders to restate the serial round contract in both initial and resume prompt paths. Keep OpenCode-specific workspace config staging inside the OpenCode backend so the change applies to every OpenCode invocation without affecting other backends.

**Tech Stack:** Python, unittest, JSON workspace config staging

---

### Task 1: Lock The New Prompt Contract With Tests

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

- [ ] Add assertions that the continuous optimize prompt says rounds must complete sequentially and that subagents must not execute multiple rounds in parallel.
- [ ] Add assertions that the continuous resume prompt repeats the same serial-round restriction.
- [ ] Run the targeted prompt tests and confirm they fail before implementation.

### Task 2: Lock OpenCode Workspace Config Staging With Tests

**Files:**
- Modify: `tests/test_opencode_runner.py`
- Test: `tests/test_opencode_runner.py`

- [ ] Add a runner test that inspects `.opencode/opencode.json` during process launch and verifies `agent.build.permission.task.general = "deny"` and `agent.plan.permission.task.general = "deny"`.
- [ ] Add a runner test that verifies the staged config file is cleaned up after the run.
- [ ] Add a runner test that verifies an existing `.opencode/opencode.json` causes an explicit failure instead of overwrite.
- [ ] Run the targeted OpenCode tests and confirm they fail before implementation.

### Task 3: Implement Prompt Serialization Guidance

**Files:**
- Modify: `src/helix/optimize/prompts.py`
- Test: `tests/test_cli.py`

- [ ] Add a shared continuous-round serialization rule block used by the initial continuous prompt and the continuous resume prompt path.
- [ ] Keep the wording explicit that only one round may be active at once and that subagents may not advance multiple rounds in parallel.
- [ ] Re-run the targeted prompt tests and confirm they pass.

### Task 4: Implement OpenCode Workspace Config Staging

**Files:**
- Modify: `src/helix/backends/opencode.py`
- Test: `tests/test_opencode_runner.py`

- [ ] Add a small OpenCode workspace config staging helper that writes `.opencode/opencode.json` with the schema URL and `permission.task.general = "deny"` for the built-in primary agents we launch.
- [ ] Fail explicitly when the config path already exists.
- [ ] Ensure the staged config file is always cleaned up after the run, including failure paths.
- [ ] Re-run the targeted OpenCode tests and confirm they pass.

### Task 5: Verify The Full Change Set

**Files:**
- Verify: `tests/test_cli.py`
- Verify: `tests/test_opencode_runner.py`

- [ ] Run the targeted unittest command that covers both prompt and OpenCode backend changes.
- [ ] If the targeted suite passes, run the repository verification commands needed for confidence proportional to this change.
- [ ] Record any residual risk if broader verification is skipped or blocked.
