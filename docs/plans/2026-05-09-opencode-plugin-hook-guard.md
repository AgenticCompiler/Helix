# OpenCode Plugin Hook Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--enable-agent-hooks` support for the OpenCode backend by staging a project-local OpenCode plugin that blocks shell reads outside the workspace and shell reads of `.opencode/skills/*/scripts/**`.

**Architecture:** Reuse the existing `AgentHookManager` lifecycle and add an OpenCode-specific template under `hooks/opencode/`. Keep OpenCode's default `--pure` behavior unchanged, but omit `--pure` when hooks are enabled so OpenCode can load the staged project plugin.

**Tech Stack:** Python 3.12, JavaScript OpenCode plugin, Node.js for focused plugin tests, Python `unittest`, existing backend runner abstractions.

---

## File Map

- Create: `hooks/opencode/triton-agent-hook-guard.js`
- Modify: `src/triton_agent/agent_hooks.py`
- Modify: `src/triton_agent/backends/opencode.py`
- Modify: `tests/test_agent_hooks.py`
- Create: `tests/test_opencode_hook_guard.py`
- Modify: `tests/test_opencode_runner.py`
- Modify: `README.md`

## Task 1: OpenCode Hook Staging

- [ ] Add failing tests in `tests/test_agent_hooks.py` for OpenCode staging, generated policy values, cleanup, existing plugin rejection, and existing owned hook directory rejection.
- [ ] Run `uv run python -m unittest tests.test_agent_hooks -v` and confirm the new OpenCode tests fail.
- [ ] Add OpenCode constants and `_prepare_opencode_hooks()` to `src/triton_agent/agent_hooks.py`.
- [ ] Render `.opencode/triton-agent-hooks/policy.json` with `.opencode/skills/*/scripts/**` in the deny glob.
- [ ] Run `uv run python -m unittest tests.test_agent_hooks -v` and confirm the hook manager tests pass.

## Task 2: OpenCode Plugin Guard

- [ ] Add failing Node-backed plugin tests in `tests/test_opencode_hook_guard.py`.
- [ ] Run `uv run python -m unittest tests.test_opencode_hook_guard -v` and confirm the tests fail because the plugin template does not exist.
- [ ] Create `hooks/opencode/triton-agent-hook-guard.js`.
- [ ] Implement `tool.execute.before` handling for `bash` commands, read-command detection, path extraction, path resolution, glob matching, and policy denial.
- [ ] Run `uv run python -m unittest tests.test_opencode_hook_guard -v` and confirm the plugin tests pass.

## Task 3: OpenCode Runner Command Shape

- [ ] Add failing tests in `tests/test_opencode_runner.py` showing default commands keep `--pure` and hook-enabled commands omit `--pure`.
- [ ] Run `uv run python -m unittest tests.test_opencode_runner -v` and confirm the new hook-enabled expectations fail.
- [ ] Update `OpenCodeRunner.build_command()` to include `--pure` only when `request.enable_agent_hooks` is false.
- [ ] Run `uv run python -m unittest tests.test_opencode_runner -v` and confirm runner tests pass.

## Task 4: Documentation

- [ ] Update `README.md` to rename the optional hook section from Codex-only to agent hook guard.
- [ ] Document Codex staged paths and OpenCode staged paths separately.
- [ ] Explain that OpenCode hook support uses a project plugin and therefore hook-enabled OpenCode launches omit `--pure`.
- [ ] Run `uv run python -m unittest tests.test_agent_hooks tests.test_opencode_hook_guard tests.test_opencode_runner tests.test_codex_pretooluse_guard -v`.

## Task 5: Final Verification

- [ ] Run `uv run --group dev ruff check`.
- [ ] Run `uv run pyright`.
- [ ] Run `uv run python -m unittest tests.test_agent_hooks tests.test_opencode_hook_guard tests.test_opencode_runner tests.test_codex_pretooluse_guard tests.test_backends_base -v`.
- [ ] Inspect `git diff --stat` and `git diff --check`.
