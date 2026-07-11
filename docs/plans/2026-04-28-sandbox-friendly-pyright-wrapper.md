# Sandbox-Friendly Pyright Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-local wrapper script for strict skill-script `pyright` checks that works inside the sandbox by default, and update `AGENTS.md` to require using it.

**Architecture:** Keep the behavior outside the CLI by adding a small shell script under `scripts/`. The script owns the writable `UV_CACHE_DIR` setup and temporary strict `pyright` project generation, while `AGENTS.md` becomes the stable project-level rule that points contributors to the wrapper.

**Tech Stack:** Bash, `uv`, `pyright`, Markdown docs, Python `unittest`

---

### Task 1: Lock the contract with a failing test

**Files:**
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add a failing contract test for the wrapper script and `AGENTS.md` guidance**

```python
def test_skill_script_pyright_wrapper_is_documented_in_agents(self) -> None:
    agents = _read("AGENTS.md")
    wrapper = _read("scripts/run-skill-script-pyright.sh")
    self.assertIn("scripts/run-skill-script-pyright.sh", agents)
    self.assertIn("UV_CACHE_DIR", wrapper)
```

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_skill_script_pyright_wrapper_is_documented_in_agents -v`
Expected: FAIL because the wrapper script and guidance do not exist yet.

### Task 2: Implement the wrapper and update stable guidance

**Files:**
- Create: `scripts/run-skill-script-pyright.sh`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add the wrapper script**

```bash
#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/helix-uv-cache}"
```

- [ ] **Step 2: Make the script generate a temporary strict `pyright` config and run from repo root**

```bash
uv run pyright --project "$config_path" "${targets[@]}"
```

- [ ] **Step 3: Update `AGENTS.md` so skill-script strict checks prefer the wrapper**

```md
- When modifying Python files under `skills/*/scripts/`, always run the additional file-scoped strict check via `scripts/run-skill-script-pyright.sh`.
```

### Task 3: Verify the contract and runtime behavior

**Files:**
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Run the targeted contract test**

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_skill_script_pyright_wrapper_is_documented_in_agents -v`
Expected: PASS.

- [ ] **Step 2: Run the wrapper against the updated bench runner skill script**

Run: `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runner.py`
Expected: `0 errors`.
