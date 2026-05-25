# Optimize PT Cleanup Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `check-baseline` always preserve archived PT files, make ordinary optimize cleanup preserve them by default, and add `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES` to re-enable ordinary optimize PT cleanup without changing `--reset-optimize`.

**Architecture:** Keep PT cleanup ownership inside the existing optimize-check skill contract and the optimize runtime bridge. Remove baseline-side mutation entirely, gate round and end-of-run PT cleanup through one skill-owned environment-variable helper, and leave `reset_optimize_workspace()` unchanged so destructive fresh resets keep their current semantics.

**Tech Stack:** Python 3, `unittest`, `os.environ`, existing optimize skill bridge, repo-local `uv` commands, file-scoped strict pyright wrapper

---

### Task 1: Lock The New Baseline And Round Semantics With Failing Tests

**Files:**
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_optimize_checks.py`

- [ ] **Step 1: Change the baseline script test to the new contract**

Update `tests/test_skill_command_script.py` so the existing baseline-check scenario now expects `baseline/test_result.pt` to remain present after `check-baseline` succeeds.

- [ ] **Step 2: Add a failing round-check test for the default preserve behavior**

Add one focused test in `tests/test_optimize_checks.py` that writes `opt-round-1/test_result.pt`, calls `optimize_checks.check_round(round_dir)`, asserts the result still passes, and asserts the PT file still exists when `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES` is unset.

- [ ] **Step 3: Add a failing round-check test for the opt-in delete behavior**

Add one focused test in `tests/test_optimize_checks.py` that uses `patch.dict(os.environ, {"TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES": "1"}, clear=False)`, calls `optimize_checks.check_round(round_dir)`, asserts a passing result, and asserts the PT file is deleted.

- [ ] **Step 4: Run the focused tests and confirm RED**

Run:

```bash
uv run python -m unittest tests.test_skill_command_script tests.test_optimize_checks -v
```

Expected: failures showing that baseline and/or round cleanup still delete PT files under the old implementation.

### Task 2: Implement The Skill-Owned PT Cleanup Gate

**Files:**
- Modify: `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py`
- Modify: `skills/triton-npu-optimize-check/scripts/optimize_check.py`
- Modify: `src/triton_agent/optimize/pt_cleanup.py`

- [ ] **Step 1: Add the environment-variable helper in the skill contract**

In `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py`, add:

- a module-level constant for `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES`
- a small helper that returns `True` only for case-insensitive values `1`, `true`, `yes`, or `on`
- default `False` for unset or any other value

Keep the helper local to the skill-side script so runtime can reuse it through the bridge without creating a reverse import from `skills/` into `src/`.

- [ ] **Step 2: Remove baseline PT deletion entirely**

Update `check_baseline()` in `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py` so a passing baseline check returns `_build_result(kind="baseline", decision="pass", issues=())` directly, without calling `cleanup_dir_pt_files()` for either `baseline/` or the workspace root.

- [ ] **Step 3: Gate round PT deletion behind the new helper**

Update `check_round()` in `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py` so it:

- preserves the existing validation flow
- only calls `cleanup_dir_pt_files(round_dir)` when the new helper returns `True`
- keeps the current cleanup message text unchanged when deletion is enabled
- returns the normal no-issues passing result when deletion is disabled

- [ ] **Step 4: Re-export the helper through the bridge module**

Update `skills/triton-npu-optimize-check/scripts/optimize_check.py` to import and expose the new helper in `__all__` so `src/triton_agent/optimize/pt_cleanup.py` can reuse the same policy through `optimize_check_module()`.

- [ ] **Step 5: Gate end-of-run runtime cleanup through the same helper**

Update `src/triton_agent/optimize/pt_cleanup.py` so `cleanup_workspace_pt_files(workdir)` first checks the bridged helper and returns `[]` immediately when ordinary PT cleanup is disabled. Keep the existing root-directory and `opt-round-*` scan behavior unchanged when the helper returns `True`.

- [ ] **Step 6: Run the focused tests and the strict skill-script pyright checks**

Run:

```bash
uv run python -m unittest tests.test_skill_command_script tests.test_optimize_checks -v
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-check/scripts/optimize_check_contract.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-check/scripts/optimize_check.py
```

Expected: the focused tests pass, and both strict pyright checks succeed.

### Task 3: Add Runtime And Reset Coverage For The New Cleanup Boundary

**Files:**
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Add a failing runtime test for the default preserve behavior**

Add a focused test that creates:

- one workspace-root PT artifact such as `kernel_result.pt`
- one round PT artifact such as `opt-round-1/test_result.pt`

Call `cleanup_workspace_pt_files(workdir)` with the environment variable unset and assert:

- the returned cleaned list is empty
- both PT files still exist

- [ ] **Step 2: Add a failing runtime test for the opt-in delete behavior**

Add a second focused test that sets `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES=1`, calls `cleanup_workspace_pt_files(workdir)`, and asserts:

- the returned cleaned list includes both the root artifact name and the round-prefixed artifact name
- both PT files are removed

- [ ] **Step 3: Add reset coverage proving reset stays destructive**

Add one focused test for `reset_optimize_workspace()` that writes a workspace-root `*_result.pt`, sets `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES` to a falsey or unrelated value, runs reset, and asserts the workspace-root PT artifact is still deleted.

- [ ] **Step 4: Run the runtime-focused tests and confirm GREEN**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime -v
```

Expected: the new runtime and reset coverage passes.

### Task 4: Document The New Environment Variable And Finish Verification

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `README.md`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add the variable to CLI environment help**

Update `src/triton_agent/cli.py` so the top-level environment-variable help includes `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES` with wording that makes these semantics explicit:

- it only affects ordinary optimize PT cleanup
- default behavior preserves PT files
- `--reset-optimize` is not controlled by it

- [ ] **Step 2: Update README optimize/runtime docs**

Update `README.md` to describe:

- the new variable under runtime environment variables
- default PT preservation during normal optimize runs
- `check-baseline` never deletes PT files
- `--reset-optimize` still deletes known optimize PT artifacts

- [ ] **Step 3: Add CLI help coverage for the new variable**

Update `tests/test_cli.py` so the existing environment-variable help assertions now include `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES`.

- [ ] **Step 4: Run the focused docs/help tests**

Run:

```bash
uv run python -m unittest tests.test_cli -v
```

Expected: the CLI help test covering environment variables passes.

### Task 5: Final Verification

**Files:**
- Modify: none

- [ ] **Step 1: Run the combined focused regression suite**

Run:

```bash
uv run python -m unittest \
  tests.test_skill_command_script \
  tests.test_optimize_checks \
  tests.test_optimize_runtime \
  tests.test_cli \
  -v
```

- [ ] **Step 2: Re-run the strict skill-script pyright checks**

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-check/scripts/optimize_check_contract.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-check/scripts/optimize_check.py
```

- [ ] **Step 3: Run the repository-standard verification commands**

Run:

```bash
uv run pyright
uv run python -m unittest discover -s tests -v
```

- [ ] **Step 4: If any verification fails, fix it and rerun before claiming completion**

Keep the final report grounded in the fresh command output, including any residual failures or intentionally skipped checks.
