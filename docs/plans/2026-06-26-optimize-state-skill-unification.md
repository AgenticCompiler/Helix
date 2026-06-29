# Optimize State Skill Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the optimize start-round, submit-round, and submit-baseline workflow skills into one structured common `ascend-npu-optimize-state` skill, then migrate runtime and tests onto that new ownership model.

**Architecture:** Add one new common skill with a single public `scripts/cli.py` CLI, keep `baseline/` and `round/` focused on durable check logic, place CLI-facing workflow entrypoints under `state_manage/`, and extend `skill_loader` so runtime code can import nested skill scripts directly. Migrate runtime bridges, prompt references, contract readers, and tests to the new skill before deleting the three old skill directories.

**Tech Stack:** Python 3.11, `argparse`, `json`, `pathlib`, `unittest`, `load_skill_script_module`, `uv`, skill-side strict pyright checks

---

## File Map

- Create: `skills/common/ascend-npu-optimize-state/SKILL.md`
- Create: `skills/common/ascend-npu-optimize-state/references/baseline-contract.json`
- Create: `skills/common/ascend-npu-optimize-state/references/round-contract.json`
- Create: `skills/common/ascend-npu-optimize-state/scripts/cli.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/baseline/contract.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/baseline/check.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/round/contract.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/round/check.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/round/kernel_continuity.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/round/local_optimum.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/state_manage/submit_baseline.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/state_manage/workflow.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/state_manage/start_round.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/state_manage/submit_round.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/shared/json_io.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/shared/models.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/shared/paths.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/shared/results.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/shared/round_naming.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/shared/cli.py`
- Modify: `src/triton_agent/skill_loader.py`
- Modify: `src/triton_agent/skill_catalog.py`
- Modify: `src/triton_agent/skill_staging.py`
- Modify: `src/triton_agent/optimize/checks.py`
- Modify: `src/triton_agent/optimize/contract.py`
- Modify: `src/triton_agent/optimize/skill_contract.py`
- Modify: `src/triton_agent/optimize/workflow_state.py`
- Modify: `src/triton_agent/optimize/prompts.py`
- Modify: `src/triton_agent/optimize/memory_file.py`
- Modify: `src/triton_agent/optimize/execution.py`
- Modify: `src/triton_agent/log_check/log_check_launcher.py`
- Modify: `src/triton_agent/backends/claude_trace.py`
- Modify: `src/triton_agent/backends/codex_trace.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `hooks/codex/tool_trace_hook.py`
- Modify: `hooks/opencode/triton-agent-hook-guard.js`
- Modify: `hooks/shared/tool_use_guard_policy.py`
- Modify: `skills/common/ascend-npu-prepare-optimize-baseline/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/tilelang/tilelang-npu-optimize/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/script/update-artifacts.py`
- Modify: `skills/triton/triton-npu-optimize/references/artifacts.md`
- Delete: `skills/common/ascend-npu-optimize-start-round/`
- Delete: `skills/common/ascend-npu-optimize-submit-baseline/`
- Delete: `skills/common/ascend-npu-optimize-submit-round/`
- Modify: `tests/test_run_skill_loader.py`
- Modify: `tests/test_optimize_baseline.py`
- Modify: `tests/test_optimize_round_contract.py`
- Modify: `tests/test_optimize_workflow_state.py`
- Modify: `tests/test_optimize_checks.py`
- Modify: `tests/test_optimize_contract.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_skills.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_codex_pretooluse_guard.py`
- Modify: `tests/test_claude_trace.py`
- Modify: `tests/test_codex_trace.py`
- Modify: `tests/test_opencode_hook_guard.py`
- Modify: `tests/test_models.py`

### Task 1: Extend `skill_loader` for nested skill-relative script paths

**Files:**
- Modify: `tests/test_run_skill_loader.py`
- Modify: `src/triton_agent/skill_loader.py`

- [ ] **Step 1: Add a failing loader-path test for nested scripts**

```python
    def test_skill_script_path_supports_nested_skill_relative_scripts(self) -> None:
        expected = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-state"
            / "scripts"
            / "state_manage"
            / "workflow.py"
        )
        path = skill_script_path("ascend-npu-optimize-state", "state_manage/workflow")
        self.assertEqual(path, expected)
```

- [ ] **Step 2: Add a failing loader test for nested module imports and cache stability**

```python
    def test_load_skill_script_module_supports_nested_skill_relative_scripts(self) -> None:
        first = load_skill_script_module("ascend-npu-optimize-state", "state_manage/workflow")
        second = load_skill_script_module("ascend-npu-optimize-state", "state_manage/workflow")
        self.assertIs(first, second)
        self.assertTrue(hasattr(first, "bootstrap_state"))
```

- [ ] **Step 3: Run the loader tests and verify the new nested-path expectations fail**

Run:

```bash
uv run python -m unittest tests.test_run_skill_loader.RunSkillLoaderTests.test_skill_script_path_supports_nested_skill_relative_scripts tests.test_run_skill_loader.RunSkillLoaderTests.test_load_skill_script_module_supports_nested_skill_relative_scripts -v
```

Expected: FAIL because `skill_script_path()` only resolves top-level scripts and the new skill does not exist yet.

- [ ] **Step 4: Implement nested-path resolution and `scripts/`-root import support**

```python
def skill_script_path(skill_name: str, script_name: str) -> Path:
    relative = Path(script_name + ".py")
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid skill script path: {script_name!r}")
    path = skill_script_root(skill_name) / "scripts" / relative
    if not path.exists():
        raise FileNotFoundError(f"Skill script does not exist: {path}")
    return path


@lru_cache(maxsize=None)
def load_skill_script_module(skill_name: str, script_name: str) -> ModuleType:
    path = skill_script_path(skill_name, script_name)
    module_name = f"skill_{skill_name.replace('-', '_')}_{script_name.replace('/', '_').replace('-', '_')}"
    scripts_root = str(skill_script_root(skill_name) / "scripts")
    ...
```

- [ ] **Step 5: Re-run the focused loader tests**

Run:

```bash
uv run python -m unittest tests.test_run_skill_loader.RunSkillLoaderTests.test_skill_script_path_supports_nested_skill_relative_scripts tests.test_run_skill_loader.RunSkillLoaderTests.test_load_skill_script_module_supports_nested_skill_relative_scripts -v
```

Expected: PASS

### Task 2: Build the new common optimize-state skill behind failing workflow tests

**Files:**
- Modify: `tests/test_optimize_workflow_state.py`
- Create: `skills/common/ascend-npu-optimize-state/...`

- [ ] **Step 1: Retarget workflow-state tests to the new module location**

```python
def load_workflow_state_module():
    return load_skill_script_module("ascend-npu-optimize-state", "state_manage/workflow")
```

- [ ] **Step 2: Add a failing CLI-ownership test for `start-round` through the new skill**

```python
    def test_optimize_state_skill_exposes_start_round_module(self) -> None:
        module = load_skill_script_module("ascend-npu-optimize-state", "state_manage/start_round")
        self.assertTrue(hasattr(module, "build_parser"))
        self.assertTrue(hasattr(module, "main"))
```

- [ ] **Step 3: Run the focused workflow-state tests and verify failure**

Run:

```bash
uv run python -m unittest tests.test_optimize_workflow_state -v
```

Expected: FAIL because the new skill tree and modules do not exist yet.

- [ ] **Step 4: Create the new skill references and shared helpers by copying the current contract content without behavior changes**

```python
# shared/json_io.py
def load_json_object(path: Path, *, display_name: str) -> dict[str, Any]:
    ...


# shared/paths.py
def baseline_dir(workspace: Path) -> Path:
    return workspace / "baseline"
```

- [ ] **Step 5: Implement `state_manage/workflow.py` and `state_manage/start_round.py` using the current workflow-state and start-round behavior**

```python
# state_manage/start_round.py
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start-round")
    start.add_argument("--round-dir", required=True)
    return parser
```

- [ ] **Step 6: Keep `baseline/check.py` and `round/check.py` focused on durable validation helpers, then add `state_manage/submit_baseline.py`, `state_manage/submit_round.py`, and `cli.py` as the CLI-facing workflow entrypoints**

```python
# cli.py
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_baseline_parser(subparsers)
    register_start_round_parser(subparsers)
    register_round_parser(subparsers)
    return parser
```

- [ ] **Step 7: Re-run the focused workflow-state tests**

Run:

```bash
uv run python -m unittest tests.test_optimize_workflow_state -v
```

Expected: PASS

### Task 3: Migrate runtime bridges and contract readers to the new skill

**Files:**
- Modify: `src/triton_agent/optimize/checks.py`
- Modify: `src/triton_agent/optimize/contract.py`
- Modify: `src/triton_agent/optimize/skill_contract.py`
- Modify: `src/triton_agent/optimize/workflow_state.py`
- Modify: `tests/test_optimize_baseline.py`
- Modify: `tests/test_optimize_round_contract.py`
- Modify: `tests/test_optimize_contract.py`

- [ ] **Step 1: Add failing runtime bridge tests that expect the new skill names and script paths**

```python
        module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "baseline/check",
        )
```

- [ ] **Step 2: Run the focused runtime bridge tests and verify failure**

Run:

```bash
uv run python -m unittest tests.test_optimize_baseline tests.test_optimize_round_contract tests.test_optimize_contract -v
```

Expected: FAIL because runtime bridge modules still target the deleted skills.

- [ ] **Step 3: Update runtime bridges to load nested scripts from the new skill**

```python
def _workflow_module():
    return load_skill_script_module("ascend-npu-optimize-state", "state_manage/workflow")
```

- [ ] **Step 4: Update the contract reader to the renamed JSON files**

```python
_BASELINE_CONTRACT_PATH = (
    skills_root()
    / "common"
    / "ascend-npu-optimize-state"
    / "references"
    / "baseline-contract.json"
)
```

- [ ] **Step 5: Re-run the focused runtime bridge tests**

Run:

```bash
uv run python -m unittest tests.test_optimize_baseline tests.test_optimize_round_contract tests.test_optimize_contract -v
```

Expected: PASS

### Task 4: Migrate staging, prompt surfaces, and command-script coverage

**Files:**
- Modify: `src/triton_agent/skill_catalog.py`
- Modify: `src/triton_agent/skill_staging.py`
- Modify: `src/triton_agent/optimize/prompts.py`
- Modify: `src/triton_agent/optimize/memory_file.py`
- Modify: `src/triton_agent/optimize/execution.py`
- Modify: `src/triton_agent/log_check/log_check_launcher.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `skills/common/ascend-npu-prepare-optimize-baseline/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/tilelang/tilelang-npu-optimize/SKILL.md`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_skills.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_codex_pretooluse_guard.py`
- Modify: `tests/test_opencode_hook_guard.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Add failing expectations that stage only `ascend-npu-optimize-state` and invoke `python3 scripts/cli.py ...`**

```python
        self.assertTrue((target / "ascend-npu-optimize-state" / "SKILL.md").exists())
        self.assertFalse((target / "ascend-npu-optimize-submit-round").exists())
```

- [ ] **Step 2: Run the focused staging and command-script tests and verify failure**

Run:

```bash
uv run python -m unittest tests.test_skills tests.test_skill_command_script tests.test_run_skill_loader tests.test_cli -v
```

Expected: FAIL because staging, prompts, and script expectations still mention the old skills.

- [ ] **Step 3: Update catalog, staging rules, optimize prompts, help strings, and skill docs to cite only the new skill**

```python
CommandKind.OPTIMIZE: StageRule(
    directives=(
        "+{language}-npu-optimize",
        "+{language}-npu-optimize-knowledge",
        "+ascend-npu-prepare-optimize-baseline",
        "+ascend-npu-optimize-state",
        ...
    ),
)
```

- [ ] **Step 4: Update command-script tests to call the unified entrypoint**

```python
completed = subprocess.run(
    [sys.executable, str(skill_root / "scripts" / "cli.py"), "submit-baseline", "--baseline-dir", str(baseline_dir)],
    ...
)
```

- [ ] **Step 5: Re-run the focused staging, prompt, and command-script tests**

Run:

```bash
uv run python -m unittest tests.test_skills tests.test_skill_command_script tests.test_run_skill_loader tests.test_cli tests.test_generation_contracts tests.test_codex_pretooluse_guard tests.test_opencode_hook_guard tests.test_models -v
```

Expected: PASS

### Task 5: Sync artifacts, delete old skills, and run full verification

**Files:**
- Modify: `skills/triton/triton-npu-optimize/script/update-artifacts.py`
- Modify: `skills/triton/triton-npu-optimize/references/artifacts.md`
- Delete: `skills/common/ascend-npu-optimize-start-round/`
- Delete: `skills/common/ascend-npu-optimize-submit-baseline/`
- Delete: `skills/common/ascend-npu-optimize-submit-round/`
- Modify: `tests/test_optimize_checks.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Update the artifact-sync script to the renamed contract files**

```python
BASELINE_CONTRACT_PATH = (
    _resolve_skill_source_dir("ascend-npu-optimize-state")
    / "references"
    / "baseline-contract.json"
)
```

- [ ] **Step 2: Re-run artifact sync required by `AGENTS.md`**

Run:

```bash
python3 skills/triton/triton-npu-optimize/script/update-artifacts.py
```

Expected: Prints `skills/triton/triton-npu-optimize/references/artifacts.md`

- [ ] **Step 3: Delete the three old skill directories and update any remaining tests that still mention them**

```text
skills/common/ascend-npu-optimize-start-round
skills/common/ascend-npu-optimize-submit-baseline
skills/common/ascend-npu-optimize-submit-round
```

- [ ] **Step 4: Run strict pyright per modified skill script**

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/cli.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/baseline/check.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/baseline/contract.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/round/check.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/round/contract.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/round/kernel_continuity.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/round/local_optimum.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/state_manage/submit_baseline.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/state_manage/submit_round.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/state_manage/workflow.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/state_manage/start_round.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/shared/json_io.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/shared/models.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/shared/paths.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/shared/results.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/shared/round_naming.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/shared/cli.py
```

Expected: All PASS

- [ ] **Step 5: Run repository verification**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS
