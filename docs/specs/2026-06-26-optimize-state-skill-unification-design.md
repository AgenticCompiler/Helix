# Optimize State Skill Unification Design

## Summary

- Replace the three common optimize workflow skills
  - `ascend-npu-optimize-start-round`
  - `ascend-npu-optimize-submit-round`
  - `ascend-npu-optimize-submit-baseline`
  with one common skill: `ascend-npu-optimize-state`.
- Move the temporary optimize workflow-state helper out of `triton-npu-optimize` and into the new common skill.
- Keep the agent-facing subcommand names `submit-baseline`, `start-round`, and `submit-round`.
- Give the new skill one public CLI entrypoint: `python3 scripts/cli.py <subcommand> ...`.
- Extend `src/helix/skill_loader.py` so runtime code can load nested skill scripts by skill-relative path such as `state_manage/workflow`.

## Goals

- Make optimize state management a single common workflow contract shared by Triton and TileLang.
- Keep baseline gating, round gating, and temporary workflow-phase state owned by one skill.
- Preserve the current workflow semantics while removing duplicated skill boundaries.
- Let runtime code load the new structured skill scripts directly instead of relying on a top-level compatibility wrapper.
- Keep the optimize CLI thin and keep workflow logic inside skills.

## Non-Goals

- Do not redesign optimize baseline preparation. That remains owned by `ascend-npu-prepare-optimize-baseline`.
- Do not redesign the `triton-npu-optimize` or `tilelang-npu-optimize` optimization loop itself.
- Do not redesign the existing optimize workflow actions beyond renaming round completion to `submit-round`.
- Do not move skill-side state-management logic into `src/helix/`.
- Do not preserve compatibility shims for the three deleted skill names.

## User-Visible Semantics

- Agents should see one common optimize state-management skill: `ascend-npu-optimize-state`.
- The new skill should document three workflow actions:
  - formally accept `baseline/` through `submit-baseline`
  - formally open one `opt-round-N/` through `start-round`
  - formally accept one completed round through `submit-round`
- The public command examples should become:

```bash
python3 scripts/cli.py submit-baseline --baseline-dir baseline
python3 scripts/cli.py start-round --round-dir opt-round-1
python3 scripts/cli.py submit-round --round-dir opt-round-1
```

- Open-ended optimization work remains owned by `triton-npu-optimize` and `tilelang-npu-optimize`.
- Baseline preparation remains owned by `ascend-npu-prepare-optimize-baseline`.
- The workflow meaning of `submit-baseline`, `start-round`, and `submit-round` should stay the same; only the owning skill and script layout change.

## Problem

The current optimize workflow state is split across four places:

- `ascend-npu-optimize-start-round`
- `ascend-npu-optimize-submit-round`
- `ascend-npu-optimize-submit-baseline`
- `triton-npu-optimize/scripts/optimize_workflow_state.py`

This creates four concrete problems:

1. One conceptual workflow contract is spread across multiple skills.
2. Shared runtime state ownership currently lives under the Triton optimize skill even though the contract is common.
3. The current layout makes the relationship between durable artifact checks and temporary workflow-phase state harder to understand.
4. Runtime code can only load top-level `scripts/<name>.py` files, which blocks a more structured skill-side script layout.

## Proposed Skill Layout

Create one new skill:

```text
skills/common/ascend-npu-optimize-state/
  SKILL.md
  references/
    baseline-contract.json
    round-contract.json
  scripts/
    cli.py
    baseline/
      contract.py
      check.py
    round/
      contract.py
      check.py
      kernel_continuity.py
      local_optimum.py
    state_manage/
      submit_baseline.py
      start_round.py
      submit_round.py
      workflow.py
    shared/
      cli.py
      json_io.py
      models.py
      paths.py
      results.py
      round_naming.py
```

Delete these directories entirely:

- `skills/common/ascend-npu-optimize-start-round`
- `skills/common/ascend-npu-optimize-submit-round`
- `skills/common/ascend-npu-optimize-submit-baseline`

## Ownership Boundaries

The new skill should separate three concerns clearly.

### `baseline/`

Owns durable baseline artifact validation helpers:

- reading `baseline/state.json`
- checking required baseline artifact files
- emitting baseline pass/fail results

This layer is about durable optimize artifacts, not temporary runner state or CLI ownership.

### `round/`

Owns durable round artifact validation helpers:

- reading `opt-round-N/round-state.json`
- checking required round-local files
- running kernel continuity and local-optimum checks
- emitting round pass/fail results

This layer is also about durable optimize artifacts, not temporary runner state or CLI ownership.

### `state_manage/`

Owns temporary runner-managed optimize workflow state and the CLI-facing workflow entrypoints:

- reading and validating `.helix/state.json`
- enforcing phase transitions
- recording baseline acceptance and active round transitions
- rendering phase summaries
- archiving completed round timing data
- exposing `submit-baseline`, `start-round`, and `submit-round` subcommand handlers

This layer must not be confused with durable files like `baseline/state.json` or `opt-round-N/round-state.json`.

`submit-baseline`, `start-round`, and `submit-round` belong to this layer. These entrypoint modules should read runner-managed workflow state when needed, delegate durable artifact validation to `baseline/check.py` or `round/check.py`, and emit the user-facing JSON payloads for the agent.

`start-round` is the one intentional bridge across the temporary and durable layers: it must read `.helix/state.json` to enforce workflow phase rules, then open the next durable `opt-round-N/` target that the worker is about to edit.

## CLI Design

`scripts/cli.py` should be the only documented command entrypoint for the skill.

It should:

- build the top-level parser
- register `submit-baseline`, `start-round`, and `submit-round`
- dispatch to the structured modules
- print JSON payloads in the same style as today's scripts

It should not own baseline contract parsing, round contract parsing, or workflow-phase mutation logic directly.

The `submit-baseline`, `start-round`, and `submit-round` subcommands should dispatch to `state_manage/submit_baseline.py`, `state_manage/start_round.py`, and `state_manage/submit_round.py` respectively, while the reusable state-transition functions remain in `state_manage/workflow.py`.

## Contract References

The new skill should rename the machine-readable contract files to make their ownership explicit:

- `references/baseline-contract.json`
- `references/round-contract.json`

This avoids carrying two unrelated `contract.json` files inside one merged skill and makes the source of truth clearer for both humans and tests.

## Runtime And Loader Integration

### Runtime Bridge Target

After this refactor, runtime code should stop loading workflow-state helpers from `triton-npu-optimize`.

Instead:

- `src/helix/optimize/workflow_state.py` should load `ascend-npu-optimize-state` script `state_manage/workflow`
- `src/helix/optimize/checks.py` should load:
  - `ascend-npu-optimize-state` script `baseline/check`
  - `ascend-npu-optimize-state` script `round/check`
- `src/helix/optimize/skill_contract.py` should point to the same new structured modules

No top-level `scripts/optimize_workflow_state.py` compatibility wrapper should be kept.

### `skill_loader` Extension

Keep the existing loader API names:

- `skill_script_path(skill_name, script_name)`
- `load_skill_script_module(skill_name, script_name)`

Extend `script_name` semantics so it may be a skill-relative script path without `.py`.

Examples:

- `run-command`
- `baseline/check`
- `round/check`
- `state_manage/workflow`

Resolution rule:

- resolve `scripts_root = <skill>/scripts`
- resolve `<scripts_root>/<script_name>.py`

The loader must reject invalid traversal inputs such as absolute paths or `..` segments.

### Import Behavior For Nested Skill Scripts

Supporting nested script paths requires one loader change beyond path resolution.

Today the loader prepends `path.parent` to `sys.path`. That works for top-level scripts, but it is not enough for nested modules such as `state_manage/workflow.py`.

After this refactor, the loader should prepend the skill `scripts/` root to `sys.path` while executing the module. This preserves current top-level behavior and also allows nested structured imports such as:

```python
from baseline.check import check_baseline
from round.check import check_round
from state_manage.workflow import start_round
```

Because loaded skill scripts are still executed as standalone modules, nested scripts should use skill-local absolute imports rooted at `scripts/`. They should not rely on package-relative imports such as `from .workflow import ...`.

Because the loader is using `scripts/`-rooted absolute imports rather than package-relative imports, the structured directories do not need `__init__.py` files. Keep them as plain directories to avoid implying package semantics that the loader design is not using.

### Loader Cache And Synthetic Module Names

`load_skill_script_module()` should continue caching loaded modules.

For nested script names, the synthetic module name should incorporate the skill-relative path in a collision-safe way, for example by replacing `/` with `_`. This keeps `baseline/check` and `round/check` distinct in the module cache.

## Skill And Prompt Migration

The repository should migrate all optimize workflow references to the new common skill.

### Catalog And Staging

- `src/helix/skill_catalog.py`
  - remove the three old common skill entries
  - add one entry for `ascend-npu-optimize-state`
- `src/helix/skill_staging.py`
  - replace staged references to the three old skills with the new common skill
  - update `LOG_CHECK`, `LOG_CHECK_BATCH`, and `OPTIMIZE`

### Optimize Workflow Callers

Update optimize runtime and prompt surfaces so they name only `ascend-npu-optimize-state`.

This includes:

- `src/helix/optimize/contract.py`
- `src/helix/optimize/prompts.py`
- `src/helix/optimize/memory_file.py`
- `src/helix/optimize/execution.py`
- `src/helix/cli.py`
- `src/helix/log_check/log_check_launcher.py`
- guard and trace wording that currently cites the deleted skill names

`src/helix/optimize/contract.py` must stop hardcoding the deleted skill directories and instead read:

- `skills/common/ascend-npu-optimize-state/references/baseline-contract.json`
- `skills/common/ascend-npu-optimize-state/references/round-contract.json`

### Triton And TileLang Skill Contracts

Update both optimize skills so they refer to one sibling skill:

- `skills/triton/triton-npu-optimize/SKILL.md`
- `skills/tilelang/tilelang-npu-optimize/SKILL.md`
- `skills/common/ascend-npu-prepare-optimize-baseline/SKILL.md`

The guidance should name `ascend-npu-optimize-state` while continuing to use the existing subcommand names.

### Contract Artifact Sync

`skills/triton/triton-npu-optimize/script/update-artifacts.py` must be updated to read the renamed contract files under `ascend-npu-optimize-state/references/`.

After updating those paths, rerun:

```bash
python3 skills/triton/triton-npu-optimize/script/update-artifacts.py
```

This is required by the repository contract-update rule in `AGENTS.md`.

## Implementation Notes

### Module Mapping

The current scripts should migrate into the new structure roughly as follows:

- `optimize_start_round.py`
  - move into `state_manage/start_round.py`
  - keep the CLI-facing `start-round` payload construction there
  - delegate legal phase transitions to `state_manage/workflow.py`
- `optimize_submit_baseline.py` and `optimize_submit_baseline_contract.py`
  - move into `baseline/check.py` and `baseline/contract.py`
- `optimize_submit_round.py` and `optimize_submit_round_contract.py`
  - move into `round/check.py` and `round/contract.py`
- `kernel_continuity_check.py`
  - move into `round/kernel_continuity.py`
- `local_optimum_check.py`
  - move into `round/local_optimum.py`
- `triton-npu-optimize/scripts/optimize_workflow_state.py`
  - move into `state_manage/workflow.py`

### Cross-Module Reuse

Skill-side shared helpers such as JSON loading, path normalization, and CLI result rendering should move into `scripts/shared/` instead of being recopied across baseline and round modules.

### Durable Versus Temporary State

The merged skill must preserve the distinction between:

- durable artifact contract files under `baseline/` and `opt-round-N/`
- temporary runner-managed phase state under `.helix/state.json`

The merged directory layout should make that distinction easier to see, not blur it.

## Verification

### Test Migration

Update all tests that hardcode the deleted skill names so they point to `ascend-npu-optimize-state` instead.

At minimum, update these files:

- `tests/test_cli.py`
- `tests/test_codex_pretooluse_guard.py`
- `tests/test_generation_contracts.py`
- `tests/test_models.py`
- `tests/test_opencode_hook_guard.py`
- `tests/test_optimize_baseline.py`
- `tests/test_optimize_checks.py`
- `tests/test_optimize_contract.py`
- `tests/test_optimize_round_contract.py`
- `tests/test_optimize_runtime.py`
- `tests/test_run_skill_loader.py`
- `tests/test_skill_command_script.py`
- `tests/test_skills.py`

Also update runtime-bridge tests that load the workflow helper by old location, including:

- `tests/test_optimize_workflow_state.py`

These test updates should change both skill-name string literals and any direct script-path expectations that still point at the deleted skill directories.

### Commands And Checks

Update and run at least these verification layers:

- loader tests for nested skill-relative script paths
- runtime bridge tests for:
  - baseline checks
  - round checks
  - workflow-state loading
- skill staging and catalog tests so only `ascend-npu-optimize-state` is required
- skill command-script tests so the public entrypoint is `python3 scripts/cli.py ...`
- prompt and trace tests that mention optimize workflow skills

Because this refactor changes skill-side Python under `skills/*/scripts/`, run strict per-file pyright checks for every modified skill script via:

```bash
bash scripts/run-skill-script-pyright.sh <skill-script-path>
```

Also run the repository verification commands:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

## Rollout Shape

Implement this as one hard cutover:

1. add the new `ascend-npu-optimize-state` skill and its structured scripts while the three old skills still exist
2. extend `skill_loader` for nested script paths and add loader coverage for the new path form
3. migrate runtime bridges, prompt surfaces, contract-path readers, help strings, and skill documents so all live references point at the new skill while the old skills still remain present
4. update the affected tests so they expect the new single-skill ownership and keep the repository passing at this stage
5. update `skills/triton/triton-npu-optimize/script/update-artifacts.py` and rerun it so `references/artifacts.md` stays synchronized with the renamed contract files
6. delete the three old skill directories only after all references and tests have been moved
7. rerun the full verification suite after deletion

Do not keep compatibility wrappers for the deleted skill names.
