# Optimize-State API And Unified Helix Skill Bridge Design

## Summary

Deliver two coordinated refactors:

1. Give `ascend-npu-optimize-state` one explicit Python facade,
   `scripts/optimize_state_api.py`.
2. Introduce `src/helix/skill_bridges/` as the single Helix-side integration
   layer for both `ascend-npu-optimize-state` and the previously refactored
   `ascend-npu-run-eval` skill.

The first bridge set is:

```text
ascend-npu-optimize-state/scripts/optimize_state_api.py
  -> helix.skill_bridges.optimize_state
  -> helix.optimize.*, helix.status.*, helix.verify.*, helix.optimize_upload.*

ascend-npu-run-eval/scripts/*_api.py
  -> helix.skill_bridges.run_{test,bench,profile,probe,simulator,comparison,remote,cli}
  -> helix.eval.*, helix.commands.comparison, helix.remote.*, helix.status.*, helix.verify.*
```

The skill facade defines the stable, deliberate set of operations that Helix
may consume. The Helix bridge is the only Helix business module allowed to load
those facades through `helix.skills.loader`. All other `src/helix` modules
import typed bridge functions and Helix-owned models instead of loading a
script, holding a script module object, or reaching into a script's
implementation file.

This is a boundary correction, not a workflow-contract redesign. The staged
optimize-state and run-eval CLIs, their artifact paths, JSON/output behavior,
validation, exceptions, and decisions remain unchanged.

## Current State And Problem

`ascend-npu-optimize-state` already has well-separated internal implementation
files, but no single Python interface intended for Helix:

- `baseline/check.py` owns baseline state loading, inspection, gating, and
  checking.
- `round/check.py` owns round state, artifact inspection, check results,
  artifact naming, round enumeration, and cleanup helpers.
- `state_manage/state_machine.py` owns transient workflow-state transitions.
- `state_manage/{submit_baseline,start_round,set_current_round_state,submit_round}.py`
  own CLI parsing, JSON presentation, and subcommand-specific guidance.

Helix currently bypasses that structure in several ways:

- `helix.optimize.checks` loads `baseline/check` and `round/check` independently
  for each operation.
- `helix.optimize.skill_contract` exposes raw `ModuleType` objects; consumers
  in `models`, `naming`, `round_contract`, cleanup modules, and upload
  collection then read attributes from those modules directly.
- `helix.optimize.execution` loads `round/check` directly for terminal-round
  enumeration.
- `helix.optimize.baseline` and `helix.optimize.workflow_state` delegate to
  `hook_runtime` modules that independently load optimize-state implementation
  files.

The result is that the stable contract is implicit, type information leaks
through `Any` and `ModuleType`, and a script-file rename requires finding every
direct consumer. It also prevents applying the successful `run_*_api.py`
convention consistently to non-eval skills.

The completed run-eval API/worker refactor made this broader issue visible:
the skill has proper `run_test_api.py`, `run_bench_api.py`,
`run_profile_api.py`, and `run_probe_api.py` facades, but
`helix.eval.runners` still loads them itself. Comparison, perf-artifact,
simulator, remote-environment, and remote-spec consumers also still load
run-eval implementation scripts directly. The unified bridge task moves all
of these existing Helix integrations now, rather than leaving run-eval as a
later follow-up.

## Goals And Non-Goals

### Goals

- Add one stable facade, `optimize_state_api.py`, to the optimize-state skill.
- Add `src/helix/skill_bridges/` as the explicit skill-to-Helix integration
  layer; use the plural package name because it holds focused bridges by skill
  contract, not a single catch-all bridge per skill.
- Migrate every direct optimize-state dependency in `src/helix` to
  `helix.skill_bridges.optimize_state`.
- Migrate every existing `ascend-npu-run-eval` script load and script-path
  lookup in `src/helix` to focused `helix.skill_bridges.run_eval_*` modules.
- Replace raw script-module attributes and script-owned type aliases in Helix
  with explicit bridge functions and Helix-owned normalized models.
- Preserve lazy loading, user-visible errors, and all existing optimize-state
  semantics.
- Enforce the finished rule: all Helix business access to a skill script,
  including loading a script module or resolving its executable script path,
  belongs in `helix.skill_bridges`.

### Non-Goals

- Do not move skill implementation into `src/helix`, or permit skill scripts
  to import `helix`.
- Do not change the staged optimize-state command line (`submit-baseline`,
  `start-round`, `set-current-round-state`, `submit-round`) or add a Helix CLI
  command for it.
- Do not change baseline/round JSON schemas, status values, timing files,
  cleanup policy, or artifact naming.
- Do not migrate skills other than `ascend-npu-optimize-state` and
  `ascend-npu-run-eval` in this change.
- Do not make `hook_runtime` import `helix`. It is an independent runtime
  package and must remain free of a reverse Helix dependency.

## Target Architecture

```text
                           staged skill scripts root
                                      |
                                      v
                         optimize_state_api.py
                  /-----------|-----------\\
                  v                       v
            baseline/check.py        round/check.py
                  |                       |
                  +----------+------------+
                             v
                  state_manage/state_machine.py

                                      |
                  load_skill_script_module("ascend-npu-optimize-state",
                                           "optimize_state_api")
                                      |
                                      v
                    helix.skill_bridges.optimize_state
                                      |
     +--------------------------------+--------------------------------+
     v                                v                                v
helix.optimize.*                helix.status/verify             helix.optimize_upload
```

The facade is a skill-local import boundary. The bridge is a Helix-local type
and ownership boundary. Neither is a second implementation of baseline/round
rules.

### Skill Facade: `optimize_state_api.py`

The new root-level script imports selected concrete functions and data classes
from the existing implementation modules and defines an explicit `__all__`.
It has no `main()`, no argument parser, no filesystem policy of its own, and no
Helix imports. The existing `cli.py` remains the only aggregate CLI entrypoint.

The public surface is intentionally the exact set of current Helix consumers.
It does not export skill data classes because the bridge normalizes those into
Helix-owned models, and it does not export helpers solely to support a
Helix-side forwarding wrapper.

| Area | Facade exports |
| --- | --- |
| Baseline read/check | `load_baseline_state`, `inspect_baseline_artifacts`, `baseline_gate_issues`, `check_baseline` |
| Round read/check | `load_round_state`, `inspect_round_artifacts`, `check_round`, `iter_terminal_round_directories`, `count_terminal_round_directories`, `count_completed_round_directories`, `best_completed_round_geomean_speedup` |
| Artifact resolution | `resolve_round_operator_file`, `resolve_round_perf_file` |
| Cleanup policy | `ordinary_optimize_pt_cleanup_mode`, `cleanup_pt_file`, `cleanup_dir_pt_files`, `cleanup_workspace_profile_artifacts` |
| Workflow state | `load_state`, `bootstrap_state`, `mark_baseline_passed`, `render_phase_summary` |

The facade does not export CLI `main` or `build_parser` functions. Helix owns
neither agent-facing JSON rendering nor the subcommand command line; it calls
the underlying domain operations already used by those CLI modules.

An export is added only when a real Helix consumer needs it. Internal helpers,
implementation-only dependency functions, and operations used only through a
redundant forwarding module stay private to their existing modules.

### Helix Bridge Package

Create:

```text
src/helix/skill_bridges/
  __init__.py
  optimize_state.py
  run_eval_test.py
  run_eval_bench.py
  run_eval_profile.py
  run_eval_probe.py
  run_eval_simulator.py
  run_eval_comparison.py
  run_eval_remote.py
  run_eval_cli.py
```

Each bridge module has the same ownership model: it is a narrow typed adapter
for one skill contract, not a generic script registry or a new aggregate
runner. In particular,
`helix.skill_bridges.optimize_state` has these responsibilities:

- lazily and cache-safely load only `optimize_state_api`;
- declare a protocol for exactly the facade surface it consumes;
- expose typed, named wrappers for each Helix operation;
- normalize foreign script result objects into Helix-owned domain models;
- validate return shapes at the boundary and raise clear `TypeError` or
  `ValueError` for an invalid facade implementation;
- never expose `ModuleType`, `getattr`, or a generic "load skill module" API
  to its callers.

The bridge does not contain baseline validation, round validation, naming, or
state-machine logic. Those remain owned by the skill scripts.

Run-eval uses eight focused bridge modules so its prior runner aggregation does
not reappear under a different name:

| Bridge module | Skill facade loaded | Public responsibility |
| --- | --- | --- |
| `run_eval_test.py` | existing `run_test_api.py` | Test metadata, local/remote execution, case payloads, and remote differential comparison |
| `run_eval_bench.py` | existing `run_bench_api.py` | Benchmark metadata, kernel resolution, mode normalization, and local/remote execution |
| `run_eval_profile.py` | existing `run_profile_api.py` | Local/remote profile execution |
| `run_eval_probe.py` | existing `run_probe_api.py` | Local/remote probe execution |
| `run_eval_simulator.py` | new `run_simulator_api.py` | Local simulator execution |
| `run_eval_comparison.py` | new `compare_result_api.py`, `perf_artifacts_api.py` | Result-payload comparison/lookup plus perf comparison/parsing |
| `run_eval_remote.py` | new `remote_execution_env_api.py`, `run_runtime_api.py` | Remote environment controls and remote-spec parsing |
| `run_eval_cli.py` | no skill facade; it is the dedicated script-path bridge | Managed MCP run-eval CLI script path |

Each module owns the corresponding lazy facade loader, protocol, argument
forwarding, and result-shape boundary. It does not import another bridge
module merely to re-export it. `helix.eval.runners` remains the Helix-domain
adapter that converts run-eval mapping results into `AgentResult` and exposes
the existing Helix public wrapper signatures; it calls the specific feature
bridge rather than loading any skill script.

Execution bridge protocols and wrappers use the concrete signatures consumed
by `helix.eval.runners`; they do not use variadic `*args` or `**kwargs` as a
dynamic-module escape hatch. This preserves static checking of command inputs
while keeping the dynamic loading boundary inside the bridge.

The four execution bridges must load existing `run_test_api`, `run_bench_api`,
`run_profile_api`, and `run_probe_api` facades, never their local/remote API,
worker, or execution modules. The remaining run-eval concerns need small,
root-level facades before their bridges land:

| New facade | Selected exports | Replaces direct load of |
| --- | --- | --- |
| `compare_result_api.py` | comparison functions and payload lookup | `compare_result.py` |
| `perf_artifacts_api.py` | comparison and parse functions consumed by Helix | `perf_artifacts.py` |
| `run_simulator_api.py` | `run_local_simulator` | `simulator_runner.py` |
| `remote_execution_env_api.py` | environment helpers | `remote_execution_env.py` |
| `run_runtime_api.py` | `parse_remote_spec` | `run_runtime.py` |

These are API facades only. They must not move comparison, perf parsing,
simulator, SSH, or CLI implementation logic from their current owners.

`helix.skills.loader` remains the low-level mechanism and may continue to
export generic loader and path functions for tests and the bridge package.
Apart from that low-level package and modules under `helix.skill_bridges`, no
`src/helix` business module may import it. This change migrates the complete
current optimize-state and run-eval inventory and adds a repository test for
the rule.

There is no broad `from helix.skill_bridges import *` facade. Consumers import
the specific bridge module, for example:

```python
from helix.skill_bridges import optimize_state

result = optimize_state.check_round(round_dir, current_round=round_number)
```

### Helix Models And Normalization

The current `helix.optimize.models` assigns `BaselineState`, `RoundState`, and
related classes directly from dynamically loaded script modules. That leaks a
script implementation type through the Helix domain model and forces pyright
suppression.

During migration, define equivalent frozen dataclasses in
`helix.optimize.models` with the current fields, field order, defaults, and
value semantics. `OptimizeCheckResult` already has a Helix-owned dataclass and
remains there. The bridge converts facade values by attribute/field name to
these Helix models. Existing Helix consumers keep importing the same names from
`helix.optimize.models`; their value behavior stays unchanged while their type
identity becomes stable and independent of dynamic script loading.

The bridge returns `Path` instances and tuple fields as-is only after type
validation. It must not silently coerce malformed script values other than the
existing result normalizations (`str`/numeric return forms) already accepted by
`helix.optimize.checks`.

## Migration Scope

### First Migration: Optimize-State Consumers in `src/helix`

| Existing location | Current dependency | Target dependency |
| --- | --- | --- |
| `helix.optimize.checks` | Pure check/counter forwarding | Delete; execution and orchestration call the named bridge functions directly |
| `helix.optimize.skill_contract` | Raw `ModuleType` access | Delete after callers use the bridge; do not leave a shim |
| `helix.optimize.models` | Dynamic aliases to script dataclasses | Helix dataclasses; bridge normalization |
| `helix.optimize.baseline` | `hook_runtime.optimize.baseline` facade | Retain only the Helix-owned `baseline_dir`; direct consumers call bridge baseline functions |
| `helix.optimize.round_contract` | Pure round contract forwarding | Delete; status and verification call the bridge directly |
| `helix.optimize.naming` | Raw round module attributes | Retain batch-discovery logic only; delete unused round forwarding helpers |
| `helix.optimize.pt_cleanup` | Raw round module attributes | Retain workspace/run-test cleanup orchestration; call bridge for leaf operations |
| `helix.optimize.profile_cleanup` | Raw round module attributes | Retain only the workspace cleanup operation; call bridge directly |
| `helix.optimize.execution` | Direct round module for latest terminal round | Bridge terminal-round iterator |
| `helix.optimize_upload.collector` | Raw round module attributes | Bridge artifact resolvers |
| downstream status/verify/orchestration | Existing `helix.optimize.*` forwarding functions | Call the specific bridge directly |

`helix.optimize.workflow_state` needs a deliberate treatment. Its public API
stays unchanged. Its Helix wrapper should own its orchestration logic and call
the optimize-state bridge for state-machine operations, instead of merely
delegating to a `hook_runtime` module that loads skill scripts. The existing
`hook_runtime.optimize.workflow_state` remains an independent runtime path for
its own callers until a separate runtime-boundary migration is designed.

This avoids introducing `hook_runtime -> helix` while ensuring that the Helix
path follows the new bridge rule.

### Run-Eval Consumer Migration

The following inventory is part of this change, alongside optimize-state. No
Helix consumer of run-eval retains a direct loader or path lookup afterward:

| Existing location | Current direct access | Target bridge API |
| --- | --- | --- |
| `helix.eval.runners` | `run_test_api`, `run_bench_api`, `run_profile_api`, `run_probe_api`, `simulator_runner` | `run_eval_test`, `run_eval_bench`, `run_eval_profile`, `run_eval_probe`, `run_eval_simulator` |
| `helix.commands.comparison` | `compare_result`, `perf_artifacts` | `run_eval_comparison` |
| `helix.status.core` | `perf_artifacts` | `run_eval_comparison` perf parse wrappers |
| `helix.verify.core` | `perf_artifacts` | `run_eval_comparison` perf parse wrappers |
| `helix.remote.env` | `remote_execution_env` | `run_eval_remote` environment wrappers |
| `helix.remote.ssh_preflight` | `run_runtime.parse_remote_spec` | `run_eval_remote.parse_remote_spec` |
| `helix.eval.mcp_server` | direct run-eval `cli.py` path lookup | `run_eval_cli.cli_script_path()` |

The existing bridge-facing public wrappers in these modules retain their
signatures, result values, error handling, and command behavior. Only their
script integration dependency moves.

Each bridge has a narrow typed surface; there is deliberately no generic
`load_script(skill_name, script_name)` bridge. Once this inventory and the
optimize-state table both migrate, the global static test permits loader imports
and script-path lookups only in `helix.skills.loader` and
`helix.skill_bridges/**`.

## Compatibility Rules

- `python skills/common/ascend-npu-optimize-state/scripts/cli.py` and all four
  subcommands retain their parser, JSON payload, stdout/stderr, and exit-code
  behavior.
- The new facade must load when only the skill's `scripts/` root is on
  `sys.path`; it may not rely on repository-root imports or `helix`.
- Retain only Helix business functions that add path, workspace, trigger, or
  optional-state semantics. Delete pure forwarding modules and functions
  instead of treating them as a public compatibility surface.
- Script loading stays lazy. Importing unrelated Helix commands must not load
  optimize-state scripts merely because the bridge package exists.
- `OptimizeCheckResult` status/kind/summary/issue handling retains the current
  accepted result forms. Baseline and round inspection objects retain their
  current dataclass fields and comparable equality behavior.
- The reference JSON contracts remain loaded from the skill directory; this
  refactor does not duplicate contract field lists in Helix.
- No compatibility module remains for `helix.optimize.skill_contract` after
  callers migrate, because its raw-module interface is exactly the boundary
  being removed.

## Implementation Sequence And Rollback

1. Add `optimize_state_api.py` and the four missing run-eval facades, each with
   an explicit `__all__` and facade test. Do not change Helix callers yet.
2. Add `helix.skill_bridges.optimize_state` plus the eight focused
   `run_eval_*` bridge modules, with lazy loaders, narrow protocols, typed
   wrappers, and normalization tests.
3. Move `BaselineState`, `BaselineArtifactsInspection`, `RoundState`, and
   `RoundArtifactsInspection` into `helix.optimize.models`; prove bridge
   results preserve their fields and equality behavior.
4. Migrate direct optimize-state consumers in small groups. Delete pure
   forwarding modules (`checks`, `round_contract`) and leaf forwarding
   functions after callers use the bridge directly. Delete
   `helix.optimize.skill_contract` after its final caller moves, without
   leaving a shim.
5. Migrate `helix.eval.runners`, comparison, status, verification, remote
   environment/preflight, and MCP CLI-path access to their focused run-eval
   bridge wrappers.
6. Add the global static boundary test that rejects direct loader imports and
   skill-script path lookups outside `helix.skills.loader` and
   `helix.skill_bridges`. Update tests to patch bridge functions rather than
   raw script module loaders.

Each step is independently reversible: a caller can temporarily return to its
previous implementation module without changing CLI protocols or workspace
artifacts. No rollback requires a schema migration.

## Validation

### Targeted Tests

- `optimize_state_api.py` and every new run-eval facade export exactly their
  approved surface, import with the staged-script loader, and do not import
  `helix`.
- The optimize-state bridge loads `optimize_state_api`, never `baseline/check`,
  `round/check`, or `state_manage/state_machine` directly. Each run-eval
  bridge loads only its named `*_api.py` facade, never a worker or execution
  module.
- Bridge tests cover baseline/round result normalization, malformed facade
  values, counters, path resolvers, cleanup helpers, and workflow state calls.
- Existing baseline, round-contract, workflow-state, cleanup, optimize
  execution, status, verification, upload, eval runner, comparison, remote
  environment/preflight, and MCP tests continue through the bridge with
  unchanged observable results. Tests assert that deleted pure forwarding
  modules cannot be imported.
- A static import test permits skill-script module loads and skill-script path
  lookups only in `helix.skills.loader` and `helix.skill_bridges/**`; it also
  confirms optimize-state and run-eval scripts do not import `helix`.
- Direct calls to the four optimize-state CLI subcommands retain their JSON and
  return-code behavior.

### Quality Gates

Run after implementation:

```bash
bash scripts/run-skill-script-pyright.sh \
  skills/common/ascend-npu-optimize-state/scripts/optimize_state_api.py
bash scripts/run-skill-script-pyright.sh \
  skills/common/ascend-npu-run-eval/scripts/compare_result_api.py \
  skills/common/ascend-npu-run-eval/scripts/perf_artifacts_api.py \
  skills/common/ascend-npu-run-eval/scripts/run_simulator_api.py \
  skills/common/ascend-npu-run-eval/scripts/remote_execution_env_api.py \
  skills/common/ascend-npu-run-eval/scripts/run_runtime_api.py
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
git diff --check
```

No R154 execution is required for this refactor: the run-eval bridge calls the
same already-validated facades and does not alter worker code, staging, SSH,
or remote execution. The focused bridge and full regression suites verify
unchanged delegation. Any later change to a run-eval worker, staging, or remote
API still requires the server validation workflow.

## Open Follow-Up

The new `helix.skill_bridges` rule intentionally applies only to `src/helix`.
`hook_runtime` currently has independent optimize-state and run-test loaders so
it can stay usable without importing Helix. A later design should decide
whether `hook_runtime` receives its own runtime bridge package or whether its
skill-facing logic moves behind a shared lower-level boundary. That decision
must preserve the no-reverse-dependency rule and is not folded into this
migration.
