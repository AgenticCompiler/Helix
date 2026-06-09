# Unified Bench Case Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the split standalone/msprof benchmark-file contracts with one shared import-only benchmark module contract, move case selection to `case_id`, and delete obsolete msprof benchmark CLI code paths.

**Architecture:** Benchmark generation will emit one module shape for both modes: metadata plus `build_operator_api(...)`, `build_bench_cases()`, and `build_bench_case_fn(...)`. A shared `bench_runtime.py` helper will own module loading, case validation, case selection, and one-case execution, while `run-bench`, `profile-bench`, and IR capture will choose only the profiling wrapper (`torch_npu.profiler` vs `msprof`). Old `msprof` benchmark-file CLI assumptions, numeric `--bench` selectors, and standalone-only runtime naming will be removed rather than preserved.

**Tech Stack:** Python 3, `unittest`, `pytest`, `argparse`, existing run-eval skill scripts, MCP server tool metadata, `ruff`, `pyright`

---

## File Structure

- `docs/specs/2026-06-09-unified-bench-case-contract-design.md`
  - Approved design and migration boundary.
- `docs/plans/2026-06-09-unified-bench-case-contract.md`
  - This implementation plan.
- `skills/triton-npu-gen-bench/SKILL.md`
  - Generator workflow contract for benchmark creation and validation.
- `skills/triton-npu-gen-bench/references/bench-standalone-spec.md`
  - Must be rewritten to the unified import-only benchmark contract while keeping `# bench-mode: standalone`.
- `skills/triton-npu-gen-bench/references/bench-msprof-spec.md`
  - Must be rewritten to the same import-only benchmark contract while keeping `# bench-mode: msprof`.
- `skills/triton-npu-run-eval/SKILL.md`
  - User-facing run-eval workflow contract.
- `skills/triton-npu-run-eval/references/run-bench.md`
  - Run-bench behavior docs for the unified runtime helper.
- `skills/triton-npu-run-eval/references/profile-bench.md`
  - Profile-bench docs with `case_id`-only case selection.
- `skills/triton-npu-run-eval-mcp/references/run-bench.md`
  - MCP-facing run-bench docs.
- `skills/triton-npu-run-eval-mcp/references/profile-bench.md`
  - MCP-facing profile-bench docs and parameter semantics.
- `skills/triton-npu-profile-operator/SKILL.md`
  - Profiler workflow contract; remove numeric case selection.
- `skills/triton-npu-run-eval/scripts/bench_runtime.py`
  - New shared runtime helper that loads benchmark modules, validates cases, selects `case_id`, and executes one case.
- `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
  - Delete after behavior is moved into `bench_runtime.py`.
- `skills/triton-npu-run-eval/scripts/bench_runner.py`
  - Shared bench-runner facade and runtime support-path staging.
- `skills/triton-npu-run-eval/scripts/bench_runner_deps.py`
  - Shared runner protocol types; rename standalone-specific runtime protocol pieces.
- `skills/triton-npu-run-eval/scripts/bench_runner_standalone.py`
  - Keep standalone profiling logic, but make it consume the shared runtime helper.
- `skills/triton-npu-run-eval/scripts/bench_runner_msprof.py`
  - Replace `--num-bench` / `--bench` benchmark-file CLI execution with shared runtime-helper execution by `case_id`.
- `skills/triton-npu-run-eval/scripts/profile_runner.py`
  - Remove numeric benchmark selection and switch both modes to `case_id`.
- `skills/triton-npu-run-eval/scripts/run-command.py`
  - CLI parser and orchestration for `profile-bench`; remove `--bench`.
- `skills/triton-npu-analyze-ir/scripts/capture_ir.py`
  - Replace `--bench` with `--case-id` and render runtime-helper execution commands for both modes.
- `src/triton_agent/run_eval_mcp_server.py`
  - Remove the numeric `bench` parameter from `profile-bench` tool metadata and argument forwarding.
- `tests/test_generation_contracts.py`
  - Contract docs assertions for benchmark specs and profiler skill docs.
- `tests/test_standalone_bench_runtime.py`
  - Migrate to the shared runtime helper test surface or rename to a neutral runtime test file if the implementation does so.
- `tests/test_bench_runner.py`
  - Run-bench runtime integration tests for local/remote standalone and msprof behavior.
- `tests/test_profile_runner.py`
  - `profile-bench` local/remote tests; replace numeric selectors with `case_id`.
- `tests/test_ascend_operator_ir_analyzer.py`
  - IR capture command-shape tests; remove numeric `--bench`.
- `tests/test_skill_command_script.py`
  - CLI help and argument-forwarding tests for `run-command.py`.
- `tests/test_run_eval_mcp_server_tool_metadata.py`
  - MCP tool schema assertions.
- `tests/test_cli.py`
  - Top-level CLI parser and execution tests.

## Task 1: Lock The New Contract In Tests And Docs First

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_run_eval_mcp_server_tool_metadata.py`
- Modify: `skills/triton-npu-gen-bench/SKILL.md`
- Modify: `skills/triton-npu-gen-bench/references/bench-standalone-spec.md`
- Modify: `skills/triton-npu-gen-bench/references/bench-msprof-spec.md`
- Modify: `skills/triton-npu-run-eval/references/run-bench.md`
- Modify: `skills/triton-npu-run-eval/references/profile-bench.md`
- Modify: `skills/triton-npu-run-eval-mcp/references/run-bench.md`
- Modify: `skills/triton-npu-run-eval-mcp/references/profile-bench.md`
- Modify: `skills/triton-npu-profile-operator/SKILL.md`

- [ ] **Step 1: Rewrite generation-contract assertions for the unified benchmark module shape**

Update `tests/test_generation_contracts.py` so the benchmark-doc assertions require:

```python
self.assertIn("build_operator_api(operator_module)", standalone)
self.assertIn("build_bench_cases()", standalone)
self.assertIn("build_bench_case_fn(operator_api, case)", standalone)
self.assertNotIn("build_standalone_bench_cases(operator_api)", standalone)
self.assertNotIn('parser.add_argument("--operator-file"', standalone)

self.assertIn("build_operator_api(operator_module)", msprof)
self.assertIn("build_bench_cases()", msprof)
self.assertIn("build_bench_case_fn(operator_api, case)", msprof)
self.assertNotIn("If `--bench N` is provided, then `--operator-file` is required.", msprof)
self.assertNotIn("--num-bench", msprof)
```

Also update the profiler-skill assertions so they require `--case-id` for both modes and explicitly reject `"first query \`--num-bench\`"`.

- [ ] **Step 2: Update `run-command.py --help` and MCP metadata tests to reject numeric bench selection**

Change the help and tool-metadata tests to require:

```python
self.assertIn("--case-id", completed.stdout)
self.assertNotIn("--bench", completed.stdout)
```

and:

```python
self.assertIn("case_id", tools["profile-bench"].parameters["properties"])
self.assertNotIn("bench", tools["profile-bench"].parameters["properties"])
```

- [ ] **Step 3: Run the focused contract and metadata tests to confirm they fail**

Run:
`uv run python -m unittest tests.test_generation_contracts tests.test_skill_command_script tests.test_run_eval_mcp_server_tool_metadata -v`

Expected: FAIL because the docs, help output, and MCP metadata still describe `build_standalone_bench_cases`, `--bench`, and `--num-bench`.

- [ ] **Step 4: Rewrite the benchmark and profiling docs to the new contract**

Make these documentation changes:

- `skills/triton-npu-gen-bench/SKILL.md`
  - describe one import-only benchmark shape for both modes
  - replace `build_standalone_bench_cases(operator_api)` with `build_bench_cases()` and `build_bench_case_fn(operator_api, case)`
  - remove the sentence that says msprof keeps an executable benchmark CLI shape
- `bench-standalone-spec.md` and `bench-msprof-spec.md`
  - describe the same hook set
  - keep mode-specific profiling behavior only in the execution sections
  - remove `main()`, `argparse`, `--bench`, and `--num-bench` requirements from the msprof spec
- `profile-bench` docs and profiler skill docs
  - use `--case-id` for both modes
  - remove the `first query --num-bench` flow

- [ ] **Step 5: Re-run the focused contract and metadata tests**

Run:
`uv run python -m unittest tests.test_generation_contracts tests.test_skill_command_script tests.test_run_eval_mcp_server_tool_metadata -v`

Expected: PASS

- [ ] **Step 6: Commit the contract-doc rewrite**

```bash
git add tests/test_generation_contracts.py tests/test_skill_command_script.py tests/test_run_eval_mcp_server_tool_metadata.py skills/triton-npu-gen-bench/SKILL.md skills/triton-npu-gen-bench/references/bench-standalone-spec.md skills/triton-npu-gen-bench/references/bench-msprof-spec.md skills/triton-npu-run-eval/references/run-bench.md skills/triton-npu-run-eval/references/profile-bench.md skills/triton-npu-run-eval-mcp/references/run-bench.md skills/triton-npu-run-eval-mcp/references/profile-bench.md skills/triton-npu-profile-operator/SKILL.md
git commit -m "test: lock unified benchmark contract docs"
```

## Task 2: Add A Shared Benchmark Runtime Helper

**Files:**
- Create: `skills/triton-npu-run-eval/scripts/bench_runtime.py`
- Delete: `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner_deps.py`
- Modify: `tests/test_standalone_bench_runtime.py`

- [ ] **Step 1: Add failing runtime tests for the new hook set**

In `tests/test_standalone_bench_runtime.py` (rename the class or file later only if the implementation truly needs it), add tests that build temporary benchmark modules with:

```python
def build_operator_api(operator_module):
    return operator_module.kernel

def build_bench_cases():
    return [{"id": "case-a", "shape": (16,)}]

def build_bench_case_fn(operator_api, case):
    def _run():
        operator_api(case["shape"])
    return _run
```

Add assertions that the runtime helper:

- loads all cases without executing them eagerly
- rejects duplicate ids
- rejects empty case lists
- rejects non-callable results from `build_bench_case_fn(...)`
- selects one case by `case_id`
- auto-selects a single case when `case_id is None`
- rejects missing `case_id` when multiple cases exist

- [ ] **Step 2: Run the focused runtime tests to confirm failure**

Run:
`uv run python -m unittest tests.test_standalone_bench_runtime -v`

Expected: FAIL because the current runtime expects `build_standalone_bench_cases(operator_api)` and has no `build_bench_cases()` / `build_bench_case_fn(...)` model.

- [ ] **Step 3: Implement `bench_runtime.py` with neutral runtime types and helper CLI**

Create `skills/triton-npu-run-eval/scripts/bench_runtime.py` with:

- a neutral runtime case dataclass such as:

```python
@dataclass(frozen=True)
class BenchCase:
    case_id: str
    fn: Callable[[], object]
    warmup: int
    repeats: int
    case_data: Mapping[str, object]
```

- functions that:
  - load benchmark and operator modules
  - call `build_operator_api(...)`
  - validate `build_bench_cases()` output
  - build one callable with `build_bench_case_fn(...)`
  - provide `load_bench_cases(...)`, `select_bench_case(...)`, `run_one_bench_case(...)`, and `profile_local_bench_case(...)`
- an internal CLI that supports:
  - `list-cases`
  - `run-one`
  - `profile-one`

- [ ] **Step 4: Replace standalone-specific protocol names in `bench_runner_deps.py`**

Update `skills/triton-npu-run-eval/scripts/bench_runner_deps.py` so the runtime protocol becomes neutral, for example:

```python
class BenchRuntimeModule(Protocol):
    def load_bench_cases(...): ...
    def run_local_bench(...): ...
    def runtime_support_paths(...): ...
```

Do not leave `StandaloneRuntimeModule` or `load_standalone_bench_cases(...)` in the protocol layer once the shared runtime exists.

- [ ] **Step 5: Re-run focused runtime tests**

Run:
`uv run python -m unittest tests.test_standalone_bench_runtime -v`

Expected: PASS

- [ ] **Step 6: Run required strict pyright for the new helper**

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runtime.py`

Expected: PASS

- [ ] **Step 7: Commit the shared runtime helper**

```bash
git add skills/triton-npu-run-eval/scripts/bench_runtime.py skills/triton-npu-run-eval/scripts/bench_runner_deps.py tests/test_standalone_bench_runtime.py
git rm skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py
git commit -m "feat: add shared benchmark runtime helper"
```

## Task 3: Rewire Run-Bench To Use Shared Cases In Both Modes

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner_standalone.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner_msprof.py`
- Modify: `tests/test_bench_runner.py`

- [ ] **Step 1: Add failing run-bench tests for msprof `case_id` execution**

In `tests/test_bench_runner.py`, replace or supplement the existing msprof CLI-shape tests with fixtures that define the new hooks and assert that msprof commands now use:

```python
[
    "msprof",
    f"--output={output_dir}",
    sys.executable,
    "bench_runtime.py",
    "run-one",
    "--bench-file",
    "bench_kernel.py",
    "--operator-file",
    "kernel.py",
    "--case-id",
    "case-a",
]
```

Also add assertions that perf labels become `case-a`, `case-b` instead of `case-1`, `case-2`.

- [ ] **Step 2: Run focused bench-runner tests to confirm failure**

Run:
`uv run python -m unittest tests.test_bench_runner -v`

Expected: FAIL because the current msprof runner still calls `bench.py --num-bench` and `bench.py --bench <N>`.

- [ ] **Step 3: Update shared runner facade and support-path staging**

In `skills/triton-npu-run-eval/scripts/bench_runner.py`:

- replace `_standalone_runtime_script_path()` and `_standalone_runtime_support_paths()` with neutral `bench_runtime` equivalents
- update dynamic runtime loading cache names
- keep public `run_local_bench(...)` and `run_remote_bench(...)` entrypoints stable

In `bench_runner_standalone.py`:

- consume `load_bench_cases(...)`
- iterate by declared `case_id`
- stage `bench_runtime.py` support paths instead of `standalone_bench_runtime.py`

In `bench_runner_msprof.py`:

- resolve all cases through the shared runtime helper
- use stable `case_id` labels in perf output and errors
- wrap `bench_runtime.py run-one --case-id ...` in `msprof`
- remove all code that parses `--num-bench` output or loops over numeric case indexes

- [ ] **Step 4: Re-run focused bench-runner tests**

Run:
`uv run python -m unittest tests.test_bench_runner -v`

Expected: PASS

- [ ] **Step 5: Run required strict pyright for bench runner scripts**

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runner.py`

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runner_msprof.py`

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runner_standalone.py`

Expected: PASS

- [ ] **Step 6: Commit the runner rewrite**

```bash
git add skills/triton-npu-run-eval/scripts/bench_runner.py skills/triton-npu-run-eval/scripts/bench_runner_standalone.py skills/triton-npu-run-eval/scripts/bench_runner_msprof.py tests/test_bench_runner.py
git commit -m "feat: unify run-bench case execution"
```

## Task 4: Replace Numeric Profile And IR Selection With `case_id`

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/profile_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`
- Modify: `skills/triton-npu-analyze-ir/scripts/capture_ir.py`
- Modify: `src/triton_agent/run_eval_mcp_server.py`
- Modify: `tests/test_profile_runner.py`
- Modify: `tests/test_ascend_operator_ir_analyzer.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing tests that remove numeric selectors**

Update tests so they require:

- `profile-bench` parser/help exposes `--case-id` and rejects `--bench`
- `profile_runner.py` uses `--case-id` for both modes
- `capture_ir.py` parser accepts `--case-id` and rejects `--bench`
- MCP `profile-bench` tool metadata no longer exposes numeric `bench`

Use command expectations such as:

```python
[
    "msprof",
    sys.executable,
    "bench_runtime.py",
    "run-one",
    "--bench-file",
    "bench_kernel.py",
    "--operator-file",
    "kernel.py",
    "--case-id",
    "case-b",
]
```

- [ ] **Step 2: Run focused profile, IR, and CLI tests to confirm failure**

Run:
`uv run python -m unittest tests.test_profile_runner tests.test_ascend_operator_ir_analyzer tests.test_cli -v`

Expected: FAIL because parsers, command builders, and MCP forwarding still expose numeric `--bench`.

- [ ] **Step 3: Implement `case_id`-only profiling and IR capture**

In `profile_runner.py`:

- remove `bench_case` arguments from local and remote helpers
- resolve selected cases by `case_id`
- for msprof, wrap `bench_runtime.py run-one --case-id ...` in `msprof`
- if no `case_id` is provided, auto-select only when exactly one case exists

In `run-command.py`:

- remove `profile-bench --bench`
- update protocol signatures and argument forwarding

In `capture_ir.py`:

- replace `--bench` with `--case-id`
- route both modes through `bench_runtime.py run-one`

In `src/triton_agent/run_eval_mcp_server.py`:

- remove the numeric `bench` parameter from the `profile-bench` tool
- forward only `case_id`

- [ ] **Step 4: Re-run focused profile, IR, and CLI tests**

Run:
`uv run python -m unittest tests.test_profile_runner tests.test_ascend_operator_ir_analyzer tests.test_cli -v`

Expected: PASS

- [ ] **Step 5: Run required strict pyright for profiling and IR scripts**

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/profile_runner.py`

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-analyze-ir/scripts/capture_ir.py`

Expected: PASS

- [ ] **Step 6: Commit the `case_id` selection rewrite**

```bash
git add skills/triton-npu-run-eval/scripts/profile_runner.py skills/triton-npu-run-eval/scripts/run-command.py skills/triton-npu-analyze-ir/scripts/capture_ir.py src/triton_agent/run_eval_mcp_server.py tests/test_profile_runner.py tests/test_ascend_operator_ir_analyzer.py tests/test_cli.py
git commit -m "feat: switch benchmark profiling to case ids"
```

## Task 5: Remove Stale Fixtures, Rename Runtime Coverage, And Run Full Verification

**Files:**
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_profile_runner.py`
- Modify: `tests/test_ascend_operator_ir_analyzer.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/run_skill_test_utils.py`
- Modify: any remaining code/test references to `standalone_bench_runtime.py`, `build_standalone_bench_cases`, `--num-bench`, or `--bench` under benchmark profiling and capture flows

- [ ] **Step 1: Search for stale benchmark-contract and runtime references**

Run:
`rg -n "standalone_bench_runtime|build_standalone_bench_cases|--num-bench|--bench\\b|case-1|case-2" skills src tests -S`

Expected: only intentional non-benchmark references remain; benchmark runtime, profiling, capture, docs, and tests should no longer depend on the removed contract.

- [ ] **Step 2: Delete or rewrite remaining stale fixtures and utility helpers**

Remove or update:

- fake msprof benchmark files that are just script bodies
- helper loaders that assume `standalone_bench_runtime.py`
- stale test names or comments that describe standalone-only runtime ownership when the helper is now shared

- [ ] **Step 3: Run focused regression suites**

Run:
`uv run python -m unittest tests.test_generation_contracts tests.test_standalone_bench_runtime tests.test_bench_runner tests.test_profile_runner tests.test_ascend_operator_ir_analyzer tests.test_skill_command_script tests.test_run_eval_mcp_server_tool_metadata -v`

Expected: PASS

- [ ] **Step 4: Run repository verification commands**

Run:
`uv run --group dev ruff check`

Run:
`uv run pyright`

Run:
`uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

Expected: PASS

- [ ] **Step 5: Commit the cleanup and verification sweep**

```bash
git add tests/test_skill_command_script.py tests/test_bench_runner.py tests/test_profile_runner.py tests/test_ascend_operator_ir_analyzer.py tests/test_cli.py tests/test_generation_contracts.py tests/run_skill_test_utils.py skills src
git commit -m "refactor: remove obsolete benchmark cli paths"
```
