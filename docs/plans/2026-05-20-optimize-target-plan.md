# Optimize Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--optimize-target {kernel,operator}` to `optimize` and `optimize-batch`, keep `kernel` as the default, switch optimize prompts and workspace guidance so `operator` mode targets end-to-end operator latency while still requiring a real Triton Ascend NPU computation path, and adapt optimize to record one resolved `effective_metric_source` for compare-perf-based round conclusions.

**Architecture:** Thread one explicit optimize-target field through optimize CLI parsing, optimize run options, request construction, prompt builders, resume handling, and temporary optimize guidance rendering. Then extend optimize round metadata and validation with one recorded `effective_metric_source` field. Keep all existing optimize artifacts and orchestration structure unchanged, make kernel mode use kernel-oriented auto comparison, and make operator mode show both kernel and total-op diagnostics while keeping total-op as the canonical round basis.

**Tech Stack:** Python 3, `argparse`, `unittest`, existing optimize orchestration, prompt, and session-artifact helpers

---

## File Map

- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/optimize.py`
- Modify: `src/helix/models.py`
- Modify: `src/helix/optimize/models.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `src/helix/optimize/prompts.py`
- Modify: `src/helix/prompts.py`
- Modify: `src/helix/backends/base.py`
- Modify: `src/helix/optimize/run_loop.py`
- Modify: `src/helix/commands/comparison.py`
- Modify: `src/helix/optimize/session_artifacts.py`
- Modify: `src/helix/optimize/memory_file.py`
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton-npu-optimize-submit-round/references/contract.json`
- Modify: `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round_contract.py`
- Modify: `skills/triton-npu-run-eval/scripts/perf_artifacts.py`
- Modify: `skills/triton-npu-run-eval/references/compare-perf.md`
- Modify: `README.md`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_comparison_commands.py`
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_checks.py`
- Modify: `tests/test_optimize_round_contract.py`

## Task 1: Add Failing CLI And Request Plumbing Tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write the failing parser tests for `--optimize-target`**

Add CLI parser coverage near the existing optimize option tests in `tests/test_cli.py`:

```python
    def test_optimize_command_defaults_optimize_target_to_kernel(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertEqual(args.optimize_target, "kernel")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_target, "kernel")

    def test_optimize_command_accepts_operator_optimize_target(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize", "-i", "kernel.py", "--optimize-target", "operator"]
        )
        self.assertEqual(args.optimize_target, "operator")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_target, "operator")

    def test_optimize_batch_defaults_optimize_target_to_kernel(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels"])
        self.assertEqual(args.optimize_target, "kernel")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_target, "kernel")

    def test_optimize_batch_accepts_operator_optimize_target(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "kernels", "--optimize-target", "operator"]
        )
        self.assertEqual(args.optimize_target, "operator")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_target, "operator")
```

- [ ] **Step 2: Write the failing model and request-plumbing tests**

Extend the existing `AgentRequest.with_prompt()` preservation test in `tests/test_models.py` so it proves the new field survives prompt replacement:

```python
        request = AgentRequest(
            command_kind=CommandKind.OPTIMIZE,
            input_path=Path("/tmp/op.py"),
            operator_path=Path("/tmp/op.py"),
            output_path=Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            interact=False,
            verbose=False,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            skill_name="triton-npu-optimize",
            prompt="original",
            workdir=Path("/tmp"),
            optimize_target="operator",
        )

        updated = request.with_prompt("updated")

        self.assertEqual(updated.optimize_target, "operator")
```

Add request-construction coverage in `tests/test_optimize_runtime.py`:

```python
    def test_build_optimize_request_defaults_optimize_target_to_kernel(self) -> None:
        options = OptimizeRunOptions(
            agent_name="codex",
            interact=False,
            verbose=False,
            show_output=False,
            remote=None,
            remote_workdir=None,
            min_rounds=None,
            resume_mode="auto",
            reset_optimize=False,
            no_agent_session=False,
            supervise="off",
            output=None,
            test_mode=None,
            bench_mode=None,
            prompt=None,
            optimize_target="kernel",
        )
        request = build_optimize_request(operator, workdir, options)
        self.assertEqual(request.optimize_target, "kernel")

    def test_build_optimize_request_preserves_operator_optimize_target(self) -> None:
        options = OptimizeRunOptions(
            agent_name="codex",
            interact=False,
            verbose=False,
            show_output=False,
            remote=None,
            remote_workdir=None,
            min_rounds=None,
            resume_mode="auto",
            reset_optimize=False,
            no_agent_session=False,
            supervise="off",
            output=None,
            test_mode=None,
            bench_mode=None,
            prompt=None,
            optimize_target="operator",
        )
        request = build_optimize_request(operator, workdir, options)
        self.assertEqual(request.optimize_target, "operator")
```

- [ ] **Step 3: Run the new tests and confirm they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_optimize_command_defaults_optimize_target_to_kernel tests.test_cli.CliParserTests.test_optimize_command_accepts_operator_optimize_target tests.test_cli.CliParserTests.test_optimize_batch_defaults_optimize_target_to_kernel tests.test_cli.CliParserTests.test_optimize_batch_accepts_operator_optimize_target tests.test_models.AgentRequestTests.test_with_prompt_preserves_all_other_fields tests.test_optimize_runtime.OptimizeRuntimeTests.test_build_optimize_request_defaults_optimize_target_to_kernel tests.test_optimize_runtime.OptimizeRuntimeTests.test_build_optimize_request_preserves_operator_optimize_target -v`

Expected: `FAIL` because the parser does not yet expose `--optimize-target`, optimize option models do not yet carry the field, and `AgentRequest` does not yet preserve it.

## Task 2: Add Failing Compare-Perf And Round-Contract Tests

**Files:**
- Modify: `tests/test_comparison_commands.py`
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_optimize_checks.py`
- Modify: `tests/test_optimize_round_contract.py`

- [ ] **Step 1: Add failing compare-perf tests for operator all-view reporting**

Extend the compare-perf tests so an `all` mode becomes the analysis view used by operator-target optimize runs.

Add coverage that asserts:

- `compare-perf` accepts `--metric-source all`
- `all` prints both a kernel section and a total-op section
- each section prints its own per-case lines plus aggregate summaries
- unavailable sections print an explicit unavailable reason instead of silently falling back

Add focused behavior tests in `tests/test_bench_runner.py`, for example:

```python
    def test_compare_perf_all_prints_kernel_and_total_op_sections(self) -> None:
        ...
        exit_code = module.compare_perf_files(
            baseline,
            compare,
            metric_source="all",
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("Metric source section: kernel", output)
        self.assertIn("Metric source section: total-op", output)
```

- [ ] **Step 2: Add failing round-contract tests for `effective_metric_source`**

Update optimize round-contract tests so benchmark-passing rounds must now provide:

- `perf_summary_source=compare-perf`
- `effective_metric_source`

Add one passing fixture with `effective_metric_source="kernel"` and one failing fixture that omits the field.

Add optimize-check coverage so:

- kernel-target rounds with `effective_metric_source="total-op"` or `"mixed"` still pass
- those rounds return a warning-oriented issue or summary note instead of failing
- operator-target rounds can still use `total-op` as canonical while retaining a separate diagnostic kernel view outside the round-state contract

- [ ] **Step 3: Run the new tests and confirm they fail**

Run: `uv run python -m unittest tests.test_comparison_commands tests.test_bench_runner tests.test_optimize_checks tests.test_optimize_round_contract -v`

Expected: `FAIL` because compare-perf does not yet support `all`, and optimize round contracts do not yet require or validate `effective_metric_source`.

## Task 3: Add Failing Prompt And Guidance Contract Tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write the failing prompt tests for worker, unsupervised, resume, and supervisor modes**

Add prompt coverage in `tests/test_cli.py`:

```python
    def test_build_optimize_worker_prompt_mentions_operator_target_contract(self) -> None:
        prompt = build_optimize_worker_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            optimize_target="operator",
        )
        self.assertIn("Target optimization scope for this optimize session: operator.", prompt)
        self.assertIn("Optimize end-to-end operator latency.", prompt)
        self.assertIn("wrapper logic, data movement, scheduling, pre-processing, post-processing, and kernel code", prompt)
        self.assertIn("pure PyTorch rewrite", prompt)
        self.assertNotIn("must continue optimizing the Triton Ascend NPU kernel path itself", prompt)

    def test_build_optimize_unsupervised_prompt_mentions_operator_target_contract(self) -> None:
        prompt = build_optimize_unsupervised_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            optimize_target="operator",
        )
        self.assertIn("Target optimization scope for this optimize session: operator.", prompt)
        self.assertIn("Optimize end-to-end operator latency.", prompt)
        self.assertIn("wrapper logic, data movement, scheduling, pre-processing, post-processing, and kernel code", prompt)
        self.assertNotIn("must continue optimizing the Triton Ascend NPU kernel path itself", prompt)

    def test_build_optimize_resume_prompt_mentions_operator_target_contract(self) -> None:
        prompt = build_optimize_resume_prompt(
            "Round gate passed.",
            optimize_target="operator",
        )
        self.assertIn("Target optimization scope for this optimize session: operator.", prompt)
        self.assertIn("Optimize end-to-end operator latency.", prompt)

    def test_build_optimize_supervisor_prompt_mentions_operator_target_contract(self) -> None:
        prompt = build_optimize_supervisor_prompt(
            Path("/tmp"),
            latest_round_dir=Path("/tmp/opt-round-3"),
            optimize_target="operator",
        )
        self.assertIn("Target optimization scope for this optimize session: operator.", prompt)
        self.assertIn("whole-operator restructuring", prompt)
        self.assertIn("pure PyTorch computation", prompt)
```

- [ ] **Step 2: Write the failing optimize guidance tests**

Add guidance assertions in `tests/test_optimize_guidance.py`:

```python
    def test_prepare_unsupervised_session_mentions_operator_target_when_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_unsupervised_session(
                workdir,
                operator_path=operator,
                agent_name="codex",
                test_mode="differential",
                bench_mode="standalone",
                optimize_target="operator",
            )

            guidance_content = (workdir / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("Target optimization scope: operator.", guidance_content)
            self.assertIn("Optimize end-to-end operator latency.", guidance_content)
            manager.cleanup_unsupervised_session(state)
```

Add runtime coverage in `tests/test_optimize_runtime.py` by extending the fake-runner unsupervised guidance test to assert the operator-target wording when `request.optimize_target == "operator"`.

- [ ] **Step 3: Run the prompt and guidance tests and confirm they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_build_optimize_worker_prompt_mentions_operator_target_contract tests.test_cli.CliParserTests.test_build_optimize_unsupervised_prompt_mentions_operator_target_contract tests.test_cli.CliParserTests.test_build_optimize_resume_prompt_mentions_operator_target_contract tests.test_cli.CliParserTests.test_build_optimize_supervisor_prompt_mentions_operator_target_contract tests.test_optimize_guidance.OptimizeSessionArtifactsManagerTests.test_prepare_unsupervised_session_mentions_operator_target_when_selected tests.test_optimize_runtime.OptimizeRuntimeTests.test_run_optimize_request_uses_unsupervised_session_artifacts -v`

Expected: `FAIL` because optimize prompts and temporary guidance files do not yet know about the operator-target mode.

## Task 4: Implement The Optimize Target Field

**Files:**
- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/optimize.py`
- Modify: `src/helix/models.py`
- Modify: `src/helix/optimize/models.py`
- Modify: `src/helix/optimize/orchestration.py`

- [ ] **Step 1: Add the CLI enum and plumb it into optimize run options**

Update `src/helix/cli.py` to register the new option in the optimize-options block:

```python
            subparser.add_argument(
                "--optimize-target",
                default="kernel",
                choices=("kernel", "operator"),
            )
```

Update `src/helix/optimize/models.py`:

```python
    optimize_target: Literal["kernel", "operator"] = "kernel"
```

Update `src/helix/commands/optimize.py`:

```python
    optimize_target = cast(
        Literal["kernel", "operator"],
        getattr(args, "optimize_target", "kernel"),
    )
```

and pass it into `OptimizeRunOptions(...)`.

- [ ] **Step 2: Add the field to `AgentRequest` and request construction**

Update `src/helix/models.py`:

```python
    optimize_target: Literal["kernel", "operator"] = "kernel"
```

Update `src/helix/optimize/orchestration.py` so `build_optimize_request()` passes:

```python
        optimize_target=options.optimize_target,
```

- [ ] **Step 3: Run the Task 1 tests and make sure they pass**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_optimize_command_defaults_optimize_target_to_kernel tests.test_cli.CliParserTests.test_optimize_command_accepts_operator_optimize_target tests.test_cli.CliParserTests.test_optimize_batch_defaults_optimize_target_to_kernel tests.test_cli.CliParserTests.test_optimize_batch_accepts_operator_optimize_target tests.test_models.AgentRequestTests.test_with_prompt_preserves_all_other_fields tests.test_optimize_runtime.OptimizeRuntimeTests.test_build_optimize_request_defaults_optimize_target_to_kernel tests.test_optimize_runtime.OptimizeRuntimeTests.test_build_optimize_request_preserves_operator_optimize_target -v`

Expected: `PASS`

## Task 5: Adapt Compare-Perf Usage And Record Effective Metric Source

**Files:**
- Modify: `src/helix/commands/comparison.py`
- Modify: `skills/triton-npu-run-eval/scripts/perf_artifacts.py`
- Modify: `skills/triton-npu-run-eval/references/compare-perf.md`
- Modify: `README.md`
- Modify: `skills/triton-npu-optimize-submit-round/references/contract.json`
- Modify: `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round_contract.py`
- Modify: `tests/test_comparison_commands.py`
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_optimize_checks.py`
- Modify: `tests/test_optimize_round_contract.py`

- [ ] **Step 1: Teach compare-perf to accept `all`**

Extend the metric-source surface so `compare-perf` accepts:

```python
choices=("auto", "kernel", "total-op", "all")
```

In `all` mode:

- print a kernel comparison section when kernel comparison is possible
- print a total-op comparison section when total-op comparison is possible
- print explicit unavailable reasons when a section cannot be produced
- do not silently collapse `all` into one fallback view

- [ ] **Step 2: Define target-to-compare-perf behavior**

Use this mapping inside optimize:

- `optimize_target="kernel"` -> `compare-perf --metric-source auto`
- `optimize_target="operator"` -> `compare-perf --metric-source all`

For kernel-target optimize:

- summary output should focus on the resolved kernel-oriented result
- if the resolved source becomes `total-op` or `mixed`, keep the round eligible for best-round selection but emit a warning

For operator-target optimize:

- show both kernel and total-op results to the agent
- record only one canonical round basis, `effective_metric_source="total-op"`

- [ ] **Step 3: Extend optimize round-state contract**

Add `effective_metric_source` to `skills/triton-npu-optimize-submit-round/references/contract.json` and load it through the optimize-check contract model.

Allowed values:

- `kernel`
- `total-op`
- `mixed`

Update round checking so:

- missing `effective_metric_source` is a revise-required issue
- kernel-target rounds with `effective_metric_source in {"total-op", "mixed"}` pass with warning semantics
- operator-target rounds treat `total-op` as canonical

- [ ] **Step 4: Keep `Total speedup` computed but remove it from default optimize-facing display**

Do not remove the underlying calculation.

Instead:

- keep `Total speedup` available in internal compare-perf data
- avoid making it part of the default optimize-target-facing summary requirements for this iteration

- [ ] **Step 5: Run the focused comparison and round-contract tests**

Run: `uv run python -m unittest tests.test_comparison_commands tests.test_bench_runner tests.test_optimize_checks tests.test_optimize_round_contract -v`

Expected: `PASS`

## Task 6: Switch Optimize Prompt Contracts By Target

**Files:**
- Modify: `src/helix/optimize/prompts.py`
- Modify: `src/helix/prompts.py`
- Modify: `src/helix/backends/base.py`
- Modify: `src/helix/optimize/run_loop.py`

- [ ] **Step 1: Add explicit target-aware contract lines in optimize prompt helpers**

In `src/helix/optimize/prompts.py`, add a helper that renders the target-specific contract:

```python
def optimize_target_lines(*, optimize_target: str) -> list[str]:
    if optimize_target == "operator":
        return [
            "Target optimization scope for this optimize session: operator.",
            "Optimize end-to-end operator latency.",
            "You may optimize wrapper logic, data movement, scheduling, pre-processing, post-processing, and kernel code together.",
            "When reporting round performance, show both kernel and total-op comparison views.",
            "Treat total-op as the canonical round conclusion in operator mode.",
            "Preserve a real Triton Ascend NPU computation path.",
            "A pure PyTorch rewrite that bypasses the Triton Ascend NPU path does not count as a successful optimize round.",
        ]
    return [
        "Target optimization scope for this optimize session: kernel.",
        "PyTorch-facing public API may remain as a wrapper when that is the intended operator entrypoint.",
        "You must continue optimizing the Triton Ascend NPU kernel path itself.",
        "Do not replace the core computation with a pure PyTorch implementation just to improve final outputs or benchmark numbers.",
        "A round that bypasses the Triton kernel path with pure PyTorch code does not count as a successful optimize round.",
        "Use the kernel-oriented compare-perf view as the primary round conclusion.",
        "If the comparison falls back away from pure kernel results, record the resolved effective metric source and surface a warning.",
    ]
```

Thread `optimize_target` through:

- `_shared_optimize_prompt_lines()`
- `build_optimize_worker_prompt()`
- `build_optimize_unsupervised_prompt()`
- `build_optimize_supervisor_prompt()`
- `build_optimize_resume_prompt()`

In operator mode, supervisor wording should explicitly allow whole-operator restructuring while still rejecting pure-PyTorch bypasses. In kernel mode, supervisor wording should also allow fallback-driven rounds to pass while flagging their `effective_metric_source` mismatch as a warning.

- [ ] **Step 2: Propagate the new prompt parameter through callers**

Update `src/helix/prompts.py`, `src/helix/backends/base.py`, and `src/helix/optimize/run_loop.py` so resume and initial prompt construction always pass `optimize_target`.

- [ ] **Step 3: Run the prompt tests and make sure they pass**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_build_optimize_worker_prompt_mentions_operator_target_contract tests.test_cli.CliParserTests.test_build_optimize_unsupervised_prompt_mentions_operator_target_contract tests.test_cli.CliParserTests.test_build_optimize_resume_prompt_mentions_operator_target_contract tests.test_cli.CliParserTests.test_build_optimize_supervisor_prompt_mentions_operator_target_contract -v`

Expected: `PASS`

## Task 7: Switch Optimize Workspace Guidance By Target

**Files:**
- Modify: `src/helix/optimize/session_artifacts.py`
- Modify: `src/helix/optimize/memory_file.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Add target-aware guidance rendering**

Update `src/helix/optimize/memory_file.py` so unsupervised and shared guidance accept `optimize_target` and include target-aware lines such as:

```python
def optimize_target_guidance_lines(*, optimize_target: str) -> list[str]:
    if optimize_target == "operator":
        return [
            "Target optimization scope: operator.",
            "Optimize end-to-end operator latency.",
            "You may improve wrapper logic, data movement, scheduling, pre-processing, post-processing, and kernel code in this session.",
            "Do not replace the Triton Ascend NPU computation path with a pure PyTorch rewrite.",
        ]
    return [
        "Target optimization scope: kernel.",
        "Optimize the Triton Ascend NPU kernel path itself.",
        "Do not replace the Triton Ascend NPU computation path with a pure PyTorch rewrite.",
    ]
```

Thread `optimize_target` through:

- `MemoryFileManager.prepare_unsupervised()`
- `MemoryFileManager.prepare_shared()`
- `OptimizeSessionArtifactsManager.prepare_unsupervised_session()`
- `OptimizeSessionArtifactsManager.prepare_supervised_session()`
- `run_optimize_request()`

- [ ] **Step 2: Update tests to assert the new guidance behavior**

Extend guidance and runtime tests so:

- default behavior still shows kernel-target wording
- operator mode shows operator-target wording

- [ ] **Step 3: Run the guidance tests and make sure they pass**

Run: `uv run python -m unittest tests.test_optimize_guidance tests.test_optimize_runtime.OptimizeRuntimeTests.test_run_optimize_request_uses_unsupervised_session_artifacts -v`

Expected: `PASS`

## Task 8: Update Skill And README Contracts

**Files:**
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `README.md`

- [ ] **Step 1: Update optimize skill guidance for explicit target modes**

In `skills/triton/triton-npu-optimize/SKILL.md`, revise the hard rules so they describe both modes:

- keep the kernel-focused wording as the default contract
- add operator-target wording that allows whole-operator optimization work
- document that kernel-target optimize uses kernel-oriented auto comparison while operator-target optimize must show both kernel and total-op views
- keep the pure-PyTorch prohibition in both modes

Do not turn the skill into a CLI parser mirror. Keep it focused on execution semantics.

- [ ] **Step 2: Document the new option in `README.md`**

Add `--optimize-target kernel|operator` to optimize and optimize-batch option lists, document the default, add one operator-target example command, and describe the target-specific compare-perf behavior:

- kernel target uses kernel-oriented auto comparison
- operator target shows both kernel and total-op views
- only one `effective_metric_source` is recorded for each round

- [ ] **Step 3: Run focused documentation-adjacent tests if any prompt text assertions depend on these semantics**

Run: `uv run python -m unittest tests.test_cli -v`

Expected: `PASS`

## Task 9: Run Final Verification

**Files:**
- No code changes

- [ ] **Step 1: Run the focused optimize regression slice**

Run: `uv run python -m unittest tests.test_cli tests.test_models tests.test_comparison_commands tests.test_bench_runner tests.test_optimize_guidance tests.test_optimize_runtime tests.test_optimize_checks tests.test_optimize_round_contract -v`

Expected: `PASS`

- [ ] **Step 2: Run repository verification commands required by project guidance**

Run: `uv run --group dev ruff check`

Expected: `PASS`

Run: `uv run pyright`

Expected: `PASS`

Run: `uv run python -m unittest discover -s tests -v`

Expected: `PASS`

## Self-Review

- Spec coverage: the plan covers CLI exposure, request plumbing, prompt contracts, resume flows, supervisor audit wording, workspace guidance, skill docs, and README updates.
- Placeholder scan: every task names concrete files, commands, and expected outcomes.
- Type consistency: the plan uses one target field name, `optimize_target`, one target value set, `kernel|operator`, and one resolved round-comparison field, `effective_metric_source`, across request construction, prompt rendering, compare-perf usage, and round validation.
