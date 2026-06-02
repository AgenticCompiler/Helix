# Optimize Round Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `--supervise/--supervisor` with `--round-mode {continuous,checked,supervised}`, add CLI-owned baseline preflight and technical round gating, keep supervisor as an audit-only layer after technical validation, and remove `round-brief.md` from round-to-round execution.

**Architecture:** Keep one continuous optimize path for long-running sessions, add one multi-invocation path for `checked` and `supervised`, and reuse a single CLI technical gate plus baseline preflight across both multi-invocation modes. Let the CLI inject continuation context directly into the next worker prompt, keep no live handoff file in `checked`, and preserve only `supervisor-report.md` plus optional history in `supervised`.

**Tech Stack:** Python 3, `argparse`, dataclasses, existing optimize prompt/render helpers, `unittest`, `ruff`, `pyright`

---

## File Map

- Modify: `src/triton_agent/cli.py`
  - Replace optimize-only `--supervise/--supervisor` parser wiring with `--round-mode`.
- Modify: `src/triton_agent/commands/optimize.py`
  - Parse and validate `round_mode` instead of `supervise`.
- Modify: `src/triton_agent/models.py`
  - Replace `AgentRequest.supervise` with `AgentRequest.round_mode` and carry prompt-rebuild context needed by runtime.
- Modify: `src/triton_agent/optimize/models.py`
  - Replace `OptimizeRunOptions.supervise` with `round_mode` and add any small enums/dataclasses needed for baseline preflight.
- Modify: `src/triton_agent/prompts.py`
  - Route optimize prompt building by `round_mode`.
- Modify: `src/triton_agent/optimize/prompts.py`
  - Rename continuous/round prompt helpers, add a baseline-focused prompt, and narrow supervisor wording.
- Modify: `src/triton_agent/optimize/orchestration.py`
  - Populate the new request fields and dispatch to continuous vs multi-invocation execution.
- Modify: `src/triton_agent/optimize/execution.py`
  - Implement baseline preflight, CLI technical gate, and the checked/supervised multi-invocation controller.
- Modify: `src/triton_agent/optimize/run_loop.py`
  - Keep this module focused on continuous recovery/min-round resume logic only.
- Modify: `src/triton_agent/optimize/runtime_handoff.py`
  - Narrow the live runtime tree to supervisor-report handling only, or remove the module if the remaining behavior folds cleanly into session artifacts.
- Modify: `src/triton_agent/optimize/session_artifacts.py`
  - Split continuous, checked, and supervised session artifact preparation/cleanup without a checked-mode handoff file.
- Modify: `src/triton_agent/optimize/memory_file.py`
  - Render separate continuous vs round-gated shared guidance text.
- Modify: `README.md`
  - Replace optimize supervise docs with round-mode docs and baseline preflight behavior.
- Test: `tests/test_cli.py`
  - Parser coverage, prompt wording coverage, and request-building coverage for `round_mode`.
- Test: `tests/test_optimize_commands.py`
  - `optimize_run_options_from_args()` coverage for the new flag.
- Test: `tests/test_optimize_runtime.py`
  - Runtime dispatch, baseline preflight, technical gate, checked flow, and supervised flow.
- Test: `tests/test_optimize_guidance.py`
  - Guidance/artifact preparation for continuous, checked, and supervised sessions.
- Test: `tests/test_supervisor.py`
  - Remaining run-loop coverage after supervised loop ownership moves into the multi-invocation runtime.
- Test: `tests/test_backends_base.py`, `tests/test_codex_runner.py`, `tests/test_claude_runner.py`, `tests/test_opencode_runner.py`, `tests/test_pi_runner.py`, `tests/test_traecli_runner.py`
  - Adjust any optimize request construction fixtures that still set `supervise="on"` or `supervise="off"`.

### Task 1: Replace The CLI And Request Surface With `round_mode`

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/optimize/models.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_optimize_commands.py`

- [ ] **Step 1: Write the failing parser and option-mapping tests**

```python
def test_optimize_command_accepts_round_modes(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize", "-i", "kernel.py", "--round-mode", "checked"])
    self.assertEqual(args.round_mode, "checked")
    options = optimize_run_options_from_args(args)
    self.assertEqual(options.round_mode, "checked")


def test_optimize_command_defaults_round_mode_continuous(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize", "-i", "kernel.py"])
    self.assertEqual(args.round_mode, "continuous")
    options = optimize_run_options_from_args(args)
    self.assertEqual(options.round_mode, "continuous")


def test_optimize_batch_accepts_round_modes(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["optimize-batch", "-i", "kernels", "--round-mode", "supervised"]
    )
    self.assertEqual(args.round_mode, "supervised")
    options = optimize_run_options_from_args(args)
    self.assertEqual(options.round_mode, "supervised")
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_optimize_command_accepts_round_modes tests.test_cli.CliParserTests.test_optimize_command_defaults_round_mode_continuous tests.test_cli.CliParserTests.test_optimize_batch_accepts_round_modes -v`

Expected: parser or option-mapping assertions fail because `--round-mode` does not exist and defaults still use `supervise`.

- [ ] **Step 3: Replace `supervise` with `round_mode` in the CLI, options, and request models**

```python
# src/triton_agent/cli.py
_ROUND_MODE_CHOICES = ("continuous", "checked", "supervised")

subparser.add_argument(
    "--round-mode",
    default="continuous",
    choices=_ROUND_MODE_CHOICES,
)
```

```python
# src/triton_agent/commands/optimize.py
def _validate_round_mode(
    args: argparse.Namespace,
) -> Literal["continuous", "checked", "supervised"]:
    value = str(getattr(args, "round_mode", "continuous"))
    if value not in {"continuous", "checked", "supervised"}:
        raise ValueError(
            "--round-mode must be one of 'continuous', 'checked', or 'supervised'"
        )
    return cast(Literal["continuous", "checked", "supervised"], value)
```

```python
# src/triton_agent/optimize/models.py
@dataclass(frozen=True)
class OptimizeRunOptions:
    agent_name: str
    interact: bool
    verbose: bool
    show_output: bool
    remote: str | None
    remote_workdir: str | None
    min_rounds: int | None
    resume_mode: str
    reset_optimize: bool
    no_agent_session: bool
    round_mode: Literal["continuous", "checked", "supervised"]
    output: str | None
    test_mode: str | None
    bench_mode: str | None
    prompt: str | None
```

```python
# src/triton_agent/models.py
@dataclass
class AgentRequest:
    command_kind: CommandKind
    input_path: Path
    operator_path: Optional[Path]
    output_path: Optional[Path]
    test_mode: Optional[str]
    bench_mode: Optional[str]
    interact: bool
    verbose: bool
    show_output: bool
    force_overwrite: bool
    agent_name: str
    skill_name: str
    prompt: str
    workdir: Path
    round_mode: Literal["continuous", "checked", "supervised"] = "continuous"
    remote: Optional[str] = None
    remote_workdir: Optional[str] = None
    additional_user_prompt: Optional[str] = None
    optimize_role: str | None = None
```

```python
# src/triton_agent/optimize/orchestration.py
return AgentRequest(
    command_kind=CommandKind.OPTIMIZE,
    input_path=input_path,
    operator_path=input_path,
    output_path=output_path,
    test_mode=test_mode,
    bench_mode=bench_mode,
    interact=options.interact,
    verbose=options.verbose,
    show_output=options.show_output,
    force_overwrite=False,
    agent_name=options.agent_name,
    skill_name=COMMAND_TO_SKILL[CommandKind.OPTIMIZE],
    prompt=prompt,
    workdir=workdir,
    round_mode=options.round_mode,
    remote=options.remote,
    remote_workdir=options.remote_workdir,
    additional_user_prompt=options.prompt,
    optimize_role="worker" if options.round_mode != "continuous" else None,
    min_rounds=options.min_rounds,
    continue_optimize=resolution.resume_existing_session,
    no_agent_session=options.no_agent_session,
)
```

- [ ] **Step 4: Run the updated parser and command tests**

Run: `uv run python -m unittest tests.test_cli tests.test_optimize_commands -v`

Expected: `OK`

- [ ] **Step 5: Commit the surface rename**

```bash
git add src/triton_agent/cli.py src/triton_agent/commands/optimize.py src/triton_agent/models.py src/triton_agent/optimize/models.py src/triton_agent/optimize/orchestration.py tests/test_cli.py tests/test_optimize_commands.py
git commit -m "feat: replace optimize supervise flag with round mode"
```

### Task 2: Introduce Phase-Specific Optimize Prompt Builders

**Files:**
- Modify: `src/triton_agent/prompts.py`
- Modify: `src/triton_agent/optimize/prompts.py`
- Modify: `src/triton_agent/optimize/orchestration.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_backends_base.py`
- Test: `tests/test_codex_runner.py`
- Test: `tests/test_claude_runner.py`
- Test: `tests/test_opencode_runner.py`
- Test: `tests/test_pi_runner.py`
- Test: `tests/test_traecli_runner.py`

- [ ] **Step 1: Write the failing prompt and request-context tests**

```python
def test_build_optimize_round_prompt_mentions_cli_validation(self) -> None:
    prompt = build_optimize_round_prompt(
        Path("/tmp/op.py"),
        Path("/tmp/opt_op.py"),
        test_mode="differential",
        bench_mode="standalone",
        round_mode="checked",
        baseline_ready=True,
    )
    self.assertIn("This invocation owns exactly one round.", prompt)
    self.assertIn("The CLI will validate this round after the invocation exits.", prompt)
    self.assertNotIn("run `check-round` and repair the round until it passes", prompt)


def test_build_optimize_baseline_prompt_mentions_baseline_only_scope(self) -> None:
    prompt = build_optimize_baseline_prompt(
        Path("/tmp/op.py"),
        Path("/tmp/opt_op.py"),
        test_mode="differential",
        bench_mode="standalone",
        round_mode="checked",
        baseline_state="needs-repair",
    )
    self.assertIn("Repair or establish `baseline/` before the round loop begins.", prompt)
    self.assertIn("Do not open a new optimization round yet.", prompt)
```

- [ ] **Step 2: Run the targeted prompt tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_build_optimize_round_prompt_mentions_cli_validation tests.test_cli.CliParserTests.test_build_optimize_baseline_prompt_mentions_baseline_only_scope -v`

Expected: import or assertion failures because the new prompt builders do not exist yet.

- [ ] **Step 3: Add continuous, round, and baseline prompt builders plus round-mode dispatch**

```python
# src/triton_agent/optimize/prompts.py
def build_optimize_continuous_prompt(
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str = "A5",
    optimize_target: str = "kernel",
    min_rounds: int | None = None,
    resume_existing_session: bool = False,
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
    baseline_ready: bool | None = None,
) -> str:
    lines = [
        "This invocation is a continuous optimize run.",
        "Own the end-to-end optimize session and continue optimizing until the session should stop.",
    ]
    if baseline_ready is True:
        lines.append("The baseline has already been validated. Reuse it and do not rebuild it.")
    elif baseline_ready is False:
        lines.append("Repair or establish `baseline/` before opening round 1.")
    lines.extend(
        _shared_optimize_prompt_lines(
            target_chip=target_chip,
            optimize_check_line="Use the staged `triton-npu-optimize-check` skill to validate every completed round.",
            optimize_target=optimize_target,
        )
    )
    return _finalize_optimize_prompt_lines(
        lines=lines,
        resume_existing_session=resume_existing_session,
        compiler_source_path=compiler_source_path,
        compiler_source_commit=compiler_source_commit,
        enable_cann_ext_api=enable_cann_ext_api,
    )


def build_optimize_round_prompt(
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str = "A5",
    optimize_target: str = "kernel",
    resume_existing_session: bool = False,
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
    round_mode: Literal["checked", "supervised"],
    baseline_ready: bool = True,
) -> str:
    lines = [
        "This invocation owns exactly one round.",
        "Read `.triton-agent/round-brief.md` before acting.",
        "The baseline has already been validated before this round.",
        "Produce all required round artifacts before stopping.",
        "The CLI will validate this round after the invocation exits.",
        "If the round needs repairs, a later invocation will return with a repair brief.",
    ]
    if round_mode == "supervised":
        lines.append(
            "After the CLI validates this round, a supervisor audit pass will review it."
        )
    lines.extend(
        _shared_optimize_prompt_lines(
            target_chip=target_chip,
            optimize_check_line="The CLI will run the technical round check after this invocation exits.",
            optimize_target=optimize_target,
        )
    )
    return _finalize_optimize_prompt_lines(
        lines=lines,
        resume_existing_session=resume_existing_session,
        compiler_source_path=compiler_source_path,
        compiler_source_commit=compiler_source_commit,
        enable_cann_ext_api=enable_cann_ext_api,
    )


def build_optimize_baseline_prompt(
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str = "A5",
    optimize_target: str = "kernel",
    resume_existing_session: bool = False,
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
    baseline_state: str,
    round_mode: str,
) -> str:
    return "\n".join(
        [
            "This invocation repairs the optimize baseline before the round loop begins.",
            f"Baseline preflight result: {baseline_state}.",
            "Repair or establish `baseline/` before the round loop begins.",
            "Use the staged `triton-npu-optimize-check` skill to run `check-baseline` until it passes.",
            "Do not open a new optimization round yet.",
        ]
    )
```

```python
# src/triton_agent/prompts.py
if command_kind == CommandKind.OPTIMIZE:
    if round_mode == "continuous":
        lines.extend(
            build_optimize_continuous_prompt(
                input_path,
                output_path,
                test_mode=test_mode,
                bench_mode=bench_mode,
                target_chip=target_chip or "A5",
                optimize_target=optimize_target,
                min_rounds=min_rounds,
                resume_existing_session=should_resume_existing_session,
                compiler_source_path=compiler_source_path,
                compiler_source_commit=compiler_source_commit,
                enable_cann_ext_api=enable_cann_ext_api,
            ).splitlines()
        )
    else:
        lines.extend(
            build_optimize_round_prompt(
                input_path,
                output_path,
                test_mode=test_mode,
                bench_mode=bench_mode,
                target_chip=target_chip or "A5",
                optimize_target=optimize_target,
                resume_existing_session=should_resume_existing_session,
                compiler_source_path=compiler_source_path,
                compiler_source_commit=compiler_source_commit,
                enable_cann_ext_api=enable_cann_ext_api,
                round_mode=round_mode,
                baseline_ready=True,
            ).splitlines()
        )
```

```python
# src/triton_agent/optimize/orchestration.py
prompt = append_additional_user_instructions(
    build_prompt(
        CommandKind.OPTIMIZE,
        input_path,
        input_path,
        output_path,
        test_mode,
        bench_mode,
        False,
        options.remote,
        options.remote_workdir,
        round_mode=options.round_mode,
        min_rounds=options.min_rounds,
        continue_optimize=resolution.resume_existing_session,
        target_chip=options.target_chip,
        optimize_target=options.optimize_target,
        compiler_source_path=compiler_source.path if compiler_source is not None else None,
        compiler_source_commit=compiler_source.commit if compiler_source is not None else None,
        enable_cann_ext_api=options.enable_cann_ext_api,
    ),
    options.prompt,
)
```

- [ ] **Step 4: Run prompt and backend-fixture tests**

Run: `uv run python -m unittest tests.test_cli tests.test_backends_base tests.test_codex_runner tests.test_claude_runner tests.test_opencode_runner tests.test_pi_runner tests.test_traecli_runner -v`

Expected: `OK`

- [ ] **Step 5: Commit the prompt split**

```bash
git add src/triton_agent/prompts.py src/triton_agent/optimize/prompts.py src/triton_agent/optimize/orchestration.py tests/test_cli.py tests/test_backends_base.py tests/test_codex_runner.py tests/test_claude_runner.py tests/test_opencode_runner.py tests/test_pi_runner.py tests/test_traecli_runner.py
git commit -m "refactor: split optimize prompts by round mode"
```

### Task 3: Implement Baseline Preflight And The Multi-Invocation Runtime

**Files:**
- Modify: `src/triton_agent/optimize/execution.py`
- Modify: `src/triton_agent/optimize/run_loop.py`
- Modify: `src/triton_agent/optimize/orchestration.py`
- Modify: `src/triton_agent/optimize/models.py`
- Test: `tests/test_optimize_runtime.py`
- Test: `tests/test_supervisor.py`

- [ ] **Step 1: Write failing runtime tests for checked and supervised control flow**

```python
def test_run_optimize_request_delegates_checked_flow_to_helper(self) -> None:
    workdir = Path(tmp)
    operator = workdir / "kernel.py"
    operator.write_text("print('x')\n", encoding="utf-8")
    request = AgentRequest(
        command_kind=CommandKind.OPTIMIZE,
        input_path=operator,
        operator_path=operator,
        output_path=workdir / "opt_kernel.py",
        test_mode="differential",
        bench_mode="standalone",
        interact=False,
        verbose=False,
        show_output=False,
        force_overwrite=False,
        agent_name="codex",
        skill_name="triton-npu-optimize",
        prompt="Optimize this operator",
        workdir=workdir,
        round_mode="checked",
        optimize_role="worker",
    )
    with patch.object(execution_module, "execute_multi_invocation_optimize", return_value=expected) as mocked:
        result = run_optimize_request(request)
    self.assertIs(result, expected)
    mocked.assert_called_once()


def test_multi_invocation_optimize_repairs_invalid_baseline_before_first_round(self) -> None:
    operator = workdir / "kernel.py"
    operator.write_text("print('x')\n", encoding="utf-8")
    self.assertEqual(runner.requests[0].prompt.splitlines()[0], "This invocation repairs the optimize baseline before the round loop begins.")
    self.assertIn("This invocation owns exactly one round.", runner.requests[1].prompt)


def test_multi_invocation_checked_stops_without_supervisor(self) -> None:
    self._write_baseline(workdir)
    self._write_round(workdir, "opt-round-1", parent_round="round-0", round_disposition="stop")
    self.assertEqual(runner.events, ["baseline-run", "round-run"])


def test_multi_invocation_supervised_runs_supervisor_only_after_technical_pass(self) -> None:
    self._write_baseline(workdir)
    self._write_round(workdir, "opt-round-1", parent_round="round-0", round_disposition="continue")
    self.assertEqual(runner.events, ["round-run", "supervisor-run"])
```

- [ ] **Step 2: Run the targeted runtime tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_runtime tests.test_supervisor -v`

Expected: helper-name assertions or flow assertions fail because only the old supervised/unsupervised split exists.

- [ ] **Step 3: Add baseline preflight, CLI technical gate, and the new multi-invocation controller**

```python
# src/triton_agent/optimize/models.py
class BaselinePreflightState(str, Enum):
    READY = "ready"
    NEEDS_PREPARE = "needs-prepare"
    NEEDS_REPAIR = "needs-repair"


@dataclass(frozen=True)
class BaselinePreflightResult:
    state: BaselinePreflightState
    issues: tuple[str, ...]
```

```python
# src/triton_agent/optimize/execution.py
def execute_multi_invocation_optimize(
    runner: AgentRunner,
    artifacts_manager: OptimizeSessionArtifactsManager,
    request: AgentRequest,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    verbose_stream: TextIO,
) -> AgentResult:
    controller = MultiInvocationOptimizeController(
        runner,
        artifacts_manager,
        stdout=stdout,
        stderr=stderr,
        verbose_stream=verbose_stream,
    )
    baseline_result = controller.preflight_baseline(request)
    if baseline_result.state is not BaselinePreflightState.READY:
        baseline_fix_result = controller.run_baseline_phase(request, baseline_result)
        if not baseline_fix_result.succeeded:
            return baseline_fix_result
    return controller.run_round_loop(request)
```

```python
class MultiInvocationOptimizeController:
    def technical_gate(self, request: AgentRequest) -> GateResult:
        latest_round_dir = _latest_round_dir(request.workdir)
        if latest_round_dir is None:
            return GateResult(
                decision=GateDecision.REVISE_REQUIRED,
                blocking_issues=("missing opt-round-* directory after round run",),
            )

        check_result = optimize_checks.check_round(
            latest_round_dir,
            optimize_target=request.optimize_target,
        )
        if check_result.decision == "hard-fail":
            return GateResult(
                decision=GateDecision.HARD_FAIL,
                blocking_issues=check_result.issues,
            )
        if check_result.decision == "revise-required":
            return GateResult(
                decision=GateDecision.REVISE_REQUIRED,
                blocking_issues=check_result.issues,
            )

        round_state = load_round_state(latest_round_dir)
        round_count = _count_round_directories(request.workdir)
        if request.min_rounds is not None and round_count < request.min_rounds:
            return GateResult(
                decision=GateDecision.PASS_CONTINUE,
                blocking_issues=(
                    f"minimum round requirement not yet satisfied: {round_count}/{request.min_rounds}",
                ),
            )
        decision = (
            GateDecision.PASS_STOP
            if round_state.round_disposition == "stop"
            else GateDecision.PASS_CONTINUE
        )
        return GateResult(decision=decision, blocking_issues=check_result.issues)
```

```python
# src/triton_agent/optimize/orchestration.py
if request.round_mode == "continuous":
    return optimize_execution.execute_continuous_optimize(
        runner,
        artifacts_manager,
        request,
        stdout=stdout,
        stderr=stderr,
        verbose_stream=verbose_stream,
    )
return optimize_execution.execute_multi_invocation_optimize(
    runner,
    artifacts_manager,
    request,
    stdout=stdout,
    stderr=stderr,
    verbose_stream=verbose_stream,
)
```

```python
# src/triton_agent/optimize/run_loop.py
class OptimizeRunLoop:
    def run(self, runner: SupportsOptimizeRecovery, request: AgentRequest) -> AgentResult:
        attempt = 0
        current_request = request
        resume_summary: str | None = None
        return self._run_unsupervised_loop(
            runner,
            current_request,
            attempt=attempt,
            resume_summary=resume_summary,
        )
```

- [ ] **Step 4: Run the runtime suite again**

Run: `uv run python -m unittest tests.test_optimize_runtime tests.test_supervisor -v`

Expected: `OK`

- [ ] **Step 5: Commit the runtime controller**

```bash
git add src/triton_agent/optimize/execution.py src/triton_agent/optimize/run_loop.py src/triton_agent/optimize/orchestration.py src/triton_agent/optimize/models.py tests/test_optimize_runtime.py tests/test_supervisor.py
git commit -m "feat: add checked optimize round controller"
```

### Task 4: Split Checked And Supervised Session Artifacts And Refresh Docs

**Files:**
- Modify: `src/triton_agent/optimize/runtime_handoff.py`
- Modify: `src/triton_agent/optimize/session_artifacts.py`
- Modify: `src/triton_agent/optimize/memory_file.py`
- Modify: `README.md`
- Test: `tests/test_optimize_guidance.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing artifact/guidance tests**

```python
def test_prepare_checked_session_creates_round_brief_without_supervisor_report(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        operator = workdir / "kernel.py"
        operator.write_text("print('x')\n", encoding="utf-8")

        manager = OptimizeSessionArtifactsManager()
        state = manager.prepare_checked_session(workdir, agent_name="codex")

        guidance_content = (workdir / "AGENTS.md").read_text(encoding="utf-8")
        self.assertTrue(state.round_brief_path.exists())
        self.assertIsNone(state.supervisor_report_path)
        self.assertIn("Use `.triton-agent/round-brief.md` as the live handoff file.", guidance_content)
        self.assertNotIn("supervisor-report.md", guidance_content)
```

```python
def test_supervised_session_still_creates_supervisor_report(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        operator = workdir / "kernel.py"
        operator.write_text("print('x')\n", encoding="utf-8")
        manager = OptimizeSessionArtifactsManager()
        state = manager.prepare_supervised_session(workdir, agent_name="codex")
    self.assertTrue(state.supervisor_report_path.exists())
    self.assertTrue(state.history_dir.exists())
```

- [ ] **Step 2: Run the targeted guidance tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_guidance.OptimizeSessionArtifactsManagerTests -v`

Expected: attribute or assertion failures because checked-mode artifacts do not exist yet.

- [ ] **Step 3: Add checked-mode artifact preparation and update documentation text**

```python
# src/triton_agent/optimize/runtime_handoff.py
@dataclass
class RuntimeHandoffState:
    runtime_root: Path
    round_brief_path: Path
    supervisor_report_path: Path | None
    history_dir: Path | None
    created_paths: tuple[Path, ...]


def prepare(self, workdir: Path, *, include_supervisor: bool) -> RuntimeHandoffState:
    runtime_root = workdir / ".triton-agent"
    runtime_root.mkdir(parents=True, exist_ok=True)
    round_brief_path.write_text("# Optimize Round Brief\n\nPending runtime handoff.\n", encoding="utf-8")
    if include_supervisor:
        supervisor_report_path = runtime_root / "supervisor-report.md"
        history_dir = runtime_root / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        supervisor_report_path.write_text(
            "# Optimize Supervisor Report\n\nPending first supervisor pass.\n",
            encoding="utf-8",
        )
    else:
        supervisor_report_path = None
        history_dir = None
```

```python
# src/triton_agent/optimize/session_artifacts.py
def prepare_checked_session(
    self,
    workdir: Path,
    agent_name: str,
    optimize_target: str = "kernel",
) -> OptimizeSessionArtifactsState:
    runtime_handoff_state = self._runtime_handoffs.prepare(workdir, include_supervisor=False)
    archive_state = self._archives.prepare(workdir, include_shared_guidance_snapshot=True)
    memory_file_state = self._memory_files.prepare_round_gated(
        workdir,
        agent_name=agent_name,
        optimize_target=optimize_target,
        include_supervisor_handoff=False,
    )
    return OptimizeSessionArtifactsState(
        memory_file=memory_file_state,
        archive=archive_state,
        runtime_handoff=runtime_handoff_state,
    )


def prepare_supervised_session(
    self,
    workdir: Path,
    agent_name: str,
    optimize_target: str = "kernel",
) -> OptimizeSessionArtifactsState:
    runtime_handoff_state = self._runtime_handoffs.prepare(workdir, include_supervisor=True)
    archive_state = self._archives.prepare(workdir, include_shared_guidance_snapshot=True)
    memory_file_state = self._memory_files.prepare_round_gated(
        workdir,
        agent_name=agent_name,
        optimize_target=optimize_target,
        include_supervisor_handoff=True,
    )
    return OptimizeSessionArtifactsState(
        memory_file=memory_file_state,
        archive=archive_state,
        runtime_handoff=runtime_handoff_state,
    )
```

```python
# src/triton_agent/optimize/memory_file.py
_ROUND_GATED_GUIDANCE_TEMPLATE = dedent(
    \"\"\"\
    # {guidance_filename}

    ## Triton Agent Optimize Round Loop

    This workspace is under an optimize round loop.

    \"\"\"
) + _OPTIMIZE_GUIDANCE_RULES_BLOCK + dedent(
    \"\"\"\
    Use the staged workspace skills as the workflow source of truth.
    Role-specific behavior comes from the launch prompt.
    Use `.triton-agent/round-brief.md` as the live handoff file.
    Treat `baseline/` as the canonical optimize baseline.
    Use `compare-perf` as the authoritative source for round performance summaries.
    {analysis_block}{high_priority_pattern_block}{compiler_source_block}{cann_ext_api_block}\
    \"\"\"
)
```

```markdown
<!-- README.md -->
- `--round-mode continuous|checked|supervised`: default is `continuous`
- `continuous`: one long-running optimize agent owns multiple rounds
- `checked`: one round per invocation; the CLI validates each round and decides whether to continue
- `supervised`: one round per invocation; the CLI validates each round and a supervisor audit decides whether to continue
- For `checked` and `supervised`, optimize runs a baseline preflight before the round loop and repairs `baseline/` first when needed.
```

- [ ] **Step 4: Run the guidance/runtime/docs-adjacent tests and full repo verification**

Run: `uv run python -m unittest tests.test_optimize_guidance tests.test_optimize_runtime -v`

Expected: `OK`

Run: `uv run --group dev ruff check`

Expected: exit code `0`

Run: `uv run pyright`

Expected: exit code `0`

Run: `uv run python -m unittest discover -s tests -v`

Expected: `OK`

- [ ] **Step 5: Commit the artifact split and docs update**

```bash
git add src/triton_agent/optimize/runtime_handoff.py src/triton_agent/optimize/session_artifacts.py src/triton_agent/optimize/memory_file.py README.md tests/test_optimize_guidance.py tests/test_optimize_runtime.py
git commit -m "feat: add checked optimize artifacts and docs"
```
