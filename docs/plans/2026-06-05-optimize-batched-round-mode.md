# Optimize Batched Round Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace optimize's `continuous` round mode with a batched checked/supervised flow that uses `--round-batch-size`, validates every new round in a batch, and runs one supervisor pass per batch.

**Architecture:** Keep optimize orchestration in the CLI/runtime layer while preserving the staged optimize skills as the workflow source of truth. Unify optimize round execution around one batched multi-invocation controller, move session-level stop policy out of the per-round checker, and update worker/supervisor prompts so each worker launch owns a bounded round range instead of either one round or an unconstrained continuous session.

**Tech Stack:** Python `argparse`, dataclasses, existing optimize orchestration and prompt builders, optimize skill contract scripts, Python `unittest`

---

### Task 1: Update CLI And Request Models For Batched Round Mode

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/optimize/validation.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_models.py`
- Test: `tests/test_optimize_commands.py`

- [ ] **Step 1: Write failing parser and model tests for the new round-mode surface**

Add parser coverage in `tests/test_cli.py` for:

```python
def test_optimize_command_defaults_round_mode_checked(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize", "-i", "kernel.py"])
    self.assertEqual(args.round_mode, "checked")
    options = optimize_run_options_from_args(args)
    self.assertEqual(options.round_mode, "checked")

def test_optimize_command_accepts_round_batch_size(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize", "-i", "kernel.py", "--round-batch-size", "3"])
    self.assertEqual(args.round_batch_size, 3)
    options = optimize_run_options_from_args(args)
    self.assertEqual(options.round_batch_size, 3)

def test_optimize_batch_defaults_round_batch_size_to_ten(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize-batch", "-i", "kernels"])
    self.assertEqual(args.round_batch_size, 10)
    options = optimize_run_options_from_args(args)
    self.assertEqual(options.round_batch_size, 10)
```

Replace the old continuous-default tests and add a model test in `tests/test_models.py` such as:

```python
def test_round_mode_defaults_to_checked(self) -> None:
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
    )

    self.assertEqual(request.round_mode, "checked")
    self.assertEqual(request.round_batch_size, 10)
```

Add command validation coverage in `tests/test_optimize_commands.py`:

```python
def test_handle_optimize_rejects_interactive_batched_mode(self) -> None:
    parser = build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        operator = Path(tmp) / "kernel.py"
        operator.write_text("print('x')\n", encoding="utf-8")
        args = parser.parse_args(["optimize", "-i", str(operator), "--interact"])

        with self.assertRaises(SystemExit) as exc:
            handle_optimize(parser, args)

        self.assertEqual(exc.exception.code, 2)
```

- [ ] **Step 2: Run the focused CLI and model tests to verify they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.TestCli.test_optimize_command_defaults_round_mode_checked \
  tests.test_cli.TestCli.test_optimize_command_accepts_round_batch_size \
  tests.test_cli.TestCli.test_optimize_batch_defaults_round_batch_size_to_ten \
  tests.test_models.AgentRequestTests.test_round_mode_defaults_to_checked \
  tests.test_optimize_commands.OptimizeCommandHandlerTests.test_handle_optimize_rejects_interactive_batched_mode \
  -v
```

Expected: FAIL because optimize still defaults to `continuous`, does not expose `round_batch_size`, and still treats interactive support as continuous-only.

- [ ] **Step 3: Implement the new CLI and model surface**

Update `src/triton_agent/cli.py` so optimize commands expose:

```python
_ROUND_MODE_CHOICES = ("checked", "supervised")
```

and:

```python
subparser.add_argument("--round-batch-size", type=int, default=10)
```

Update `src/triton_agent/optimize/models.py`:

```python
@dataclass(frozen=True)
class OptimizeRunOptions:
    ...
    round_mode: Literal["checked", "supervised"]
    round_batch_size: int = 10
```

Update `src/triton_agent/models.py`:

```python
@dataclass
class AgentRequest:
    ...
    round_mode: Literal["checked", "supervised"] = "checked"
    round_batch_size: int = 10
```

Update `src/triton_agent/commands/optimize.py` so `_validate_round_mode()` only accepts `checked` and `supervised`, `optimize_run_options_from_args()` maps `round_batch_size`, and `_validate_agent_options()` rejects all optimize `--interact` usage with a batched-mode error.

Update `src/triton_agent/optimize/validation.py`:

```python
def validate_optimize_options(..., round_batch_size: int) -> None:
    if round_batch_size < 1:
        raise ValueError("--round-batch-size must be at least 1")
```

Wire the new argument through `handle_optimize()` and `handle_optimize_batch()`.

- [ ] **Step 4: Run the focused CLI and model tests to verify they pass**

Run:

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_models \
  tests.test_optimize_commands \
  -v
```

Expected: PASS for the updated round-mode defaults, round-batch-size mapping, and interactive rejection behavior.

- [ ] **Step 5: Commit**

```bash
git add src/triton_agent/cli.py src/triton_agent/commands/optimize.py src/triton_agent/models.py src/triton_agent/optimize/models.py src/triton_agent/optimize/validation.py tests/test_cli.py tests/test_models.py tests/test_optimize_commands.py
git commit -m "feat: add optimize batched round mode options"
```

### Task 2: Replace Continuous Prompting With Batched Worker Prompt Contracts

**Files:**
- Modify: `src/triton_agent/prompts.py`
- Modify: `src/triton_agent/optimize/prompts.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_codex_runner.py`
- Test: `tests/test_opencode_runner.py`
- Test: `tests/test_claude_runner.py`
- Test: `tests/test_pi_runner.py`
- Test: `tests/test_traecli_runner.py`
- Test: `tests/test_backends_base.py`

- [ ] **Step 1: Write failing prompt tests for batched worker semantics**

Add or replace prompt tests in `tests/test_cli.py` such as:

```python
def test_optimize_prompt_defaults_to_checked_batch_mode(self) -> None:
    prompt = build_prompt(
        CommandKind.OPTIMIZE,
        Path("/tmp/kernel.py"),
        Path("/tmp/kernel.py"),
        Path("/tmp/opt_kernel.py"),
        "differential",
        "standalone",
        False,
        min_rounds=5,
        round_mode="checked",
    )

    self.assertIn("This invocation owns rounds 1 through 5.", prompt)
    self.assertIn("Do not pre-plan the full batch before acting.", prompt)
```

Add a direct worker-prompt test:

```python
def test_build_optimize_round_prompt_mentions_current_and_final_round(self) -> None:
    prompt = build_optimize_round_prompt(
        Path("/tmp/kernel.py"),
        Path("/tmp/opt_kernel.py"),
        test_mode="differential",
        bench_mode="standalone",
        round_mode="checked",
        current_round=2,
        final_round=4,
        round_batch_size=3,
    )

    self.assertIn("This invocation owns rounds 2 through 4.", prompt)
    self.assertIn("Execute those rounds strictly one at a time.", prompt)
    self.assertIn("Do not pre-plan the full batch before acting.", prompt)
```

Update resume-prompt tests across backend prompt-preservation suites so checked/supervised continuation prompts still preserve base context while no longer referring to `continuous`.

- [ ] **Step 2: Run the focused prompt tests to verify they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_backends_base \
  tests.test_codex_runner \
  tests.test_opencode_runner \
  tests.test_claude_runner \
  tests.test_pi_runner \
  tests.test_traecli_runner \
  -v
```

Expected: FAIL because prompt builders still split between continuous and one-round worker semantics and do not mention batch round ranges.

- [ ] **Step 3: Update optimize prompt builders for batched workers and batch supervisors**

In `src/triton_agent/optimize/prompts.py`, replace the optimize prompt split so `build_optimize_round_prompt()` accepts:

```python
def build_optimize_round_prompt(
    ...,
    round_mode: Literal["checked", "supervised"],
    current_round: int,
    final_round: int,
    round_batch_size: int,
    ...
) -> str:
```

Add prompt lines like:

```python
lines = [
    f"This invocation owns rounds {current_round} through {final_round}.",
    "Execute those rounds strictly one at a time.",
    "Do not pre-plan the full batch before acting.",
    "Before each round, re-evaluate the next bottleneck and choose the right analysis depth from the current evidence.",
    "The CLI will validate the completed batch after the invocation exits.",
]
```

Keep `build_optimize_resume_prompt()` for checked/supervised continuation, but remove any continuous-only assumptions and ensure it reinforces per-round sequential work without saying the invocation is a continuous optimize task.

Update `build_optimize_supervisor_prompt()` so it can describe the last completed batch rather than only the last completed round.

In `src/triton_agent/prompts.py`, remove optimize's dependency on `build_optimize_continuous_prompt()` and always route optimize prompt generation through the batched worker prompt path for checked/supervised.

- [ ] **Step 4: Run the focused prompt tests to verify they pass**

Run:

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_backends_base \
  tests.test_codex_runner \
  tests.test_opencode_runner \
  tests.test_claude_runner \
  tests.test_pi_runner \
  tests.test_traecli_runner \
  -v
```

Expected: PASS with batched worker wording and preserved continuation-context behavior.

- [ ] **Step 5: Commit**

```bash
git add src/triton_agent/prompts.py src/triton_agent/optimize/prompts.py tests/test_cli.py tests/test_backends_base.py tests/test_codex_runner.py tests/test_opencode_runner.py tests/test_claude_runner.py tests/test_pi_runner.py tests/test_traecli_runner.py
git commit -m "feat: add optimize batched worker prompts"
```

### Task 3: Change The Round Checker Contract From `min_rounds` To `current_round` / `final_round`

**Files:**
- Modify: `src/triton_agent/optimize/checks.py`
- Modify: `skills/triton-npu-optimize-submit-round/SKILL.md`
- Modify: `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round.py`
- Modify: `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round_contract.py`
- Test: `tests/test_optimize_checks.py`

- [ ] **Step 1: Write failing checker tests for the new round-range contract**

Replace the old `min_rounds`-driven tests in `tests/test_optimize_checks.py` with explicit batch-range tests:

```python
def test_check_round_with_remaining_batch_rounds_names_next_round_and_reflection(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        self._write_baseline(workdir)
        round_dir = self._write_round(workdir, "opt-round-2", round_disposition="continue")

        result = optimize_checks.check_round(
            round_dir,
            current_round=2,
            final_round=4,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.decision, "pass")
        self.assertEqual(result.next_option, "opt-round-3")
        self.assertIn("Round 2/4 in the current worker batch is complete", result.summary)
```

Add:

```python
def test_check_round_final_batch_round_says_batch_target_is_complete(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        self._write_baseline(workdir)
        round_dir = self._write_round(workdir, "opt-round-4", round_disposition="continue")

        result = optimize_checks.check_round(
            round_dir,
            current_round=4,
            final_round=4,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.decision, "pass")
        self.assertIsNone(result.next_option)
        self.assertIn("This round satisfied the current worker batch target.", result.summary)
```

- [ ] **Step 2: Run the focused checker tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_optimize_checks -v
```

Expected: FAIL because the checker still accepts `min_rounds` and emits session-level minimum-round text.

- [ ] **Step 3: Implement the new checker CLI and summary contract**

In `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round.py`, replace:

```python
round_parser.add_argument("--min-rounds", "--min-round", dest="min_rounds", type=int, default=None)
```

with:

```python
round_parser.add_argument("--current-round", type=int, default=None)
round_parser.add_argument("--final-round", type=int, default=None)
```

Update the call site:

```python
result = check_round(
    Path(args.round_dir).expanduser().resolve(),
    current_round=args.current_round,
    final_round=args.final_round,
    optimize_target=args.optimize_target,
)
```

In `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round_contract.py`, change `check_round()` to:

```python
def check_round(
    round_dir: Path,
    *,
    current_round: int | None = None,
    final_round: int | None = None,
    optimize_target: Literal["kernel", "operator"] | None = None,
) -> OptimizeCheckResult:
```

On pass:

```python
if current_round is not None and final_round is not None and current_round < final_round:
    next_round_name = f"opt-round-{current_round + 1}"
    result = _build_result(
        kind="round",
        decision="pass",
        issues=result.issues,
        summary=_append_pass_issues_to_summary(
            f"round check passed. Round {current_round}/{final_round} in the current worker batch is complete. "
            f"Next round: {next_round_name}. "
            "Do not rush into the next code change. "
            "First decide which operator, kernel path, or wrapper bottleneck should anchor the next round. "
            "Decide whether existing evidence is already sufficient or whether profiling, IR, or compiler-source analysis is needed first. "
            "Do not use agents or subagents to optimize multiple rounds in parallel. "
            "Do not treat the next round as a parameter-only tuning sweep. "
            "Do not use a script to create multiple optimize rounds where each round only adjusts parameters in order to speed up the optimization process. "
            "This is cheating behavior and is strictly prohibited.",
            result.issues,
        ),
        next_option=next_round_name,
    )
elif current_round is not None and final_round is not None:
    result = _build_result(
        kind="round",
        decision="pass",
        issues=result.issues,
        summary=_append_pass_issues_to_summary(
            "round check passed. This round satisfied the current worker batch target.",
            result.issues,
        ),
        next_option=None,
    )
```

Update `src/triton_agent/optimize/checks.py` to pass through the new keyword arguments and keep the wrapper types aligned.

Update `skills/triton-npu-optimize-submit-round/SKILL.md` examples and prose so it documents `--current-round` and `--final-round`.

- [ ] **Step 4: Run the checker tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_optimize_checks -v
```

Expected: PASS with batch-relative checker summaries.

- [ ] **Step 5: Run the required strict skill-script type check**

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round.py
```

Expected: PASS because AGENTS.md requires strict pyright for modified skill-side Python scripts.

- [ ] **Step 6: Commit**

```bash
git add src/triton_agent/optimize/checks.py skills/triton-npu-optimize-submit-round/SKILL.md skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round.py skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round_contract.py tests/test_optimize_checks.py
git commit -m "feat: switch optimize round checker to batch range contract"
```

### Task 4: Replace Continuous Runtime With A Batched Optimize Controller

**Files:**
- Modify: `src/triton_agent/optimize/orchestration.py`
- Modify: `src/triton_agent/optimize/execution.py`
- Modify: `src/triton_agent/optimize/run_loop.py`
- Modify: `src/triton_agent/optimize/batch.py`
- Test: `tests/test_optimize_runtime.py`
- Test: `tests/test_supervisor.py`

- [ ] **Step 1: Write failing runtime tests for batched checked and supervised flows**

Add runtime tests in `tests/test_optimize_runtime.py` such as:

```python
def test_multi_invocation_controller_checked_batch_validates_all_new_rounds(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        operator = workdir / "kernel.py"
        operator.write_text("print('x')\n", encoding="utf-8")
        self._write_baseline(workdir)
        guidance_state = self._build_checked_guidance_state(workdir)

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
            min_rounds=4,
            round_mode="checked",
            round_batch_size=2,
            optimize_role="worker",
        )
```

The fake runner should create `opt-round-1` and `opt-round-2` in one worker run, and the assertions should verify the controller launches one worker for the batch and one follow-up worker when `min_rounds` is still unmet.

Add a supervised batch test:

```python
def test_run_optimize_request_supervised_runs_one_supervisor_per_batch(self) -> None:
    ...
    self.assertEqual(
        [req.optimize_role for req in runner.requests],
        ["worker", "supervisor", "worker", "supervisor"],
    )
```

Add a regression test that `round_batch_size=1` still behaves like one worker batch plus one supervisor audit per round.

- [ ] **Step 2: Run the focused runtime tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime tests.test_supervisor -v
```

Expected: FAIL because runtime still routes `continuous` separately and checked/supervised still assume one worker per round or one supervisor per round.

- [ ] **Step 3: Implement the batched controller and remove optimize's continuous path**

In `src/triton_agent/optimize/orchestration.py`:

- remove optimize's special dispatch to `execute_continuous_optimize()`
- always route optimize through the batched multi-invocation flow
- pass `round_batch_size` into `AgentRequest`
- stop setting `optimize_role=None` based on `continuous`

In `src/triton_agent/optimize/execution.py`, replace the old round loop with a batched controller shape:

```python
def _next_batch_bounds(self, request: AgentRequest) -> tuple[int, int]:
    accepted_rounds = self._count_accepted_rounds(request.workdir)
    batch_start = accepted_rounds + 1
    batch_end = min(accepted_rounds + request.round_batch_size, cast(int, request.min_rounds))
    return batch_start, batch_end
```

When running a worker batch:

```python
round_request = replace(
    current_request,
    optimize_role="worker",
    prompt=self._build_worker_batch_prompt(current_request, batch_start, batch_end),
)
```

Validate each expected round in order:

```python
for round_number in range(batch_start, batch_end + 1):
    round_dir = request.workdir / f"opt-round-{round_number}"
    check_result = check_round(
        round_dir,
        current_round=round_number,
        final_round=batch_end,
        optimize_target=request.optimize_target,
    )
```

Advance accepted progress only through the longest passing prefix. If an intermediate round fails, keep later rounds unaccepted and include that in the continuation summary.

For supervised mode, run one supervisor pass after the CLI has built the batch follow-up summary and teach the supervisor prompt to reference the batch range.

In `src/triton_agent/optimize/run_loop.py`, keep only generic continuation-prompt helpers that are still needed. Remove optimize runtime ownership that only existed for the old continuous path.

In `src/triton_agent/optimize/batch.py`, preserve `round_batch_size` in per-workspace optimize requests.

- [ ] **Step 4: Run the focused runtime tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime tests.test_supervisor -v
```

Expected: PASS with checked and supervised batch orchestration, batch-size-one degeneration, and no optimize dependency on the old continuous path.

- [ ] **Step 5: Commit**

```bash
git add src/triton_agent/optimize/orchestration.py src/triton_agent/optimize/execution.py src/triton_agent/optimize/run_loop.py src/triton_agent/optimize/batch.py tests/test_optimize_runtime.py tests/test_supervisor.py
git commit -m "feat: batch optimize worker and supervisor orchestration"
```

### Task 5: Refresh Remaining Plumbing, Docs, And End-To-End Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/specs/2026-06-05-optimize-batched-round-mode-design.md`
- Test: `tests/test_cli.py`
- Test: `tests/test_optimize_runtime.py`
- Test: `tests/test_optimize_checks.py`
- Test: `tests/test_backends_base.py`
- Test: `tests/test_codex_runner.py`
- Test: `tests/test_opencode_runner.py`
- Test: `tests/test_claude_runner.py`
- Test: `tests/test_pi_runner.py`
- Test: `tests/test_traecli_runner.py`

- [ ] **Step 1: Update user-facing docs for the new optimize contract**

In `README.md`, update optimize examples and semantics so they show:

```text
--round-mode {checked,supervised}
--round-batch-size 10
```

and describe:

- `checked` as the default
- `supervised` as one supervisor audit per batch
- `round_batch_size=1` as the compatibility/degenerate case

- [ ] **Step 2: Run the focused regression suite**

Run:

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_optimize_runtime \
  tests.test_optimize_checks \
  tests.test_backends_base \
  tests.test_codex_runner \
  tests.test_opencode_runner \
  tests.test_claude_runner \
  tests.test_pi_runner \
  tests.test_traecli_runner \
  -v
```

Expected: PASS across CLI, prompt preservation, checker contract, and batched runtime orchestration.

- [ ] **Step 3: Run repository verification commands**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS. If failures are unrelated pre-existing issues, record them explicitly before closing the work.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/specs/2026-06-05-optimize-batched-round-mode-design.md docs/plans/2026-06-05-optimize-batched-round-mode.md
git commit -m "docs: document optimize batched round mode"
```

