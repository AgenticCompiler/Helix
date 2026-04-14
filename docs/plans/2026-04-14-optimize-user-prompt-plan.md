# Optimize User Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add additive `--prompt` support to `optimize` and `optimize-batch` so user instructions are appended to optimize worker prompts, preserved across resume/continue, and excluded from supervisor prompts.

**Architecture:** Extend optimize-only CLI parsing and option models with a `prompt` field, then centralize prompt appending in the prompt-building layer so single-workspace, batch, and resume flows all reuse one formatting rule. Keep supervisor prompt construction isolated so supervised audit passes continue using only the dedicated supervisor guidance.

**Tech Stack:** Python, `argparse`, dataclasses, existing optimize prompt/runtime helpers, Python `unittest`

---

### Task 1: Add CLI And Option Plumbing For Optimize User Prompts

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/optimize/models.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests for `--prompt`**

```python
def test_main_optimize_passes_user_prompt_to_request(self) -> None:
    captured: dict[str, object] = {}

    def _fake_build_prompt(*args, **kwargs):
        captured["base_prompt_called"] = True
        return "Prompt body"

    with patch("triton_agent.optimize.runtime.build_prompt", side_effect=_fake_build_prompt):
        ...
        exit_code = main(
            ["optimize", "-i", str(operator), "--prompt", "Focus on memory coalescing."]
        )

    self.assertEqual(exit_code, 0)
    request = mocked_supervisor_run.call_args.args[1]
    self.assertIn("Additional user instructions:", request.prompt)
    self.assertIn("Focus on memory coalescing.", request.prompt)


def test_main_optimize_batch_accepts_user_prompt(self) -> None:
    args = build_parser().parse_args(
        ["optimize-batch", "-i", str(root), "--prompt", "Avoid numerics changes."]
    )
    options = optimize_run_options_from_args(args)
    self.assertEqual(options.prompt, "Avoid numerics changes.")
```

- [ ] **Step 2: Run the targeted CLI tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.TestCLI -v`
Expected: FAIL because optimize parsers and `OptimizeRunOptions` do not yet expose `--prompt`.

- [ ] **Step 3: Add the minimal parser and option-model support**

```python
# src/triton_agent/optimize/models.py
@dataclass(frozen=True)
class OptimizeRunOptions:
    ...
    prompt: str | None


# src/triton_agent/cli.py
if spec.has_optimize_options:
    ...
    subparser.add_argument("--prompt")


# src/triton_agent/commands/optimize.py
return OptimizeRunOptions(
    ...
    prompt=getattr(args, "prompt", None),
)
```

- [ ] **Step 4: Re-run the targeted CLI tests**

Run: `uv run python -m unittest tests.test_cli.TestCLI -v`
Expected: PASS for the new `--prompt` coverage and existing optimize CLI regressions.

- [ ] **Step 5: Commit the CLI plumbing**

```bash
git add src/triton_agent/cli.py src/triton_agent/commands/optimize.py src/triton_agent/optimize/models.py tests/test_cli.py
git commit -m "feat: add optimize prompt option plumbing"
```

### Task 2: Append User Instructions To Worker Prompts And Keep Resume Behavior

**Files:**
- Modify: `src/triton_agent/prompts.py`
- Modify: `src/triton_agent/optimize/runtime.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_supervisor.py`

- [ ] **Step 1: Write failing prompt-construction tests**

```python
def test_optimize_prompt_appends_additional_user_instructions(self) -> None:
    prompt = append_additional_user_instructions(
        "Optimize the operator implementation.",
        "Prefer shared-memory reductions.",
    )
    self.assertIn("Additional user instructions:", prompt)
    self.assertIn("Prefer shared-memory reductions.", prompt)


def test_optimize_prompt_skips_blank_user_instructions(self) -> None:
    prompt = append_additional_user_instructions("Optimize the operator implementation.", "   ")
    self.assertEqual(prompt, "Optimize the operator implementation.")


def test_supervised_continue_prompt_preserves_user_instructions(self) -> None:
    base_prompt = append_additional_user_instructions(
        build_prompt(..., supervise="on"),
        "Keep launch geometry unchanged unless evidence says otherwise.",
    )
    resumed = build_optimize_resume_prompt("need one more round", base_prompt=base_prompt, supervise="on")
    self.assertIn("Additional user instructions:", resumed)
    self.assertIn("Keep launch geometry unchanged", resumed)
```

- [ ] **Step 2: Run the prompt/resume tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli tests.test_supervisor -v`
Expected: FAIL because there is no helper for additive optimize user instructions and no assertions currently pin the new resume behavior.

- [ ] **Step 3: Implement shared prompt-appending logic and use it in optimize request building**

```python
# src/triton_agent/prompts.py
def append_additional_user_instructions(prompt: str, user_prompt: str | None) -> str:
    if user_prompt is None or not user_prompt.strip():
        return prompt
    return f"{prompt}\n\nAdditional user instructions:\n{user_prompt.strip()}"


# src/triton_agent/optimize/runtime.py
prompt = append_additional_user_instructions(
    build_prompt(...),
    options.prompt,
)
```

- [ ] **Step 4: Re-run the prompt/resume tests**

Run: `uv run python -m unittest tests.test_cli tests.test_supervisor -v`
Expected: PASS with the new additive prompt block preserved in resume prompts.

- [ ] **Step 5: Commit the prompt-building change**

```bash
git add src/triton_agent/prompts.py src/triton_agent/optimize/runtime.py tests/test_cli.py tests/test_supervisor.py
git commit -m "feat: append optimize user instructions to worker prompts"
```

### Task 3: Verify Batch And Supervised Runtime Boundaries

**Files:**
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing runtime tests for batch propagation and supervisor isolation**

```python
def test_run_optimize_batch_applies_user_prompt_to_each_workspace_request(self) -> None:
    options = OptimizeRunOptions(..., prompt="Avoid changing numerics.")
    captured_prompts: list[str] = []

    def _fake_run_request(request, *_args):
        captured_prompts.append(request.prompt)
        return AgentResult(return_code=0, stdout="", stderr="")

    exit_code = run_optimize_batch(root, options, max_concurrency=1, run_request=_fake_run_request)
    self.assertEqual(exit_code, 0)
    self.assertTrue(captured_prompts)
    self.assertTrue(all("Additional user instructions:" in prompt for prompt in captured_prompts))


def test_run_optimize_request_supervisor_prompt_excludes_user_instructions(self) -> None:
    request = AgentRequest(..., prompt="...Additional user instructions:\nFocus on occupancy.", supervise="on")
    ...
    self.assertNotIn("Additional user instructions:", supervisor_request.prompt)
    self.assertNotIn("Focus on occupancy.", supervisor_request.prompt)
```

- [ ] **Step 2: Run the runtime test module to verify the new cases fail**

Run: `uv run python -m unittest tests.test_optimize_runtime -v`
Expected: FAIL until the new assertions are covered by the current request-building and supervised orchestration behavior.

- [ ] **Step 3: Adjust runtime expectations only where needed**

```python
# Prefer no production-code change here unless the new tests expose a real gap.
# If a gap appears, keep the fix inside optimize runtime/request construction
# and do not thread user prompt text into build_optimize_supervisor_prompt().
```

- [ ] **Step 4: Re-run the runtime tests**

Run: `uv run python -m unittest tests.test_optimize_runtime -v`
Expected: PASS, showing batch requests inherit the user prompt while supervisor prompts remain isolated.

- [ ] **Step 5: Commit the runtime boundary coverage**

```bash
git add tests/test_optimize_runtime.py
git commit -m "test: cover optimize user prompt runtime boundaries"
```

### Task 4: Document The New Optimize Prompt Option And Run Full Verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_cli.py`
- Test: `tests/test_supervisor.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Update README optimize examples**

```markdown
uv run triton-agent optimize --input a.py --prompt "Prioritize memory-coalescing improvements."
uv run triton-agent optimize-batch --input operators_root --prompt "Avoid changing numerics unless correctness requires it."
```

- [ ] **Step 2: Run the focused optimize verification suite**

Run: `uv run python -m unittest tests.test_cli tests.test_supervisor tests.test_optimize_runtime -v`
Expected: PASS

- [ ] **Step 3: Run lint and type checks required by the repository**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

- [ ] **Step 4: Run the full unittest suite**

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS

- [ ] **Step 5: Commit the docs and verified change set**

```bash
git add README.md tests/test_cli.py tests/test_supervisor.py tests/test_optimize_runtime.py src/triton_agent/cli.py src/triton_agent/commands/optimize.py src/triton_agent/optimize/models.py src/triton_agent/prompts.py src/triton_agent/optimize/runtime.py
git commit -m "feat: support additive optimize user prompts"
```
