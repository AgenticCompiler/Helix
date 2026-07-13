# Optimize Resume Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `optimize --continue` with `--resume {auto,continue,fresh}` for both single-workspace and batch optimize flows, while preserving explicit failure behavior for partial optimize state.

**Architecture:** Keep resume-mode parsing, workspace classification, and effective fresh-vs-continue resolution in the CLI orchestration layer. Parse the raw user intent as a string resume mode, classify each optimize workspace before request construction, then build optimize requests from the resolved execution path so prompts, mode reuse, and batch summaries all stay consistent with the existing optimize wrapper design.

**Tech Stack:** Python `argparse`, `dataclasses`, existing optimize metadata parsers, Python `unittest`

---

### Task 1: Replace The CLI Surface With `--resume`

**Files:**
- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/optimize.py`
- Modify: `src/helix/optimize/models.py`
- Modify: `src/helix/models.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing parser and option-mapping tests**

Add tests that lock the new CLI surface:

```python
def test_optimize_command_defaults_resume_to_auto(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize", "-i", "kernel.py"])
    self.assertEqual(args.resume, "auto")

def test_optimize_command_accepts_resume_modes(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize", "-i", "kernel.py", "--resume", "fresh"])
    self.assertEqual(args.resume, "fresh")

def test_optimize_batch_accepts_resume_modes(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize-batch", "-i", "kernels", "--resume", "continue"])
    self.assertEqual(args.resume, "continue")
```

- [ ] **Step 2: Run parser tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: FAIL because `--resume` is not implemented and parser still exposes `--continue`

- [ ] **Step 3: Implement the minimal CLI/model changes**

Update parser and optimize option plumbing so optimize commands expose a string resume mode:

```python
subparser.add_argument(
    "--resume",
    default="auto",
    choices=["auto", "continue", "fresh"],
)
```

Update optimize option/request models to carry `resume_mode: str` instead of `continue_optimize: bool` in raw CLI-facing state:

```python
@dataclass(frozen=True)
class OptimizeRunOptions:
    ...
    resume_mode: str
    no_agent_session: bool
```

```python
def optimize_run_options_from_args(args: argparse.Namespace) -> OptimizeRunOptions:
    return OptimizeRunOptions(
        ...
        resume_mode=str(getattr(args, "resume", "auto")),
        no_agent_session=bool(getattr(args, "no_agent_session", False)),
        ...
    )
```

- [ ] **Step 4: Run parser tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: PASS

### Task 2: Add Workspace Classification And Resume Validation

**Files:**
- Create: `src/helix/optimize/resume.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `src/helix/optimize/validation.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing classification and validation tests**

Add focused tests for the three workspace states and mode-specific failures:

```python
def test_main_optimize_resume_auto_uses_fresh_for_no_session(self) -> None:
    ...
    exit_code = main(["optimize", "-i", str(operator), "--resume", "auto"])
    self.assertEqual(captured["resume_existing_session"], False)

def test_main_optimize_resume_auto_uses_continue_for_resumable_session(self) -> None:
    ...
    exit_code = main(["optimize", "-i", str(operator), "--resume", "auto"])
    self.assertTrue(captured["resume_existing_session"])

def test_main_optimize_resume_auto_rejects_partial_session(self) -> None:
    ...
    self.assertIn("resume auto found partial optimize state", stderr.getvalue())

def test_main_optimize_resume_fresh_rejects_existing_optimize_artifacts(self) -> None:
    ...
    self.assertIn("resume fresh refused because optimize artifacts already exist", stderr.getvalue())
```

Also replace the old `--continue` validation tests with `--resume continue` coverage and add continue-path mode-assertion validation cases (matching modes succeed, conflicting modes fail).

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests -v`
Expected: FAIL because the runtime still only understands boolean continue behavior

- [ ] **Step 3: Implement minimal classification and validation helpers**

Create a small optimize-specific helper module that classifies workspaces and resolves effective modes:

```python
@dataclass(frozen=True)
class ResumeResolution:
    workspace_state: str
    resume_existing_session: bool
    test_mode: str | None
    bench_mode: str | None
```

```python
def classify_optimize_workspace(input_path: Path, workdir: Path) -> str:
    ...
    return "no-session" | "resumable-session" | "partial-session"
```

```python
def resolve_optimize_resume(input_path: Path, workdir: Path, *, resume_mode: str) -> ResumeResolution:
    ...
```

Use this helper from `build_optimize_request`, and keep `validate_optimize_options` responsible for pure argument-level checks while the runtime handles filesystem-backed session detection.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests -v`
Expected: PASS

### Task 3: Wire Effective Continuation Through Requests And Prompts

**Files:**
- Modify: `src/helix/prompts.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `src/helix/agent.py`
- Modify: `src/helix/models.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing prompt and request tests**

Add tests that assert continuation wording depends on the resolved execution path, not just the raw CLI flag:

```python
def test_optimize_prompt_mentions_continue_mode_for_resolved_resume(self) -> None:
    prompt = build_prompt(
        CommandKind.OPTIMIZE,
        Path("/tmp/op.py"),
        Path("/tmp/op.py"),
        Path("/tmp/opt_op.py"),
        test_mode="differential",
        bench_mode="standalone",
        force_overwrite=False,
        resume_existing_session=True,
    )
    self.assertIn("Continue the existing optimization session", prompt)
```

Add request-building assertions that:
- `--resume auto` + `no-session` keeps explicit `--test-mode` / `--bench-mode`
- `--resume auto` + `resumable-session` reuses metadata and treats explicit mode flags as assertions: matching values succeed, conflicting values fail
- resume requests keep the effective continuation flag when `AgentRunner.resume()` clones the request

- [ ] **Step 2: Run the prompt/request tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.PromptTests tests.test_cli.PathResolutionTests -v`
Expected: FAIL because prompt wiring and request cloning still use `continue_optimize`

- [ ] **Step 3: Implement the minimal prompt/request changes**

Rename the prompt input away from the raw legacy flag and carry the effective continuation bit through `AgentRequest`:

```python
def build_prompt(
    ...,
    min_rounds: int | None = None,
    resume_existing_session: bool = False,
) -> str:
    ...
    if command_kind == CommandKind.OPTIMIZE and resume_existing_session:
        lines.extend([...])
```

Update request construction and agent resume cloning to preserve the effective continuation path:

```python
return AgentRequest(
    ...
    resume_existing_session=resolution.resume_existing_session,
    no_agent_session=options.no_agent_session,
)
```

- [ ] **Step 4: Run the prompt/request tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.PromptTests tests.test_cli.PathResolutionTests -v`
Expected: PASS

### Task 4: Update Batch Behavior And User-Facing Documentation

**Files:**
- Modify: `src/helix/optimize/batch.py`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/notes/2026-04-02-optimize-continue-mode.md`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing batch and documentation-facing tests**

Add batch tests that prove per-workspace resolution stays isolated:

```python
def test_optimize_batch_resume_auto_mixes_fresh_and_continue_workspaces(self) -> None:
    ...
    self.assertEqual(messages["fresh-workspace"], "optimized kernel.py")
    self.assertEqual(messages["resumed-workspace"], "optimized kernel.py")

def test_optimize_batch_resume_auto_reports_partial_session_failure(self) -> None:
    ...
    self.assertIn("resume auto found partial optimize state", rendered_summary)
```

Also update any existing help-text assertions that mention `--continue`.

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests tests.test_cli.OutputRenderingTests -v`
Expected: FAIL because batch summaries and docs still describe `--continue`

- [ ] **Step 3: Implement the minimal batch/doc changes**

Keep batch orchestration shape unchanged while letting each workspace request use the resolved resume path, then rewrite user docs to match:

```markdown
- `optimize` accepts `--resume {auto,continue,fresh}`.
- `optimize-batch` applies the same resume-mode validation per workspace.
- `resume auto` continues complete sessions, starts fresh in empty workspaces, and fails for partial optimize state.
```

Update `AGENTS.md` only where the durable rule changed from `--continue` semantics to explicit resume-mode semantics.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests tests.test_cli.OutputRenderingTests -v`
Expected: PASS

### Task 5: Run Repository Verification

**Files:**
- Modify: none
- Test: repository-wide verification commands

- [ ] **Step 1: Run the full CLI test module**

Run: `uv run python -m unittest tests.test_cli -v`
Expected: PASS

- [ ] **Step 2: Run lint**

Run: `uv run --group dev ruff check`
Expected: PASS

- [ ] **Step 3: Run static typing**

Run: `uv run pyright`
Expected: PASS

- [ ] **Step 4: Run the full test suite**

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
