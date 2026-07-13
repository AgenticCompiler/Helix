# Compiler Source Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in compiler-source analysis capability that lets optimize runs use a CLI-provisioned, read-only AscendNPU-IR checkout as evidence when profile and IR analysis are not enough.

**Architecture:** Keep compiler source provisioning in the CLI/runtime layer and compiler-source reasoning in a dedicated skill. Add CLI options and request fields for the feature, implement a focused provisioning module that shallow-clones or validates the local checkout, then pass the resolved path/commit/dirty state into optimize prompts and guidance. Optimize already stages the full repository skill set, so compiler source options control provisioning and prompt authorization, not skill visibility.

**Tech Stack:** Python 3.12, `argparse`, `dataclasses`, `pathlib`, `subprocess`, existing optimize prompt/guidance plumbing, Markdown skills and docs, Python `unittest`

---

### Task 1: Lock In CLI Option Semantics And Data Model

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_commands.py`
- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/optimize.py`
- Modify: `src/helix/optimize/models.py`
- Modify: `src/helix/models.py`

- [ ] **Step 1: Write failing parser tests for compiler source options**

Add tests in `tests/test_cli.py`:

```python
def test_optimize_accepts_compiler_source_analysis_options(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "optimize",
            "-i",
            "kernel.py",
            "--enable-compiler-source-analysis",
            "--compiler-source-path",
            "/tmp/AscendNPU-IR",
        ]
    )

    self.assertTrue(args.enable_compiler_source_analysis)
    self.assertEqual(args.compiler_source_path, "/tmp/AscendNPU-IR")


def test_optimize_batch_accepts_compiler_source_analysis_options(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "optimize-batch",
            "-i",
            "operators",
            "--enable-compiler-source-analysis",
            "--compiler-source-path",
            "/tmp/AscendNPU-IR",
        ]
    )

    self.assertTrue(args.enable_compiler_source_analysis)
    self.assertEqual(args.compiler_source_path, "/tmp/AscendNPU-IR")
```

- [ ] **Step 2: Write failing command option mapping tests**

Add tests in `tests/test_optimize_commands.py` or an existing option-mapping test class:

```python
def test_optimize_run_options_maps_compiler_source_analysis(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "optimize",
            "-i",
            "kernel.py",
            "--enable-compiler-source-analysis",
            "--compiler-source-path",
            "/tmp/AscendNPU-IR",
        ]
    )

    options = optimize_run_options_from_args(args)

    self.assertEqual(options.compiler_source_analysis, "auto")
    self.assertEqual(options.compiler_source_path, "/tmp/AscendNPU-IR")
```

- [ ] **Step 3: Write failing validation test for path without enable flag**

Add a test that calls `handle_optimize()` with `--compiler-source-path` but without `--enable-compiler-source-analysis` and expects `SystemExit(2)` with a parser error. Patch agent execution if needed so the test stops at validation.

- [ ] **Step 4: Run targeted tests and confirm failure**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_optimize_commands.OptimizeCommandHandlerTests -v`

Expected: FAIL because the parser and option model do not expose compiler-source options yet.

- [ ] **Step 5: Add parser arguments**

In `src/helix/cli.py`, inside the `spec.has_optimize_options` block, add:

```python
subparser.add_argument("--enable-compiler-source-analysis", action="store_true")
subparser.add_argument("--compiler-source-path")
```

- [ ] **Step 6: Extend optimize option models**

In `src/helix/optimize/models.py`, extend `OptimizeRunOptions`:

```python
compiler_source_analysis: Literal["off", "auto"] = "off"
compiler_source_path: str | None = None
```

In `src/helix/models.py`, extend `AgentRequest`:

```python
compiler_source_analysis: Literal["off", "auto"] = "off"
compiler_source_path: Optional[Path] = None
compiler_source_commit: Optional[str] = None
compiler_source_dirty: Optional[bool] = None
```

- [ ] **Step 7: Map and validate CLI args**

In `optimize_run_options_from_args()`, map:

```python
compiler_source_enabled = bool(getattr(args, "enable_compiler_source_analysis", False))
compiler_source_path = getattr(args, "compiler_source_path", None)
```

Return `compiler_source_analysis="auto"` when enabled, otherwise `"off"`.

In `handle_optimize()` and `handle_optimize_batch()`, reject `--compiler-source-path` unless compiler-source analysis is enabled. Prefer adding this to `validate_optimize_options()` only if that function already has the right argument context; otherwise keep it local in `commands/optimize.py`.

- [ ] **Step 8: Re-run targeted tests**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_optimize_commands.OptimizeCommandHandlerTests -v`

Expected: PASS for the new CLI/model tests.

### Task 2: Implement Compiler Source Provisioning

**Files:**
- Create: `src/helix/optimize/compiler_source.py`
- Create: `tests/test_compiler_source.py`
- Modify: `src/helix/optimize/orchestration.py`

- [ ] **Step 1: Write failing tests for default path resolution**

Create `tests/test_compiler_source.py` with tests for a fake home directory. Avoid touching the real home directory.

```python
def test_default_compiler_source_path_uses_helix_home(self) -> None:
    root = Path("/tmp/fake-home")

    path = default_compiler_source_path(root)

    self.assertEqual(path, root / "compiler-sources" / "AscendNPU-IR")
```

- [ ] **Step 2: Write failing tests for missing checkout clone command**

Patch the command runner so no network is used:

```python
def test_prepare_clones_missing_default_checkout_depth_one(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / ".helix"
        calls: list[list[str]] = []

        def fake_run(args: list[str], cwd: Path | None = None) -> str:
            calls.append(args)
            if args[:2] == ["git", "clone"]:
                target = Path(args[-1])
                target.mkdir(parents=True)
                (target / ".git").mkdir()
                return ""
            if args == ["git", "rev-parse", "HEAD"]:
                return "abc123\n"
            if args == ["git", "status", "--porcelain"]:
                return ""
            raise AssertionError(args)

        result = prepare_compiler_source(
            mode="auto",
            source_path=None,
            helix_home=home,
            run_git=fake_run,
        )

        self.assertEqual(result.path, home / "compiler-sources" / "AscendNPU-IR")
        self.assertEqual(result.commit, "abc123")
        self.assertFalse(result.dirty)
        self.assertIn(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://gitcode.com/Ascend/AscendNPU-IR.git",
                str(home / "compiler-sources" / "AscendNPU-IR"),
            ],
            calls,
        )
```

- [ ] **Step 3: Write failing tests for existing checkout reuse**

Create an existing directory with `.git/`, patch `run_git`, and assert:

- no `git clone`
- no `git fetch`
- no `git pull`
- commit and dirty state are inspected

- [ ] **Step 4: Write failing tests for invalid existing path**

Cover:

- path exists but is a file
- path exists but has no `.git`
- explicit `--compiler-source-path` points to a missing path

Expected: `ValueError` with short actionable messages.

- [ ] **Step 5: Run provisioning tests and confirm failure**

Run: `uv run python -m unittest tests.test_compiler_source -v`

Expected: FAIL because the module does not exist yet.

- [ ] **Step 6: Implement the provisioning module**

Add `src/helix/optimize/compiler_source.py`:

```python
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

COMPILER_SOURCE_REPO_URL = "https://gitcode.com/Ascend/AscendNPU-IR.git"
COMPILER_SOURCE_DIR_NAME = "AscendNPU-IR"
CompilerSourceMode = Literal["off", "auto"]


@dataclass(frozen=True)
class CompilerSourceInfo:
    path: Path
    commit: str
    dirty: bool


RunGit = Callable[[list[str], Path | None], str]
```

Implement:

- `helix_home() -> Path`
- `default_compiler_source_path(home: Path | None = None) -> Path`
- `prepare_compiler_source(...) -> CompilerSourceInfo | None`
- `_run_git(args: list[str], cwd: Path | None) -> str`
- `_validate_git_checkout(path: Path) -> None`
- `_inspect_commit(path: Path, run_git: RunGit) -> str`
- `_inspect_dirty(path: Path, run_git: RunGit) -> bool`

Use `subprocess.run(..., check=False, capture_output=True, text=True)` and convert failures to `ValueError` with concise messages.

- [ ] **Step 7: Re-run provisioning tests**

Run: `uv run python -m unittest tests.test_compiler_source -v`

Expected: PASS.

### Task 3: Thread Provisioning Through Optimize Request Construction

**Files:**
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_optimize_batch.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `src/helix/optimize/batch.py` if needed

- [ ] **Step 1: Write failing request-building tests**

Add tests that patch `prepare_compiler_source()` inside `helix.optimize.orchestration` and assert:

- disabled mode does not provision source
- enabled mode provisions source before building the prompt
- request fields carry `compiler_source_path`, `compiler_source_commit`, and `compiler_source_dirty`

Example:

```python
with patch(
    "helix.optimize.orchestration.prepare_compiler_source",
    return_value=CompilerSourceInfo(path=source_path, commit="abc123", dirty=False),
) as mocked:
    request = build_optimize_request(operator, workdir, options)

mocked.assert_called_once()
self.assertEqual(request.compiler_source_path, source_path)
self.assertEqual(request.compiler_source_commit, "abc123")
self.assertFalse(request.compiler_source_dirty)
```

- [ ] **Step 2: Write failing optimize-batch plumbing test**

Add a test that runs `run_optimize_batch()` with a fake `run_request` and options containing `compiler_source_analysis="auto"` plus an explicit compiler source path. Assert each generated request carries the same compiler-source mode and resolved source info.

Patch provisioning to avoid clone and to make the test deterministic.

- [ ] **Step 3: Run targeted tests and confirm failure**

Run: `uv run python -m unittest tests.test_optimize_runtime tests.test_optimize_batch -v`

Expected: FAIL because request construction does not provision or carry compiler source metadata yet.

- [ ] **Step 4: Implement request construction plumbing**

In `src/helix/optimize/orchestration.py`:

- import `prepare_compiler_source`
- call it in `build_optimize_request()` after resume resolution and before `build_prompt()`
- convert `options.compiler_source_path` to an expanded resolved `Path` only when present
- pass resolved compiler-source fields into `build_prompt()`
- set the same fields on `AgentRequest`

Do not alter `staged_skill_names=None`; optimize should continue staging all repository skills.

- [ ] **Step 5: Re-run targeted tests**

Run: `uv run python -m unittest tests.test_optimize_runtime tests.test_optimize_batch -v`

Expected: PASS for the new provisioning plumbing tests.

### Task 4: Add Prompt, Resume Prompt, And Guidance Text

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `src/helix/prompts.py`
- Modify: `src/helix/optimize/guidance.py`
- Modify: `src/helix/optimize/execution.py`
- Modify: `src/helix/optimize/run_loop.py`

- [ ] **Step 1: Write failing prompt tests**

Add tests for `build_prompt()`, `build_optimize_worker_prompt()`, and `build_optimize_unsupervised_prompt()` that assert enabled compiler-source analysis includes:

- `Compiler source analysis is enabled`
- the local source path
- the source commit
- read-only checkout rules
- no `https://gitcode.com/Ascend/AscendNPU-IR.git` URL
- evidence ladder language that places compiler source after benchmark/profile/IR

- [ ] **Step 2: Write failing guidance tests**

In `tests/test_optimize_guidance.py`, add unsupervised and supervised guidance tests that pass compiler-source metadata and assert the guidance contains the path, commit, read-only rules, and no repo URL.

- [ ] **Step 3: Write failing resume prompt test**

Add or update tests covering resume prompts so resumed worker/unsupervised prompts preserve compiler-source instructions when the request has compiler-source metadata.

- [ ] **Step 4: Run targeted tests and confirm failure**

Run: `uv run python -m unittest tests.test_cli tests.test_optimize_guidance tests.test_optimize_runtime -v`

Expected: FAIL because prompt and guidance functions do not accept compiler-source metadata.

- [ ] **Step 5: Add a small prompt formatter**

In `src/helix/prompts.py`, add a helper such as:

```python
def compiler_source_analysis_lines(
    *,
    compiler_source_path: Path | None,
    compiler_source_commit: str | None,
    compiler_source_dirty: bool | None,
) -> list[str]:
    if compiler_source_path is None or compiler_source_commit is None:
        return []
    dirty_text = "dirty" if compiler_source_dirty else "clean"
    return [
        "Compiler source analysis is enabled for this optimize run.",
        f"Compiler source path: {compiler_source_path}",
        f"Compiler source commit: {compiler_source_commit} ({dirty_text}).",
        "Treat the compiler source checkout as read-only.",
        "Do not run git clone, git fetch, git pull, or modify files in the compiler source checkout.",
        "Use the staged `triton-npu-analyze-compiler-source` skill only when compiler source evidence is needed.",
        "Prefer the evidence ladder first: benchmark and correctness results, then profiler evidence, then IR evidence, then compiler source.",
    ]
```

Thread optional compiler-source parameters into:

- `build_prompt()`
- `build_optimize_worker_prompt()`
- `build_optimize_unsupervised_prompt()`
- `build_optimize_resume_prompt()`
- any helper that constructs resumed worker prompts

- [ ] **Step 6: Update guidance rendering**

Add an optional `compiler_source_info` argument or explicit fields to:

- `OptimizeGuidanceManager.prepare_unsupervised_session()`
- `OptimizeGuidanceManager.prepare_supervised_session()`
- `_render_unsupervised_guidance()`
- `_render_shared_guidance()`

Append the same read-only local-path instructions when compiler-source metadata is present.

- [ ] **Step 7: Thread request fields into execution guidance**

Update `src/helix/optimize/execution.py` and `src/helix/optimize/run_loop.py` so supervised and unsupervised session preparation receives request compiler-source metadata.

- [ ] **Step 8: Re-run targeted tests**

Run: `uv run python -m unittest tests.test_cli tests.test_optimize_guidance tests.test_optimize_runtime -v`

Expected: PASS for prompt and guidance tests.

### Task 5: Add The Compiler Source Analysis Skill

**Files:**
- Create: `skills/triton/triton-npu-analyze-compiler-source/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/references/workflow.md`
- Modify: `skills/triton-npu-analyze-round-performance/SKILL.md`
- Modify: `tests/test_skills.py` if skill discovery has explicit expectations

- [ ] **Step 1: Write or update skill discovery test if needed**

If existing skill tests enumerate expected skill names, add `triton-npu-analyze-compiler-source`. If they only test generic staging, no test change is required.

- [ ] **Step 2: Create the skill**

Add `skills/triton/triton-npu-analyze-compiler-source/SKILL.md` with front matter:

```markdown
---
name: triton-npu-analyze-compiler-source
description: Use when an optimize round has CLI-provisioned AscendNPU-IR compiler source and needs source-backed explanation for a compiler error, suspicious IR pass transition, lowering symptom, or stalled optimization direction that profile and IR evidence alone cannot explain.
---
```

Include sections:

- Overview
- Inputs
- When To Use
- When Not To Use
- Working Rules
- Analysis Workflow
- Output Contract
- Evidence Quality Rules

The skill must explicitly say:

- use only the CLI-provided local source path
- do not clone, fetch, pull, or modify compiler source
- cite source paths and commit
- treat version mismatch as an evidence gap
- write `opt-round-N/compiler-analysis.md` when used
- detailed index/helper behavior is deferred

- [ ] **Step 3: Update optimize skill references**

In `skills/triton/triton-npu-optimize/SKILL.md` and `references/workflow.md`, add compiler source analysis as an escalation after profiler and IR evidence. Keep this conditional on the feature being enabled by the launch prompt/guidance.

- [ ] **Step 4: Update round performance analysis skill**

In `skills/triton-npu-analyze-round-performance/SKILL.md`, mention that compiler source analysis may be used after profile and IR evidence when source-level explanation is needed and the launch prompt says it is enabled.

- [ ] **Step 5: Run skill-related tests**

Run: `uv run python -m unittest tests.test_skills -v`

Expected: PASS.

### Task 6: Update README And Finish Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/specs/2026-04-21-compiler-source-analysis-design.md` only if implementation details force a spec correction
- Modify: `docs/plans/2026-04-21-compiler-source-analysis.md`

- [ ] **Step 1: Update README optimize options**

Document:

- `--enable-compiler-source-analysis`
- `--compiler-source-path <path>`
- the default cache path under `~/.helix/compiler-sources/AscendNPU-IR/`
- that the CLI provisions the checkout and agents must treat it as read-only
- that the option enables escalation, not mandatory compiler-source analysis every round

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_optimize_commands \
  tests.test_compiler_source \
  tests.test_optimize_guidance \
  tests.test_optimize_runtime \
  tests.test_optimize_batch \
  tests.test_skills \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run lint**

Run: `uv run --group dev ruff check`

Expected: PASS.

- [ ] **Step 4: Run type checking**

Run: `uv run pyright`

Expected: PASS.

- [ ] **Step 5: Run the full unit suite**

Run: `uv run python -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 6: Commit when verification passes**

Run:

```bash
git add \
  src/helix \
  tests \
  skills \
  README.md \
  docs/specs/2026-04-21-compiler-source-analysis-design.md \
  docs/plans/2026-04-21-compiler-source-analysis.md
git commit -m "feat: add compiler source analysis option"
```
