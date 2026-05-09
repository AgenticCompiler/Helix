# Optimize Knowledge Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--optimize-knowledge {v1,v2}` to `optimize` and `optimize-batch`, stage the selected knowledge library before launching the agent, and preserve the staged workspace skill name `triton-npu-optimize-knowledge`.

**Architecture:** Keep the agent-facing skill contract stable by leaving optimize prompts and `staged_skill_names` unchanged, then add one narrow staging override path that maps the stable target name to an alternate repository source directory only when `v2` is selected. Thread the selected version through CLI parsing, optimize request construction, and skill staging without changing non-optimize commands.

**Tech Stack:** Python 3, `argparse`, `unittest`, existing optimize orchestration and skill staging helpers

---

## File Map

- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/optimize/orchestration.py`
- Modify: `src/triton_agent/skills.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_skills.py`

No prompt text changes are planned in `src/triton_agent/optimize/prompts.py`, because the staged skill name stays stable.

### Task 1: Add Failing CLI And Request Coverage

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write the failing parser tests for the new option**

Add CLI parser coverage near the existing optimize option tests in `tests/test_cli.py`:

```python
    def test_optimize_command_defaults_optimize_knowledge_to_v1(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertEqual(args.optimize_knowledge, "v1")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_knowledge, "v1")

    def test_optimize_command_accepts_optimize_knowledge_v2(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize", "-i", "kernel.py", "--optimize-knowledge", "v2"]
        )
        self.assertEqual(args.optimize_knowledge, "v2")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_knowledge, "v2")

    def test_optimize_batch_defaults_optimize_knowledge_to_v1(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels"])
        self.assertEqual(args.optimize_knowledge, "v1")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_knowledge, "v1")

    def test_optimize_batch_accepts_optimize_knowledge_v2(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "kernels", "--optimize-knowledge", "v2"]
        )
        self.assertEqual(args.optimize_knowledge, "v2")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_knowledge, "v2")
```

- [ ] **Step 2: Write the failing request/model tests for version plumbing**

Extend `tests/test_models.py` so `AgentRequest.with_prompt()` proves the new staging override field is preserved:

```python
        request = AgentRequest(
            command_kind=CommandKind.OPTIMIZE,
            input_path=Path("/tmp/op.py"),
            operator_path=Path("/tmp/op.py"),
            output_path=Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            interact=False,
            verbose=True,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            skill_name="triton-npu-optimize",
            prompt="original",
            workdir=Path("/tmp"),
            staged_skill_names=("triton-npu-optimize", "triton-npu-optimize-knowledge"),
            staged_skill_sources={"triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v2"},
        )

        updated = request.with_prompt("updated")

        self.assertEqual(updated.staged_skill_sources, request.staged_skill_sources)
```

Add runtime tests in `tests/test_optimize_runtime.py`:

```python
    def test_build_optimize_request_defaults_optimize_knowledge_to_v1(self) -> None:
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
            optimize_knowledge="v1",
        )
        request = build_optimize_request(operator, workdir, options)
        self.assertEqual(request.staged_skill_sources, None)

    def test_build_optimize_request_maps_v2_knowledge_to_stable_staged_name(self) -> None:
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
            optimize_knowledge="v2",
        )
        request = build_optimize_request(operator, workdir, options)
        self.assertIn("triton-npu-optimize-knowledge", request.staged_skill_names or ())
        self.assertEqual(
            request.staged_skill_sources,
            {"triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v2"},
        )
```

- [ ] **Step 3: Run the new tests and confirm they fail before implementation**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_optimize_command_defaults_optimize_knowledge_to_v1 tests.test_cli.CliParserTests.test_optimize_command_accepts_optimize_knowledge_v2 tests.test_cli.CliParserTests.test_optimize_batch_defaults_optimize_knowledge_to_v1 tests.test_cli.CliParserTests.test_optimize_batch_accepts_optimize_knowledge_v2 tests.test_models.AgentRequestTests.test_with_prompt_preserves_all_other_fields tests.test_optimize_runtime.OptimizeRuntimeTests.test_build_optimize_request_defaults_optimize_knowledge_to_v1 tests.test_optimize_runtime.OptimizeRuntimeTests.test_build_optimize_request_maps_v2_knowledge_to_stable_staged_name -v`

Expected: `FAIL` because the parser does not yet expose `--optimize-knowledge`, `OptimizeRunOptions` has no `optimize_knowledge` field, `AgentRequest` has no staging override field, and optimize requests cannot describe a `v2` source override.

### Task 2: Add Failing Skill Staging Alias Coverage

**Files:**
- Modify: `tests/test_skills.py`

- [ ] **Step 1: Write the failing skill staging alias test**

Add a focused alias test to `tests/test_skills.py`:

```python
    def test_prepare_skills_can_stage_alternate_source_under_stable_target_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "triton-npu-optimize-knowledge-v2").mkdir()
            (source / "triton-npu-optimize-knowledge-v2" / "SKILL.md").write_text(
                "v2 knowledge\n",
                encoding="utf-8",
            )

            manager = SkillLinkManager(source)
            links = manager.prepare_skills(
                "codex",
                workspace,
                skill_names=("triton-npu-optimize-knowledge",),
                skill_sources={
                    "triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v2",
                },
            )

            target = self._skills_target(workspace, "codex")
            staged_dir = target / "triton-npu-optimize-knowledge"
            self.assertTrue(staged_dir.is_dir())
            self.assertEqual(
                (staged_dir / "SKILL.md").read_text(encoding="utf-8"),
                "v2 knowledge\n",
            )
            self.assertFalse((target / "triton-npu-optimize-knowledge-v2").exists())
            manager.cleanup(links)
```

- [ ] **Step 2: Run the staging alias test and confirm it fails**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests.test_prepare_skills_can_stage_alternate_source_under_stable_target_name -v`

Expected: `FAIL` because `SkillLinkManager.prepare_skills()` does not yet accept a source override mapping and only copies from a source directory whose name exactly matches the staged target name.

### Task 3: Implement Optimize Knowledge Selection With Minimal Plumbing

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/optimize/orchestration.py`
- Modify: `src/triton_agent/skills.py`

- [ ] **Step 1: Add the CLI enum and plumb it into optimize options**

Update `src/triton_agent/cli.py`:

```python
_OPTIMIZE_KNOWLEDGE_CHOICES = ("v1", "v2")
```

Extend the optimize option block:

```python
            subparser.add_argument(
                "--optimize-knowledge",
                default="v1",
                choices=_OPTIMIZE_KNOWLEDGE_CHOICES,
            )
```

Update `src/triton_agent/optimize/models.py`:

```python
    optimize_knowledge: Literal["v1", "v2"] = "v1"
```

Update `src/triton_agent/commands/optimize.py`:

```python
        optimize_knowledge=cast(Literal["v1", "v2"], getattr(args, "optimize_knowledge", "v1")),
```

- [ ] **Step 2: Add one request field for staged skill source overrides**

Update `src/triton_agent/models.py`:

```python
    staged_skill_sources: dict[str, str] | None = None
```

Keep `with_prompt()` untouched so `dataclasses.replace()` preserves the new field automatically.

- [ ] **Step 3: Teach optimize orchestration to derive the knowledge override**

In `src/triton_agent/optimize/orchestration.py`, add a helper:

```python
_OPTIMIZE_KNOWLEDGE_TARGET = "triton-npu-optimize-knowledge"
_OPTIMIZE_KNOWLEDGE_VERSION_TO_SOURCE = {
    "v1": "triton-npu-optimize-knowledge",
    "v2": "triton-npu-optimize-knowledge-v2",
}


def _optimize_skill_sources(
    *,
    optimize_knowledge: str,
) -> dict[str, str] | None:
    source_name = _OPTIMIZE_KNOWLEDGE_VERSION_TO_SOURCE[optimize_knowledge]
    if source_name == _OPTIMIZE_KNOWLEDGE_TARGET:
        return None
    return {_OPTIMIZE_KNOWLEDGE_TARGET: source_name}
```

Pass the override into the built request:

```python
        staged_skill_sources=_optimize_skill_sources(
            optimize_knowledge=options.optimize_knowledge,
        ),
```

Leave `_BASE_OPTIMIZE_STAGED_SKILLS` and prompt strings unchanged so the staged target name remains stable.

- [ ] **Step 4: Teach `SkillLinkManager` to honor source overrides while preserving target names**

In `src/triton_agent/skills.py`, update the selected-skill iteration and copy logic:

```python
    def _iter_selected_skill_dirs(
        self,
        skill_names: tuple[str, ...] | None,
        skill_sources: dict[str, str] | None = None,
    ) -> Iterable[tuple[str, Path]]:
        if skill_names is None:
            for entry in self._iter_skill_dirs():
                yield entry.name, entry
            return

        seen: set[str] = set()
        for skill_name in skill_names:
            if skill_name in seen:
                continue
            seen.add(skill_name)
            source_name = skill_sources.get(skill_name, skill_name) if skill_sources else skill_name
            skill_dir = self.skills_root / source_name
            if not skill_dir.exists() or not skill_dir.is_dir():
                raise RuntimeError(f"Requested skill does not exist: {skill_dir}")
            yield skill_name, skill_dir
```

Then copy to the stable target name:

```python
        for staged_name, skill_dir in self._iter_selected_skill_dirs(skill_names, skill_sources):
            staged_path = target / staged_name
```

Update `prepare_skills()` to accept and forward `skill_sources`, and update the optimize call site in `run_optimize_request()`:

```python
    links = manager.prepare_skills(
        request.agent_name,
        request.workdir,
        skill_names=request.staged_skill_names,
        skill_sources=request.staged_skill_sources,
    )
```

- [ ] **Step 5: Re-run the targeted tests and make sure they all pass**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_optimize_command_defaults_optimize_knowledge_to_v1 tests.test_cli.CliParserTests.test_optimize_command_accepts_optimize_knowledge_v2 tests.test_cli.CliParserTests.test_optimize_batch_defaults_optimize_knowledge_to_v1 tests.test_cli.CliParserTests.test_optimize_batch_accepts_optimize_knowledge_v2 tests.test_models.AgentRequestTests.test_with_prompt_preserves_all_other_fields tests.test_optimize_runtime.OptimizeRuntimeTests.test_build_optimize_request_defaults_optimize_knowledge_to_v1 tests.test_optimize_runtime.OptimizeRuntimeTests.test_build_optimize_request_maps_v2_knowledge_to_stable_staged_name tests.test_skills.SkillLinkManagerTests.test_prepare_skills_can_stage_alternate_source_under_stable_target_name -v`

Expected: `OK`

### Task 4: Run Broader Verification

**Files:**
- Modify: none

- [ ] **Step 1: Run the broader optimize regression slice**

Run: `uv run python -m unittest tests.test_cli tests.test_models tests.test_optimize_runtime tests.test_skills -v`

Expected: `OK`

- [ ] **Step 2: Run repository-standard verification if the focused slice stays green**

Run: `uv run python -m unittest -v`

Expected: `OK`, or a clearly unrelated pre-existing failure that should be reported with the exact failing test name.

## Self-Review

- Spec coverage: the plan covers the CLI option, `OptimizeRunOptions` plumbing, stable staged skill naming, alias-aware skill staging, and explicit failure behavior without prompt changes.
- Placeholder scan: no `TODO` or vague “handle appropriately” steps remain; every task lists exact files and commands.
- Type consistency: the plan uses one consistent option name, `optimize_knowledge`, one stable staged target name, `triton-npu-optimize-knowledge`, and one request field name, `staged_skill_sources`, across all tasks.
