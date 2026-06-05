# Run-Eval MCP Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--enable-mcp` for agent-backed run-eval workflows so the CLI can switch between the legacy script-backed `triton-npu-run-eval` skill and an MCP-backed skill source staged under the same visible skill name.

**Architecture:** Thread an explicit `enable_mcp` flag from CLI parsing through options, staging, request construction, and batch/orchestration control flow. Keep the staged skill name stable as `triton-npu-run-eval`, switch its source directory with `skill_sources`, and let the shared HTTP MCP server own NPU slot arbitration only when MCP mode is enabled.

**Tech Stack:** Python 3, argparse, dataclasses, unittest/pytest, FastMCP, existing backend MCP config emitters.

---

## File Structure

### CLI and option propagation

- Modify: `src/triton_agent/cli.py`
  - add `--enable-mcp` only to `gen-eval`, `gen-eval-batch`, `convert`, `convert-batch`, `optimize`, and `optimize-batch`
- Modify: `src/triton_agent/generation/models.py`
  - add `enable_mcp` to `GenerationOptions`
- Modify: `src/triton_agent/convert/models.py`
  - add `enable_mcp` to `ConvertOptions`
- Modify: `src/triton_agent/optimize/models.py`
  - add `enable_mcp` to `OptimizeRunOptions`
- Modify: `src/triton_agent/models.py`
  - add `enable_mcp` to `AgentRequest`
- Modify: `src/triton_agent/commands/generation.py`
  - populate `GenerationOptions.enable_mcp`
- Modify: `src/triton_agent/commands/convert.py`
  - populate `ConvertOptions.enable_mcp`
- Modify: `src/triton_agent/commands/optimize.py`
  - populate `OptimizeRunOptions.enable_mcp`

### Staging and orchestration

- Modify: `src/triton_agent/skill_staging.py`
  - make run-eval staging switch source directory when MCP mode is enabled
- Modify: `src/triton_agent/mcp.py`
  - derive managed MCP server names only when both run-eval is staged and `enable_mcp` is true
- Modify: `src/triton_agent/generation/orchestration.py`
  - pass `enable_mcp` into staging/MCP helpers and stage `skill_sources`
- Modify: `src/triton_agent/convert/orchestration.py`
  - same as generation
- Modify: `src/triton_agent/optimize/orchestration.py`
  - same as optimize path, preserving existing optimize-specific staging overrides

### Batch behavior

- Modify: `src/triton_agent/generation/batch.py`
  - skip batch affinity capacity validation in MCP mode
- Modify: `src/triton_agent/convert/batch.py`
  - same as generation batch
- Modify: `src/triton_agent/optimize/batch.py`
  - same as optimize batch

### MCP skill content

- Modify: `skills/triton-npu-run-eval-mcp/SKILL.md`
  - make it tool-first and remove script guidance / `compare-result`
- Modify: `skills/triton-npu-run-eval-mcp/references/run-test.md`
  - explain `run-test-baseline` and `run-test-optimize` as MCP tools
- Modify: `skills/triton-npu-run-eval-mcp/references/run-bench.md`
  - explain `run-bench` as an MCP tool
- Modify: `skills/triton-npu-run-eval-mcp/references/profile-bench.md`
  - explain `profile-bench` as an MCP tool
- Modify: `skills/triton-npu-run-eval-mcp/references/profile-report.md`
  - explain `profile-report` as an MCP tool
- Modify: `skills/triton-npu-run-eval-mcp/references/compare-perf.md`
  - explain `compare-perf` as an MCP tool
- Delete: `skills/triton-npu-run-eval-mcp/references/compare-result.md`

### MCP server tools

- Modify: `src/triton_agent/run_eval_mcp_server.py`
  - add `profile-report` and `compare-perf`
  - ensure only device-bound tools lease NPU slots

### Tests

- Modify: `tests/test_cli.py`
- Modify: `tests/test_skill_staging.py`
- Modify: `tests/test_generation_commands.py`
- Modify: `tests/test_convert_commands.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_generation_batch.py`
- Modify: `tests/test_run_eval_mcp_server.py`

### Documentation

- Modify: `docs/specs/2026-06-04-run-eval-mcp-server-design.md`
  - align the older MCP design doc with the new toggle-based rollout if needed

### Verification

- Run: `uv run --group dev ruff check`
- Run: `uv run pyright`
- Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

### Task 1: Add CLI and model support for `--enable-mcp`

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/generation/models.py`
- Modify: `src/triton_agent/convert/models.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/commands/generation.py`
- Modify: `src/triton_agent/commands/convert.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI parser tests for the new option**

```python
def test_enable_mcp_is_available_on_agent_backed_commands(self) -> None:
    parser = build_parser()
    for command, input_value in (
        ("gen-eval", "kernel.py"),
        ("gen-eval-batch", "kernels"),
        ("convert", "kernel.py"),
        ("convert-batch", "kernels"),
        ("optimize", "kernel.py"),
        ("optimize-batch", "kernels"),
    ):
        with self.subTest(command=command):
            args = parser.parse_args([command, "-i", input_value, "--enable-mcp"])
            self.assertTrue(args.enable_mcp)


def test_enable_mcp_is_not_available_on_non_agent_run_eval_commands(self) -> None:
    parser = build_parser()
    stderr = StringIO()
    with self.assertRaises(SystemExit) as exc, redirect_stderr(stderr):
        parser.parse_args(["run-bench", "--bench-file", "bench.py", "--operator-file", "kernel.py", "--enable-mcp"])
    self.assertEqual(exc.exception.code, 2)
    self.assertIn("--enable-mcp", stderr.getvalue())
```

- [ ] **Step 2: Run the focused CLI tests and verify they fail first**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py -k enable_mcp`

Expected: FAIL because the parser does not define `--enable-mcp` yet.

- [ ] **Step 3: Add `enable_mcp` to the option dataclasses and CLI option parsing**

```python
@dataclass(frozen=True)
class GenerationOptions:
    interact: bool
    verbose: bool
    show_output: bool
    force_overwrite: bool
    agent_name: str
    remote: str | None
    remote_workdir: str | None
    min_rounds: int | None
    continue_optimize: bool
    output: str | None
    test_mode: str | None
    bench_mode: str | None
    prompt: str | None = None
    log_tools: bool = False
    enable_mcp: bool = False
```

```python
return GenerationOptions(
    interact=bool(getattr(args, "interact", False)),
    verbose=bool(getattr(args, "verbose", False)),
    show_output=bool(getattr(args, "show_output", False)),
    force_overwrite=bool(getattr(args, "force_overwrite", False)),
    agent_name=args.agent,
    remote=getattr(args, "remote", None),
    remote_workdir=getattr(args, "remote_workdir", None),
    min_rounds=getattr(args, "min_rounds", None),
    continue_optimize=bool(getattr(args, "continue_optimize", False)),
    output=getattr(args, "output", None),
    test_mode=getattr(args, "test_mode", None),
    bench_mode=getattr(args, "bench_mode", None),
    prompt=getattr(args, "prompt", None),
    log_tools=bool(getattr(args, "log_tools", False)),
    enable_mcp=bool(getattr(args, "enable_mcp", False)),
)
```

```python
if spec.has_agent and command_kind in {
    CommandKind.GEN_EVAL,
    CommandKind.GEN_EVAL_BATCH,
    CommandKind.CONVERT,
    CommandKind.CONVERT_BATCH,
    CommandKind.OPTIMIZE,
    CommandKind.OPTIMIZE_BATCH,
}:
    parser.add_argument(
        "--enable-mcp",
        action="store_true",
        help="Stage the MCP-backed run-eval skill and configure request-scoped MCP servers.",
    )
```

- [ ] **Step 4: Add `enable_mcp` to `AgentRequest` and populate it from command helpers**

```python
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
    remote: Optional[str] = None
    remote_workdir: Optional[str] = None
    extra_env: dict[str, str] | None = None
    min_rounds: Optional[int] = None
    continue_optimize: bool = False
    no_agent_session: bool = False
    round_mode: Literal["continuous", "checked", "supervised"] = "continuous"
    staged_skill_names: tuple[str, ...] | None = None
    staged_skill_sources: dict[str, str] | None = None
    optimize_role: str | None = None
    supervisor_report_path: Optional[Path] = None
    target_chip: Literal["A3", "A5"] = "A5"
    optimize_target: Literal["kernel", "operator"] = "kernel"
    compiler_source_analysis: Literal["off", "auto"] = "off"
    compiler_source_path: Optional[Path] = None
    compiler_source_commit: Optional[str] = None
    enable_subagent: bool = False
    enable_agent_hooks: bool = False
    log_tools: bool = False
    enable_mcp: bool = False
    mcp_servers: tuple[str, ...] | None = None
    show_output_label: str = ""
    run_id: str = ""
```

- [ ] **Step 5: Run the focused CLI tests and option-construction tests**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py tests/test_generation_commands.py tests/test_convert_commands.py -k enable_mcp`

Expected: PASS for parser coverage, with later orchestration-specific tests still failing until the next task.

- [ ] **Step 6: Commit the CLI/model groundwork**

```bash
git add src/triton_agent/cli.py src/triton_agent/generation/models.py src/triton_agent/convert/models.py src/triton_agent/optimize/models.py src/triton_agent/models.py src/triton_agent/commands/generation.py src/triton_agent/commands/convert.py src/triton_agent/commands/optimize.py tests/test_cli.py
git commit -m "feat: add run-eval mcp toggle option"
```

### Task 2: Switch run-eval skill staging by source while keeping a stable staged name

**Files:**
- Modify: `src/triton_agent/skill_staging.py`
- Modify: `src/triton_agent/generation/orchestration.py`
- Modify: `src/triton_agent/convert/orchestration.py`
- Modify: `src/triton_agent/optimize/orchestration.py`
- Test: `tests/test_skill_staging.py`
- Test: `tests/test_generation_commands.py`
- Test: `tests/test_convert_commands.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing staging tests for MCP source switching**

```python
def test_resolve_staged_skills_for_gen_eval_uses_mcp_source_when_enabled(self) -> None:
    names, sources = resolve_staged_skills(CommandKind.GEN_EVAL, enable_mcp=True)
    self.assertIn("triton-npu-run-eval", names or ())
    self.assertEqual(sources, {"triton-npu-run-eval": "triton-npu-run-eval-mcp"})


def test_resolve_staged_skills_for_gen_eval_keeps_legacy_source_by_default(self) -> None:
    names, sources = resolve_staged_skills(CommandKind.GEN_EVAL)
    self.assertIn("triton-npu-run-eval", names or ())
    self.assertIsNone(sources)
```

- [ ] **Step 2: Run the focused staging tests and verify they fail first**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_staging.py -k mcp_source`

Expected: FAIL because `resolve_staged_skills(...)` does not accept `enable_mcp` yet.

- [ ] **Step 3: Extend `resolve_staged_skills(...)` to support MCP source overrides**

```python
def resolve_staged_skills(
    command_kind: CommandKind,
    *,
    optimize_knowledge: str | None = None,
    optimize_target: str = "kernel",
    enable_cann_ext_api: bool = False,
    enable_mcp: bool = False,
) -> tuple[tuple[str, ...] | None, dict[str, str] | None]:
    rule = STAGE_RULES.get(command_kind)
    if rule is None:
        return None, None

    staged_skill_names = _apply_stage_directives(rule.directives)
    if (
        enable_mcp
        and staged_skill_names is not None
        and "triton-npu-run-eval" in staged_skill_names
    ):
        staged_skill_sources = {"triton-npu-run-eval": "triton-npu-run-eval-mcp"}
    else:
        staged_skill_sources = None
```

- [ ] **Step 4: Thread `enable_mcp` into request building and pass `skill_sources` when staging skills**

```python
staged_skill_names, staged_skill_sources = resolve_staged_skills(
    command_kind,
    enable_mcp=options.enable_mcp,
)
```

```python
links = manager.prepare_skills(
    request.agent_name,
    request.workdir,
    skill_names=request.staged_skill_names,
    skill_sources=request.staged_skill_sources,
)
```

- [ ] **Step 5: Update request-builder tests to cover the new staged source behavior**

```python
request = build_convert_request(
    Path("/tmp/kernel.py"),
    Path("/tmp/kernel.py"),
    Path("/tmp"),
    ConvertOptions(
        interact=False,
        verbose=False,
        show_output=False,
        force_overwrite=False,
        agent_name="codex",
        remote=None,
        remote_workdir=None,
        output=None,
        test_mode="differential",
        prompt=None,
        enable_mcp=True,
    ),
)
self.assertEqual(request.staged_skill_sources, {"triton-npu-run-eval": "triton-npu-run-eval-mcp"})
```

- [ ] **Step 6: Run the focused staging/request tests**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_staging.py tests/test_generation_commands.py tests/test_convert_commands.py tests/test_optimize_runtime.py -k mcp`

Expected: PASS for staging overrides and request propagation, with MCP server selection details still to be completed in the next task.

- [ ] **Step 7: Commit the staging-source switch**

```bash
git add src/triton_agent/skill_staging.py src/triton_agent/generation/orchestration.py src/triton_agent/convert/orchestration.py src/triton_agent/optimize/orchestration.py tests/test_skill_staging.py tests/test_generation_commands.py tests/test_convert_commands.py tests/test_optimize_runtime.py
git commit -m "feat: stage run-eval mcp skill behind toggle"
```

### Task 3: Gate managed MCP activation explicitly on `enable_mcp`

**Files:**
- Modify: `src/triton_agent/mcp.py`
- Modify: `src/triton_agent/generation/orchestration.py`
- Modify: `src/triton_agent/convert/orchestration.py`
- Modify: `src/triton_agent/optimize/orchestration.py`
- Test: `tests/test_generation_commands.py`
- Test: `tests/test_convert_commands.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing tests that non-MCP requests no longer attach `mcp_servers`**

```python
def test_build_generation_request_for_gen_test_omits_mcp_servers_by_default(self) -> None:
    request = build_generation_request(
        CommandKind.GEN_TEST,
        Path("/tmp/kernel.py"),
        Path("/tmp/kernel.py"),
        Path("/tmp"),
        GenerationOptions(
            interact=False,
            verbose=False,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            remote=None,
            remote_workdir=None,
            min_rounds=None,
            continue_optimize=False,
            output=None,
            test_mode="standalone",
            bench_mode=None,
            prompt=None,
            enable_mcp=False,
        ),
    )
    self.assertIsNone(request.mcp_servers)
```

- [ ] **Step 2: Run the focused request tests and verify they fail first**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_generation_commands.py tests/test_convert_commands.py tests/test_optimize_runtime.py -k mcp_servers`

Expected: FAIL because current request builders still attach MCP servers whenever run-eval is staged.

- [ ] **Step 3: Replace implicit skill-only MCP activation with explicit request mode checks**

```python
def managed_mcp_server_names_for_request(
    staged_skill_names: tuple[str, ...] | None,
    *,
    enable_mcp: bool,
) -> tuple[str, ...] | None:
    if not enable_mcp or staged_skill_names is None:
        return None
    if "triton-npu-run-eval" not in staged_skill_names:
        return None
    return (RUN_EVAL_MCP_SERVER_NAME,)
```

```python
mcp_servers = managed_mcp_server_names_for_request(
    staged_skill_names,
    enable_mcp=options.enable_mcp,
)
```

- [ ] **Step 4: Store `enable_mcp` in `AgentRequest` and use it consistently in orchestration**

```python
return AgentRequest(
    command_kind=command_kind,
    input_path=input_path,
    operator_path=operator_path,
    output_path=output_path,
    test_mode=options.test_mode,
    bench_mode=options.bench_mode,
    interact=options.interact,
    verbose=options.verbose,
    show_output=options.show_output,
    force_overwrite=options.force_overwrite,
    agent_name=options.agent_name,
    skill_name=COMMAND_TO_SKILL[command_kind],
    prompt=prompt,
    workdir=workdir,
    enable_mcp=options.enable_mcp,
    staged_skill_names=staged_skill_names,
    staged_skill_sources=staged_skill_sources,
    mcp_servers=mcp_servers,
)
```

- [ ] **Step 5: Run the focused MCP-gating tests**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_generation_commands.py tests/test_convert_commands.py tests/test_optimize_runtime.py -k "mcp_servers or enable_mcp"`

Expected: PASS with MCP servers attached only when the toggle is enabled.

- [ ] **Step 6: Commit the explicit MCP activation gate**

```bash
git add src/triton_agent/mcp.py src/triton_agent/generation/orchestration.py src/triton_agent/convert/orchestration.py src/triton_agent/optimize/orchestration.py tests/test_generation_commands.py tests/test_convert_commands.py tests/test_optimize_runtime.py
git commit -m "feat: gate managed run-eval mcp on explicit toggle"
```

### Task 4: Preserve legacy batch validation and relax it in MCP mode

**Files:**
- Modify: `src/triton_agent/generation/batch.py`
- Modify: `src/triton_agent/convert/batch.py`
- Modify: `src/triton_agent/optimize/batch.py`
- Test: `tests/test_generation_batch.py`
- Test: `tests/test_convert_commands.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing batch tests for MCP-mode validation skipping**

```python
def test_run_gen_eval_batch_skips_affinity_capacity_validation_when_mcp_enabled(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workspace = root / "alpha"
        workspace.mkdir()
        (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

        with patch("triton_agent.generation.batch.validate_batch_affinity_capacity") as mocked:
            exit_code = run_gen_eval_batch(
                root,
                GenerationOptions(
                    interact=False,
                    verbose=False,
                    show_output=False,
                    force_overwrite=False,
                    agent_name="codex",
                    remote=None,
                    remote_workdir=None,
                    min_rounds=None,
                    continue_optimize=False,
                    output=None,
                    test_mode="differential",
                    bench_mode="standalone",
                    prompt=None,
                    enable_mcp=True,
                ),
                max_concurrency=8,
                run_request=lambda request, stdout=None, stderr=None: AgentResult(return_code=0, stdout="ok", stderr=""),
            )

    self.assertEqual(exit_code, 0)
    mocked.assert_not_called()
```

- [ ] **Step 2: Run the focused batch tests and verify they fail first**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_generation_batch.py tests/test_optimize_runtime.py -k "affinity or enable_mcp"`

Expected: FAIL because batch paths still validate affinity capacity unconditionally.

- [ ] **Step 3: Split batch validation logic by `enable_mcp`**

```python
devices = configured_batch_npu_devices()
if not options.enable_mcp:
    validate_batch_affinity_capacity(devices, max_concurrency=max_concurrency)
```

```python
staged_skill_names, _ = resolve_staged_skills(
    CommandKind.GEN_EVAL,
    enable_mcp=options.enable_mcp,
)
scope = (
    managed_mcp_scope()
    if managed_mcp_server_names_for_request(staged_skill_names, enable_mcp=options.enable_mcp)
    else nullcontext()
)
```

- [ ] **Step 4: Add explicit regression tests that legacy mode still validates capacity**

```python
def test_run_gen_eval_batch_preserves_affinity_capacity_validation_without_mcp(self) -> None:
    with patch("triton_agent.generation.batch.validate_batch_affinity_capacity") as mocked:
        run_gen_eval_batch(
            root,
            GenerationOptions(..., enable_mcp=False),
            max_concurrency=2,
            run_request=_fake_run,
        )
    mocked.assert_called_once()
```

- [ ] **Step 5: Run the focused batch tests**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_generation_batch.py tests/test_convert_commands.py tests/test_optimize_runtime.py -k "enable_mcp or affinity"`

Expected: PASS with non-MCP validation preserved and MCP mode skipping only the batch-time capacity check.

- [ ] **Step 6: Commit the batch behavior split**

```bash
git add src/triton_agent/generation/batch.py src/triton_agent/convert/batch.py src/triton_agent/optimize/batch.py tests/test_generation_batch.py tests/test_convert_commands.py tests/test_optimize_runtime.py
git commit -m "feat: relax batch affinity checks in run-eval mcp mode"
```

### Task 5: Rewrite the MCP skill docs to be tool-first and drop `compare-result`

**Files:**
- Modify: `skills/triton-npu-run-eval-mcp/SKILL.md`
- Modify: `skills/triton-npu-run-eval-mcp/references/run-test.md`
- Modify: `skills/triton-npu-run-eval-mcp/references/run-bench.md`
- Modify: `skills/triton-npu-run-eval-mcp/references/profile-bench.md`
- Modify: `skills/triton-npu-run-eval-mcp/references/profile-report.md`
- Modify: `skills/triton-npu-run-eval-mcp/references/compare-perf.md`
- Delete: `skills/triton-npu-run-eval-mcp/references/compare-result.md`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Write failing contract assertions for the MCP skill content**

```python
def test_run_eval_mcp_skill_does_not_reference_compare_result(self) -> None:
    skill = _read("skills/triton-npu-run-eval-mcp/SKILL.md")
    self.assertNotIn("compare-result", skill)
    self.assertFalse((REPO_ROOT / "skills" / "triton-npu-run-eval-mcp" / "references" / "compare-result.md").exists())


def test_run_eval_mcp_skill_uses_tool_first_guidance(self) -> None:
    skill = _read("skills/triton-npu-run-eval-mcp/SKILL.md")
    self.assertIn("use the corresponding MCP tool", skill)
    self.assertNotIn("python3 ./scripts/run-command.py", skill)
```

- [ ] **Step 2: Run the focused contract tests and verify they fail first**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_generation_contracts.py -k run_eval_mcp`

Expected: FAIL because the current MCP skill still references scripts and `compare-result`.

- [ ] **Step 3: Rewrite `SKILL.md` and references in tool-first language**

```md
# Run-Eval Router

Use the corresponding MCP tool for run-eval actions in this staged skill.

Primary MCP tools:

- `run-test-baseline`
- `run-test-optimize`
- `run-bench`
- `profile-bench`
- `profile-report`
- `compare-perf`

During normal agent use:

- use the MCP tool instead of calling `python3 ./scripts/run-command.py`
- keep arguments and artifact expectations aligned with the focused reference
- do not read unrelated command guides
```

- [ ] **Step 4: Delete the MCP `compare-result` reference and update link coverage tests**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_generation_contracts.py -k compare_result`

Expected: PASS with only the legacy skill still referencing `compare-result`.

- [ ] **Step 5: Commit the MCP skill documentation refresh**

```bash
git add skills/triton-npu-run-eval-mcp/SKILL.md skills/triton-npu-run-eval-mcp/references/run-test.md skills/triton-npu-run-eval-mcp/references/run-bench.md skills/triton-npu-run-eval-mcp/references/profile-bench.md skills/triton-npu-run-eval-mcp/references/profile-report.md skills/triton-npu-run-eval-mcp/references/compare-perf.md tests/test_generation_contracts.py
git rm skills/triton-npu-run-eval-mcp/references/compare-result.md
git commit -m "docs: make run-eval mcp skill tool-first"
```

### Task 6: Add `profile-report` and `compare-perf` MCP tools without device leasing

**Files:**
- Modify: `src/triton_agent/run_eval_mcp_server.py`
- Modify: `tests/test_run_eval_mcp_server.py`

- [ ] **Step 1: Write failing MCP server tests for the new tool registrations**

```python
def test_server_registers_expected_tools(self) -> None:
    server = module.create_server(slot_pool=module.NpuDevicePool(("0",)))

    async def _list_tool_names() -> list[str]:
        tools = await server.list_tools()
        return sorted(tool.name for tool in tools)

    self.assertEqual(
        asyncio.run(_list_tool_names()),
        [
            "compare-perf",
            "profile-bench",
            "profile-report",
            "run-bench",
            "run-test-baseline",
            "run-test-optimize",
        ],
    )
```

- [ ] **Step 2: Add failing tests that artifact-only tools do not lease a device**

```python
def test_compare_perf_tool_does_not_lease_device(self) -> None:
    server = module.create_server(slot_pool=module.NpuDevicePool(("0",)))
    observed: dict[str, object] = {}

    def fake_run_subcommand(subcommand: str, arguments: list[str], *, leased_device: Optional[str] = None, workspace: Path):
        observed["subcommand"] = subcommand
        observed["leased_device"] = leased_device
        observed["arguments"] = arguments
        return {"return_code": 0, "stdout": "ok\n", "stderr": ""}

    async def _call_tool():
        with (
            patch.object(module, "_run_subcommand", side_effect=fake_run_subcommand),
            patch.object(module, "current_workspace", return_value=Path("/tmp/ws")),
        ):
            return await server.call_tool(
                "compare-perf",
                {"baseline": "/tmp/base.txt", "compare": "/tmp/candidate.txt", "metric_source": "kernel"},
            )

    asyncio.run(_call_tool())
    self.assertEqual(observed["subcommand"], "compare-perf")
    self.assertIsNone(observed["leased_device"])
```

- [ ] **Step 3: Run the focused MCP server tests and verify they fail first**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_run_eval_mcp_server.py -k "profile_report or compare_perf or expected_tools"`

Expected: FAIL because the current server exposes only four tools.

- [ ] **Step 4: Implement the two new tools in `run_eval_mcp_server.py`**

```python
@server.tool(name="profile-report")
def profile_report(
    profile_dir: str,
    target_op: str | None = None,
    format: str | None = None,
    top: int | None = None,
) -> dict[str, object]:
    workspace = current_workspace()
    arguments = ["--profile-dir", profile_dir]
    if target_op is not None:
        arguments.extend(["--target-op", target_op])
    if format is not None:
        arguments.extend(["--format", format])
    if top is not None:
        arguments.extend(["--top", str(top)])
    return _run_subcommand(
        "profile-report",
        arguments,
        leased_device=None,
        workspace=workspace,
    )
```

```python
@server.tool(name="compare-perf")
def compare_perf(
    baseline: str,
    compare: str,
    skip_latency_errors: bool = False,
    metric_source: str | None = None,
) -> dict[str, object]:
    workspace = current_workspace()
    arguments = ["--baseline", baseline, "--compare", compare]
    if skip_latency_errors:
        arguments.append("--skip-latency-errors")
    if metric_source is not None:
        arguments.extend(["--metric-source", metric_source])
    return _run_subcommand(
        "compare-perf",
        arguments,
        leased_device=None,
        workspace=workspace,
    )
```

- [ ] **Step 5: Run the MCP server tests**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_run_eval_mcp_server.py`

Expected: PASS with the new tools registered and no device leasing for artifact-only analysis.

- [ ] **Step 6: Commit the MCP server expansion**

```bash
git add src/triton_agent/run_eval_mcp_server.py tests/test_run_eval_mcp_server.py
git commit -m "feat: add report and perf compare tools to run-eval mcp"
```

### Task 7: Run full validation and refresh affected design docs if needed

**Files:**
- Modify: `docs/specs/2026-06-04-run-eval-mcp-server-design.md`
- Modify: `docs/specs/2026-06-05-run-eval-mcp-toggle-design.md`

- [ ] **Step 1: Align the older MCP design doc with the toggle-based rollout if it still claims MCP is unconditional**

```md
## Update Note

This document now describes the shared HTTP MCP runtime used when `--enable-mcp` is enabled for supported agent-backed run-eval workflows. The legacy script-backed run-eval path remains the default when the toggle is omitted.
```

- [ ] **Step 2: Run lint and type checks**

Run: `uv run --group dev ruff check`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 3: Run the full test suite**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

Expected: all tests pass.

- [ ] **Step 4: Commit the final integration pass**

```bash
git add docs/specs/2026-06-04-run-eval-mcp-server-design.md docs/specs/2026-06-05-run-eval-mcp-toggle-design.md
git commit -m "docs: align run-eval mcp design with toggle rollout"
```

## Self-Review

- Spec coverage:
  - `--enable-mcp` only on six agent-backed commands is covered in Task 1.
  - Stable staged name with source switching is covered in Task 2.
  - Explicit MCP activation and request-scoped MCP wiring are covered in Task 3.
  - Batch validation split is covered in Task 4.
  - MCP skill rewrite and `compare-result` removal from the MCP path are covered in Task 5.
  - `profile-report` and `compare-perf` tools are covered in Task 6.
  - Full repo verification and doc alignment are covered in Task 7.
- Placeholder scan:
  - Removed generic “add tests” language; every task names exact files, commands, and example code.
- Type consistency:
  - `enable_mcp` is used consistently across `GenerationOptions`, `ConvertOptions`, `OptimizeRunOptions`, and `AgentRequest`.
  - The MCP helper name used in the plan is `managed_mcp_server_names_for_request(...)` everywhere after the refactor.
