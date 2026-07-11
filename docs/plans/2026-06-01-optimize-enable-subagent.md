# Optimize Enable Subagent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--enable-subagent` to optimize workflows so `codex`, `opencode`, and `claude` workspaces can stage a diagnosis-only performance subagent that may read staged skills and collect benchmark/profile/IR evidence without ever editing operator implementations.

**Architecture:** Keep the CLI surface small and thread one boolean through the existing optimize request path. Add a generic backend-native subagent staging layer plus one optimize-owned diagnosis subagent definition, then let `OptimizeSessionArtifactsManager` own its lifecycle alongside guidance files so setup and cleanup stay per-run and auditable.

**Tech Stack:** Python 3.11, argparse, unittest, existing optimize/session-artifact infrastructure, backend-native agent config formats for Codex, OpenCode, and Claude

---

## File Structure

- `src/helix/cli.py`
  Parse `--enable-subagent` for `optimize` and `optimize-batch`.
- `src/helix/commands/optimize.py`
  Map parsed args into `OptimizeRunOptions` and reject unsupported backend combinations before filesystem checks.
- `src/helix/optimize/models.py`
  Add `enable_subagent` to optimize options.
- `src/helix/models.py`
  Add `enable_subagent` to `AgentRequest` so execution and artifacts can see it.
- `src/helix/subagents.py`
  Hold generic staging data structures, backend path mapping, prepare/cleanup, and collision checks.
- `src/helix/optimize/subagents.py`
  Hold the built-in `helix-perf-diagnosis-advisor` definition, backend renderers, and optimize-specific recommendation text.
- `src/helix/optimize/orchestration.py`
  Carry `enable_subagent` into the optimize request and pass staged-skill context into artifact preparation.
- `src/helix/optimize/session_artifacts.py`
  Stage subagents together with optimize guidance and clean them up after each run.
- `src/helix/optimize/prompts.py`
  Add worker prompt guidance that recommends using the diagnosis subagent without making it mandatory.
- `src/helix/optimize/memory_file.py`
  Add matching workspace guidance text for `AGENTS.md` / `CLAUDE.md`.
- `tests/test_cli.py`
  Cover parser acceptance, default values, and unsupported backend rejection.
- `tests/test_subagents.py`
  Cover generic staging, rendering, collisions, and cleanup.
- `tests/test_optimize_guidance.py`
  Cover guidance-file staging and subagent recommendation text.
- `tests/test_optimize_runtime.py`
  Cover optimize runtime integration so staged subagents appear during session setup and disappear on cleanup.
- `tests/test_opencode_runner.py`
  Lock in OpenCode config/launch compatibility for subagent-enabled optimize requests.
- `README.md`
  Document `--enable-subagent`, supported backends, and the diagnosis-only contract.

### Task 1: Lock In The CLI Contract

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/optimize.py`
- Modify: `src/helix/optimize/models.py`

- [ ] **Step 1: Add failing parser tests for `--enable-subagent` on `optimize` and `optimize-batch`**

```python
def test_optimize_accepts_enable_subagent_option(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize", "-i", "kernel.py", "--enable-subagent"])
    self.assertTrue(args.enable_subagent)
    options = optimize_run_options_from_args(args)
    self.assertTrue(options.enable_subagent)


def test_optimize_batch_accepts_enable_subagent_option(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize-batch", "-i", "kernels", "--enable-subagent"])
    self.assertTrue(args.enable_subagent)
    options = optimize_run_options_from_args(args)
    self.assertTrue(options.enable_subagent)
```

- [ ] **Step 2: Add failing command-validation tests that reject unsupported backends before input-path checks**

```python
def test_handle_optimize_rejects_enable_subagent_for_pi(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["optimize", "-i", "kernel.py", "--agent", "pi", "--enable-subagent"]
    )
    stderr = StringIO()
    with self.assertRaises(SystemExit) as exc, redirect_stderr(stderr):
        optimize_commands.handle_optimize(parser, args)
    self.assertEqual(exc.exception.code, 2)
    self.assertIn("--enable-subagent only supports", stderr.getvalue())
```

- [ ] **Step 3: Run the targeted parser/validation tests and confirm they fail before implementation**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_optimize_accepts_enable_subagent_option \
  tests.test_cli.CliParserTests.test_optimize_batch_accepts_enable_subagent_option \
  tests.test_cli.CliParserTests.test_handle_optimize_rejects_enable_subagent_for_pi \
  -v
```

Expected:

```text
FAIL: ... enable_subagent attribute missing or validation not enforced yet
```

- [ ] **Step 4: Implement the CLI wiring and option validation**

```python
@dataclass(frozen=True)
class OptimizeRunOptions:
    ...
    enable_subagent: bool = False


def optimize_run_options_from_args(args: argparse.Namespace) -> OptimizeRunOptions:
    ...
    subagent_enabled = bool(getattr(args, "enable_subagent", False))
    return OptimizeRunOptions(
        ...
        enable_subagent=subagent_enabled,
    )


def _validate_agent_options(..., options: OptimizeRunOptions) -> None:
    ...
    if options.enable_subagent and options.agent_name not in {"codex", "opencode", "claude"}:
        parser.error("--enable-subagent only supports --agent codex, opencode, or claude.")
```

- [ ] **Step 5: Re-run the targeted tests and make them pass**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_optimize_accepts_enable_subagent_option \
  tests.test_cli.CliParserTests.test_optimize_batch_accepts_enable_subagent_option \
  tests.test_cli.CliParserTests.test_handle_optimize_rejects_enable_subagent_for_pi \
  -v
```

Expected:

```text
OK
```

- [ ] **Step 6: Commit the CLI contract change**

```bash
git add tests/test_cli.py src/helix/cli.py src/helix/commands/optimize.py src/helix/optimize/models.py
git commit -m "feat: add optimize enable-subagent option"
```

### Task 2: Add Generic Subagent Staging Infrastructure

**Files:**
- Create: `src/helix/subagents.py`
- Create: `src/helix/optimize/subagents.py`
- Create: `tests/test_subagents.py`

- [ ] **Step 1: Write failing staging tests for backend-native file paths, collisions, cleanup, and rendered analysis restrictions**

```python
def test_prepare_codex_subagent_stages_toml_and_cleans_up(self) -> None:
    manager = SubagentManager()
    definition = perf_diagnosis_subagent_definition(
        backend="codex",
        optimize_target="kernel",
        enable_cann_ext_api=False,
    )
    state = manager.prepare("codex", workspace, (definition,))
    agent_path = workspace / ".codex" / "agents" / "helix-perf-diagnosis-advisor.toml"
    self.assertTrue(agent_path.exists())
    self.assertIn("must not perform optimization work", agent_path.read_text(encoding="utf-8"))
    self.assertEqual(manager.cleanup(state), [])
    self.assertFalse(agent_path.exists())
```

- [ ] **Step 2: Run the staging tests and confirm they fail before the new modules exist**

Run:

```bash
uv run python -m unittest tests.test_subagents -v
```

Expected:

```text
ERROR: No module named 'helix.subagents'
```

- [ ] **Step 3: Implement the generic staging layer in `src/helix/subagents.py`**

```python
@dataclass(frozen=True)
class RenderedSubagent:
    relative_path: Path
    content: str


@dataclass(frozen=True)
class SubagentDefinition:
    id: str
    supported_backends: tuple[str, ...]
    render: Callable[[str], RenderedSubagent]


@dataclass
class SubagentStageSet:
    created_paths: list[Path]


class SubagentManager:
    def prepare(self, backend: str, workdir: Path, definitions: tuple[SubagentDefinition, ...]) -> SubagentStageSet:
        ...

    def cleanup(self, stage_set: SubagentStageSet) -> list[str]:
        ...
```

- [ ] **Step 4: Implement the optimize-owned diagnosis subagent definition in `src/helix/optimize/subagents.py`**

```python
def perf_diagnosis_subagent_definition(
    *,
    backend: str,
    optimize_target: str,
    enable_cann_ext_api: bool,
) -> SubagentDefinition:
    ...


def optimize_subagent_recommendation_lines() -> list[str]:
    return [
        "A diagnosis subagent named `helix-perf-diagnosis-advisor` is available in this workspace.",
        "Use it when the bottleneck hypothesis is still unclear before deeper optimize edits.",
    ]
```

- [ ] **Step 5: Re-run the staging tests and make them pass**

Run:

```bash
uv run python -m unittest tests.test_subagents -v
```

Expected:

```text
OK
```

- [ ] **Step 6: Commit the staging infrastructure**

```bash
git add src/helix/subagents.py src/helix/optimize/subagents.py tests/test_subagents.py
git commit -m "feat: add backend-native subagent staging"
```

### Task 3: Thread Subagents Through Optimize Session Artifacts

**Files:**
- Modify: `src/helix/models.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `src/helix/optimize/session_artifacts.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Add failing artifact-manager tests that stage and clean diagnosis subagents with optimize sessions**

```python
def test_prepare_continuous_session_stages_subagent_when_enabled(self) -> None:
    manager = OptimizeSessionArtifactsManager()
    state = manager.prepare_continuous_session(
        workdir,
        operator_path=workdir / "op.py",
        test_mode="differential",
        bench_mode="standalone",
        agent_name="codex",
        enable_subagent=True,
        staged_skill_names=("triton-npu-optimize-knowledge",),
    )
    self.assertTrue(
        (workdir / ".codex" / "agents" / "helix-perf-diagnosis-advisor.toml").exists()
    )
    self.assertEqual(manager.cleanup_continuous_session(state), [])
```

- [ ] **Step 2: Run the targeted optimize-guidance/runtime tests and confirm they fail for missing subagent state**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_guidance \
  tests.test_optimize_runtime \
  -v
```

Expected:

```text
FAIL: prepare_*_session does not accept enable_subagent or does not stage subagent files
```

- [ ] **Step 3: Add `enable_subagent` to `AgentRequest` and carry it through optimize request construction**

```python
@dataclass
class AgentRequest:
    ...
    enable_subagent: bool = False


return AgentRequest(
    ...
    enable_subagent=options.enable_subagent,
)
```

- [ ] **Step 4: Extend `OptimizeSessionArtifactsManager` to own subagent staging and cleanup**

```python
@dataclass
class SharedOptimizeSessionArtifactsState:
    memory_file: MemoryFileState
    archive: ArchiveState
    subagents: SubagentStageSet | None = None


subagent_state = None
if enable_subagent:
    subagent_state = SubagentManager().prepare(
        agent_name,
        workdir,
        optimize_subagent_definitions(...),
    )
...
warnings.extend(self._subagents.cleanup(state.subagents))
```

- [ ] **Step 5: Re-run the targeted optimize artifact tests and make them pass**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_guidance \
  tests.test_optimize_runtime \
  -v
```

Expected:

```text
OK
```

- [ ] **Step 6: Commit the optimize artifact integration**

```bash
git add src/helix/models.py src/helix/optimize/orchestration.py src/helix/optimize/session_artifacts.py tests/test_optimize_guidance.py tests/test_optimize_runtime.py
git commit -m "feat: stage optimize diagnosis subagents per session"
```

### Task 4: Inject Diagnosis Guidance And Backend Compatibility

**Files:**
- Modify: `src/helix/optimize/prompts.py`
- Modify: `src/helix/optimize/memory_file.py`
- Modify: `tests/test_opencode_runner.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing guidance tests that only mention the diagnosis subagent when the option is enabled**

```python
def test_prepare_continuous_session_mentions_diagnosis_subagent_when_enabled(self) -> None:
    state = manager.prepare_continuous_session(
        workdir,
        operator_path=workdir / "op.py",
        test_mode="differential",
        bench_mode="standalone",
        agent_name="codex",
        enable_subagent=True,
    )
    content = state.guidance_path.read_text(encoding="utf-8")
    self.assertIn("helix-perf-diagnosis-advisor", content)
    self.assertIn("cannot perform optimization edits", content)
```

- [ ] **Step 2: Add a failing OpenCode regression test that proves subagent-enabled optimize requests keep the existing project-config launch contract**

```python
def test_subagent_enabled_optimize_keeps_pure_and_thinking(self) -> None:
    request = AgentRequest(
        command_kind=CommandKind.OPTIMIZE,
        input_path=workspace / "op.py",
        operator_path=workspace / "op.py",
        output_path=workspace / "opt_op.py",
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=False,
        show_output=False,
        force_overwrite=False,
        agent_name="opencode",
        skill_name="triton-npu-optimize",
        prompt="Prompt body",
        workdir=workspace,
        enable_subagent=True,
    )
    command = OpenCodeRunner().build_command(request)
    self.assertEqual(command[:3], ["opencode", "run", "--dir"])
    self.assertIn("--pure", command)
    self.assertIn("--thinking", command)
```

- [ ] **Step 3: Add a failing OpenCode config regression test that proves `.opencode/opencode.json` keeps the existing built-in subagent deny behavior even when `enable_subagent=True`**

```python
def test_run_with_enable_subagent_keeps_general_and_explore_denied(self) -> None:
    request = AgentRequest(
        command_kind=CommandKind.OPTIMIZE,
        input_path=workspace / "op.py",
        operator_path=workspace / "op.py",
        output_path=workspace / "opt_op.py",
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=False,
        show_output=False,
        force_overwrite=False,
        agent_name="opencode",
        skill_name="triton-npu-optimize",
        prompt="Prompt body",
        workdir=workspace,
        enable_subagent=True,
    )
    ...
    self.assertEqual(config["agent"]["build"]["permission"]["task"]["general"], "deny")
    self.assertEqual(config["agent"]["build"]["permission"]["task"]["explore"], "deny")
```

- [ ] **Step 4: Update optimize prompts and guidance rendering to include the recommendation block only when `enable_subagent` is true**

```python
def diagnosis_subagent_lines(*, enabled: bool) -> list[str]:
    if not enabled:
        return []
    return [
        "A diagnosis subagent named `helix-perf-diagnosis-advisor` is available in this workspace.",
        "It may inspect harnesses and collect benchmark, profiler, or IR evidence for diagnosis.",
        "It must not perform optimization edits or apply candidate patches.",
    ]
```

- [ ] **Step 5: Re-run the targeted guidance and OpenCode tests and make them pass**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_guidance \
  tests.test_opencode_runner \
  tests.test_cli \
  -v
```

Expected:

```text
OK
```

- [ ] **Step 6: Commit the guidance and backend-compatibility changes**

```bash
git add src/helix/optimize/prompts.py src/helix/optimize/memory_file.py tests/test_optimize_guidance.py tests/test_opencode_runner.py tests/test_cli.py
git commit -m "feat: recommend optimize diagnosis subagent"
```

### Task 5: Document The New Optimize Option

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the optimize and optimize-batch option lists with `--enable-subagent`**

```markdown
- `--enable-subagent`: stage a diagnosis-only workspace subagent for `codex`, `opencode`, or `claude`. The subagent may inspect harnesses and collect benchmark/profile/IR evidence, but it must not perform optimization edits.
```

- [ ] **Step 2: Add one short behavior note near the optimize workflow description**

```markdown
When `--enable-subagent` is set, optimize stages a backend-native `helix-perf-diagnosis-advisor` subagent into the workspace. The main agent may call it for diagnosis help, but the subagent is limited to analysis work and must not edit operator implementations.
```

- [ ] **Step 3: Re-read the spec and this plan and make sure the documented user contract still matches the implementation tasks exactly**

Run:

```bash
sed -n '1,220p' docs/specs/2026-06-01-optimize-enable-subagent-design.md
sed -n '1,260p' docs/plans/2026-06-01-optimize-enable-subagent.md
```

Expected:

```text
The README wording matches the spec: supported backends, diagnosis-only behavior, evidence collection allowed, optimization edits forbidden.
```

- [ ] **Step 4: Commit the docs update**

```bash
git add README.md docs/specs/2026-06-01-optimize-enable-subagent-design.md docs/plans/2026-06-01-optimize-enable-subagent.md
git commit -m "docs: describe optimize subagent support"
```

### Task 6: Verify The Whole Change

**Files:**
- Modify: `src/helix/cli.py`
- Modify: `src/helix/subagents.py`
- Modify: `src/helix/optimize/subagents.py`
- Modify: `src/helix/optimize/session_artifacts.py`
- Modify: `src/helix/optimize/prompts.py`
- Modify: `src/helix/optimize/memory_file.py`
- Modify: `src/helix/backends/opencode.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_subagents.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_opencode_runner.py`
- Modify: `README.md`

- [ ] **Step 1: Run the targeted subagent-related unittest set**

Run:

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_subagents \
  tests.test_optimize_guidance \
  tests.test_optimize_runtime \
  tests.test_opencode_runner \
  -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Run the full repository unittest suite**

Run:

```bash
uv run python -m unittest discover -s tests -v
```

Expected:

```text
OK
```

- [ ] **Step 3: Run lint and type checks**

Run:

```bash
uv run --group dev ruff check
uv run pyright
```

Expected:

```text
All checks pass with no new optimize/subagent regressions.
```

- [ ] **Step 4: If verification fails, fix only the regressions introduced by this plan and re-run the failing command until clean**

```bash
uv run python -m unittest tests.test_subagents -v
uv run --group dev ruff check
uv run pyright
```

- [ ] **Step 5: Commit the verified final implementation**

```bash
git add src tests README.md
git commit -m "feat: stage optimize diagnosis subagents"
```
