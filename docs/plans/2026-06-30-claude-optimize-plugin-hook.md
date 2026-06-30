# Claude Optimize Plugin Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repository build script that produces a self-contained Claude Code plugin directory for the optimize hook workflow, including one optimize agent, the minimum optimize skills, and plugin-managed `.triton-agent/` lifecycle hooks.

**Architecture:** Reuse the repository's existing optimize guidance and skill staging contracts to render one plugin-scoped optimize agent and copy only the required optimize skills into a generated plugin tree. Move Claude optimize runtime bootstrap from CLI request-scoped staging into plugin-local `SessionStart`, `SessionEnd`, and `PreToolUse` hooks that conservatively recover `.triton-agent/state.json` from durable optimize artifacts.

**Tech Stack:** Python 3.12, existing `triton_agent` prompt/guidance modules, Claude plugin manifest + hooks, `unittest`, `claude plugin validate`

**Workflow-state command semantics update:** `submit-baseline` should auto-bootstrap missing `.triton-agent/state.json` before marking the baseline accepted. `start-round` and `set-current-round-state` should keep treating missing workflow state as a structured failure that the agent must repair through workflow commands rather than direct internal state-file edits, while `start-round` should create a missing `opt-round-N/` directory. `submit-round` should fail with structured JSON when either the workflow state or the round directory is missing instead of silently skipping completion or raising an uncaught exception.

**Builder placement update:** Keep the Claude optimize plugin builder implementation directly inside `scripts/build-claude-optimize-plugin.py`; do not maintain a separate `src/triton_agent/claude_optimize_plugin.py` module for this packaging-only flow.

---

### Task 1: Add Reusable Claude Optimize Plugin Rendering Helpers

**Files:**
- Modify: `scripts/build-claude-optimize-plugin.py`
- Modify: `src/triton_agent/optimize/memory_file.py`
- Modify: `src/triton_agent/optimize/prompts.py`
- Test: `tests/test_claude_optimize_plugin.py`

- [ ] **Step 1: Write failing tests for plugin guidance rendering and optimize skill resolution**

```python
class ClaudeOptimizePluginBuilderTests(unittest.TestCase):
    def test_plugin_builder_uses_optimize_skill_staging_contract(self) -> None:
        skill_names, _ = resolve_staged_skills(CommandKind.OPTIMIZE)
        asset = build_claude_optimize_plugin_assets()

        self.assertIsNotNone(skill_names)
        self.assertEqual(tuple(sorted(asset.skill_names)), tuple(sorted(skill_names or ())))

    def test_plugin_builder_renders_single_optimize_agent_without_standalone_claude_md(self) -> None:
        asset = build_claude_optimize_plugin_assets()

        self.assertIn("agents/triton-agent-optimize.md", asset.text_files)
        self.assertNotIn("CLAUDE.md", asset.text_files)
        self.assertNotIn("prompts.md", asset.text_files)
        agent_text = asset.text_files["agents/triton-agent-optimize.md"]
        self.assertIn("Complete optimize rounds strictly one at a time in sequence.", agent_text)
        self.assertIn("Use the staged `ascend-npu-optimize-state` skill's `start-round` subcommand", agent_text)
```

- [ ] **Step 2: Run the focused builder test file and verify it fails because the builder module does not exist yet**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin -v`

Expected: FAIL because `build-claude-optimize-plugin.py` does not yet expose the builder helpers.

- [ ] **Step 3: Expose reusable optimize-guidance rendering helpers from the existing prompt/guidance modules**

```python
# src/triton_agent/optimize/memory_file.py
class MemoryFileManager:
    def render_round_gated_guidance(
        self,
        *,
        agent_name: str,
        language: str = "triton",
        optimize_target: str = "kernel",
        include_supervisor_handoff: bool = True,
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
        enable_cann_ext_api: bool = False,
        enable_subagent: bool = False,
        optimize_knowledge_skill_name: str | None = None,
    ) -> str:
        guidance_filename = self.guidance_filename(agent_name)
        return self._render_round_gated_guidance(
            guidance_filename=guidance_filename,
            language=language,
            optimize_target=optimize_target,
            include_supervisor_handoff=include_supervisor_handoff,
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
            enable_cann_ext_api=enable_cann_ext_api,
            enable_subagent=enable_subagent,
            optimize_knowledge_skill_name=optimize_knowledge_skill_name,
        )
```

```python
# src/triton_agent/optimize/prompts.py
def build_optimize_plugin_prompt(
    *,
    language: str = "triton",
    optimize_target: str = "kernel",
    enable_cann_ext_api: bool = False,
    enable_subagent: bool = False,
) -> str:
    lines = [
        *optimize_target_lines(optimize_target=optimize_target, language=language),
        *sequential_round_execution_lines(),
        *layered_analysis_lines(round_scope="the session", language=language),
        *next_round_reflection_lines(language=language),
    ]
    if enable_subagent:
        lines.extend(optimize_subagent_recommendation_lines(language=language))
    lines.extend(cann_ext_api_lines(enabled=enable_cann_ext_api, language=language))
    return "\n".join(lines)
```

- [ ] **Step 4: Implement the builder asset module that assembles the plugin agent text and selected optimize skills**

```python
# scripts/build-claude-optimize-plugin.py
@dataclass(frozen=True)
class ClaudeOptimizePluginAssets:
    text_files: dict[str, str]
    skill_names: tuple[str, ...]
    skill_sources: dict[str, str] | None


def build_claude_optimize_plugin_assets(
    *,
    language: str = "triton",
    optimize_target: str = "kernel",
    enable_cann_ext_api: bool = False,
    enable_subagent: bool = False,
) -> ClaudeOptimizePluginAssets:
    skill_names, skill_sources = resolve_staged_skills(
        CommandKind.OPTIMIZE,
        language=language,
        optimize_target=optimize_target,
        enable_cann_ext_api=enable_cann_ext_api,
    )
    if skill_names is None:
        raise RuntimeError("Optimize plugin packaging requires an explicit optimize skill list.")

    memory_files = MemoryFileManager()
    guidance_text = memory_files.render_round_gated_guidance(
        workdir=Path("."),
        agent_name="claude",
        language=language,
        optimize_target=optimize_target,
        include_supervisor_handoff=False,
        enable_cann_ext_api=enable_cann_ext_api,
        enable_subagent=enable_subagent,
    )
    prompt_text = build_optimize_plugin_prompt(
        language=language,
        optimize_target=optimize_target,
        enable_cann_ext_api=enable_cann_ext_api,
        enable_subagent=enable_subagent,
    )
    agent_text = render_claude_optimize_agent(guidance_text=guidance_text, prompt_text=prompt_text)
    return ClaudeOptimizePluginAssets(
        text_files={"agents/triton-agent-optimize.md": agent_text},
        skill_names=skill_names,
        skill_sources=skill_sources,
    )
```

- [ ] **Step 5: Re-run the focused builder tests and verify they pass**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin -v`

Expected: PASS for guidance rendering, single-agent packaging, and optimize skill resolution.

- [ ] **Step 6: Commit the helper-layer change**

```bash
git add docs/plans/2026-06-30-claude-optimize-plugin-hook.md tests/test_claude_optimize_plugin.py scripts/build-claude-optimize-plugin.py src/triton_agent/optimize/memory_file.py src/triton_agent/optimize/prompts.py
git commit -m "feat: add claude optimize plugin builder helpers"
```

### Task 2: Add Plugin Hook Assets And Conservative Workflow-State Bootstrap

**Files:**
- Create: `hooks/claude_plugin/hooks.json`
- Create: `hooks/claude_plugin/session_start.py`
- Create: `hooks/claude_plugin/session_end.py`
- Create: `hooks/claude_plugin/state_bootstrap.py`
- Create: `hooks/claude_plugin/pretooluse_guard.py`
- Test: `tests/test_claude_optimize_plugin_hooks.py`

- [ ] **Step 1: Write failing tests for SessionStart bootstrap, SessionEnd cleanup, and plugin-mode guard diagnostics**

```python
class ClaudeOptimizePluginHookTests(unittest.TestCase):
    def test_session_start_creates_runtime_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            result = run_session_start(workspace, agent_name="triton-agent-optimize")
            self.assertEqual(result.returncode, 0)
            self.assertTrue((workspace / ".triton-agent").is_dir())

    def test_session_start_recovers_awaiting_round_start_when_baseline_state_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(json.dumps({"established": True}), encoding="utf-8")
            run_session_start(workspace, agent_name="triton-agent-optimize")
            payload = json.loads((workspace / ".triton-agent" / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "awaiting_round_start")

    def test_session_end_removes_runtime_dir_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".triton-agent").mkdir()
            (workspace / "baseline").mkdir()
            run_session_end(workspace, agent_name="triton-agent-optimize")
            self.assertFalse((workspace / ".triton-agent").exists())
            self.assertTrue((workspace / "baseline").exists())
```

- [ ] **Step 2: Run the hook test file and verify it fails because the plugin hook assets do not exist yet**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin_hooks -v`

Expected: FAIL because `hooks/claude_plugin/` files and test helpers are missing.

- [ ] **Step 3: Add a self-contained plugin bootstrap helper that conservatively reconstructs workflow state**

```python
# hooks/claude_plugin/state_bootstrap.py
def bootstrap_runtime_state(workspace: Path) -> BootstrapResult:
    runtime_dir = workspace / ".triton-agent"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state_path = runtime_dir / "state.json"
    if state_path.exists():
        return validate_existing_state(state_path)

    phase = infer_phase_from_workspace(workspace)
    payload = build_minimal_state_payload(workspace=workspace, phase=phase)
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return BootstrapResult(created_runtime_dir=True, wrote_state=True, phase=phase)
```

```python
def infer_phase_from_workspace(workspace: Path) -> str:
    baseline_state = workspace / "baseline" / "state.json"
    if baseline_state.is_file() and baseline_looks_established(baseline_state):
        return "awaiting_round_start"
    return "baseline"
```

- [ ] **Step 4: Add SessionStart, SessionEnd, and plugin-local PreToolUse wrappers**

```python
# hooks/claude_plugin/session_start.py
def main() -> int:
    payload = json.load(sys.stdin)
    if payload.get("agent_type") != "triton-agent-optimize":
        return 0
    result = bootstrap_runtime_state(Path(payload["cwd"]))
    if result.additional_context:
        json.dump(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": result.additional_context,
                }
            },
            sys.stdout,
        )
    return 0
```

```python
# hooks/claude_plugin/session_end.py
def main() -> int:
    payload = json.load(sys.stdin)
    if payload.get("agent_type") != "triton-agent-optimize":
        return 0
    cleanup_runtime_tree(Path(payload["cwd"]) / ".triton-agent")
    return 0
```

```python
# hooks/claude_plugin/pretooluse_guard.py
def main(argv: list[str] | None = None) -> int:
    policy = _load_json(Path(args.policy))
    payload = json.load(sys.stdin)
    bootstrap_reason = deny_reason_for_missing_plugin_workflow_state(payload, policy)
    if bootstrap_reason is not None:
        json.dump(_build_denial_output(bootstrap_reason), sys.stdout)
        return 0
    reason = _deny_reason_for_tool_use(policy, payload)
    if reason is not None:
        json.dump(_build_denial_output(reason), sys.stdout)
    return 0
```

- [ ] **Step 5: Write the plugin hook manifest and verify it wires SessionStart, SessionEnd, and PreToolUse**

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|clear|compact",
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/session_start.py\""
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/session_end.py\""
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash|Read|Grep|Glob|Edit|MultiEdit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/pretooluse_guard.py\" --policy \"${CLAUDE_PLUGIN_ROOT}/hooks/policy.json\""
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 6: Re-run the hook tests and verify they pass**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin_hooks -v`

Expected: PASS for runtime-dir creation, conservative phase recovery, malformed state detection, and cleanup-only semantics.

- [ ] **Step 7: Commit the plugin hook asset layer**

```bash
git add tests/test_claude_optimize_plugin_hooks.py hooks/claude_plugin/hooks.json hooks/claude_plugin/session_start.py hooks/claude_plugin/session_end.py hooks/claude_plugin/state_bootstrap.py hooks/claude_plugin/pretooluse_guard.py
git commit -m "feat: add claude optimize plugin runtime hooks"
```

### Task 3: Add The Build Script And End-To-End Plugin Packaging Validation

**Files:**
- Create: `scripts/build-claude-optimize-plugin.py`
- Modify: `scripts/build-claude-optimize-plugin.py`
- Modify: `tests/test_claude_optimize_plugin.py`

- [ ] **Step 1: Extend the builder tests with end-to-end filesystem packaging expectations**

```python
def test_build_script_writes_valid_plugin_tree(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "claude-optimize-hook"
        build_claude_optimize_plugin(output_dir)

        self.assertTrue((output_dir / ".claude-plugin" / "plugin.json").exists())
        self.assertTrue((output_dir / "agents" / "triton-agent-optimize.md").exists())
        self.assertTrue((output_dir / "hooks" / "hooks.json").exists())
        self.assertTrue((output_dir / "skills").is_dir())
        self.assertFalse((output_dir / "CLAUDE.md").exists())
        self.assertFalse((output_dir / "prompts.md").exists())
```

- [ ] **Step 2: Run the builder test file again and verify it fails because the script entry point and on-disk copy logic do not exist yet**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin -v`

Expected: FAIL because `build_claude_optimize_plugin()` and the CLI script do not yet materialize the plugin tree.

- [ ] **Step 3: Implement the filesystem writer that copies hook assets and selected skills into the output directory**

```python
# scripts/build-claude-optimize-plugin.py
def build_claude_optimize_plugin(output_dir: Path, *, skills_root: Path | None = None) -> Path:
    assets = build_claude_optimize_plugin_assets()
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_plugin_manifest(root / ".claude-plugin" / "plugin.json")
    write_text_files(root, assets.text_files)
    copy_hook_assets(root / "hooks")
    copy_selected_skills(root / "skills", assets.skill_names, assets.skill_sources, skills_root=skills_root)
    write_plugin_readme(root / "README.md")
    return root
```

- [ ] **Step 4: Add the repository script entry point**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="dist/claude-optimize-hook")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    build_claude_optimize_plugin(output_dir)
    print(output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Validate the generated plugin with Claude and run focused repository tests**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin tests.test_claude_optimize_plugin_hooks -v`

Expected: PASS for all focused packaging and hook tests.

Run: `python3 scripts/build-claude-optimize-plugin.py /tmp/claude-optimize-hook`

Expected: prints the generated plugin directory path and creates the plugin tree on disk.

Run: `claude plugin validate /tmp/claude-optimize-hook`

Expected: validation passes, ideally with no errors.

- [ ] **Step 6: Run repository verification commands**

Run: `uv run --group dev ruff check`

Expected: PASS

Run: `uv run pyright`

Expected: PASS

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

Expected: PASS

- [ ] **Step 7: Commit the build script and end-to-end packaging path**

```bash
git add scripts/build-claude-optimize-plugin.py tests/test_claude_optimize_plugin.py
git commit -m "feat: build claude optimize plugin"
```
