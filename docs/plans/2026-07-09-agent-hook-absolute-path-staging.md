# Agent Hook Absolute Path Staging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite staged Claude and Codex hook configs so their hook commands use absolute workspace paths for staged scripts and `policy.json`.

**Architecture:** Keep the repository hook JSON files as readable structure templates and encode workspace-root placeholders in them: `${CLAUDE_PROJECT_DIR}` for Claude and `${CODEX_PROJECT_DIR}` for Codex. During staging, render those placeholders to the resolved workspace path so runtime hook execution no longer depends on cwd while the checked-in templates still show the intended path shape.

**Tech Stack:** Python 3.12, `pathlib`, `json`, `unittest`, `pytest`, `uv`

---

### Task 1: Lock The New Staged Config Behavior With Red Tests

**Files:**
- Modify: `tests/test_agent_hooks.py`
- Test: `tests/test_agent_hooks.py`

- [ ] **Step 1: Write the failing Claude and Codex staging assertions**

```python
self.assertEqual(
    hooks_config["hooks"]["PreToolUse"],
    [
        {
            "matcher": "Bash|Read|Grep|Glob|Edit|MultiEdit|Write",
            "hooks": [
                {
                    "type": "command",
                    "command": (
                        f"python \"{workspace.resolve() / '.codex' / 'helix-hooks' / 'tool_trace_hook.py'}\" "
                        f"--policy \"{workspace.resolve() / '.codex' / 'helix-hooks' / 'policy.json'}\" "
                        "--event PreToolUse"
                    ),
                },
                {
                    "type": "command",
                    "command": (
                        f"python3 \"{workspace.resolve() / '.codex' / 'helix-hooks' / 'pretooluse_guard.py'}\" "
                        f"--policy \"{workspace.resolve() / '.codex' / 'helix-hooks' / 'policy.json'}\""
                    ),
                },
            ],
        },
    ],
)
```

```python
self.assertEqual(
    settings["hooks"]["PreToolUse"],
    [
        {
            "matcher": "Bash|Read|Grep|Glob|Edit|MultiEdit|Write",
            "hooks": [
                {
                    "type": "command",
                    "command": "python3",
                    "args": [
                        str(workspace.resolve() / ".claude" / "helix-hooks" / "pretooluse_guard.py"),
                        "--policy",
                        str(workspace.resolve() / ".claude" / "helix-hooks" / "policy.json"),
                    ],
                }
            ],
        }
    ],
)
```

- [ ] **Step 2: Run the focused hook staging tests to verify they fail**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_agent_hooks.py -k "prepare_codex_hooks_stages_workspace_policy or prepare_claude_hooks_stages_workspace_settings"
```

Expected: FAIL because the checked-in templates do not yet use project-dir placeholders and the staged config files do not yet render quoted absolute Codex paths.

### Task 2: Rewrite Staged Hook Config Paths Minimally

**Files:**
- Modify: `src/helix/backends/claude_hooks.py`
- Modify: `src/helix/backends/codex_hooks.py`
- Test: `tests/test_agent_hooks.py`

- [ ] **Step 1: Replace verbatim config copies with placeholder-rendered staged JSON writers**

```python
def _write_claude_settings(settings_template: Path, settings_path: Path, project_dir: Path) -> None:
    settings = json.loads(settings_template.read_text(encoding="utf-8"))
    settings = replace_string_placeholder(settings, "${CLAUDE_PROJECT_DIR}", str(project_dir))
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
```

```python
def _write_codex_hooks_config(template_path: Path, hooks_path: Path, project_dir: Path) -> None:
    hooks_config = json.loads(template_path.read_text(encoding="utf-8"))
    hooks_config = replace_string_placeholder(hooks_config, "${CODEX_PROJECT_DIR}", str(project_dir))
    hooks_path.write_text(json.dumps(hooks_config, indent=2) + "\n", encoding="utf-8")
```

- [ ] **Step 2: Run the focused hook staging tests to verify they pass**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_agent_hooks.py -k "prepare_codex_hooks_stages_workspace_policy or prepare_claude_hooks_stages_workspace_settings"
```

Expected: PASS.

- [ ] **Step 3: Run the broader relevant verification**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_agent_hooks.py tests/test_claude_runner.py tests/test_codex_runner.py
```

Expected: PASS.
