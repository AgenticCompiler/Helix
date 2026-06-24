import json
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Optional


_node_available = shutil.which("node") is not None
_skip_if_no_node = unittest.skipUnless(_node_available, "node is not available")


class OpenCodeHookGuardTests(unittest.TestCase):
    @_skip_if_no_node
    def test_allows_in_workspace_non_protected_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            readme = workspace / "README.md"
            readme.write_text("hello\n", encoding="utf-8")

            result = _evaluate_plugin(_policy(workspace), "bash", f"sed -n '1,20p' {readme}", workspace)

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_blocks_outside_workspace_absolute_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside.txt"
            workspace.mkdir()
            outside.write_text("secret\n", encoding="utf-8")

            result = _evaluate_plugin(_policy(workspace), "bash", f"cat {outside}", workspace)

            self.assertEqual(result, {"allowed": False, "message": _DENY_MESSAGE})

    @_skip_if_no_node
    def test_blocks_outside_workspace_parent_escape_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside.txt"
            workspace.mkdir()
            outside.write_text("secret\n", encoding="utf-8")

            result = _evaluate_plugin(_policy(workspace), "bash", "sed -n '1,20p' ../outside.txt", workspace)

            self.assertEqual(result, {"allowed": False, "message": _DENY_MESSAGE})

    @_skip_if_no_node
    def test_blocks_staged_skill_script_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".opencode" / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "run-command.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")

            result = _evaluate_plugin(_policy(workspace), "bash", f"sed -n '1,80p' {script}", workspace)

            self.assertEqual(result, {"allowed": False, "message": _DENY_MESSAGE})

    @_skip_if_no_node
    def test_allows_python_one_liner_opening_protected_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".opencode" / "skills" / "skill-a" / "scripts" / "helper.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")

            result = _evaluate_plugin(
                _policy(workspace),
                "bash",
                f"python3 -c \"print(open('{script}').read())\"",
                workspace,
            )

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_allows_python_entrypoint_for_staged_helper_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".opencode" / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "run-command.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")

            result = _evaluate_plugin(
                _policy(workspace),
                "bash",
                f"python3 {script} run-test-optimize --test-file differential_test_file.py",
                workspace,
            )

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_allows_relative_python_entrypoint_for_staged_helper_script_with_redirection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".opencode" / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "run-command.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            bench_file = workspace / "bench_triton_5_MoeInitRouting.py"
            bench_file.write_text("pass\n", encoding="utf-8")
            operator_dir = workspace / "baseline"
            operator_dir.mkdir()
            operator_file = operator_dir / "opt_triton_5_MoeInitRouting.py"
            operator_file.write_text("pass\n", encoding="utf-8")

            result = _evaluate_plugin(
                _policy(workspace),
                "bash",
                "python3 .opencode/skills/ascend-npu-run-eval/scripts/run-command.py "
                "run-bench --bench-file bench_triton_5_MoeInitRouting.py "
                "--operator-file baseline/opt_triton_5_MoeInitRouting.py "
                "--bench-mode msprof 2>&1",
                workspace,
            )

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_allows_heredoc_write_when_body_mentions_protected_runtime_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()

            result = _evaluate_plugin(
                _policy(workspace),
                "bash",
                "cat > learned_lessons.md << 'ENDOFFILE'\n"
                "reference .triton-agent/state.json in prose\n"
                "ENDOFFILE",
                workspace,
            )

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_blocks_redirected_read_from_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside.txt"
            workspace.mkdir()
            outside.write_text("secret\n", encoding="utf-8")

            result = _evaluate_plugin(
                _policy(workspace),
                "bash",
                f"cat {outside} > learned_lessons.md",
                workspace,
            )

            self.assertEqual(result, {"allowed": False, "message": _DENY_MESSAGE})

    @_skip_if_no_node
    def test_blocks_read_tool_for_protected_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".opencode" / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "run-command.py"
            workspace.mkdir()
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")

            result = _evaluate_plugin(_policy(workspace), "read", str(script), workspace)

            self.assertEqual(result, {"allowed": False, "message": _DENY_MESSAGE})

    @_skip_if_no_node
    def test_allows_read_tool_for_in_workspace_non_protected_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            readme = workspace / "README.md"
            workspace.mkdir()
            readme.write_text("hello\n", encoding="utf-8")

            result = _evaluate_plugin(_policy(workspace), "read", str(readme), workspace)

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_allows_read_tool_relative_path_against_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            nested = workspace / "subdir" / "nested"
            readme = nested / "README.md"
            nested.mkdir(parents=True)
            readme.write_text("hello\n", encoding="utf-8")

            result = _evaluate_plugin(_policy(workspace), "read", "nested/README.md", workspace / "subdir")

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_blocks_triton_agent_logs_bash_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            log_file = workspace / "triton-agent-logs" / "gen-test.show-output.log"
            log_file.parent.mkdir(parents=True)
            log_file.write_text("log output\n", encoding="utf-8")

            result = _evaluate_plugin(_policy(workspace), "bash", f"sed -n '1,20p' {log_file}", workspace)

            self.assertEqual(result, {"allowed": False, "message": _DENY_MESSAGE})

    @_skip_if_no_node
    def test_blocks_triton_agent_logs_bare_relative_bash_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            log_file = workspace / "triton-agent-logs" / "gen-test.show-output.log"
            log_file.parent.mkdir(parents=True)
            log_file.write_text("log output\n", encoding="utf-8")

            result = _evaluate_plugin(
                _policy(workspace),
                "bash",
                "cat triton-agent-logs/gen-test.show-output.log",
                workspace,
            )

            self.assertEqual(result, {"allowed": False, "message": _DENY_MESSAGE})

    @_skip_if_no_node
    def test_blocks_triton_agent_logs_read_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            log_file = workspace / "triton-agent-logs" / "gen-test.show-output.log"
            log_file.parent.mkdir(parents=True)
            log_file.write_text("log output\n", encoding="utf-8")

            result = _evaluate_plugin(_policy(workspace), "read", str(log_file), workspace)

            self.assertEqual(result, {"allowed": False, "message": _DENY_MESSAGE})

    @_skip_if_no_node
    def test_allows_python_one_liner_opening_relative_triton_agent_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            log_file = workspace / "triton-agent-logs" / "gen-test.show-output.log"
            log_file.parent.mkdir(parents=True)
            log_file.write_text("log output\n", encoding="utf-8")

            result = _evaluate_plugin(
                _policy(workspace),
                "bash",
                'python3 -c "print(open(\'triton-agent-logs/gen-test.show-output.log\').read())"',
                workspace,
            )

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_allows_read_outside_triton_agent_logs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            readme = workspace / "triton-agent-readme.md"
            readme.write_text("not a log\n", encoding="utf-8")

            result = _evaluate_plugin(_policy(workspace), "bash", f"cat {readme}", workspace)

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_allows_read_from_extra_allow_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            compiler_source = Path(tmp) / "compiler-sources" / "AscendNPU-IR"
            source_file = compiler_source / "passes" / "lowering.cc"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("pass\n", encoding="utf-8")
            workspace.mkdir()

            result = _evaluate_plugin(
                _policy(workspace, extra_allow_roots=[compiler_source]),
                "bash",
                f"sed -n '1,20p' {source_file}",
                workspace,
            )

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_malformed_shell_payload_fails_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()

            result = _evaluate_plugin(_policy(workspace), "bash", None, workspace)

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_baseline_phase_allows_native_write_to_source_operator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            operator_file = workspace / "kernel.py"
            operator_file.write_text("pass\n", encoding="utf-8")
            _write_workflow_state(
                workspace,
                phase="baseline",
                baseline_status="pending",
                source_operator="kernel.py",
            )

            result = _evaluate_plugin_args(
                _policy(workspace),
                "write",
                {"filePath": str(operator_file), "content": "updated\n", "cwd": str(workspace)},
                workspace,
            )

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_awaiting_round_start_blocks_native_write_with_start_round_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            round_file = workspace / "opt-round-1" / "opt_kernel.py"
            round_file.parent.mkdir(parents=True)
            _write_workflow_state(
                workspace,
                phase="awaiting_round_start",
                baseline_status="passed",
                source_operator="kernel.py",
            )

            result = _evaluate_plugin_args(
                _policy(workspace),
                "write",
                {"filePath": str(round_file), "content": "updated\n", "cwd": str(workspace)},
                workspace,
            )

            self.assertFalse(result["allowed"])
            self.assertIn("awaiting_round_start", str(result["message"]))
            self.assertIn("ascend-npu-optimize-start-round", str(result["message"]))

    @_skip_if_no_node
    def test_round_active_allows_native_write_inside_current_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            round_file = workspace / "opt-round-2" / "attempts.md"
            round_file.parent.mkdir(parents=True)
            _write_workflow_state(
                workspace,
                phase="round_active",
                baseline_status="passed",
                source_operator="kernel.py",
                current_round=2,
                rounds={
                    "2": {
                        "status": "active",
                        "round_dir": "opt-round-2",
                        "started_at": "2026-06-23T08:00:00Z",
                        "ended_at": None,
                    }
                },
            )

            result = _evaluate_plugin_args(
                _policy(workspace),
                "write",
                {"filePath": str(round_file), "content": "updated\n", "cwd": str(workspace)},
                workspace,
            )

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_round_active_blocks_native_write_outside_current_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            operator_file = workspace / "kernel.py"
            operator_file.write_text("pass\n", encoding="utf-8")
            _write_workflow_state(
                workspace,
                phase="round_active",
                baseline_status="passed",
                source_operator="kernel.py",
                current_round=2,
                rounds={
                    "2": {
                        "status": "active",
                        "round_dir": "opt-round-2",
                        "started_at": "2026-06-23T08:00:00Z",
                        "ended_at": None,
                    }
                },
            )

            result = _evaluate_plugin_args(
                _policy(workspace),
                "write",
                {"filePath": str(operator_file), "content": "updated\n", "cwd": str(workspace)},
                workspace,
            )

            self.assertFalse(result["allowed"])
            self.assertIn("Current active round is opt-round-2", str(result["message"]))
            self.assertIn("must stay inside `opt-round-2/`", str(result["message"]))
            self.assertIn("ascend-npu-optimize-submit-round", str(result["message"]))
            self.assertNotIn("First-version scope", str(result["message"]))

    @_skip_if_no_node
    def test_missing_workflow_state_blocks_native_write_with_restart_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            round_file = workspace / "opt-round-1" / "attempts.md"
            round_file.parent.mkdir(parents=True)

            result = _evaluate_plugin_args(
                _policy(workspace),
                "write",
                {"filePath": str(round_file), "content": "updated\n", "cwd": str(workspace)},
                workspace,
            )

            self.assertFalse(result["allowed"])
            self.assertIn("temporary optimize workflow state", str(result["message"]))
            self.assertIn("restart the optimize session", str(result["message"]))
            self.assertNotIn(".triton-agent/state.json", str(result["message"]))

    @_skip_if_no_node
    def test_blocks_runtime_state_read_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            state_path = workspace / ".triton-agent" / "state.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text("{}", encoding="utf-8")

            result = _evaluate_plugin(_policy(workspace), "read", str(state_path), workspace)

            self.assertEqual(result, {"allowed": False, "message": _DENY_MESSAGE})

    @_skip_if_no_node
    def test_blocks_hidden_runtime_directory_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            runtime_dir = workspace / ".triton-agent"
            runtime_dir.mkdir(parents=True)

            result = _evaluate_plugin(_policy(workspace), "bash", "ls .triton-agent", workspace)

            self.assertEqual(result, {"allowed": False, "message": _DENY_MESSAGE})

    @_skip_if_no_node
    def test_trace_tool_call_summary_is_shorter_than_full_bash_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()

            events = _trace_plugin_args(
                _policy(workspace),
                "bash",
                {
                    "command": "python3 .opencode/skills/ascend-npu-run-eval/scripts/run-command.py "
                    "run-bench --bench-file bench_kernel.py --bench-mode msprof",
                    "cwd": str(workspace),
                },
                workspace,
            )

            tool_event = next(event for event in events if event.get("type") == "tool_call" and event.get("phase") == "start")
            command_event = next(event for event in events if event.get("type") == "command" and event.get("phase") == "start")

            self.assertEqual(tool_event["summary"], "bash: benchmark")
            self.assertEqual(
                command_event["command"],
                "python3 .opencode/skills/ascend-npu-run-eval/scripts/run-command.py "
                "run-bench --bench-file bench_kernel.py --bench-mode msprof",
            )

    @_skip_if_no_node
    def test_trace_events_omit_source_confidence_and_duration_source_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()

            events = _trace_plugin_args(
                _policy(workspace),
                "bash",
                {
                    "command": "python3 .opencode/skills/ascend-npu-run-eval/scripts/run-command.py "
                    "run-bench --bench-file bench_kernel.py --bench-mode msprof",
                    "cwd": str(workspace),
                },
                workspace,
            )

            self.assertGreaterEqual(len(events), 4)
            for event in events:
                self.assertNotIn("source", event)
                self.assertNotIn("confidence", event)
                self.assertNotIn("duration_source", event)


_DENY_MESSAGE = (
    "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
    "and do not inspect protected runner-managed files (temporary optimize runtime files, "
    "staged skill implementation files under .opencode/skills/*/scripts/, or triton-agent-logs/ output). "
    "Use the skill instructions and documented command interface instead."
)


def _policy(workspace: Path, extra_allow_roots: Optional[list[Path]] = None) -> dict[str, object]:
    root = workspace.resolve()
    allow_read_roots = [str(root)]
    if extra_allow_roots is not None:
        allow_read_roots.extend(str(path.resolve()) for path in extra_allow_roots)
    return {
        "workspace_root": str(root),
        "allow_read_roots": allow_read_roots,
        "deny_read_globs": [
            str(root / ".triton-agent"),
            str(root / ".triton-agent" / "**"),
            str(root / ".opencode" / "plugins" / "triton-agent-hook-guard.js"),
            str(root / ".opencode" / "triton-agent-hooks"),
            str(root / ".opencode" / "triton-agent-hooks" / "**"),
            str(root / "triton-agent-logs" / "**"),
            str(root / ".opencode" / "skills" / "*" / "scripts" / "**"),
            str(root / ".opencode" / "skills" / "*" / "*" / "scripts" / "**"),
        ],
        "deny_message": _DENY_MESSAGE,
    }


def _evaluate_plugin(
    policy: dict[str, object],
    tool: str,
    command: Optional[str],
    cwd: Path,
) -> dict[str, object]:
    return _evaluate_plugin_args(policy, tool, None if command is None else command, cwd)


def _evaluate_plugin_args(
    policy: dict[str, object],
    tool: str,
    args: object,
    cwd: Path,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.mjs"
        harness.write_text(_node_harness_source(), encoding="utf-8")
        payload = {
            "policy": policy,
            "tool": tool,
            "args": args,
            "cwd": str(cwd),
        }
        result = subprocess.run(
            ["node", str(harness)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            self_output = result.stderr or result.stdout
            raise AssertionError(f"node harness failed with exit {result.returncode}: {self_output}")
    return json.loads(result.stdout)


def _trace_plugin_args(
    policy: dict[str, object],
    tool: str,
    args: object,
    cwd: Path,
) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "trace-harness.mjs"
        harness.write_text(_trace_harness_source(), encoding="utf-8")
        payload = {
            "policy": policy,
            "tool": tool,
            "args": args,
            "cwd": str(cwd),
        }
        result = subprocess.run(
            ["node", str(harness)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            self_output = result.stderr or result.stdout
            raise AssertionError(f"node trace harness failed with exit {result.returncode}: {self_output}")
    return json.loads(result.stdout)


def _node_harness_source() -> str:
    plugin_path = Path(__file__).resolve().parents[1] / "hooks" / "opencode" / "triton-agent-hook-guard.js"
    return textwrap.dedent(
        f"""
        import fs from "node:fs/promises";
        import os from "node:os";
        import path from "node:path";
        import {{ TritonAgentHookGuard }} from {json.dumps(plugin_path.as_uri())};

        const rawInput = await new Promise((resolve, reject) => {{
          let data = "";
          process.stdin.setEncoding("utf8");
          process.stdin.on("data", (chunk) => {{
            data += chunk;
          }});
          process.stdin.on("end", () => resolve(data));
          process.stdin.on("error", reject);
        }});
        const input = JSON.parse(rawInput);
        const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "triton-agent-opencode-hook-"));
        const context = {{
          directory: tempRoot,
          project: {{ path: {{ root: "/" }} }},
          experimental_workspace: {{}},
        }};
        await fs.mkdir(path.join(tempRoot, ".opencode", "triton-agent-hooks"), {{ recursive: true }});
        await fs.writeFile(
          path.join(tempRoot, ".opencode", "triton-agent-hooks", "policy.json"),
          JSON.stringify(input.policy),
          "utf8",
        );

        const plugin = await TritonAgentHookGuard(context);
        const hook = plugin["tool.execute.before"];
        const args = input.args === null
          ? {{}}
          : typeof input.args === "string"
            ? input.tool === "read"
              ? {{ filePath: input.args }}
              : {{ command: input.args, cwd: input.cwd }}
            : input.args;
        const hookInput = {{ tool: input.tool, cwd: input.cwd }};
        const output = {{ args }};

        try {{
          await hook(hookInput, output);
          process.stdout.write(JSON.stringify({{ allowed: true }}));
        }} catch (error) {{
          process.stdout.write(JSON.stringify({{ allowed: false, message: error.message }}));
        }} finally {{
          await fs.rm(tempRoot, {{ recursive: true, force: true }});
        }}
        """
    )


def _trace_harness_source() -> str:
    plugin_path = Path(__file__).resolve().parents[1] / "hooks" / "opencode" / "triton-agent-hook-guard.js"
    return textwrap.dedent(
        f"""
        import fs from "node:fs/promises";
        import os from "node:os";
        import path from "node:path";
        import {{ TritonAgentHookGuard }} from {json.dumps(plugin_path.as_uri())};

        const rawInput = await new Promise((resolve, reject) => {{
          let data = "";
          process.stdin.setEncoding("utf8");
          process.stdin.on("data", (chunk) => {{
            data += chunk;
          }});
          process.stdin.on("end", () => resolve(data));
          process.stdin.on("error", reject);
        }});
        const input = JSON.parse(rawInput);
        const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "triton-agent-opencode-trace-"));
        const tracePath = path.join(tempRoot, "trace.jsonl");
        const policy = {{
          ...input.policy,
          trace: {{
            enabled: true,
            path: tracePath,
            run_id: "trace-run",
          }},
        }};
        const context = {{
          directory: tempRoot,
          project: {{ path: {{ root: "/" }} }},
          experimental_workspace: {{}},
        }};
        await fs.mkdir(path.join(tempRoot, ".opencode", "triton-agent-hooks"), {{ recursive: true }});
        await fs.writeFile(
          path.join(tempRoot, ".opencode", "triton-agent-hooks", "policy.json"),
          JSON.stringify(policy),
          "utf8",
        );

        const plugin = await TritonAgentHookGuard(context);
        const beforeHook = plugin["tool.execute.before"];
        const afterHook = plugin["tool.execute.after"];
        const args = typeof input.args === "string"
          ? input.tool === "read"
            ? {{ filePath: input.args }}
            : {{ command: input.args, cwd: input.cwd }}
          : input.args;
        const hookInput = {{ tool: input.tool, cwd: input.cwd }};
        const output = {{ args, meta: {{ tool_use_id: "tool-call-1" }} }};

        await beforeHook(hookInput, output);
        await afterHook(hookInput, output);

        const events = (await fs.readFile(tracePath, "utf8"))
          .trim()
          .split("\\n")
          .filter(Boolean)
          .map((line) => JSON.parse(line));
        process.stdout.write(JSON.stringify(events));
        await fs.rm(tempRoot, {{ recursive: true, force: true }});
        """
    )


def _write_workflow_state(
    workspace: Path,
    *,
    phase: str,
    baseline_status: str,
    source_operator: str,
    current_round: Optional[int] = None,
    rounds: Optional[dict[str, object]] = None,
) -> None:
    state_path = workspace / ".triton-agent" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "run_id": "optimize-20260623-guard",
        "phase": phase,
        "source_operator": source_operator,
        "current_round": current_round,
        "baseline": {
            "status": baseline_status,
            "submitted_at": None if baseline_status == "pending" else "2026-06-23T07:55:00Z",
        },
        "rounds": rounds or {},
    }
    state_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
