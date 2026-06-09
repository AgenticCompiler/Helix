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
            script = workspace / ".opencode" / "skills" / "triton-npu-run-eval" / "scripts" / "run-command.py"
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
            script = workspace / ".opencode" / "skills" / "triton-npu-run-eval" / "scripts" / "run-command.py"
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
            script = workspace / ".opencode" / "skills" / "triton-npu-run-eval" / "scripts" / "run-command.py"
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
                "python3 .opencode/skills/triton-npu-run-eval/scripts/run-command.py "
                "run-bench --bench-file bench_triton_5_MoeInitRouting.py "
                "--operator-file baseline/opt_triton_5_MoeInitRouting.py "
                "--bench-mode msprof 2>&1",
                workspace,
            )

            self.assertEqual(result, {"allowed": True})

    @_skip_if_no_node
    def test_blocks_read_tool_for_protected_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".opencode" / "skills" / "triton-npu-run-eval" / "scripts" / "run-command.py"
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


_DENY_MESSAGE = (
    "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
    "and do not inspect protected files (staged skill implementation files under "
    ".opencode/skills/*/scripts/ or triton-agent-logs/ output). "
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
            str(root / "triton-agent-logs" / "**"),
            str(root / ".opencode" / "skills" / "*" / "scripts" / "**"),
        ],
        "deny_message": _DENY_MESSAGE,
    }


def _evaluate_plugin(
    policy: dict[str, object],
    tool: str,
    command: Optional[str],
    cwd: Path,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.mjs"
        harness.write_text(_node_harness_source(), encoding="utf-8")
        payload = {
            "policy": policy,
            "tool": tool,
            "command": command,
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
        const args = input.command === null
          ? {{}}
          : input.tool === "read"
            ? {{ filePath: input.command }}
            : {{ command: input.command, cwd: input.cwd }};
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


if __name__ == "__main__":
    unittest.main()
