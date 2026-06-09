import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


_SRC = Path(__file__).resolve().parents[1] / "src"


class FastMcpOptionalImportTests(unittest.TestCase):
    def test_cli_help_does_not_require_fastmcp(self) -> None:
        script = textwrap.dedent(
            f"""
            import builtins
            import sys

            real_import = builtins.__import__

            def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "fastmcp" or name.startswith("fastmcp."):
                    raise ModuleNotFoundError("No module named 'fastmcp'")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = blocked_import
            sys.path.insert(0, {str(_SRC)!r})

            import triton_agent.cli as cli

            try:
                cli.main(["--help"])
            except SystemExit as exc:
                raise SystemExit(0 if exc.code == 0 else exc.code)
            """
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout)
