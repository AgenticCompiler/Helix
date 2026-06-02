from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from triton_agent.clean import clean_workspace, discover_clean_workspaces, is_cleanable_workspace


class WorkspaceCleaningTests(unittest.TestCase):
    def test_clean_workspace_preserves_operator_and_cases_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('source')\n", encoding="utf-8")
            test_harness = workspace / "test_kernel.py"
            test_harness.write_text("# test-mode: standalone\n", encoding="utf-8")
            bench_harness = workspace / "bench_kernel.py"
            bench_harness.write_text("# bench-mode: standalone\n", encoding="utf-8")
            generated = workspace / "opt_kernel.py"
            generated.write_text("print('opt')\n", encoding="utf-8")

            result = clean_workspace(workspace, deep=False)

            self.assertTrue(operator.exists())
            self.assertTrue(test_harness.exists())
            self.assertTrue(bench_harness.exists())
            self.assertFalse(generated.exists())
            self.assertIn(generated, result.removed)

    def test_clean_workspace_includes_cases_for_deep_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('source')\n", encoding="utf-8")
            test_harness = workspace / "differential_test_kernel.py"
            test_harness.write_text("# test-mode: differential\n", encoding="utf-8")
            bench_harness = workspace / "bench_kernel.py"
            bench_harness.write_text("# bench-mode: msprof\n", encoding="utf-8")

            clean_workspace(workspace, deep=True)

            self.assertTrue(operator.exists())
            self.assertFalse(test_harness.exists())
            self.assertFalse(bench_harness.exists())

    def test_clean_workspace_removes_prof_and_extra_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            extra_info = workspace / "extra-info.json"
            extra_info.write_text("{}", encoding="utf-8")
            prof_dir = workspace / "PROF_demo"
            prof_dir.mkdir()

            result = clean_workspace(workspace, deep=False)

            self.assertFalse(extra_info.exists())
            self.assertFalse(prof_dir.exists())
            self.assertIn(extra_info, result.removed)
            self.assertIn(prof_dir, result.removed)

    def test_clean_workspace_unlinks_symlink_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            outside = Path(tmp) / "outside"
            outside.mkdir()
            real_logs = outside / "logs-real"
            real_logs.mkdir()
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            logs_link = workspace / "triton-agent-logs"
            logs_link.symlink_to(real_logs, target_is_directory=True)

            clean_workspace(workspace, deep=False)

            self.assertFalse(logs_link.exists())
            self.assertTrue(real_logs.exists())

    def test_is_cleanable_workspace_detects_generated_only_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            generated = workspace / "triton_kernel.py"
            generated.write_text("print('gen')\n", encoding="utf-8")

            self.assertTrue(is_cleanable_workspace(workspace))
            self.assertEqual(discover_clean_workspaces(workspace), [workspace])


if __name__ == "__main__":
    unittest.main()
