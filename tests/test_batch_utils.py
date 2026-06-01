import tempfile
import unittest
from pathlib import Path

from triton_agent.batch_utils import RESERVED_BATCH_SUBDIR_NAMES, discover_batch_workspaces

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class BatchUtilsTests(unittest.TestCase):
    def test_discover_batch_workspaces_skips_completed_subdir(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            root = Path(tmp)
            active = root / "kernel_a"
            active.mkdir()
            (active / "kernel_a.py").write_text("pass\n", encoding="utf-8")
            completed = root / "_completed"
            completed.mkdir()
            (completed / "ignored.py").write_text("pass\n", encoding="utf-8")

            discovered, failures = discover_batch_workspaces(
                root,
                resolve_operator_file=lambda workspace: next(workspace.glob("*.py")),
            )

            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0][0].name, "kernel_a")
            self.assertEqual(failures, [])
            self.assertIn("_completed", RESERVED_BATCH_SUBDIR_NAMES)


if __name__ == "__main__":
    unittest.main()
