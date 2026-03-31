import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize_guidance import OptimizeGuidanceManager


class OptimizeGuidanceManagerTests(unittest.TestCase):
    def test_prepare_creates_temporary_agents_file_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare(workdir, operator, test_mode="differential", bench_mode="standalone")

            agents_path = workdir / "AGENTS.md"
            content = agents_path.read_text(encoding="utf-8")
            self.assertTrue(agents_path.exists())
            self.assertIsNone(state.backup_path)
            self.assertIn("## Triton Agent Optimize Session", content)
            self.assertIn("## Mission", content)
            self.assertIn("## Baseline", content)
            self.assertIn("## Gates", content)
            self.assertIn("## Search", content)
            self.assertIn("## Records", content)
            self.assertIn("Never edit the original operator in place.", content)
            self.assertIn("Record a baseline correctness and benchmark result", content)
            self.assertIn("Keep useful validated branches", content)
            self.assertIn("Use `differential` correctness validation", content)
            self.assertIn("Use `standalone` benchmark validation", content)
            self.assertIn("Update `attempts.md` throughout each round", content)

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])
            self.assertFalse(agents_path.exists())

    def test_prepare_backs_up_existing_agents_file_and_restores_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            agents_path = workdir / "AGENTS.md"
            agents_path.write_text("original content\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare(workdir, operator, test_mode="standalone", bench_mode="msprof")

            self.assertIsNotNone(state.backup_path)
            self.assertTrue(state.backup_path is not None and state.backup_path.exists())
            content = agents_path.read_text(encoding="utf-8")
            self.assertIn("## Triton Agent Optimize Session", content)
            self.assertIn("Use `standalone` correctness validation", content)
            self.assertIn("Use `msprof` benchmark validation", content)
            self.assertIn("Update `attempts.md` throughout each round", content)

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])
            self.assertEqual(agents_path.read_text(encoding="utf-8"), "original content\n")
            self.assertFalse(state.backup_path is not None and state.backup_path.exists())


if __name__ == "__main__":
    unittest.main()
