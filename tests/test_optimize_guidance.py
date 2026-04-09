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
            state = manager.prepare(
                workdir,
                operator,
                test_mode="differential",
                bench_mode="standalone",
                agent_name="codex",
            )

            agents_path = workdir / "AGENTS.md"
            content = agents_path.read_text(encoding="utf-8")
            self.assertTrue(agents_path.exists())
            self.assertEqual(state.guidance_path, agents_path)
            self.assertIsNone(state.backup_path)
            self.assertIn("## Triton Agent Optimize Session", content)
            self.assertIn("## Mission", content)
            self.assertIn("## Baseline", content)
            self.assertIn("## Gates", content)
            self.assertIn("## Search", content)
            self.assertIn("## Records", content)
            self.assertIn("Never edit the original operator in place.", content)
            self.assertIn("Record a baseline correctness and benchmark result", content)
            self.assertIn("Check whether correctness tests and benchmark cases already exist", content)
            self.assertIn("Do not regenerate them when reusable harnesses are already present.", content)
            self.assertIn("Keep useful validated branches", content)
            self.assertIn("Use `differential` correctness validation", content)
            self.assertIn("Use `standalone` benchmark validation", content)
            self.assertIn("Update `attempts.md` throughout each round", content)
            self.assertIn("Write a short diagnosis summary before the first code-changing round.", content)
            self.assertIn("State the hypothesis, why it may help, and what evidence supports it before editing code.", content)
            self.assertIn("If you skip profiling or IR capture for a round, explain why the existing evidence is sufficient.", content)
            self.assertIn("Record `Geomean speedup` and `Total speedup`", content)
            self.assertIn("Use `Geomean speedup` as the headline metric", content)
            self.assertIn("Optimize only the existing NPU Triton operator implementation.", content)
            self.assertIn("Do not replace Triton operator calls with direct PyTorch operator", content)
            self.assertIn("If you need to generate or regenerate correctness tests, include multiple test cases", content)
            self.assertIn("If you need to generate or regenerate benchmark cases, include multiple benchmark cases", content)
            self.assertIn("Start by consulting the staged `optimize` skill", content)
            self.assertIn("Use the staged `ascend-npu-operator-profiler` skill", content)
            self.assertIn("Use the staged `ascend-operator-ir-analyzer` skill", content)

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])
            self.assertFalse(agents_path.exists())

    def test_prepare_uses_claude_file_and_restores_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            guidance_path = workdir / "CLAUDE.md"
            guidance_path.write_text("original content\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare(
                workdir,
                operator,
                test_mode="standalone",
                bench_mode="msprof",
                agent_name="claude",
            )

            self.assertIsNotNone(state.backup_path)
            self.assertTrue(state.backup_path is not None and state.backup_path.exists())
            self.assertEqual(state.guidance_path, guidance_path)
            content = guidance_path.read_text(encoding="utf-8")
            self.assertIn("# CLAUDE.md", content)
            self.assertIn("## Triton Agent Optimize Session", content)
            self.assertIn("Use `standalone` correctness validation", content)
            self.assertIn("Use `msprof` benchmark validation", content)
            self.assertIn("Update `attempts.md` throughout each round", content)
            self.assertIn("Check whether correctness tests and benchmark cases already exist", content)
            self.assertIn("Do not regenerate them when reusable harnesses are already present.", content)
            self.assertIn("Write a short diagnosis summary before the first code-changing round.", content)
            self.assertIn("State the hypothesis, why it may help, and what evidence supports it before editing code.", content)
            self.assertIn("If you skip profiling or IR capture for a round, explain why the existing evidence is sufficient.", content)
            self.assertIn("Record `Geomean speedup` and `Total speedup`", content)
            self.assertIn("Use `Geomean speedup` as the headline metric", content)
            self.assertIn("Optimize only the existing NPU Triton operator implementation.", content)
            self.assertIn("Do not replace Triton operator calls with direct PyTorch operator", content)
            self.assertIn("If you need to generate or regenerate correctness tests, include multiple test cases", content)
            self.assertIn("If you need to generate or regenerate benchmark cases, include multiple benchmark cases", content)
            self.assertIn("Start by consulting the staged `optimize` skill", content)
            self.assertIn("Use the staged `ascend-npu-operator-profiler` skill", content)
            self.assertIn("Use the staged `ascend-operator-ir-analyzer` skill", content)

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])
            self.assertEqual(guidance_path.read_text(encoding="utf-8"), "original content\n")
            self.assertFalse(state.backup_path is not None and state.backup_path.exists())

    def test_prepare_mentions_strict_analysis_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare(
                workdir,
                operator,
                test_mode="differential",
                bench_mode="standalone",
                agent_name="codex",
                require_analysis=True,
            )

            content = state.guidance_path.read_text(encoding="utf-8")
            self.assertIn("Before the first code-changing round, gather profiling or IR-backed evidence.", content)
            self.assertIn("Do not begin with blind tiling or launch-parameter search.", content)

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
