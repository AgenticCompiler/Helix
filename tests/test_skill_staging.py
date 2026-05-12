import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import CommandKind
from triton_agent.skill_staging import _apply_stage_directives, resolve_staged_skills


class SkillStagingTests(unittest.TestCase):
    def test_resolve_staged_skills_for_gen_eval(self) -> None:
        names, sources = resolve_staged_skills(CommandKind.GEN_EVAL)

        self.assertEqual(
            names,
            (
                "triton-npu-gen-eval-suite",
                "triton-npu-gen-test",
                "triton-npu-gen-bench",
                "triton-npu-run-eval",
                "triton-npu-repair-guide",
            ),
        )
        self.assertIsNone(sources)

    def test_resolve_staged_skills_for_gen_test_includes_validation_and_repair_support(self) -> None:
        names, sources = resolve_staged_skills(CommandKind.GEN_TEST)

        self.assertEqual(
            names,
            (
                "triton-npu-gen-test",
                "triton-npu-run-eval",
                "triton-npu-repair-guide",
            ),
        )
        self.assertIsNone(sources)

    def test_resolve_staged_skills_for_gen_bench_includes_validation_and_repair_support(self) -> None:
        names, sources = resolve_staged_skills(CommandKind.GEN_BENCH)

        self.assertEqual(
            names,
            (
                "triton-npu-gen-bench",
                "triton-npu-run-eval",
                "triton-npu-repair-guide",
            ),
        )
        self.assertIsNone(sources)

    def test_resolve_staged_skills_for_optimize_v2_maps_knowledge_source(self) -> None:
        names, sources = resolve_staged_skills(
            CommandKind.OPTIMIZE,
            optimize_knowledge="v2",
        )

        self.assertIn("triton-npu-optimize-knowledge", names or ())
        self.assertEqual(
            sources,
            {"triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v2"},
        )

    def test_resolve_staged_skills_for_optimize_v3_maps_knowledge_source(self) -> None:
        names, sources = resolve_staged_skills(
            CommandKind.OPTIMIZE,
            optimize_knowledge="v3",
        )

        self.assertIn("triton-npu-optimize-knowledge", names or ())
        self.assertEqual(
            sources,
            {"triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v3"},
        )

    def test_resolve_staged_skills_for_optimize_can_include_cann_ext_api(self) -> None:
        names, sources = resolve_staged_skills(
            CommandKind.OPTIMIZE,
            enable_cann_ext_api=True,
        )

        self.assertIn("triton-npu-cann-ext-api-patterns", names or ())
        self.assertIsNone(sources)

    def test_apply_stage_directives_supports_add_remove_and_full_copy(self) -> None:
        self.assertEqual(_apply_stage_directives(("+a", "+b", "-a", "+c")), ("b", "c"))
        self.assertIsNone(_apply_stage_directives(("*", "+a")))


if __name__ == "__main__":
    unittest.main()
