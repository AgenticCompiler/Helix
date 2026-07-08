import unittest

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.stages import (
    Stage,
    default_stage_graph,
    load_stage_graph,
)


_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "triton"
    / "triton-npu-optimize"
    / "references"
    / "stages.json"
)


class StageGraphContractTests(unittest.TestCase):
    """Self-consistency of the stages.json contract."""

    def setUp(self) -> None:
        self.graph = default_stage_graph()

    def test_canonical_contract_loads(self) -> None:
        graph = load_stage_graph(_CONTRACT_PATH)
        self.assertEqual(graph.stage_ids, self.graph.stage_ids)

    def test_seven_stages_present(self) -> None:
        self.assertEqual(
            set(self.graph.stage_ids),
            {
                Stage.BOUNDARY,
                Stage.PARALLEL,
                Stage.MEMORY_ACCESS,
                Stage.ALGORITHMIC,
                Stage.PIPELINE,
                Stage.COMPILE_HINTS,
                Stage.PARAMETERIZATION,
            },
        )

    def test_boundary_has_no_prereqs(self) -> None:
        self.assertEqual(self.graph.prereqs(Stage.BOUNDARY), ())

    def test_parameterization_requires_compile_hints(self) -> None:
        # parameterization's direct prereq is compile_hints (which transitively
        # brings in all earlier stages).
        self.assertIn(Stage.COMPILE_HINTS, set(self.graph.prereqs(Stage.PARAMETERIZATION)))

    def test_pipeline_requires_memory_and_algorithmic(self) -> None:
        prereqs = set(self.graph.prereqs(Stage.PIPELINE))
        self.assertIn(Stage.MEMORY_ACCESS, prereqs)
        self.assertIn(Stage.ALGORITHMIC, prereqs)
        self.assertIn(Stage.BOUNDARY, prereqs)

    def test_parallel_requires_boundary(self) -> None:
        self.assertEqual(
            set(self.graph.prereqs(Stage.PARALLEL)), {Stage.BOUNDARY}
        )

    def test_no_dependency_cycle(self) -> None:
        self.assertFalse(self.graph.has_cycle())

    def test_all_dependency_edges_reference_defined_stages(self) -> None:
        known = set(self.graph.stage_ids)
        for before, after in self.graph.dependencies:
            self.assertIn(before, known)
            self.assertIn(after, known)
            self.assertNotEqual(before, after)

    def test_every_stage_has_patterns(self) -> None:
        for descriptor in self.graph.stages:
            self.assertTrue(
                descriptor.patterns,
                f"stage {descriptor.id.value} has no patterns",
            )


class StageGraphGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = default_stage_graph()

    def test_fresh_history_only_boundary_runnable(self) -> None:
        allowable = set(self.graph.allowable_stages(addressed=set(), skipped=set()))
        self.assertEqual(allowable, {Stage.BOUNDARY})

    def test_after_boundary_three_runnable(self) -> None:
        allowable = set(
            self.graph.allowable_stages(
                addressed={Stage.BOUNDARY}, skipped=set()
            )
        )
        self.assertIn(Stage.PARALLEL, allowable)
        self.assertIn(Stage.MEMORY_ACCESS, allowable)
        self.assertIn(Stage.ALGORITHMIC, allowable)
        # pipeline still needs memory_access + algorithmic
        self.assertNotIn(Stage.PIPELINE, allowable)
        self.assertNotIn(Stage.COMPILE_HINTS, allowable)
        self.assertNotIn(Stage.PARAMETERIZATION, allowable)

    def test_pipeline_runnable_after_memory_and_algorithmic(self) -> None:
        allowable = set(
            self.graph.allowable_stages(
                addressed={Stage.BOUNDARY, Stage.MEMORY_ACCESS, Stage.ALGORITHMIC},
                skipped=set(),
            )
        )
        self.assertIn(Stage.PIPELINE, allowable)

    def test_skipped_stage_satisfies_prereq(self) -> None:
        allowable = set(
            self.graph.allowable_stages(addressed=set(), skipped={Stage.BOUNDARY})
        )
        self.assertIn(Stage.PARALLEL, allowable)

    def test_parameterization_unlocked_only_when_all_prereqs_resolved(self) -> None:
        allowable = set(
            self.graph.allowable_stages(
                addressed={Stage.BOUNDARY, Stage.MEMORY_ACCESS, Stage.ALGORITHMIC},
                skipped=set(),
            )
        )
        self.assertNotIn(Stage.PARAMETERIZATION, allowable)
        allowable = set(
            self.graph.allowable_stages(
                addressed={
                    Stage.BOUNDARY,
                    Stage.PARALLEL,
                    Stage.MEMORY_ACCESS,
                    Stage.ALGORITHMIC,
                    Stage.PIPELINE,
                    Stage.COMPILE_HINTS,
                },
                skipped=set(),
            )
        )
        self.assertIn(Stage.PARAMETERIZATION, allowable)

    def test_gate_allows_runnable_stage(self) -> None:
        result = self.graph.gate(
            Stage.BOUNDARY, addressed=set(), skipped=set()
        )
        self.assertTrue(result.allowed)
        self.assertIsNone(result.redirect_to)

    def test_gate_blocks_jump_to_parameterization(self) -> None:
        result = self.graph.gate(
            Stage.PARAMETERIZATION, addressed=set(), skipped=set()
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.redirect_to, Stage.BOUNDARY)

    def test_gate_redirects_to_first_unmet_prereq(self) -> None:
        result = self.graph.gate(
            Stage.PIPELINE,
            addressed={Stage.BOUNDARY},
            skipped=set(),
        )
        self.assertFalse(result.allowed)
        self.assertIn(result.redirect_to, self.graph.prereqs(Stage.PIPELINE))

    def test_blocked_stages_lists_first_unmet_prereq(self) -> None:
        blocked = dict(self.graph.blocked_stages(addressed=set(), skipped=set()))
        self.assertEqual(blocked[Stage.PARAMETERIZATION], Stage.BOUNDARY)
        self.assertNotIn(Stage.BOUNDARY, blocked)


if __name__ == "__main__":
    unittest.main()
