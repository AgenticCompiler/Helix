import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.issue_detection import (
    Issue,
    coerce_stage_verdict,
    validate_and_coerce,
    write_stage_verdict,
)
from triton_agent.optimize.stages import Stage, default_stage_graph


class ValidateAndCoerceTests(unittest.TestCase):
    def test_passes_through_typed_issue(self) -> None:
        issue = Issue(issue_type="missing_autotune", severity=4)
        self.assertIs(validate_and_coerce(issue), issue)

    def test_coerces_dict_with_string_issue_type(self) -> None:
        raw = {
            "issue_type": "permute_contiguous_materialization",
            "severity": 5,
            "location": "forward():42",
            "description": "movedim().contiguous()",
            "suggested_fix": "use strided kernel",
        }
        issue = validate_and_coerce(raw)
        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertEqual(issue.issue_type, "permute_contiguous_materialization")
        self.assertEqual(issue.severity, 5)
        self.assertEqual(issue.location, "forward():42")

    def test_accepts_any_nonempty_issue_type_string(self) -> None:
        """issue_type is open — any non-empty string is valid."""
        for issue_type in [
            "missing_autotune",
            "exact-tile-no-boundary-fast-path",
            "some_new_pattern_not_in_any_enum",
            "my_custom_observation",
        ]:
            issue = validate_and_coerce({"issue_type": issue_type})
            self.assertIsNotNone(issue, f"issue_type {issue_type!r} should be accepted")
            assert issue is not None
            self.assertEqual(issue.issue_type, issue_type)

    def test_drops_empty_issue_type(self) -> None:
        self.assertIsNone(validate_and_coerce({"issue_type": ""}))
        self.assertIsNone(validate_and_coerce({"issue_type": "  "}))
        # non-empty unknown issue_type is ACCEPTED (open vocabulary)
        issue = validate_and_coerce({"issue_type": "totally_made_up_but_nonempty"})
        self.assertIsNotNone(issue)

    def test_drops_non_dict(self) -> None:
        self.assertIsNone(validate_and_coerce("not a dict"))
        self.assertIsNone(validate_and_coerce(None))

    def test_clamps_severity_to_1_5(self) -> None:
        issue = validate_and_coerce({"issue_type": "missing_autotune", "severity": 99})
        assert issue is not None
        self.assertEqual(issue.severity, 5)
        issue = validate_and_coerce({"issue_type": "missing_autotune", "severity": 0})
        assert issue is not None
        self.assertEqual(issue.severity, 1)

    def test_suggested_stage_only_honored_for_open_ended(self) -> None:
        issue = validate_and_coerce(
            {"issue_type": "open_ended", "suggested_stage": "memory_access"}
        )
        assert issue is not None
        self.assertEqual(issue.suggested_stage, Stage.MEMORY_ACCESS)
        issue = validate_and_coerce(
            {"issue_type": "missing_autotune", "suggested_stage": "memory_access"}
        )
        assert issue is not None
        self.assertIsNone(issue.suggested_stage)


class MergeIssuesTests(unittest.TestCase):
    def test_merge_issues_concatenates(self) -> None:
        from triton_agent.optimize.issue_detection import merge_issues

        a = [Issue(issue_type="missing_autotune")]
        b = [Issue(issue_type="static_range_unroll")]
        merged = merge_issues(a, b)
        self.assertEqual(len(merged), 2)


class StageVerdictTests(unittest.TestCase):
    def test_coerce_clean_and_issues(self) -> None:
        raw = {
            "round": "opt-round-1",
            "verdicts": [
                {"stage": "algorithmic", "verdict": "issues",
                 "issues": [{"issue_type": "manual_k_reduction", "severity": 4}]},
                {"stage": "boundary", "verdict": "clean", "rationale": "none"},
            ],
            "determined_stage": "algorithmic",
        }
        verdict = coerce_stage_verdict(raw)
        self.assertIsNotNone(verdict)
        assert verdict is not None
        self.assertEqual(verdict.determined_stage, Stage.ALGORITHMIC)
        self.assertEqual(set(verdict.clean_stages), {Stage.BOUNDARY})
        self.assertIn(Stage.ALGORITHMIC, verdict.issues_by_stage)
        self.assertEqual(len(verdict.issues_by_stage[Stage.ALGORITHMIC]), 1)

    def test_coerce_accepts_arbitrary_issue_types(self) -> None:
        """issue_type is open — stage-verdict can contain any issue_type string."""
        raw = {
            "verdicts": [
                {"stage": "parameterization", "verdict": "issues",
                 "issues": [{"issue_type": "exact-tile-no-boundary-fast-path", "severity": 5}]},
            ],
            "determined_stage": "parameterization",
        }
        verdict = coerce_stage_verdict(raw)
        assert verdict is not None
        issues = verdict.issues_by_stage.get(Stage.PARAMETERIZATION, [])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, "exact-tile-no-boundary-fast-path")

    def test_coerce_returns_none_for_non_object(self) -> None:
        self.assertIsNone(coerce_stage_verdict([]))
        self.assertIsNone(coerce_stage_verdict({"verdicts": "not a list"}))


if __name__ == "__main__":
    unittest.main()
