import json
import sys
import unittest
from unittest import mock
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import helix.status.render as status_render
from helix.status.models import OptimizeStatusRound, OptimizeStatusWorkspace
from helix.status.render import render_optimize_status_results


class _TTYStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class OptimizeRenderTests(unittest.TestCase):
    def test_render_optimize_status_best_view_dispatches_to_best_renderer(self) -> None:
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/alpha"),
                state="ok",
                avg_improvement=0.2,
                geomean_speedup=1.25,
                best_round="round-2",
                logged_best="round-2",
                warnings=(),
            )
        ]

        with mock.patch.object(status_render, "render_optimize_status_best_results", return_value=5) as render_best:
            exit_code = render_optimize_status_results(results, stdout=StringIO(), output_format="text", view="best")

        self.assertEqual(exit_code, 5)
        render_best.assert_called_once_with(results, stdout=mock.ANY, output_format="text")

    def test_render_optimize_status_best_text_dispatches_to_text_renderer(self) -> None:
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/alpha"),
                state="ok",
                avg_improvement=0.2,
                geomean_speedup=1.25,
                best_round="round-2",
                logged_best="round-2",
                warnings=(),
            )
        ]

        with mock.patch.object(status_render, "render_optimize_status_text", return_value=7) as render_text:
            exit_code = render_optimize_status_results(results, stdout=StringIO(), output_format="text", view="best")

        self.assertEqual(exit_code, 7)
        render_text.assert_called_once_with(results, stdout=mock.ANY)

    def test_render_optimize_status_sorts_no_session_first_then_remaining_by_name(self) -> None:
        stream = StringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/alpha"),
                state="ok",
                avg_improvement=0.2,
                geomean_speedup=1.25,
                best_round="round-2",
                logged_best="round-2",
                warnings=(),
                latest_verify_state=None,
                verified=False,
                verified_geomean_speedup=None,
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/gamma"),
                state="no-session",
                avg_improvement=None,
                geomean_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=(),
                latest_verify_state=None,
                verified=False,
                verified_geomean_speedup=None,
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/zeta"),
                state="warning",
                avg_improvement=None,
                geomean_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=("missing perf artifact for opt-round-28",),
                latest_verify_state=None,
                verified=False,
                verified_geomean_speedup=None,
            ),
        ]

        render_optimize_status_results(results, stdout=stream)

        rendered = stream.getvalue()
        self.assertLess(rendered.index("[NO-SESSION] gamma"), rendered.index("[OK] alpha"))
        self.assertLess(rendered.index("[OK] alpha"), rendered.index("[WARN] zeta"))

    def test_render_optimize_status_uses_plain_text_when_not_tty(self) -> None:
        stream = StringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/layernorm"),
                state="warning",
                avg_improvement=None,
                geomean_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=("missing perf artifact for opt-round-28",),
                latest_verify_state=None,
                verified=False,
                verified_geomean_speedup=None,
            )
        ]

        render_optimize_status_results(results, stdout=stream)

        rendered = stream.getvalue()
        self.assertIn("[WARN] layernorm", rendered)
        self.assertIn("  Warning: missing perf artifact for opt-round-28", rendered)
        self.assertIn("  Geomean speedup: unknown", rendered)
        self.assertNotIn("\033[", rendered)

    def test_render_optimize_status_uses_tty_colors_for_titles_and_faint_warnings(self) -> None:
        stream = _TTYStringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/layernorm"),
                state="warning",
                avg_improvement=None,
                geomean_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=("missing perf artifact for opt-round-28",),
                latest_verify_state=None,
                verified=False,
                verified_geomean_speedup=None,
            )
        ]

        render_optimize_status_results(results, stdout=stream)

        rendered = stream.getvalue()
        self.assertIn("\033[36m[WARN] layernorm\033[0m", rendered)
        self.assertIn("\033[37m  Geomean speedup: unknown\033[0m", rendered)
        self.assertIn(
            "\033[90m  Warning: missing perf artifact for opt-round-28\033[0m",
            rendered,
        )

    def test_render_optimize_status_markdown_table_filters_no_session_and_uses_dash(self) -> None:
        stream = StringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/omega"),
                state="no-session",
                avg_improvement=None,
                geomean_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=(),
                latest_verify_state=None,
                verified=False,
                verified_geomean_speedup=None,
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/zeta"),
                state="warning",
                avg_improvement=None,
                geomean_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=("missing comparable round perf data",),
                latest_verify_state=None,
                verified=False,
                verified_geomean_speedup=None,
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/beta"),
                state="ok",
                avg_improvement=0.2,
                geomean_speedup=1.25,
                best_round="round-2",
                logged_best="round-2",
                warnings=(),
                latest_verify_state=Path("/tmp/beta/opt-verify/verify-20260421-120000/verify-state.json"),
                verified=True,
                verified_geomean_speedup=1.22,
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/gamma"),
                state="ok",
                avg_improvement=0.3,
                geomean_speedup=1.49,
                best_round="round-2",
                logged_best="round-1",
                warnings=(
                    "numeric best round != logged best. "
                    "computed speedup: 1.49x, 1.58x; logged speedup: 1.16x, 1.18x",
                ),
                latest_verify_state=Path("/tmp/gamma/opt-verify/verify-20260421-120000/verify-state.json"),
                verified=False,
                verified_geomean_speedup=None,
            ),
        ]

        render_optimize_status_results(results, stdout=stream, output_format="markdown")

        rendered = stream.getvalue()
        self.assertIn(
            "| 名称 | Geomean speedup | Verified | "
            "Verified Geomean speedup | Notes |",
            rendered,
        )
        self.assertIn("| beta | 1.25x | Verified | 1.22x | - |", rendered)
        self.assertIn("| gamma | 1.49x | - |  | best≠log |", rendered)
        self.assertIn("| zeta | - | - |  | warn |", rendered)
        self.assertLess(rendered.index("| beta |"), rendered.index("| gamma |"))
        self.assertLess(rendered.index("| gamma |"), rendered.index("| zeta |"))
        self.assertNotIn("omega", rendered)
        self.assertNotIn("Summary:", rendered)

    def test_render_optimize_status_best_json_includes_all_operators(self) -> None:
        stream = StringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/nope"),
                state="no-session",
                avg_improvement=None,
                geomean_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=(),
                latest_verify_state=None,
                verified=False,
                verified_geomean_speedup=None,
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/op"),
                state="ok",
                avg_improvement=0.25,
                geomean_speedup=1.5,
                best_round="round-2",
                logged_best="round-1",
                warnings=(
                    "numeric best round != logged best. "
                    "computed speedup: 1.50x; logged speedup: 1.20x",
                ),
                latest_verify_state=Path("/tmp/op/opt-verify/verify-20260421-120000/verify-state.json"),
                verified=True,
                verified_geomean_speedup=1.4,
            ),
        ]

        render_optimize_status_results(results, stdout=stream, output_format="json", view="best")

        payload = json.loads(stream.getvalue())
        self.assertEqual([item["name"] for item in payload["operators"]], ["nope", "op"])
        self.assertEqual(payload["operators"][0]["state"], "no-session")
        self.assertEqual(payload["operators"][1]["geomean_speedup"], 1.5)
        self.assertEqual(payload["operators"][1]["verified_geomean_speedup"], 1.4)

    def test_render_optimize_status_trend_text_table_uses_round_union(self) -> None:
        stream = StringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/beta"),
                state="ok",
                avg_improvement=0.3,
                geomean_speedup=1.3,
                best_round="round-3",
                logged_best=None,
                warnings=(),
                rounds=(
                    OptimizeStatusRound("round-1", "auto", 0.1, 1.1, 9.0),
                    OptimizeStatusRound("round-3", "auto", 0.3, 1.3, 7.0),
                ),
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/alpha"),
                state="ok",
                avg_improvement=0.2,
                geomean_speedup=1.2,
                best_round="round-2",
                logged_best=None,
                warnings=(),
                rounds=(
                    OptimizeStatusRound("round-2", "auto", 0.2, 1.2, 8.0),
                ),
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/empty"),
                state="no-session",
                avg_improvement=None,
                geomean_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=(),
            ),
        ]

        render_optimize_status_results(results, stdout=stream, view="trend")

        rendered = stream.getvalue()
        self.assertIn("Name", rendered)
        self.assertIn("round-1", rendered)
        self.assertIn("round-2", rendered)
        self.assertIn("round-3", rendered)
        self.assertLess(rendered.index("alpha"), rendered.index("beta"))
        self.assertRegex(rendered, r"alpha\s+-\s+1\.20x\s+-")
        self.assertRegex(rendered, r"beta\s+1\.10x\s+-\s+1\.30x")
        self.assertNotIn("empty", rendered)

    def test_render_optimize_status_trend_markdown_table_uses_round_union(self) -> None:
        stream = StringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/beta"),
                state="ok",
                avg_improvement=0.3,
                geomean_speedup=1.3,
                best_round="round-3",
                logged_best=None,
                warnings=(),
                rounds=(
                    OptimizeStatusRound("round-1", "auto", 0.1, 1.1, 9.0),
                    OptimizeStatusRound("round-3", "auto", 0.3, 1.3, 7.0),
                ),
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/alpha"),
                state="ok",
                avg_improvement=0.2,
                geomean_speedup=1.2,
                best_round="round-2",
                logged_best=None,
                warnings=(),
                rounds=(
                    OptimizeStatusRound("round-2", "auto", 0.2, 1.2, 8.0),
                ),
            ),
        ]

        render_optimize_status_results(results, stdout=stream, output_format="markdown", view="trend")

        rendered = stream.getvalue()
        self.assertIn("| Name | round-1 | round-2 | round-3 |", rendered)
        self.assertIn("| alpha | - | 1.20x | - |", rendered)
        self.assertIn("| beta | 1.10x | - | 1.30x |", rendered)

    def test_render_optimize_status_trend_json_filters_no_session_and_fills_nulls(self) -> None:
        stream = StringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/empty"),
                state="no-session",
                avg_improvement=None,
                geomean_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=(),
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/beta"),
                state="ok",
                avg_improvement=0.3,
                geomean_speedup=1.3,
                best_round="round-3",
                logged_best=None,
                warnings=(),
                rounds=(
                    OptimizeStatusRound("round-1", "auto", 0.1, 1.1, 9.0),
                    OptimizeStatusRound("round-3", "auto", 0.3, 1.3, 7.0),
                ),
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/alpha"),
                state="ok",
                avg_improvement=0.2,
                geomean_speedup=1.2,
                best_round="round-2",
                logged_best=None,
                warnings=(),
                rounds=(
                    OptimizeStatusRound("round-2", "auto", 0.2, 1.2, 8.0),
                ),
            ),
        ]

        render_optimize_status_results(results, stdout=stream, output_format="json", view="trend")

        payload = json.loads(stream.getvalue())
        self.assertEqual(
            payload["operators"],
            [
                {
                    "name": "alpha",
                    "round_speedups": {
                        "round-1": None,
                        "round-2": 1.2,
                        "round-3": None,
                    },
                },
                {
                    "name": "beta",
                    "round_speedups": {
                        "round-1": 1.1,
                        "round-2": None,
                        "round-3": 1.3,
                    },
                },
            ],
        )

    def test_render_optimize_status_trend_html_renders_static_report(self) -> None:
        stream = StringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/beta"),
                state="ok",
                avg_improvement=0.3,
                geomean_speedup=1.3,
                best_round="round-3",
                logged_best=None,
                warnings=(),
                rounds=(
                    OptimizeStatusRound("round-1", "auto", 0.1, 1.1, 9.0),
                    OptimizeStatusRound("round-3", "auto", 0.3, 1.3, 7.0),
                ),
            ),
        ]

        render_optimize_status_results(results, stdout=stream, output_format="html", view="trend")

        rendered = stream.getvalue()
        self.assertIn("<!doctype html>", rendered.lower())
        self.assertIn("Operator Speedup Trends", rendered)
        self.assertIn("beta", rendered)
        self.assertIn("<svg", rendered)

    def test_render_optimize_status_trend_html_dispatches_results_to_html_builder(self) -> None:
        stream = StringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/beta"),
                state="ok",
                avg_improvement=0.3,
                geomean_speedup=1.3,
                best_round="round-3",
                logged_best=None,
                warnings=(),
                rounds=(
                    OptimizeStatusRound("round-1", "auto", 0.1, 1.1, 9.0),
                    OptimizeStatusRound("round-3", "auto", 0.3, 1.3, 7.0),
                ),
            ),
        ]

        with mock.patch.object(status_render, "_build_optimize_status_trend_html", return_value="<html></html>") as build_html:
            exit_code = status_render.render_optimize_status_trend_html(results, stdout=stream)

        self.assertEqual(exit_code, 0)
        build_html.assert_called_once_with(results)
        self.assertEqual(stream.getvalue(), "<html></html>\n")

    def test_render_optimize_status_best_html_is_unsupported(self) -> None:
        stream = StringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/beta"),
                state="ok",
                avg_improvement=0.2,
                geomean_speedup=1.25,
                best_round="round-2",
                logged_best="round-2",
                warnings=(),
            ),
        ]

        with self.assertRaisesRegex(ValueError, "HTML format only supports --view trend"):
            render_optimize_status_results(results, stdout=stream, output_format="html", view="best")


if __name__ == "__main__":
    unittest.main()
