import sys
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.models import OptimizeStatusWorkspace
from triton_agent.optimize.render import render_optimize_status_results


class _TTYStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class OptimizeRenderTests(unittest.TestCase):
    def test_render_optimize_status_sorts_no_session_first_then_remaining_by_name(self) -> None:
        stream = StringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/alpha"),
                state="ok",
                baseline_mean=10.0,
                best_mean=8.0,
                avg_improvement=0.2,
                geomean_speedup=1.25,
                total_speedup=1.3,
                best_round="round-2",
                logged_best="round-2",
                warnings=(),
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/gamma"),
                state="no-session",
                baseline_mean=None,
                best_mean=None,
                avg_improvement=None,
                geomean_speedup=None,
                total_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=(),
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/zeta"),
                state="warning",
                baseline_mean=12.0,
                best_mean=None,
                avg_improvement=None,
                geomean_speedup=None,
                total_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=("missing perf artifact for opt-round-28",),
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
                baseline_mean=12.0,
                best_mean=None,
                avg_improvement=None,
                geomean_speedup=None,
                total_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=("missing perf artifact for opt-round-28",),
            )
        ]

        render_optimize_status_results(results, stdout=stream)

        rendered = stream.getvalue()
        self.assertIn("[WARN] layernorm", rendered)
        self.assertIn("  Warning: missing perf artifact for opt-round-28", rendered)
        self.assertIn("  Geomean speedup: unknown", rendered)
        self.assertIn("  Total speedup: unknown", rendered)
        self.assertNotIn("\033[", rendered)

    def test_render_optimize_status_uses_tty_colors_for_titles_and_faint_warnings(self) -> None:
        stream = _TTYStringIO()
        results = [
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/layernorm"),
                state="warning",
                baseline_mean=12.0,
                best_mean=None,
                avg_improvement=None,
                geomean_speedup=None,
                total_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=("missing perf artifact for opt-round-28",),
            )
        ]

        render_optimize_status_results(results, stdout=stream)

        rendered = stream.getvalue()
        self.assertIn("\033[36m[WARN] layernorm\033[0m", rendered)
        self.assertIn("\033[37m  Baseline mean: 12.000000\033[0m", rendered)
        self.assertIn("\033[37m  Geomean speedup: unknown\033[0m", rendered)
        self.assertIn("\033[37m  Total speedup: unknown\033[0m", rendered)
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
                baseline_mean=None,
                best_mean=None,
                avg_improvement=None,
                geomean_speedup=None,
                total_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=(),
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/zeta"),
                state="warning",
                baseline_mean=12.0,
                best_mean=None,
                avg_improvement=None,
                geomean_speedup=None,
                total_speedup=None,
                best_round=None,
                logged_best=None,
                warnings=("missing comparable round perf data",),
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/beta"),
                state="ok",
                baseline_mean=10.0,
                best_mean=8.0,
                avg_improvement=0.2,
                geomean_speedup=1.25,
                total_speedup=1.3,
                best_round="round-2",
                logged_best="round-2",
                warnings=(),
            ),
            OptimizeStatusWorkspace(
                workspace=Path("/tmp/gamma"),
                state="ok",
                baseline_mean=15.0,
                best_mean=9.5,
                avg_improvement=0.3,
                geomean_speedup=1.49,
                total_speedup=1.58,
                best_round="round-2",
                logged_best="round-1",
                warnings=(
                    "numeric best round != logged best. "
                    "computed speedup: 1.49x, 1.58x; logged speedup: 1.16x, 1.18x",
                ),
            ),
        ]

        render_optimize_status_results(results, stdout=stream, output_format="markdown")

        rendered = stream.getvalue()
        self.assertIn("| 名称 | Geomean speedup | Total speedup | Notes |", rendered)
        self.assertIn("| beta | 1.25x | 1.30x | - |", rendered)
        self.assertIn("| gamma | 1.49x | 1.58x | best≠log |", rendered)
        self.assertIn("| zeta | - | - | warn |", rendered)
        self.assertLess(rendered.index("| beta |"), rendered.index("| gamma |"))
        self.assertLess(rendered.index("| gamma |"), rendered.index("| zeta |"))
        self.assertNotIn("omega", rendered)
        self.assertNotIn("Summary:", rendered)


if __name__ == "__main__":
    unittest.main()
