import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.execution import MultiInvocationOptimizeController
from triton_agent.optimize.stages import Stage


def _write_round_state(
    round_dir: Path,
    *,
    correctness: str = "passed",
    benchmark: str = "passed",
    stage: str | None = None,
) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "round": round_dir.name,
        "parent_round": "round-0",
        "hypothesis": "test",
        "evidence_sources": ["benchmark"],
        "correctness_status": correctness,
        "benchmark_status": benchmark,
        "perf_artifact": "perf.txt",
        "comparison_target": "../baseline/perf.txt",
        "effective_metric_source": "kernel",
        "summary_path": "summary.md",
        "opt_note_updated": True,
    }
    if stage is not None:
        payload["stage"] = stage
    (round_dir / "round-state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class LatestAcceptedRoundDirTests(unittest.TestCase):
    def test_returns_highest_passed_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            _write_round_state(workdir / "opt-round-1")  # passed
            _write_round_state(
                workdir / "opt-round-2", benchmark="failed"
            )  # failed
            _write_round_state(workdir / "opt-round-3")  # passed
            result = MultiInvocationOptimizeController._latest_accepted_round_dir(
                workdir
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.name, "opt-round-3")

    def test_skips_failed_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            _write_round_state(workdir / "opt-round-1")  # passed
            _write_round_state(
                workdir / "opt-round-2", correctness="failed"
            )  # failed
            result = MultiInvocationOptimizeController._latest_accepted_round_dir(
                workdir
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.name, "opt-round-1")

    def test_returns_none_when_no_passed_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            _write_round_state(
                workdir / "opt-round-1", benchmark="failed"
            )
            result = MultiInvocationOptimizeController._latest_accepted_round_dir(
                workdir
            )
        self.assertIsNone(result)

    def test_returns_none_for_empty_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                MultiInvocationOptimizeController._latest_accepted_round_dir(
                    Path(tmp)
                )
            )

    def test_malformed_round_state_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "opt-round-1").mkdir()
            (workdir / "opt-round-1" / "round-state.json").write_text(
                "{not json", encoding="utf-8"
            )
            _write_round_state(workdir / "opt-round-2")  # passed
            result = MultiInvocationOptimizeController._latest_accepted_round_dir(
                workdir
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.name, "opt-round-2")


class ReadRoundStageTests(unittest.TestCase):
    def test_reads_declared_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp)
            _write_round_state(round_dir, stage="algorithmic")
            self.assertEqual(
                MultiInvocationOptimizeController._read_round_stage(round_dir),
                Stage.ALGORITHMIC,
            )

    def test_returns_none_when_stage_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp)
            _write_round_state(round_dir)  # no stage
            self.assertIsNone(
                MultiInvocationOptimizeController._read_round_stage(round_dir)
            )

    def test_returns_none_for_unknown_stage_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp)
            _write_round_state(round_dir, stage="totally_made_up")
            self.assertIsNone(
                MultiInvocationOptimizeController._read_round_stage(round_dir)
            )

    def test_returns_none_when_round_state_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                MultiInvocationOptimizeController._read_round_stage(Path(tmp))
            )

    def test_returns_none_for_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp)
            (round_dir / "round-state.json").write_text("{not json", encoding="utf-8")
            self.assertIsNone(
                MultiInvocationOptimizeController._read_round_stage(round_dir)
            )


class StageAddressedMarkerTests(unittest.TestCase):
    def test_write_marker_creates_file_with_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            MultiInvocationOptimizeController._write_stage_addressed_marker(
                round_dir, Stage.ALGORITHMIC
            )
            marker = round_dir / "stage-addressed.json"
            self.assertTrue(marker.is_file())
            data = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(
                data, {"stage": "algorithmic", "progress": True}
            )

    def test_clear_marker_removes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            MultiInvocationOptimizeController._write_stage_addressed_marker(
                round_dir, Stage.BOUNDARY
            )
            MultiInvocationOptimizeController._clear_stage_addressed_marker(round_dir)
            self.assertFalse((round_dir / "stage-addressed.json").exists())

    def test_clear_marker_is_idempotent_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            # must not raise even when no marker exists
            MultiInvocationOptimizeController._clear_stage_addressed_marker(round_dir)

    def test_scan_returns_only_marked_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            # round-1: algorithmic marked; round-2: no marker (e.g. gate-rejected);
            # round-3: boundary marked.
            MultiInvocationOptimizeController._write_stage_addressed_marker(
                workdir / "opt-round-1", Stage.ALGORITHMIC
            )
            (workdir / "opt-round-2").mkdir()
            MultiInvocationOptimizeController._write_stage_addressed_marker(
                workdir / "opt-round-3", Stage.BOUNDARY
            )
            addressed = (
                MultiInvocationOptimizeController._get_stages_addressed_from_rounds(
                    workdir
                )
            )
        self.assertEqual(addressed, {Stage.ALGORITHMIC, Stage.BOUNDARY})

    def test_scan_ignores_malformed_and_invalid_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            MultiInvocationOptimizeController._write_stage_addressed_marker(
                workdir / "opt-round-1", Stage.PARAMETERIZATION
            )
            # malformed JSON
            (workdir / "opt-round-2").mkdir()
            (workdir / "opt-round-2" / "stage-addressed.json").write_text(
                "{not json", encoding="utf-8"
            )
            # invalid stage id
            (workdir / "opt-round-3").mkdir()
            (workdir / "opt-round-3" / "stage-addressed.json").write_text(
                json.dumps({"stage": "totally_made_up"}), encoding="utf-8"
            )
            addressed = (
                MultiInvocationOptimizeController._get_stages_addressed_from_rounds(
                    workdir
                )
            )
        self.assertEqual(addressed, {Stage.PARAMETERIZATION})

    def test_scan_empty_workdir_returns_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                set(),
                MultiInvocationOptimizeController._get_stages_addressed_from_rounds(
                    Path(tmp)
                ),
            )


class ExhaustedStagesTests(unittest.TestCase):
    """Spinning guard: consecutive no-progress rounds on one stage -> exhausted."""

    @staticmethod
    def _mark(workdir: Path, round_number: int, stage: Stage, progress: bool) -> None:
        round_dir = workdir / f"opt-round-{round_number}"
        MultiInvocationOptimizeController._write_stage_addressed_marker(
            round_dir, stage, progress=progress, speedup=None
        )

    def test_five_consecutive_no_progress_exhausts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            # 5 consecutive micro_tuning no-progress rounds.
            for n in range(1, 6):
                self._mark(workdir, n, Stage.PARAMETERIZATION, progress=False)
            exhausted = MultiInvocationOptimizeController._get_exhausted_stages(workdir)
        self.assertIn(Stage.PARAMETERIZATION, exhausted)

    def test_progress_round_resets_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            # 4 no-progress, 1 progress (reset), 2 no-progress -> not exhausted.
            self._mark(workdir, 1, Stage.PARAMETERIZATION, progress=False)
            self._mark(workdir, 2, Stage.PARAMETERIZATION, progress=False)
            self._mark(workdir, 3, Stage.PARAMETERIZATION, progress=False)
            self._mark(workdir, 4, Stage.PARAMETERIZATION, progress=False)
            self._mark(workdir, 5, Stage.PARAMETERIZATION, progress=True)
            self._mark(workdir, 6, Stage.PARAMETERIZATION, progress=False)
            self._mark(workdir, 7, Stage.PARAMETERIZATION, progress=False)
            exhausted = MultiInvocationOptimizeController._get_exhausted_stages(workdir)
        self.assertNotIn(Stage.PARAMETERIZATION, exhausted)

    def test_stage_switch_resets_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            # 3 micro_tuning no-progress, switch to boundary, 3 more micro_tuning no-progress.
            for n in range(1, 4):
                self._mark(workdir, n, Stage.PARAMETERIZATION, progress=False)
            self._mark(workdir, 4, Stage.BOUNDARY, progress=False)
            for n in range(5, 8):
                self._mark(workdir, n, Stage.PARAMETERIZATION, progress=False)
            exhausted = MultiInvocationOptimizeController._get_exhausted_stages(workdir)
        # micro_tuning never reached 5 consecutive (3 + 3, broken by boundary).
        self.assertNotIn(Stage.PARAMETERIZATION, exhausted)

    def test_exhaustion_is_sticky(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            # 5 no-progress -> exhausted; then a progress round -> still exhausted.
            for n in range(1, 6):
                self._mark(workdir, n, Stage.PARAMETERIZATION, progress=False)
            self._mark(workdir, 6, Stage.PARAMETERIZATION, progress=True)
            exhausted = MultiInvocationOptimizeController._get_exhausted_stages(workdir)
        self.assertIn(Stage.PARAMETERIZATION, exhausted)

    def test_old_markers_without_progress_default_to_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            # 5 markers with NO progress field (legacy) -> default True -> not exhausted.
            for n in range(1, 6):
                round_dir = workdir / f"opt-round-{n}"
                round_dir.mkdir(parents=True, exist_ok=True)
                (round_dir / "stage-addressed.json").write_text(
                    json.dumps({"stage": "micro_tuning"}), encoding="utf-8"
                )
            exhausted = MultiInvocationOptimizeController._get_exhausted_stages(workdir)
        self.assertNotIn(Stage.PARAMETERIZATION, exhausted)

    def test_no_markers_no_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                set(),
                MultiInvocationOptimizeController._get_exhausted_stages(Path(tmp)),
            )


if __name__ == "__main__":
    unittest.main()
