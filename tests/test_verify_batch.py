import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.cli import main
from helix.cli import build_parser
from helix.commands.verify import handle_verify_batch
from helix.remote.env import remote_target_env_name, remote_workdir_env_name
from helix.verify.batch import run_verify_batch
from helix.verify.core import VerifyOptions
from helix.verify.core import VerifyResult


class VerifyBatchTests(unittest.TestCase):
    def _write_verify_state(
        self,
        workspace: Path,
        verify_name: str,
        *,
        test_status: str = "passed",
        baseline_bench_status: str = "passed",
        best_bench_status: str = "passed",
        compare_status: str = "passed",
    ) -> Path:
        verify_dir = workspace / "opt-verify" / verify_name
        verify_dir.mkdir(parents=True)
        state_path = verify_dir / "verify-state.json"
        state_path.write_text(
            json.dumps(
                {
                    "verify-result": {
                        "test": {"status": test_status},
                        "rerun_baseline_bench": {"status": baseline_bench_status},
                        "rerun_best_bench": {"status": best_bench_status},
                        "compare_perf": {"status": compare_status},
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return state_path

    def test_run_verify_batch_reuses_latest_verify_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
            latest_state = self._write_verify_state(workspace, "verify-20260421-120000")
            stream = StringIO()

            with patch("helix.verify.batch.prepare_verify_target") as prepare_target:
                with patch("helix.verify.batch.run_verify") as run_verify:
                    exit_code = run_verify_batch(root, stdout=stream)

            self.assertEqual(exit_code, 0)
            prepare_target.assert_not_called()
            run_verify.assert_not_called()
            self.assertIn("[OK] matmul: reused verify-state.json", stream.getvalue())
            self.assertIn(str(latest_state), stream.getvalue())

    def test_run_verify_batch_force_verify_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
            self._write_verify_state(workspace, "verify-20260421-120000")
            stream = StringIO()
            verify_dir = workspace / "opt-verify" / "verify-20260421-130000"
            verify_dir.mkdir(parents=True)
            state_path = verify_dir / "verify-state.json"

            with patch(
                "helix.verify.batch.prepare_verify_target",
                return_value=object(),
            ) as prepare_target:
                with patch(
                    "helix.verify.batch.run_verify",
                    return_value=VerifyResult(
                        return_code=0,
                        verify_dir=verify_dir,
                        state_path=state_path,
                    ),
                ) as run_verify:
                    exit_code = run_verify_batch(root, force_verify=True, stdout=stream)

            self.assertEqual(exit_code, 0)
            prepare_target.assert_called_once_with(workspace)
            run_verify.assert_called_once()
            self.assertIn("[OK] matmul: verified", stream.getvalue())
            self.assertIn(str(state_path), stream.getvalue())

    def test_run_verify_batch_propagates_shared_remote_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
            stream = StringIO()
            verify_dir = workspace / "opt-verify" / "verify-20260421-130000"
            verify_dir.mkdir(parents=True)
            state_path = verify_dir / "verify-state.json"
            options = VerifyOptions(
                remote="alice@example.com",
                remote_workdir="/tmp/helix",
                keep_remote_workdir=True,
                verbose=True,
            )

            with patch(
                "helix.verify.batch.prepare_verify_target",
                return_value=object(),
            ):
                with patch(
                    "helix.verify.batch.run_verify",
                    return_value=VerifyResult(
                        return_code=0,
                        verify_dir=verify_dir,
                        state_path=state_path,
                    ),
                ) as run_verify:
                    exit_code = run_verify_batch(
                        root,
                        force_verify=True,
                        stdout=stream,
                        options=options,
                    )

            self.assertEqual(exit_code, 0)
            _target, passed_options = run_verify.call_args.args
            self.assertEqual(passed_options.remote, "alice@example.com")
            self.assertEqual(passed_options.remote_workdir, "/tmp/helix")
            self.assertTrue(passed_options.keep_remote_workdir)
            self.assertTrue(passed_options.verbose)

    def test_run_verify_batch_skips_non_verifiable_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "broken"
            workspace.mkdir()
            stream = StringIO()

            with patch(
                "helix.verify.batch.prepare_verify_target",
                side_effect=ValueError("missing baseline"),
            ) as prepare_target:
                exit_code = run_verify_batch(root, stdout=stream)

            self.assertEqual(exit_code, 0)
            prepare_target.assert_called_once_with(workspace)
            self.assertIn("[SKIP] broken: missing baseline", stream.getvalue())

    def test_run_verify_batch_continues_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok_workspace = root / "ok"
            bad_workspace = root / "bad"
            ok_workspace.mkdir()
            bad_workspace.mkdir()
            stream = StringIO()

            def prepare_side_effect(workspace: Path) -> Path:
                return workspace

            def run_side_effect(target: Path, options: object) -> VerifyResult:
                workspace = target
                if workspace == bad_workspace:
                    return VerifyResult(
                        return_code=1,
                        verify_dir=workspace / "opt-verify" / "verify-20260421-120000",
                        state_path=workspace / "opt-verify" / "verify-20260421-120000" / "verify-state.json",
                    )
                return VerifyResult(
                    return_code=0,
                    verify_dir=workspace / "opt-verify" / "verify-20260421-120000",
                    state_path=workspace / "opt-verify" / "verify-20260421-120000" / "verify-state.json",
                )

            with patch("helix.verify.batch.prepare_verify_target", side_effect=prepare_side_effect):
                with patch("helix.verify.batch.run_verify", side_effect=run_side_effect):
                    exit_code = run_verify_batch(root, force_verify=True, stdout=stream)

            self.assertEqual(exit_code, 1)
            rendered = stream.getvalue()
            self.assertIn("[FAIL] bad: verify exited with return code 1", rendered)
            self.assertIn("[OK] ok: verified", rendered)

    def test_main_verify_batch_dispatches_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "matmul").mkdir()

            with patch(
                "helix.commands.verify.run_verify_batch",
                return_value=0,
            ) as run_batch:
                exit_code = main(["verify-batch", "-i", str(root), "--force-verify"])

            self.assertEqual(exit_code, 0)
            run_batch.assert_called_once()

    def test_handle_verify_batch_passes_shared_remote_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parser = build_parser()
            args = Namespace(
                input=str(root),
                force_verify=True,
                remote="alice@example.com",
                remote_workdir="/tmp/helix",
                keep_remote_workdir=True,
                verbose=True,
            )

            with patch(
                "helix.commands.verify.run_verify_batch",
                return_value=0,
            ) as run_batch:
                exit_code = handle_verify_batch(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(run_batch.call_args.args[0], root.resolve())
            self.assertTrue(run_batch.call_args.kwargs["force_verify"])
            options = run_batch.call_args.kwargs["options"]
            self.assertEqual(options.remote, "alice@example.com")
            self.assertEqual(options.remote_workdir, "/tmp/helix")
            self.assertTrue(options.keep_remote_workdir)
            self.assertTrue(options.verbose)

    def test_handle_verify_batch_uses_remote_env_when_flag_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parser = build_parser()
            args = Namespace(
                input=str(root),
                force_verify=False,
                remote=None,
                remote_workdir=None,
                keep_remote_workdir=False,
                verbose=False,
            )

            with patch.dict(
                "os.environ",
                {
                    remote_target_env_name(): "alice@example.com",
                    remote_workdir_env_name(): "/tmp/helix",
                },
                clear=False,
            ):
                with patch(
                    "helix.commands.verify.run_verify_batch",
                    return_value=0,
                ) as run_batch:
                    exit_code = handle_verify_batch(parser, args)

            self.assertEqual(exit_code, 0)
            options = run_batch.call_args.kwargs["options"]
            self.assertEqual(options.remote, "alice@example.com")
            self.assertEqual(options.remote_workdir, "/tmp/helix")


if __name__ == "__main__":
    unittest.main()
