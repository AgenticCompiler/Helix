import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentResult


class RemoteExecutionTests(unittest.TestCase):
    def test_parse_remote_spec_supports_optional_port(self) -> None:
        from triton_agent.remote_execution import parse_remote_spec

        spec = parse_remote_spec("alice@example.com:2200")

        self.assertEqual(spec.user_host, "alice@example.com")
        self.assertEqual(spec.port, 2200)

    def test_parse_remote_spec_without_port(self) -> None:
        from triton_agent.remote_execution import parse_remote_spec

        spec = parse_remote_spec("alice@example.com")

        self.assertEqual(spec.user_host, "alice@example.com")
        self.assertIsNone(spec.port)

    def test_parse_remote_spec_rejects_invalid_port(self) -> None:
        from triton_agent.remote_execution import parse_remote_spec

        with self.assertRaises(ValueError):
            parse_remote_spec("alice@example.com:notaport")

    def test_verbose_remote_copy_logs_scp_command(self) -> None:
        from triton_agent.remote_execution import copy_file_to_remote, parse_remote_spec

        stderr = StringIO()
        with patch(
            "triton_agent.remote_execution.run_process",
            return_value=AgentResult(return_code=0, stdout="", stderr=""),
        ):
            copy_file_to_remote(
                parse_remote_spec("alice@example.com:2200"),
                Path("/tmp/local.txt"),
                "/tmp/remote.txt",
                verbose=True,
                stderr=stderr,
            )

        self.assertIn("[remote]", stderr.getvalue())
        self.assertIn("scp -P 2200 /tmp/local.txt alice@example.com:/tmp/remote.txt", stderr.getvalue())

    def test_run_remote_test_keeps_workspace_when_requested(self) -> None:
        from triton_agent.test_runner import run_remote_test

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "test_kernel.py"
            operator_file = root / "kernel.py"
            test_file.write_text("# test-mode: standalone\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch(
                "triton_agent.test_runner.create_remote_workspace",
                return_value=("spec", "/tmp/remote-keep"),
            ), patch("triton_agent.test_runner.copy_file_to_remote"), patch(
                "triton_agent.test_runner.run_remote_command_streaming",
                return_value=AgentResult(return_code=0, stdout="", stderr=""),
            ), patch("triton_agent.test_runner.cleanup_remote_workspace") as cleanup:
                result, archived_result, remote_workspace = run_remote_test(
                    test_file,
                    operator_file,
                    "standalone",
                    "alice@example.com",
                    None,
                    keep_remote_workdir=True,
                )

        self.assertEqual(result.return_code, 0)
        self.assertIsNone(archived_result)
        self.assertEqual(remote_workspace, "/tmp/remote-keep")
        cleanup.assert_not_called()

    def test_run_remote_bench_cleans_workspace_by_default(self) -> None:
        from triton_agent.bench_runner import run_remote_bench

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: standalone\n# kernel: k\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch(
                "triton_agent.bench_runner.create_remote_workspace",
                return_value=("spec", "/tmp/remote-clean"),
            ), patch("triton_agent.bench_runner.copy_file_to_remote"), patch(
                "triton_agent.bench_runner.run_remote_command_streaming",
                return_value=AgentResult(return_code=0, stdout="latency-a: 1.0\n", stderr=""),
            ), patch("triton_agent.bench_runner.cleanup_remote_workspace") as cleanup:
                result, perf_path, remote_workspace = run_remote_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                    "alice@example.com",
                    None,
                )

        self.assertEqual(result.return_code, 0)
        self.assertIsNotNone(perf_path)
        self.assertEqual(remote_workspace, "/tmp/remote-clean")
        cleanup.assert_called_once_with("spec", "/tmp/remote-clean", verbose=False, stderr=None)


if __name__ == "__main__":
    unittest.main()
