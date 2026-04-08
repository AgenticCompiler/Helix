import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_capture_ir_module():
    script = (
        REPO_ROOT
        / "skills"
        / "ascend-operator-ir-analyzer"
        / "scripts"
        / "capture_ir.py"
    )
    spec = importlib.util.spec_from_file_location("capture_ir_test_module", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AscendOperatorIrAnalyzerTests(unittest.TestCase):
    def test_build_parser_accepts_ir_dir_argument(self) -> None:
        module = _load_capture_ir_module()

        args = module.build_parser().parse_args(
            ["--ir-dir", "ir", "--bench-file", "bench.py", "--operator-file", "kernel.py"]
        )

        self.assertEqual(args.ir_dir, "ir")
        self.assertEqual(args.bench_file, "bench.py")
        self.assertEqual(args.operator_file, "kernel.py")

    def test_build_execution_command_uses_bench_and_operator_names(self) -> None:
        module = _load_capture_ir_module()

        command = module.build_execution_command(
            bench_file=Path("/tmp/work/bench_matmul.py"),
            operator_file=Path("/tmp/work/matmul.py"),
        )

        self.assertEqual(
            command,
            ["python3", "bench_matmul.py", "--operator-file", "matmul.py"],
        )

    def test_extract_capture_details_reads_dump_path_and_compile_command(self) -> None:
        module = _load_capture_ir_module()

        details = module.extract_capture_details(
            "\n".join(
                [
                    "warmup",
                    "Dumping intermediate results to /tmp/triton-dump",
                    "[DEBUG] cmd_list: /opt/bin/bishengir-compile /tmp/kernel.ttadapter.mlir --target=Ascend910",
                ]
            )
        )

        self.assertEqual(details.dumped_ir_dir, "/tmp/triton-dump")
        self.assertEqual(
            details.compile_command,
            [
                "/opt/bin/bishengir-compile",
                "/tmp/kernel.ttadapter.mlir",
                "--target=Ascend910",
            ],
        )

    def test_extract_capture_details_requires_both_markers(self) -> None:
        module = _load_capture_ir_module()

        with self.assertRaisesRegex(RuntimeError, "Dumping intermediate results"):
            module.extract_capture_details("[DEBUG] cmd_list: /opt/bin/bishengir-compile /tmp/kernel.ttadapter.mlir")

        with self.assertRaisesRegex(RuntimeError, r"\[DEBUG\] cmd_list"):
            module.extract_capture_details("Dumping intermediate results to /tmp/triton-dump")

    def test_rewrite_compile_command_quotes_append_bisheng_options_and_updates_ir_flags(self) -> None:
        module = _load_capture_ir_module()

        command = [
            "/opt/bin/bishengir-compile",
            "/tmp/kernel.ttadapter.mlir",
            "--target=Ascend910_9589",
            "--append-bisheng-options=-cce-link-aicore-ll-module",
            "/opt/lib/libdevice.10.bc",
            "--bishengir-print-ir-after=hivm-inject-sync",
            "-o",
            "/tmp/kernel",
        ]

        rewritten = module.rewrite_compile_command(
            command,
            archived_input=Path("/archive/triton_dump/kernel.ttadapter.mlir"),
            stage_dir=Path("/archive/bishengir_stages"),
        )

        self.assertEqual(rewritten[0], "/opt/bin/bishengir-compile")
        self.assertEqual(rewritten[1], "/archive/triton_dump/kernel.ttadapter.mlir")
        self.assertIn(
            "--append-bisheng-options=-cce-link-aicore-ll-module /opt/lib/libdevice.10.bc",
            rewritten,
        )
        self.assertNotIn("--bishengir-print-ir-after=hivm-inject-sync", rewritten)
        self.assertIn("--mlir-print-ir-after-all", rewritten)
        self.assertIn("--mlir-print-ir-tree-dir=/archive/bishengir_stages", rewritten)

    def test_write_manifest_records_commands_and_paths(self) -> None:
        module = _load_capture_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp) / "archive"
            archive_dir.mkdir()
            manifest_path = module.write_manifest(
                archive_dir,
                bench_file=Path("/tmp/work/bench_matmul.py"),
                operator_file=Path("/tmp/work/matmul.py"),
                rendered_command=["python3", "bench_matmul.py", "--operator-file", "matmul.py"],
                remote="alice@example.com:2200",
                dumped_ir_dir="/tmp/triton-dump",
                original_compile_command=["/opt/bin/bishengir-compile", "/tmp/kernel.ttadapter.mlir"],
                replay_compile_command=["/opt/bin/bishengir-compile", "/archive/kernel.ttadapter.mlir"],
                archived_input=archive_dir / "triton_dump" / "kernel.ttadapter.mlir",
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["bench_file"], "/tmp/work/bench_matmul.py")
        self.assertEqual(manifest["operator_file"], "/tmp/work/matmul.py")
        self.assertEqual(
            manifest["rendered_command"],
            ["python3", "bench_matmul.py", "--operator-file", "matmul.py"],
        )
        self.assertEqual(manifest["remote"], "alice@example.com:2200")
        self.assertEqual(manifest["dumped_ir_dir"], "/tmp/triton-dump")
        self.assertEqual(manifest["archived_input"], str(archive_dir / "triton_dump" / "kernel.ttadapter.mlir"))
        self.assertEqual(manifest["replay_compile_command"], ["/opt/bin/bishengir-compile", "/archive/kernel.ttadapter.mlir"])

    def test_capture_remote_archive_keeps_workspace_when_requested(self) -> None:
        module = _load_capture_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "archive"
            bench_file = root / "bench.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("print('bench')\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            def _fake_copy_directory(_spec, _remote_path, local_path, **_kwargs):
                (local_path / "triton_dump").mkdir(parents=True)
                (local_path / "triton_dump" / "kernel.ttadapter.mlir").write_text(
                    "module {}\n",
                    encoding="utf-8",
                )
                (local_path / "bishengir_stages").mkdir()
                (local_path / "all-ir.txt").write_text("stderr\n", encoding="utf-8")
                module.write_manifest(
                    local_path,
                    bench_file=bench_file,
                    operator_file=operator_file,
                    rendered_command=["python3", "bench.py", "--operator-file", "kernel.py"],
                    remote="alice@example.com",
                    dumped_ir_dir="/tmp/triton-dump",
                    original_compile_command=["bishengir-compile", "/tmp/kernel.ttadapter.mlir"],
                    replay_compile_command=["bishengir-compile", "/tmp/archive/kernel.ttadapter.mlir"],
                    archived_input=local_path / "triton_dump" / "kernel.ttadapter.mlir",
                )

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-ir"),
            ), patch.object(module, "copy_file_to_remote") as copy_file, patch.object(
                module,
                "copy_directory_from_remote",
                side_effect=_fake_copy_directory,
            ) as copy_back, patch.object(
                module,
                "run_remote_command_buffered",
                side_effect=[
                    module.make_result(return_code=0, stdout="ok\n", stderr=""),
                    module.make_result(
                        return_code=0,
                        stdout=(
                            "Dumping intermediate results to /tmp/triton-dump\n"
                            "[DEBUG] cmd_list: bishengir-compile /tmp/kernel.ttadapter.mlir --target=Ascend910\n"
                        ),
                        stderr="",
                    ),
                    module.make_result(return_code=0, stdout="ok\n", stderr=""),
                    module.make_result(return_code=0, stdout="ok\n", stderr=""),
                ],
            ) as remote_run, patch.object(module, "cleanup_remote_workspace") as cleanup:
                manifest_path, remote_workspace = module.capture_remote_archive(
                    bench_file=bench_file,
                    operator_file=operator_file,
                    archive_dir=archive_dir,
                    remote="alice@example.com",
                    remote_workdir=None,
                    keep_remote_workdir=True,
                )

        self.assertEqual(remote_workspace, "/tmp/remote-ir")
        self.assertEqual(manifest_path, archive_dir / "capture-manifest.json")
        self.assertGreaterEqual(remote_run.call_count, 2)
        copy_back.assert_called_once_with(
            "spec",
            "/tmp/remote-ir/archive",
            archive_dir,
            verbose=False,
            stderr=None,
        )
        copied_names = [call.args[2].rsplit("/", 1)[-1] for call in copy_file.call_args_list]
        self.assertEqual(copied_names, ["bench.py", "kernel.py"])
        self.assertIn(
            "python3 bench.py --operator-file kernel.py",
            remote_run.call_args_list[1].args[2],
        )
        cleanup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
