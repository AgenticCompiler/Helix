import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_capture_ir_module():
    script = (
        REPO_ROOT
        / "skills"
        / "triton-npu-analyze-ir"
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

    def test_build_parser_accepts_explicit_msprof_bench_case(self) -> None:
        module = _load_capture_ir_module()

        args = module.build_parser().parse_args(
            [
                "--ir-dir",
                "ir",
                "--bench-file",
                "bench.py",
                "--operator-file",
                "kernel.py",
                "--bench",
                "5",
            ]
        )

        self.assertEqual(args.bench, 5)

    def test_build_parser_rejects_bench_abbreviation(self) -> None:
        module = _load_capture_ir_module()

        with self.assertRaises(SystemExit):
            module.build_parser().parse_args(
                [
                    "--ir-dir",
                    "ir",
                    "--bench-file",
                    "bench.py",
                    "--operator-file",
                    "kernel.py",
                    "--ben",
                    "5",
                ]
            )

    def test_build_execution_command_uses_runtime_helper_for_standalone_benches(self) -> None:
        module = _load_capture_ir_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_matmul.py"
            operator_file = root / "matmul.py"
            bench_file.write_text("# bench-mode: standalone\n", encoding="utf-8")
            operator_file.write_text("def matmul():\n    pass\n", encoding="utf-8")

            command = module.build_execution_command(
                bench_file=bench_file,
                operator_file=operator_file,
            )

        self.assertEqual(
            command,
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "skills"
                    / "triton-npu-run-eval"
                    / "scripts"
                    / "standalone_bench_runtime.py"
                ),
                "run-one",
                "--bench-file",
                "bench_matmul.py",
                "--operator-file",
                "matmul.py",
            ],
        )

    def test_build_execution_command_forwards_bench_case_for_msprof_benches(self) -> None:
        module = _load_capture_ir_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_matmul.py"
            operator_file = root / "matmul.py"
            bench_file.write_text("# bench-mode: msprof\n", encoding="utf-8")
            operator_file.write_text("def matmul():\n    pass\n", encoding="utf-8")

            command = module.build_execution_command(
                bench_file=bench_file,
                operator_file=operator_file,
                bench_case=5,
            )

        self.assertEqual(
            command,
            [
                sys.executable,
                "bench_matmul.py",
                "--operator-file",
                "matmul.py",
                "--bench",
                "5",
            ],
        )

    def test_build_execution_command_rejects_bench_case_for_standalone_benches(self) -> None:
        module = _load_capture_ir_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_matmul.py"
            operator_file = root / "matmul.py"
            bench_file.write_text("# bench-mode: standalone\n", encoding="utf-8")
            operator_file.write_text("def matmul():\n    pass\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "--bench is only valid for msprof benchmark capture"):
                module.build_execution_command(
                    bench_file=bench_file,
                    operator_file=operator_file,
                    bench_case=5,
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

    def test_rewrite_compile_command_updates_ir_flags(self) -> None:
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

    def test_build_remote_replay_command_quotes_only_append_bisheng_options_value(self) -> None:
        module = _load_capture_ir_module()

        command = [
            "/opt/bin/bishengir-compile",
            "/archive/triton_dump/kernel.ttadapter.mlir",
            "--target=Ascend910_9589",
            "--append-bisheng-options=-cce-link-aicore-ll-module /opt/lib/libdevice.10.bc",
            "--mlir-print-ir-after-all",
            "--mlir-print-ir-tree-dir=/archive/bishengir_stages",
        ]

        remote_command = module._build_remote_replay_command(command, "/archive")

        self.assertIn(
            "--append-bisheng-options='-cce-link-aicore-ll-module /opt/lib/libdevice.10.bc'",
            remote_command,
        )
        self.assertNotIn(
            "'--append-bisheng-options=-cce-link-aicore-ll-module /opt/lib/libdevice.10.bc'",
            remote_command,
        )

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
        self.assertEqual(manifest["archived_input"], (archive_dir / "triton_dump" / "kernel.ttadapter.mlir").as_posix())
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
        self.assertEqual(copied_names, ["bench.py", "kernel.py", "standalone_bench_runtime.py"])
        self.assertIn(
            "python3 standalone_bench_runtime.py run-one --bench-file bench.py --operator-file kernel.py",
            remote_run.call_args_list[1].args[2],
        )
        cleanup.assert_not_called()

    def test_capture_remote_archive_forwards_bench_case_for_msprof_bench(self) -> None:
        module = _load_capture_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "archive"
            bench_file = root / "bench.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: msprof\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            def _fake_copy_directory(_spec, _remote_path, local_path, **_kwargs):
                (local_path / "triton_dump").mkdir(parents=True)
                (local_path / "triton_dump" / "kernel.ttadapter.mlir").write_text(
                    "module {}\n",
                    encoding="utf-8",
                )
                (local_path / "bishengir_stages").mkdir()
                (local_path / "all-ir.txt").write_text("stderr\n", encoding="utf-8")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-ir"),
            ), patch.object(module, "copy_file_to_remote") as copy_file, patch.object(
                module,
                "copy_directory_from_remote",
                side_effect=_fake_copy_directory,
            ), patch.object(
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
            ) as remote_run, patch.object(module, "cleanup_remote_workspace"):
                module.capture_remote_archive(
                    bench_file=bench_file,
                    operator_file=operator_file,
                    archive_dir=archive_dir,
                    remote="alice@example.com",
                    remote_workdir=None,
                    keep_remote_workdir=False,
                    bench_case=5,
                )

        copied_names = [call.args[2].rsplit("/", 1)[-1] for call in copy_file.call_args_list]
        self.assertEqual(copied_names, ["bench.py", "kernel.py"])
        self.assertIn(
            "python3 bench.py --operator-file kernel.py --bench 5",
            remote_run.call_args_list[1].args[2],
        )

    def test_run_local_replay_failure_includes_command_stdout_and_stderr(self) -> None:
        module = _load_capture_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            stderr_path = Path(tmp) / "all-ir.txt"
            with patch.object(
                module.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=1,
                    stdout="pipeline stdout\n",
                    stderr="ub overflow\n",
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "ub overflow"):
                    module._run_local_replay(["bishengir-compile", "kernel.ttadapter.mlir"], stderr_path)

            self.assertEqual(stderr_path.read_text(encoding="utf-8"), "ub overflow\n")

    def test_capture_remote_archive_replay_failure_includes_remote_stdout_and_stderr(self) -> None:
        module = _load_capture_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "archive"
            bench_file = root / "bench.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("print('bench')\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-ir"),
            ), patch.object(module, "copy_file_to_remote"), patch.object(
                module,
                "copy_directory_from_remote",
            ), patch.object(
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
                    module.make_result(
                        return_code=1,
                        stdout="Failed to run BiShengIR HIVM pipeline\n",
                        stderr="ub overflow, requires 3672064 bits\n",
                    ),
                ],
            ), patch.object(module, "cleanup_remote_workspace"):
                with self.assertRaisesRegex(RuntimeError, "ub overflow, requires 3672064 bits"):
                    module.capture_remote_archive(
                        bench_file=bench_file,
                        operator_file=operator_file,
                        archive_dir=archive_dir,
                        remote="alice@example.com",
                        remote_workdir=None,
                        keep_remote_workdir=False,
                    )


if __name__ == "__main__":
    unittest.main()
