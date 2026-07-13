import sys
import tempfile
import unittest
from os import environ
from pathlib import Path
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.generation.batch import (
    is_batch_gen_eval_operator_candidate,
    run_gen_eval_batch,
    resolve_batch_gen_eval_operator_file,
    summarize_batch_gen_eval_failure,
)
from helix.generation.models import GenerationOptions
from helix.models import AgentResult


class GenerationBatchHelpersTests(unittest.TestCase):
    def test_resolve_batch_gen_eval_operator_file_excludes_generated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")
            (workspace / "test_kernel.py").write_text("", encoding="utf-8")
            (workspace / "differential_test_kernel.py").write_text("", encoding="utf-8")
            (workspace / "bench_kernel.py").write_text("", encoding="utf-8")
            (workspace / "opt_kernel.py").write_text("", encoding="utf-8")
            (workspace / "__init__.py").write_text("", encoding="utf-8")

            resolved = resolve_batch_gen_eval_operator_file(workspace)

            self.assertEqual(resolved, workspace / "kernel.py")

    def test_resolve_batch_gen_eval_operator_file_rejects_multiple_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "a.py").write_text("print('a')\n", encoding="utf-8")
            (workspace / "b.py").write_text("print('b')\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "multiple candidate operator files"):
                resolve_batch_gen_eval_operator_file(workspace)

    def test_resolve_batch_gen_eval_operator_file_applies_operator_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('a')\n", encoding="utf-8")
            (workspace / "kernel_fp16.py").write_text("print('b')\n", encoding="utf-8")

            resolved = resolve_batch_gen_eval_operator_file(
                workspace,
                operator_filter="*_fp16.py",
            )

            self.assertEqual(resolved, workspace / "kernel_fp16.py")

    def test_resolve_batch_gen_eval_operator_file_treats_empty_filter_as_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('a')\n", encoding="utf-8")

            resolved = resolve_batch_gen_eval_operator_file(
                workspace,
                operator_filter="",
            )

            self.assertEqual(resolved, workspace / "kernel.py")

    def test_resolve_batch_gen_eval_operator_file_treats_blank_filter_as_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('a')\n", encoding="utf-8")

            resolved = resolve_batch_gen_eval_operator_file(
                workspace,
                operator_filter="   \t",
            )

            self.assertEqual(resolved, workspace / "kernel.py")

    def test_resolve_batch_gen_eval_operator_file_reports_no_match_after_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('a')\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"found no candidate operator file after applying --operator-filter 'triton_\*\.py'",
            ):
                resolve_batch_gen_eval_operator_file(
                    workspace,
                    operator_filter="triton_*.py",
                )

    def test_resolve_batch_gen_eval_operator_file_reports_multiple_matches_after_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel_a.py").write_text("print('a')\n", encoding="utf-8")
            (workspace / "kernel_b.py").write_text("print('b')\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"found multiple candidate operator files after applying --operator-filter 'kernel_\*\.py':",
            ):
                resolve_batch_gen_eval_operator_file(
                    workspace,
                    operator_filter="kernel_*.py",
                )

    def test_is_batch_gen_eval_operator_candidate_filters_non_operator_names(self) -> None:
        workspace = Path("/tmp")

        self.assertTrue(is_batch_gen_eval_operator_candidate(workspace / "kernel.py"))
        self.assertFalse(is_batch_gen_eval_operator_candidate(workspace / "test_kernel.py"))
        self.assertFalse(
            is_batch_gen_eval_operator_candidate(workspace / "differential_test_kernel.py")
        )
        self.assertFalse(is_batch_gen_eval_operator_candidate(workspace / "bench_kernel.py"))
        self.assertFalse(is_batch_gen_eval_operator_candidate(workspace / "opt_kernel.py"))
        self.assertFalse(is_batch_gen_eval_operator_candidate(workspace / "__init__.py"))
        self.assertFalse(is_batch_gen_eval_operator_candidate(workspace / "kernel.txt"))

    def test_summarize_batch_gen_eval_failure_prefers_last_non_blank_stderr_line(self) -> None:
        result = AgentResult(return_code=1, stdout="stdout line\n", stderr="\nfirst\nsecond\n")

        summary = summarize_batch_gen_eval_failure(result)

        self.assertEqual(summary, "second")

    def test_summarize_batch_gen_eval_failure_falls_back_to_return_code(self) -> None:
        result = AgentResult(return_code=7, stdout="   \n", stderr="")

        summary = summarize_batch_gen_eval_failure(result)

        self.assertEqual(summary, "gen-eval exited with return code 7")

    def test_run_gen_eval_batch_applies_user_prompt_to_each_workspace_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("kernel_a", "kernel_b"):
                workspace = root / name
                workspace.mkdir()
                (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            prompts: list[str] = []

            def _fake_run(request, stdout=None, stderr=None):
                del stdout, stderr
                prompts.append(request.prompt)
                return AgentResult(return_code=0, stdout="ok", stderr="")

            exit_code = run_gen_eval_batch(
                root,
                GenerationOptions(
                    interact=False,
                    verbose=False,
                    stream_output=False,
                    force_overwrite=False,
                    agent_name="codex",
                    remote=None,
                    remote_workdir=None,
                    min_rounds=None,
                    continue_optimize=False,
                    output=None,
                    test_mode="differential",
                    bench_mode="torch-npu-profiler",
                    prompt="Avoid changing numerics.",
                ),
                max_concurrency=1,
                run_request=_fake_run,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(prompts), 2)
            for prompt in prompts:
                self.assertIn("Additional user instructions:", prompt)
                self.assertIn("Avoid changing numerics.", prompt)

    def test_run_gen_eval_batch_assigns_affinity_env_per_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("alpha", "beta"):
                workspace = root / name
                workspace.mkdir()
                (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            seen_devices: list[Optional[str]] = []

            def _fake_run(request, stdout=None, stderr=None):
                del stdout, stderr
                seen_devices.append((request.extra_env or {}).get("ASCEND_RT_VISIBLE_DEVICES"))
                return AgentResult(return_code=0, stdout="ok", stderr="")

            with patch.dict(environ, {"HELIX_BATCH_NPU_DEVICES": "0,1"}, clear=False):
                exit_code = run_gen_eval_batch(
                    root,
                    GenerationOptions(
                        interact=False,
                        verbose=False,
                        stream_output=False,
                        force_overwrite=False,
                        agent_name="codex",
                        remote=None,
                        remote_workdir=None,
                        min_rounds=None,
                        continue_optimize=False,
                        output=None,
                        test_mode="differential",
                        bench_mode="torch-npu-profiler",
                        prompt=None,
                    ),
                    max_concurrency=2,
                    run_request=_fake_run,
                )

            self.assertEqual(exit_code, 0)
            self.assertCountEqual(seen_devices, ["0", "1"])

    def test_run_gen_eval_batch_allows_same_device_when_workers_per_npu_gt_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("alpha", "beta"):
                workspace = root / name
                workspace.mkdir()
                (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            seen_devices: list[Optional[str]] = []

            def _fake_run(request, stdout=None, stderr=None):
                del stdout, stderr
                seen_devices.append((request.extra_env or {}).get("ASCEND_RT_VISIBLE_DEVICES"))
                return AgentResult(return_code=0, stdout="ok", stderr="")

            env_vars = {
                "HELIX_BATCH_NPU_DEVICES": "0",
                "HELIX_BATCH_WORKERS_PER_NPU": "2",
            }
            with patch.dict(environ, env_vars, clear=False):
                exit_code = run_gen_eval_batch(
                    root,
                    GenerationOptions(
                        interact=False,
                        verbose=False,
                        stream_output=False,
                        force_overwrite=False,
                        agent_name="codex",
                        remote=None,
                        remote_workdir=None,
                        min_rounds=None,
                        continue_optimize=False,
                        output=None,
                        test_mode="differential",
                        bench_mode="torch-npu-profiler",
                        prompt=None,
                    ),
                    max_concurrency=2,
                    run_request=_fake_run,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(seen_devices), 2)
            self.assertEqual(seen_devices, ["0", "0"])

    def test_run_gen_eval_batch_does_not_inject_affinity_env_when_mcp_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("alpha", "beta"):
                workspace = root / name
                workspace.mkdir()
                (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            seen_devices: list[Optional[str]] = []

            def _fake_run(request, stdout=None, stderr=None):
                del stdout, stderr
                seen_devices.append((request.extra_env or {}).get("ASCEND_RT_VISIBLE_DEVICES"))
                return AgentResult(return_code=0, stdout="ok", stderr="")

            with patch.dict(environ, {"HELIX_BATCH_NPU_DEVICES": "0,1"}, clear=False):
                exit_code = run_gen_eval_batch(
                    root,
                    GenerationOptions(
                        interact=False,
                        verbose=False,
                        stream_output=False,
                        force_overwrite=False,
                        agent_name="codex",
                        remote=None,
                        remote_workdir=None,
                        min_rounds=None,
                        continue_optimize=False,
                        output=None,
                        test_mode="differential",
                        bench_mode="torch-npu-profiler",
                        prompt=None,
                        enable_mcp=True,
                    ),
                    max_concurrency=2,
                    run_request=_fake_run,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_devices, [None, None])

    def test_run_gen_eval_batch_passes_explicit_affinity_into_managed_mcp_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            class _DummyScope:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            with patch("helix.generation.batch.managed_mcp_scope", return_value=_DummyScope()) as mocked:
                exit_code = run_gen_eval_batch(
                    root,
                    GenerationOptions(
                        interact=False,
                        verbose=False,
                        stream_output=False,
                        force_overwrite=False,
                        agent_name="codex",
                        remote=None,
                        remote_workdir=None,
                        min_rounds=None,
                        continue_optimize=False,
                        output=None,
                        test_mode="differential",
                        bench_mode="torch-npu-profiler",
                        npu_devices="0,1",
                        workers_per_npu="2",
                        prompt=None,
                        enable_mcp=True,
                    ),
                    max_concurrency=1,
                    run_request=lambda request, stdout=None, stderr=None: AgentResult(
                        return_code=0,
                        stdout="ok",
                        stderr="",
                    ),
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(npu_devices="0,1", workers_per_npu="2")

    def test_run_gen_eval_batch_preserves_affinity_capacity_validation_without_mcp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            with patch("helix.generation.batch.validate_batch_affinity_capacity") as mocked:
                exit_code = run_gen_eval_batch(
                    root,
                    GenerationOptions(
                        interact=False,
                        verbose=False,
                        stream_output=False,
                        force_overwrite=False,
                        agent_name="codex",
                        remote=None,
                        remote_workdir=None,
                        min_rounds=None,
                        continue_optimize=False,
                        output=None,
                        test_mode="differential",
                        bench_mode="torch-npu-profiler",
                        prompt=None,
                        enable_mcp=False,
                    ),
                    max_concurrency=1,
                    run_request=lambda request, stdout=None, stderr=None: AgentResult(
                        return_code=0,
                        stdout="ok",
                        stderr="",
                    ),
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once()

    def test_run_gen_eval_batch_validates_physical_capacity_when_mcp_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            with patch("helix.generation.batch.validate_batch_affinity_capacity") as mocked:
                exit_code = run_gen_eval_batch(
                    root,
                    GenerationOptions(
                        interact=False,
                        verbose=False,
                        stream_output=False,
                        force_overwrite=False,
                        agent_name="codex",
                        remote=None,
                        remote_workdir=None,
                        min_rounds=None,
                        continue_optimize=False,
                        output=None,
                        test_mode="differential",
                        bench_mode="torch-npu-profiler",
                        npu_devices="0",
                        workers_per_npu="2",
                        prompt=None,
                        enable_mcp=True,
                    ),
                    max_concurrency=8,
                    run_request=lambda request, stdout=None, stderr=None: AgentResult(
                        return_code=0,
                        stdout="ok",
                        stderr="",
                    ),
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                ("0",),
                max_concurrency=8,
                workers_per_npu_raw="2",
                ignore_workers_per_npu=True,
            )


if __name__ == "__main__":
    unittest.main()
