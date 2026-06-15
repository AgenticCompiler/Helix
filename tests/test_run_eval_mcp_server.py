import asyncio
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Optional, cast
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent import run_eval_mcp_server as module


class RunEvalMCPServerTests(unittest.TestCase):
    def test_build_slot_pool_ignores_workers_per_npu(self) -> None:
        self.assertEqual(
            module.build_slot_pool("0,1", 2),
            ("0", "1"),
        )

    def test_configured_slot_pool_defaults_to_device_zero_and_one_worker(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRITON_AGENT_BATCH_NPU_DEVICES": "",
                "TRITON_AGENT_BATCH_WORKERS_PER_NPU": "",
            },
            clear=False,
        ):
            pool = module.configured_slot_pool()

        seen_devices: list[str] = []
        with pool.acquire() as device:
            seen_devices.append(device)
        self.assertEqual(seen_devices, ["0"])

    def test_configured_slot_pool_ignores_workers_per_npu_when_devices_are_configured(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRITON_AGENT_BATCH_NPU_DEVICES": "0,1",
                "TRITON_AGENT_BATCH_WORKERS_PER_NPU": "3",
            },
            clear=False,
        ):
            pool = module.configured_slot_pool()

        seen_devices: list[str] = []
        with pool.acquire() as first:
            seen_devices.append(first)
            with pool.acquire() as second:
                seen_devices.append(second)

        self.assertEqual(seen_devices, ["0", "1"])

    def test_build_mcp_url_embeds_absolute_workspace_query(self) -> None:
        workspace = Path("/tmp/demo-workspace").resolve()
        url = module.build_mcp_url(port=8765, workspace=workspace)
        self.assertEqual(url, f"http://127.0.0.1:8765/mcp?workspace={workspace}")

    def test_server_registers_expected_tools(self) -> None:
        server = module.create_server(slot_pool=module.NpuDevicePool(("0",)))

        async def _list_tool_names() -> list[str]:
            tools = await server.list_tools()
            return sorted(tool.name for tool in tools)

        self.assertEqual(
            asyncio.run(_list_tool_names()),
            [
                "compare-perf",
                "profile-bench",
                "profile-report",
                "run-bench",
                "run-test-baseline",
                "run-test-optimize",
            ],
        )

    def test_run_bench_tool_uses_leased_device_and_workspace(self) -> None:
        server = module.create_server(slot_pool=module.NpuDevicePool(("0",)))
        observed: dict[str, object] = {}

        def fake_run_subcommand(
            subcommand: str,
            arguments: list[str],
            *,
            leased_device: Optional[str] = None,
            workspace: Path,
        ):
            observed["leased_device"] = leased_device
            observed["subcommand"] = subcommand
            observed["arguments"] = arguments
            observed["workspace"] = workspace
            return {
                "return_code": 0,
                "stdout": "Perf file: /tmp/kernel_perf.txt\n",
                "stderr": "",
                "perf_path": "/tmp/kernel_perf.txt",
            }

        async def _call_tool():
            with (
                patch.object(module, "_run_subcommand", side_effect=fake_run_subcommand),
                patch.object(module, "current_workspace", return_value=Path("/tmp/ws")),
            ):
                return await server.call_tool(
                    "run-bench",
                    {
                        "bench_file": "/tmp/bench_kernel.py",
                        "operator_file": "/tmp/kernel.py",
                        "bench_mode": "torch-npu-profiler",
                    },
                )

        result = asyncio.run(_call_tool())

        self.assertEqual(observed["leased_device"], "0")
        self.assertEqual(observed["subcommand"], "run-bench")
        self.assertEqual(observed["workspace"], Path("/tmp/ws"))
        arguments = cast(list[str], observed["arguments"])
        self.assertIn("--bench-file", arguments)
        self.assertIn("--operator-file", arguments)
        self.assertEqual(
            result.structured_content,
            {
                "return_code": 0,
                "stdout": "Perf file: /tmp/kernel_perf.txt\n",
                "stderr": "",
                "perf_path": "/tmp/kernel_perf.txt",
            },
        )

    def test_run_test_tools_reuse_released_slot(self) -> None:
        server = module.create_server(slot_pool=module.NpuDevicePool(("0",)))
        seen_devices: list[Optional[str]] = []

        def fake_run_subcommand(
            subcommand: str,
            arguments: list[str],
            *,
            leased_device: Optional[str] = None,
            workspace: Path,
        ):
            del workspace
            self.assertIn(subcommand, {"run-test-baseline", "run-test-optimize"})
            self.assertIn("--test-file", arguments)
            self.assertIn("--operator-file", arguments)
            seen_devices.append(leased_device)
            return {
                "return_code": 0,
                "stdout": "Archived result: /tmp/kernel_result.pt\n",
                "stderr": "",
                "archived_result": "/tmp/kernel_result.pt",
            }

        async def _call_tools() -> None:
            with (
                patch.object(module, "_run_subcommand", side_effect=fake_run_subcommand),
                patch.object(module, "current_workspace", return_value=Path("/tmp/ws")),
            ):
                await server.call_tool(
                    "run-test-baseline",
                    {
                        "test_file": "/tmp/test_kernel.py",
                        "operator_file": "/tmp/kernel.py",
                        "test_mode": "standalone",
                    },
                )
                await server.call_tool(
                    "run-test-optimize",
                    {
                        "test_file": "/tmp/differential_test_kernel.py",
                        "operator_file": "/tmp/opt_kernel.py",
                        "test_mode": "differential",
                        "baseline_operator_file": "/tmp/kernel.py",
                    },
                )

        asyncio.run(_call_tools())

        self.assertEqual(seen_devices, ["0", "0"])

    def test_compare_perf_tool_does_not_lease_device(self) -> None:
        server = module.create_server(slot_pool=module.NpuDevicePool(("0",)))
        observed: dict[str, object] = {}

        def fake_run_subcommand(
            subcommand: str,
            arguments: list[str],
            *,
            leased_device: Optional[str] = None,
            workspace: Path,
        ):
            observed["subcommand"] = subcommand
            observed["arguments"] = arguments
            observed["leased_device"] = leased_device
            observed["workspace"] = workspace
            return {"return_code": 0, "stdout": "ok\n", "stderr": ""}

        async def _call_tool():
            with (
                patch.object(module, "_run_subcommand", side_effect=fake_run_subcommand),
                patch.object(module, "current_workspace", return_value=Path("/tmp/ws")),
            ):
                return await server.call_tool(
                    "compare-perf",
                    {
                        "baseline": "/tmp/base.txt",
                        "compare": "/tmp/candidate.txt",
                        "metric_source": "kernel",
                    },
                )

        result = asyncio.run(_call_tool())

        self.assertEqual(observed["subcommand"], "compare-perf")
        self.assertIsNone(observed["leased_device"])
        self.assertEqual(observed["workspace"], Path("/tmp/ws"))
        self.assertIn("--baseline", cast(list[str], observed["arguments"]))
        self.assertIn("--compare", cast(list[str], observed["arguments"]))
        self.assertEqual(
            result.structured_content,
            {"return_code": 0, "stdout": "ok\n", "stderr": ""},
        )

    def test_compare_perf_tool_maps_skip_errors_to_legacy_subcommand_flag(self) -> None:
        server = module.create_server(slot_pool=module.NpuDevicePool(("0",)))
        observed: dict[str, object] = {}

        def fake_run_subcommand(
            subcommand: str,
            arguments: list[str],
            *,
            leased_device: Optional[str] = None,
            workspace: Path,
        ):
            observed["subcommand"] = subcommand
            observed["arguments"] = arguments
            observed["leased_device"] = leased_device
            observed["workspace"] = workspace
            return {"return_code": 0, "stdout": "ok\n", "stderr": ""}

        async def _call_tool():
            with (
                patch.object(module, "_run_subcommand", side_effect=fake_run_subcommand),
                patch.object(module, "current_workspace", return_value=Path("/tmp/ws")),
            ):
                return await server.call_tool(
                    "compare-perf",
                    {
                        "baseline": "/tmp/base.txt",
                        "compare": "/tmp/candidate.txt",
                        "skip_error": True,
                    },
                )

        asyncio.run(_call_tool())

        self.assertEqual(observed["subcommand"], "compare-perf")
        self.assertIn("--skip-error", cast(list[str], observed["arguments"]))

    def test_profile_report_tool_does_not_lease_device(self) -> None:
        server = module.create_server(slot_pool=module.NpuDevicePool(("0",)))
        observed: dict[str, object] = {}

        def fake_run_subcommand(
            subcommand: str,
            arguments: list[str],
            *,
            leased_device: Optional[str] = None,
            workspace: Path,
        ):
            observed["subcommand"] = subcommand
            observed["arguments"] = arguments
            observed["leased_device"] = leased_device
            observed["workspace"] = workspace
            return {"return_code": 0, "stdout": "report\n", "stderr": ""}

        async def _call_tool():
            with (
                patch.object(module, "_run_subcommand", side_effect=fake_run_subcommand),
                patch.object(module, "current_workspace", return_value=Path("/tmp/ws")),
            ):
                return await server.call_tool(
                    "profile-report",
                    {
                        "profile_dir": "/tmp/PROF_0001",
                        "target_op": "MatMul",
                        "format": "json",
                        "top": 3,
                    },
                )

        result = asyncio.run(_call_tool())

        self.assertEqual(observed["subcommand"], "profile-report")
        self.assertIsNone(observed["leased_device"])
        self.assertEqual(observed["workspace"], Path("/tmp/ws"))
        self.assertIn("--profile-dir", cast(list[str], observed["arguments"]))
        self.assertIn("--target-op", cast(list[str], observed["arguments"]))
        self.assertEqual(
            result.structured_content,
            {"return_code": 0, "stdout": "report\n", "stderr": ""},
        )

    def test_current_workspace_requires_query_parameter(self) -> None:
        request = type("Request", (), {"scope": {"query_string": b""}})()
        with patch.object(module, "get_http_request", return_value=request):
            with self.assertRaisesRegex(ValueError, "workspace query parameter is required"):
                module.current_workspace()

    def test_current_workspace_requires_absolute_path(self) -> None:
        request = type("Request", (), {"scope": {"query_string": b"workspace=relative/path"}})()
        with patch.object(module, "get_http_request", return_value=request):
            with self.assertRaisesRegex(ValueError, "must be an absolute path"):
                module.current_workspace()

    def test_current_workspace_resolves_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            request = type("Request", (), {"scope": {"query_string": f"workspace={workspace}".encode("utf-8")}})()
            with patch.object(module, "get_http_request", return_value=request):
                self.assertEqual(module.current_workspace(), workspace)

    def test_serve_http_server_forever_prints_endpoint_and_closes_on_keyboard_interrupt(self) -> None:
        events: list[str] = []

        class _FakeServer:
            port = 8765
            endpoint = "http://127.0.0.1:8765/mcp"

            def url_for_workspace(self, workspace: Path) -> str:
                return module.build_mcp_url(port=self.port, workspace=workspace)

            def close(self) -> None:
                events.append("closed")

        fake_server = _FakeServer()

        def _stop_forever() -> None:
            raise KeyboardInterrupt

        stdout = StringIO()
        with (
            patch.object(module, "start_http_server", return_value=fake_server),
            patch.object(module, "_serve_forever", side_effect=_stop_forever),
            redirect_stdout(stdout),
        ):
            exit_code = module.serve_http_server_forever(port=0)

        self.assertEqual(exit_code, 0)
        self.assertEqual(events, ["closed"])
        rendered = stdout.getvalue()
        self.assertIn(fake_server.endpoint, rendered)
        self.assertIn("workspace=", rendered)


if __name__ == "__main__":
    unittest.main()
