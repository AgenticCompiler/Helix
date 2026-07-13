import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.eval import mcp_server as module


class RunEvalMCPServerToolMetadataTests(unittest.TestCase):
    def test_tools_expose_descriptions_and_parameter_help(self) -> None:
        server = module.create_server(slot_pool=module.NpuDevicePool(("0",)))

        async def _tool_map():
            tools = await server.list_tools()
            return {tool.name: tool for tool in tools}

        tools = asyncio.run(_tool_map())

        self.assertEqual(
            tools["run-test-baseline"].description,
            "Run the baseline operator against a test case and return any archived differential result it produces.",
        )
        self.assertEqual(
            tools["run-test-baseline"].parameters["properties"]["test_file"]["description"],
            "Absolute path to the test entry file.",
        )
        self.assertEqual(
            tools["run-test-baseline"].parameters["properties"]["operator_file"]["description"],
            "Absolute path to the operator implementation file.",
        )
        self.assertEqual(
            tools["run-test-baseline"].parameters["properties"]["test_mode"]["description"],
            "Optional test mode override. Supported values: standalone, differential.",
        )
        self.assertNotIn("accuracy_mode", tools["run-test-baseline"].parameters["properties"])
        self.assertNotIn("atol", tools["run-test-baseline"].parameters["properties"])
        self.assertNotIn("rtol", tools["run-test-baseline"].parameters["properties"])
        self.assertNotIn("compare_level", tools["run-test-baseline"].parameters["properties"])
        self.assertNotIn("ref_result", tools["run-test-baseline"].parameters["properties"])
        self.assertNotIn("ref_operator_file", tools["run-test-baseline"].parameters["properties"])

        self.assertEqual(
            tools["run-test-convert"].description,
            "Run the converted operator against a test case and compare it with reference evidence.",
        )
        self.assertEqual(
            tools["run-test-convert"].parameters["properties"]["ref_operator_file"]["description"],
            "Absolute path to the reference operator file used to produce comparison output.",
        )
        self.assertEqual(
            tools["run-test-convert"].parameters["properties"]["ref_result"]["description"],
            "Absolute path to an archived reference result used for differential comparison.",
        )
        self.assertNotIn("accuracy_mode", tools["run-test-convert"].parameters["properties"])
        self.assertNotIn("atol", tools["run-test-convert"].parameters["properties"])
        self.assertNotIn("rtol", tools["run-test-convert"].parameters["properties"])
        self.assertNotIn("compare_level", tools["run-test-convert"].parameters["properties"])

        self.assertEqual(
            tools["run-test-optimize"].description,
            "Run the optimized operator against a test case and compare it with reference evidence.",
        )
        self.assertEqual(
            tools["run-test-optimize"].parameters["properties"]["ref_operator_file"]["description"],
            "Absolute path to the reference operator file used to produce comparison output.",
        )
        self.assertNotIn("accuracy_mode", tools["run-test-optimize"].parameters["properties"])
        self.assertNotIn("atol", tools["run-test-optimize"].parameters["properties"])
        self.assertNotIn("rtol", tools["run-test-optimize"].parameters["properties"])
        self.assertNotIn("compare_level", tools["run-test-optimize"].parameters["properties"])

        self.assertEqual(
            tools["run-bench"].description,
            "Run a benchmark workload on the operator and return the generated perf artifact path.",
        )
        self.assertEqual(
            tools["run-bench"].parameters["properties"]["bench_file"]["description"],
            "Absolute path to the benchmark entry file.",
        )
        self.assertEqual(
            tools["run-bench"].parameters["properties"]["bench_mode"]["description"],
            "Optional benchmark mode override. Supported values: torch-npu-profiler, msprof, perf-counter.",
        )
        self.assertEqual(
            tools["run-bench"].parameters["properties"]["baseline_operator_file"]["description"],
            "Optional absolute path to the baseline operator file used for automatic perf comparison.",
        )
        self.assertEqual(
            tools["run-bench"].parameters["properties"]["metric_source"]["description"],
            "Optional perf comparison metric-source override for automatic baseline comparison.",
        )
        self.assertEqual(
            tools["run-bench"].parameters["properties"]["skip_latency_errors"]["description"],
            "Optional flag to keep automatic baseline comparison running when latency-error entries are present.",
        )

        self.assertEqual(
            tools["profile-bench"].description,
            "Run a benchmark profile collection and return the generated profile directory.",
        )
        self.assertEqual(
            tools["profile-bench"].parameters["properties"]["case_id"]["description"],
            "Optional benchmark case id to profile.",
        )
        self.assertEqual(
            tools["profile-bench"].parameters["properties"]["target_op"]["description"],
            "Optional operator name to highlight in the generated profile summary.",
        )
        self.assertNotIn("bench_mode", tools["profile-bench"].parameters["properties"])
        self.assertNotIn("bench", tools["profile-bench"].parameters["properties"])

        self.assertEqual(
            tools["profile-report"].description,
            "Summarize an existing profile directory without running a new benchmark.",
        )
        self.assertEqual(
            tools["profile-report"].parameters["properties"]["profile_dir"]["description"],
            "Absolute path to an existing profile output directory.",
        )

        self.assertEqual(
            tools["compare-perf"].description,
            "Compare two existing perf artifacts and report latency regressions or improvements.",
        )
        self.assertEqual(
            tools["compare-perf"].parameters["properties"]["baseline"]["description"],
            "Absolute path to the baseline perf artifact.",
        )
        self.assertEqual(
            tools["compare-perf"].parameters["properties"]["metric_source"]["description"],
            "Metric source selection for the comparison view. Supported values: auto, kernel, total-op, all.",
        )
        self.assertEqual(
            tools["compare-perf"].parameters["properties"]["skip_error"]["description"],
            "Skip parse errors encountered while reading perf artifacts and continue comparing valid entries.",
        )
        self.assertNotIn("skip_latency_errors", tools["compare-perf"].parameters["properties"])

        hidden_parameters = {"verbose", "keep_remote_workdir"}
        for tool_name in ("run-test-baseline", "run-test-convert", "run-test-optimize", "run-bench", "profile-bench"):
            properties = tools[tool_name].parameters["properties"]
            self.assertTrue(hidden_parameters.isdisjoint(properties), tool_name)
