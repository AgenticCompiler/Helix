import builtins
import importlib.util
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_parse_bin_module(*, block_tabulate: bool):
    script = REPO_ROOT / "skills" / "triton-npu-profile-operator" / "scripts" / "parse_bin.py"
    spec = importlib.util.spec_from_file_location("parse_bin_test_module", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    if not block_tabulate:
        spec.loader.exec_module(module)
        return module

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tabulate":
            raise ModuleNotFoundError("No module named 'tabulate'")
        return original_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        spec.loader.exec_module(module)
    return module


class MsprofParseBinTests(unittest.TestCase):
    def test_parse_bin_loads_without_tabulate_dependency(self) -> None:
        module = _load_parse_bin_module(block_tabulate=True)

        info = module.BaseInfo(
            name="demo",
            duration=12.5,
            op_type="vector",
            block_dim=4,
            head_name=("Stage", "Cycles"),
            block_detail=(("Load", "10"), ("Compute", "20")),
        )

        rendered = info.print_info()

        self.assertIn("**Name:** demo", rendered)
        self.assertIn("| Stage | Cycles |", rendered)
        self.assertIn("| --- | --- |", rendered)
        self.assertIn("| Load | 10 |", rendered)
        self.assertIn("| Compute | 20 |", rendered)

    def test_summarize_results_exposes_structured_binary_signal_sections(self) -> None:
        module = _load_parse_bin_module(block_tabulate=False)

        results = [
            {
                "json": {
                    "name": "demo",
                    "duration": 12.5,
                    "op_type": "vector",
                    "block_dim": 4,
                    "block_detail": {
                        "head_name": ["Stage", "Cycles"],
                        "row": [{"value": ["Load", "10"]}, {"value": ["Compute", "20"]}],
                    },
                }
            },
            {
                "json": {
                    "subblock_detail": [
                        {"block_id": "0", "name": "Vector Pipe", "value": "82.5"},
                        {"block_id": "1", "name": "Scalar Pipe", "value": "18.0"},
                    ]
                }
            },
            {
                "json": {
                    "subblock_detail": [
                        {"block_id": "0", "name": "Vector Wait", "value": "33.0"},
                        {"block_id": "0", "name": "Vector Compute Data Size", "value": "1024"},
                        {"block_id": "0", "name": "Vector Add", "value": "100"},
                    ]
                }
            },
            {
                "json": {
                    "core_memory_map": [
                        {
                            "core_no": 0,
                            "advice": ["Prefer reuse in UB"],
                            "memory_unit": [
                                {"memory_path": 12, "bandwidth": "120.0", "request": "16"},
                                {"memory_path": 14, "bandwidth": "80.0", "request": "32"},
                            ],
                            "L2cache": {"hit_ratio": "87.5"},
                            "Vector": {"ratio": "65.0"},
                        }
                    ]
                }
            },
            {
                "json": {
                    "table_per_block": [
                        {
                            "block_id": 0,
                            "advice": ["Investigate load imbalance"],
                            "table_detail": [
                                {
                                    "table_name": "UB",
                                    "header_name": ["Name", "Load"],
                                    "row": [{"name": "Vector0", "value": ["75.0"]}],
                                }
                            ],
                        }
                    ]
                }
            },
        ]

        summary = module.summarize_results(results)

        self.assertEqual(summary["base_info"]["name"], "demo")
        self.assertEqual(summary["base_info"]["op_type"], "vector")
        self.assertIn("pipe_utilization", summary)
        self.assertEqual(summary["pipe_utilization"]["top_pipe"]["name"], "Vector Pipe")
        self.assertIn("instruction_wait_signals", summary)
        self.assertEqual(summary["instruction_wait_signals"]["vector_wait_by_block"]["0"], 33.0)
        self.assertIn("memory_path_signals", summary)
        self.assertEqual(summary["memory_path_signals"]["l2_hit_ratio_by_core"]["0"], 87.5)
        self.assertIn("memory_load_signals", summary)
        self.assertEqual(summary["memory_load_signals"]["tables_by_block"]["0"][0]["table_name"], "UB")


if __name__ == "__main__":
    unittest.main()
