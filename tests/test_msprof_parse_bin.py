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


if __name__ == "__main__":
    unittest.main()
