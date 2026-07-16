import sys
import unittest
import warnings
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.skills.loader import load_operator_eval_script_module


class TorchNpuWarningsTests(unittest.TestCase):
    def test_suppresses_only_collect_env_owner_mismatch_warning(self) -> None:
        module = load_operator_eval_script_module("torch_npu_warnings")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            module.suppress_torch_npu_owner_mismatch_warning()
            warnings.warn_explicit(
                "Warning: The /home/pkg/CANN owner does not match the current owner.",
                UserWarning,
                filename="/home/pkg/ascend_py311/lib/python3.11/site-packages/torch_npu/utils/collect_env.py",
                lineno=58,
                module="torch_npu.utils.collect_env",
            )
            warnings.warn_explicit(
                "unrelated torch_npu warning",
                UserWarning,
                filename="/home/pkg/ascend_py311/lib/python3.11/site-packages/torch_npu/utils/collect_env.py",
                lineno=60,
                module="torch_npu.utils.collect_env",
            )

        self.assertEqual([str(item.message) for item in captured], ["unrelated torch_npu warning"])

    def test_fixed_remote_worker_installs_filter(self) -> None:
        worker_path = (
            Path(__file__).resolve().parents[1]
            / "skills/common/ascend-npu-run-eval/scripts/run_test_remote_worker.py"
        )
        worker_source = worker_path.read_text(encoding="utf-8")

        self.assertIn("suppress_torch_npu_owner_mismatch_warning", worker_source)


if __name__ == "__main__":
    unittest.main()
