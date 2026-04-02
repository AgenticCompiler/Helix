import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures" / "ascend_npu_operator_profiler"


def _load_profile_summary_module():
    script = (
        REPO_ROOT
        / "skills"
        / "ascend-npu-operator-profiler"
        / "scripts"
        / "profile_summary.py"
    )
    spec = importlib.util.spec_from_file_location("profile_summary_test_module", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AscendNpuOperatorProfilerTests(unittest.TestCase):
    def test_build_profile_report_handles_realistic_parent_layout_fixture(self) -> None:
        module = _load_profile_summary_module()

        profile_dir = FIXTURES_ROOT / "realistic_parent_layout"
        rendered = module.build_profile_report(profile_dir)

        self.assertIn("Target operator: `matmul_kernel`", rendered)
        self.assertIn("Selection: inferred from the hottest `op_statistic` row", rendered)
        self.assertIn("Matched op_summary rows: `5`", rendered)
        self.assertIn("Summed task duration: `1749.438 us`", rendered)
        self.assertIn("Average task duration: `349.888 us`", rendered)
        self.assertIn("Min task duration: `346.544 us`", rendered)
        self.assertIn("Max task duration: `355.962 us`", rendered)

    def test_build_profile_report_accepts_parent_of_mindstudio_output(self) -> None:
        module = _load_profile_summary_module()

        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp)
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)

            (output_dir / "op_statistic_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,MatMul,AI_CORE,4,400.0,90.0,100.0,110.0,80.0",
                        "0,ElementWise,AI_VECTOR_CORE,8,100.0,10.0,12.5,20.0,20.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "op_summary_1.csv").write_text(
                "\n".join(
                    [
                        "Model Name,Op Name,Task Duration(us)",
                        "demo,MatMul,101.0",
                        "demo,MatMul,99.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rendered = module.build_profile_report(profile_dir)

        self.assertIn("Target operator: `MatMul`", rendered)
        self.assertIn("Profile directory:", rendered)

    def test_build_profile_report_summarizes_target_operator(self) -> None:
        module = _load_profile_summary_module()

        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp) / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)

            (output_dir / "op_statistic_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,MatMul,AI_CORE,4,400.0,90.0,100.0,110.0,80.0",
                        "0,ElementWise,AI_VECTOR_CORE,8,100.0,10.0,12.5,20.0,20.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "op_summary_1.csv").write_text(
                "\n".join(
                    [
                        "Model Name,Op Name,Task Duration(us)",
                        "demo,MatMul,101.0",
                        "demo,MatMul,99.0",
                        "demo,ElementWise,12.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rendered = module.build_profile_report(profile_dir, target_op="MatMul")

        self.assertIn("Target operator: `MatMul`", rendered)
        self.assertIn("Average time: `100.0 us`", rendered)
        self.assertIn("Matched op_summary rows: `2`", rendered)
        self.assertIn("Top operators by total time", rendered)

    def test_build_profile_report_ignores_blank_op_summary_durations(self) -> None:
        module = _load_profile_summary_module()

        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp) / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)

            (output_dir / "op_statistic_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,MatMul,AI_CORE,3,300.0,90.0,100.0,110.0,100.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "op_summary_1.csv").write_text(
                "\n".join(
                    [
                        "Op Name,Task Duration(us)",
                        "MatMul,120.0",
                        "MatMul,",
                        "MatMul,80.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rendered = module.build_profile_report(profile_dir, target_op="MatMul")

        self.assertIn("Matched op_summary rows: `3`", rendered)
        self.assertIn("Average task duration: `100.0 us`", rendered)


if __name__ == "__main__":
    unittest.main()
