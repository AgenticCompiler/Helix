import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures" / "ascend_npu_operator_profiler"
_SKILL_SCRIPTS = (
    REPO_ROOT / "skills" / "triton-npu-profile-operator" / "scripts"
)


def _load_reporter_module():
    script = _SKILL_SCRIPTS / "reporter.py"
    if str(_SKILL_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SKILL_SCRIPTS))
    spec = importlib.util.spec_from_file_location("reporter_test_module", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AscendNpuOperatorProfilerTests(unittest.TestCase):
    def test_build_report_handles_realistic_parent_layout_fixture(self) -> None:
        module = _load_reporter_module()

        profile_dir = FIXTURES_ROOT / "realistic_parent_layout"
        rendered = module.build_report(profile_dir)

        self.assertIn("Target operator: `matmul_kernel`", rendered)
        self.assertIn("Selection: inferred from the hottest `op_statistic` row", rendered)
        self.assertIn("Matched op_summary rows: `5`", rendered)
        self.assertIn("Summed task duration: `1749.438 us`", rendered)
        self.assertIn("Average task duration: `349.888 us`", rendered)
        self.assertIn("Min task duration: `346.544 us`", rendered)
        self.assertIn("Max task duration: `355.962 us`", rendered)

    def test_build_report_accepts_parent_of_mindstudio_output(self) -> None:
        module = _load_reporter_module()

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

            rendered = module.build_report(profile_dir)

        self.assertIn("Target operator: `MatMul`", rendered)
        self.assertIn("Profile directory:", rendered)

    def test_build_report_summarizes_target_operator(self) -> None:
        module = _load_reporter_module()

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

            rendered = module.build_report(profile_dir, target_op="MatMul")

        self.assertIn("Target operator: `MatMul`", rendered)
        self.assertIn("Average time: `100.0 us`", rendered)
        self.assertIn("Matched op_summary rows: `2`", rendered)
        self.assertIn("Top operators by total time", rendered)

    def test_build_report_ignores_blank_op_summary_durations(self) -> None:
        module = _load_reporter_module()

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

            rendered = module.build_report(profile_dir, target_op="MatMul")

        self.assertIn("Matched op_summary rows: `3`", rendered)
        self.assertIn("Average task duration: `100.0 us`", rendered)

    def test_build_report_includes_core_type_totals_and_transfer_signals(self) -> None:
        module = _load_reporter_module()

        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp) / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)

            (output_dir / "op_statistic_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,VectorKernel,AI_VECTOR_CORE,4,450.0,100.0,112.5,130.0,45.0",
                        "0,ScalarFixup,AI_SCALAR_CORE,6,250.0,30.0,41.667,55.0,25.0",
                        "0,CubeKernel,AI_CUBE_CORE,2,200.0,95.0,100.0,105.0,20.0",
                        "0,TransData,AI_CPU,3,80.0,20.0,26.667,30.0,8.0",
                        "0,MemcpyAsync,AI_CPU,1,20.0,20.0,20.0,20.0,2.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "op_summary_1.csv").write_text(
                "Op Name,Task Duration(us)\nVectorKernel,112.5\n",
                encoding="utf-8",
            )

            rendered = module.build_report(profile_dir, target_op="VectorKernel")

        self.assertIn("## Core type totals", rendered)
        self.assertIn("scalar", rendered.lower())
        self.assertIn("vector", rendered.lower())
        self.assertIn("cube", rendered.lower())
        self.assertIn("## Data movement hotspots", rendered)
        self.assertIn("TransData", rendered)
        self.assertIn("MemcpyAsync", rendered)

    def test_build_report_supports_json_output(self) -> None:
        module = _load_reporter_module()

        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp) / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)

            (output_dir / "op_statistic_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,VectorKernel,AI_VECTOR_CORE,4,450.0,100.0,112.5,130.0,45.0",
                        "0,ScalarFixup,AI_SCALAR_CORE,6,250.0,30.0,41.667,55.0,25.0",
                        "0,CubeKernel,AI_CUBE_CORE,2,200.0,95.0,100.0,105.0,20.0",
                        "0,TransData,AI_CPU,3,80.0,20.0,26.667,30.0,8.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "op_summary_1.csv").write_text(
                "Op Name,Task Duration(us)\nVectorKernel,112.5\n",
                encoding="utf-8",
            )

            rendered = module.build_report(
                profile_dir,
                target_op="VectorKernel",
                output_format="json",
            )

        payload = json.loads(rendered)
        self.assertEqual(payload["target_operator"], "VectorKernel")
        self.assertEqual(payload["profile_dir"], str(profile_dir.resolve()))
        self.assertIn("core_type_totals", payload)
        self.assertIn("vector", payload["core_type_totals"])
        self.assertIn("data_movement_hotspots", payload)
        self.assertEqual(payload["data_movement_hotspots"][0]["op_type"], "TransData")
        self.assertIn("top_ops", payload)
        self.assertEqual(payload["top_ops"][0]["op_type"], "VectorKernel")

    def test_build_report_emits_operator_type_bound_and_pipeline_signals(self) -> None:
        module = _load_reporter_module()

        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp) / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)

            (output_dir / "op_statistic_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,VectorKernel,AI_VECTOR_CORE,4,450.0,100.0,112.5,130.0,45.0",
                        "0,CubeKernel,AI_CUBE_CORE,2,300.0,140.0,150.0,160.0,30.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "op_summary_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,Op Name,Task Duration(us),Task Wait Time(us),Block Dim,aic_mac_ratio,aic_scalar_ratio,aic_mte1_ratio,aic_mte2_ratio,aic_mte3_ratio,aiv_vec_ratio,aiv_scalar_ratio,aiv_mte2_ratio,aiv_mte3_ratio,cube_utilization(%)",
                        "0,VectorKernel,120.0,18.0,8,0.0,8.0,0.0,6.0,5.0,62.0,22.0,12.0,4.0,5.0",
                        "0,VectorKernel,110.0,22.0,8,0.0,10.0,0.0,7.0,4.0,58.0,24.0,11.0,5.0,6.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rendered = module.build_report(
                profile_dir,
                target_op="VectorKernel",
                output_format="json",
            )

        payload = json.loads(rendered)
        self.assertIn("operator_type_guess", payload)
        self.assertEqual(payload["operator_type_guess"]["kind"], "vector")
        self.assertIn("bound_analysis", payload)
        self.assertIn(payload["bound_analysis"]["classification"], {"memory-bound", "scalar-overhead", "mixed"})
        self.assertIn("pipeline_signals", payload)
        self.assertGreater(payload["pipeline_signals"]["task_wait_time_us"]["avg"], 0.0)
        self.assertIn("aiv_vec_ratio", payload["pipeline_signals"]["ratios"])
        self.assertIn("cube_utilization_percent", payload["pipeline_signals"])

    def test_build_report_emits_task_time_api_and_msprof_signals(self) -> None:
        module = _load_reporter_module()

        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp) / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)

            (output_dir / "op_statistic_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,VectorKernel,AI_VECTOR_CORE,4,450.0,100.0,112.5,130.0,45.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "op_summary_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,Op Name,Task Duration(us)",
                        "0,VectorKernel,120.0",
                        "0,VectorKernel,110.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "task_time_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,kernel_name,kernel_type,stream_id,task_id,task_time(us),task_start(us),task_stop(us)",
                        "0,VectorKernel,AI_VECTOR_CORE,1,11,120.0,0.0,120.0",
                        "0,VectorKernel,AI_VECTOR_CORE,1,12,100.0,180.0,280.0",
                        "0,OtherKernel,AI_VECTOR_CORE,2,13,90.0,200.0,290.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "api_statistic_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,Level,API Name,Time(us),Count,Avg(us),Min(us),Max(us),Variance",
                        "0,ACL,aclrtLaunchKernel,200.0,2,100.0,95.0,105.0,25.0",
                        "0,ACL,aclopCompileAndExecute,150.0,1,150.0,150.0,150.0,0.0",
                        "0,ACL,aclrtSynchronizeStream,120.0,4,30.0,10.0,45.0,20.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "msprof_1.json").write_text(
                json.dumps(
                    [
                        {"name": "aclrtLaunchKernel", "ph": "X", "pid": 1, "tid": 10, "ts": 0, "dur": 40},
                        {"name": "VectorKernel", "ph": "X", "pid": 2, "tid": 20, "ts": 50, "dur": 120},
                        {"name": "OtherKernel", "ph": "X", "pid": 2, "tid": 21, "ts": 60, "dur": 100},
                    ]
                ),
                encoding="utf-8",
            )

            rendered = module.build_report(
                profile_dir,
                target_op="VectorKernel",
                output_format="json",
            )

        payload = json.loads(rendered)
        self.assertIn("task_timeline_signals", payload)
        self.assertEqual(payload["task_timeline_signals"]["matched_rows"], 2)
        self.assertGreater(payload["task_timeline_signals"]["max_gap_us"], 0.0)
        self.assertIn("host_api_signals", payload)
        self.assertEqual(payload["host_api_signals"]["top_apis"][0]["api_name"], "aclrtLaunchKernel")
        self.assertTrue(payload["host_api_signals"]["launch_related_present"])
        self.assertIn("msprof_timeline_signals", payload)
        self.assertGreaterEqual(payload["msprof_timeline_signals"]["stream_like_tracks"], 2)

    def test_build_report_from_real_msprof_output(self) -> None:
        module = _load_reporter_module()
        fixture = FIXTURES_ROOT / "msprof_real_output"
        rendered = module.build_report(fixture)
        self.assertIn("Target operator: `matmul_kernel`", rendered)
        self.assertIn("Selection: inferred from the hottest", rendered)
        self.assertIn("Core type: `AI_CORE`", rendered)
        self.assertIn("Operator type guess: `cube`", rendered)
        self.assertIn("Bound analysis:", rendered)
        self.assertIn("| matmul_kernel | AI_CORE |", rendered)
        self.assertIn("| cube | AI_CORE |", rendered)

    def test_build_report_from_real_msprof_json_output(self) -> None:
        module = _load_reporter_module()
        fixture = FIXTURES_ROOT / "msprof_real_output"
        rendered = module.build_report(fixture, output_format="json")
        payload = json.loads(rendered)
        self.assertEqual(payload["target_operator"], "matmul_kernel")
        self.assertTrue(payload["selection"].startswith("inferred"))
        self.assertIn("pipeline_signals", payload)
        ratios = payload["pipeline_signals"].get("ratios", {})
        self.assertIn("aic_mac_ratio", ratios)
        self.assertIn("aiv_vec_ratio", ratios)
        self.assertGreater(payload["task_timeline_signals"]["matched_rows"], 0)
        self.assertTrue(payload["host_api_signals"]["launch_related_present"])

    def test_build_report_from_real_standalone_output(self) -> None:
        module = _load_reporter_module()
        fixture = FIXTURES_ROOT / "standalone_real_output"
        rendered = module.build_report(fixture, target_op="matmul_kernel")
        self.assertIn("Target operator: `matmul_kernel`", rendered)
        self.assertIn("op_summary` file: `kernel_details.csv`", rendered)
        self.assertIn("Core type: `AI_CORE`", rendered)
        self.assertIn("| matmul_kernel | AI_CORE |", rendered)
        self.assertIn("cube | AI_CORE", rendered)
        self.assertIn("Task timeline matched rows: `0`", rendered)

    def test_build_report_from_real_standalone_json_output(self) -> None:
        module = _load_reporter_module()
        fixture = FIXTURES_ROOT / "standalone_real_output"
        rendered = module.build_report(fixture, target_op="matmul_kernel", output_format="json")
        payload = json.loads(rendered)
        self.assertEqual(payload["op_summary_file"], "kernel_details.csv")
        self.assertEqual(payload["target_operator"], "matmul_kernel")
        self.assertIn("pipeline_signals", payload)
        self.assertEqual(payload["task_timeline_signals"]["matched_rows"], 0)
        self.assertEqual(payload["task_time_file"], None)


if __name__ == "__main__":
    unittest.main()
