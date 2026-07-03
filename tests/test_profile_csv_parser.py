import tempfile
import unittest
from dataclasses import fields
from io import StringIO
from pathlib import Path

from tests.run_skill_test_utils import load_profile_csv_parser_module


class ProfileCsvParserTests(unittest.TestCase):
    def test_op_statistic_csv_row_type_has_named_fields(self) -> None:
        module = load_profile_csv_parser_module()

        self.assertEqual(
            [field.name for field in fields(module.OpStatisticCsvRow)],
            ["op_type", "native_avg_time_us", "total_time_us"],
        )

    def test_kernel_details_aggregation_type_has_named_fields(self) -> None:
        module = load_profile_csv_parser_module()

        self.assertEqual(
            [field.name for field in fields(module.KernelDetailsAggregation)],
            [
                "total_time_us",
                "total_duration_us_by_op",
                "op_order",
                "total_duration_us_by_step",
            ],
        )

    def test_parse_op_statistic_csv_reads_plain_file(self) -> None:
        module = load_profile_csv_parser_module()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "op_statistic.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,KernelA,AI_CORE,5,20,1,4.0,6,80",
                        "0,KernelB,AI_VECTOR_CORE,5,5,0.5,1.0,2,20",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parsed = module.parse_op_statistic_csv(csv_path)

        self.assertEqual(parsed.source_path, csv_path)
        self.assertEqual(
            parsed.ops,
            [
                {"op_type": "KernelA", "avg_time_us": 4.0},
                {"op_type": "KernelB", "avg_time_us": 1.0},
            ],
        )
        self.assertEqual(parsed.total_time_us, 25.0)

    def test_parse_op_statistic_csv_uses_count_derived_step_proxy_without_active_count(self) -> None:
        module = load_profile_csv_parser_module()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "op_statistic.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,engram_hash_kernel_kernel,MIX_AIC,45,9227.36,204.78,205.052,205.34,56.884",
                        "0,BroadcastTo,AI_VECTOR_CORE,180,1913.52,5.72,10.63,16.4,11.796",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parsed = module.parse_op_statistic_csv(csv_path)

        self.assertEqual(
            parsed.ops,
            [
                {"op_type": "engram_hash_kernel_kernel", "avg_time_us": 205.052444},
                {"op_type": "BroadcastTo", "avg_time_us": 42.522667},
            ],
        )
        self.assertAlmostEqual(parsed.total_op_avg_time_us, 247.575111, places=6)

    def test_parse_op_statistic_csv_prefers_active_count_when_it_differs_from_inferred_step_count(self) -> None:
        module = load_profile_csv_parser_module()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "op_statistic.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,engram_hash_kernel_kernel,MIX_AIC,45,9227.36,204.78,205.052,205.34,56.884",
                        "0,BroadcastTo,AI_VECTOR_CORE,180,1913.52,5.72,10.63,16.4,11.796",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            stderr = StringIO()
            parsed = module.parse_op_statistic_csv(
                csv_path,
                active_count=50,
                verbose=True,
                stderr=stderr,
            )

        self.assertEqual(
            parsed.ops,
            [
                {"op_type": "engram_hash_kernel_kernel", "avg_time_us": 184.5472},
                {"op_type": "BroadcastTo", "avg_time_us": 38.2704},
            ],
        )
        self.assertAlmostEqual(parsed.total_op_avg_time_us, 222.8176, places=6)
        self.assertIn("inferred_step_count=45 differs from active_count=50", stderr.getvalue())
        self.assertIn("using active_count=50 as the benchmark-provided step proxy", stderr.getvalue())

    def test_parse_kernel_details_csv_reads_plain_file(self) -> None:
        module = load_profile_csv_parser_module()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "kernel_details.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "Name,Duration(us),Wait Time(us),Block Dim",
                        "KernelA,9.0,0,3",
                        "KernelA,6.0,1,3",
                        "KernelB,1.5,2,1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parsed = module.parse_kernel_details_csv(csv_path, active_count=3)

        self.assertEqual(parsed.source_path, csv_path)
        self.assertEqual(
            parsed.ops,
            [
                {"op_type": "KernelA", "avg_time_us": 5.0},
                {"op_type": "KernelB", "avg_time_us": 0.5},
            ],
        )
        self.assertEqual(parsed.total_time_us, 16.5)


if __name__ == "__main__":
    unittest.main()
