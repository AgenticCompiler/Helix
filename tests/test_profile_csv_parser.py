import tempfile
import unittest
from pathlib import Path

from tests.run_skill_test_utils import load_profile_csv_parser_module


class ProfileCsvParserTests(unittest.TestCase):
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
