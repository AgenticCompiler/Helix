import unittest

from triton_agent.npu_affinity import (
    BatchNpuAffinityPool,
    affinity_env_for_device,
    parse_batch_npu_devices,
)


class BatchNpuAffinityTests(unittest.TestCase):
    def test_parse_batch_npu_devices_returns_none_when_unset(self) -> None:
        self.assertIsNone(parse_batch_npu_devices(None))

    def test_parse_batch_npu_devices_trims_whitespace(self) -> None:
        self.assertEqual(parse_batch_npu_devices(" 0, 1 ,2 "), ("0", "1", "2"))

    def test_parse_batch_npu_devices_rejects_empty_entries(self) -> None:
        with self.assertRaisesRegex(ValueError, "TRITON_AGENT_BATCH_NPU_DEVICES"):
            parse_batch_npu_devices("0,,1")

    def test_parse_batch_npu_devices_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_batch_npu_devices("0,1,0")

    def test_affinity_env_for_device_uses_visible_devices_and_diagnostic_env(self) -> None:
        self.assertEqual(
            affinity_env_for_device("3"),
            {
                "ASCEND_RT_VISIBLE_DEVICES": "3",
                "TRITON_AGENT_ASSIGNED_NPU": "3",
            },
        )

    def test_pool_reuses_released_devices(self) -> None:
        pool = BatchNpuAffinityPool(("0", "1"))
        with pool.acquire() as first:
            self.assertEqual(first, "0")
        with pool.acquire() as second:
            self.assertEqual(second, "1")
        with pool.acquire() as third:
            self.assertEqual(third, "0")


if __name__ == "__main__":
    unittest.main()
