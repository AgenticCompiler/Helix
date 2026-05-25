import os
import unittest
from unittest import mock

from triton_agent.npu_affinity import (
    BatchNpuAffinityPool,
    affinity_env_for_device,
    configured_batch_npu_slots,
    configured_batch_workers_per_npu,
    parse_batch_npu_devices,
    parse_batch_workers_per_npu,
    validate_batch_affinity_capacity,
)


class BatchNpuAffinityTests(unittest.TestCase):
    def test_parse_batch_npu_devices_returns_none_when_unset(self) -> None:
        self.assertIsNone(parse_batch_npu_devices(None))

    def test_parse_batch_npu_devices_trims_whitespace(self) -> None:
        self.assertEqual(parse_batch_npu_devices(" 0, 1 ,2 "), ("0", "1", "2"))

    def test_parse_batch_npu_devices_expands_numeric_ranges(self) -> None:
        self.assertEqual(
            parse_batch_npu_devices("0,3-5,8-9"),
            ("0", "3", "4", "5", "8", "9"),
        )

    def test_parse_batch_npu_devices_rejects_empty_entries(self) -> None:
        with self.assertRaisesRegex(ValueError, "TRITON_AGENT_BATCH_NPU_DEVICES"):
            parse_batch_npu_devices("0,,1")

    def test_parse_batch_npu_devices_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_batch_npu_devices("0,1,0")

    def test_parse_batch_npu_devices_rejects_descending_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "range"):
            parse_batch_npu_devices("5-3")

    def test_parse_batch_npu_devices_rejects_malformed_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "range"):
            parse_batch_npu_devices("1-3-5")

    def test_affinity_env_for_device_uses_visible_devices(self) -> None:
        self.assertEqual(
            affinity_env_for_device("3"),
            {
                "ASCEND_RT_VISIBLE_DEVICES": "3",
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

    # -- parse_batch_workers_per_npu --

    def test_parse_workers_per_npu_returns_1_when_none(self) -> None:
        self.assertEqual(parse_batch_workers_per_npu(None), 1)

    def test_parse_workers_per_npu_rejects_empty_string(self) -> None:
        with self.assertRaisesRegex(ValueError, "TRITON_AGENT_BATCH_WORKERS_PER_NPU"):
            parse_batch_workers_per_npu("")

    def test_parse_workers_per_npu_rejects_whitespace_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "TRITON_AGENT_BATCH_WORKERS_PER_NPU"):
            parse_batch_workers_per_npu("  ")

    def test_parse_workers_per_npu_parses_valid_integer(self) -> None:
        self.assertEqual(parse_batch_workers_per_npu("3"), 3)

    def test_parse_workers_per_npu_rejects_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "TRITON_AGENT_BATCH_WORKERS_PER_NPU"):
            parse_batch_workers_per_npu("0")

    def test_parse_workers_per_npu_rejects_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "TRITON_AGENT_BATCH_WORKERS_PER_NPU"):
            parse_batch_workers_per_npu("-1")

    def test_parse_workers_per_npu_rejects_non_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "TRITON_AGENT_BATCH_WORKERS_PER_NPU"):
            parse_batch_workers_per_npu("abc")

    # -- configured_batch_workers_per_npu --

    def test_configured_workers_per_npu_defaults_to_1(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_batch_workers_per_npu(), 1)

    def test_configured_workers_per_npu_reads_env(self) -> None:
        with mock.patch.dict(os.environ, {"TRITON_AGENT_BATCH_WORKERS_PER_NPU": "4"}, clear=True):
            self.assertEqual(configured_batch_workers_per_npu(), 4)

    # -- configured_batch_npu_slots --

    def test_configured_slots_returns_none_when_devices_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(configured_batch_npu_slots())

    def test_configured_slots_no_duplication_when_workers_unset(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"TRITON_AGENT_BATCH_NPU_DEVICES": "0,1"},
            clear=True,
        ):
            self.assertEqual(configured_batch_npu_slots(), ("0", "1"))

    def test_configured_slots_expands_devices_by_workers(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TRITON_AGENT_BATCH_NPU_DEVICES": "0,1",
                "TRITON_AGENT_BATCH_WORKERS_PER_NPU": "2",
            },
            clear=True,
        ):
            self.assertEqual(
                configured_batch_npu_slots(),
                ("0", "0", "1", "1"),
            )

    def test_configured_slots_with_ranges_and_workers(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TRITON_AGENT_BATCH_NPU_DEVICES": "0,3-4",
                "TRITON_AGENT_BATCH_WORKERS_PER_NPU": "3",
            },
            clear=True,
        ):
            self.assertEqual(
                configured_batch_npu_slots(),
                ("0", "0", "0", "3", "3", "3", "4", "4", "4"),
            )

    # -- validate_batch_affinity_capacity --

    def test_validate_capacity_allows_within_device_count(self) -> None:
        validate_batch_affinity_capacity(("0", "1"), max_concurrency=2)

    def test_validate_capacity_allows_within_effective_capacity(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"TRITON_AGENT_BATCH_WORKERS_PER_NPU": "2"},
        ):
            validate_batch_affinity_capacity(("0", "1"), max_concurrency=4)

    def test_validate_capacity_rejects_exceeding_effective_capacity(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"TRITON_AGENT_BATCH_WORKERS_PER_NPU": "2"},
        ):
            with self.assertRaisesRegex(ValueError, "TRITON_AGENT_BATCH_WORKERS_PER_NPU"):
                validate_batch_affinity_capacity(("0", "1"), max_concurrency=5)

    def test_validate_capacity_error_mentions_both_env_vars(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"TRITON_AGENT_BATCH_WORKERS_PER_NPU": "3"},
        ):
            with self.assertRaises(ValueError) as ctx:
                validate_batch_affinity_capacity(("0", "1", "2"), max_concurrency=10)
            message = str(ctx.exception)
            self.assertIn("TRITON_AGENT_BATCH_NPU_DEVICES", message)
            self.assertIn("TRITON_AGENT_BATCH_WORKERS_PER_NPU", message)

    def test_validate_capacity_noop_when_devices_none(self) -> None:
        validate_batch_affinity_capacity(None, max_concurrency=100)

    # -- pool with repeated slots (shared devices) --

    def test_pool_allows_concurrent_same_device_when_slots_duplicated(self) -> None:
        pool = BatchNpuAffinityPool(("0", "0", "1", "1"))
        with pool.acquire() as first:
            with pool.acquire() as second:
                self.assertEqual(first, "0")
                self.assertEqual(second, "0")

    def test_pool_leases_all_expanded_slots(self) -> None:
        pool = BatchNpuAffinityPool(("0", "0"))
        acquired: list[str] = []
        with pool.acquire() as a:
            acquired.append(a)
            with pool.acquire() as b:
                acquired.append(b)
        self.assertCountEqual(acquired, ["0", "0"])


if __name__ == "__main__":
    unittest.main()
