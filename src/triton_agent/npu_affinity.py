from __future__ import annotations

import os
import queue
from collections.abc import Iterator
from contextlib import contextmanager


_BATCH_NPU_DEVICES_ENV = "TRITON_AGENT_BATCH_NPU_DEVICES"


def parse_batch_npu_devices(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    devices = tuple(part.strip() for part in raw.split(","))
    if not devices or any(not part for part in devices):
        raise ValueError(f"{_BATCH_NPU_DEVICES_ENV} must be a comma-separated non-empty device list.")
    if len(set(devices)) != len(devices):
        raise ValueError(f"{_BATCH_NPU_DEVICES_ENV} must not contain duplicate devices: {raw!r}")
    return devices


def configured_batch_npu_devices() -> tuple[str, ...] | None:
    return parse_batch_npu_devices(os.environ.get(_BATCH_NPU_DEVICES_ENV))


def validate_batch_affinity_capacity(
    devices: tuple[str, ...] | None,
    *,
    max_concurrency: int,
) -> None:
    if devices is None:
        return
    if max_concurrency > len(devices):
        raise ValueError(
            "--max-concurrency must not exceed the number of devices configured by "
            "TRITON_AGENT_BATCH_NPU_DEVICES."
        )


def affinity_env_for_device(device: str) -> dict[str, str]:
    return {
        "ASCEND_RT_VISIBLE_DEVICES": device,
        "TRITON_AGENT_ASSIGNED_NPU": device,
    }


class BatchNpuAffinityPool:
    def __init__(self, devices: tuple[str, ...]) -> None:
        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        for device in devices:
            self._queue.put(device)

    @contextmanager
    def acquire(self) -> Iterator[str]:
        device = self._queue.get()
        try:
            yield device
        finally:
            self._queue.put(device)
