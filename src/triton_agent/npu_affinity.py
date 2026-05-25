from __future__ import annotations

import os
import queue
from collections.abc import Iterator
from contextlib import contextmanager


_BATCH_NPU_DEVICES_ENV = "TRITON_AGENT_BATCH_NPU_DEVICES"
_BATCH_WORKERS_PER_NPU_ENV = "TRITON_AGENT_BATCH_WORKERS_PER_NPU"


def parse_batch_npu_devices(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    raw_parts = tuple(part.strip() for part in raw.split(","))
    if not raw_parts or any(not part for part in raw_parts):
        raise ValueError(f"{_BATCH_NPU_DEVICES_ENV} must be a comma-separated non-empty device list.")
    devices: list[str] = []
    for part in raw_parts:
        devices.extend(_expand_device_token(part))
    expanded = tuple(devices)
    if len(set(expanded)) != len(expanded):
        raise ValueError(f"{_BATCH_NPU_DEVICES_ENV} must not contain duplicate devices: {raw!r}")
    return expanded


def configured_batch_npu_devices() -> tuple[str, ...] | None:
    return parse_batch_npu_devices(os.environ.get(_BATCH_NPU_DEVICES_ENV))


def parse_batch_workers_per_npu(raw: str | None) -> int:
    if raw is None:
        return 1
    stripped = raw.strip()
    if stripped == "":
        raise ValueError(
            f"{_BATCH_WORKERS_PER_NPU_ENV} is set but empty; "
            f"must be a positive integer when {_BATCH_NPU_DEVICES_ENV} is configured."
        )
    try:
        value = int(stripped)
    except ValueError:
        raise ValueError(
            f"{_BATCH_WORKERS_PER_NPU_ENV} must be a positive integer, got {raw!r}"
        )
    if value < 1:
        raise ValueError(
            f"{_BATCH_WORKERS_PER_NPU_ENV} must be a positive integer, got {value}"
        )
    return value


def configured_batch_workers_per_npu() -> int:
    return parse_batch_workers_per_npu(os.environ.get(_BATCH_WORKERS_PER_NPU_ENV))


def configured_batch_npu_slots() -> tuple[str, ...] | None:
    devices = configured_batch_npu_devices()
    if devices is None:
        return None
    workers = configured_batch_workers_per_npu()
    slots: list[str] = []
    for device in devices:
        slots.extend([device] * workers)
    return tuple(slots)


def validate_batch_affinity_capacity(
    devices: tuple[str, ...] | None,
    *,
    max_concurrency: int,
) -> None:
    if devices is None:
        return
    workers = configured_batch_workers_per_npu()
    effective_capacity = len(devices) * workers
    if max_concurrency > effective_capacity:
        raise ValueError(
            f"--max-concurrency ({max_concurrency}) must not exceed the effective "
            f"capacity ({effective_capacity}) derived from "
            f"{_BATCH_NPU_DEVICES_ENV} ({len(devices)} device(s)) "
            f"and {_BATCH_WORKERS_PER_NPU_ENV} ({workers})."
        )


def affinity_env_for_device(device: str) -> dict[str, str]:
    return {"ASCEND_RT_VISIBLE_DEVICES": device}


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


def _expand_device_token(token: str) -> list[str]:
    if "-" not in token:
        return [token]
    if token.count("-") != 1:
        raise ValueError(f"{_BATCH_NPU_DEVICES_ENV} range token is invalid: {token!r}")
    start_text, end_text = token.split("-", 1)
    if not start_text.isdigit() or not end_text.isdigit():
        raise ValueError(f"{_BATCH_NPU_DEVICES_ENV} range token is invalid: {token!r}")
    start = int(start_text)
    end = int(end_text)
    if start > end:
        raise ValueError(f"{_BATCH_NPU_DEVICES_ENV} range token must be ascending: {token!r}")
    return [str(value) for value in range(start, end + 1)]
