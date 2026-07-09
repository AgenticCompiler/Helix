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


def configured_batch_npu_slots(
    npu_devices_raw: str | None = None,
    workers_per_npu_raw: str | None = None,
) -> tuple[str, ...] | None:
    devices = (
        configured_batch_npu_devices()
        if npu_devices_raw is None
        else parse_batch_npu_devices(npu_devices_raw)
    )
    if devices is None:
        return None
    workers = (
        configured_batch_workers_per_npu()
        if workers_per_npu_raw is None
        else parse_batch_workers_per_npu(workers_per_npu_raw)
    )
    slots: list[str] = []
    for device in devices:
        slots.extend([device] * workers)
    return tuple(slots)


def effective_batch_affinity_capacity(
    npu_devices_raw: str | None = None,
    workers_per_npu_raw: str | None = None,
    *,
    ignore_workers_per_npu: bool = False,
) -> int | None:
    devices = (
        configured_batch_npu_devices()
        if npu_devices_raw is None
        else parse_batch_npu_devices(npu_devices_raw)
    )
    if devices is None:
        return None
    if ignore_workers_per_npu:
        if workers_per_npu_raw is None:
            configured_batch_workers_per_npu()
        else:
            parse_batch_workers_per_npu(workers_per_npu_raw)
        return len(devices)
    workers = (
        configured_batch_workers_per_npu()
        if workers_per_npu_raw is None
        else parse_batch_workers_per_npu(workers_per_npu_raw)
    )
    return len(devices) * workers


def resolve_batch_concurrency(
    requested: int | str,
    npu_devices_raw: str | None = None,
    workers_per_npu_raw: str | None = None,
    *,
    ignore_workers_per_npu: bool = False,
) -> int:
    if requested == "max":
        capacity = effective_batch_affinity_capacity(
            npu_devices_raw,
            workers_per_npu_raw,
            ignore_workers_per_npu=ignore_workers_per_npu,
        )
        if capacity is None:
            raise ValueError(
                f"--concurrency max requires {_BATCH_NPU_DEVICES_ENV} to be set."
            )
        return capacity
    if not isinstance(requested, int):
        raise ValueError(f"--concurrency must be a positive integer or 'max', got {requested!r}")
    if requested < 1:
        raise ValueError("--concurrency must be at least 1")
    return requested


def validate_batch_affinity_capacity(
    devices: tuple[str, ...] | None,
    *,
    max_concurrency: int,
    workers_per_npu_raw: str | None = None,
    ignore_workers_per_npu: bool = False,
) -> None:
    if devices is None:
        return
    if ignore_workers_per_npu:
        if workers_per_npu_raw is None:
            configured_batch_workers_per_npu()
        else:
            parse_batch_workers_per_npu(workers_per_npu_raw)
        effective_capacity = len(devices)
        if max_concurrency > effective_capacity:
            raise ValueError(
                f"--concurrency ({max_concurrency}) must not exceed the managed MCP "
                f"capacity ({effective_capacity}) derived from "
                f"{_BATCH_NPU_DEVICES_ENV} ({len(devices)} device(s)); "
                f"{_BATCH_WORKERS_PER_NPU_ENV} is ignored when --enable-mcp is set."
            )
        return
    workers = (
        configured_batch_workers_per_npu()
        if workers_per_npu_raw is None
        else parse_batch_workers_per_npu(workers_per_npu_raw)
    )
    effective_capacity = len(devices) * workers
    if max_concurrency > effective_capacity:
        raise ValueError(
            f"--concurrency ({max_concurrency}) must not exceed the effective "
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
