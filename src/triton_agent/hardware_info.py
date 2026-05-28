from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def capture_hardware_info(target_chip: str) -> dict[str, Any]:
    chip_name = _query_npu_smi_chip_name()
    cann_version = _query_cann_version()
    driver_version = _query_driver_version()
    return {
        "chip_name": chip_name,
        "cann_version": cann_version,
        "driver_version": driver_version,
    }


def write_env_info(workdir: Path, target_chip: str) -> Path:
    hardware = capture_hardware_info(target_chip)
    info: dict[str, Any] = {
        "target_chip": target_chip,
        "hardware": hardware,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = workdir / "env-info.json"
    output_path.write_text(
        json.dumps(info, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _query_npu_smi_chip_name() -> str:
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _fallback_npu_smi()
    if result.returncode != 0:
        return _fallback_npu_smi()
    output = result.stdout.strip()
    if not output:
        return "unknown"
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Chip Name"):
            value = line.split(":", 1)[-1].strip()
            if value:
                return value
    return _fallback_npu_smi()


def _fallback_npu_smi() -> str:
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-m"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    for line in result.stdout.splitlines():
        line = line.strip()
        if "Chip Name" in line:
            value = line.split(":", 1)[-1].strip()
            if value:
                return value
    return "unknown"


def _query_cann_version() -> str:
    toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
    if toolkit_home:
        toolkit_path = Path(toolkit_home)
        if toolkit_path.is_dir():
            return toolkit_path.name
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    for line in result.stdout.splitlines():
        line = line.strip()
        if "Chip Version" in line:
            value = line.split(":", 1)[-1].strip()
            if value:
                return value
    return "unknown"


def _query_driver_version() -> str:
    driver_info_path = Path("/usr/local/Ascend/driver/version.info")
    if driver_info_path.is_file():
        try:
            content = driver_info_path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                return content
        except OSError:
            pass
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-m"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            line = line.strip()
            if "Driver Version" in line:
                value = line.split(":", 1)[-1].strip()
                if value:
                    return value
    return "unknown"


__all__ = ["capture_hardware_info", "write_env_info"]
