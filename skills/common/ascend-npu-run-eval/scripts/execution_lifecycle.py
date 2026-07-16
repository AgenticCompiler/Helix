"""Shared optimize-round lifecycle helpers for test and benchmark commands."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterator, Literal, cast

from env_registry import HELIX_OPTIMIZE_DELETE_PT_FILES, TRITON_ALL_BLOCKS_PARALLEL


_BLOCKS_PARALLEL_UNSAFE_VALUE = "1"
_BLOCKS_PARALLEL_SAFE_VALUE = "0"
_PT_CLEANUP_MODES = frozenset({"never", "round", "run-test"})
_LEGACY_ROUND_CLEANUP_VALUES = frozenset({"1", "true", "yes", "on"})
_LEGACY_NEVER_CLEANUP_VALUES = frozenset({"0", "false", "no", "off"})
PtCleanupMode = Literal["never", "round", "run-test"]


def pt_cleanup_mode() -> PtCleanupMode:
    raw_value = os.environ.get(HELIX_OPTIMIZE_DELETE_PT_FILES)
    if raw_value is None:
        return "round"
    value = raw_value.strip().lower()
    if value in _PT_CLEANUP_MODES:
        return cast(PtCleanupMode, value)
    if value in _LEGACY_ROUND_CLEANUP_VALUES:
        return "round"
    if value in _LEGACY_NEVER_CLEANUP_VALUES:
        return "never"
    return "round"


def cleanup_run_test_pt_files(paths: tuple[Path | None, ...]) -> list[str]:
    if pt_cleanup_mode() != "run-test":
        return []
    cleaned: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        if _is_ordinary_pt_result_file(resolved_path) and resolved_path.is_file():
            try:
                resolved_path.unlink()
            except OSError:
                continue
            cleaned.append(str(resolved_path))
    return cleaned


def active_optimize_round_context(*paths: Path) -> dict[str, str] | None:
    state_path = _find_optimize_state_path(*paths)
    if state_path is None:
        return None
    try:
        raw_payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw_payload, dict):
        return None
    payload = cast(dict[str, object], raw_payload)
    if payload.get("phase") != "round_active":
        return None
    run_id = payload.get("run_id")
    current_round = payload.get("current_round")
    rounds = payload.get("rounds")
    if not isinstance(run_id, str) or not run_id:
        return None
    if not isinstance(current_round, int) or not isinstance(rounds, dict):
        return None
    round_entry = cast(dict[str, object], rounds).get(str(current_round))
    if not isinstance(round_entry, dict):
        return None
    round_entry_dict = cast(dict[str, object], round_entry)
    round_dir = round_entry_dict.get("round_dir")
    if round_entry_dict.get("status") != "active" or not isinstance(round_dir, str) or not round_dir:
        return None
    return {
        "run_id": run_id,
        "round": round_dir,
        "workspace_root": str(state_path.parent.parent),
    }


def append_optimize_timing_event(
    context: dict[str, str] | None,
    *,
    event: str,
    command: str,
    return_code: int | None = None,
    test_file: Path | None = None,
    bench_file: Path | None = None,
    operator_file: Path | None = None,
) -> None:
    if context is None:
        return
    try:
        workspace_root = Path(context["workspace_root"])
        log_path = workspace_root / ".helix" / "round-timings" / f"{context['round']}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "event": event,
            "timestamp": _utc_now(),
            "run_id": context["run_id"],
            "round": context["round"],
            "command": command,
        }
        if return_code is not None:
            payload["return_code"] = return_code
        for key, path in (
            ("test_file", test_file),
            ("bench_file", bench_file),
            ("operator_file", operator_file),
        ):
            if path is not None:
                payload[key] = _timing_display_path(path, workspace_root)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except OSError:
        return


@contextlib.contextmanager
def guard_operator_execution_env(command: str) -> Iterator[None]:
    if command not in {
        "run-test-baseline",
        "run-test-convert",
        "run-test-optimize",
        "run-bench",
        "profile-bench",
    }:
        yield
        return
    previous = os.environ.get(TRITON_ALL_BLOCKS_PARALLEL)
    if previous != _BLOCKS_PARALLEL_UNSAFE_VALUE:
        yield
        return
    os.environ[TRITON_ALL_BLOCKS_PARALLEL] = _BLOCKS_PARALLEL_SAFE_VALUE
    try:
        yield
    finally:
        os.environ[TRITON_ALL_BLOCKS_PARALLEL] = previous


def _is_ordinary_pt_result_file(path: Path) -> bool:
    name_lower = path.name.lower()
    return name_lower == "test_result.pt" or name_lower.endswith("_result.pt")


def _find_optimize_state_path(*paths: Path) -> Path | None:
    search_roots = [Path.cwd().resolve(), *(path.resolve().parent for path in paths)]
    seen: set[Path] = set()
    for root in search_roots:
        current = root
        while current not in seen:
            seen.add(current)
            state_path = current / ".helix" / "state.json"
            if state_path.is_file():
                return state_path
            if current.parent == current:
                break
            current = current.parent
    return None


def _timing_display_path(path: Path, workspace_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace_root).as_posix()
    except ValueError:
        return str(resolved)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
