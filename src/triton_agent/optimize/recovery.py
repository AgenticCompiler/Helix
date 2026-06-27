from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Callable, Literal

from triton_agent.models import AgentResult
from triton_agent.optimize.checks import check_round
from triton_agent.transient_failures import contains_transient_agent_failure_text


WorkerFailureKind = Literal["stall", "transient", "fatal"]


@dataclass(frozen=True)
class ProgressSnapshot:
    latest_mtime: float | None
    file_fingerprints: tuple[tuple[str, int, float], ...]


@dataclass(frozen=True)
class RangeProgress:
    last_accepted_round: int
    first_unresolved_round: int
    next_batch_start: int
    next_batch_end: int


def classify_worker_failure(result: AgentResult) -> WorkerFailureKind:
    if result.stalled:
        return "stall"
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.retryable_failure or contains_transient_agent_failure_text(combined):
        return "transient"
    return "fatal"


def is_optimize_progress_path(path: Path, workspace: Path) -> bool:
    if not path.is_file():
        return False
    relative = path.relative_to(workspace)
    if relative == Path("opt-note.md") or relative == Path("learned_lessons.md"):
        return True
    parts = relative.parts
    if not parts:
        return False
    if parts[0] == "baseline":
        return True
    if re.fullmatch(r"opt-round-\d+", parts[0]) is not None:
        return True
    return False


def scan_optimize_progress(workspace: Path) -> ProgressSnapshot:
    entries: list[tuple[str, int, float]] = []
    _append_progress_file_fingerprint(workspace / "opt-note.md", workspace, entries)
    _append_progress_file_fingerprint(workspace / "learned_lessons.md", workspace, entries)
    _append_progress_tree_fingerprints(workspace / "baseline", workspace, entries)
    for round_dir in sorted(path for path in workspace.glob("opt-round-*") if path.is_dir()):
        _append_progress_tree_fingerprints(round_dir, workspace, entries)
    entries.sort(key=lambda item: item[0])
    latest = max((item[2] for item in entries), default=None)
    return ProgressSnapshot(latest_mtime=latest, file_fingerprints=tuple(entries))


def _append_progress_tree_fingerprints(
    root: Path,
    workspace: Path,
    entries: list[tuple[str, int, float]],
) -> None:
    if not root.is_dir():
        return

    for current_root, _dirnames, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=lambda _error: None,
    ):
        current_path = Path(current_root)
        for filename in sorted(filenames):
            _append_progress_file_fingerprint(current_path / filename, workspace, entries)


def _append_progress_file_fingerprint(
    path: Path,
    workspace: Path,
    entries: list[tuple[str, int, float]],
) -> None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return
    if not path.is_file():
        return
    rel = path.relative_to(workspace).as_posix()
    entries.append((rel, stat.st_size, stat.st_mtime))


def build_optimize_progress_probe(workspace: Path) -> Callable[[], float | None]:
    state = {"snapshot": scan_optimize_progress(workspace)}

    def probe() -> float | None:
        snapshot = scan_optimize_progress(workspace)
        if snapshot != state["snapshot"]:
            state["snapshot"] = snapshot
            return snapshot.latest_mtime
        return None

    return probe


def compute_range_progress(
    workdir: Path,
    *,
    batch_start: int,
    batch_end: int,
    optimize_target: Literal["kernel", "operator"],
) -> RangeProgress:
    """Compute accepted progress for the current worker-owned range only."""
    last_accepted = batch_start - 1
    for round_number in range(batch_start, batch_end + 1):
        round_dir = workdir / f"opt-round-{round_number}"
        if not round_dir.is_dir():
            break
        try:
            result = check_round(
                round_dir,
                current_round=round_number,
                final_round=batch_end,
                optimize_target=optimize_target,
            )
        except Exception:
            break
        if result.status != "pass":
            break
        last_accepted = round_number
    first_unresolved = last_accepted + 1
    return RangeProgress(
        last_accepted_round=last_accepted,
        first_unresolved_round=first_unresolved,
        next_batch_start=first_unresolved,
        next_batch_end=batch_end,
    )


def render_transient_recovery_note(*, batch_start: int, batch_end: int) -> str:
    return "\n".join(
        [
            "CLI recovery note:",
            "The previous invocation ended in a transient backend failure.",
            f"Retry the current target range and complete rounds {batch_start} through {batch_end}.",
        ]
    )


def render_stall_recovery_note(
    *,
    original_batch_start: int,
    last_accepted_round: int,
    first_unresolved_round: int,
    batch_end: int,
) -> str:
    lines = [
        "CLI recovery note:",
        "The previous invocation stalled.",
    ]
    if last_accepted_round >= original_batch_start:
        lines.append(
            f"Rounds {original_batch_start} through {last_accepted_round} are already accepted as session progress and must not be rerun."
        )
    lines.extend(
        [
            f"Resume from round {first_unresolved_round} and continue through round {batch_end}.",
            f"Inspect existing artifacts for round {first_unresolved_round} before deciding whether to repair or finish it.",
        ]
    )
    return "\n".join(lines)


class RecoveryBudget:
    def __init__(self, max_attempts: int = 3) -> None:
        self._max_attempts = max_attempts
        self._attempts: dict[int, int] = {}

    def consume(self, round_number: int) -> None:
        self._attempts[round_number] = self._attempts.get(round_number, 0) + 1

    def remaining(self, round_number: int) -> int:
        return self._max_attempts - self._attempts.get(round_number, 0)

    def exhausted(self, round_number: int) -> bool:
        return self.remaining(round_number) <= 0
