from __future__ import annotations

from pathlib import Path


def cleanup_dir_pt_files(directory: Path) -> list[str]:
    cleaned: list[str] = []
    for pt_file in sorted(directory.iterdir()):
        if not pt_file.is_file():
            continue
        name_lower = pt_file.name.lower()
        if not (name_lower == "test_result.pt" or name_lower.endswith("_result.pt")):
            continue
        try:
            pt_file.unlink()
            cleaned.append(pt_file.name)
        except OSError:
            pass
    return cleaned


def cleanup_workspace_pt_files(workdir: Path) -> list[str]:
    cleaned: list[str] = []
    cleaned.extend(cleanup_dir_pt_files(workdir))
    for round_dir in sorted(workdir.glob("opt-round-*")):
        if not round_dir.is_dir():
            continue
        for name in cleanup_dir_pt_files(round_dir):
            cleaned.append(f"{round_dir.name}/{name}")
    return cleaned
