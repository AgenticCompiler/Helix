from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from triton_agent.skills.catalog import resolve_skill_source_dir


def optimize_knowledge_skill_name(language: str) -> str:
    return f"{language}-npu-optimize-knowledge"


def ensure_editable_knowledge_skill(skills_dir: Path, *, language: str = "triton") -> Path:
    knowledge_name = optimize_knowledge_skill_name(language)
    skills_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir = skills_dir / knowledge_name
    if knowledge_dir.exists():
        if not knowledge_dir.is_dir():
            raise ValueError(f"Skills workspace entry is not a directory: {knowledge_dir}")
        return knowledge_dir

    bundled = resolve_skill_source_dir(knowledge_name)
    if not bundled.is_dir():
        raise ValueError(f"Bundled knowledge skill does not exist: {bundled}")
    shutil.copytree(bundled, knowledge_dir, symlinks=False)
    return knowledge_dir


def rebuild_pattern_index(knowledge_dir: Path) -> None:
    script = knowledge_dir / "scripts" / "build_pattern_index.py"
    patterns_dir = knowledge_dir / "references" / "patterns"
    output = knowledge_dir / "references" / "pattern_index.md"
    if not script.exists():
        raise ValueError(f"Pattern index builder does not exist: {script}")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--patterns-dir",
            str(patterns_dir),
            "--output",
            str(output),
        ],
        cwd=knowledge_dir,
        check=True,
    )


def promote_editable_knowledge_skill(
    source_knowledge_dir: Path,
    *,
    language: str = "triton",
) -> Path:
    if not source_knowledge_dir.is_dir():
        raise ValueError(f"Converged knowledge skill does not exist: {source_knowledge_dir}")
    destination = resolve_skill_source_dir(optimize_knowledge_skill_name(language))
    if source_knowledge_dir.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _remove_existing_path(destination)
        shutil.copytree(source_knowledge_dir, destination, symlinks=False)
    rebuild_pattern_index(destination)
    return destination


def _remove_existing_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    if path.exists():
        path.unlink()


def snapshot_pattern_card_texts(knowledge_dir: Path) -> dict[str, str]:
    patterns_dir = knowledge_dir / "references" / "patterns"
    if not patterns_dir.is_dir():
        return {}
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(patterns_dir.glob("*.md"))
    }


def find_changed_pattern_cards(
    knowledge_dir: Path,
    pattern_snapshot: dict[str, str],
) -> list[Path]:
    patterns_dir = knowledge_dir / "references" / "patterns"
    if not patterns_dir.is_dir():
        return []
    changed: list[Path] = []
    for path in sorted(patterns_dir.glob("*.md")):
        current = path.read_text(encoding="utf-8")
        if pattern_snapshot.get(path.name) != current:
            changed.append(path)
    return changed


def export_changed_pattern_cards(
    source_knowledge_dir: Path,
    update_skills_dir: Path,
    *,
    language: str = "triton",
    pattern_snapshot: dict[str, str],
    updated_pattern_names: list[str] | None = None,
) -> list[str]:
    changed_paths = find_changed_pattern_cards(source_knowledge_dir, pattern_snapshot)
    for name in updated_pattern_names or []:
        resolved = find_pattern_card(source_knowledge_dir, name)
        if resolved is not None and resolved not in changed_paths:
            changed_paths.append(resolved)
    if not changed_paths:
        return []

    dest_knowledge = update_skills_dir / optimize_knowledge_skill_name(language)
    dest_patterns = dest_knowledge / "references" / "patterns"
    dest_patterns.mkdir(parents=True, exist_ok=True)
    _ensure_index_scripts(source_knowledge_dir, dest_knowledge)

    exported: list[str] = []
    for path in changed_paths:
        shutil.copy2(path, dest_patterns / path.name)
        exported.append(path.name)

    try:
        rebuild_pattern_index(dest_knowledge)
    except Exception as exc:
        print(f"Warning: export pattern index regeneration failed: {exc}", file=sys.stderr)
    manifest = {
        "exported_patterns": exported,
        "updated_pattern_names": list(updated_pattern_names or []),
    }
    update_skills_dir.mkdir(parents=True, exist_ok=True)
    (update_skills_dir / "updated_patterns.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return exported


def find_pattern_card(knowledge_dir: Path, name: str) -> Path | None:
    patterns_dir = knowledge_dir / "references" / "patterns"
    if not patterns_dir.is_dir():
        return None
    slug = _slugify(name)
    candidates = (
        patterns_dir / f"{name}.md",
        patterns_dir / f"{slug}.md",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for path in patterns_dir.glob("*.md"):
        if path.stem == slug:
            return path
        text = path.read_text(encoding="utf-8")
        first_line = text.splitlines()[0] if text else ""
        if first_line == f"# {name}":
            return path
    return None


def _ensure_index_scripts(source_knowledge_dir: Path, dest_knowledge: Path) -> None:
    source_scripts = source_knowledge_dir / "scripts"
    dest_scripts = dest_knowledge / "scripts"
    if source_scripts.is_dir() and not dest_scripts.exists():
        shutil.copytree(source_scripts, dest_scripts, symlinks=False)


def _slugify(name: str) -> str:
    value = name.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    return value.strip("-")
