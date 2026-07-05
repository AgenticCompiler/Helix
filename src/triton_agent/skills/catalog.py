from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from triton_agent.paths import application_root


@dataclass(frozen=True)
class SkillCatalogEntry:
    """A single entry in the repository skill catalog.

    Each entry maps a logical skill name to its physical source location
    within the repository. The catalog is the source of truth for:
    - staging (which directory to copy into a flat workspace)
    - loading (where to find skill scripts at runtime)
    - tests (where to locate live skill files)
    """

    logical_name: str
    source_group: str
    physical_path: str

    @property
    def physical_source_dir(self) -> Path:
        return application_root() / self.physical_path


_COMMON_SKILLS: tuple[SkillCatalogEntry, ...] = (
    SkillCatalogEntry(
        logical_name="ascend-npu-optimize-state",
        source_group="common",
        physical_path="skills/common/ascend-npu-optimize-state",
    ),
    SkillCatalogEntry(
        logical_name="ascend-npu-prepare-optimize-baseline",
        source_group="common",
        physical_path="skills/common/ascend-npu-prepare-optimize-baseline",
    ),
    SkillCatalogEntry(
        logical_name="ascend-npu-gen-test",
        source_group="common",
        physical_path="skills/common/ascend-npu-gen-test",
    ),
    SkillCatalogEntry(
        logical_name="ascend-npu-gen-bench",
        source_group="common",
        physical_path="skills/common/ascend-npu-gen-bench",
    ),
    SkillCatalogEntry(
        logical_name="ascend-npu-gen-eval-suite",
        source_group="common",
        physical_path="skills/common/ascend-npu-gen-eval-suite",
    ),
    SkillCatalogEntry(
        logical_name="ascend-npu-run-eval",
        source_group="common",
        physical_path="skills/common/ascend-npu-run-eval",
    ),
    SkillCatalogEntry(
        logical_name="ascend-npu-run-eval-mcp",
        source_group="common",
        physical_path="skills/common/ascend-npu-run-eval-mcp",
    ),
    SkillCatalogEntry(
        logical_name="ascend-npu-report",
        source_group="common",
        physical_path="skills/common/ascend-npu-report",
    ),
    SkillCatalogEntry(
        logical_name="ascend-npu-distill-patterns",
        source_group="common",
        physical_path="skills/common/ascend-npu-distill-patterns",
    ),
    SkillCatalogEntry(
        logical_name="ascend-npu-profile-operator",
        source_group="common",
        physical_path="skills/common/ascend-npu-profile-operator",
    ),
    SkillCatalogEntry(
        logical_name="ascend-npu-analyze-round-performance",
        source_group="common",
        physical_path="skills/common/ascend-npu-analyze-round-performance",
    ),
    SkillCatalogEntry(
        logical_name="ascend-npu-plan-git-operator-workspaces",
        source_group="common",
        physical_path="skills/common/ascend-npu-plan-git-operator-workspaces",
    ),
)

_TRITON_SKILLS: tuple[SkillCatalogEntry, ...] = (
    SkillCatalogEntry(
        logical_name="triton-npu-optimize",
        source_group="triton",
        physical_path="skills/triton/triton-npu-optimize",
    ),
    SkillCatalogEntry(
        logical_name="triton-npu-convert-pytorch-operator",
        source_group="triton",
        physical_path="skills/triton/triton-npu-convert-pytorch-operator",
    ),
    SkillCatalogEntry(
        logical_name="triton-npu-repair-guide",
        source_group="triton",
        physical_path="skills/triton/triton-npu-repair-guide",
    ),
    SkillCatalogEntry(
        logical_name="triton-npu-analyze-ir",
        source_group="triton",
        physical_path="skills/triton/triton-npu-analyze-ir",
    ),
    SkillCatalogEntry(
        logical_name="triton-npu-analyze-compiler-source",
        source_group="triton",
        physical_path="skills/triton/triton-npu-analyze-compiler-source",
    ),
    SkillCatalogEntry(
        logical_name="triton-npu-cann-ext-api-patterns",
        source_group="triton",
        physical_path="skills/triton/triton-npu-cann-ext-api-patterns",
    ),
    SkillCatalogEntry(
        logical_name="triton-npu-optimize-knowledge",
        source_group="triton",
        physical_path="skills/triton/triton-npu-optimize-knowledge",
    ),
    SkillCatalogEntry(
        logical_name="triton-npu-optimize-knowledge-v2",
        source_group="triton",
        physical_path="skills/triton/triton-npu-optimize-knowledge-v2",
    ),
    SkillCatalogEntry(
        logical_name="triton-npu-optimize-knowledge-v3",
        source_group="triton",
        physical_path="skills/triton/triton-npu-optimize-knowledge-v3",
    ),
    SkillCatalogEntry(
        logical_name="torch-npu-optimize-knowledge",
        source_group="triton",
        physical_path="skills/triton/torch-npu-optimize-knowledge",
    ),
    SkillCatalogEntry(
        logical_name="triton-npu-pattern-signal",
        source_group="triton",
        physical_path="skills/triton/triton-npu-pattern-signal",
    ),
)

_TILELANG_SKILLS: tuple[SkillCatalogEntry, ...] = (
    SkillCatalogEntry(
        logical_name="tilelang-npu-analyze-compiler-source",
        source_group="tilelang",
        physical_path="skills/tilelang/tilelang-npu-analyze-compiler-source",
    ),
    SkillCatalogEntry(
        logical_name="tilelang-npu-api-reference",
        source_group="tilelang",
        physical_path="skills/tilelang/tilelang-npu-api-reference",
    ),
    SkillCatalogEntry(
        logical_name="tilelang-npu-analyze-ir",
        source_group="tilelang",
        physical_path="skills/tilelang/tilelang-npu-analyze-ir",
    ),
    SkillCatalogEntry(
        logical_name="tilelang-npu-convert-pytorch-operator",
        source_group="tilelang",
        physical_path="skills/tilelang/tilelang-npu-convert-pytorch-operator",
    ),
    SkillCatalogEntry(
        logical_name="tilelang-npu-optimize",
        source_group="tilelang",
        physical_path="skills/tilelang/tilelang-npu-optimize",
    ),
    SkillCatalogEntry(
        logical_name="tilelang-npu-optimize-knowledge",
        source_group="tilelang",
        physical_path="skills/tilelang/tilelang-npu-optimize-knowledge",
    ),
    SkillCatalogEntry(
        logical_name="tilelang-npu-cann-ext-api-patterns",
        source_group="tilelang",
        physical_path="skills/tilelang/tilelang-npu-cann-ext-api-patterns",
    ),
    SkillCatalogEntry(
        logical_name="tilelang-npu-repair-guide",
        source_group="tilelang",
        physical_path="skills/tilelang/tilelang-npu-repair-guide",
    ),
)

SKILL_CATALOG: tuple[SkillCatalogEntry, ...] = _COMMON_SKILLS + _TRITON_SKILLS + _TILELANG_SKILLS

_SKILL_BY_NAME: dict[str, SkillCatalogEntry] = {
    entry.logical_name: entry for entry in SKILL_CATALOG
}

_PHYSICAL_PATHS: set[str] = {entry.physical_path for entry in SKILL_CATALOG}


def get_skill_catalog_entry(skill_name: str) -> SkillCatalogEntry:
    """Return the catalog entry for a logical skill name.

    Raises KeyError if the skill is not in the catalog.
    """
    if skill_name not in _SKILL_BY_NAME:
        raise KeyError(f"Skill not found in catalog: {skill_name!r}")
    return _SKILL_BY_NAME[skill_name]


def resolve_skill_source_dir(skill_name: str) -> Path:
    """Return the physical source directory for a logical skill name."""
    entry = get_skill_catalog_entry(skill_name)
    physical_dir = entry.physical_source_dir
    return physical_dir


def list_catalog_skill_names() -> tuple[str, ...]:
    """Return all logical skill names in the catalog."""
    return tuple(_SKILL_BY_NAME.keys())


def is_catalog_skill(skill_name: str) -> bool:
    """Return True if the given logical name is a repository-owned catalog skill."""
    return skill_name in _SKILL_BY_NAME


def assert_catalog_completeness(skills_root: Path) -> None:
    """Assert that every repository-owned skill directory appears exactly once in the catalog.

    This should be called from tests to guard against orphaned skill directories.
    """
    found_dirs: set[str] = set()
    for group_dir in skills_root.iterdir():
        if not group_dir.is_dir():
            continue
        for skill_dir in group_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                rel_path = str(skill_dir.relative_to(skills_root.parent))
                found_dirs.add(rel_path)

    catalog_dirs = _PHYSICAL_PATHS
    missing_in_catalog = found_dirs - catalog_dirs
    orphaned_in_catalog = catalog_dirs - found_dirs

    errors: list[str] = []
    if missing_in_catalog:
        errors.append(
            f"Skill directories not in catalog: {sorted(missing_in_catalog)}"
        )
    if orphaned_in_catalog:
        errors.append(
            f"Catalog entries without matching directories: {sorted(orphaned_in_catalog)}"
        )

    if errors:
        raise AssertionError("; ".join(errors))
