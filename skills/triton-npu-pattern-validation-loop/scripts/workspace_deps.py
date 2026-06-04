"""Dependency layout helpers for pattern-validation workspaces."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_DEPENDENCY_DIR = "deps"
STRATEGY_REPO_PATH = "repo_path"
STRATEGY_DEPS_COPY = "deps_copy"
REPO_PATH_BOOTSTRAP_START = "# pattern-validation: repo import paths (auto-generated, do not remove)"
REPO_PATH_BOOTSTRAP_END = "# pattern-validation: end repo import paths"
DEPS_PATH_BOOTSTRAP_START = "# pattern-validation: deps import paths (auto-generated, do not remove)"
DEPS_PATH_BOOTSTRAP_END = "# pattern-validation: end deps import paths"
_IMPORT_SMOKE_TIMEOUT_SECONDS = 120

_REPO_SOURCE_ROOTS: dict[str, Path] = {
    "fla": Path("src/kernels/fla"),
}
_DEPS_PACKAGE_DIRS: dict[str, str] = {
    "fla": "fla",
}
_SRC_KERNELS_FLA_PREFIX = "src.kernels.fla"


def normalize_dependency_dir(value: object) -> str:
    text = str(value or "").strip() or DEFAULT_DEPENDENCY_DIR
    if "{" in text or "}" in text or text != DEFAULT_DEPENDENCY_DIR:
        return DEFAULT_DEPENDENCY_DIR
    return text


def extract_imported_modules(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _extract_imported_modules_regex(source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return _stable_unique(modules)


def repo_scoped_import_modules(source: str) -> list[str]:
    modules: list[str] = []
    for module in extract_imported_modules(source):
        root = module.split(".", 1)[0]
        if root in _REPO_SOURCE_ROOTS or module.startswith(_SRC_KERNELS_FLA_PREFIX):
            modules.append(module)
    return _stable_unique(modules)


def has_repo_path_bootstrap(source: str) -> bool:
    return REPO_PATH_BOOTSTRAP_START in source


def has_deps_path_bootstrap(source: str) -> bool:
    return DEPS_PATH_BOOTSTRAP_START in source


def build_deps_path_bootstrap_block(*, dependency_dir: str) -> str:
    return "\n".join(
        [
            DEPS_PATH_BOOTSTRAP_START,
            "import sys",
            "from pathlib import Path",
            "",
            f"_DEPS_ROOT = Path(__file__).resolve().parent / {dependency_dir!r}",
            'if _DEPS_ROOT.is_dir() and str(_DEPS_ROOT) not in sys.path:',
            "    sys.path.insert(0, str(_DEPS_ROOT))",
            DEPS_PATH_BOOTSTRAP_END,
            "",
        ],
    )


def inject_deps_path_bootstrap(operator_path: Path, *, dependency_dir: str) -> bool:
    text = operator_path.read_text(encoding="utf-8")
    if has_deps_path_bootstrap(text):
        return False
    block = build_deps_path_bootstrap_block(dependency_dir=dependency_dir)
    if text.startswith("#!"):
        lines = text.splitlines(keepends=True)
        insert_at = 1
        while insert_at < len(lines) and lines[insert_at].strip().startswith("#"):
            insert_at += 1
        updated = "".join(lines[:insert_at] + [block] + lines[insert_at:])
    else:
        updated = block + text
    operator_path.write_text(updated, encoding="utf-8")
    return True


def build_repo_path_bootstrap_block(*, repo_relative_from_workspace: str) -> str:
    rel = repo_relative_from_workspace.replace("\\", "/").strip() or "."
    ws_expr = "Path(__file__).resolve().parent"
    if rel != ".":
        for part in Path(rel).parts:
            if part == "..":
                ws_expr = f"({ws_expr}).parent"
            elif part != ".":
                ws_expr = f'{ws_expr} / "{part}"'
    return "\n".join(
        [
            REPO_PATH_BOOTSTRAP_START,
            "import sys",
            "from pathlib import Path",
            "",
            f"_REPO_ROOT = ({ws_expr}).resolve()",
            '_KERNELS_ROOT = _REPO_ROOT / "src" / "kernels"',
            "for _import_path in (_KERNELS_ROOT, _REPO_ROOT):",
            '    _entry = str(_import_path)',
            "    if _import_path.is_dir() and _entry not in sys.path:",
            "        sys.path.insert(0, _entry)",
            REPO_PATH_BOOTSTRAP_END,
            "",
        ],
    )


def inject_repo_path_bootstrap(
    operator_path: Path,
    *,
    repo_root: Path,
    workspace: Path,
) -> bool:
    repo_root = repo_root.resolve()
    workspace = workspace.resolve()
    rel = os.path.relpath(repo_root, workspace)
    block = build_repo_path_bootstrap_block(repo_relative_from_workspace=rel)
    text = operator_path.read_text(encoding="utf-8")
    if has_repo_path_bootstrap(text):
        return False
    if text.startswith("#!"):
        lines = text.splitlines(keepends=True)
        insert_at = 1
        while insert_at < len(lines) and lines[insert_at].strip().startswith("#"):
            insert_at += 1
        updated = "".join(lines[:insert_at] + [block] + lines[insert_at:])
    else:
        updated = block + text
    operator_path.write_text(updated, encoding="utf-8")
    return True


def run_operator_import_smoke(
    workspace: Path,
    operator_filename: str,
    *,
    dependency_dir: str = DEFAULT_DEPENDENCY_DIR,
) -> tuple[bool, str]:
    stem = Path(operator_filename).stem
    if not stem:
        return False, "operator_filename has no module stem"
    command = (
        "import importlib, sys; "
        f"importlib.import_module({stem!r}); "
        "print('ok')"
    )
    env = os.environ.copy()
    deps_root = (workspace / dependency_dir).resolve()
    if deps_root.is_dir() and any(deps_root.rglob("*.py")):
        deps_entry = str(deps_root)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            deps_entry if not existing else f"{deps_entry}{os.pathsep}{existing}"
        )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=_IMPORT_SMOKE_TIMEOUT_SECONDS,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"import smoke timed out after {_IMPORT_SMOKE_TIMEOUT_SECONDS}s"
    if completed.returncode == 0:
        return True, ""
    detail = (completed.stderr or completed.stdout or "").strip()
    return False, detail or f"import smoke exited {completed.returncode}"


def verify_dependency_issues(
    workspace: Path,
    meta: dict[str, Any],
    *,
    operator_text: str,
    operator_filename: str,
) -> list[str]:
    issues: list[str] = []
    raw_dependency_dir = str(meta.get("dependency_dir", DEFAULT_DEPENDENCY_DIR)).strip()
    if raw_dependency_dir and (
        "{" in raw_dependency_dir
        or "}" in raw_dependency_dir
        or raw_dependency_dir != DEFAULT_DEPENDENCY_DIR
    ):
        issues.append(
            f"dependency_dir must be the literal name {DEFAULT_DEPENDENCY_DIR!r}, "
            f"not {raw_dependency_dir!r} (template placeholders like '{{deps}}' are invalid)",
        )

    if (workspace / "{deps}").is_dir():
        issues.append(
            f"remove or rename directory '{{deps}}' to {DEFAULT_DEPENDENCY_DIR!r}; "
            "run sync_workspace_dependencies.py",
        )

    strategy = str(meta.get("dependency_strategy", "")).strip() or STRATEGY_REPO_PATH
    dependency_dir = normalize_dependency_dir(meta.get("dependency_dir"))
    deps_root = workspace / dependency_dir

    if strategy == STRATEGY_REPO_PATH and operator_text and not has_repo_path_bootstrap(operator_text):
        issues.append(
            "operator is missing repo path bootstrap; run "
            "sync_workspace_dependencies.py --repo <repo_root>",
        )

    if strategy == STRATEGY_DEPS_COPY:
        repo_modules = repo_scoped_import_modules(operator_text)
        py_files = list(deps_root.rglob("*.py")) if deps_root.is_dir() else []
        if repo_modules and not py_files:
            issues.append(
                f"dependency_strategy is {STRATEGY_DEPS_COPY!r} but {dependency_dir}/ has no .py files",
            )
        missing = missing_repo_modules_in_deps(repo_modules, deps_root)
        if missing:
            issues.append(
                "operator imports repo modules not present under "
                f"{dependency_dir}/: {', '.join(missing)}",
            )

    for entry in _string_list(meta.get("copied_dependencies")):
        normalized = entry.replace("\\", "/").lstrip("./")
        target = workspace / normalized
        if not target.is_file():
            issues.append(f"copied_dependencies entry missing on disk: {entry}")

    if operator_filename and operator_text:
        smoke_ok, smoke_err = run_operator_import_smoke(
            workspace,
            operator_filename,
            dependency_dir=dependency_dir,
        )
        if not smoke_ok:
            issues.append(
                "operator import smoke failed"
                + (f": {smoke_err}" if smoke_err else "")
                + "; fix imports or run sync_workspace_dependencies.py --copy-deps",
            )
    return issues


def missing_repo_modules_in_deps(modules: list[str], deps_root: Path) -> list[str]:
    missing: list[str] = []
    for module in modules:
        if not _module_exists_under_deps(module, deps_root):
            missing.append(module)
    return missing


def sync_workspace_dependencies(
    workspace: Path,
    repo_root: Path,
    *,
    meta_path: Path | None = None,
    force_deps_copy: bool = False,
) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    meta_path = meta_path or (workspace / "validation-meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"invalid validation-meta.json: {meta_path}")

    operator_name = str(meta.get("operator_filename", "")).strip()
    if not operator_name:
        raise ValueError(f"validation-meta.json missing operator_filename: {meta_path}")
    operator_path = workspace / operator_name
    operator_text = operator_path.read_text(encoding="utf-8")

    dependency_dir = normalize_dependency_dir(meta.get("dependency_dir"))
    deps_root = workspace / dependency_dir
    deps_root.mkdir(parents=True, exist_ok=True)
    _migrate_brace_deps_dir(workspace, deps_root)

    warnings: list[str] = []
    copied: list[str] = []
    unresolved: list[str] = []
    strategy = STRATEGY_DEPS_COPY if force_deps_copy else STRATEGY_REPO_PATH

    if strategy == STRATEGY_REPO_PATH:
        inject_repo_path_bootstrap(operator_path, repo_root=repo_root, workspace=workspace)
        operator_text = operator_path.read_text(encoding="utf-8")

    smoke_ok, smoke_err = run_operator_import_smoke(
        workspace,
        operator_name,
        dependency_dir=dependency_dir,
    )
    if strategy == STRATEGY_DEPS_COPY:
        copied, unresolved = copy_deps_closure(
            workspace=workspace,
            repo_root=repo_root,
            operator_text=operator_text,
            deps_root=deps_root,
            dependency_dir=dependency_dir,
        )
        if copied:
            _write_deps_conftest(workspace, dependency_dir)
            inject_deps_path_bootstrap(operator_path, dependency_dir=dependency_dir)
            operator_text = operator_path.read_text(encoding="utf-8")
        smoke_ok, smoke_err = run_operator_import_smoke(
            workspace,
            operator_name,
            dependency_dir=dependency_dir,
        )
    elif not smoke_ok:
        warnings.append(
            "repo path bootstrap import smoke failed; falling back to deps/ copy: "
            + (smoke_err or "unknown error"),
        )
        copied, unresolved = copy_deps_closure(
            workspace=workspace,
            repo_root=repo_root,
            operator_text=operator_path.read_text(encoding="utf-8"),
            deps_root=deps_root,
            dependency_dir=dependency_dir,
        )
        strategy = STRATEGY_DEPS_COPY
        if copied:
            _write_deps_conftest(workspace, dependency_dir)
            inject_deps_path_bootstrap(operator_path, dependency_dir=dependency_dir)
            operator_text = operator_path.read_text(encoding="utf-8")
        smoke_ok, smoke_err = run_operator_import_smoke(
            workspace,
            operator_name,
            dependency_dir=dependency_dir,
        )

    meta["dependency_dir"] = dependency_dir
    meta["dependency_strategy"] = strategy
    meta["repo_root"] = repo_root.as_posix()
    meta["repo_path_injected"] = strategy == STRATEGY_REPO_PATH or has_repo_path_bootstrap(
        operator_path.read_text(encoding="utf-8"),
    )
    meta["import_smoke_passed"] = smoke_ok
    meta["copied_dependencies"] = copied
    if unresolved:
        meta["unresolved_repo_imports"] = unresolved
    elif "unresolved_repo_imports" in meta:
        del meta["unresolved_repo_imports"]
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "workspace": workspace.name,
        "dependency_dir": dependency_dir,
        "dependency_strategy": strategy,
        "import_smoke_passed": smoke_ok,
        "import_smoke_error": smoke_err or None,
        "copied_dependencies": copied,
        "unresolved_repo_imports": unresolved,
        "warnings": warnings,
        "repo_imports": repo_scoped_import_modules(operator_text),
    }


def copy_deps_closure(
    *,
    workspace: Path,
    repo_root: Path,
    operator_text: str,
    deps_root: Path,
    dependency_dir: str,
) -> tuple[list[str], list[str]]:
    copied: list[str] = []
    unresolved: list[str] = []
    pending = list(repo_scoped_import_modules(operator_text))
    seen_modules: set[str] = set()
    while pending:
        module = pending.pop(0)
        if module in seen_modules:
            continue
        seen_modules.add(module)
        source_file = _resolve_repo_module_file(repo_root, module)
        if source_file is None:
            unresolved.append(module)
            continue
        destination = _deps_destination_for_source(
            deps_root,
            module,
            source_file=source_file,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)
        _ensure_package_inits(destination.parent, deps_root)
        rel = destination.relative_to(workspace).as_posix()
        if rel not in copied:
            copied.append(rel)
        nested_source = destination.read_text(encoding="utf-8")
        for nested in repo_scoped_import_modules(nested_source):
            if nested not in seen_modules:
                pending.append(nested)
    return copied, unresolved


def _deps_destination_for_source(
    deps_root: Path,
    module: str,
    *,
    source_file: Path,
) -> Path:
    if source_file.name == "__init__.py":
        if module.startswith(_SRC_KERNELS_FLA_PREFIX):
            suffix_parts = module[len(_SRC_KERNELS_FLA_PREFIX) + 1 :].split(".")
        else:
            suffix_parts = module.split(".")[1:]
        destination = deps_root / "fla"
        if suffix_parts and suffix_parts[0]:
            destination = destination.joinpath(*suffix_parts)
        return destination / "__init__.py"
    return _deps_destination_for_module(deps_root, module)


def _deps_destination_for_module(deps_root: Path, module: str) -> Path:
    if module.startswith(_SRC_KERNELS_FLA_PREFIX):
        suffix_parts = module[len(_SRC_KERNELS_FLA_PREFIX) + 1 :].split(".")
        destination = deps_root / "fla"
        if suffix_parts and suffix_parts[0]:
            destination = destination.joinpath(*suffix_parts)
    else:
        package = module.split(".", 1)[0]
        suffix_parts = module.split(".")[1:] if "." in module else []
        destination = deps_root / _DEPS_PACKAGE_DIRS[package]
        if suffix_parts:
            destination = destination.joinpath(*suffix_parts)
    if destination.suffix != ".py":
        if (destination / "__init__.py").exists():
            return destination / "__init__.py"
        return destination.with_suffix(".py")
    return destination


def _resolve_repo_module_file(repo_root: Path, module: str) -> Path | None:
    if module.startswith(_SRC_KERNELS_FLA_PREFIX):
        suffix = module[len(_SRC_KERNELS_FLA_PREFIX) + 1 :]
        base = repo_root / _REPO_SOURCE_ROOTS["fla"]
        if suffix:
            module_path = base.joinpath(*suffix.split("."))
            candidates = [module_path.with_suffix(".py"), module_path / "__init__.py"]
        else:
            candidates = [base / "__init__.py", base.with_suffix(".py")]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    root_name = module.split(".", 1)[0]
    package_root = _REPO_SOURCE_ROOTS.get(root_name)
    if package_root is None:
        return None
    suffix = module.split(".", 1)[1] if "." in module else ""
    base = repo_root / package_root
    if suffix:
        module_path = base.joinpath(*suffix.split("."))
        candidates = [module_path.with_suffix(".py"), module_path / "__init__.py"]
    else:
        candidates = [base / "__init__.py", base.with_suffix(".py")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _module_exists_under_deps(module: str, deps_root: Path) -> bool:
    destination = _deps_destination_for_module(deps_root, module)
    if destination.is_file():
        return True
    init_candidate = (
        destination.parent / "__init__.py"
        if destination.suffix == ".py" and destination.name != "__init__.py"
        else destination / "__init__.py"
    )
    return init_candidate.is_file()


def _migrate_brace_deps_dir(workspace: Path, deps_root: Path) -> None:
    brace_dir = workspace / "{deps}"
    if not brace_dir.is_dir() or brace_dir == deps_root:
        return
    if not any(deps_root.iterdir()) and any(brace_dir.iterdir()):
        for item in brace_dir.iterdir():
            target = deps_root / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    shutil.rmtree(brace_dir)


def _write_deps_conftest(workspace: Path, dependency_dir: str) -> None:
    conftest = workspace / "conftest.py"
    if conftest.is_file():
        return
    conftest.write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                "",
                f"_deps = Path(__file__).resolve().parent / {dependency_dir!r}",
                'if _deps.is_dir() and str(_deps) not in sys.path:',
                "    sys.path.insert(0, str(_deps))",
                "",
            ],
        ),
        encoding="utf-8",
    )


def _ensure_package_inits(leaf_dir: Path, deps_root: Path) -> None:
    current = leaf_dir
    while deps_root in current.parents or current == deps_root:
        init_path = current / "__init__.py"
        if not init_path.exists():
            init_path.write_text("", encoding="utf-8")
        if current == deps_root:
            break
        current = current.parent


def _extract_imported_modules_regex(source: str) -> list[str]:
    modules: list[str] = []
    for match in re.finditer(r"^\s*from\s+([a-zA-Z0-9_.]+)\s+import", source, re.MULTILINE):
        modules.append(match.group(1))
    for match in re.finditer(r"^\s*import\s+([a-zA-Z0-9_.]+)", source, re.MULTILINE):
        modules.append(match.group(1))
    return _stable_unique(modules)


def _stable_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
