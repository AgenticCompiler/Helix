from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_NAME = "triton-agent"


DEFAULT_SPEC = '''from pathlib import Path


SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent if SPEC_DIR.name == "packaging" else SPEC_DIR


def collect_skills():
    skills_root = ROOT / "skills"
    datas = []
    for path in sorted(skills_root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        target_dir = Path("skills") / path.relative_to(skills_root).parent
        datas.append((str(path), str(target_dir)))
    return datas


a = Analysis(
    [str(ROOT / "src" / "triton_agent" / "cli.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=collect_skills(),
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tests"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="triton-agent",
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="triton-agent",
)
'''


def platform_tag() -> str:
    system = platform.system().lower()
    if system == "darwin":
        os_name = "macos"
    elif system == "windows":
        os_name = "windows"
    elif system == "linux":
        os_name = "linux"
    else:
        os_name = system or "unknown-os"

    machine = platform.machine().lower()
    machine = machine.replace("amd64", "x86_64").replace("arm64", "aarch64")
    return f"{os_name}-{machine or 'unknown-arch'}"


def resolve_repo(path: str) -> Path:
    repo = Path(path).expanduser().resolve()
    if not (repo / "pyproject.toml").is_file():
        raise FileNotFoundError(f"pyproject.toml not found under repository path: {repo}")
    if not (repo / "skills").is_dir():
        raise FileNotFoundError(f"skills directory not found under repository path: {repo}")
    return repo


def ensure_within_repo(path: Path, repo: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(repo.resolve())
    return resolved


def remove_path(path: Path, repo: Path) -> None:
    target = ensure_within_repo(path, repo)
    if not target.exists():
        return
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
        return
    target.unlink()


def run_command(command: Sequence[str], cwd: Path) -> None:
    print("+ " + subprocess.list2cmdline(list(command)), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def ensure_spec_file(spec_path: Path, repo: Path) -> Path:
    spec = ensure_within_repo(spec_path, repo)
    if spec.is_file():
        return spec
    if spec.exists():
        raise FileExistsError(f"PyInstaller spec path exists but is not a file: {spec}")
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(DEFAULT_SPEC, encoding="utf-8")
    print(f"Created default PyInstaller spec: {spec}", flush=True)
    return spec


def zip_directory(source_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, path.relative_to(source_dir.parent))


def executable_path(bundle_dir: Path) -> Path:
    name = f"{PROJECT_NAME}.exe" if os.name == "nt" else PROJECT_NAME
    return bundle_dir / name


def build(args: argparse.Namespace) -> int:
    repo = resolve_repo(args.repo)
    spec_path = ensure_spec_file(repo / args.spec, repo)

    tag = args.platform_tag or platform_tag()
    artifact_root = ensure_within_repo(repo / args.artifact_dir, repo)
    platform_root = artifact_root / f"{PROJECT_NAME}-{tag}"
    work_path = ensure_within_repo(repo / "build" / "pyinstaller" / tag, repo)
    bundle_dir = platform_root / PROJECT_NAME
    archive_path = artifact_root / f"{PROJECT_NAME}-{tag}.zip"

    if args.clean:
        remove_path(platform_root, repo)
        remove_path(work_path, repo)
        remove_path(archive_path, repo)

    artifact_root.mkdir(parents=True, exist_ok=True)
    command = [
        args.uv,
        "run",
        "pyinstaller",
        "--noconfirm",
        "--distpath",
        str(platform_root),
        "--workpath",
        str(work_path),
    ]
    if args.pyinstaller_clean:
        command.append("--clean")
    command.append(str(spec_path))

    run_command(command, repo)

    exe = executable_path(bundle_dir)
    if not exe.is_file():
        raise FileNotFoundError(f"Expected packaged executable was not created: {exe}")

    skills_dir = bundle_dir / "_internal" / "skills"
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"Expected bundled skills directory was not created: {skills_dir}")

    if not args.no_zip:
        zip_directory(bundle_dir, archive_path)
        print(f"Created archive: {archive_path}", flush=True)

    print(f"Created bundle: {bundle_dir}", flush=True)
    print(f"Executable: {exe}", flush=True)
    print(f"Bundled skills: {skills_dir}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the triton-agent PyInstaller bundle for the current OS. "
            "Run separately on Windows, Linux, and macOS to produce native artifacts."
        )
    )
    parser.add_argument("--repo", default=".", help="Path to the triton-agent repository.")
    parser.add_argument("--clean", action="store_true", help="Remove old artifacts for this platform first.")
    parser.add_argument(
        "--no-pyinstaller-clean",
        dest="pyinstaller_clean",
        action="store_false",
        help="Do not pass --clean to PyInstaller.",
    )
    parser.add_argument("--no-zip", action="store_true", help="Do not create the zip archive.")
    parser.add_argument("--platform-tag", help="Override auto-detected tag, e.g. windows-x86_64.")
    parser.add_argument(
        "--artifact-dir",
        default="dist/pyinstaller",
        help="Artifact output directory relative to the repository root.",
    )
    parser.add_argument(
        "--spec",
        default="packaging/triton-agent.spec",
        help="Spec path relative to the repository root.",
    )
    parser.add_argument("--uv", default="uv", help="uv executable to invoke.")
    parser.set_defaults(pyinstaller_clean=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return build(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
