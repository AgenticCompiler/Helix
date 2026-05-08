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
    a.binaries,
    a.datas,
    [],
    name="triton-agent",
    strip=False,
    upx=False,
    console=True,
)
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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

    machine = platform.machine().lower().replace("amd64", "x86_64")
    machine = machine.replace("arm64", "aarch64")
    return f"{os_name}-{machine or 'unknown-arch'}"


def ensure_within_repo(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def remove_path(path: Path, root: Path) -> None:
    target = ensure_within_repo(path, root)
    if not target.exists():
        return
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
        return
    target.unlink()


def run_command(command: Sequence[str], cwd: Path) -> None:
    print("+ " + subprocess.list2cmdline(list(command)), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def ensure_spec_file(spec_path: Path, root: Path) -> Path:
    spec = ensure_within_repo(spec_path, root)
    if spec.is_file():
        return spec
    if spec.exists():
        raise FileExistsError(f"PyInstaller spec path exists but is not a file: {spec}")
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(DEFAULT_SPEC, encoding="utf-8")
    print(f"Created default PyInstaller spec: {spec}", flush=True)
    return spec


def zip_executable(executable: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(executable, executable.relative_to(archive_path.parent))


def executable_name() -> str:
    return f"{PROJECT_NAME}.exe" if os.name == "nt" else PROJECT_NAME


def build(args: argparse.Namespace) -> int:
    root = repo_root()
    spec_path = ensure_spec_file(root / args.spec, root)

    tag = args.platform_tag or platform_tag()
    artifact_root = ensure_within_repo(root / args.artifact_dir, root)
    platform_root = artifact_root / f"{PROJECT_NAME}-{tag}"
    dist_path = platform_root
    work_path = ensure_within_repo(root / "build" / "pyinstaller" / tag, root)
    executable = dist_path / executable_name()
    archive_path = artifact_root / f"{PROJECT_NAME}-{tag}.zip"

    if args.clean:
        remove_path(platform_root, root)
        remove_path(work_path, root)
        remove_path(archive_path, root)

    artifact_root.mkdir(parents=True, exist_ok=True)
    command = [
        args.uv,
        "run",
        "pyinstaller",
        "--noconfirm",
        "--distpath",
        str(dist_path),
        "--workpath",
        str(work_path),
    ]
    if args.pyinstaller_clean:
        command.append("--clean")
    command.append(str(spec_path))

    run_command(command, root)

    if not executable.is_file():
        raise FileNotFoundError(f"Expected onefile executable was not created: {executable}")

    if not args.no_zip:
        zip_executable(executable, archive_path)
        print(f"Created archive: {archive_path}", flush=True)

    print(f"Created onefile executable: {executable}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the triton-agent PyInstaller onefile executable for the current OS. "
            "Run this script separately on Windows, Linux, and macOS to produce "
            "platform-specific executables."
        )
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove this platform's previous build, executable artifact, and zip before building.",
    )
    parser.add_argument(
        "--no-pyinstaller-clean",
        dest="pyinstaller_clean",
        action="store_false",
        help="Do not pass --clean to PyInstaller.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Build the onefile executable but do not create a zip archive.",
    )
    parser.add_argument(
        "--platform-tag",
        help="Override the auto-detected artifact tag, for example windows-x86_64.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="dist/pyinstaller",
        help="Directory for platform executable artifacts and zip archives.",
    )
    parser.add_argument(
        "--spec",
        default="packaging/triton-agent.spec",
        help="Path to the PyInstaller spec file, relative to the repository root.",
    )
    parser.add_argument(
        "--uv",
        default="uv",
        help="uv executable to invoke.",
    )
    parser.set_defaults(pyinstaller_clean=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
