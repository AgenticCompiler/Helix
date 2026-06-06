from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata


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


def collect_optional_metadata(package_name):
    try:
        return copy_metadata(package_name)
    except Exception:
        return []


a = Analysis(
    [str(ROOT / "src" / "triton_agent" / "cli.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=collect_skills() + collect_optional_metadata("fastmcp"),
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
