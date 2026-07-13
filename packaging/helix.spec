from pathlib import Path


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


def _collect_build_meta():
    meta_path = ROOT / "src" / "helix" / "_build_meta.json"
    if meta_path.is_file():
        return [(str(meta_path), "helix")]
    return []


a = Analysis(
    [str(ROOT / "src" / "helix" / "cli.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=collect_skills() + _collect_build_meta(),
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
    name="helix",
    strip=False,
    upx=False,
    console=True,
)
