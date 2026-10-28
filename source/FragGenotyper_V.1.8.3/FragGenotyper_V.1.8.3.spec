# -*- mode: python ; coding: utf-8 -*-
"""Native macOS app bundle; build Windows with build_windows.ps1 instead."""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


if sys.platform != "darwin":
    raise SystemExit("This spec builds the macOS app. On Windows use build_windows.ps1.")

project_dir = Path(SPECPATH).resolve()
application_name = "FragGenotyper_V.1.8.3"
icon_path = project_dir / "aq8s1-jliqi.icns"


def application_submodule(name):
    # Scientific wheels contain large test suites and optional testing dependencies.
    return not any(part in {"tests", "test", "_tests"} for part in name.split("."))


datas = [(str(icon_path), ".")]
binaries = []
hiddenimports = ["matplotlib.backends.backend_qtagg"]
# Older PyInstaller NumPy hooks miss modules imported from NumPy 2 C code,
# including numpy._core._exceptions. Collect runtime submodules explicitly.
hiddenimports += collect_submodules(
    "numpy", filter=lambda name: application_submodule(name)
    and not name.startswith(("numpy.f2py", "numpy.distutils")),
)
for scientific_package in ("scipy", "sklearn"):
    package_datas, package_binaries, package_imports = collect_all(
        scientific_package,
        include_py_files=False,
        filter_submodules=application_submodule,
        exclude_datas=["**/tests/**", "**/test/**", "**/_tests/**"],
        on_error="warn once",
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports

a = Analysis(
    [str(project_dir / "FragGenotyper_V.1.8.3.py")],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["QtAgg"]}},
    runtime_hooks=[str(project_dir / "runtime_cache.py")],
    excludes=["PySide6", "PySide2", "PyQt5", "tkinter", "torch", "tensorflow", "jax", "jaxlib", "IPython", "notebook"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=application_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # Build with a Python interpreter matching the target Mac.
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path),
)
collection = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=application_name,
)
app = BUNDLE(
    collection,
    name=application_name + ".app",
    icon=str(icon_path),
    bundle_identifier="org.fraggenotyper.desktop",
    info_plist={
        "CFBundleShortVersionString": "1.8.3",
        "CFBundleVersion": "1.8.3",
        "NSHighResolutionCapable": True,
    },
)
