"""The early PyInstaller hook preserves user intent without loading packages."""
import builtins
import os
from pathlib import Path
import runpy
import sys

import pytest


HOOK_PATH = Path(__file__).resolve().parents[1] / "runtime_cache.py"
ATTRIBUTE = "_fraggenotyper_original_mplconfigdir"


@pytest.mark.parametrize("original", [None, "", "/explicit/cache with spaces"])
def test_hook_records_original_setting_without_changing_environment_or_loading_heavy_packages(monkeypatch, original):
    if original is None:
        monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    else:
        monkeypatch.setenv("MPLCONFIGDIR", original)
    # Register the attribute with monkeypatch so it is removed/restored at exit.
    monkeypatch.setattr(sys, ATTRIBUTE, object(), raising=False)
    environment_before = dict(os.environ)
    import_module = builtins.__import__
    forbidden = {"numpy", "pandas", "matplotlib", "scipy", "sklearn", "PyQt6",
                 "startup", "app_window"}

    def guarded_import(name, *args, **kwargs):
        assert name.partition(".")[0] not in forbidden, f"Runtime hook imported {name}"
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runpy.run_path(str(HOOK_PATH), run_name="fraggenotyper_runtime_cache_probe")

    assert getattr(sys, ATTRIBUTE) == original
    assert dict(os.environ) == environment_before
