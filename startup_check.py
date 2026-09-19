"""Fail-fast checks for the Project 1 GUI runtime."""
from __future__ import annotations

import importlib
import os
import site
import sys
from pathlib import Path


REQUIRED = ("numpy", "scipy", "matplotlib", "tkinter", "PIL")
OPTIONAL_COMPILED = ("pandas", "pyarrow", "numexpr", "bottleneck")


def run_startup_check() -> dict:
    expected = Path(os.environ.get("HEATSINK_GUI_ENV", "")).resolve()
    executable = Path(sys.executable).resolve()
    if expected and expected.exists() and executable.parent != expected:
        raise RuntimeError(f"Wrong Python interpreter.\nExpected environment: {expected}\nRunning: {executable}")
    if "heatsink-gui" not in str(executable).lower():
        raise RuntimeError(f"Wrong Python interpreter.\nThis application must run from heatsink-gui, not:\n{executable}")
    if site.ENABLE_USER_SITE:
        raise RuntimeError("The Python user site is enabled. Launch through launch_heatsink_gui.cmd so packages cannot be mixed.")
    versions = {}
    failures = []
    for name in REQUIRED + OPTIONAL_COMPILED:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "ok")
        except Exception as error:
            if name in REQUIRED:
                failures.append(f"{name}: {error}")
            else:
                versions[name] = f"unavailable ({error})"
    if failures:
        raise RuntimeError("Required imports failed:\n" + "\n".join(failures))
    return {"executable": str(executable), "python": sys.version.split()[0], "versions": versions}
