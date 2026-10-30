"""Common PyInstaller-aware bundled resource and user-data paths."""
from __future__ import annotations

from pathlib import Path
import os
import sys


def resource_path(relative: str | Path) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


def close_startup_splash() -> None:
    """Close PyInstaller's one-file splash when the main window is ready."""
    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass


def user_data_path(*parts: str) -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ChineseSemanticToolkit"
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

