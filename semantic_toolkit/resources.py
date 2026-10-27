"""Common PyInstaller-aware bundled resource and user-data paths."""
from __future__ import annotations

from pathlib import Path
import os
import sys


def resource_path(relative: str | Path) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


def user_data_path(*parts: str) -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ChineseSemanticToolkit"
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

