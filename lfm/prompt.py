"""The owner's own system prompt, kept in a file outside the repository.

Nothing in this module contains prompt text: it only finds and reads the file.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import config_home

DEFAULT_NAME = "system.md"


def system_file() -> Path:
    """Where the local system prompt lives (LFM_SYSTEM_FILE overrides)."""
    override = os.environ.get("LFM_SYSTEM_FILE")
    if override:
        return Path(override).expanduser()
    return config_home() / DEFAULT_NAME


def load_system(fallback: str) -> tuple[str, Path | None]:
    """Return (prompt, path). Falls back when the file is missing or empty."""
    path = system_file()
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return fallback, None
    if not text:
        return fallback, None
    return text, path


def prompt_source(path: Path | None) -> str:
    return str(path) if path is not None else "built-in default"
