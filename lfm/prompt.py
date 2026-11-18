"""Where the system prompt comes from.

Order: ``LFM_SYSTEM_FILE`` → ``~/.config/lfm/system.md`` → the packaged
``lfm/system.md`` → the built-in default. The first readable, non-empty file
wins, so a clone works out of the box and a local file overrides it.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import config_home

DEFAULT_NAME = "system.md"


def system_file() -> Path:
    """The per-user override (LFM_SYSTEM_FILE wins over the config dir)."""
    override = os.environ.get("LFM_SYSTEM_FILE")
    if override:
        return Path(override).expanduser()
    return config_home() / DEFAULT_NAME


def packaged_file() -> Path:
    """The prompt shipped with the package."""
    return Path(__file__).resolve().parent / DEFAULT_NAME


def _read(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def load_system(fallback: str) -> tuple[str, Path | None]:
    """Return (prompt, path) for the first source that has text."""
    for path in (system_file(), packaged_file()):
        text = _read(path)
        if text is not None:
            return text, path
    return fallback, None


def prompt_source(path: Path | None) -> str:
    return str(path) if path is not None else "built-in default"
