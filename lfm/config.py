"""Configuration and model discovery."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

APP = "lfm"

DEFAULT_MODEL_FILE = "LFM3-1.7B-Uncensored-AD-Q4_K_M.gguf"
MODEL_REPO = "AtomicChat/LFM3-1.7B-Uncensored-GGUF"
MODEL_URL = (
    f"https://huggingface.co/{MODEL_REPO}/resolve/main/{DEFAULT_MODEL_FILE}?download=true"
)

DEFAULT_SYSTEM = (
    "You are lfm, a local AI assistant running on the user's own computer via "
    "llama.cpp. Be direct, warm and concise. Answer in the language the user "
    "writes in."
)

MODEL_DIRS = (
    Path.cwd() / "models",
    Path.home() / "models",
)


def config_home() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return Path(base) / APP if base else Path.home() / ".config" / APP


def data_home() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    return Path(base) / APP if base else Path.home() / ".local" / "share" / APP


def config_path() -> Path:
    return config_home() / "config.toml"


def models_home() -> Path:
    return data_home() / "models"


@dataclass
class Config:
    model: str = ""
    ctx: int = 8192
    threads: int = 0
    gpu_layers: int = 0
    host: str = "127.0.0.1"
    port: int = 0
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = -1
    thinking: bool = False
    agent: bool = False
    agent_steps: int = 8
    system_prompt: str = DEFAULT_SYSTEM

    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        target = path or config_path()
        cfg = cls()
        if not target.is_file():
            return cfg
        try:
            data = tomllib.loads(target.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return cfg
        for field in fields(cls):
            if field.name in data:
                value = data[field.name]
                if isinstance(getattr(cfg, field.name), bool):
                    value = bool(value)
                setattr(cfg, field.name, value)
        return cfg

    def save(self, path: Path | None = None) -> Path:
        target = path or config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_toml(), encoding="utf-8")
        return target

    def to_toml(self) -> str:
        lines = ["# lfm — local LFM3 chat over llama.cpp", ""]
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, str):
                rendered = json.dumps(value, ensure_ascii=False)
            else:
                rendered = repr(value)
            lines.append(f"{field.name} = {rendered}")
        return "\n".join(lines) + "\n"


def _score(path: Path) -> int:
    name = path.name.lower()
    score = 0
    if "lfm3" in name or "lfm-3" in name:
        score += 100
    elif "lfm" in name:
        score += 40
    if "q4_k_m" in name:
        score += 25
    elif "q4" in name:
        score += 10
    if "uncensored" in name:
        score += 2
    return score


def find_model(explicit: str | None = None) -> Path | None:
    """Locate a .gguf, preferring LFM3 Q4_K_M builds."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None

    env = os.environ.get("LFM_MODEL")
    if env:
        path = Path(env).expanduser()
        if path.is_file():
            return path

    cfg = Config.load()
    if cfg.model:
        path = Path(cfg.model).expanduser()
        if path.is_file():
            return path

    candidates: list[Path] = []
    for directory in (*MODEL_DIRS, models_home(), Path.home() / "Downloads"):
        if not directory.is_dir():
            continue
        try:
            candidates.extend(p for p in directory.glob("*.gguf") if p.is_file())
        except OSError:
            continue
    if not candidates:
        return None

    lfm = [p for p in candidates if "lfm" in p.name.lower()]
    pool = lfm or candidates
    return max(pool, key=lambda p: (_score(p), p.stat().st_mtime))


def find_server_binary(explicit: str | None = None) -> Path | None:
    """Locate llama-server: env, project vendor build, lfm data dir, PATH."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None

    env = os.environ.get("LFM_LLAMA_SERVER")
    if env:
        path = Path(env).expanduser()
        if path.is_file():
            return path

    from importlib.resources import files

    candidates = [
        data_home() / "bin" / "llama-server",
        Path(__file__).resolve().parent.parent / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server",
    ]
    for path in candidates:
        if path.is_file():
            return path

    from shutil import which

    found = which("llama-server")
    return Path(found) if found else None


MODEL_DOWNLOAD_HINT = (
    "Download the model into ~/.local/share/lfm/models/:\n"
    f"  curl -L -o ~/.local/share/lfm/models/{DEFAULT_MODEL_FILE} \\\n"
    f'    "{MODEL_URL}"'
)
