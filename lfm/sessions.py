"""Chat transcripts on disk: one JSON file per conversation."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .config import data_home


def sessions_dir() -> Path:
    return data_home() / "sessions"


@dataclass
class Session:
    id: str
    created: float
    title: str = ""
    model: str = ""
    system: str = ""
    messages: list[dict] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def new(cls, model: str = "", system: str = "") -> "Session":
        return cls(id=uuid.uuid4().hex[:8], created=time.time(), model=model, system=system)

    def add(self, role: str, content: str, **extra) -> None:
        message = {"role": role, "content": content, "ts": time.time(), **extra}
        self.messages.append(message)
        if role == "user" and not self.title:
            self.title = " ".join(content.split())[:60]

    def save(self) -> Path:
        directory = sessions_dir()
        directory.mkdir(parents=True, exist_ok=True)
        if self.path is None:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.created))
            self.path = directory / f"{stamp}-{self.id}.json"
        data = {
            "id": self.id,
            "created": self.created,
            "updated": time.time(),
            "title": self.title,
            "model": self.model,
            "system": self.system,
            "messages": self.messages,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)
        return self.path
