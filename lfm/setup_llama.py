"""One-shot installers: llama.cpp binaries and the default GGUF."""

from __future__ import annotations

import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import httpx

from .config import DEFAULT_MODEL_FILE, MODEL_URL, data_home, models_home

Progress = Callable[[int, int | None], None]


def _target_asset() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in ("x86_64", "amd64"):
        return "-bin-ubuntu-x64.tar.gz"
    if system == "darwin" and machine in ("arm64", "aarch64"):
        return "-bin-macos-arm64.zip"
    raise RuntimeError(f"no prebuilt llama.cpp binaries for {system}/{machine}; build it manually")


def install_llama(progress: Progress | None = None) -> Path:
    """Download the newest nightly llama.cpp build into ~/.local/share/lfm/bin."""
    suffix = _target_asset()
    with httpx.Client(timeout=httpx.Timeout(60.0), follow_redirects=True) as client:
        releases = client.get(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases", params={"per_page": 15}
        ).json()
        asset = None
        tag = ""
        for release in releases:
            for candidate in release.get("assets", []):
                if candidate["name"].startswith("llama-b") and candidate["name"].endswith(suffix):
                    asset = candidate
                    tag = release["tag_name"]
                    break
            if asset:
                break
        if asset is None:
            raise RuntimeError("could not find a suitable llama.cpp release")

        target_dir = data_home() / "bin"
        target_dir.mkdir(parents=True, exist_ok=True)
        total = int(asset["size"])
        archive = Path(tempfile.gettempdir()) / asset["name"]

        with client.stream("GET", asset["browser_download_url"]) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                done = 0
                for block in response.iter_bytes(256 * 1024):
                    handle.write(block)
                    done += len(block)
                    if progress:
                        progress(done, total)

        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(target_dir)
        else:
            with tarfile.open(archive) as bundle:
                members = bundle.getmembers()
                root = members[0].name.split("/")[0] + "/"
                for member in members:
                    if member.name.startswith(root):
                        member.name = member.name[len(root):]
                    if member.name:
                        bundle.extract(member, target_dir)
        archive.unlink(missing_ok=True)

    binary = target_dir / "llama-server"
    binary.chmod(0o755)
    for extra in target_dir.glob("*.so*"):
        extra.chmod(0o755)
    (data_home() / "version.txt").write_text(f"{tag}\n", encoding="utf-8")
    return binary


def download_model(progress: Progress | None = None) -> Path:
    target = models_home() / DEFAULT_MODEL_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")

    with httpx.Client(timeout=httpx.Timeout(60.0, read=None), follow_redirects=True) as client:
        with client.stream("GET", MODEL_URL) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0) or None
            done = 0
            with part.open("wb") as handle:
                for block in response.iter_bytes(1024 * 1024):
                    handle.write(block)
                    done += len(block)
                    if progress:
                        progress(done, total)
    part.replace(target)
    return target
