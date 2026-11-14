"""llama-server lifecycle and the streaming chat client."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterator

import httpx

from .config import Config


class ServerError(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LlamaServer:
    """Owns one llama-server process and speaks OpenAI-compatible HTTP to it."""

    def __init__(self, binary: Path, model: Path, cfg: Config) -> None:
        self.binary = Path(binary)
        self.model = Path(model)
        self.cfg = cfg
        self.port = cfg.port or free_port()
        self.proc: subprocess.Popen[str] | None = None
        self.log: deque[str] = deque(maxlen=200)
        self._reader: threading.Thread | None = None
        self.on_line: Callable[[str], None] | None = None

    # ------------------------------------------------------------------ #

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.host}:{self.port}"

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def command(self) -> list[str]:
        cfg = self.cfg
        threads = cfg.threads or os.cpu_count() or 4
        cmd = [
            str(self.binary),
            "-m", str(self.model),
            "--host", cfg.host,
            "--port", str(self.port),
            "-c", str(cfg.ctx),
            "-t", str(threads),
            "-ngl", str(cfg.gpu_layers),
            "-np", "1",
            "--jinja",
        ]
        if cfg.thinking:
            cmd += ["--reasoning", "on", "--reasoning-format", "deepseek"]
        else:
            cmd += ["--reasoning", "off"]
        return cmd

    # ------------------------------------------------------------------ #

    def start(self, timeout: float = 300.0, on_line: Callable[[str], None] | None = None) -> None:
        if self.alive:
            return
        self.on_line = on_line
        self.log.clear()

        for attempt in range(2):
            self.proc = subprocess.Popen(
                self.command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._reader = threading.Thread(target=self._pump, daemon=True)
            self._reader.start()

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not self.alive:
                    tail = "\n".join(list(self.log)[-14:])
                    if "address already in use" in tail.lower() and attempt == 0:
                        self.port = free_port()
                        break
                    raise ServerError(
                        f"llama-server failed to start (code {self.proc.returncode if self.proc else '?'}):\n{tail}"
                    )
                try:
                    response = httpx.get(f"{self.base_url}/health", timeout=1.0)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            else:
                self.stop()
                raise ServerError(
                    "llama-server did not become healthy in time:\n"
                    + "\n".join(list(self.log)[-14:])
                )
            self.stop()

        raise ServerError("could not find a free port for llama-server")

    def _pump(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            self.log.append(line)
            callback = self.on_line
            if callback is not None:
                try:
                    callback(line)
                except Exception:
                    pass

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except ProcessLookupError:
                return
        try:
            proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()

    # ------------------------------------------------------------------ #

    def chat_stream(
        self,
        messages: list[dict],
        cancel: threading.Event | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[dict]:
        cfg = self.cfg
        payload: dict = {
            "model": "lfm",
            "messages": messages,
            "stream": True,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "cache_prompt": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if cfg.max_tokens and cfg.max_tokens > 0:
            payload["max_tokens"] = cfg.max_tokens

        timings: dict = {}
        usage: dict = {}
        calls: dict[int, dict] = {}
        timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None)

        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream(
                    "POST", f"{self.base_url}/v1/chat/completions", json=payload
                ) as response:
                    if response.status_code != 200:
                        body = response.read().decode("utf-8", "replace")
                        yield {"type": "error", "message": f"HTTP {response.status_code}: {body[:600]}"}
                        return
                    for line in response.iter_lines():
                        if cancel is not None and cancel.is_set():
                            return
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if not line:
                            continue
                        if line == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("timings"):
                            timings = chunk["timings"]
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            reasoning = delta.get("reasoning_content")
                            if reasoning:
                                yield {"type": "reasoning", "text": reasoning}
                            content = delta.get("content")
                            if content:
                                yield {"type": "content", "text": content}
                            for call in delta.get("tool_calls") or []:
                                slot = calls.setdefault(
                                    int(call.get("index") or 0),
                                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                                )
                                if call.get("id"):
                                    slot["id"] = call["id"]
                                function = call.get("function") or {}
                                if function.get("name"):
                                    slot["function"]["name"] += function["name"]
                                if function.get("arguments"):
                                    slot["function"]["arguments"] += function["arguments"]
        except httpx.HTTPError as exc:
            yield {"type": "error", "message": f"llama-server connection: {exc}"}
            return

        if calls:
            yield {"type": "tool_calls", "calls": [calls[index] for index in sorted(calls)]}
        yield {"type": "done", "timings": timings, "usage": usage}


def tail(server: LlamaServer, lines: int = 12) -> str:
    return "\n".join(list(server.log)[-lines:])
