"""Local tools the model may call when agent mode is on.

Everything runs inside one sandbox root (the directory lfm was started in).
Read-only tools run immediately; anything that writes or runs a command is
marked dangerous and the UI asks the user first.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path

MAX_OUTPUT = 6000
MAX_READ_BYTES = 400_000
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache", ".pytest_cache"}


class ToolError(RuntimeError):
    """A tool failure that is safe to hand back to the model."""


SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries of a directory inside the project.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "directory, default '.'"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "description": "1-based, default 1"},
                    "max_lines": {"type": "integer", "description": "default 400"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents with a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "directory, default '.'"},
                    "glob": {"type": "string", "description": "file name filter, e.g. '*.py'"},
                    "ignore_case": {"type": "boolean"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given text.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace one unique occurrence of old_string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "seconds, default 60"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a web page and return its text.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]

DANGEROUS = {"write_file", "edit", "run_shell"}


def _short(value: str, limit: int = MAX_OUTPUT) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n… truncated ({len(value) - limit} more chars)"


def describe(name: str, args: dict) -> str:
    """One-line summary of a call, for the transcript."""
    if name == "run_shell":
        return f"run_shell  {args.get('command', '')}"
    if name == "grep":
        extra = f" in {args['path']}" if args.get("path") else ""
        return f"grep  {args.get('pattern', '')}{extra}"
    if name == "fetch_url":
        return f"fetch_url  {args.get('url', '')}"
    if name in ("read_file", "write_file", "edit", "list_dir"):
        return f"{name}  {args.get('path', '.')}"
    return name


class Toolbox:
    """Executes tool calls, confined to one directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()

    # ------------------------------------------------------------------ #

    def dangerous(self, name: str) -> bool:
        return name in DANGEROUS

    def resolve(self, raw: str | None) -> Path:
        if not raw:
            return self.root
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.root / path
        try:
            real = path.resolve()
        except OSError as exc:
            raise ToolError(f"bad path {raw!r}: {exc}") from exc
        if real != self.root and self.root not in real.parents:
            raise ToolError(
                f"{raw!r} is outside the sandbox {self.root}; "
                "agent mode only touches the project directory"
            )
        return real

    def run(self, name: str, args: dict) -> str:
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            raise ToolError(f"unknown tool {name!r}")
        return handler(args)

    # ------------------------------------------------------------------ #

    def tool_list_dir(self, args: dict) -> str:
        path = self.resolve(args.get("path"))
        if not path.is_dir():
            raise ToolError(f"not a directory: {args.get('path')!r}")
        rows: list[str] = []
        for entry in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if entry.is_dir():
                rows.append(f"{entry.name}/")
            else:
                try:
                    rows.append(f"{entry.name}  {entry.stat().st_size} B")
                except OSError:
                    rows.append(entry.name)
        if not rows:
            return "(empty directory)"
        return _short("\n".join(rows[:400]))

    def tool_read_file(self, args: dict) -> str:
        path = self.resolve(args.get("path"))
        if not path.is_file():
            raise ToolError(f"no such file: {args.get('path')!r}")
        if path.stat().st_size > MAX_READ_BYTES:
            raise ToolError(f"file is too big ({path.stat().st_size} bytes); use grep")
        text = path.read_bytes().decode("utf-8", "replace")
        start = max(1, int(args.get("start_line") or 1))
        limit = max(1, min(2000, int(args.get("max_lines") or 400)))
        lines = text.splitlines()
        chunk = lines[start - 1:start - 1 + limit]
        if not chunk:
            return f"(no lines from {start} in a {len(lines)}-line file)"
        numbered = "\n".join(f"{start + i}\t{line}" for i, line in enumerate(chunk))
        if len(lines) > start - 1 + limit:
            numbered += f"\n… {len(lines) - (start - 1 + limit)} more lines"
        return _short(numbered)

    def tool_grep(self, args: dict) -> str:
        pattern = args.get("pattern")
        if not pattern:
            raise ToolError("pattern is required")
        flags = re.IGNORECASE if args.get("ignore_case") else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error as exc:
            raise ToolError(f"bad regex {pattern!r}: {exc}") from exc
        start = self.resolve(args.get("path"))
        name_glob = args.get("glob") or "*"
        hits: list[str] = []
        for file in self._walk(start):
            if not fnmatch.fnmatch(file.name, name_glob):
                continue
            try:
                data = file.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:4096]:
                continue
            text = data.decode("utf-8", "replace")
            for number, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{file.relative_to(self.root)}:{number}: {line.strip()[:200]}")
                    if len(hits) >= 200:
                        return _short("\n".join(hits) + "\n… stopped at 200 matches")
        return _short("\n".join(hits)) if hits else "no matches"

    def tool_write_file(self, args: dict) -> str:
        path = self.resolve(args.get("path"))
        content = args.get("content")
        if content is None:
            raise ToolError("content is required")
        if path.is_dir():
            raise ToolError(f"{args.get('path')!r} is a directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path.relative_to(self.root)}"

    def tool_edit(self, args: dict) -> str:
        path = self.resolve(args.get("path"))
        old = args.get("old_string")
        new = args.get("new_string")
        if not old:
            raise ToolError("old_string is required")
        if new is None:
            raise ToolError("new_string is required")
        if not path.is_file():
            raise ToolError(f"no such file: {args.get('path')!r}")
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            raise ToolError("old_string not found in the file")
        if count > 1:
            raise ToolError(f"old_string appears {count} times; add more context to make it unique")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {path.relative_to(self.root)}"

    def tool_run_shell(self, args: dict) -> str:
        command = (args.get("command") or "").strip()
        if not command:
            raise ToolError("command is required")
        timeout = max(1, min(300, int(args.get("timeout") or 60)))
        try:
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {timeout}s")
        parts = [f"exit {result.returncode}"]
        if result.stdout.strip():
            parts.append(result.stdout.rstrip())
        if result.stderr.strip():
            parts.append("[stderr]\n" + result.stderr.rstrip())
        return _short("\n".join(parts))

    def tool_fetch_url(self, args: dict) -> str:
        url = (args.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            raise ToolError("only http:// and https:// urls are allowed")
        import httpx

        try:
            response = httpx.get(
                url,
                timeout=20,
                follow_redirects=True,
                headers={"user-agent": "lfm/0.1 (+local agent)"},
            )
        except httpx.HTTPError as exc:
            raise ToolError(f"fetch failed: {exc}") from exc
        if response.status_code >= 400:
            raise ToolError(f"HTTP {response.status_code} for {url}")
        body = response.text
        if "html" in response.headers.get("content-type", ""):
            body = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", body)
            body = re.sub(r"(?s)<[^>]+>", " ", body)
            body = re.sub(r"[ \t]+", " ", body)
            body = re.sub(r"\n\s*\n+", "\n\n", body)
        return _short(body.strip())

    # ------------------------------------------------------------------ #

    def _walk(self, start: Path) -> list[Path]:
        if start.is_file():
            return [start]
        files: list[Path] = []
        for current, dirs, names in os.walk(start):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in names:
                files.append(Path(current) / name)
                if len(files) >= 5000:
                    return files
        return files
