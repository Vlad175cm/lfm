"""The Textual application: black and white, square, no decoration."""

from __future__ import annotations

import asyncio
import json
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static, TextArea

from ..config import Config
from ..server import LlamaServer
from ..sessions import Session, sessions_dir
from ..tools import SCHEMAS, ToolError, Toolbox, describe
from . import mono

mono.install()

BG = "#000000"
FG = "#ffffff"
DIM = "#8a8a8a"
FAINT = "#555555"
PANEL = "#0a0a0a"

CSS = f"""
Screen {{
    background: {BG};
    color: {FG};
}}

#header {{
    height: 1;
    padding: 0 2;
    background: {BG};
}}

#brand {{
    width: auto;
    color: {FG};
    text-style: bold;
}}

#info {{
    width: 1fr;
    text-align: right;
    color: {DIM};
}}

#log {{
    width: 1fr;
    padding: 1 2 0 2;
    background: {BG};
    scrollbar-size-vertical: 0;
    scrollbar-size-horizontal: 0;
}}

#composer {{
    height: auto;
    padding: 1 2 1 2;
    background: {BG};
}}

#prompt-mark {{
    width: 2;
    height: 1;
    color: {DIM};
}}

#prompt {{
    height: 1;
    border: none;
    background: {BG};
    color: {FG};
    padding: 0;
}}

#prompt:focus {{
    border: none;
    background: {BG};
}}

#prompt .text-area--placeholder {{
    color: {FAINT};
}}

#prompt .text-area--cursor {{
    background: {FG};
    color: {BG};
}}

.user-msg {{
    margin: 1 0 0 0;
    color: {FG};
}}

.assistant-msg {{
    height: auto;
    margin: 1 0 0 0;
}}

.gutter {{
    width: 2;
    color: {FG};
}}

.msg-body {{
    width: 1fr;
    height: auto;
}}

.assistant-msg .answer {{
    background: transparent;
    padding: 0;
    margin: 0;
    height: auto;
}}

.assistant-msg .meta {{
    height: auto;
    color: {FAINT};
    margin-top: 1;
}}

.copy {{
    height: auto;
    width: auto;
    color: {DIM};
    margin-top: 1;
}}

.copy:hover {{
    color: {FG};
}}

.note {{
    height: auto;
    margin-top: 1;
    color: {DIM};
}}

.note.error {{
    color: {FG};
    text-style: bold;
}}

.tool-step {{
    height: auto;
    margin-top: 1;
    padding-left: 2;
}}

#approval {{
    width: 72;
    height: auto;
    background: {BG};
    border: solid #333333;
    padding: 1 2;
}}

Approval {{
    align: center middle;
    background: #000000 60%;
}}

#approval-title {{
    text-style: bold;
}}

#approval-text {{
    margin-top: 1;
    color: {DIM};
}}

#approval-hint {{
    margin-top: 1;
    color: {FAINT};
}}

Markdown {{
    background: transparent;
    padding: 0;
    margin: 0;
}}

MarkdownBlock > .code_inline {{
    background: #1a1a1a;
    color: {FG};
}}

MarkdownFence {{
    background: {PANEL};
    color: {FG};
    border-left: solid #333333;
    padding-left: 1;
}}

MarkdownFence > Label {{
    background: {PANEL};
}}
"""


class ChatLog(VerticalScroll):
    """Scrolling transcript: shift+up/down only, the mouse wheel is ignored."""

    def __init__(self, **kwargs) -> None:
        super().__init__(id="log", **kwargs)
        self.follow = True

    def watch_scroll_y(self, old: float, new: float) -> None:
        self.follow = new >= self.max_scroll_y - 1

    def _on_mouse_scroll_up(self, event) -> None:
        event.stop()

    def _on_mouse_scroll_down(self, event) -> None:
        event.stop()

    def stick(self) -> None:
        if self.follow:
            self.scroll_end(animate=False, immediate=True)


class Prompt(TextArea):
    """Multi-line composer: grows with the text, Enter sends."""

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    MAX_LINES = 8

    def __init__(self, **kwargs) -> None:
        super().__init__(
            compact=True,
            soft_wrap=True,
            show_line_numbers=False,
            highlight_cursor_line=False,
            tab_behavior="focus",
            **kwargs,
        )

    def on_mount(self) -> None:
        self._fit()

    def on_resize(self) -> None:
        self._fit()

    def on_text_area_changed(self, event) -> None:
        self._fit()

    def _fit(self) -> None:
        height = max(1, min(self.MAX_LINES, self.wrapped_document.height))
        if self.styles.height != height:
            self.styles.height = height

    async def _on_key(self, event) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self.text))
            return
        if event.key in ("shift+enter", "alt+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if event.key in ("shift+up", "shift+down"):
            # TextArea would select text by lines; the chat needs these keys.
            event.prevent_default()
            event.stop()
            if event.key == "shift+up":
                self.app.action_scroll_up()
            else:
                self.app.action_scroll_down()
            return
        await super()._on_key(event)


class UserMessage(Static):
    def __init__(self, text: str) -> None:
        body = Text()
        body.append("> ", style=FG)
        body.append(text, style=FG)
        super().__init__(body, classes="user-msg", markup=False)


class Note(Static):
    def __init__(self, text: str, error: bool = False) -> None:
        super().__init__(
            Text(text, style=FG if error else DIM),
            classes="note",
            markup=False,
        )


class AssistantMessage(Horizontal):
    def __init__(self) -> None:
        super().__init__(classes="assistant-msg")
        self.answer = ""
        self.thinking = ""
        self.blocks: list[str] = []
        self.started = time.monotonic()
        self.md = Markdown("", classes="answer")
        self.meta_view = Static("", classes="meta")

    def compose(self):
        yield Static("▌", classes="gutter")
        with Vertical(classes="msg-body"):
            yield self.md
            yield self.meta_view

    def on_mount(self) -> None:
        self.meta_view.display = False

    async def show(self, answer: str, streaming: bool = True) -> None:
        self.answer = answer
        suffix = "▌" if streaming else ""
        await self.md.update(answer + suffix)

    async def finish(self, timings: dict, elapsed: float, error: str | None, cancelled: bool) -> None:
        await self.md.update(self.answer)
        self.blocks = [] if error else code_blocks(self.answer)
        if self.blocks:
            body = self.query_one(".msg-body", Vertical)
            body.mount(CopyButton(self.blocks), before=self.meta_view)
        self.meta_view.display = True
        if error:
            self.meta_view.update(Text(error, style=FG))
            return

        bits: list[str] = []
        if cancelled:
            bits.append("stopped")
        tokens = int(timings.get("predicted_n") or 0)
        speed = timings.get("predicted_per_second")
        if tokens:
            bits.append(f"{tokens} tok")
        if speed:
            bits.append(f"{speed:.1f} t/s")
        bits.append(f"{elapsed:.1f}s")
        self.meta_view.update(Text(" · ".join(bits), style=FAINT))


CODE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def code_blocks(text: str) -> list[str]:
    """Every fenced code block in a markdown answer, without the fences."""
    blocks = [block.strip("\n") for block in CODE_RE.findall(text)]
    return [block for block in blocks if block.strip()]


class CopyButton(Static):
    def __init__(self, blocks: list[str]) -> None:
        self.blocks = blocks
        label = "[ copy code ]" if len(blocks) == 1 else f"[ copy code · {len(blocks)} ]"
        self.label = label
        super().__init__(label, classes="copy", markup=False)

    def on_click(self, event) -> None:
        event.stop()
        self.app.copy_code(self.blocks)
        self.update("[ copied ]")
        self.set_timer(1.4, lambda: self.update(self.label))


AGENT_HINT = (
    "Agent mode is on: you have tools and you are expected to use them. "
    "Inspect files and run commands instead of guessing, then answer from "
    "their output. Emit tool calls through the tool interface only — never "
    "write a call as text, a JSON blob, or a code block. Use one or two "
    "calls per turn. When you know the answer, answer briefly."
)


class ToolStep(Static):
    """One executed tool call, with a short preview of its output."""

    def __init__(self, summary: str) -> None:
        body = Text()
        body.append("● ", style=FG)
        body.append(summary, style=DIM)
        super().__init__(body, classes="tool-step", markup=False)
        self.body = body

    def show_result(self, result: str) -> None:
        lines = result.strip().replace("\t", "    ").splitlines() or [""]
        text = "\n".join("  " + line for line in lines[:6])
        if len(lines) > 6:
            text += f"\n  … {len(lines) - 6} more lines"
        self.body.append("\n")
        self.body.append(text, style=FAINT)
        self.update(self.body)


class Approval(ModalScreen[bool]):
    """Yes/no gate in front of anything that writes or runs a command."""

    BINDINGS = [
        Binding("y", "allow", "allow"),
        Binding("n", "deny", "deny"),
        Binding("enter", "allow", "allow", show=False),
        Binding("escape", "deny", "deny", show=False),
    ]

    def __init__(self, summary: str) -> None:
        super().__init__()
        self.summary = summary

    def compose(self):
        with Vertical(id="approval"):
            yield Static("run this?", id="approval-title", markup=False)
            yield Static(self.summary, id="approval-text", markup=False)
            yield Static("[ y ] allow    [ n ] deny", id="approval-hint", markup=False)

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


def _fmt_int(value: int | float) -> str:
    value = int(value)
    if value >= 1000:
        return f"{value / 1000:.1f}k".replace(".0k", "k")
    return str(value)


class LfmApp(App):
    CSS = CSS
    TITLE = "lfm"
    BINDINGS = [
        Binding("ctrl+c", "quit", "quit", priority=True),
        Binding("ctrl+l", "clear", "clear"),
        Binding("escape", "stop", "stop"),
        Binding("ctrl+y", "copy_code", "copy code"),
        Binding("shift+up", "scroll_up", "scroll up", show=False),
        Binding("shift+down", "scroll_down", "scroll down", show=False),
    ]

    def __init__(self, cfg: Config, model: Path, binary: Path) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_path = model
        self.binary = binary
        self.server: LlamaServer | None = None
        self.messages: list[dict] = []
        self.session = Session.new(model=model.stem, system=cfg.system_prompt)
        self.state = "boot"
        self._status_kind = "boot"
        self._status_text = "loading model"
        self._q: queue.Queue = queue.Queue()
        self._active = False
        self._widget: AssistantMessage | None = None
        self._buf = ""
        self._think = ""
        self._timings: dict = {}
        self._error: str | None = None
        self._cancel = threading.Event()
        self._t0 = time.monotonic()
        self._spin = 0
        self._last_timings: dict = {}
        self.agent = cfg.agent
        self.toolbox = Toolbox(Path.cwd())
        self._calls: list[dict] = []
        self._steps = 0

    # ------------------------------------------------------------------ #

    def compose(self):
        with Horizontal(id="header"):
            yield Static("lfm", id="brand")
            yield Static("", id="info")
        yield ChatLog()
        with Horizontal(id="composer"):
            yield Static("❯", id="prompt-mark")
            yield Prompt(placeholder="message", id="prompt")

    def on_mount(self) -> None:
        self.chatlog = self.query_one(ChatLog)
        self.prompt = self.query_one("#prompt", Prompt)
        self.info = self.query_one("#info", Static)

        self.prompt.disabled = True
        self._render_info()
        self.set_interval(0.05, self._drain)
        self.set_interval(0.2, self._spinner)
        self._boot()

    # ------------------------------------------------------------------ #

    @work(thread=True, exclusive=True, group="boot")
    def _boot(self) -> None:
        self._t0 = time.monotonic()
        try:
            self.server = LlamaServer(self.binary, self.model_path, self.cfg)
            self.server.start(timeout=300)
            self.call_from_thread(self._server_ready)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._boot_failed, str(exc))

    def _server_ready(self) -> None:
        self.state = "idle"
        self.prompt.disabled = False
        self.prompt.focus()
        self._set_status("idle", "ready")
        self._render_info()

    def _boot_failed(self, message: str) -> None:
        self.state = "error"
        self._set_status("error", "error")
        self._render_info()
        self.note("could not start llama.cpp:\n" + message, error=True)

    def _spinner(self) -> None:
        if self.state != "boot":
            return
        frames = "◐◓◑◒"
        self._spin = (self._spin + 1) % len(frames)
        elapsed = int(time.monotonic() - self._t0)
        self._set_status("boot", f"{frames[self._spin]} loading {elapsed}s")

    # ------------------------------------------------------------------ #

    def _set_status(self, kind: str, text: str) -> None:
        self._status_kind = kind
        self._status_text = text
        self._render_info()

    def _render_info(self) -> None:
        timings = self._last_timings
        used = int(timings.get("prompt_n") or 0) + int(timings.get("predicted_n") or 0)
        total = max(1, self.cfg.ctx)
        speed = timings.get("predicted_per_second")

        line = Text()
        glyph = {
            "boot": "◌",
            "idle": "●",
            "thinking": "✻",
            "error": "!",
        }.get(self._status_kind, "●")
        line.append(glyph, style=FG)
        line.append(" " + self._status_text, style=DIM)
        if self.agent:
            line.append("  ·  agent", style=DIM)
        line.append("  ·  ", style=FAINT)
        line.append(f"ctx {_fmt_int(used)}/{_fmt_int(total)}", style=DIM)
        line.append("  ·  ", style=FAINT)
        line.append(f"{speed:.1f} tok/s" if speed else "— tok/s", style=DIM)
        self.info.update(line)

    # ------------------------------------------------------------------ #

    async def on_prompt_submitted(self, event: Prompt.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if text.startswith("/"):
            self.prompt.text = ""
            self.command(text)
            return
        if self.state != "idle" or self._active:
            return
        self.prompt.text = ""
        await self.send(text)

    async def send(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self.session.add("user", text)
        self.session.save()
        await self.chatlog.mount(UserMessage(text))
        self.chatlog.follow = True
        self.chatlog.scroll_end(animate=False, immediate=True)
        self.call_after_refresh(self.chatlog.stick)

        widget = AssistantMessage()
        self._widget = widget
        self._active = True
        self.state = "stream"
        self._buf = ""
        self._think = ""
        self._timings = {}
        self._error = None
        self._calls = []
        self._steps = 0
        self._cancel.clear()
        self._t0 = time.monotonic()
        self._set_status("thinking", "thinking")
        await self.chatlog.mount(widget)
        self.call_after_refresh(self.chatlog.stick)
        self._stream(self._build_messages(), self._tools())

    def _tools(self) -> list[dict] | None:
        return SCHEMAS if self.agent else None

    def _build_messages(self) -> list[dict]:
        payload: list[dict] = []
        system = self.cfg.system_prompt.strip()
        if self.agent:
            system = (system + "\n\n" if system else "") + AGENT_HINT
            system += f"\nAll paths are inside {self.toolbox.root}."
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(self.messages)
        return payload

    @work(thread=True, exclusive=True, group="stream")
    def _stream(self, messages: list[dict], tools: list[dict] | None = None) -> None:
        try:
            assert self.server is not None
            for event in self.server.chat_stream(messages, self._cancel, tools):
                self._q.put(event)
        except Exception as exc:  # noqa: BLE001
            self._q.put({"type": "error", "message": str(exc)})
        finally:
            self._q.put({"type": "eof"})

    async def _drain(self) -> None:
        if not self._active:
            return
        changed = False
        finished = False
        while True:
            try:
                event = self._q.get_nowait()
            except queue.Empty:
                break
            kind = event.get("type")
            if kind == "content":
                self._buf += event.get("text", "")
                changed = True
            elif kind == "reasoning":
                self._think += event.get("text", "")
            elif kind == "tool_calls":
                self._calls = event.get("calls") or []
            elif kind == "done":
                self._timings = event.get("timings") or {}
            elif kind == "error":
                self._error = event.get("message") or "unknown error"
                finished = True
            elif kind == "eof":
                finished = True

        if changed and self._widget is not None:
            await self._widget.show(self._buf)
            self.chatlog.stick()
        if finished:
            await self._finish()

    async def _finish(self) -> None:
        self._active = False
        widget = self._widget
        self._widget = None
        elapsed = time.monotonic() - self._t0
        cancelled = self._cancel.is_set()
        calls = self._calls
        self._calls = []

        if widget is not None and widget.is_mounted:
            widget.thinking = self._think
            if calls and not self._buf.strip() and not self._error and not cancelled:
                widget.display = False
            else:
                await widget.finish(self._timings, elapsed, self._error, cancelled)

        if widget is not None and (self._buf.strip() or calls):
            message: dict = {"role": "assistant", "content": self._buf}
            extra: dict = {}
            if calls:
                message["tool_calls"] = calls
                extra["tool_calls"] = calls
            self.messages.append(message)
            self.session.add("assistant", self._buf, **extra)
            self.session.save()
        if self._timings:
            self._last_timings = self._timings

        if not self._buf.strip() and not calls and self._steps and not cancelled and not self._error:
            self.note("the model returned an empty answer — send it again")

        if calls and not cancelled and not self._error:
            if self.agent:
                if self._steps < max(1, int(self.cfg.agent_steps)):
                    self._steps += 1
                    self._set_status("thinking", f"step {self._steps}")
                    await self._run_tools(calls)
                    return
                self.note(f"stopped after {self._steps} tool steps — /agent off or ask again")
            else:
                self.note("the model asked for a tool, but agent mode is off — /agent to allow it")
        self._steps = 0

        self.state = "idle"
        if self._error:
            self._set_status("error", "error")
        else:
            self._set_status("idle", "ready")
        self.prompt.disabled = False
        self.prompt.focus()
        self.call_after_refresh(self.chatlog.stick)

    # ------------------------------------------------------------------ #

    async def _run_tools(self, calls: list[dict]) -> None:
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name") or "?"
            raw = function.get("arguments") or "{}"
            try:
                parsed = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                parsed = None
            bad_args = not isinstance(parsed, dict)
            args = parsed if isinstance(parsed, dict) else {}

            summary = describe(name, args)
            step = ToolStep(summary)
            await self.chatlog.mount(step)
            self.call_after_refresh(self.chatlog.stick)

            if bad_args:
                result = f"error: arguments must be a JSON object, got {raw!r}"
                step.show_result("bad arguments")
            elif self.toolbox.dangerous(name) and not await self._approve(summary):
                result = (
                    "DENIED by the user: nothing ran and nothing changed. "
                    "Tell the user the action was not done and ask what to do instead."
                )
                step.show_result("denied — nothing was executed")
            else:
                try:
                    result = await asyncio.to_thread(self.toolbox.run, name, args)
                except ToolError as exc:
                    result = f"error: {exc}"
                except Exception as exc:  # noqa: BLE001
                    result = f"error: {type(exc).__name__}: {exc}"
                step.show_result(result or "(no output)")

            self.messages.append(
                {"role": "tool", "tool_call_id": call.get("id", ""), "name": name, "content": result}
            )
            self.session.add("tool", result, tool_call_id=call.get("id", ""), name=name)
            self.call_after_refresh(self.chatlog.stick)

        self.session.save()

        widget = AssistantMessage()
        self._widget = widget
        self._active = True
        self.state = "stream"
        self._buf = ""
        self._think = ""
        self._timings = {}
        self._error = None
        self._cancel.clear()
        self._t0 = time.monotonic()
        await self.chatlog.mount(widget)
        self.call_after_refresh(self.chatlog.stick)
        self._stream(self._build_messages(), self._tools())

    async def _approve(self, summary: str) -> bool:
        # push_screen_wait() needs a worker; this runs from a timer callback.
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        def answer(result: bool | None) -> None:
            if not future.done():
                future.set_result(bool(result))

        try:
            self.push_screen(Approval(summary), answer)
        except Exception:  # noqa: BLE001
            return False
        return await future

    # ------------------------------------------------------------------ #

    def action_stop(self) -> None:
        if self._active:
            self._cancel.set()
            self._set_status("thinking", "stopping")

    def action_scroll_up(self) -> None:
        if len(self.screen_stack) > 1:
            return
        self.chatlog.follow = False
        self.chatlog.scroll_relative(y=-3, animate=False, immediate=True)

    def action_scroll_down(self) -> None:
        if len(self.screen_stack) > 1:
            return
        self.chatlog.scroll_relative(y=3, animate=False, immediate=True)
        if self.chatlog.is_vertical_scroll_end:
            self.chatlog.follow = True

    def action_copy_code(self) -> None:
        for widget in reversed(self.chatlog.query(AssistantMessage)):
            if widget.blocks:
                self.copy_code(widget.blocks)
                return
        self._flash("no code to copy")

    def copy_code(self, blocks: list[str]) -> None:
        text = "\n\n".join(blocks)
        if self._clipboard_write(text):
            self._flash(f"copied {len(text)} chars")
        else:
            self._flash("could not copy")

    def _clipboard_write(self, text: str) -> bool:
        for command in (
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
            ["pbcopy"],
        ):
            if not shutil.which(command[0]):
                continue
            try:
                subprocess.run(command, input=text, text=True, check=True, timeout=5)
                return True
            except (OSError, subprocess.SubprocessError):
                continue
        try:
            self.copy_to_clipboard(text)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _flash(self, text: str) -> None:
        self._set_status("idle", text)
        self.set_timer(1.6, self._restore_status)

    def _restore_status(self) -> None:
        if self.state == "idle" and not self._active:
            self._set_status("idle", "ready")

    def action_clear(self) -> None:
        if self._active:
            self._cancel.set()
            self._widget = None
        self.messages.clear()
        self._calls = []
        self._steps = 0
        self.session = Session.new(model=self.model_path.stem, system=self.cfg.system_prompt)
        self._last_timings = {}
        self.chatlog.remove_children()
        self._render_info()
        if self.state == "idle":
            self._set_status("idle", "ready")

    def action_quit(self) -> None:
        self._cancel.set()
        if self.server is not None:
            self.server.stop()
        self.exit()

    # ------------------------------------------------------------------ #

    def note(self, text: str, error: bool = False) -> None:
        self.chatlog.mount(Note(text, error=error))
        self.call_after_refresh(self.chatlog.stick)

    def command(self, raw: str) -> None:
        name, _, arg = raw[1:].partition(" ")
        name = name.strip().lower()
        arg = arg.strip()
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            self.note(f"unknown command /{name} — try /help")
            return
        handler(arg)

    def cmd_help(self, _: str = "") -> None:
        self.note(
            "/help            this list\n"
            "/agent [on|off]  toggle tools (read auto, write/shell ask)\n"
            "/clear           wipe the conversation\n"
            "/model           which model is loaded\n"
            "/temp <0..2>     sampling temperature\n"
            "/system <text>   replace the system prompt\n"
            "/stats           tokens, context, speed\n"
            "/chats           where transcripts are saved\n"
            "/quit            exit\n"
            "\n"
            "enter            send\n"
            "shift+enter      new line\n"
            "shift+up/down    scroll the transcript\n"
            "esc              stop generating\n"
            "ctrl+l           clear"
        )

    def cmd_agent(self, arg: str = "") -> None:
        choice = arg.strip().lower()
        if choice in ("on", "1", "true"):
            self.agent = True
        elif choice in ("off", "0", "false"):
            self.agent = False
        elif not choice:
            self.agent = not self.agent
        else:
            self.note("usage: /agent [on|off]")
            return
        self.cfg.agent = self.agent
        self._render_info()
        if self.agent:
            self.note(
                "agent mode on\n"
                f"root: {self.toolbox.root}\n"
                "auto: list_dir, read_file, grep, fetch_url\n"
                "asks: write_file, edit, run_shell\n"
                f"step limit: {self.cfg.agent_steps} · /agent off to stop"
            )
        else:
            self.note("agent mode off")

    def cmd_clear(self, _: str = "") -> None:
        self.action_clear()

    def cmd_model(self, _: str = "") -> None:
        size = self.model_path.stat().st_size / 1e9 if self.model_path.exists() else 0
        self.note(f"{self.model_path}\n{size:.2f} GB · ctx {self.cfg.ctx} · temp {self.cfg.temperature}")

    def cmd_temp(self, arg: str = "") -> None:
        try:
            value = float(arg.replace(",", "."))
        except ValueError:
            self.note(f"temperature is {self.cfg.temperature}; usage: /temp 0.9")
            return
        self.cfg.temperature = min(2.0, max(0.0, value))
        self.note(f"temperature → {self.cfg.temperature}")

    def cmd_system(self, arg: str = "") -> None:
        if not arg:
            self.note("system prompt:\n" + (self.cfg.system_prompt or "(empty)"))
            return
        self.cfg.system_prompt = arg
        self.session.system = arg
        self.session.save()
        self.note("system prompt updated")

    def cmd_stats(self, _: str = "") -> None:
        t = self._last_timings
        bits = [
            f"context: {_fmt_int(int(t.get('prompt_n') or 0) + int(t.get('predicted_n') or 0))} / {_fmt_int(self.cfg.ctx)}",
            f"tokens: {int(t.get('predicted_n') or 0)}",
            f"speed: {t.get('predicted_per_second', 0):.1f} tok/s" if t else "speed: —",
            f"messages: {len(self.messages)}",
        ]
        self.note("\n".join(bits))

    def cmd_chats(self, _: str = "") -> None:
        path = self.session.save()
        self.note(f"transcripts: {sessions_dir()}\nthis chat: {path}")

    def cmd_quit(self, _: str = "") -> None:
        self.action_quit()
