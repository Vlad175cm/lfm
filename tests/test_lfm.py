from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lfm.config import Config, find_model
from lfm.prompt import load_system
from lfm.server import LlamaServer, free_port
from lfm.sessions import Session, sessions_dir
from lfm.tools import ToolError, Toolbox
from lfm.ui.app import (
    AGENT_HINT,
    Approval,
    AssistantMessage,
    CopyButton,
    LfmApp,
    Note,
    ToolStep,
    code_blocks,
)


class ConfigTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            cfg = Config()
            cfg.ctx = 4096
            cfg.temperature = 0.3
            cfg.thinking = True
            cfg.system_prompt = 'a string "with quotes"\nand a newline'
            cfg.save(path)

            loaded = Config.load(path)
            self.assertEqual(loaded.ctx, 4096)
            self.assertAlmostEqual(loaded.temperature, 0.3)
            self.assertTrue(loaded.thinking)
            self.assertEqual(loaded.system_prompt, cfg.system_prompt)

    def test_missing_config_is_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(Config.load(Path(tmp) / "nope.toml").ctx, 8192)

    def test_find_model_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.gguf"
            path.write_bytes(b"gguf")
            self.assertEqual(find_model(str(path)), path)
            self.assertIsNone(find_model(str(Path(tmp) / "missing.gguf")))


class ServerTest(unittest.TestCase):
    def test_free_port(self):
        port = free_port()
        self.assertGreater(port, 1024)
        self.assertLess(port, 65536)

    def test_command_flags(self):
        cfg = Config()
        server = LlamaServer(Path("/bin/true"), Path("/tmp/m.gguf"), cfg)
        cmd = server.command()
        self.assertIn("--jinja", cmd)
        self.assertIn("--reasoning", cmd)
        self.assertIn("off", cmd)
        self.assertNotIn("deepseek", cmd)

        cfg.thinking = True
        server = LlamaServer(Path("/bin/true"), Path("/tmp/m.gguf"), cfg)
        cmd = server.command()
        self.assertIn("deepseek", cmd)
        self.assertIn("on", cmd)


class FakeServer:
    answer = "Hello, world"
    script: list[list[dict]] = []

    def __init__(self, *args, **kwargs):
        self.stopped = False

    def start(self, timeout=0, on_line=None):
        pass

    def stop(self):
        self.stopped = True

    def chat_stream(self, messages, cancel=None, tools=None):
        if FakeServer.script:
            for event in FakeServer.script.pop(0):
                yield event
            return
        for index in range(0, len(self.answer), 5):
            yield {"type": "content", "text": self.answer[index:index + 5]}
        yield {"type": "done", "timings": {"predicted_n": 3, "predicted_per_second": 12.5}}


def tool_turn(name: str, arguments: dict, call_id: str = "call_1") -> list[dict]:
    return [
        {
            "type": "tool_calls",
            "calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        {"type": "done", "timings": {}},
    ]


class ToolboxTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.box = Toolbox(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sandbox_refuses_to_escape(self):
        with self.assertRaises(ToolError):
            self.box.run("read_file", {"path": "../escape.txt"})
        with self.assertRaises(ToolError):
            self.box.run("list_dir", {"path": "/etc"})

    def test_write_read_edit(self):
        self.box.run("write_file", {"path": "sub/a.txt", "content": "one\ntwo\n"})
        self.assertEqual((self.root / "sub/a.txt").read_text(), "one\ntwo\n")

        read = self.box.run("read_file", {"path": "sub/a.txt"})
        self.assertIn("1\tone", read)
        self.assertIn("2\ttwo", read)

        self.box.run("edit", {"path": "sub/a.txt", "old_string": "two", "new_string": "three"})
        self.assertEqual((self.root / "sub/a.txt").read_text(), "one\nthree\n")

    def test_edit_requires_a_unique_match(self):
        (self.root / "x.txt").write_text("a a")
        with self.assertRaises(ToolError):
            self.box.run("edit", {"path": "x.txt", "old_string": "a", "new_string": "b"})
        with self.assertRaises(ToolError):
            self.box.run("edit", {"path": "x.txt", "old_string": "zzz", "new_string": "b"})

    def test_grep_finds_lines(self):
        (self.root / "a.py").write_text("print('needle')\n")
        (self.root / "b.txt").write_text("needle\n")
        hits = self.box.run("grep", {"pattern": "needle", "glob": "*.py"})
        self.assertIn("a.py:1", hits)
        self.assertNotIn("b.txt", hits)

    def test_run_shell_uses_the_root(self):
        (self.root / "here.txt").write_text("x")
        out = self.box.run("run_shell", {"command": "ls"})
        self.assertIn("here.txt", out)
        self.assertIn("exit 0", out)

    def test_dangerous_tools(self):
        self.assertTrue(self.box.dangerous("run_shell"))
        self.assertTrue(self.box.dangerous("write_file"))
        self.assertFalse(self.box.dangerous("read_file"))
        self.assertFalse(self.box.dangerous("grep"))


class PromptFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "system.md"
        self.env = patch.dict(os.environ, {"LFM_SYSTEM_FILE": str(self.path)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_file_wins(self):
        self.path.write_text("you are a cat", encoding="utf-8")
        self.assertEqual(load_system("fallback"), ("you are a cat", self.path))

    def test_empty_file_falls_back(self):
        self.path.write_text("   \n\n", encoding="utf-8")
        self.assertEqual(load_system("fallback"), ("fallback", None))

    def test_missing_file_falls_back(self):
        self.assertEqual(load_system("fallback"), ("fallback", None))


class CodeTest(unittest.TestCase):
    def test_extracts_fenced_blocks(self):
        text = "intro\n```python\nx = 1\n```\nmid\n```\nplain\n```\n"
        self.assertEqual(code_blocks(text), ["x = 1", "plain"])

    def test_ignores_inline_and_unclosed(self):
        self.assertEqual(code_blocks("no code here"), [])
        self.assertEqual(code_blocks("```py\nx = 1"), [])
        self.assertEqual(code_blocks("`inline` only"), [])


class AppTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old_data = os.environ.get("XDG_DATA_HOME")
        os.environ["XDG_DATA_HOME"] = self.tmp.name
        self._prompt_env = patch.dict(
            os.environ, {"LFM_SYSTEM_FILE": str(Path(self.tmp.name) / "system.md")}
        )
        self._prompt_env.start()
        FakeServer.answer = "Hello, world"
        FakeServer.script = []
        self.patcher = patch("lfm.ui.app.LlamaServer", FakeServer)
        self.patcher.start()
        self.app = LfmApp(Config(), Path("/tmp/model.gguf"), Path("/bin/true"))
        self.ctx = self.app.run_test(size=(110, 36))
        self.pilot = await self.ctx.__aenter__()
        for _ in range(200):
            if self.app.state == "idle":
                break
            await self.pilot.pause(0.02)
        self.app.prompt.focus()
        await self.pilot.pause(0.05)

    async def asyncTearDown(self):
        self.app.exit()
        await self.ctx.__aexit__(None, None, None)
        self.patcher.stop()
        self._prompt_env.stop()
        if self._old_data is None:
            os.environ.pop("XDG_DATA_HOME", None)
        else:
            os.environ["XDG_DATA_HOME"] = self._old_data
        self.tmp.cleanup()

    async def send(self, text: str):
        self.app.prompt.text = text
        await self.pilot.press("enter")
        for _ in range(500):
            if not self.app._active and self.app.state == "idle":
                break
            await self.pilot.pause(0.02)
        await self.pilot.pause(0.05)

    async def wait_for(self, predicate, timeout: float = 5.0):
        steps = int(timeout / 0.02)
        for _ in range(steps):
            if predicate():
                return
            await self.pilot.pause(0.02)
        self.fail(f"condition was never true; messages={self.app.messages!r} state={self.app.state}")

    async def test_boot_ready(self):
        self.assertEqual(self.app.state, "idle")
        self.assertFalse(self.app.prompt.disabled)

    async def test_chat_streams_into_history(self):
        await self.send("hello")
        self.assertEqual(len(self.app.messages), 2)
        self.assertEqual(self.app.messages[0], {"role": "user", "content": "hello"})
        self.assertEqual(self.app.messages[1]["content"], "Hello, world")

        widget = self.app.query_one(AssistantMessage)
        self.assertEqual(widget.answer, "Hello, world")

    async def test_help_and_clear(self):
        await self.send("/help")
        notes = self.app.query(".note")
        self.assertTrue(any("/temp" in str(n.render()) for n in notes))

        await self.send("hello")
        self.assertEqual(len(self.app.messages), 2)
        await self.send("/clear")
        await self.pilot.pause(0.1)
        self.assertEqual(self.app.messages, [])

    async def test_temp_command(self):
        await self.send("/temp 0.5")
        self.assertAlmostEqual(self.app.cfg.temperature, 0.5)

    async def test_transcript_is_saved(self):
        await self.send("hello")
        files = list(sessions_dir().glob("*.json"))
        self.assertEqual(len(files), 1, files)
        data = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual([m["role"] for m in data["messages"]], ["user", "assistant"])
        self.assertEqual(data["messages"][1]["content"], "Hello, world")
        self.assertEqual(data["title"], "hello")

    async def test_clear_starts_a_new_transcript(self):
        await self.send("hello")
        first = self.app.session.path
        await self.send("/clear")
        await self.send("again")
        await self.pilot.pause(0.2)
        files = sorted(sessions_dir().glob("*.json"))
        self.assertEqual(len(files), 2, files)
        self.assertNotEqual(self.app.session.path, first)

    async def test_system_command_refuses_to_edit(self):
        before = self.app.cfg.system_prompt
        await self.send("/system you are a cat")
        self.assertEqual(self.app.cfg.system_prompt, before)
        notes = [str(n.render()) for n in self.app.query(".note")]
        self.assertTrue(any("not edited from the chat" in n for n in notes), notes)
        await self.send("hello")

    async def test_system_command_shows_the_file(self):
        await self.send("/system")
        notes = [str(n.render()) for n in self.app.query(".note")]
        self.assertTrue(any("system.md" in n for n in notes), notes)

    async def test_system_reloads_from_the_file(self):
        path = Path(self.tmp.name) / "system.md"
        path.write_text("you are a cat", encoding="utf-8")
        self.app.reload_system()
        self.assertEqual(self.app.cfg.system_prompt, "you are a cat")
        self.assertEqual(self.app.session.system, "you are a cat")
        self.assertEqual(self.app.system_path, path)

    async def test_agent_hint_is_subordinate_to_the_persona(self):
        self.app.agent = True
        self.app.cfg.system_prompt = "you are a cat"
        system = self.app._build_messages()[0]["content"]
        self.assertTrue(system.startswith(AGENT_HINT))
        self.assertTrue(system.endswith("you are a cat"))

    async def test_copy_button_appears_for_code(self):
        FakeServer.answer = "try this:\n```python\nx = 1\n```\n"
        await self.send("code please")
        await self.wait_for(lambda: len(self.app.query(CopyButton)) == 1)

        buttons = self.app.query(CopyButton)
        self.assertEqual(len(buttons), 1)
        button = buttons.first()
        self.assertEqual(button.blocks, ["x = 1"])

        copied: list[str] = []
        self.app._clipboard_write = lambda text: copied.append(text) or True
        button.on_click(type("Event", (), {"stop": lambda self: None})())
        self.assertEqual(copied[-1], "x = 1")

    async def fill_log(self, lines: int = 120):
        self.app.chatlog.follow = True
        await self.app.chatlog.mount(Note("\n".join(f"line {i}" for i in range(lines))))
        await self.pilot.pause(0.2)
        self.app.chatlog.scroll_end(animate=False, immediate=True)
        await self.pilot.pause(0.1)

    async def test_shift_arrows_scroll_with_text_in_the_prompt(self):
        await self.fill_log()
        self.app.prompt.text = "one\ntwo\nthree"
        await self.pilot.pause(0.1)
        start = self.app.chatlog.scroll_y

        await self.pilot.press("shift+up")
        await self.pilot.pause(0.1)

        self.assertLess(self.app.chatlog.scroll_y, start)
        self.assertEqual(self.app.prompt.text, "one\ntwo\nthree")
        self.assertEqual(self.app.prompt.selected_text, "")

    async def test_shift_down_returns_to_the_bottom(self):
        await self.fill_log()
        for _ in range(5):
            await self.pilot.press("shift+up")
        await self.pilot.pause(0.1)
        self.assertFalse(self.app.chatlog.follow)

        for _ in range(60):
            await self.pilot.press("shift+down")
            if self.app.chatlog.follow:
                break
            await self.pilot.pause(0.01)

        self.assertTrue(self.app.chatlog.follow)
        self.assertTrue(self.app.chatlog.is_vertical_scroll_end)

    async def test_plain_arrows_and_page_keys_do_not_scroll(self):
        await self.fill_log()
        start = self.app.chatlog.scroll_y
        await self.pilot.press("up")
        await self.pilot.press("down")
        await self.pilot.press("pageup")
        await self.pilot.press("pagedown")
        await self.pilot.pause(0.1)
        self.assertEqual(self.app.chatlog.scroll_y, start)

    async def test_sending_returns_the_log_to_the_bottom(self):
        await self.fill_log()
        for _ in range(4):
            await self.pilot.press("shift+up")
        await self.pilot.pause(0.1)
        self.assertFalse(self.app.chatlog.follow)

        await self.send("hello")

        self.assertTrue(self.app.chatlog.follow)
        self.assertTrue(self.app.chatlog.is_vertical_scroll_end)

    async def test_shift_arrows_are_ignored_while_a_dialog_is_open(self):
        await self.fill_log()
        start = self.app.chatlog.scroll_y
        self.app.push_screen(Approval("write_file x"))
        await self.pilot.pause(0.1)

        await self.pilot.press("shift+up")
        await self.pilot.pause(0.1)

        self.assertEqual(self.app.chatlog.scroll_y, start)
        await self.pilot.press("n")
        await self.pilot.pause(0.1)

    async def test_copy_shortcut_takes_the_last_code(self):
        FakeServer.answer = "```\nprint(1)\n```\n\nmore\n\n```\nprint(2)\n```\n"
        await self.send("code")
        await self.wait_for(
            lambda: any(w.blocks for w in self.app.query(AssistantMessage))
        )

        copied: list[str] = []
        self.app._clipboard_write = lambda text: copied.append(text) or True
        self.app.action_copy_code()
        self.assertEqual(copied[-1], "print(1)\n\nprint(2)")

    # ------------------------------------------------------------------ #
    # agent mode

    async def test_agent_command_toggles_tools(self):
        self.assertFalse(self.app.agent)
        await self.send("/agent")
        self.assertTrue(self.app.agent)
        self.assertTrue(self.app.cfg.agent)
        await self.send("/agent off")
        self.assertFalse(self.app.agent)

    async def test_agent_reads_a_file_without_asking(self):
        self.app.toolbox = Toolbox(self.tmp.name)
        (Path(self.tmp.name) / "note.txt").write_text("the secret is 42", encoding="utf-8")
        FakeServer.script = [
            tool_turn("read_file", {"path": "note.txt"}),
            [{"type": "content", "text": "It is 42."}, {"type": "done", "timings": {}}],
        ]
        await self.send("/agent")
        await self.send("what is in note.txt?")

        roles = [m["role"] for m in self.app.messages]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
        tool_message = self.app.messages[2]
        self.assertEqual(tool_message["name"], "read_file")
        self.assertIn("the secret is 42", tool_message["content"])
        self.assertEqual(self.app.messages[-1]["content"], "It is 42.")
        self.assertEqual(len(self.app.query(ToolStep)), 1)

    async def test_agent_denied_write_changes_nothing(self):
        self.app.toolbox = Toolbox(self.tmp.name)
        asked: list[str] = []

        async def deny(summary: str) -> bool:
            asked.append(summary)
            return False

        self.app._approve = deny
        FakeServer.script = [
            tool_turn("write_file", {"path": "new.txt", "content": "nope"}),
            [{"type": "content", "text": "ok, skipped"}, {"type": "done", "timings": {}}],
        ]
        await self.send("/agent")
        await self.send("create new.txt")

        self.assertEqual(len(asked), 1)
        self.assertIn("write_file", asked[0])
        self.assertFalse((Path(self.tmp.name) / "new.txt").exists())
        self.assertIn("denied", self.app.messages[2]["content"].lower())

    async def test_agent_allowed_write_creates_the_file(self):
        self.app.toolbox = Toolbox(self.tmp.name)

        async def allow(summary: str) -> bool:
            return True

        self.app._approve = allow
        FakeServer.script = [
            tool_turn("write_file", {"path": "new.txt", "content": "written"}),
            [{"type": "content", "text": "done"}, {"type": "done", "timings": {}}],
        ]
        await self.send("/agent")
        await self.send("create new.txt")

        self.assertEqual((Path(self.tmp.name) / "new.txt").read_text(), "written")

    async def test_agent_modal_approves_and_writes(self):
        self.app.toolbox = Toolbox(self.tmp.name)
        FakeServer.script = [
            tool_turn("write_file", {"path": "ok.txt", "content": "yes"}),
            [{"type": "content", "text": "done"}, {"type": "done", "timings": {}}],
        ]
        await self.send("/agent")
        self.app.prompt.text = "create ok.txt"
        await self.pilot.press("enter")

        for _ in range(300):
            if isinstance(self.app.screen, Approval):
                break
            await self.pilot.pause(0.02)
        self.assertIsInstance(self.app.screen, Approval, "no approval dialog appeared")

        await self.pilot.press("y")
        for _ in range(500):
            if not self.app._active and self.app.state == "idle":
                break
            await self.pilot.pause(0.02)

        self.assertTrue((Path(self.tmp.name) / "ok.txt").exists())
        self.assertIn("wrote", self.app.messages[2]["content"])

    async def test_agent_modal_denies_and_writes_nothing(self):
        self.app.toolbox = Toolbox(self.tmp.name)
        FakeServer.script = [
            tool_turn("write_file", {"path": "no.txt", "content": "no"}),
            [{"type": "content", "text": "fine"}, {"type": "done", "timings": {}}],
        ]
        await self.send("/agent")
        self.app.prompt.text = "create no.txt"
        await self.pilot.press("enter")

        for _ in range(300):
            if isinstance(self.app.screen, Approval):
                break
            await self.pilot.pause(0.02)
        await self.pilot.press("n")
        for _ in range(500):
            if not self.app._active and self.app.state == "idle":
                break
            await self.pilot.pause(0.02)

        self.assertFalse((Path(self.tmp.name) / "no.txt").exists())
        self.assertIn("denied", self.app.messages[2]["content"].lower())

    async def test_agent_stops_at_the_step_limit(self):
        self.app.cfg.agent_steps = 2
        self.app.toolbox = Toolbox(self.tmp.name)
        FakeServer.script = [
            tool_turn("list_dir", {"path": "."}),
            tool_turn("list_dir", {"path": "."}),
            tool_turn("list_dir", {"path": "."}),
        ]
        await self.send("/agent")
        await self.send("look around")

        self.assertEqual(len(self.app.query(ToolStep)), 2)
        self.assertEqual([m["role"] for m in self.app.messages if m["role"] == "tool"], ["tool", "tool"])
        self.assertEqual(self.app._steps, 0)
        self.assertEqual(self.app.state, "idle")

    async def test_agent_tool_without_agent_mode_is_refused(self):
        FakeServer.script = [tool_turn("list_dir", {"path": "."})]
        await self.send("look around")
        self.assertEqual(self.app.messages[-1]["role"], "assistant")
        self.assertEqual(len(self.app.query(ToolStep)), 0)

    async def test_approval_modal_keys(self):
        accepted: list[bool] = []
        self.app.push_screen(Approval("write_file x"), accepted.append)
        await self.pilot.pause(0.1)
        await self.pilot.press("n")
        await self.pilot.pause(0.1)

        self.app.push_screen(Approval("write_file x"), accepted.append)
        await self.pilot.pause(0.1)
        await self.pilot.press("y")
        await self.pilot.pause(0.1)

        self.assertEqual(accepted, [False, True])


if __name__ == "__main__":
    unittest.main()
