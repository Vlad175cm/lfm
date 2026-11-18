# lfm

[![tests](https://github.com/Vlad175cm/lfm/actions/workflows/tests.yml/badge.svg)](https://github.com/Vlad175cm/lfm/actions/workflows/tests.yml)

A local chat with **LFM3-1.7B** in your terminal. It starts `llama-server` for
you, streams the answer, and — if you want — lets the model read files, edit
them and run commands inside the project. Black and white, square, nothing else
on screen.

```
$ lfm
```

## Install

```bash
git clone https://github.com/Vlad175cm/lfm.git
cd lfm
./install.sh
```

That's it. `install.sh` creates a virtualenv inside the clone, installs the
package, links `lfm` into `~/.local/bin`, downloads a prebuilt `llama.cpp`
(~23 MB) and the Q4_K_M model (~1.1 GB), and writes `~/.config/lfm/config.toml`.
Everything goes into your home directory; no sudo.

Skip the model download (it happens on demand later):

```bash
./install.sh --no-model
```

Requirements: Linux x86_64, Python 3.11+, ~1.3 GB of disk. Network is needed
once, for the installer. macOS arm64 is not tested.

Re-running `./install.sh` is safe: it reuses the venv and skips what is already
downloaded. The clone has to stay where it is — `~/.local/bin/lfm` points into
it.

## What it does

- starts `llama-server` with the model and shuts it down on exit;
- SSE streaming, markdown and code in answers, token and tok/s counters;
- monochrome UI, no borders, no rounded corners;
- context and speed in the top-right corner;
- commands `/help`, `/agent`, `/clear`, `/model`, `/temp`, `/system`, `/stats`,
  `/chats`, `/quit`;
- scroll the transcript with `shift+↑` / `shift+↓` (3 lines a press), even while
  typing; `esc` stops a generation.

## Agent mode

`/agent` (or `lfm --agent`) hands the model seven local tools:

| tool | needs confirmation |
|---|---|
| `list_dir`, `read_file`, `grep`, `fetch_url` | no |
| `write_file`, `edit`, `run_shell` | yes — `y` / `n` |

Everything runs inside the directory `lfm` was started in; paths that escape it
are refused. Tool calls and their output show up in the transcript as compact
`● read_file note.txt` steps, and the model gets the result and keeps talking.
`agent_steps` (default 8) caps how many rounds one answer may take.

Reality check: LFM3-1.7B is small. One or two tool steps are dependable, three
or more are a coin flip — it may start reasoning in the answer instead of
calling a tool, or give up counting. It also runs at ~12 tok/s on a CPU, so
every step costs seconds. `esc` stops the whole run.

## Flags

```
--model PATH     path to a .gguf file
--ctx N          context size (default: 8192)
--threads N      CPU threads
--port N         llama-server port
--temp F         temperature
--thinking       enable the model's reasoning (slower)
--agent          start with tools and file access on
--setup          install llama.cpp (and the model with --download-model)
```

## Transcripts

Every chat is saved to `~/.local/share/lfm/sessions/` as one JSON file per
conversation (`YYYYMMDD-HHMMSS-<id>.json`), written after each exchange.
`/chats` prints the folder and the file of the current conversation.

## Copying code

When an answer contains code, a `[ copy code ]` label appears under it — click
it to copy. `ctrl+y` copies the code of the last answer. Uses
`wl-copy`/`xclip`/`xsel` when available, otherwise OSC 52.

## Config

`~/.config/lfm/config.toml`, created by the installer. It holds the model path,
`ctx`, `threads`, `temperature`, `thinking`, `agent` and `agent_steps`.

## System prompt

The prompt ships as `lfm/system.md` and applies out of the box. Sources are
tried in this order:

1. `LFM_SYSTEM_FILE` if set,
2. `~/.config/lfm/system.md` — your own override,
3. `lfm/system.md` from the package,
4. a short built-in default.

Inside `lfm`, `/system` shows which source is active, and `/system reload`
re-reads it without restarting. The prompt is not editable from the chat: one
file, one source of truth.

## Manual install

Same thing, step by step:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
ln -sf "$PWD/.venv/bin/lfm" ~/.local/bin/lfm
lfm --setup --download-model   # llama.cpp + model, or drop the flag
```

The installer uses the same code under the hood, so both routes end up
identical.

## How it is built

- `install.sh` — the one-command installer;
- `lfm/server.py` — starts `llama-server`, waits for `/health`, SSE client;
- `lfm/config.py` — config plus model/binary discovery;
- `lfm/setup_llama.py` — installs the binary and downloads the model;
- `lfm/tools.py` — the seven local tools, sandboxed to the working directory;
- `lfm/ui/app.py` — the Textual app, streaming, agent loop and commands;
- `lfm/ui/mono.py` — forces monochrome syntax highlighting.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

36 tests, no model or network needed — they use a fake `llama-server`. CI runs
them on Python 3.11–3.13 for every push.

## License

The code in this repository is MIT (see `LICENSE`). The model is downloaded
from Hugging Face and is licensed separately under the
[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE);
`llama.cpp` is MIT.
