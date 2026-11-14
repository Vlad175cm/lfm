"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from .config import (
    MODEL_DOWNLOAD_HINT,
    Config,
    find_model,
    find_server_binary,
)
from .setup_llama import download_model, install_llama

VERSION = "1.0.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lfm",
        description="Local LFM3 chat in your terminal. Brings up llama.cpp for you.",
    )
    parser.add_argument("--model", help="path to a .gguf file")
    parser.add_argument("--ctx", type=int, help="context size (default: 8192)")
    parser.add_argument("--threads", type=int, help="CPU threads")
    parser.add_argument("--port", type=int, help="llama-server port")
    parser.add_argument("--temp", type=float, help="sampling temperature")
    parser.add_argument("--thinking", action="store_true", help="enable the model's reasoning")
    parser.add_argument("--agent", action="store_true", help="start with tools and file access on")
    parser.add_argument("--config", help="path to config.toml")
    parser.add_argument("--setup", action="store_true", help="install llama.cpp and exit")
    parser.add_argument("--download-model", action="store_true", help="fetch the model during --setup")
    parser.add_argument("--force", action="store_true", help="reinstall llama.cpp")
    parser.add_argument("--version", action="version", version=f"lfm {VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    cfg = Config.load(Path(args.config).expanduser() if args.config else None)

    if args.model:
        cfg.model = args.model
    if args.ctx:
        cfg.ctx = args.ctx
    if args.threads:
        cfg.threads = args.threads
    if args.port:
        cfg.port = args.port
    if args.temp is not None:
        cfg.temperature = args.temp
    if args.thinking:
        cfg.thinking = True
    if args.agent:
        cfg.agent = True

    if args.setup:
        return _setup(cfg, console, download=args.download_model, force=args.force)

    model = find_model(args.model or cfg.model)
    if model is None:
        console.print("[bold red]model not found[/]")
        console.print(MODEL_DOWNLOAD_HINT, style="dim")
        console.print("\nor in one command: [bold]lfm --setup --download-model[/]")
        return 2

    binary = find_server_binary()
    if binary is None:
        console.print("[bold red]llama-server not found[/]")
        console.print("install a prebuilt binary with [bold]lfm --setup[/]", style="dim")
        return 2

    from .ui.app import LfmApp

    app = LfmApp(cfg, model, binary)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        if app.server is not None:
            app.server.stop()
    return 0


def _setup(cfg: Config, console: Console, *, download: bool, force: bool) -> int:
    from .config import data_home

    binary = find_server_binary()
    if binary is None or force:
        console.print("[dim]installing llama.cpp…[/]")
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("llama.cpp", total=None)

            def report(done: int, total: int | None) -> None:
                progress.update(
                    task,
                    total=total,
                    completed=done,
                    description=f"llama.cpp · {done / 1e6:.0f} MB",
                )

            try:
                binary = install_llama(report)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]failed:[/] {exc}")
                return 1
        console.print(f"[green]✓[/] llama.cpp: [dim]{binary}[/]")
        console.print(f"[dim]version: {(data_home() / 'version.txt').read_text().strip()}[/]")
    else:
        console.print(f"[green]✓[/] llama.cpp already installed: [dim]{binary}[/]")

    model = find_model(cfg.model)
    if model is None and download:
        console.print("[dim]downloading the model (~1.1 GB)…[/]")
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("model", total=None)

            def report_model(done: int, total: int | None) -> None:
                progress.update(
                    task,
                    total=total,
                    completed=done,
                    description=f"model · {done / 1e6:.0f} MB",
                )

            try:
                model = download_model(report_model)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]failed:[/] {exc}")
                return 1
        console.print(f"[green]✓[/] model: [dim]{model}[/]")

    if model is None:
        console.print("\n[yellow]model not found[/]")
        console.print(MODEL_DOWNLOAD_HINT, style="dim")
        console.print("\nor: [bold]lfm --setup --download-model[/]")
    else:
        console.print(f"[green]✓[/] model: [dim]{model}[/]")
        cfg.model = str(model)

    path = cfg.save()
    console.print(f"[dim]config: {path}[/]")
    console.print("\nrun it: [bold]lfm[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
