#!/usr/bin/env python3
"""Run MuSLR baseline and LogiCAM sequentially with a Rich terminal UI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


WORKSPACE = Path(__file__).resolve().parents[2]
ENV_FILE = WORKSPACE / ".env"
PYTHON = WORKSPACE / ".venv" / "bin" / "python"
console = Console()


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as error:
        raise SystemExit(f"{name} must be an integer") from error


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.1f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate both configurations without calling the model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Sample limit used only with --dry-run (default: 1).",
    )
    args = parser.parse_args()

    if not ENV_FILE.is_file():
        parser.error(f"missing configuration: {ENV_FILE}")
    if not PYTHON.is_file():
        parser.error(f"missing Python environment: {PYTHON}")
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    load_dotenv(ENV_FILE, override=False)
    output_dir = Path(
        os.environ.get(
            "MUSLR_OUTPUT_DIR",
            WORKSPACE / "outputs",
        )
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = (
        ("baseline", env_int("MUSLR_BASELINE_MAX_TOKENS", 2048)),
        ("logicam", env_int("MUSLR_LOGICAM_MAX_TOKENS", 4096)),
    )
    console.print(
        Panel.fit(
            "[bold]MuSLR sequential inference[/bold]\n"
            "baseline [dim]→[/dim] LogiCAM\n"
            f"[dim]Outputs:[/dim] {output_dir}",
            border_style="cyan",
        )
    )

    completed: list[tuple[str, Path, float]] = []
    total_started = monotonic()
    try:
        for index, (method, max_tokens) in enumerate(methods, start=1):
            suffix = ".dry-run" if args.dry_run else ""
            output = output_dir / f"muslr_{method}{suffix}.jsonl"
            console.rule(f"[bold cyan]Stage {index}/2 · {method}[/bold cyan]")
            console.print(
                f"[bold]Output:[/bold] {output}\n"
                f"[bold]Max tokens:[/bold] {max_tokens}\n"
                "[dim]Existing checkpoints will be resumed automatically.[/dim]"
            )

            command = [
                str(PYTHON),
                "-u",
                "-m",
                "legacy.free_text_run",
                "--method",
                method,
                "--max-tokens",
                str(max_tokens),
                "--output",
                str(output),
            ]
            if args.dry_run:
                command.extend(("--dry-run", "--limit", str(args.limit)))

            started = monotonic()
            subprocess.run(command, cwd=WORKSPACE, check=True)
            duration = monotonic() - started
            completed.append((method, output, duration))
            console.print(
                f"[bold green]✓ {method} finished[/bold green] "
                f"[dim]({format_duration(duration)})[/dim]"
            )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted. Re-run to resume checkpoints.[/bold yellow]")
        raise SystemExit(130) from None
    except subprocess.CalledProcessError as error:
        console.print(
            f"\n[bold red]Stage failed with exit code {error.returncode}.[/bold red]\n"
            "Fix the reported issue, then re-run to resume checkpoints."
        )
        raise SystemExit(error.returncode) from None

    summary = Table(title="MuSLR run summary", show_header=True, header_style="bold cyan")
    summary.add_column("Method")
    summary.add_column("Duration", justify="right")
    summary.add_column("Output")
    for method, output, duration in completed:
        summary.add_row(method, format_duration(duration), str(output))
    console.print(summary)
    console.print(
        f"[bold green]All stages finished in "
        f"{format_duration(monotonic() - total_started)}.[/bold green]"
    )


if __name__ == "__main__":
    main()
