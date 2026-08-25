"""soup pipeline — run training.pipeline's stages against a checkpoint.

Reads the SAME YAML config `soup train` would use, and runs whichever of
`training.pipeline.{activation_scan,compress,distill}` are enabled, in
their fixed order (see utils/pipeline_orchestrator.py's module docstring
for why this is a separate, explicit command rather than something `soup
train` runs automatically).

Usage:
    soup pipeline run config.yaml
    soup pipeline run config.yaml --checkpoint-dir /local/snapshot/dir
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

app = typer.Typer(help="Run training.pipeline's activation_scan/compress/distill stages.")
console = Console()


@app.command("run")
def run(
    config_path: Path = typer.Argument(..., help="Training YAML with a training.pipeline block."),
    checkpoint_dir: Optional[str] = typer.Option(
        None,
        "--checkpoint-dir",
        help=(
            "Local directory of the checkpoint to scan/compress. Defaults to "
            "`base:` in the config, which only works if that's already a "
            "local path — a HF Hub model id needs `soup pull`/manual "
            "snapshot_download first (this command does not download "
            "multi-GB weights as a side effect)."
        ),
    ),
) -> None:
    """Run the enabled training.pipeline stages and report what they produced."""
    from soup_cli.config.loader import load_config
    from soup_cli.utils.pipeline_orchestrator import run_pipeline_stages

    console.print(f"[dim]Loading config from {config_path}...[/]")
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Config error:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    if cfg.training.pipeline is None:
        console.print(
            "[yellow]No training.pipeline block in this config — nothing to run.[/]\n"
            "[dim]See docs/pipeline.md for activation_scan/compress/distill stage syntax.[/]"
        )
        raise typer.Exit(code=0)

    resolved_dir = checkpoint_dir or cfg.base
    if not Path(resolved_dir).is_dir():
        console.print(
            f"[red]--checkpoint-dir/base is not a local directory: {escape(resolved_dir)}[/]\n"
            f"[dim]If base: is a HF Hub model id, download it first (e.g. "
            f"`soup pull {escape(str(resolved_dir))}`) and pass "
            f"--checkpoint-dir pointing at the local snapshot.[/]"
        )
        raise typer.Exit(code=1)

    try:
        results = run_pipeline_stages(cfg, checkpoint_dir=resolved_dir)
    except Exception as exc:
        console.print(f"[red]Pipeline stage failed:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    if not results:
        console.print(
            "[yellow]training.pipeline is set but every stage is disabled "
            "(enabled: false / unset) — nothing ran.[/]"
        )
        raise typer.Exit(code=0)

    table = Table(title="Pipeline stages", show_header=True)
    table.add_column("Stage", style="bold")
    table.add_column("Detail")
    table.add_column("Output")
    for r in results:
        table.add_row(r.stage, r.detail, r.output_path or "-")
    console.print(table)

    final = results[-1]
    if final.output_path and final.stage != "activation_scan":
        console.print(
            f"\n[green]Done.[/] To train against the result, point `base:` at:\n"
            f"  {final.output_path}"
        )
