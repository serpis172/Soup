"""soup compress — optional model-density tools (v0.73.6).

Two independent, opt-in analyses over a model's ``.safetensors`` weights,
streamed one tensor at a time (same bounded-peak-RSS approach as
``soup spectrum``, reused not reimplemented):

- ``soup compress importance``: ranks output neurons by weight-magnitude —
  which ones contribute least, per layer. Analysis only, never modifies
  anything.
- ``soup compress neurons``: finds MLP intermediate neurons that are
  near-duplicates of each other (safe to merge without a meaningful output
  change) and, with ``--apply``, actually performs the merge into a new,
  smaller checkpoint.

See ``soup_cli/utils/neuron_compress.py`` for the algorithms and their
accuracy characterisation (median/p90 relative output error vs merge
threshold, measured on synthetic data — printed as guidance below too).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(
    no_args_is_help=True,
    help="Optional model-density tools: neuron importance + similar-neuron merging.",
)


@app.callback()
def _compress() -> None:
    """Optional model-density tools (v0.73.6)."""


@app.command()
def importance(
    model: str = typer.Option(
        ..., "--model", "-m", help="HF Hub id or local model directory."
    ),
    modules: str = typer.Option(
        "mlp,attn",
        "--modules",
        help="Comma list of module types to rank: e.g. 'mlp,attn' — or 'all'.",
    ),
    bottom_k: int = typer.Option(
        10, "--bottom-k", help="Show the N least-important neurons per layer group."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write the full ranking as JSON (must stay under cwd)."
    ),
) -> None:
    """Rank output neurons by weight-magnitude importance (streamed, no calibration data).

    Analysis only — this never modifies the model. It tells you which
    neurons *could* be pruning candidates; deciding a ratio and acting on
    it is a separate, deliberate step (there's no auto-prune here).
    """
    from soup_cli.utils.neuron_compress import rank_importance
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    if output is not None:
        from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

        try:
            enforce_under_cwd_and_no_symlink(output, "output")
        except (ValueError, TypeError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc

    try:
        weights_dir = resolve_model_weights(model)
        results = rank_importance(weights_dir, modules=modules)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        console.print(f"[red]Importance scan failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if not results:
        console.print("[yellow]No scannable 2-D weight matrices found (check --model and --modules).[/]")
        raise typer.Exit(1)

    groups: dict = defaultdict(list)
    for r in results:
        groups[r.group].append(r)

    table = Table(title=f"Neuron importance (weight-magnitude) — {escape(model)}")
    table.add_column("Group")
    table.add_column("Layers", justify="right")
    table.add_column("Neurons/layer", justify="right")
    table.add_column(f"Bottom {bottom_k} (avg norm)", justify="right")
    table.add_column("Median norm", justify="right")

    import statistics

    for group in sorted(groups):
        items = groups[group]
        all_norms = [n for r in items for n in r.row_norms]
        bottom_avgs = [
            sum(v for _i, v in r.least_important(bottom_k)) / max(1, min(bottom_k, r.n_neurons))
            for r in items
        ]
        table.add_row(
            escape(group),
            str(len(items)),
            str(items[0].n_neurons),
            f"{sum(bottom_avgs) / len(bottom_avgs):.4f}",
            f"{statistics.median(all_norms):.4f}",
        )
    console.print(table)
    console.print(
        "[dim]Lower norm = smaller possible output contribution from that neuron's "
        "weight row alone. This is a weight-only proxy (no calibration data) — a "
        "neuron with large weights that happen to cancel out on real inputs won't "
        "show up here; that needs activation-based scoring (e.g. Wanda), which this "
        "streaming pass deliberately doesn't do (would require loading the full "
        "model + running data through it).[/]"
    )

    if output is not None:
        from soup_cli.utils.paths import atomic_write_text

        payload = {
            "model": model,
            "modules": modules,
            "layers": [
                {
                    "param_name": r.param_name,
                    "group": r.group,
                    "module_type": r.module_type,
                    "n_neurons": r.n_neurons,
                    "least_important": [
                        {"index": i, "norm": v} for i, v in r.least_important(bottom_k)
                    ],
                }
                for r in results
            ],
        }
        atomic_write_text(json.dumps(payload, indent=2), output, field="output")
        console.print(f"[green]Wrote report:[/] {escape(output)}")


@app.command()
def neurons(
    model: str = typer.Option(
        ..., "--model", "-m", help="HF Hub id or local model directory."
    ),
    threshold: float = typer.Option(
        0.98,
        "--threshold",
        help=(
            "Minimum joint gate/up cosine similarity to call two neurons "
            "redundant. Measured on synthetic weights: median relative "
            "output error is ~0.3% at 0.998 sim, ~1.3% at 0.97 sim — 0.98 "
            "is a conservative default, lower it for a more aggressive "
            "merge at the cost of more drift."
        ),
    ),
    max_pairs_per_layer: int = typer.Option(
        50, "--max-pairs-per-layer", help="Cap merge candidates considered per layer."
    ),
    apply: bool = typer.Option(
        False, "--apply", help="Actually perform the merge and write a new checkpoint."
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", help="Required with --apply: where to write the merged model."
    ),
    allow_nonuniform: bool = typer.Option(
        False,
        "--allow-nonuniform",
        help=(
            "Standard HF configs have one global intermediate_size for every "
            "layer, so by default every MLP layer merges the same number of "
            "pairs (the minimum found anywhere), and refuses if that's 0. "
            "This flag instead merges each layer independently, producing a "
            "checkpoint with a per-layer-varying MLP width that plain "
            "AutoModelForCausalLM.from_pretrained can't load — only for "
            "custom deployment code that reads _soup_per_layer_intermediate_size."
        ),
    ),
) -> None:
    """Find (and optionally merge) near-duplicate MLP intermediate neurons.

    Two neurons are candidates only when BOTH their gate_proj and up_proj
    rows are highly similar (cosine >= --threshold) — that joint condition
    is what makes ``act_i(x) ≈ act_j(x)`` for any input, without needing
    calibration data. Without --apply this only reports candidates; nothing
    is written. MLP/FFN only — see module docstring in
    soup_cli/utils/neuron_compress.py for why attention heads aren't in
    scope here.
    """
    from soup_cli.utils.neuron_compress import find_merge_candidates
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    if apply and not output_dir:
        console.print("[red]--apply requires --output-dir.[/]")
        raise typer.Exit(2)
    if output_dir is not None:
        from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

        try:
            enforce_under_cwd_and_no_symlink(output_dir, "output-dir")
        except (ValueError, TypeError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc

    try:
        weights_dir = resolve_model_weights(model)
        candidates = find_merge_candidates(
            weights_dir, threshold=threshold, max_pairs_per_layer=max_pairs_per_layer
        )
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        console.print(f"[red]Neuron-merge scan failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if not candidates:
        console.print(
            f"[yellow]No merge candidates found at threshold={threshold} "
            f"(no MLP layers, or nothing similar enough — try lowering "
            f"--threshold).[/]"
        )
        raise typer.Exit(0)

    total_pairs = sum(len(v) for v in candidates.values())
    table = Table(title=f"Merge candidates — {escape(model)} (threshold={threshold})")
    table.add_column("Layer", justify="right")
    table.add_column("Pairs", justify="right")
    table.add_column("Best joint sim", justify="right")
    table.add_column("Worst joint sim", justify="right")
    for layer_idx in sorted(candidates):
        sims = [c.joint_similarity for c in candidates[layer_idx]]
        table.add_row(str(layer_idx), str(len(sims)), f"{max(sims):.4f}", f"{min(sims):.4f}")
    console.print(table)
    console.print(
        f"[dim]{total_pairs} candidate pair(s) across {len(candidates)} layer(s). "
        f"Each merged pair shrinks that layer's intermediate_size by 1.[/]"
    )

    if not apply:
        console.print(
            "[cyan]Dry run — re-run with --apply --output-dir <path> to write "
            "a merged checkpoint.[/]"
        )
        return

    console.print(f"[cyan]Applying {total_pairs} merge(s) and writing to {escape(output_dir)}...[/]")
    from soup_cli.utils.neuron_compress import apply_merges_to_checkpoint

    try:
        summary = apply_merges_to_checkpoint(
            weights_dir, output_dir, candidates, allow_nonuniform=allow_nonuniform
        )
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        console.print(f"[red]Merge apply failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    console.print(
        Panel(
            "\n".join(
                f"layer {layer_idx}: intermediate_size {before} -> {after}"
                for layer_idx, before, after in summary
            )
            or "(no MLP triplets found to shrink)",
            title="Merge summary",
            border_style="green",
        )
    )
    console.print(f"[green]Wrote merged model:[/] {escape(output_dir)}")
    console.print(
        "[yellow]Reminder: this changed intermediate_size for the merged layers. "
        "Re-run your eval suite before shipping — the joint-similarity screen "
        "bounds *expected* drift, it doesn't guarantee zero quality loss.[/]"
    )
