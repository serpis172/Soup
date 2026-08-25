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
from typing import List, Optional

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
    metric: str = typer.Option(
        "magnitude",
        "--metric",
        help=(
            "'magnitude' (default): weight-only, streamed, no calibration data "
            "needed. 'wanda': activation-weighted (Sun et al. 2024) — needs the "
            "full model loaded plus --calibration-data; catches neurons with "
            "large weights that cancel out on real inputs, which magnitude "
            "alone misses, at the cost of a heavier scan."
        ),
    ),
    calibration_data: Optional[List[str]] = typer.Option(
        None,
        "--calibration-data",
        help=(
            "JSONL with a 'text' field per line, used only when --metric wanda. "
            "Repeatable — pass --calibration-data multiple times to draw the "
            "calibration set from more than one dataset (e.g. code + chat + a "
            "domain corpus); samples are pooled across all of them."
        ),
    ),
    calibration_samples: int = typer.Option(
        64, "--calibration-samples", help="How many calibration lines to use (wanda only)."
    ),
    max_length: int = typer.Option(
        512, "--max-length", help="Max tokens per calibration sample (wanda only)."
    ),
    bottom_k: int = typer.Option(
        10, "--bottom-k", help="Show the N least-important neurons per layer group."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write the full ranking as JSON (must stay under cwd)."
    ),
) -> None:
    """Rank output neurons by importance — weight-magnitude (default) or
    Wanda (activation-weighted). Analysis only — this never modifies the
    model. It tells you which neurons *could* be pruning candidates;
    deciding a ratio and acting on it is a separate, deliberate step
    (there's no auto-prune here).
    """
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    if metric not in ("magnitude", "wanda"):
        console.print(f"[red]--metric must be 'magnitude' or 'wanda', got {metric!r}[/]")
        raise typer.Exit(2)
    if metric == "wanda" and not calibration_data:
        console.print("[red]--metric wanda requires at least one --calibration-data <file.jsonl>[/]")
        raise typer.Exit(2)

    if output is not None:
        from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

        try:
            enforce_under_cwd_and_no_symlink(output, "output")
        except (ValueError, TypeError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc

    try:
        if metric == "wanda":
            from soup_cli.utils.neuron_compress import (
                load_calibration_texts,
                rank_importance_wanda,
            )

            texts = load_calibration_texts(
                calibration_data, max_samples_per_file=calibration_samples
            )
            if not texts:
                sources = ", ".join(escape(p) for p in calibration_data)
                console.print(f"[red]No usable 'text' field found in: {sources}[/]")
                raise typer.Exit(1)
            n_sources = len(calibration_data)
            source_note = f"{n_sources} dataset{'s' if n_sources != 1 else ''}"
            console.print(
                f"[dim]Loading {escape(model)} for Wanda scan "
                f"({len(texts)} calibration samples from {source_note})...[/]"
            )
            results = rank_importance_wanda(model, texts, modules=modules, max_length=max_length)
        else:
            from soup_cli.utils.neuron_compress import rank_importance

            weights_dir = resolve_model_weights(model)
            results = rank_importance(weights_dir, modules=modules)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        console.print(f"[red]Importance scan failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if not results:
        if metric == "wanda":
            console.print(
                "[yellow]No nn.Linear modules matched --model/--modules. Note: "
                "Wanda scoring only targets nn.Linear (covers Llama/Qwen/Mistral/"
                "Gemma-family models — it does not cover legacy Conv1D-based "
                "architectures like GPT-2).[/]"
            )
        else:
            console.print("[yellow]No scannable 2-D weight matrices found (check --model and --modules).[/]")
        raise typer.Exit(1)

    groups: dict = defaultdict(list)
    for r in results:
        groups[r.group].append(r)

    metric_label = "Wanda (activation-weighted)" if metric == "wanda" else "weight-magnitude"
    table = Table(title=f"Neuron importance ({metric_label}) — {escape(model)}")
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
    if metric == "wanda":
        console.print(
            "[dim]Wanda score = |weight| x activation-norm, aggregated per output "
            f"neuron over {len(texts)} calibration samples. Catches neurons whose "
            "large weights cancel out on real inputs, which pure magnitude misses.[/]"
        )
    else:
        console.print(
            "[dim]Lower norm = smaller possible output contribution from that neuron's "
            "weight row alone. This is a weight-only proxy (no calibration data) — a "
            "neuron with large weights that happen to cancel out on real inputs won't "
            "show up here; re-run with --metric wanda --calibration-data <file> for "
            "activation-weighted scoring instead.[/]"
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


def build_distill_config_yaml(
    *,
    student_base: str,
    teacher_model: str,
    data_train: Optional[str] = None,
    mode: str = "token",
    divergence: str = "forward_kl",
    temperature: float = 2.0,
    output: str = "./output/distilled",
) -> str:
    """Generate a ready-to-run distillation config for the *existing*
    trainer (soup_cli.trainer.distill.DistillTrainerWrapper — token-level
    KL or sequence-level KD, already wired to `soup train`/`task: distill`).

    This is a convenience bridge from `soup compress` to that trainer, not
    a new training implementation: after pruning/merging/SVD-factorizing a
    model, the natural recovery step is to distill from the pre-compression
    original (or any other teacher) back into the compressed student —
    exactly what `task: distill` already does when `base` is the student
    and `training.teacher_model` is the teacher.
    """
    import yaml as yaml_mod

    doc = {
        "base": student_base,
        "task": "distill",
        "data": {
            "train": data_train or "./data/train.jsonl",
            "format": "alpaca",
            "val_split": 0.1,
            "max_length": 2048,
        },
        "training": {
            "epochs": 1,
            "lr": 1e-5,
            "teacher_model": teacher_model,
            "distill_mode": mode,
            "distill_divergence": divergence,
            "distill_temperature": temperature,
        },
        "output": output,
    }
    yaml_text = yaml_mod.dump(doc, default_flow_style=False, sort_keys=False)
    if not data_train:
        # A real top-of-file comment, not text embedded in the field value
        # (that produced technically-parseable but ugly, easy-to-miss YAML
        # where the placeholder note became part of the literal path).
        yaml_text = (
            "# TODO: point data.train at your training data, or a dedicated\n"
            "# distillation set (prompts the teacher should be conditioned on).\n"
        ) + yaml_text
    return yaml_text


@app.command()
def distill_config(
    student: str = typer.Option(..., "--student", help="The compressed/pruned/merged model."),
    teacher: str = typer.Option(
        ..., "--teacher", help="Teacher model — typically the pre-compression original."
    ),
    data: Optional[str] = typer.Option(None, "--data", help="Training/distillation data path."),
    mode: str = typer.Option("token", "--mode", help="'token' (logit KL) or 'sequence' (teacher-generated text)."),
    divergence: str = typer.Option("forward_kl", "--divergence", help="forward_kl | reverse_kl | js"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write YAML here instead of stdout."),
) -> None:
    """Print (or write) a ready-to-run distillation config for recovering
    quality after `soup compress` — student=compressed model,
    teacher=original (or any other model). Uses the existing distillation
    trainer (`task: distill`) — this command only generates the config.
    """
    yaml_text = build_distill_config_yaml(
        student_base=student, teacher_model=teacher, data_train=data, mode=mode, divergence=divergence
    )
    if output:
        from soup_cli.utils.paths import atomic_write_text

        atomic_write_text(yaml_text, output, field="output")
        console.print(f"[green]Wrote distillation config:[/] {escape(output)}")
        console.print(f"  [dim]Run it with:[/] soup train --config {escape(output)}")
    else:
        console.print(yaml_text)
        console.print("[dim]Save this and run: soup train --config <file>.yaml[/]")


@app.command()
def svd(
    model: str = typer.Option(..., "--model", "-m", help="HF Hub id or local model directory."),
    modules: str = typer.Option("mlp,attn", "--modules", help="Comma list of module types to analyze."),
    energy: str = typer.Option(
        "0.90,0.95,0.99", "--energy", help="Comma list of energy-retention thresholds to report rank for."
    ),
    mode: str = typer.Option(
        "denoise", "--mode", help="'denoise' (same shape, always loadable) or 'factorize' (smaller, needs custom loading)."
    ),
    rank_at_energy: float = typer.Option(
        0.95, "--rank-at-energy", help="With --apply, pick each matrix's rank from this energy threshold."
    ),
    apply: bool = typer.Option(False, "--apply", help="Actually write a new checkpoint."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Required with --apply."),
) -> None:
    """SVD-based weight compression: denoise (default, safe, same-shape) or
    factorize (real size reduction, non-standard checkpoint — needs custom
    loading, see the printed manifest note). Without --apply this only
    reports the rank/energy profile; nothing is written.
    """
    from soup_cli.utils.spectrum_scan import resolve_model_weights
    from soup_cli.utils.svd_compress import analyze_svd

    thresholds = tuple(float(x) for x in energy.split(","))
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
        analysis = analyze_svd(weights_dir, modules=modules, energy_thresholds=thresholds)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        console.print(f"[red]SVD scan failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if not analysis:
        console.print("[yellow]No scannable 2-D weight matrices found.[/]")
        raise typer.Exit(1)

    table = Table(title=f"SVD rank/energy — {escape(model)} (mode={mode})")
    table.add_column("Param")
    table.add_column("Shape")
    for t in thresholds:
        table.add_column(f"Rank@{t}", justify="right")
    table.add_column(f"Ratio@{rank_at_energy}", justify="right")
    for a in analysis:
        row = [escape(a.param_name), f"{a.shape[0]}x{a.shape[1]}"]
        for t in thresholds:
            row.append(str(a.rank_at_energy[t]))
        rank = a.rank_at_energy.get(rank_at_energy, a.rank_at_energy[thresholds[-1]])
        row.append(f"{a.compression_ratio(rank):.3f}")
        table.add_row(*row)
    console.print(table)

    if not apply:
        console.print("[cyan]Dry run — re-run with --apply --output-dir <path> to write a checkpoint.[/]")
        return

    plan = {
        f"{a.param_name}.weight" if not a.param_name.endswith(".weight") else a.param_name: a.rank_at_energy.get(
            rank_at_energy, a.rank_at_energy[thresholds[-1]]
        )
        for a in analysis
    }
    console.print(f"[cyan]Applying SVD ({mode}) to {len(plan)} matrices, writing to {escape(output_dir)}...[/]")
    from soup_cli.utils.svd_compress import apply_svd_to_checkpoint

    try:
        report = apply_svd_to_checkpoint(weights_dir, output_dir, plan, mode=mode)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        console.print(f"[red]SVD apply failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    console.print(f"[green]Wrote {mode} checkpoint:[/] {escape(output_dir)} ({len(report)} matrices touched)")
    if mode == "factorize":
        console.print(
            "[yellow]Non-standard checkpoint — see svd_manifest.json in the output "
            "dir. NOT loadable by plain AutoModelForCausalLM.from_pretrained "
            "without custom code that reconstructs W = svd_u @ svd_v.[/]"
        )
    console.print(
        "[dim]Consider a distillation recovery pass:[/]\n"
        f"  soup compress distill-config --student {escape(output_dir)} --teacher {escape(model)}"
    )


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
    eval_data: Optional[str] = typer.Option(
        None,
        "--eval-data",
        help=(
            "JSONL with a 'text' field per line — with --apply, compares "
            "perplexity before/after the merge on these samples (loads both "
            "models, heavier than a plain apply — opt-in)."
        ),
    ),
    eval_samples: int = typer.Option(20, "--eval-samples", help="How many --eval-data lines to use."),
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

    if eval_data:
        import json as json_mod

        texts = []
        with open(eval_data, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json_mod.loads(line)
                text = row.get("text") if isinstance(row, dict) else None
                if text:
                    texts.append(text)
                if len(texts) >= eval_samples:
                    break
        if not texts:
            console.print(f"[red]--eval-data given but no usable 'text' field in {escape(eval_data)}[/]")
        else:
            console.print(f"[cyan]Quick eval: comparing perplexity on {len(texts)} samples...[/]")
            from soup_cli.utils.neuron_compress import quick_eval_merge

            try:
                result = quick_eval_merge(model, output_dir, texts)
                console.print(
                    f"  perplexity: {result['perplexity_before']:.3f} -> "
                    f"{result['perplexity_after']:.3f} "
                    f"({result['relative_increase_pct']:+.2f}%)"
                )
            except (ValueError, RuntimeError, OSError) as exc:
                console.print(f"[red]Quick eval failed:[/] {escape(str(exc))}")

    console.print(
        "[dim]Consider a distillation recovery pass:[/]\n"
        f"  soup compress distill-config --student {escape(output_dir)} --teacher {escape(model)}"
    )
