"""SVD-based weight compression (v0.73.7) — two independent, opt-in modes.

Every 2-D weight matrix has a singular value spectrum; most of it carries
little "signal" (same idea ``spectrum_scan.py`` already uses for SNR-based
LoRA targeting, applied here to compression instead of target selection).

- **denoise** (default, always stock-compatible): replace W [m, n] with its
  best rank-k reconstruction, reshaped back to the *same* [m, n] shape.
  Doesn't reduce stored parameter count — this is about removing low-energy
  noise from the weights, not shrinking the file. Zero architecture risk:
  any stock ``AutoModelForCausalLM.from_pretrained`` loads the result
  unchanged, because nothing about the module structure changed.

- **factorize**: replace W [m, n] with two smaller matrices U [m, k] and V
  [k, n] such that W ≈ U @ V — a genuine parameter reduction whenever
  k*(m+n) < m*n. This changes module structure (one Linear becomes two),
  so — same honest limitation as neuron merging's non-uniform case — the
  result needs custom loading code and is not loadable by
  ``AutoModelForCausalLM.from_pretrained`` out of the box. A
  ``svd_manifest.json`` documents exactly what was factored, at what rank,
  so a custom loader can reconstruct the mapping.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from soup_cli.utils.spectrum_scan import classify_module, layer_type_signature

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _np():
    import numpy as np

    return np


@dataclass(frozen=True)
class SvdAnalysis:
    param_name: str
    group: str
    module_type: str
    shape: Tuple[int, int]
    # energy_threshold -> minimum rank achieving it
    rank_at_energy: Dict[float, int]

    def compression_ratio(self, rank: int) -> float:
        """factored_params / original_params — < 1.0 means smaller."""
        m, n = self.shape
        return (rank * (m + n)) / (m * n)


def _rank_for_energy(singular_values: "NDArray[Any]", threshold: float) -> int:
    np = _np()
    energy = singular_values.astype(np.float64) ** 2
    total = energy.sum()
    if total <= 0:
        return 1
    cumulative = np.cumsum(energy) / total
    idx = int(np.searchsorted(cumulative, threshold))
    return min(idx + 1, len(singular_values))


def analyze_matrix(
    name: str, matrix: "NDArray[Any]", *, energy_thresholds: Tuple[float, ...] = (0.90, 0.95, 0.99)
) -> SvdAnalysis:
    np = _np()
    a = np.asarray(matrix, dtype=np.float32)
    # full_matrices=False: economy SVD, O(min(m,n)^2 * max(m,n)) instead of
    # O(m^2*n + n^3) — only need singular values + the thin factors, never
    # the full square U/V.
    _u, s, _vt = np.linalg.svd(a, full_matrices=False)
    ranks = {t: _rank_for_energy(s, t) for t in energy_thresholds}
    return SvdAnalysis(
        param_name=name,
        group=layer_type_signature(name),
        module_type=classify_module(name) or "other",
        shape=(a.shape[0], a.shape[1]),
        rank_at_energy=ranks,
    )


def analyze_svd(
    weights_dir: str, *, modules: str = "mlp,attn", energy_thresholds: Tuple[float, ...] = (0.90, 0.95, 0.99)
) -> Tuple[SvdAnalysis, ...]:
    """Stream every kept weight matrix and report its SVD rank/energy profile.

    Same streaming approach as neuron_compress.rank_importance — peak RSS
    bounded by the largest single matrix, not the whole model.
    """
    from soup_cli.utils.spectrum_scan import iter_weight_matrices

    return tuple(
        analyze_matrix(name, matrix, energy_thresholds=energy_thresholds)
        for name, matrix in iter_weight_matrices(weights_dir, modules=modules)
    )


def denoise_matrix(matrix: "NDArray[Any]", rank: int) -> "NDArray[Any]":
    """Replace ``matrix`` with its best rank-``rank`` reconstruction, same shape."""
    np = _np()
    a = np.asarray(matrix, dtype=np.float32)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    k = min(rank, len(s))
    return (u[:, :k] * s[:k]) @ vt[:k, :]


def factorize_matrix(matrix: "NDArray[Any]", rank: int) -> Tuple["NDArray[Any]", "NDArray[Any]"]:
    """Decompose ``matrix`` [m, n] into U [m, k], V [k, n] with matrix ≈ U @ V."""
    np = _np()
    a = np.asarray(matrix, dtype=np.float32)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    k = min(rank, len(s))
    u_k = np.ascontiguousarray((u[:, :k] * s[:k]).astype(np.float32))
    v_k = np.ascontiguousarray(vt[:k, :].astype(np.float32))
    return u_k, v_k


def apply_svd_to_checkpoint(
    weights_dir: str,
    output_dir: str,
    plan: Dict[str, int],
    *,
    mode: str = "denoise",
) -> List[Dict[str, Any]]:
    """Write a new checkpoint with SVD compression applied per ``plan``.

    ``plan`` maps a weight's key (e.g. ``model.layers.0.mlp.down_proj.weight``)
    to the rank to use for it. Keys not in ``plan`` pass through unchanged.

    ``mode="denoise"``: same shape, always stock-loadable.
    ``mode="factorize"``: real reduction, writes ``<key>` as two tensors
    (``<key>.svd_u``, ``<key>.svd_v``) instead, plus ``svd_manifest.json`` —
    needs custom loading code, see module docstring.
    """
    import json as json_mod

    from safetensors import safe_open
    from safetensors.numpy import save_file

    from soup_cli.utils.spectrum_scan import _discover_safetensors, _framework

    if mode not in ("denoise", "factorize"):
        raise ValueError(f"mode must be 'denoise' or 'factorize', got {mode!r}")

    np = _np()
    os.makedirs(output_dir, exist_ok=True)
    framework, to_np = _framework()

    all_tensors: Dict[str, "NDArray[Any]"] = {}
    manifest_entries: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []

    for path in _discover_safetensors(weights_dir):
        with safe_open(path, framework=framework) as handle:
            for key in handle.keys():
                if key not in plan:
                    all_tensors[key] = np.ascontiguousarray(to_np(handle.get_tensor(key)).copy())
                    continue
                rank = plan[key]
                original = to_np(handle.get_tensor(key)).copy()
                m, n = original.shape[0], original.shape[1]
                if mode == "denoise":
                    all_tensors[key] = np.ascontiguousarray(denoise_matrix(original, rank))
                    report.append({"key": key, "mode": "denoise", "rank": rank, "shape": [m, n]})
                else:
                    u_k, v_k = factorize_matrix(original, rank)
                    all_tensors[f"{key}.svd_u"] = u_k
                    all_tensors[f"{key}.svd_v"] = v_k
                    manifest_entries.append(
                        {
                            "original_key": key,
                            "u_key": f"{key}.svd_u",
                            "v_key": f"{key}.svd_v",
                            "original_shape": [m, n],
                            "rank": rank,
                            "reconstruction": "W ~= svd_u @ svd_v",
                        }
                    )
                    report.append(
                        {
                            "key": key,
                            "mode": "factorize",
                            "rank": rank,
                            "shape": [m, n],
                            "compression_ratio": (rank * (m + n)) / (m * n),
                        }
                    )

    save_file(all_tensors, os.path.join(output_dir, "model.safetensors"))

    config_path = os.path.join(weights_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json_mod.load(fh)
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as fh:
            json_mod.dump(config, fh, indent=2)

    if mode == "factorize" and manifest_entries:
        with open(os.path.join(output_dir, "svd_manifest.json"), "w", encoding="utf-8") as fh:
            json_mod.dump(
                {
                    "format": "soup-svd-factorized-v1",
                    "note": (
                        "This checkpoint is NOT loadable by plain "
                        "AutoModelForCausalLM.from_pretrained — the factored "
                        "keys need custom model code that reconstructs "
                        "W = svd_u @ svd_v for each entry below before use."
                    ),
                    "entries": manifest_entries,
                },
                fh,
                indent=2,
            )

    return report
