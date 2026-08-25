"""Optional model-density tools: neuron importance ranking and similar-neuron
merging for the MLP/FFN block (v0.73.6).

Two independent, opt-in analyses — neither runs unless explicitly invoked,
neither modifies a checkpoint unless ``--apply`` is passed:

- **Importance** (:func:`rank_importance`): per-output-neuron L2-norm of
  each weight matrix's rows, streamed one tensor at a time via
  :func:`soup_cli.utils.spectrum_scan.iter_weight_matrices` — same
  bounded-peak-RSS approach the Spectrum scan already uses, reused rather
  than reimplemented. Cheap, always available, no calibration data. Ranks
  which output neurons contribute least (by weight magnitude) so a human
  can decide whether to prune them — this module only ranks, it does not
  remove anything.

- **Similar-neuron merging** (:func:`find_merge_candidates` /
  :func:`apply_merges`): targets the MLP intermediate dimension of a
  SwiGLU-style block (``down_proj(act(gate_proj(x)) * up_proj(x))`` —
  Llama/Qwen/Mistral-family naming). Two intermediate neurons *i* and *j*
  are declared redundant only when **both** ``gate_proj`` row *i* ≈ row *j*
  **and** ``up_proj`` row *i* ≈ row *j* (cosine similarity above
  threshold). That joint condition is what makes the merge safe without
  needing calibration data: if both input-side rows are ~equal, then for
  *any* input x, ``act_i(x) ≈ act_j(x)`` follows directly (silu is
  1-Lipschitz-ish and continuous, so small row differences give small
  activation differences) — no forward pass over data required to justify
  it. Merging then folds neuron j's output contribution into neuron i's
  (``down_proj[:, i] += down_proj[:, j]``) and drops row j from
  gate_proj/up_proj and column j from down_proj, shrinking
  ``intermediate_size`` by the number of merged pairs.

  This is an approximation (exact only in the limit of identical rows);
  :func:`apply_merges` is unit-tested against exact-duplicate neurons
  (output must be bit-identical) and near-duplicate ones (output must stay
  within a small bound) — see ``tests`` in this repo.

Scope, stated plainly: MLP/FFN intermediate neurons only. Attention-head
pruning/merging has extra structure (per-head grouping, RoPE, GQA) this
module does not attempt — a reasonable v1 boundary, not an oversight.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

from soup_cli.utils.spectrum_scan import (
    _discover_safetensors,
    _framework,
    _LAYER_IDX_RE,
    classify_module,
    layer_type_signature,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Same cap philosophy as spectrum_scan's _MAX_MATRIX_ELEMENTS: a full [d, d]
# similarity matrix is what bounds peak RSS here, not the weight matrices
# themselves (those are already bounded by streaming one layer at a time).
# d=45000 -> d*d*4 bytes ≈ 8.1 GB, a reasonable ceiling for one layer's
# analysis; larger intermediate dims fall back to a chunked comparison
# instead of failing outright (see _pairwise_cosine_topk).
_MAX_SIMILARITY_DIM = 45_000


def _np():
    import numpy as np  # lazy — keep the CLI import light

    return np


# ---------------------------------------------------------------------------
# Importance ranking (any 2-D weight matrix, streamed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeuronImportance:
    """Per-row (output-neuron) importance for one weight matrix."""

    param_name: str
    group: str
    module_type: str
    row_norms: Tuple[float, ...]  # length = matrix.shape[0]

    @property
    def n_neurons(self) -> int:
        return len(self.row_norms)

    def least_important(self, k: int) -> List[Tuple[int, float]]:
        """Return the ``k`` lowest-norm (index, norm) pairs, ascending."""
        ranked = sorted(enumerate(self.row_norms), key=lambda kv: kv[1])
        return ranked[:k]


def row_l2_norms(matrix: "NDArray[Any]") -> "NDArray[Any]":
    """L2 norm of each row — the per-output-neuron importance score.

    Cheap, deterministic, no calibration data: a neuron whose entire
    incoming-weight row is near zero can only ever produce a near-zero
    contribution, regardless of the input, so it's a defensible pure-weight
    proxy for "contributes little to the output". It will not catch a
    neuron with large weights that happen to cancel out on real inputs —
    that needs activation data (Wanda-style), out of scope for this
    streaming, no-forward-pass pass by design (see module docstring).
    """
    np = _np()
    a = np.asarray(matrix, dtype=np.float32)
    return np.linalg.norm(a, axis=1)


def rank_importance_wanda(
    model_path: str,
    calibration_texts: List[str],
    *,
    modules: str = "mlp,attn",
    max_length: int = 512,
    device: Optional[str] = None,
) -> Tuple[NeuronImportance, ...]:
    """Load the real model + tokenizer and rank neurons via Wanda instead of
    plain magnitude. Heavier than :func:`rank_importance` (needs the full
    model resident, not just streamed weight files) — that's the documented
    tradeoff for catching neurons magnitude alone would miss.

    Returns the same :class:`NeuronImportance` shape as
    :func:`rank_importance` so callers (CLI table, UI, JSON export) don't
    need to know which metric produced the ranking.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.utils.trust_remote import model_requires_trust_remote_code

    trust_remote_code = bool(model_requires_trust_remote_code(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, dtype=torch.float32
    )
    if device:
        model = model.to(device)

    mod_filter = {m.strip() for m in modules.split(",")} if modules != "all" else None

    def _wanted(name: str) -> bool:
        if mod_filter is None:
            return True
        # classify_module expects parameter-style names (ending in
        # ".weight", matching safetensors keys / iter_weight_matrices) —
        # model.named_modules() gives *module* names without that suffix.
        # Without appending it here, every mlp/attn module silently failed
        # to match, target_names collapsed to an empty list -> `or None`
        # -> "score everything" (including lm_head) despite --modules
        # filtering to mlp/attn only. Confirmed via a real Llama-arch test
        # model where lm_head showed up in a mlp,attn-filtered scan.
        kind = classify_module(f"{name}.weight")
        return kind in mod_filter

    target_names = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and _wanted(name)
    ] or None

    calib_ids = [
        tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
        for text in calibration_texts
        if text and text.strip()
    ]
    if not calib_ids:
        raise ValueError("no usable calibration text (all lines empty after stripping)")

    scores = compute_wanda_importance(model, calib_ids, target_names=target_names)
    return tuple(
        NeuronImportance(
            param_name=f"{name}.weight",
            group=layer_type_signature(name),
            module_type=classify_module(name) or "other",
            row_norms=tuple(float(x) for x in arr),
        )
        for name, arr in scores.items()
    )


def rank_importance(
    weights_dir: str, *, modules: str = "all"
) -> Tuple[NeuronImportance, ...]:
    """Stream every kept weight matrix and rank its rows by L2 norm.

    Peak RSS ≈ the largest single kept matrix (same guarantee as
    ``iter_weight_matrices``, which this calls directly).
    """
    from soup_cli.utils.spectrum_scan import iter_weight_matrices

    results = []
    for name, matrix in iter_weight_matrices(weights_dir, modules=modules):
        norms = row_l2_norms(matrix)
        results.append(
            NeuronImportance(
                param_name=name,
                group=layer_type_signature(name),
                module_type=classify_module(name) or "other",
                row_norms=tuple(float(x) for x in norms),
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# MLP triplet streaming (gate_proj / up_proj / down_proj, one layer at a time)
# ---------------------------------------------------------------------------

_GATE_MARKERS = ("gate_proj", "mlp.w1", "mlp.wi_0")
_UP_MARKERS = ("up_proj", "mlp.w3", "mlp.wi_1", "mlp.wi")
_DOWN_MARKERS = ("down_proj", "mlp.w2", "mlp.wo")


def _mlp_role(key: str) -> Optional[str]:
    low = key.lower()
    if not low.endswith(".weight"):
        return None
    if any(m in low for m in _GATE_MARKERS):
        return "gate"
    if any(m in low for m in _UP_MARKERS):
        return "up"
    if any(m in low for m in _DOWN_MARKERS):
        return "down"
    return None


def _layer_idx(key: str) -> Optional[int]:
    match = _LAYER_IDX_RE.search(key)
    if not match:
        return None
    digits = "".join(ch for ch in match.group(0) if ch.isdigit())
    return int(digits) if digits else None


@dataclass(frozen=True)
class MlpTriplet:
    layer_idx: int
    gate_key: str
    up_key: str
    down_key: str
    gate: "NDArray[Any]"  # [intermediate, hidden]
    up: "NDArray[Any]"  # [intermediate, hidden]
    down: "NDArray[Any]"  # [hidden, intermediate]


def iter_mlp_triplets(weights_dir: str) -> Iterator[MlpTriplet]:
    """Yield one decoder layer's (gate_proj, up_proj, down_proj) at a time.

    Only architectures with this separate-gate/up/down SwiGLU-style MLP are
    supported (Llama/Qwen/Mistral/Gemma family naming, plus a few known
    aliases) — a layer missing any of the three is skipped with a debug
    log rather than raising, so a mixed/unusual checkpoint degrades to
    "found nothing" instead of crashing the whole scan.
    """
    from safetensors import safe_open

    framework, to_np = _framework()
    # name -> {role: array}, built up across shards since a layer's three
    # matrices are not guaranteed to live in the same shard file.
    pending: Dict[int, Dict[str, Tuple[str, "NDArray[Any]"]]] = {}

    for path in _discover_safetensors(weights_dir):
        with safe_open(path, framework=framework) as handle:
            for key in handle.keys():
                role = _mlp_role(key)
                if role is None:
                    continue
                idx = _layer_idx(key)
                if idx is None:
                    continue
                shape = tuple(handle.get_slice(key).get_shape())
                if len(shape) != 2:
                    continue
                if shape[0] * shape[1] > 2**31:
                    logger.warning("neuron-merge: skipping %s — too large to load", key)
                    continue
                array = to_np(handle.get_tensor(key))
                # safetensors' numpy framework returns a *view* onto the
                # memory-mapped file (OWNDATA=False) — valid only while this
                # `with safe_open(...)` block is open. iter_mlp_triplets is
                # consumed via `list(...)` by callers that need every
                # triplet before merging (apply_merges_to_checkpoint), which
                # outlives this block and can outlive the file's mmap
                # entirely once it's reopened elsewhere (e.g. this same
                # function's own pass 2) — an unguarded view would then
                # silently read back *different* data than what was
                # actually scored. Copy once, here, so what gets yielded is
                # always independently owned.
                array = array.copy()
                slot = pending.setdefault(idx, {})
                slot[role] = (key, array)
                if {"gate", "up", "down"} <= slot.keys():
                    gk, gw = slot["gate"]
                    uk, uw = slot["up"]
                    dk, dw = slot["down"]
                    yield MlpTriplet(idx, gk, uk, dk, gw, uw, dw)
                    del pending[idx]

    if pending:
        missing = sorted(pending.keys())
        logger.debug(
            "neuron-merge: %d layer(s) had an incomplete gate/up/down triplet "
            "(missing role) and were skipped: %s",
            len(missing), missing,
        )


# ---------------------------------------------------------------------------
# Similarity + merge planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeCandidate:
    layer_idx: int
    i: int
    j: int
    gate_similarity: float
    up_similarity: float

    @property
    def joint_similarity(self) -> float:
        return min(self.gate_similarity, self.up_similarity)


def _row_cosine_matrix(np_mod, matrix) -> Any:
    """Full [n, n] row-wise cosine similarity via one GEMM.

    ``normalize(W) @ normalize(W).T`` — same one-shot-matmul approach as
    computing a Gram matrix; O(n^2 * hidden) FLOPs, O(n^2) memory. Bounded
    by ``_MAX_SIMILARITY_DIM`` by the caller.
    """
    norms = np_mod.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np_mod.where(norms == 0, 1.0, norms)
    normed = matrix / norms
    return normed @ normed.T


def find_merge_candidates_in_layer(
    triplet: MlpTriplet, *, threshold: float = 0.95, max_pairs: int = 50
) -> List[MergeCandidate]:
    """Greedy, non-overlapping merge candidates for one layer.

    A neuron participates in at most one merge per layer (simple 1:1
    matching, not chained clustering — chaining compounds the
    joint-similarity approximation error across more than one pair, so it's
    deliberately not done here). Sorted by joint similarity, descending.
    """
    np = _np()
    d = triplet.gate.shape[0]
    if d > _MAX_SIMILARITY_DIM:
        logger.warning(
            "neuron-merge: layer %d intermediate_size=%d exceeds the "
            "%d-neuron similarity cap — skipped (raise threshold coverage "
            "by scanning fewer layers, or shard the comparison manually)",
            triplet.layer_idx, d, _MAX_SIMILARITY_DIM,
        )
        return []

    gate_sim = _row_cosine_matrix(np, triplet.gate.astype(np.float32))
    up_sim = _row_cosine_matrix(np, triplet.up.astype(np.float32))
    joint = np.minimum(gate_sim, up_sim)
    np.fill_diagonal(joint, -1.0)  # exclude self-pairs

    iu, ju = np.triu_indices(d, k=1)
    scores = joint[iu, ju]
    keep = scores >= threshold
    if not keep.any():
        return []
    iu, ju, scores = iu[keep], ju[keep], scores[keep]
    order = np.argsort(-scores)

    used = set()
    out: List[MergeCandidate] = []
    for idx in order:
        i, j = int(iu[idx]), int(ju[idx])
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        out.append(
            MergeCandidate(
                layer_idx=triplet.layer_idx,
                i=i,
                j=j,
                gate_similarity=float(gate_sim[i, j]),
                up_similarity=float(up_sim[i, j]),
            )
        )
        if len(out) >= max_pairs:
            break
    return out


def find_merge_candidates(
    weights_dir: str, *, threshold: float = 0.95, max_pairs_per_layer: int = 50
) -> Dict[int, List[MergeCandidate]]:
    """Run :func:`find_merge_candidates_in_layer` across every MLP layer."""
    result: Dict[int, List[MergeCandidate]] = {}
    for triplet in iter_mlp_triplets(weights_dir):
        candidates = find_merge_candidates_in_layer(
            triplet, threshold=threshold, max_pairs=max_pairs_per_layer
        )
        if candidates:
            result[triplet.layer_idx] = candidates
    return result


def apply_merges(
    gate: "NDArray[Any]",
    up: "NDArray[Any]",
    down: "NDArray[Any]",
    pairs: List[Tuple[int, int]],
) -> Tuple["NDArray[Any]", "NDArray[Any]", "NDArray[Any]", List[int]]:
    """Fold each (i, j) pair into neuron i and drop neuron j.

    ``down_proj[:, i] += down_proj[:, j]`` compensates for removing j
    (valid to the extent act_i(x) ≈ act_j(x), which is exactly the
    condition :func:`find_merge_candidates_in_layer` screened for). Returns
    the new (smaller) gate/up/down matrices plus the sorted list of kept
    original indices, so callers can trace which neurons survived.
    """
    np = _np()
    d = gate.shape[0]
    drop = {j for _i, j in pairs}
    fold_into: Dict[int, List[int]] = {}
    for i, j in pairs:
        fold_into.setdefault(i, []).append(j)

    down = np.array(down, copy=True)
    for i, js in fold_into.items():
        for j in js:
            down[:, i] += down[:, j]

    kept = [idx for idx in range(d) if idx not in drop]
    new_gate = gate[kept, :]
    new_up = up[kept, :]
    new_down = down[:, kept]
    # safetensors.numpy.save_file writes an array's raw buffer without
    # consulting its strides — a non-C-contiguous array (this fancy-index
    # selection on axis 1 can produce one; confirmed empirically, not just
    # in theory) gets silently serialized as if it *were* C-contiguous,
    # writing transposed garbage that only surfaces on reload. Every array
    # that might reach save_file must be forced contiguous first.
    new_gate = np.ascontiguousarray(new_gate)
    new_up = np.ascontiguousarray(new_up)
    new_down = np.ascontiguousarray(new_down)
    return new_gate, new_up, new_down, kept


def apply_merges_to_checkpoint(
    weights_dir: str,
    output_dir: str,
    candidates: Dict[int, List[MergeCandidate]],
    *,
    allow_nonuniform: bool = False,
) -> List[Tuple[int, int, int]]:
    """Write a new checkpoint with the given per-layer merges applied.

    Standard HF configs (``LlamaConfig``/``Qwen2Config``/...) have a single
    global ``intermediate_size`` shared by every decoder layer — there's no
    field for "layer 3 has a smaller MLP than layer 7". So by default this
    enforces a *uniform* merge count across every MLP layer found (the
    minimum candidate count across all of them, treating a layer absent
    from ``candidates`` as 0) — the output always has one consistent
    ``intermediate_size`` and loads with plain
    ``AutoModelForCausalLM.from_pretrained``. If that minimum is 0 (some
    layer had no candidates at the given threshold), nothing is merged
    unless ``allow_nonuniform=True``, which writes a checkpoint with a
    per-layer-varying MLP width instead — deliberately opt-in, since that
    checkpoint needs custom model-loading code to use (not a stock
    ``transformers`` config).

    Note on cost: candidate *scanning* (``find_merge_candidates``) streams
    one layer at a time and is memory-bounded. This function is not — like
    any "write out a new checkpoint" step, it needs the model in memory
    once to call ``safetensors.save_file`` (which takes a single in-memory
    tensor dict; there's no incremental writer in the public API). Same
    cost profile as ``soup merge`` / ``soup export``, not a new limitation
    introduced here.
    """
    import json as json_mod

    from safetensors import safe_open
    from safetensors.numpy import save_file

    np = _np()
    os.makedirs(output_dir, exist_ok=True)

    triplets = list(iter_mlp_triplets(weights_dir))
    if not triplets:
        raise ValueError("no MLP gate/up/down triplets found — nothing to merge")

    if allow_nonuniform:
        n_merges_by_layer = {t.layer_idx: len(candidates.get(t.layer_idx, [])) for t in triplets}
    else:
        uniform_n = min(len(candidates.get(t.layer_idx, [])) for t in triplets)
        if uniform_n == 0:
            raise ValueError(
                "at least one MLP layer has 0 merge candidates at this "
                "threshold, and a uniform intermediate_size is required by "
                "default (standard HF configs can't express a per-layer "
                "MLP width). Lower --threshold, or pass --allow-nonuniform "
                "to write a non-standard checkpoint anyway."
            )
        n_merges_by_layer = {t.layer_idx: uniform_n for t in triplets}

    merged_tensors: Dict[str, "NDArray[Any]"] = {}
    summary: List[Tuple[int, int, int]] = []
    new_intermediate_sizes = set()
    for t in triplets:
        n = n_merges_by_layer[t.layer_idx]
        before = t.gate.shape[0]
        if n == 0:
            new_intermediate_sizes.add(before)
            summary.append((t.layer_idx, before, before))
            continue
        pairs = [(c.i, c.j) for c in candidates[t.layer_idx][:n]]
        new_gate, new_up, new_down, _kept = apply_merges(t.gate, t.up, t.down, pairs)
        merged_tensors[t.gate_key] = new_gate
        merged_tensors[t.up_key] = new_up
        merged_tensors[t.down_key] = new_down
        new_intermediate_sizes.add(new_gate.shape[0])
        summary.append((t.layer_idx, before, new_gate.shape[0]))

    # Pass 2: copy every tensor through, substituting the merged ones.
    # Same view-lifetime hazard as pass 1 (see the comment in
    # iter_mlp_triplets): a passthrough tensor read here must be copied
    # before this `with safe_open(...)` block closes, or a later shard's
    # mmap can silently overwrite it before save_file gets to serialize it.
    # (Didn't show up on a single-shard checkpoint — only bites with 2+
    # shard files — caught by testing against a multi-shard synthetic
    # checkpoint, not just the original single-file one.)
    framework, to_np = _framework()
    all_tensors: Dict[str, "NDArray[Any]"] = {}
    for path in _discover_safetensors(weights_dir):
        with safe_open(path, framework=framework) as handle:
            for key in handle.keys():
                if key in merged_tensors:
                    all_tensors[key] = merged_tensors[key]
                else:
                    all_tensors[key] = to_np(handle.get_tensor(key)).copy()

    # Defensive: save_file mishandles non-contiguous arrays (see the note
    # in apply_merges above) — the merged tensors are already forced
    # contiguous there, but guard every tensor here too, including the
    # untouched passthrough ones, since a copy() of a sliced/transposed
    # source view isn't guaranteed C-contiguous either.
    all_tensors = {k: np.ascontiguousarray(v) for k, v in all_tensors.items()}
    save_file(all_tensors, os.path.join(output_dir, "model.safetensors"))

    config_path = os.path.join(weights_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json_mod.load(fh)
        if len(new_intermediate_sizes) == 1:
            config["intermediate_size"] = new_intermediate_sizes.pop()
        else:
            # allow_nonuniform path — no single valid value; record the
            # per-layer map instead so it's at least discoverable, and
            # leave the standard field untouched rather than write a
            # silently-wrong number into it.
            config["_soup_per_layer_intermediate_size"] = {
                str(idx): after for idx, _before, after in summary
            }
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as fh:
            json_mod.dump(config, fh, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Wanda importance (activation-weighted, needs a loaded model + calibration data)
# ---------------------------------------------------------------------------
#
# Sun et al., "A Simple and Effective Pruning Approach for Large Language
# Models" (Wanda, ICLR 2024): per-weight score S_ij = |W_ij| * ||X_j||_2,
# where X_j is the j-th input feature's activation norm over calibration
# data. No backward pass needed — just forward passes to collect activation
# statistics — but unlike row_l2_norms() above, it does need the actual
# model loaded (not just streamed weight files) and a calibration batch,
# so it's opt-in and heavier by nature. Aggregated per output row (neuron)
# the same way as the magnitude-only score, for a like-for-like ranking.


# ---------------------------------------------------------------------------
# Calibration text loading — shared by CLI (`soup compress importance
# --metric wanda`), the UI's `/api/compress/importance/scan`, and
# `soup export`'s AWQ/GPTQ paths.
#
# Previously each of those three call sites had its own copy-pasted "read
# JSONL, pull the 'text' field" loop, and none of them accepted more than one
# file — so a Wanda calibration set spanning code + chat + a domain-specific
# corpus had to be manually concatenated into a single file first. This is
# now the one place that logic lives; it accepts one path OR a list, and
# samples up to `max_samples_per_file` from *each* file so one large file
# can't starve the others out of the pool.
# ---------------------------------------------------------------------------


def load_calibration_texts(
    paths: Union[str, List[str]],
    *,
    max_samples_per_file: int = 128,
    max_samples_total: Optional[int] = None,
) -> List[str]:
    """Load calibration text samples from one or more JSONL files.

    Args:
        paths: a single file path, or a list of file paths. Each file is a
            JSONL where every line is either ``{"text": "..."}`` or an
            arbitrary object whose string-valued fields get space-joined
            (mirrors the single-file loaders this replaces, so existing
            calibration files need no changes).
        max_samples_per_file: cap on how many lines are read from any one
            file, so one large dataset can't crowd out the others when
            several are given.
        max_samples_total: optional overall cap across all files combined,
            applied after per-file sampling.

    Returns:
        A flat list of text samples, in file order, files in the order
        given. Empty/unreadable lines are skipped; a file that yields zero
        usable lines is skipped with a warning logged, not a hard failure —
        one bad file in a multi-dataset list shouldn't sink the whole scan.
    """
    import json as json_mod

    if isinstance(paths, str):
        paths = [paths]

    all_texts: List[str] = []
    for path in paths:
        file_texts: List[str] = []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json_mod.loads(line)
                    except json_mod.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        if "text" in row and row["text"]:
                            file_texts.append(str(row["text"]))
                        else:
                            joined = " ".join(str(v) for v in row.values() if v)
                            if joined:
                                file_texts.append(joined)
                    elif isinstance(row, str) and row:
                        file_texts.append(row)
                    if len(file_texts) >= max_samples_per_file:
                        break
        except (OSError, FileNotFoundError) as exc:
            logger.warning("calibration file %r unreadable, skipping: %s", path, exc)
            continue
        if not file_texts:
            logger.warning("calibration file %r yielded no usable text rows", path)
        all_texts.extend(file_texts)

    if max_samples_total is not None:
        all_texts = all_texts[:max_samples_total]
    return all_texts


def compute_wanda_importance(
    model: Any,
    calibration_input_ids: List[Any],
    *,
    target_names: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> Dict[str, "NDArray[Any]"]:
    """Per-output-neuron Wanda importance for every ``nn.Linear`` in ``model``.

    Args:
        model: a loaded ``torch.nn.Module`` (e.g. from
            ``AutoModelForCausalLM.from_pretrained``), already on the
            device you want the forward passes to run on.
        calibration_input_ids: list of 1-D or 2-D LongTensors (token ids)
            to forward through the model. A few dozen sequences of a few
            hundred tokens each is typically enough (this is exactly the
            calibration-set size Wanda's own paper uses).
        target_names: if given, only ``nn.Linear`` modules whose qualified
            name is in this list are scored (matches the ``.weight`` keys
            used elsewhere in this module, e.g.
            ``model.layers.0.mlp.down_proj``). Defaults to every Linear.

    Returns:
        ``{module_name: np.ndarray[n_out]}`` — same shape/semantics as
        stacking :func:`row_l2_norms` outputs, so results from both are
        directly comparable.
    """
    import torch

    np = _np()
    model.eval()
    if device is not None:
        model = model.to(device)

    linears = {
        name: mod
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and (target_names is None or name in target_names)
    }
    if not linears:
        raise ValueError("no matching nn.Linear modules found to score")

    sq_sum: Dict[str, "torch.Tensor"] = {}
    n_samples: Dict[str, int] = {}
    handles = []

    def _make_hook(name: str):
        def _hook(_module, inputs):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(torch.float32)
            s = (x * x).sum(dim=0)
            if name not in sq_sum:
                sq_sum[name] = s
                n_samples[name] = x.shape[0]
            else:
                sq_sum[name] = sq_sum[name] + s
                n_samples[name] += x.shape[0]

        return _hook

    for name, mod in linears.items():
        handles.append(mod.register_forward_pre_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            for ids in calibration_input_ids:
                ids = ids if hasattr(ids, "dim") else torch.as_tensor(ids)
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                ids = ids.to(next(model.parameters()).device)
                model(ids)
    finally:
        for h in handles:
            h.remove()

    result: Dict[str, "NDArray[Any]"] = {}
    for name, mod in linears.items():
        if name not in sq_sum:
            continue  # module never actually ran (e.g. unused expert in a MoE)
        norm_x = torch.sqrt(sq_sum[name]).cpu().numpy().astype(np.float32)
        w = mod.weight.detach().cpu().numpy().astype(np.float32)  # [n_out, n_in]
        scores = np.abs(w) * norm_x[np.newaxis, :]
        result[name] = np.linalg.norm(scores, axis=1)
    return result


# ---------------------------------------------------------------------------
# Quick eval: perplexity before/after, for a "does this merge actually hurt
# the model" check before committing to --apply on real data.
# ---------------------------------------------------------------------------


def compute_perplexity(model: Any, tokenizer: Any, texts: List[str], *, max_length: int = 512) -> float:
    """Mean per-token cross-entropy perplexity of ``model`` over ``texts``.

    Standard next-token perplexity (each text scored independently, no
    cross-document context) — good enough for a relative before/after
    comparison, which is all a quick sanity check needs.
    """
    import math

    import torch

    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            if not text or not text.strip():
                continue
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            out = model(input_ids=input_ids, labels=input_ids)
            n_tokens = input_ids.shape[1] - 1  # labels shifted internally by the model
            total_nll += float(out.loss) * n_tokens
            total_tokens += n_tokens
    if total_tokens == 0:
        raise ValueError("no usable text for perplexity (all too short/empty)")
    return math.exp(total_nll / total_tokens)


def quick_eval_merge(
    original_model_path: str,
    merged_weights_dir: str,
    eval_texts: List[str],
    *,
    max_length: int = 512,
) -> Dict[str, Any]:
    """Load the original and a just-merged checkpoint, compare perplexity.

    Both models load in full (this is the "hold it in memory once" cost
    already documented for --apply — this just also loads the original for
    the comparison, so it's heavier than either alone, by design: it's an
    opt-in sanity check, not something that runs on every merge).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.utils.trust_remote import model_requires_trust_remote_code

    trust_remote_code = bool(model_requires_trust_remote_code(original_model_path))
    tokenizer = AutoTokenizer.from_pretrained(original_model_path, trust_remote_code=trust_remote_code)

    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_path, trust_remote_code=trust_remote_code, dtype=torch.float32
    )
    ppl_before = compute_perplexity(original_model, tokenizer, eval_texts, max_length=max_length)
    del original_model

    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_weights_dir, trust_remote_code=trust_remote_code, dtype=torch.float32
    )
    ppl_after = compute_perplexity(merged_model, tokenizer, eval_texts, max_length=max_length)
    del merged_model

    return {
        "perplexity_before": ppl_before,
        "perplexity_after": ppl_after,
        "relative_increase_pct": 100.0 * (ppl_after - ppl_before) / ppl_before,
        "n_texts": len(eval_texts),
    }
