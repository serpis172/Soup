"""Bounded, thread-safe LRU cache for resident decoder-layer weights.

This sits behind ``RamSource`` (see ``utils/layer_stream_runtime.py``) when
``training.ram_cache_gb`` is set: instead of holding every layer resident in
pinned RAM (the plain "ram" tier) or reloading from disk shards every step
(the "disk" tier), a bounded LRU subset of layers is kept warm.

Design notes (post code-review, v0.73.4):
- No background "read ahead" thread here. ``StreamPrefetcher`` in
  ``layer_stream_runtime.py`` already drives ahead-of-need prefetching by
  calling ``RamSource.get()`` for the next layer before it is needed; a
  second self-driven prefetch loop in this class only duplicated that work,
  raced on ``current_size`` accounting, and could spawn one unbounded
  thread per cache miss with no cap and no shutdown hook.
- ``max_ram_gb`` is clamped against *actually available* system RAM (via
  psutil) with a safety margin, so a value typed into the UI slider can't
  silently ask for more than the box has and get the OOM killer involved
  instead of a clear error.
"""

from __future__ import annotations

import collections
import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Leave this fraction of *available* RAM alone even if the user asks for
# more — the OS, the training process's own Python/CUDA host allocations,
# and the dataloader workers all need headroom too.
_AVAILABLE_RAM_SAFETY_MARGIN = 0.85


def clamp_ram_cache_gb(
    requested_gb: float, *, notify: Optional[Callable[[str], None]] = None
) -> float:
    """Clamp a requested RAM-cache budget to what the box can safely give.

    Returns the (possibly reduced) GB value to actually use. Falls back to
    the requested value unmodified if ``psutil`` is unavailable rather than
    failing training over a monitoring dependency.
    """
    if requested_gb <= 0:
        return 0.0
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:  # pragma: no cover - psutil missing/platform quirk
        logger.warning("psutil unavailable; ram_cache_gb not clamped to available RAM")
        return requested_gb

    safe_ceiling_gb = available_gb * _AVAILABLE_RAM_SAFETY_MARGIN
    if requested_gb > safe_ceiling_gb:
        msg = (
            f"training.ram_cache_gb={requested_gb:.1f} exceeds the safe "
            f"ceiling ({safe_ceiling_gb:.1f} GB of {available_gb:.1f} GB "
            f"available); clamping to {safe_ceiling_gb:.1f} GB."
        )
        if notify:
            notify(msg)
        else:
            logger.warning(msg)
        return max(safe_ceiling_gb, 0.0)
    return requested_gb


class RamLayerCache:
    """Bounded LRU cache mapping layer index -> {tensor_name: Tensor}."""

    def __init__(self, max_ram_gb: float, *, notify: Optional[Callable[[str], None]] = None):
        max_ram_gb = clamp_ram_cache_gb(max_ram_gb, notify=notify)
        self.max_ram_bytes = int(max_ram_gb * (1024**3))
        self.current_size = 0
        # value = (layer_dict, size_bytes) — size stored alongside the data
        # so eviction/accounting never has to recompute or drift.
        self.cache: "collections.OrderedDict[int, tuple]" = collections.OrderedDict()
        self.lock = threading.Lock()

    def get_layer(self, layer_id: int, load_fn: Callable[[int], Dict[str, Any]]) -> Dict[str, Any]:
        """Return a layer from cache, loading it via ``load_fn`` on a miss."""
        with self.lock:
            entry = self.cache.get(layer_id)
            if entry is not None:
                self.cache.move_to_end(layer_id)
                return entry[0]

        # Cache miss — load outside the lock so a slow disk read on one
        # layer doesn't block hits on every other layer.
        layer_data = load_fn(layer_id)
        if self.max_ram_bytes > 0:
            self._add_to_cache(layer_id, layer_data)
        return layer_data

    def _add_to_cache(self, layer_id: int, data: Dict[str, Any]) -> None:
        with self.lock:
            if layer_id in self.cache:
                # Another thread loaded the same layer concurrently (race
                # on the miss above) — keep the existing entry, just bump
                # its recency. Don't double-count its size.
                self.cache.move_to_end(layer_id)
                return
            data_size = self._get_size(data)
            while self.current_size + data_size > self.max_ram_bytes and self.cache:
                _, (_evicted, evicted_size) = self.cache.popitem(last=False)
                self.current_size -= evicted_size
            self.cache[layer_id] = (data, data_size)
            self.current_size += data_size

    @staticmethod
    def _get_size(obj: Any) -> int:
        """Approximate resident size in bytes of a {name: Tensor} layer dict."""
        if not hasattr(obj, "values"):
            return 0
        return sum(getattr(t, "numel", lambda: 0)() * getattr(t, "element_size", lambda: 0)() for t in obj.values())

    def stats(self) -> Dict[str, Any]:
        """Snapshot for the Web UI monitoring panel."""
        with self.lock:
            return {
                "capacity_bytes": self.max_ram_bytes,
                "used_bytes": self.current_size,
                "layers_cached": len(self.cache),
                "utilization_pct": (
                    round(100 * self.current_size / self.max_ram_bytes, 1)
                    if self.max_ram_bytes > 0
                    else 0.0
                ),
            }
