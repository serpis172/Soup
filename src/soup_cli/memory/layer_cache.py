"""Bounded, direction-aware read-ahead prefetcher for resident decoder-layer
weights.

v0.73.5 — per user request, simplified from a reactive LRU *cache* to a pure
*prefetcher*. Layer streaming with gradient checkpointing visits layers in a
fully deterministic zigzag: forward 0->n-1, then backward recompute n-1->0,
repeating every step (see trainer/stream_setup.py). Given that, hit/miss/
eviction cache bookkeeping was solving a problem this workload doesn't have.
What actually helps is read-ahead: one background thread that keeps loading
the *next* layers in whichever direction training is currently moving,
bounded by how many layers fit in the configured RAM budget, and drains
itself as the consumer advances — no separate eviction policy to reason
about.
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
    """Clamp a requested RAM-prefetch budget to what the box can safely give.

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


class LayerPrefetcher:
    """Read-ahead window over decoder layers, filled by one background thread.

    Usage:
      1. Construct with a RAM budget (GB).
      2. Optionally call ``set_bounds(n_layers)`` once the shard's layer
         count is known, so the walk knows where to stop (works fine
         without it too — an out-of-range guess just fails ``load_fn`` once
         and that direction is skipped until the bound self-corrects).
      3. Call ``get_layer(idx, load_fn)`` for every layer access, in
         whatever order training actually visits them. The background
         thread infers direction from consecutive calls and keeps the
         window filled ahead of the consumer in that direction.

    Single background thread for the process lifetime of this object — no
    thread-per-call, no per-key locks. The only synchronization is one
    lock/condition shared between that thread and the (single) consumer
    thread that calls ``get_layer``.
    """

    def __init__(self, max_ram_gb: float, *, notify: Optional[Callable[[str], None]] = None):
        max_ram_gb = clamp_ram_cache_gb(max_ram_gb, notify=notify)
        self.max_bytes = int(max_ram_gb * (1024**3))
        self.n_layers: Optional[int] = None

        self.window: "collections.OrderedDict[int, tuple]" = collections.OrderedDict()
        self.current_bytes = 0
        self._avg_layer_bytes = 0

        self._last_idx: Optional[int] = None
        self._direction = 1

        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self._load_fn: Optional[Callable[[int], Dict[str, Any]]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def set_bounds(self, n_layers: int) -> None:
        with self.lock:
            self.n_layers = n_layers
            self.cv.notify_all()

    def _ensure_thread(self, load_fn: Callable[[int], Dict[str, Any]]) -> None:
        with self.lock:
            if self._running:
                return
            self._load_fn = load_fn
            self._running = True
            self._thread = threading.Thread(
                target=self._prefetch_loop, name="soup-layer-prefetch", daemon=True
            )
            self._thread.start()

    def get_layer(self, idx: int, load_fn: Callable[[int], Dict[str, Any]]) -> Dict[str, Any]:
        if self.max_bytes <= 0:
            return load_fn(idx)

        self._ensure_thread(load_fn)
        with self.lock:
            entry = self.window.pop(idx, None)
            if entry is not None:
                data, _size = entry
                self._advance(idx)
                self.cv.notify_all()
                return data

        # Miss: load synchronously — the caller is never blocked longer than
        # a direct read would take anyway — then let the background thread
        # pick up prefetching from here.
        data = load_fn(idx)
        with self.lock:
            self._advance(idx)
            self.cv.notify_all()
        return data

    def _advance(self, idx: int) -> None:
        """Update direction and drop window entries now behind the consumer.

        Must be called with ``self.lock`` held.
        """
        if self._last_idx is not None:
            if idx == self._last_idx - 1:
                self._direction = -1
            elif idx == self._last_idx + 1:
                self._direction = 1
            # any other jump (first call, or a non-adjacent index) keeps the
            # existing direction guess — a wrong guess only costs a few
            # wasted background loads, never correctness.
        self._last_idx = idx

        stale = [
            k for k in self.window
            if (self._direction == 1 and k <= idx) or (self._direction == -1 and k >= idx)
        ]
        for k in stale:
            _data, size = self.window.pop(k)
            self.current_bytes -= size

    def _next_target(self) -> Optional[int]:
        """Must be called with ``self.lock`` held."""
        if self.max_bytes <= 0 or self._last_idx is None:
            return None
        if self._avg_layer_bytes and self.current_bytes + self._avg_layer_bytes > self.max_bytes:
            return None

        probe = self._last_idx
        # Bounded scan: at most one past the window size, since the window
        # already only holds distinct not-yet-consumed indices ahead of
        # _last_idx in the current direction.
        for _ in range(len(self.window) + 1):
            probe += self._direction
            if probe < 0:
                return None
            if self.n_layers is not None and probe >= self.n_layers:
                return None
            if probe not in self.window:
                return probe
        return None

    def _prefetch_loop(self) -> None:
        assert self._load_fn is not None
        while True:
            with self.cv:
                while self._running and self._next_target() is None:
                    self.cv.wait(timeout=1.0)
                if not self._running:
                    return
                target = self._next_target()
            if target is None:
                continue
            try:
                data = self._load_fn(target)
            except Exception:
                # Out of range / shard missing / whatever — this was a
                # forward guess past a boundary we didn't know yet. Learn
                # the bound so we stop retrying it, and keep going;
                # get_layer's synchronous fallback still covers correctness
                # if this guess was ever needed for real.
                logger.debug("layer prefetch: idx=%s failed, treating as boundary", target)
                with self.lock:
                    if self.n_layers is None:
                        self.n_layers = target if self._direction == 1 else target + 1
                continue
            size = self._get_size(data)
            with self.lock:
                if target not in self.window and self.current_bytes + size <= self.max_bytes:
                    self.window[target] = (data, size)
                    self.current_bytes += size
                    self._avg_layer_bytes = size

    @staticmethod
    def _get_size(obj: Any) -> int:
        if not hasattr(obj, "values"):
            return 0
        return sum(getattr(t, "numel", lambda: 0)() * getattr(t, "element_size", lambda: 0)() for t in obj.values())

    def stats(self) -> Dict[str, Any]:
        """Snapshot for the Web UI monitoring panel."""
        with self.lock:
            return {
                "capacity_bytes": self.max_bytes,
                "used_bytes": self.current_bytes,
                "layers_queued": len(self.window),
                "direction": self._direction,
                "utilization_pct": (
                    round(100 * self.current_bytes / self.max_bytes, 1)
                    if self.max_bytes > 0
                    else 0.0
                ),
            }

    def close(self) -> None:
        """Stop the background thread. Not required — it's a daemon thread
        and dies with the process — but tidy for long-lived Python sessions
        (e.g. the Web UI) that create/discard trainers repeatedly.
        """
        with self.lock:
            self._running = False
            self.cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# Backwards-compatible alias — anything importing the old cache-flavoured
# name keeps working; the class itself is now a pure prefetcher.
RamLayerCache = LayerPrefetcher
