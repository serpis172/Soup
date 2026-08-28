"""FastAPI application for Soup Web UI."""

import json as json_mod
import logging
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Max file read size to prevent memory exhaustion
_MAX_INSPECT_LIMIT = 500


class TrainRequest(PydanticBaseModel):
    """Request body for starting a training run."""
    config_yaml: str


class TrainStatus(PydanticBaseModel):
    """Current training process status."""
    running: bool
    pid: Optional[int] = None
    config_path: Optional[str] = None
    paused: bool = False
    phase: Optional[str] = None
    total_steps: Optional[int] = None
    exit_code: Optional[int] = None


class DataInspectRequest(PydanticBaseModel):
    """Request body for data inspection."""
    path: str
    limit: int = Field(default=50, ge=1, le=_MAX_INSPECT_LIMIT)


# Global state for training process
_train_process: Optional[subprocess.Popen] = None
_train_config_path: Optional[str] = None
_train_lock = threading.Lock()
_train_paused = False
_train_total_steps_estimate: Optional[int] = None

# Shared, in-memory log buffer for the current training run + the thread
# that fills it.
#
# Bug fix (this session): /api/train/logs used to iterate `proc.stdout`
# DIRECTLY, only while an SSE client happened to be connected. `stdout` was
# piped (subprocess.PIPE) but never otherwise read — the OS pipe buffer is
# small (~64KB on Linux), so a training run producing more than that much
# stdout with nobody watching the log page would block on its next write
# and hang silently, forever, indistinguishable from "still training" in
# /api/train/status (poll() stays None). This background thread is the one
# and only reader of proc.stdout for the lifetime of the process — started
# unconditionally in /api/train/start, whether or not anyone ever opens the
# logs SSE stream — so the pipe can never back up. /api/train/logs and the
# phase-detection in /api/train/progress both read from `_train_log_buffer`
# instead of the pipe directly.
_train_log_buffer: List[str] = []
_train_log_lock = threading.Lock()
_train_log_thread: Optional[threading.Thread] = None

# Coarse phase labels inferred from known log-line substrings the trainer
# prints at each stage (see commands/train.py) — checked in order, most
# recently matched wins. "Training" is NOT in this table: it's detected
# from the experiment tracker actually having a metrics row for the run,
# which is a far more reliable signal than matching log text for it.
_PHASE_MARKERS = [
    ("Loading config from", "Loading config"),
    ("Verifying datasets", "Verifying datasets"),
    ("Loading dataset", "Loading dataset"),
    ("Setting up model + trainer", "Setting up model + trainer"),
    ("Loaded:", "Dataset loaded"),
]


def _drain_training_stdout(proc: subprocess.Popen) -> None:
    """Background thread body: continuously read `proc.stdout` into
    `_train_log_buffer` for the lifetime of the process. See the module-level
    comment above `_train_log_buffer` for why this must run unconditionally.
    """
    try:
        for raw_line in proc.stdout:
            text = (
                raw_line.decode("utf-8", errors="replace")
                if isinstance(raw_line, bytes)
                else raw_line
            ).rstrip("\n\r")
            with _train_log_lock:
                _train_log_buffer.append(text)
    except (ValueError, OSError):
        pass


def _detect_phase(log_tail: List[str]) -> Optional[str]:
    """Best-effort phase label from the most recent matching marker in
    `log_tail`. Scans newest-first so a later stage's marker always wins
    over an earlier one still sitting in the tail."""
    for line in reversed(log_tail):
        for marker, label in _PHASE_MARKERS:
            if marker in line:
                return label
    return None


def _estimate_total_steps(config) -> Optional[int]:
    """Best-effort `epochs * (train_rows / effective_batch_size)` estimate,
    for a determinate progress bar / ETA in the UI. Returns None (shown as
    "unknown" in the UI — step count and speed still display without a
    percentage) whenever it can't be computed cheaply and reliably:

    - `training.batch_size == "auto"` — the real batch size is only known
      once the trainer's runtime GPU-memory probing picks one.
    - any `data.train` source that isn't a local, existing `.jsonl` file —
      HF dataset names and remote URIs would need a network fetch just to
      count rows, which this best-effort estimate shouldn't trigger as a
      side effect of clicking "Start Training".
    """
    try:
        tcfg = config.training
        if not isinstance(tcfg.batch_size, int):
            return None

        train_sources = config.data.train
        if isinstance(train_sources, str):
            train_sources = [train_sources]

        total_rows = 0
        for src in train_sources:
            p = Path(src)
            if p.suffix != ".jsonl" or not p.exists():
                return None
            with open(p, encoding="utf-8") as fh:
                total_rows += sum(1 for line in fh if line.strip())
        if total_rows == 0:
            return None

        if not config.data.val:
            total_rows = int(total_rows * (1.0 - config.data.val_split))
        if total_rows <= 0:
            return None

        effective_batch = max(1, tcfg.batch_size * tcfg.gradient_accumulation_steps)
        steps_per_epoch = max(1, total_rows // effective_batch)
        return steps_per_epoch * tcfg.epochs
    except Exception:
        # Best-effort only — any unexpected shape (custom config subclass,
        # future field changes) degrades to "unknown", never a 500.
        return None


# Auth token generated at startup — printed to console for the user.
# Reads/writes go through `_auth_token_lock` so token rotation never
# leaves a window where some requests see the old value and some the new.
_auth_token: str = secrets.token_urlsafe(32)
_auth_token_lock = threading.Lock()


def get_auth_token() -> str:
    """Return the current auth token (for printing at startup)."""
    with _auth_token_lock:
        return _auth_token


def set_auth_token(token: str) -> None:
    """Replace the process-wide auth token (used by `soup ui --auth-token`).

    Validates via `utils.qr_url.validate_token` so a malformed override
    can't bypass the urlsafe-base64 shape check.
    """
    from soup_cli.utils.qr_url import validate_token

    validated = validate_token(token)
    global _auth_token
    with _auth_token_lock:
        _auth_token = validated


def create_app(host: str = "127.0.0.1", port: int = 7860):
    """Create the Soup Web UI FastAPI application."""
    from fastapi import Depends, FastAPI, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Soup Web UI", version="1.0.0")

    # Restrict CORS to the origin we actually serve. When `host == "0.0.0.0"`
    # the literal `http://0.0.0.0:<port>` is never a browser origin, so we
    # allow loopback origins AND the same-LAN regex shape. The Bearer
    # token is the actual security gate on mutating endpoints.
    if host == "0.0.0.0":
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=(
                r"^https?://("
                r"localhost|127\.0\.0\.1|"
                r"10\.\d+\.\d+\.\d+|"
                r"192\.168\.\d+\.\d+|"
                r"172\.(?:1[6-9]|2[0-9]|3[01])\.\d+\.\d+"
                r")(:\d+)?$"
            ),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )
    else:
        allowed_origin = f"http://{host}:{port}"
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[allowed_origin],
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    def _verify_token(request: Request):
        """Verify Bearer token on mutating endpoints."""
        auth = request.headers.get("Authorization", "")
        with _auth_token_lock:
            expected = f"Bearer {_auth_token}"
        # Constant-time compare — a plain != leaks the token byte-by-byte via
        # response timing when `soup ui --public` is exposed on a LAN.
        if not secrets.compare_digest(auth, expected):
            raise HTTPException(status_code=401, detail="Unauthorized")

    # --- Static files ---

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # --- Runs API ---

    @app.get("/api/runs")
    def list_runs(limit: int = Query(default=50, ge=1, le=500)):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            runs = tracker.list_runs(limit=limit)
            return {"runs": runs}
        finally:
            tracker.close()

    @app.get("/api/runs/compare")
    def compare_runs(ids: str = Query(default="")):
        """Compare metrics for multiple runs."""
        from soup_cli.experiment.tracker import ExperimentTracker

        if not ids or not ids.strip():
            raise HTTPException(status_code=400, detail="ids parameter required")

        run_ids = [rid.strip() for rid in ids.split(",") if rid.strip()]
        if len(run_ids) > 5:
            raise HTTPException(
                status_code=400, detail="Maximum 5 runs per comparison"
            )
        if not run_ids:
            raise HTTPException(status_code=400, detail="ids parameter required")

        tracker = ExperimentTracker()
        try:
            result = []
            for rid in run_ids:
                run_info = tracker.get_run(rid)
                metrics = tracker.get_metrics(rid)
                config = {}
                if run_info and run_info.get("config_json"):
                    try:
                        config = json_mod.loads(run_info["config_json"])
                    except (ValueError, TypeError):
                        pass
                result.append({
                    "run_id": rid,
                    "config": config,
                    "metrics": metrics,
                })
            return {"runs": result}
        finally:
            tracker.close()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            run = tracker.get_run(run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            return run
        finally:
            tracker.close()

    @app.get("/api/runs/{run_id}/metrics")
    def get_run_metrics(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            run = tracker.get_run(run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            metrics = tracker.get_metrics(run_id)
            return {"run_id": run_id, "metrics": metrics}
        finally:
            tracker.close()

    @app.delete("/api/runs/{run_id}", dependencies=[Depends(_verify_token)])
    def delete_run(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            deleted = tracker.delete_run(run_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Run not found")
            return {"deleted": True, "run_id": run_id}
        finally:
            tracker.close()

    @app.get("/api/runs/{run_id}/eval")
    def get_run_eval(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            results = tracker.get_eval_results(run_id=run_id)
            return {"run_id": run_id, "eval_results": results}
        finally:
            tracker.close()

    # --- GPU / System Info ---

    @app.get("/api/system")
    def system_info():
        from soup_cli import __version__
        from soup_cli.utils.gpu import detect_device, get_gpu_info

        device, device_name = detect_device()
        gpu_info = get_gpu_info()
        return {
            "version": __version__,
            "device": device,
            "device_name": device_name,
            "gpu_info": gpu_info,
            "python_version": sys.version.split()[0],
        }

    # --- Templates ---

    @app.get("/api/templates")
    def list_templates():
        from soup_cli.config.schema import TEMPLATES

        return {"templates": {name: yaml_str for name, yaml_str in TEMPLATES.items()}}

    # --- Config Validation ---

    @app.post("/api/config/validate", dependencies=[Depends(_verify_token)])
    def validate_config(body: dict):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = body.get("yaml", "")
        if not yaml_str:
            raise HTTPException(status_code=400, detail="Empty config")
        try:
            config = load_config_from_string(yaml_str)
            return {"valid": True, "config": config.model_dump()}
        except Exception as exc:
            return {"valid": False, "error": str(exc)}

    # --- Training ---

    @app.post("/api/train/start", dependencies=[Depends(_verify_token)])
    def start_training(req: TrainRequest):
        global _train_process, _train_config_path, _train_log_thread, _train_paused
        global _train_total_steps_estimate

        with _train_lock:
            if _train_process and _train_process.poll() is None:
                raise HTTPException(
                    status_code=409, detail="Training already in progress"
                )

            # Validate config before writing to disk
            from soup_cli.config.loader import load_config_from_string

            try:
                config = load_config_from_string(req.config_yaml)
            except Exception as exc:
                # Bug fix (this session): this used to collapse every
                # validation failure into a bare "Invalid training
                # configuration" with the real reason only in the server
                # log — the person clicking Start Training had no way to
                # see WHAT was wrong (missing required field, bad
                # base/data.train type, an incompatible objectives/task
                # combination, etc.) without going and reading server
                # console output they likely don't have open. Mirrors what
                # /api/config/validate already does with the same
                # exception (pydantic's message is structural — field
                # names/expected types — not sensitive).
                logger.warning("Invalid training config: %s", exc)
                raise HTTPException(status_code=400, detail=str(exc))
            _train_total_steps_estimate = _estimate_total_steps(config)

            # Securely-created temp file. A FIXED name in the shared temp dir
            # let a local attacker pre-place a symlink there and redirect this
            # write; mkstemp creates a fresh O_EXCL file (no symlink following,
            # unpredictable name).
            fd, config_path = tempfile.mkstemp(
                prefix="soup_ui_config_", suffix=".yaml"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(req.config_yaml)

            _train_config_path = config_path
            _train_paused = False
            _train_process = subprocess.Popen(
                [sys.executable, "-m", "soup_cli", "train", "--config", config_path, "--yes"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            with _train_log_lock:
                _train_log_buffer.clear()
            # Unconditional drain thread — see _train_log_buffer's module
            # comment for why this can't be "only when someone's watching".
            _train_log_thread = threading.Thread(
                target=_drain_training_stdout, args=(_train_process,), daemon=True
            )
            _train_log_thread.start()
            return {"started": True, "pid": _train_process.pid}

    @app.get("/api/train/status")
    def train_status():
        global _train_process
        with _train_lock:
            proc = _train_process
            paused = _train_paused
        if proc is None:
            return TrainStatus(running=False)
        poll = proc.poll()
        if poll is None:
            with _train_log_lock:
                tail = list(_train_log_buffer[-50:])
            phase = "Paused" if paused else (_detect_phase(tail) or "Starting")
            return TrainStatus(
                running=True,
                pid=proc.pid,
                config_path=_train_config_path,
                paused=paused,
                phase=phase,
            )
        return TrainStatus(running=False, pid=proc.pid, exit_code=poll)

    @app.post("/api/train/stop", dependencies=[Depends(_verify_token)])
    def stop_training():
        global _train_process, _train_paused
        with _train_lock:
            if _train_process and _train_process.poll() is None:
                if _train_paused:
                    # A SIGSTOP'd process can't act on SIGTERM until resumed
                    # first — send SIGCONT so terminate() actually takes
                    # effect instead of leaving a stopped zombie-in-place.
                    try:
                        os.kill(_train_process.pid, signal.SIGCONT)
                    except OSError:
                        pass
                    _train_paused = False
                _train_process.terminate()
                return {"stopped": True}
            return {"stopped": False, "detail": "No training in progress"}

    @app.post("/api/train/pause", dependencies=[Depends(_verify_token)])
    def pause_training():
        """Pause the training subprocess via SIGSTOP (POSIX).

        This suspends the OS process — no gradient step is left half-applied,
        since the kernel simply stops scheduling the process, mid-instruction
        boundary is irrelevant at this granularity. It is NOT the same as a
        trainer-level checkpoint-and-resume: GPU memory (model, optimizer
        state, activations) stays allocated for the whole pause, so this
        frees compute, not VRAM. For releasing VRAM, stop the run properly
        and resume from a checkpoint instead (training.resume_from).
        """
        global _train_paused
        with _train_lock:
            if not _train_process or _train_process.poll() is not None:
                raise HTTPException(status_code=409, detail="No training in progress")
            if _train_paused:
                return {"paused": True, "detail": "Already paused"}
            if not hasattr(signal, "SIGSTOP"):
                raise HTTPException(
                    status_code=501,
                    detail="Pause is unavailable on this OS (needs POSIX SIGSTOP/SIGCONT).",
                )
            try:
                os.kill(_train_process.pid, signal.SIGSTOP)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not pause: {exc}")
            _train_paused = True
            return {"paused": True}

    @app.post("/api/train/resume", dependencies=[Depends(_verify_token)])
    def resume_training():
        """Resume a training subprocess previously paused via `/api/train/pause`."""
        global _train_paused
        with _train_lock:
            if not _train_process or _train_process.poll() is not None:
                raise HTTPException(status_code=409, detail="No training in progress")
            if not _train_paused:
                return {"paused": False, "detail": "Not paused"}
            try:
                os.kill(_train_process.pid, signal.SIGCONT)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not resume: {exc}")
            _train_paused = False
            return {"paused": False}

    # --- Data Inspection ---

    @app.post("/api/data/inspect", dependencies=[Depends(_verify_token)])
    def inspect_data(req: DataInspectRequest):
        from soup_cli.data.loader import load_raw_data
        from soup_cli.utils.paths import is_under_cwd

        # Path traversal protection. Use realpath + commonpath containment
        # (is_under_cwd) — the old str.startswith check let a sibling like
        # ".../project-secrets" pass as under ".../project".
        try:
            resolved = Path(req.path).resolve()
        except (ValueError, OSError):
            raise HTTPException(status_code=400, detail="Invalid path")

        if not is_under_cwd(req.path):
            raise HTTPException(
                status_code=403, detail="Access denied: path outside working directory"
            )

        if not resolved.exists():
            raise HTTPException(status_code=404, detail="File not found")

        try:
            raw_data = load_raw_data(resolved)
        except Exception as exc:
            logger.warning("Data inspect error: %s", exc)
            raise HTTPException(status_code=400, detail="Failed to load data file")

        total = len(raw_data)
        sample = raw_data[: req.limit]

        # Detect format
        from soup_cli.data.formats import detect_format

        fmt = detect_format(raw_data[:5]) if raw_data else "unknown"

        # Basic stats
        keys = set()
        for entry in sample:
            keys.update(entry.keys())

        return {
            "path": str(resolved),
            "total": total,
            "format": fmt,
            "keys": sorted(keys),
            "sample": sample,
        }

    # --- Training Live Monitor (SSE) ---

    @app.get("/api/train/logs")
    def stream_training_logs(request: Request):
        """SSE endpoint streaming training log lines in real time.

        Reads from `_train_log_buffer` (filled by the unconditional drain
        thread started in /api/train/start), not `proc.stdout` directly —
        see that buffer's module-level comment. Polls the buffer instead of
        blocking on pipe reads, so multiple concurrent SSE clients (and the
        phase detector in /api/train/progress) can all read the same
        buffered history independently without racing each other for lines.
        """
        from fastapi.responses import StreamingResponse

        last_event_id = request.headers.get("Last-Event-ID")
        skip_count = 0
        if last_event_id and last_event_id.isdigit():
            skip_count = int(last_event_id) + 1

        def _generate_log_events():
            with _train_lock:
                proc = _train_process
            if proc is None:
                yield "event: done\ndata: {}\n\n"
                return

            line_index = skip_count
            while True:
                with _train_log_lock:
                    new_lines = _train_log_buffer[line_index:]
                for text in new_lines:
                    data = json_mod.dumps({"line": text, "id": line_index})
                    yield f"id: {line_index}\ndata: {data}\n\n"
                    line_index += 1

                with _train_lock:
                    still_running = proc.poll() is None
                if not still_running:
                    # One more drain pass in case a few lines landed in the
                    # buffer between the poll() above and the thread's last
                    # write, then close out.
                    with _train_log_lock:
                        trailing = _train_log_buffer[line_index:]
                    for text in trailing:
                        data = json_mod.dumps({"line": text, "id": line_index})
                        yield f"id: {line_index}\ndata: {data}\n\n"
                        line_index += 1
                    yield "event: done\ndata: {}\n\n"
                    return

                time.sleep(0.4)

        return StreamingResponse(
            _generate_log_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/train/metrics/live")
    def stream_live_metrics(
        request: Request,
        run_id: Optional[str] = Query(default=None),
    ):
        """SSE endpoint streaming new metrics as they're logged."""
        from fastapi.responses import StreamingResponse

        def _generate_metrics_events():
            from soup_cli.experiment.tracker import ExperimentTracker

            with _train_lock:
                proc = _train_process
            if proc is None and run_id is None:
                yield "event: done\ndata: {}\n\n"
                return

            last_step = -1
            max_polls = 3  # For tests: limit poll cycles when process done
            polls_since_new = 0

            while True:
                tracker = ExperimentTracker()
                try:
                    if run_id:
                        metrics = tracker.get_metrics(run_id)
                    else:
                        yield "event: done\ndata: {}\n\n"
                        return
                finally:
                    tracker.close()

                new_metrics = [
                    m for m in metrics if m.get("step", 0) > last_step
                ]
                if new_metrics:
                    for m_row in new_metrics:
                        data = json_mod.dumps(m_row, default=str)
                        yield f"data: {data}\n\n"
                    last_step = max(
                        m.get("step", 0) for m in new_metrics
                    )
                    polls_since_new = 0
                else:
                    polls_since_new += 1

                # Check if training is still running
                with _train_lock:
                    proc = _train_process
                if proc is None or proc.poll() is not None:
                    if polls_since_new >= 1:
                        yield "event: done\ndata: {}\n\n"
                        return

                # Yield heartbeat
                yield ":heartbeat\n\n"

                if polls_since_new >= max_polls:
                    yield "event: done\ndata: {}\n\n"
                    return

                time.sleep(0.1)  # Short poll for tests

        return StreamingResponse(
            _generate_metrics_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/train/progress")
    def train_progress(
        run_id: Optional[str] = Query(default=None),
    ):
        """Return current training progress snapshot: step/loss/speed from
        the experiment tracker, a best-effort total_steps/eta_seconds (see
        `_estimate_total_steps`), the coarse phase label, and pause state.
        """
        with _train_lock:
            proc = _train_process
            paused = _train_paused
        is_running = proc is not None and proc.poll() is None

        with _train_log_lock:
            tail = list(_train_log_buffer[-50:])
        phase = "Paused" if (is_running and paused) else _detect_phase(tail)

        base = {
            "running": is_running,
            "paused": paused,
            "phase": phase,
            "total_steps": _train_total_steps_estimate,
        }

        if not is_running and run_id is None:
            return {**base, "current_step": 0, "run_id": None}

        # Auto-discover run_id from the tracked subprocess's PID when the
        # caller doesn't already know it — the New Training page's progress
        # poller doesn't (nothing in the frontend previously tracked a
        # run_id at all), so without this, step/loss/speed/eta could never
        # populate for it even though the tracker had them the whole time.
        if run_id is None and is_running and proc is not None:
            from soup_cli.experiment.tracker import ExperimentTracker

            tracker = ExperimentTracker()
            try:
                run_id = tracker.find_run_by_pid(proc.pid)
            finally:
                tracker.close()

        if run_id:
            from soup_cli.experiment.tracker import ExperimentTracker

            tracker = ExperimentTracker()
            try:
                metrics = tracker.get_metrics(run_id)
                latest = metrics[-1] if metrics else None
            finally:
                tracker.close()

            current_step = latest["step"] if latest else 0
            speed = latest.get("speed") if latest else None
            eta_seconds = None
            if (
                speed and speed > 0
                and _train_total_steps_estimate
                and _train_total_steps_estimate > current_step
            ):
                eta_seconds = (_train_total_steps_estimate - current_step) / speed
            if current_step > 0:
                phase = "Training"

            return {
                **base,
                "phase": phase,
                "current_step": current_step,
                "run_id": run_id,
                "loss": latest.get("loss") if latest else None,
                "speed": speed,
                "eta_seconds": eta_seconds,
            }

        return {**base, "current_step": 0, "run_id": None}

    # --- Config Builder ---

    @app.get("/api/config/schema")
    def config_schema():
        """Return config schema as JSON for form generation."""
        from soup_cli.config.schema import (
            DataConfig,
            LoraConfig,
            SoupConfig,
            TrainingConfig,
        )

        def _extract_field_info(model_cls):
            """Extract field metadata from a Pydantic model."""
            result = {}
            for name, field_info in model_cls.model_fields.items():
                info = {"type": "string", "required": field_info.is_required()}

                # Get default value
                if field_info.default is not None:
                    info["default"] = field_info.default

                # Get type annotation
                annotation = field_info.annotation
                if annotation is not None:
                    ann_str = str(annotation)
                    if "int" in ann_str:
                        info["type"] = "integer"
                    elif "float" in ann_str:
                        info["type"] = "number"
                    elif "bool" in ann_str:
                        info["type"] = "boolean"

                    # Check for Literal (enum) types
                    args = getattr(annotation, "__args__", None)
                    if args:
                        # Filter out NoneType for Optional[Literal[...]]
                        non_none = [a for a in args if a is not type(None)]
                        if non_none and all(isinstance(a, str) for a in non_none):
                            info["type"] = "enum"
                            info["options"] = list(non_none)

                # Get constraints from metadata
                for meta in (field_info.metadata or []):
                    if hasattr(meta, "ge"):
                        info["ge"] = meta.ge
                    if hasattr(meta, "le"):
                        info["le"] = meta.le

                result[name] = info
            return result

        schema = _extract_field_info(SoupConfig)
        schema["data"] = _extract_field_info(DataConfig)
        schema["training"] = _extract_field_info(TrainingConfig)
        schema["training"]["lora"] = _extract_field_info(LoraConfig)
        return schema

    @app.get("/api/recipes")
    def list_recipes():
        """Return recipe catalog as JSON."""
        from soup_cli.recipes.catalog import RECIPES

        recipes_list = []
        for name, meta in RECIPES.items():
            recipes_list.append({
                "name": name,
                "model": meta.model,
                "task": meta.task,
                "description": meta.description,
                "tags": list(meta.tags) if hasattr(meta, "tags") else [],
                "yaml": meta.yaml_str,
            })
        return {"recipes": recipes_list}

    @app.post("/api/config/from-form", dependencies=[Depends(_verify_token)])
    def form_to_yaml(body: dict):
        """Convert form field values to validated YAML string."""
        import yaml

        from soup_cli.config.loader import load_config_from_string

        # Build YAML from form values
        config_dict = {}
        for key, val in body.items():
            if val is not None and val != "" and val != {}:
                config_dict[key] = val

        try:
            yaml_str = yaml.dump(
                config_dict, default_flow_style=False, sort_keys=False
            )
            # Validate
            load_config_from_string(yaml_str)
            return {"yaml": yaml_str}
        except (ValueError, TypeError) as exc:
            # Bug fix (this session): same "swallowed real error" pattern
            # as /api/train/start had — see that endpoint's comment.
            logger.warning("Config form validation error: %s", exc)
            return {"error": str(exc)}

    # --- Chat Proxy ---

    class ChatMessage(PydanticBaseModel):
        """A single chat message."""
        role: str
        content: str

    class ChatRequest(PydanticBaseModel):
        """Request body for chat send."""
        messages: list[ChatMessage]
        endpoint: str
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        max_tokens: int = Field(default=512, ge=1, le=16384)
        top_p: float = Field(default=0.9, ge=0.0, le=1.0)
        adapter: Optional[str] = None

    @app.post("/api/chat/send", dependencies=[Depends(_verify_token)])
    def chat_send(req: ChatRequest):
        """SSE proxy endpoint streaming chat completions."""
        from urllib.parse import urlparse

        from fastapi.responses import StreamingResponse

        # Validate messages
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        # SSRF protection: localhost-only HTTP, HTTPS for remote
        parsed = urlparse(req.endpoint)
        if parsed.scheme == "http":
            import ipaddress as _ipaddr

            host = parsed.hostname or ""
            is_local = host in ("localhost", "0.0.0.0")
            if not is_local:
                try:
                    addr = _ipaddr.ip_address(host)
                    is_local = addr.is_loopback
                except ValueError:
                    is_local = False
            if not is_local:
                raise HTTPException(
                    status_code=400,
                    detail="HTTP only allowed for localhost endpoints",
                )
        elif parsed.scheme != "https":
            raise HTTPException(
                status_code=400,
                detail="Only HTTP (localhost) or HTTPS endpoints allowed",
            )

        # Validate bounds
        if req.max_tokens > 16384:
            raise HTTPException(
                status_code=400, detail="max_tokens exceeds 16384 cap"
            )
        if req.temperature < 0.0 or req.temperature > 2.0:
            raise HTTPException(
                status_code=400, detail="temperature must be 0.0-2.0"
            )
        if req.top_p < 0.0 or req.top_p > 1.0:
            raise HTTPException(
                status_code=400, detail="top_p must be 0.0-1.0"
            )

        def _stream_chat():
            import httpx

            url = req.endpoint.rstrip("/") + "/v1/chat/completions"
            payload = {
                "messages": [m.model_dump() for m in req.messages],
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
                "stream": True,
            }
            if req.adapter:
                payload["model"] = req.adapter

            try:
                with httpx.stream(
                    "POST", url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=120.0,
                ) as resp:
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                yield "data: {\"done\": true}\n\n"
                                return
                            try:
                                parsed_data = json_mod.loads(data_str)
                                delta = (
                                    parsed_data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if delta:
                                    out = json_mod.dumps({"delta": delta})
                                    yield f"data: {out}\n\n"
                            except (ValueError, IndexError, KeyError):
                                pass
                yield "data: {\"done\": true}\n\n"
            except Exception as exc:
                logger.warning("Chat proxy error: %s", exc)
                err_msg = json_mod.dumps(
                    {"error": "Connection failed"}
                )
                yield f"data: {err_msg}\n\n"

        return StreamingResponse(
            _stream_chat(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # --- v0.53.9 #94: SSE training-event stream ---

    @app.get("/api/train/stream")
    async def stream_train_events():
        """SSE endpoint streaming `TrainEvent` payloads as JSON frames.

        Per-subscriber cursor — multiple concurrent listeners each receive
        every event (no destructive drain). Uses `asyncio.sleep` so the
        uvicorn async loop is not blocked under default workers.
        """
        import asyncio

        from fastapi.responses import StreamingResponse

        from soup_cli.utils.sse_train_stream import TrainEvent, format_sse_frame
        from soup_cli.utils.train_event_buffer import get_global_buffer

        buffer = get_global_buffer()

        async def _gen():
            # Start from cursor 0 — new subscribers receive a bounded
            # catch-up of retained events (deque maxlen=1000) before
            # streaming fresh ones. Concurrent subscribers are independent.
            cursor = 0
            max_ticks = 200  # cap to keep test runs bounded; ~20s at 100ms
            empty_ticks = 0
            for _ in range(max_ticks):
                events, cursor = buffer.snapshot_since(cursor)
                if events:
                    empty_ticks = 0
                    for event in events:
                        yield format_sse_frame(event)
                else:
                    empty_ticks += 1
                    yield ":heartbeat\n\n"
                with _train_lock:
                    proc = _train_process
                if proc is None or proc.poll() is not None:
                    if empty_ticks >= 1:
                        done = TrainEvent(type="status", message="done")
                        yield format_sse_frame(done)
                        return
                await asyncio.sleep(0.1)
            done = TrainEvent(type="status", message="timeout")
            yield format_sse_frame(done)

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # --- v0.53.9 #100: Tool-call observation panel ---

    @app.get("/api/tool-outputs")
    def list_tool_outputs(
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        """Return the most recent tool-call records as JSON.

        Records are pushed by the SFT trainer's tool-calling callback
        into the process-wide `ToolOutputsBuffer`. Read-only; safe for
        cross-origin polling.
        """
        from soup_cli.utils.tool_outputs import get_global_tool_buffer

        records = get_global_tool_buffer().snapshot(limit=limit)
        return {
            "count": len(records),
            "records": [
                {
                    "name": r.name,
                    "started_ts": r.started_ts,
                    "duration_ms": r.duration_ms,
                    "success": r.success,
                    "output_preview": r.output_preview,
                    "error": r.error,
                }
                for r in records
            ],
        }

    # --- Health ---

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # --- System RAM (for the RAM-cache slider bound) ---

    @app.get("/api/system/ram")
    def system_ram():
        """Total/available host RAM in GB, for sizing the RAM-cache slider.

        Mirrors the same clamp `RamLayerCache` applies at training time
        (soup_cli.memory.layer_cache.clamp_ram_cache_gb) so the slider's max
        matches what the trainer will actually allow, instead of letting the
        user drag it past a value that gets silently reduced later.
        """
        try:
            import psutil

            vm = psutil.virtual_memory()
            from soup_cli.memory.layer_cache import _AVAILABLE_RAM_SAFETY_MARGIN

            return {
                "total_gb": round(vm.total / (1024**3), 1),
                "available_gb": round(vm.available / (1024**3), 1),
                "safe_max_gb": round(
                    vm.available / (1024**3) * _AVAILABLE_RAM_SAFETY_MARGIN, 1
                ),
            }
        except ImportError:
            return {"total_gb": None, "available_gb": None, "safe_max_gb": None}

    # --- Lightweight health banner for the Dashboard ---
    # Deliberately independent of `soup doctor` (commands/doctor.py) rather
    # than reusing its checks: those print Rich Panels directly instead of
    # returning structured data, so wiring them into a JSON endpoint would
    # mean refactoring a mature, already-relied-upon CLI command — not
    # worth the risk for a banner. `soup doctor` remains the authoritative,
    # detailed check; this is a few fast, safe signals for "is anything
    # obviously wrong before you start a run".

    @app.get("/api/system/health")
    def system_health():
        issues = []
        try:
            import torch

            if not torch.cuda.is_available() and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                issues.append("No GPU detected (CPU-only) — training will be slow.")
        except ImportError:
            issues.append("PyTorch not installed — run: pip install -e \".[train]\"")

        try:
            import shutil

            free_gb = shutil.disk_usage(".").free / (1024**3)
            if free_gb < 10:
                issues.append(f"Low disk space: {free_gb:.1f} GB free in the working directory.")
        except OSError:
            pass

        try:
            import psutil

            if psutil.virtual_memory().available / (1024**3) < 4:
                issues.append("Less than 4 GB RAM available.")
        except ImportError:
            pass

        return {"ok": len(issues) == 0, "issues": issues}

    # --- Live resource snapshot (Dashboard polls this every few seconds) ---

    @app.get("/api/system/live")
    def system_live():
        snapshot: Dict[str, Any] = {"cpu_pct": None, "ram_pct": None, "gpu": []}
        try:
            import psutil

            snapshot["cpu_pct"] = psutil.cpu_percent(interval=None)
            snapshot["ram_pct"] = psutil.virtual_memory().percent
        except ImportError:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    total = torch.cuda.get_device_properties(idx).total_memory
                    reserved = torch.cuda.memory_reserved(idx)
                    snapshot["gpu"].append(
                        {
                            "index": idx,
                            "name": torch.cuda.get_device_name(idx),
                            "memory_used_gb": round(reserved / (1024**3), 2),
                            "memory_total_gb": round(total / (1024**3), 2),
                            "memory_pct": round(100 * reserved / total, 1) if total else 0,
                        }
                    )
        except ImportError:
            pass
        return snapshot

    # --- Streaming / quantization config patch ---
    #
    # Bug fix (this session): these used to be ONE panel + ONE endpoint
    # (`/api/config/patch-training` patching both `ram_cache_gb` and
    # `custom_quant_*` in a single request). Layer RAM prefetch and
    # post-training quantization are unrelated settings — one tunes how the
    # frozen base streams during training, the other tunes how the finished
    # checkpoint is exported — so they're now two cards with two endpoints.
    # `_patch_training_yaml` is the shared helper both route through so the
    # "load → patch → re-validate → dump" logic isn't duplicated.

    def _patch_training_yaml(yaml_text: str, patch: Dict[str, Any]) -> str:
        """Apply `patch` onto a config's `training:` block and return the
        updated, re-validated YAML string. Shared by the RAM-prefetch and
        quantization patch endpoints below.
        """
        import yaml as yaml_mod

        from soup_cli.config.loader import load_config_from_string

        try:
            doc = yaml_mod.safe_load(yaml_text) or {}
        except yaml_mod.YAMLError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")
        if not isinstance(doc, dict):
            raise HTTPException(status_code=400, detail="Config root must be a mapping")

        training = doc.get("training")
        if not isinstance(training, dict):
            training = {}
        for key, value in patch.items():
            if value is None:
                continue
            if value == "":
                training.pop(key, None)
            else:
                training[key] = value
        doc["training"] = training

        try:
            new_yaml = yaml_mod.dump(doc, default_flow_style=False, sort_keys=False)
            load_config_from_string(new_yaml)  # validate before handing back
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Resulting config is invalid: {exc}")
        return new_yaml

    class RamPrefetchPatchRequest(PydanticBaseModel):
        """Patch just `training.ram_cache_gb` in a YAML config string."""
        yaml: str
        ram_cache_gb: float = Field(ge=0.0)

    @app.post("/api/config/patch-ram-prefetch", dependencies=[Depends(_verify_token)])
    def patch_ram_prefetch_config(req: RamPrefetchPatchRequest):
        """Set `training.ram_cache_gb` — the "Layer RAM Prefetch" card's own
        endpoint, separate from quantization (see comment above).
        """
        new_yaml = _patch_training_yaml(req.yaml, {"ram_cache_gb": req.ram_cache_gb})
        return {"yaml": new_yaml}

    class QuantPatchRequest(PydanticBaseModel):
        """Patch `training.custom_quant_strategy` / `custom_quant_detail`."""
        yaml: str
        custom_quant_strategy: str
        custom_quant_detail: Optional[str] = None

    @app.post("/api/config/patch-quant", dependencies=[Depends(_verify_token)])
    def patch_quant_config(req: QuantPatchRequest):
        """Set `training.custom_quant_strategy` / `custom_quant_detail` — the
        "Model Quantization" card's own endpoint, separate from RAM prefetch.
        """
        new_yaml = _patch_training_yaml(
            req.yaml,
            {
                "custom_quant_strategy": req.custom_quant_strategy,
                "custom_quant_detail": req.custom_quant_detail or "",
            },
        )
        return {"yaml": new_yaml}

    class TrainingPatchRequest(PydanticBaseModel):
        """Patch a subset of `training:` fields in a YAML config string.

        Deprecated (kept for backward compatibility with any external caller
        already using it): prefer `/api/config/patch-ram-prefetch` and
        `/api/config/patch-quant`, which the UI now calls instead.
        """
        yaml: str
        ram_cache_gb: Optional[float] = Field(default=None, ge=0.0)
        custom_quant_strategy: Optional[str] = None
        custom_quant_detail: Optional[str] = None

    @app.post("/api/config/patch-training", dependencies=[Depends(_verify_token)])
    def patch_training_config(req: TrainingPatchRequest):
        """Deprecated combined patch endpoint — see `TrainingPatchRequest`."""
        new_yaml = _patch_training_yaml(
            req.yaml,
            {
                "ram_cache_gb": req.ram_cache_gb,
                "custom_quant_strategy": req.custom_quant_strategy,
                "custom_quant_detail": req.custom_quant_detail,
            },
        )
        return {"yaml": new_yaml}

    # --- HuggingFace Hub: search + download (models / datasets / benchmarks) ---
    #
    # Search is read-only (GET, no auth — same tier as /api/templates).
    # Download writes to disk, so it requires the Bearer token like every
    # other mutating endpoint.

    def _hf_build_filter(
        library: Optional[str], license_: Optional[str], language: Optional[str] = None
    ) -> Optional[list]:
        """Build a Hub tag-filter list. `language` used to only be wired for
        dataset search (list_datasets has a native `language=` kwarg) — the
        Models tab showed the same Language field but silently dropped it,
        since list_models has no equivalent native kwarg. HF Hub models
        that declare a language do so via a `language:<code>` tag though,
        same mechanism as library/license here, so this now works for
        models too instead of only appearing to.
        """
        tags = []
        if library:
            tags.append(f"library:{library}")
        if license_:
            tags.append(f"license:{license_}")
        if language:
            tags.append(f"language:{language}")
        return tags or None

    def _hf_model_card(info) -> dict:
        return {
            "id": info.id,
            "author": info.author,
            "pipeline_tag": info.pipeline_tag,
            "library_name": info.library_name,
            "downloads": info.downloads,
            "likes": info.likes,
            "gated": bool(info.gated),
            "last_modified": str(info.last_modified) if info.last_modified else None,
            "tags": list(info.tags or [])[:12],
        }

    def _hf_dataset_card(info) -> dict:
        return {
            "id": info.id,
            "author": info.author,
            "downloads": info.downloads,
            "likes": info.likes,
            "gated": bool(info.gated),
            "last_modified": str(info.last_modified) if info.last_modified else None,
            "tags": list(info.tags or [])[:12],
        }

    @app.get("/api/fs/browse", dependencies=[Depends(_verify_token)])
    def browse_filesystem(
        path: str = Query(default="."),
        extensions: Optional[str] = Query(
            default=None, description="comma-separated, e.g. .jsonl,.json — files only, dirs always shown"
        ),
    ):
        """List directories (and matching files, if `extensions` given)
        under `path` — backs the New Training page's dataset/model/output
        Browse buttons (this session), so paths can be picked from a real
        listing instead of typed/remembered exactly.
        """
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        try:
            p = p.resolve()
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=f"Cannot resolve path: {exc}")
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {p}")
        if not p.is_dir():
            p = p.parent  # browsing a file path -> show its containing directory

        ext_filter = None
        if extensions:
            ext_filter = {e.strip().lower() for e in extensions.split(",") if e.strip()}

        entries = []
        try:
            children = sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower()))
        except PermissionError:
            raise HTTPException(status_code=403, detail=f"Permission denied: {p}")
        for child in children:
            if child.name.startswith("."):
                continue  # dotfiles/dirs are noise for this picker
            try:
                if child.is_dir():
                    entries.append({"name": child.name, "type": "dir", "path": str(child)})
                elif ext_filter is None or child.suffix.lower() in ext_filter:
                    entries.append({
                        "name": child.name, "type": "file", "path": str(child),
                        "size": child.stat().st_size,
                    })
            except OSError:
                continue  # broken symlink or similar — skip, don't fail the whole listing

        return {
            "path": str(p),
            "parent": str(p.parent) if p.parent != p else None,
            "entries": entries,
        }

    @app.get("/api/calculator/model-size")
    def calculator_model_size(model: str = Query(...)):
        """Best-effort parameter count (in billions) for the Calculator
        section, reusing the same logic `soup`'s own GPU-sizing helpers use
        (utils.gpu.model_size_from_name): reads a local checkpoint's actual
        config.json when `model` is a local path, otherwise parses the
        parameter count out of the model name (e.g. "...-8B-..." -> 8.0) —
        a heuristic, not exact, for anything not already downloaded.
        """
        from soup_cli.utils.gpu import model_size_from_name

        try:
            params_b = model_size_from_name(model)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"model": model, "params_b": params_b}

    @app.get("/api/hf/models/search")
    def hf_search_models(
        q: Optional[str] = Query(default=None),
        task: Optional[str] = Query(default=None, description="pipeline_tag, e.g. text-generation"),
        library: Optional[str] = Query(default=None, description="e.g. transformers, peft, gguf"),
        language: Optional[str] = Query(default=None, description="e.g. en, it, multilingual"),
        license: Optional[str] = Query(default=None),  # noqa: A002 - HF's own vocabulary
        sort: str = Query(default="downloads"),
        limit: int = Query(default=30, ge=1, le=100),
    ):
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError

        if sort not in ("downloads", "likes", "trending_score", "created_at", "last_modified"):
            raise HTTPException(status_code=400, detail="invalid sort")
        try:
            results = HfApi().list_models(
                search=q or None,
                pipeline_tag=task or None,
                filter=_hf_build_filter(library, license, language),
                sort=sort,
                limit=limit,
            )
            return {"results": [_hf_model_card(m) for m in results]}
        except HfHubHTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Hugging Face Hub error: {exc}")

    @app.get("/api/hf/datasets/search")
    def hf_search_datasets(
        q: Optional[str] = Query(default=None),
        task: Optional[str] = Query(default=None, description="task_categories, e.g. question-answering"),
        language: Optional[str] = Query(default=None),
        license: Optional[str] = Query(default=None),  # noqa: A002
        benchmark_only: bool = Query(default=False, description="Only datasets tagged as benchmarks"),
        sort: str = Query(default="downloads"),
        limit: int = Query(default=30, ge=1, le=100),
    ):
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError

        if sort not in ("downloads", "likes", "trending_score", "created_at", "last_modified"):
            raise HTTPException(status_code=400, detail="invalid sort")
        try:
            results = HfApi().list_datasets(
                search=q or None,
                task_categories=task or None,
                language=language or None,
                filter=_hf_build_filter(None, license),
                benchmark=True if benchmark_only else None,
                sort=sort,
                limit=limit,
            )
            return {"results": [_hf_dataset_card(d) for d in results]}
        except HfHubHTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Hugging Face Hub error: {exc}")

    # In-memory download job registry. Best-effort status only (queued /
    # downloading / done / error) — snapshot_download doesn't expose clean
    # mid-transfer cancellation, and per-file byte progress would need a
    # tqdm_class shim; skipped for now (ponytail: ship the version that
    # answers "is it done / did it fail", add byte-level progress if that
    # turns out not to be enough).
    _hf_jobs: Dict[str, Dict[str, Any]] = {}
    _hf_jobs_lock = threading.Lock()

    class HfDownloadRequest(PydanticBaseModel):
        repo_id: str
        repo_type: str = "model"  # "model" | "dataset"
        revision: Optional[str] = None
        local_dir: Optional[str] = None

    def _hf_run_download(job_id: str, req: "HfDownloadRequest", target_dir: str):
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError

        with _hf_jobs_lock:
            _hf_jobs[job_id]["status"] = "downloading"
        try:
            snapshot_download(
                repo_id=req.repo_id,
                repo_type=req.repo_type,
                revision=req.revision,
                local_dir=target_dir,
            )
            with _hf_jobs_lock:
                _hf_jobs[job_id]["status"] = "done"
                _hf_jobs[job_id]["finished_ts"] = time.time()
        except (HfHubHTTPError, OSError, ValueError) as exc:
            with _hf_jobs_lock:
                _hf_jobs[job_id]["status"] = "error"
                _hf_jobs[job_id]["error"] = str(exc)
                _hf_jobs[job_id]["finished_ts"] = time.time()

    @app.post("/api/hf/download", dependencies=[Depends(_verify_token)])
    def hf_download(req: HfDownloadRequest):
        import re

        from soup_cli.utils.paths import is_under_cwd

        if req.repo_type not in ("model", "dataset"):
            raise HTTPException(status_code=400, detail="repo_type must be 'model' or 'dataset'")
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$", req.repo_id):
            raise HTTPException(status_code=400, detail="repo_id must look like 'org/name'")

        subdir = "models" if req.repo_type == "model" else "datasets"
        default_dir = str(Path(subdir) / req.repo_id.replace("/", "__"))
        target_dir = req.local_dir or default_dir
        if not is_under_cwd(target_dir):
            raise HTTPException(
                status_code=403, detail="local_dir must stay inside the working directory"
            )

        job_id = secrets.token_urlsafe(8)
        with _hf_jobs_lock:
            _hf_jobs[job_id] = {
                "job_id": job_id,
                "repo_id": req.repo_id,
                "repo_type": req.repo_type,
                "local_dir": target_dir,
                "status": "queued",
                "error": None,
                "started_ts": time.time(),
                "finished_ts": None,
            }
        threading.Thread(
            target=_hf_run_download, args=(job_id, req, target_dir), daemon=True
        ).start()
        return {"job_id": job_id, "local_dir": target_dir}

    @app.get("/api/hf/download/jobs")
    def hf_download_jobs():
        with _hf_jobs_lock:
            return {"jobs": sorted(_hf_jobs.values(), key=lambda j: j["started_ts"], reverse=True)}

    # --- Quantization format catalog (real registry, not hardcoded in JS) ---

    @app.get("/api/quant/formats")
    def quant_formats():
        """Real quantization choices per strategy, sourced from
        utils/gguf_quant.py's registry — the same one config/schema.py
        validates training.custom_quant_detail against — so the UI can
        never offer a value the backend would then reject.

        Bug fix: this used to offer `[4, 8]` for BOTH awq and gptq. AWQ is a
        4-bit-only method (autoawq has no 8-bit kernel — offering 8 here just
        set the user up for a later export-time failure), and GPTQ actually
        supports 2/3/4/8-bit but only 4/8 were ever exposed. Both are now
        sourced from the same domain config/schema.py's
        `_validate_custom_quant_detail` enforces, so this list and that
        validator cannot drift apart again.
        """
        from soup_cli.utils.gguf_quant import STANDARD_GGUF_QUANT_TYPES, _GGUF_METADATA

        return {
            "awq": {
                "kind": "bits",
                "options": [4],
                "note": "AWQ supports only 4-bit (autoawq library limitation).",
            },
            "gptq": {"kind": "bits", "options": [2, 3, 4, 8]},
            "k-quants": {
                "kind": "named",
                "options": [
                    {"name": s.name, "bits": s.bits, "description": s.description}
                    for s in sorted(STANDARD_GGUF_QUANT_TYPES.values(), key=lambda s: s.bits)
                ],
            },
            "i-quants": {
                "kind": "named",
                "options": [
                    {"name": s.name, "bits": s.bits, "description": s.description}
                    for s in sorted(_GGUF_METADATA.values(), key=lambda s: s.bits)
                ],
            },
        }

    # --- Model density tools: importance ranking + neuron merging ---
    # Same background-job pattern as HF downloads above (scanning/merging a
    # real model's weights can take real time — never block the request).

    _compress_jobs: Dict[str, Dict[str, Any]] = {}
    _compress_jobs_lock = threading.Lock()

    def _new_compress_job(kind: str, extra: dict) -> str:
        job_id = secrets.token_urlsafe(8)
        with _compress_jobs_lock:
            _compress_jobs[job_id] = {
                "job_id": job_id,
                "kind": kind,
                "status": "running",
                "error": None,
                "result": None,
                "started_ts": time.time(),
                "finished_ts": None,
                **extra,
            }
        return job_id

    def _finish_compress_job(job_id: str, *, result=None, error=None) -> None:
        with _compress_jobs_lock:
            job = _compress_jobs[job_id]
            job["status"] = "error" if error else "done"
            job["error"] = error
            job["result"] = result
            job["finished_ts"] = time.time()

    class ImportanceScanRequest(PydanticBaseModel):
        model: str
        modules: str = "mlp,attn"
        bottom_k: int = Field(default=10, ge=1, le=200)
        metric: str = "magnitude"  # "magnitude" | "wanda"
        calibration_texts: Optional[List[str]] = None
        # Bug fix / feature (this session): the Wanda calibration set used to
        # be pasted-text-only (`calibration_texts`), so drawing calibration
        # samples from real dataset files — let alone more than one of them —
        # meant copy-pasting text into the UI by hand. This lets the panel
        # point at one or more JSONL dataset paths instead; both can be
        # combined (dataset-loaded texts are appended after any pasted ones).
        calibration_dataset_paths: Optional[List[str]] = None
        calibration_samples_per_dataset: int = Field(default=64, ge=1, le=2000)
        max_length: int = Field(default=512, ge=16, le=4096)

    def _run_importance_scan(job_id: str, req: "ImportanceScanRequest"):
        try:
            if req.metric == "wanda":
                from soup_cli.utils.neuron_compress import load_calibration_texts

                texts: List[str] = list(req.calibration_texts or [])
                if req.calibration_dataset_paths:
                    texts.extend(
                        load_calibration_texts(
                            req.calibration_dataset_paths,
                            max_samples_per_file=req.calibration_samples_per_dataset,
                        )
                    )
                if not texts:
                    raise ValueError(
                        "metric=wanda requires calibration_texts and/or "
                        "calibration_dataset_paths"
                    )
                from soup_cli.utils.neuron_compress import rank_importance_wanda

                results = rank_importance_wanda(
                    req.model, texts, modules=req.modules, max_length=req.max_length
                )
            else:
                from soup_cli.utils.neuron_compress import rank_importance
                from soup_cli.utils.spectrum_scan import resolve_model_weights

                weights_dir = resolve_model_weights(req.model)
                results = rank_importance(weights_dir, modules=req.modules)

            payload = [
                {
                    "param_name": r.param_name,
                    "group": r.group,
                    "module_type": r.module_type,
                    "n_neurons": r.n_neurons,
                    "least_important": [
                        {"index": i, "norm": v} for i, v in r.least_important(req.bottom_k)
                    ],
                }
                for r in results
            ]
            _finish_compress_job(job_id, result={"layers": payload, "metric": req.metric})
        except Exception as exc:  # background thread — never let this die silently
            _finish_compress_job(job_id, error=f"{type(exc).__name__}: {exc}")

    @app.post("/api/compress/importance/scan", dependencies=[Depends(_verify_token)])
    def compress_importance_scan(req: ImportanceScanRequest):
        job_id = _new_compress_job("importance", {"model": req.model})
        threading.Thread(target=_run_importance_scan, args=(job_id, req), daemon=True).start()
        return {"job_id": job_id}

    class NeuronScanRequest(PydanticBaseModel):
        model: str
        threshold: float = Field(default=0.98, ge=0.0, le=1.0)
        max_pairs_per_layer: int = Field(default=50, ge=1, le=1000)

    def _run_neuron_scan(job_id: str, req: "NeuronScanRequest"):
        try:
            from soup_cli.utils.neuron_compress import find_merge_candidates
            from soup_cli.utils.spectrum_scan import resolve_model_weights

            weights_dir = resolve_model_weights(req.model)
            candidates = find_merge_candidates(
                weights_dir, threshold=req.threshold, max_pairs_per_layer=req.max_pairs_per_layer
            )
            payload = {
                str(layer_idx): [
                    {"i": c.i, "j": c.j, "joint_similarity": c.joint_similarity}
                    for c in cands
                ]
                for layer_idx, cands in candidates.items()
            }
            total_pairs = sum(len(v) for v in payload.values())
            _finish_compress_job(
                job_id,
                result={
                    "candidates": payload,
                    "total_pairs": total_pairs,
                    "layers_with_candidates": len(payload),
                },
            )
        except Exception as exc:
            _finish_compress_job(job_id, error=f"{type(exc).__name__}: {exc}")

    @app.post("/api/compress/neurons/scan", dependencies=[Depends(_verify_token)])
    def compress_neurons_scan(req: NeuronScanRequest):
        job_id = _new_compress_job(
            "neurons_scan", {"model": req.model, "threshold": req.threshold}
        )
        threading.Thread(target=_run_neuron_scan, args=(job_id, req), daemon=True).start()
        return {"job_id": job_id}

    class NeuronApplyRequest(PydanticBaseModel):
        model: str
        threshold: float = Field(default=0.98, ge=0.0, le=1.0)
        max_pairs_per_layer: int = Field(default=50, ge=1, le=1000)
        output_dir: str
        allow_nonuniform: bool = False
        eval_texts: Optional[List[str]] = None

    def _run_neuron_apply(job_id: str, req: "NeuronApplyRequest", target_dir: str):
        try:
            from soup_cli.utils.neuron_compress import (
                apply_merges_to_checkpoint,
                find_merge_candidates,
            )
            from soup_cli.utils.spectrum_scan import resolve_model_weights

            weights_dir = resolve_model_weights(req.model)
            candidates = find_merge_candidates(
                weights_dir, threshold=req.threshold, max_pairs_per_layer=req.max_pairs_per_layer
            )
            summary = apply_merges_to_checkpoint(
                weights_dir, target_dir, candidates, allow_nonuniform=req.allow_nonuniform
            )
            result = {
                "output_dir": target_dir,
                "layers": [
                    {"layer_idx": idx, "before": before, "after": after}
                    for idx, before, after in summary
                ],
            }
            if req.eval_texts:
                from soup_cli.utils.neuron_compress import quick_eval_merge

                result["quick_eval"] = quick_eval_merge(req.model, target_dir, req.eval_texts)
            _finish_compress_job(job_id, result=result)
        except Exception as exc:
            _finish_compress_job(job_id, error=f"{type(exc).__name__}: {exc}")

    @app.post("/api/compress/neurons/apply", dependencies=[Depends(_verify_token)])
    def compress_neurons_apply(req: NeuronApplyRequest):
        from soup_cli.utils.paths import is_under_cwd

        if not is_under_cwd(req.output_dir):
            raise HTTPException(
                status_code=403, detail="output_dir must stay inside the working directory"
            )
        job_id = _new_compress_job(
            "neurons_apply", {"model": req.model, "output_dir": req.output_dir}
        )
        threading.Thread(
            target=_run_neuron_apply, args=(job_id, req, req.output_dir), daemon=True
        ).start()
        return {"job_id": job_id}

    @app.get("/api/compress/jobs")
    def compress_jobs():
        with _compress_jobs_lock:
            return {
                "jobs": sorted(
                    _compress_jobs.values(), key=lambda j: j["started_ts"], reverse=True
                )
            }

    # --- SVD compression ---

    class SvdScanRequest(PydanticBaseModel):
        model: str
        modules: str = "mlp,attn"
        energy_thresholds: List[float] = Field(default_factory=lambda: [0.90, 0.95, 0.99])

    def _run_svd_scan(job_id: str, req: "SvdScanRequest"):
        try:
            from soup_cli.utils.spectrum_scan import resolve_model_weights
            from soup_cli.utils.svd_compress import analyze_svd

            weights_dir = resolve_model_weights(req.model)
            analysis = analyze_svd(
                weights_dir, modules=req.modules, energy_thresholds=tuple(req.energy_thresholds)
            )
            payload = [
                {
                    "param_name": a.param_name,
                    "group": a.group,
                    "module_type": a.module_type,
                    "shape": list(a.shape),
                    "rank_at_energy": {str(k): v for k, v in a.rank_at_energy.items()},
                }
                for a in analysis
            ]
            _finish_compress_job(job_id, result={"matrices": payload})
        except Exception as exc:
            _finish_compress_job(job_id, error=f"{type(exc).__name__}: {exc}")

    @app.post("/api/compress/svd/scan", dependencies=[Depends(_verify_token)])
    def compress_svd_scan(req: SvdScanRequest):
        job_id = _new_compress_job("svd_scan", {"model": req.model})
        threading.Thread(target=_run_svd_scan, args=(job_id, req), daemon=True).start()
        return {"job_id": job_id}

    class SvdApplyRequest(PydanticBaseModel):
        model: str
        modules: str = "mlp,attn"
        rank_at_energy: float = Field(default=0.95, ge=0.0, le=1.0)
        mode: str = "denoise"  # "denoise" | "factorize"
        output_dir: str

    def _run_svd_apply(job_id: str, req: "SvdApplyRequest", target_dir: str):
        try:
            from soup_cli.utils.spectrum_scan import resolve_model_weights
            from soup_cli.utils.svd_compress import analyze_svd, apply_svd_to_checkpoint

            weights_dir = resolve_model_weights(req.model)
            analysis = analyze_svd(
                weights_dir, modules=req.modules, energy_thresholds=(req.rank_at_energy,)
            )
            plan = {a.param_name: a.rank_at_energy[req.rank_at_energy] for a in analysis}
            report = apply_svd_to_checkpoint(weights_dir, target_dir, plan, mode=req.mode)
            _finish_compress_job(
                job_id, result={"output_dir": target_dir, "matrices": report, "mode": req.mode}
            )
        except Exception as exc:
            _finish_compress_job(job_id, error=f"{type(exc).__name__}: {exc}")

    @app.post("/api/compress/svd/apply", dependencies=[Depends(_verify_token)])
    def compress_svd_apply(req: SvdApplyRequest):
        from soup_cli.utils.paths import is_under_cwd

        if req.mode not in ("denoise", "factorize"):
            raise HTTPException(status_code=400, detail="mode must be 'denoise' or 'factorize'")
        if not is_under_cwd(req.output_dir):
            raise HTTPException(
                status_code=403, detail="output_dir must stay inside the working directory"
            )
        job_id = _new_compress_job(
            "svd_apply", {"model": req.model, "output_dir": req.output_dir}
        )
        threading.Thread(
            target=_run_svd_apply, args=(job_id, req, req.output_dir), daemon=True
        ).start()
        return {"job_id": job_id}

    # --- Distillation config bridge (see commands/compress.py::distill_config
    # for the same generator, exposed here for the UI's "Distill" button —
    # reuses the *existing*, already-integrated distillation trainer via
    # `task: distill`, this only generates the config text) ---

    class DistillConfigRequest(PydanticBaseModel):
        student: str
        teacher: str
        data_train: Optional[str] = None
        mode: str = "token"
        divergence: str = "forward_kl"

    @app.post("/api/compress/distill-config", dependencies=[Depends(_verify_token)])
    def compress_distill_config(req: DistillConfigRequest):
        from soup_cli.commands.compress import build_distill_config_yaml

        if req.mode not in ("token", "sequence"):
            raise HTTPException(status_code=400, detail="mode must be 'token' or 'sequence'")
        yaml_text = build_distill_config_yaml(
            student_base=req.student,
            teacher_model=req.teacher,
            data_train=req.data_train,
            mode=req.mode,
            divergence=req.divergence,
        )
        return {"yaml": yaml_text}

    return app
