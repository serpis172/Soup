"""Tests for Web UI Training Live Monitor — SSE endpoints and progress API."""

import os
from unittest.mock import MagicMock, patch

import pytest


def _auth_headers():
    """Return auth headers with the current UI token."""
    from soup_cli.ui.app import get_auth_token
    return {"Authorization": f"Bearer {get_auth_token()}"}


class TestTrainLogsSSE:
    """Test GET /api/train/logs SSE endpoint."""

    def test_logs_endpoint_exists(self):
        """Route /api/train/logs should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/train/logs" in routes

    def test_logs_returns_event_stream(self):
        """SSE endpoint should return text/event-stream content type.

        Bug fix (this session): /api/train/logs now reads from
        `_train_log_buffer` (filled by the always-on drain thread started
        in /api/train/start) instead of iterating `proc.stdout` directly —
        see that buffer's module-level comment in ui/app.py for why
        (unattended stdout could fill the OS pipe buffer and hang the
        training process). This test populates the buffer directly rather
        than mocking `proc.stdout`, to match the new contract.
        """
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, None, 0, 0, 0]
        ui_mod._train_process = mock_proc
        with ui_mod._train_log_lock:
            ui_mod._train_log_buffer.clear()
            ui_mod._train_log_buffer.extend(["Epoch 1/3", "Loss: 2.5"])

        client = TestClient(create_app())
        try:
            with client.stream("GET", "/api/train/logs") as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                text = body.decode()
                assert "Epoch 1/3" in text
                assert "Loss: 2.5" in text
                assert "done" in text
        finally:
            ui_mod._train_process = None
            with ui_mod._train_log_lock:
                ui_mod._train_log_buffer.clear()

    def test_logs_no_training_returns_done(self):
        """When no training is running, SSE should emit done event."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        client = TestClient(create_app())
        with client.stream("GET", "/api/train/logs") as resp:
            assert resp.status_code == 200
            body = b""
            for chunk in resp.iter_bytes():
                body += chunk
            text = body.decode()
            assert "done" in text

    def test_logs_last_event_id_reconnection(self):
        """Last-Event-ID header should skip earlier lines."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, None, 0, 0, 0]
        ui_mod._train_process = mock_proc
        with ui_mod._train_log_lock:
            ui_mod._train_log_buffer.clear()
            ui_mod._train_log_buffer.extend(["Line 1", "Line 2", "Line 3"])

        client = TestClient(create_app())
        try:
            with client.stream(
                "GET", "/api/train/logs",
                headers={"Last-Event-ID": "1"},
            ) as resp:
                assert resp.status_code == 200
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                text = body.decode()
                # Last-Event-ID: 1 means "I already have id=1 (Line 2)" —
                # skip_count = 1 + 1 = 2, so resume from id=2 (Line 3) only.
                assert "Line 1" not in text
                assert "Line 2" not in text
                assert "Line 3" in text
        finally:
            ui_mod._train_process = None
            with ui_mod._train_log_lock:
                ui_mod._train_log_buffer.clear()

    def test_logs_no_auth_required(self):
        """SSE log endpoint is GET (read-only) — no auth needed."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        client = TestClient(create_app())
        # No auth header — should still succeed (not 401)
        with client.stream("GET", "/api/train/logs") as resp:
            assert resp.status_code == 200


class TestLiveMetricsSSE:
    """Test GET /api/train/metrics/live SSE endpoint."""

    def test_metrics_live_endpoint_exists(self):
        """Route /api/train/metrics/live should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/train/metrics/live" in routes

    def test_metrics_live_returns_event_stream(self, tmp_path):
        """SSE metrics endpoint should return text/event-stream content type."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        # Simulate running training
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Already finished
        ui_mod._train_process = mock_proc

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            with client.stream("GET", "/api/train/metrics/live") as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]

        ui_mod._train_process = None

    def test_metrics_live_emits_new_rows(self, tmp_path):
        """SSE should emit metric rows as they appear in the DB."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        db_path = tmp_path / "test.db"

        # Pre-populate a run with metrics
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=db_path)
        run_id = tracker.start_run(
            config_dict={"base": "test", "task": "sft"},
            device="cpu",
            device_name="CPU",
            gpu_info={"memory_total": "N/A"},
        )
        tracker.log_metrics(run_id, step=10, loss=2.5, lr=1e-5)
        tracker.close()

        # Simulate finished training
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        ui_mod._train_process = mock_proc

        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            with client.stream(
                "GET", f"/api/train/metrics/live?run_id={run_id}"
            ) as resp:
                assert resp.status_code == 200
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                text = body.decode()
                # Should contain metric data
                assert "step" in text or "done" in text

        ui_mod._train_process = None

    def test_metrics_live_no_auth_required(self, tmp_path):
        """SSE metrics endpoint is GET (read-only) — no auth needed."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        ui_mod._train_process = mock_proc

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            with client.stream("GET", "/api/train/metrics/live") as resp:
                assert resp.status_code == 200

        ui_mod._train_process = None

    def test_metrics_live_done_when_no_training(self, tmp_path):
        """Should emit done event when no training is running."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            with client.stream("GET", "/api/train/metrics/live") as resp:
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                text = body.decode()
                assert "done" in text

        ui_mod._train_process = None


class TestTrainProgress:
    """Test GET /api/train/progress endpoint."""

    def test_progress_endpoint_exists(self):
        """Route /api/train/progress should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/train/progress" in routes

    def test_progress_not_running(self):
        """Should return running=false when no training."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        client = TestClient(create_app())
        response = client.get("/api/train/progress")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is False

    def test_progress_running_with_metrics(self, tmp_path):
        """Should return progress when training is running."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        db_path = tmp_path / "test.db"

        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=db_path)
        run_id = tracker.start_run(
            config_dict={"base": "test", "task": "sft"},
            device="cpu",
            device_name="CPU",
            gpu_info={"memory_total": "N/A"},
        )
        tracker.log_metrics(run_id, step=50, loss=1.5, lr=1e-5)
        tracker.close()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 9999
        ui_mod._train_process = mock_proc

        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            response = client.get(f"/api/train/progress?run_id={run_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["running"] is True
            assert data["current_step"] == 50

        ui_mod._train_process = None

    def test_progress_auto_discovers_run_id_from_pid(self, tmp_path):
        """This session's fix: the frontend never tracked a run_id at all
        (nothing in it previously did), so /api/train/progress must be able
        to find step/loss/speed on its own from just the tracked
        subprocess's PID — via ExperimentTracker.find_run_by_pid(), fed by
        train.py's tracker.mark_running(run_id, pid=os.getpid()) call.
        Without EITHER half of that chain, this would silently return
        current_step=0 forever despite the tracker having real data.
        """
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        db_path = tmp_path / "test.db"

        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=db_path)
        run_id = tracker.start_run(
            config_dict={"base": "test", "task": "sft"},
            device="cpu",
            device_name="CPU",
            gpu_info={"memory_total": "N/A"},
        )
        tracker.mark_running(run_id, pid=8642)
        tracker.log_metrics(run_id, step=17, loss=0.9, lr=1e-5, speed=2.5)
        tracker.close()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 8642
        ui_mod._train_process = mock_proc

        try:
            with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
                client = TestClient(create_app())
                # Deliberately NOT passing run_id — this is the whole point.
                response = client.get("/api/train/progress")
                assert response.status_code == 200
                data = response.json()
                assert data["running"] is True
                assert data["run_id"] == run_id
                assert data["current_step"] == 17
                assert data["speed"] == 2.5
        finally:
            ui_mod._train_process = None

    def test_progress_returns_correct_fields(self, tmp_path):
        """Progress response should contain all required fields."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        db_path = tmp_path / "test.db"

        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=db_path)
        run_id = tracker.start_run(
            config_dict={"base": "test", "task": "sft"},
            device="cpu",
            device_name="CPU",
            gpu_info={"memory_total": "N/A"},
        )
        tracker.log_metrics(run_id, step=10, loss=2.0, lr=1e-5)
        tracker.close()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 1234
        ui_mod._train_process = mock_proc

        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            response = client.get(f"/api/train/progress?run_id={run_id}")
            data = response.json()
            expected_keys = {"running", "current_step", "run_id"}
            assert expected_keys.issubset(set(data.keys()))

        ui_mod._train_process = None

    def test_progress_no_auth_required(self):
        """Progress endpoint is GET (read-only) — no auth needed."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        client = TestClient(create_app())
        response = client.get("/api/train/progress")
        assert response.status_code == 200


class TestSSEGracefulClose:
    """Test that SSE endpoints close gracefully."""

    def test_logs_closes_when_training_stops(self):
        """Log SSE should end when training process finishes."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        mock_proc = MagicMock()
        mock_proc.stdout = iter([b"Starting\n"])
        mock_proc.poll.side_effect = [None, 0]
        ui_mod._train_process = mock_proc

        client = TestClient(create_app())
        try:
            with client.stream("GET", "/api/train/logs") as resp:
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                text = body.decode()
                # Stream should terminate with done event
                assert "done" in text
        finally:
            ui_mod._train_process = None

    def test_heartbeat_event(self):
        """SSE should include heartbeat comments to keep connection alive."""
        # This is a design test — heartbeats are sent as SSE comments (:heartbeat)
        # Testing that the generator yields at least one event
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        client = TestClient(create_app())
        with client.stream("GET", "/api/train/logs") as resp:
            body = b""
            for chunk in resp.iter_bytes():
                body += chunk
            # Stream should have some content (at least done event)
            assert len(body) > 0

        ui_mod._train_process = None


class TestTrainPauseResume:
    """Test POST /api/train/pause and /api/train/resume (this session)."""

    def test_pause_requires_auth(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None
        client = TestClient(create_app())
        resp = client.post("/api/train/pause")
        assert resp.status_code == 401

    def test_pause_without_running_training_returns_409(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None
        client = TestClient(create_app())
        resp = client.post("/api/train/pause", headers=_auth_headers())
        assert resp.status_code == 409

    def test_pause_then_resume_calls_sigstop_sigcont(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 4321
        ui_mod._train_process = mock_proc
        ui_mod._train_paused = False

        client = TestClient(create_app())
        try:
            with patch("os.kill") as mock_kill:
                import signal as signal_mod

                resp = client.post("/api/train/pause", headers=_auth_headers())
                assert resp.status_code == 200
                assert resp.json()["paused"] is True
                mock_kill.assert_called_once_with(4321, signal_mod.SIGSTOP)
                assert ui_mod._train_paused is True

                mock_kill.reset_mock()
                resp = client.post("/api/train/resume", headers=_auth_headers())
                assert resp.status_code == 200
                assert resp.json()["paused"] is False
                mock_kill.assert_called_once_with(4321, signal_mod.SIGCONT)
                assert ui_mod._train_paused is False
        finally:
            ui_mod._train_process = None
            ui_mod._train_paused = False

    def test_status_reflects_paused_state(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 555
        ui_mod._train_process = mock_proc
        ui_mod._train_paused = True

        client = TestClient(create_app())
        try:
            resp = client.get("/api/train/status")
            data = resp.json()
            assert data["running"] is True
            assert data["paused"] is True
            assert data["phase"] == "Paused"
        finally:
            ui_mod._train_process = None
            ui_mod._train_paused = False

    def test_stop_resumes_a_paused_process_before_terminating(self):
        """A SIGSTOP'd process can't act on SIGTERM until resumed — stop
        must SIGCONT first so terminate() actually takes effect."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 777
        ui_mod._train_process = mock_proc
        ui_mod._train_paused = True

        client = TestClient(create_app())
        try:
            with patch("os.kill") as mock_kill:
                import signal as signal_mod

                resp = client.post("/api/train/stop", headers=_auth_headers())
                assert resp.status_code == 200
                mock_kill.assert_called_once_with(777, signal_mod.SIGCONT)
                mock_proc.terminate.assert_called_once()
                assert ui_mod._train_paused is False
        finally:
            ui_mod._train_process = None
            ui_mod._train_paused = False


class TestEstimateTotalSteps:
    """Test _estimate_total_steps (this session's best-effort progress-bar denominator)."""

    def _config(self, tmp_path, n_rows, **overrides):
        import json

        from soup_cli.config.schema import SoupConfig

        f = tmp_path / "train.jsonl"
        f.write_text(
            "\n".join(json.dumps({"instruction": f"Q{i}", "output": f"A{i}"}) for i in range(n_rows))
            + "\n"
        )
        data = {"train": str(f), "val_split": overrides.pop("val_split", 0.0)}
        if "val" in overrides:
            data["val"] = overrides.pop("val")
        training = {
            "epochs": overrides.pop("epochs", 1),
            "batch_size": overrides.pop("batch_size", 2),
            "gradient_accumulation_steps": overrides.pop("gradient_accumulation_steps", 1),
        }
        return SoupConfig(base="some-model", data=data, training=training)

    def test_estimates_steps_for_local_jsonl(self, tmp_path):
        from soup_cli.ui.app import _estimate_total_steps

        # 20 rows, batch_size=2, grad_accum=1 -> 10 steps/epoch, epochs=3 -> 30
        cfg = self._config(tmp_path, 20, epochs=3, batch_size=2, gradient_accumulation_steps=1)
        assert _estimate_total_steps(cfg) == 30

    def test_returns_none_for_auto_batch_size(self, tmp_path):
        from soup_cli.config.schema import SoupConfig
        from soup_cli.ui.app import _estimate_total_steps

        import json

        f = tmp_path / "train.jsonl"
        f.write_text(json.dumps({"instruction": "Q", "output": "A"}) + "\n")
        cfg = SoupConfig(
            base="some-model",
            data={"train": str(f), "val_split": 0.0},
            training={"batch_size": "auto"},
        )
        assert _estimate_total_steps(cfg) is None

    def test_returns_none_for_hf_dataset_name(self, tmp_path):
        from soup_cli.config.schema import SoupConfig
        from soup_cli.ui.app import _estimate_total_steps

        cfg = SoupConfig(base="some-model", data={"train": "some-org/some-dataset"})
        assert _estimate_total_steps(cfg) is None

    def test_accounts_for_val_split(self, tmp_path):
        from soup_cli.ui.app import _estimate_total_steps

        # 100 rows, val_split=0.2 -> 80 train rows, batch=4 -> 20 steps/epoch
        cfg = self._config(tmp_path, 100, epochs=1, batch_size=4, val_split=0.2)
        assert _estimate_total_steps(cfg) == 20

    def test_explicit_val_does_not_shrink_train(self, tmp_path):
        from soup_cli.ui.app import _estimate_total_steps

        # explicit val set -> all 100 train rows count, batch=4 -> 25 steps/epoch
        cfg = self._config(tmp_path, 100, epochs=1, batch_size=4, val_split=0.2, val="ignored.jsonl")
        assert _estimate_total_steps(cfg) == 25
