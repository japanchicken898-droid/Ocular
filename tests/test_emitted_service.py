"""Tests for the emitted FastAPI wrapper service."""

import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient
from ocular.cli import compile_command
from samples.create_sample_video import generate_synthetic_screen_recording


def test_emitted_fastapi_endpoints(tmp_path: Path):
    video_path = tmp_path / "sample.mp4"
    generate_synthetic_screen_recording(video_path, fps=60, width=1920, height=1080)

    dist_dir = tmp_path / "emitted_dist"
    compile_command(input_path=str(video_path), out_dir=str(dist_dir))

    assert (dist_dir / "app.py").exists()
    assert (dist_dir / "playwright_driver.py").exists()

    sys.path.insert(0, str(dist_dir.resolve()))
    try:
        import importlib
        app_mod = importlib.import_module("app")
        app = app_mod.app

        client = TestClient(app)

        # 1. Test /health
        res_health = client.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert health_data["status"] == "ok"
        assert health_data["compiler"] == "Ocular"

        # 2. Test /api/v1/schema
        res_schema = client.get("/api/v1/schema")
        assert res_schema.status_code == 200
        schema_data = res_schema.json()
        assert "parameters_schema" in schema_data
        assert "anchors_count" in schema_data
        assert schema_data["zero_runtime_vision"] is True

        # 3. Test POST /api/v1/execute with generic execution options
        res_exec = client.post(
            "/api/v1/execute",
            json={
                "viewport_width": 1920,
                "viewport_height": 1080,
                "headless": True,
            },
        )
        assert res_exec.status_code == 200
        assert "X-Ocular-Compile-Hash" in res_exec.headers
        exec_data = res_exec.json()
        assert exec_data["success"] is True
        assert "execution_id" in exec_data
        assert exec_data["steps_executed"] >= 1
        assert "results" in exec_data

    finally:
        if str(dist_dir.resolve()) in sys.path:
            sys.path.remove(str(dist_dir.resolve()))
