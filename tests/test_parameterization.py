"""Unit tests for Dynamic Parameterization in the Ocular compiler pipeline."""

import asyncio
import importlib
import json
import subprocess
import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from ocular.cli import compile_command
from ocular.pipeline.code_generator import JinjaCodeGenerator, extract_telemetry_input_parameters
from samples.create_sample_video import generate_synthetic_screen_recording


@pytest.fixture(scope="module")
def compiled_test_dist(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("param_test")
    video_path = tmp_path / "sample.mp4"
    generate_synthetic_screen_recording(video_path, fps=30, width=1920, height=1080)
    dist_dir = tmp_path / "dist"
    ret = compile_command(input_path=str(video_path), out_dir=str(dist_dir))
    assert ret == 0
    return dist_dir


def test_pydantic_schema_generation_from_dag(compiled_test_dist: Path):
    """a) Verifies Pydantic schema generation from DAG parameters in state_graph.json and app.py."""
    state_graph_path = compiled_test_dist / "state_graph.json"
    assert state_graph_path.exists()
    state_graph_data = json.loads(state_graph_path.read_text(encoding="utf-8"))

    # Verify state_graph.json has parameters_schema
    assert "parameters_schema" in state_graph_data
    schema = state_graph_data["parameters_schema"]
    assert "ExecutionRequest" in schema["title"]
    assert schema["type"] == "object"
    assert "viewport_width" in schema["properties"]
    assert "viewport_height" in schema["properties"]
    assert "headless" in schema["properties"]
    assert "timeout_ms" in schema["properties"]

    # Verify app.py ExecutionPayload Pydantic v2 model and OpenAPI documentation
    sys.path.insert(0, str(compiled_test_dist.resolve()))
    try:
        app_mod = importlib.import_module("app")
        assert hasattr(app_mod, "ExecutionPayload")
        ExecutionPayload = app_mod.ExecutionPayload

        # Inspect model fields
        fields = ExecutionPayload.model_fields
        assert "viewport_width" in fields
        assert "viewport_height" in fields
        assert "headless" in fields
        assert "timeout_ms" in fields
        assert fields["headless"].default is True

        # Verify FastAPI OpenAPI schema exposes ExecutionPayload in Swagger /docs
        client = TestClient(app_mod.app)
        openapi_res = client.get("/openapi.json")
        assert openapi_res.status_code == 200
        openapi = openapi_res.json()
        schemas = openapi.get("components", {}).get("schemas", {})
        assert "ExecutionPayload" in schemas
        payload_schema = schemas["ExecutionPayload"]
        assert "viewport_width" in payload_schema["properties"]
        assert "headless" in payload_schema["properties"]

        # Verify /api/v1/schema endpoint exposes the schema
        schema_res = client.get("/api/v1/schema")
        assert schema_res.status_code == 200
        assert "parameters_schema" in schema_res.json()
    finally:
        if str(compiled_test_dist.resolve()) in sys.path:
            sys.path.remove(str(compiled_test_dist.resolve()))


def test_fallback_to_default_values(compiled_test_dist: Path):
    """b) Verifies fallback to default values when no runtime parameter is passed."""
    sys.path.insert(0, str(compiled_test_dist.resolve()))
    try:
        driver_mod = importlib.import_module("playwright_driver")

        # Test driver.run() with empty dict and None
        res_empty = asyncio.run(driver_mod.run({}))
        assert res_empty["status"] == "success"
        assert "steps" in res_empty
        assert "compile_hash" in res_empty

        res_none = asyncio.run(driver_mod.run(None))
        assert res_none["status"] == "success"
        assert "latency_ms" in res_none

        # Test FastAPI POST /api/v1/execute without parameters
        app_mod = importlib.import_module("app")
        client = TestClient(app_mod.app)
        res = client.post("/api/v1/execute", json={})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "execution_id" in data
        assert data["results"]["status"] == "success"
    finally:
        if str(compiled_test_dist.resolve()) in sys.path:
            sys.path.remove(str(compiled_test_dist.resolve()))


def test_value_substitution_during_execution(compiled_test_dist: Path):
    """c) Verifies value substitution during live execution via driver, API, and CLI."""
    sys.path.insert(0, str(compiled_test_dist.resolve()))
    try:
        driver_mod = importlib.import_module("playwright_driver")

        # 1. Driver run with custom parameter substitution
        res = asyncio.run(driver_mod.run(parameters={"viewport_width": 1024, "viewport_height": 768}))
        assert res["status"] == "success"
        assert "steps" in res

        # 2. FastAPI microservice execution with typed payload substitution
        app_mod = importlib.import_module("app")
        client = TestClient(app_mod.app)

        # Flat payload
        res_api_flat = client.post("/api/v1/execute", json={"viewport_width": 1024, "headless": True})
        assert res_api_flat.status_code == 200
        data_flat = res_api_flat.json()
        assert data_flat["success"] is True
        assert data_flat["results"]["status"] == "success"

        # Wrapped payload
        res_api_wrapped = client.post("/api/v1/execute", json={"parameters": {"viewport_width": 1024}})
        assert res_api_wrapped.status_code == 200
        data_wrapped = res_api_wrapped.json()
        assert data_wrapped["success"] is True
        assert data_wrapped["results"]["status"] == "success"

        # 3. CLI execution flag: python dist/playwright_driver.py --json
        driver_script = compiled_test_dist / "playwright_driver.py"
        cmd_json = [sys.executable, str(driver_script), "--json"]
        proc_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
        assert '"status": "success"' in proc_json.stdout
    finally:
        if str(compiled_test_dist.resolve()) in sys.path:
            sys.path.remove(str(compiled_test_dist.resolve()))


def test_telemetry_keydown_parameterization(tmp_path: Path):
    """d) Verifies dynamic parameter extraction when telemetry stream contains typed text."""
    video_path = tmp_path / "input_session.mp4"
    generate_synthetic_screen_recording(video_path, fps=10, width=1280, height=720)

    telem_events = [
        {"event_type": "keydown", "action_type": "fill", "key": "h", "target_selector": "input#search"},
        {"event_type": "keydown", "action_type": "fill", "key": "e", "target_selector": "input#search"},
        {"event_type": "keydown", "action_type": "fill", "key": "l", "target_selector": "input#search"},
        {"event_type": "keydown", "action_type": "fill", "key": "l", "target_selector": "input#search"},
        {"event_type": "keydown", "action_type": "fill", "key": "o", "target_selector": "input#search"},
    ]
    telem_path = tmp_path / "telemetry_events.jsonl"
    with open(telem_path, "w", encoding="utf-8") as f:
        for ev in telem_events:
            f.write(json.dumps(ev) + "\n")

    dist_dir = tmp_path / "dist_telem"
    ret = compile_command(input_path=str(video_path), out_dir=str(dist_dir), telemetry_path=str(telem_path))
    assert ret == 0

    state_graph_data = json.loads((dist_dir / "state_graph.json").read_text(encoding="utf-8"))
    schema = state_graph_data.get("parameters_schema", {})
    assert "search" in schema.get("properties", {})
    assert schema["properties"]["search"]["default"] == "hello"
