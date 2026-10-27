"""Unit tests for Dynamic Parameterization in the Ocular compiler pipeline."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from ocular.cli import compile_command
from ocular.pipeline.state_graph import DAGBuilder
from ocular.pipeline.code_generator import JinjaCodeGenerator
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
    assert schema["title"] == "WorkflowExecutionRequest"
    assert schema["type"] == "object"
    assert "interval" in schema["properties"]
    assert schema["properties"]["interval"]["default"] == "15m"
    assert "required" in schema
    assert "interval" in schema["required"]

    # Verify DAG nodes map parameter_ref
    dag_nodes = state_graph_data.get("dag", {}).get("nodes", {})
    interval_nodes = [n for n in dag_nodes.values() if n.get("parameter_ref") == "interval"]
    assert len(interval_nodes) >= 1

    # Verify app.py ExecutionPayload Pydantic v2 model and OpenAPI documentation
    sys.path.insert(0, str(compiled_test_dist.resolve()))
    try:
        app_mod = importlib.import_module("app")
        assert hasattr(app_mod, "ExecutionPayload")
        ExecutionPayload = app_mod.ExecutionPayload

        # Inspect model fields
        fields = ExecutionPayload.model_fields
        assert "interval" in fields
        assert fields["interval"].default == "15m"

        # Verify FastAPI OpenAPI schema exposes ExecutionPayload in Swagger /docs
        client = TestClient(app_mod.app)
        openapi_res = client.get("/openapi.json")
        assert openapi_res.status_code == 200
        openapi = openapi_res.json()
        schemas = openapi.get("components", {}).get("schemas", {})
        assert "ExecutionPayload" in schemas
        payload_schema = schemas["ExecutionPayload"]
        assert "interval" in payload_schema["properties"]
        assert payload_schema["properties"]["interval"]["default"] == "15m"

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
        import asyncio

        # Test driver.run() with empty dict and None
        res_empty = asyncio.run(driver_mod.run({}))
        assert res_empty["status"] == "success"
        assert res_empty.get("interval") == "15m"
        assert res_empty.get("target_anchor") == "timeframe_15m"

        res_none = asyncio.run(driver_mod.run(None))
        assert res_none["status"] == "success"
        assert res_none.get("interval") == "15m"

        # Test FastAPI POST /api/v1/execute without parameters
        app_mod = importlib.import_module("app")
        client = TestClient(app_mod.app)
        res = client.post("/api/v1/execute", json={})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["results"].get("interval") == "15m"
    finally:
        if str(compiled_test_dist.resolve()) in sys.path:
            sys.path.remove(str(compiled_test_dist.resolve()))


def test_value_substitution_during_execution(compiled_test_dist: Path):
    """c) Verifies value substitution during live execution via driver, API, and CLI."""
    sys.path.insert(0, str(compiled_test_dist.resolve()))
    try:
        driver_mod = importlib.import_module("playwright_driver")
        import asyncio

        # 1. Driver run with custom parameter substitution
        res = asyncio.run(driver_mod.run(parameters={"interval": "1h"}))
        assert res["status"] == "success"
        assert res.get("interval") == "1h"
        assert res.get("target_anchor") == "timeframe_1h"

        # 2. FastAPI microservice execution with typed payload substitution
        app_mod = importlib.import_module("app")
        client = TestClient(app_mod.app)

        # Flat payload
        res_api_flat = client.post("/api/v1/execute", json={"interval": "1h"})
        assert res_api_flat.status_code == 200
        data_flat = res_api_flat.json()
        assert data_flat["success"] is True
        assert data_flat["results"].get("interval") == "1h"

        # Wrapped payload
        res_api_wrapped = client.post("/api/v1/execute", json={"parameters": {"interval": "1h"}})
        assert res_api_wrapped.status_code == 200
        data_wrapped = res_api_wrapped.json()
        assert data_wrapped["success"] is True
        assert data_wrapped["results"].get("interval") == "1h"

        # 3. CLI execution flag: python dist/playwright_driver.py --params '{"interval": "1h"}'
        driver_script = compiled_test_dist / "playwright_driver.py"
        cmd = [sys.executable, str(driver_script), "--params", '{"interval": "1h"}']
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        cli_out = proc.stdout
        assert "1h" in cli_out
        # Also verify --json flag emits raw JSON
        cmd_json = [sys.executable, str(driver_script), "--params", '{"interval": "1h"}', "--json"]
        proc_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
        assert '"interval": "1h"' in proc_json.stdout
    finally:
        if str(compiled_test_dist.resolve()) in sys.path:
            sys.path.remove(str(compiled_test_dist.resolve()))
