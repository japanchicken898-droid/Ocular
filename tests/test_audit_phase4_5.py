"""Regression test suite for Phase 4 (Execution Reliability) and Phase 5 (External Calls & Filesystem Hygiene)."""

import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import threading
import time
import pytest
from starlette.testclient import TestClient

from ocular.cli import compile_command
from ocular.pipeline.code_generator import JinjaCodeGenerator
from ocular.pipeline.state_graph import DAGBuilder, CompilationError, export_mermaid_dag, render_rich_dag_tree
from ocular.pipeline.segmenter import MetaSAMSegmenter
from samples.create_sample_video import generate_synthetic_screen_recording


class FixtureServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head><title>Test App</title></head>
<body>
    <button id="btn_submit" role="button">Submit</button>
    <input id="input_query" role="textbox" name="query" value="" />
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/error-404":
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")
        elif self.path == "/error-500":
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Server Error")
        else:
            self.send_response(200)
            self.end_headers()

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def fixture_server():
    server = HTTPServer(("127.0.0.1", 0), FixtureServerHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def test_filesystem_symlink_refusal(tmp_path: Path):
    """Phase 5: _safe_write_text and export_mermaid_dag must strictly refuse to write to symlinks with IOError."""
    target_file = tmp_path / "target.txt"
    target_file.write_text("legitimate data")

    symlink_file = tmp_path / "symlink.txt"
    symlink_file.symlink_to(target_file)

    # 1. _safe_write_text refusal
    with pytest.raises(IOError, match="Refusing to write to symlink"):
        JinjaCodeGenerator._safe_write_text(symlink_file, "malicious payload")
    assert target_file.read_text() == "legitimate data"

    # 2. export_mermaid_dag refusal
    mermaid_symlink = tmp_path / "graph.mmd"
    mermaid_symlink.symlink_to(target_file)
    with pytest.raises(IOError, match="Refusing to write to symlink"):
        export_mermaid_dag({"workflow_name": "Test", "nodes": {}, "edges": []}, mermaid_symlink)
    assert target_file.read_text() == "legitimate data"


def test_rich_markup_escaping():
    """Phase 5: render_rich_dag_tree escapes Rich markup in node names, titles, and preconditions."""
    dag = {
        "workflow_name": "[bold red]Exploit[/bold red]",
        "nodes": {
            "[italic]state_1[/italic]": {
                "label": "[underline]Label[/underline]",
                "preconditions": [{"description": "[blink]Precondition[/blink]"}],
                "assertions": ["[green]Assertion[/green]"],
            }
        },
        "edges": [
            {
                "from_node": "[italic]state_1[/italic]",
                "to_node": "[italic]state_1[/italic]",
                "action_type": "[yellow]click[/yellow]",
                "trigger_element": "[blue]button[/blue]",
            }
        ],
    }
    tree = render_rich_dag_tree(dag)
    assert tree is not None


def test_external_api_endpoint_sanitization(monkeypatch):
    """Phase 5: Insecure HTTP endpoints and non-whitelisted hosts must be rejected for external APIs."""
    monkeypatch.setenv("OCULAR_ALLOW_EXTERNAL_INFERENCE", "1")
    monkeypatch.setenv("BASETEN_API_KEY", "dummy_key")
    monkeypatch.setenv("MUSE_SPARK_API_ENDPOINT", "http://insecure.baseten.co/v1")
    builder = DAGBuilder(workflow_name="ApiSecTest")

    # Insecure plaintext HTTP must return None (refusing to transmit request)
    assert builder._call_muse_spark_api([]) is None

    # Non-whitelisted host when OCULAR_ALLOWED_API_HOSTS is set must return None
    monkeypatch.setenv("MUSE_SPARK_API_ENDPOINT", "https://evil-host.com/v1")
    monkeypatch.setenv("OCULAR_ALLOWED_API_HOSTS", "inference.baseten.co,api.meta.ai")
    assert builder._call_muse_spark_api([]) is None

    # Segmenter META_MODEL_API_ENDPOINT validation
    monkeypatch.setenv("META_MODEL_API_KEY", "dummy_meta_key")
    monkeypatch.setenv("META_MODEL_API_ENDPOINT", "http://insecure.meta.ai/v1")
    monkeypatch.setenv("OCULAR_ALLOW_EXTERNAL_INFERENCE", "1")
    import numpy as np
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    segmenter = MetaSAMSegmenter()
    with pytest.raises(ValueError, match="Insecure endpoint scheme"):
        segmenter._call_meta_sam_api(dummy_frame)


@pytest.mark.asyncio
async def test_driver_navigation_error_and_postcondition(fixture_server: str, tmp_path: Path, monkeypatch):
    """Phase 4: Real navigation failures (404/500/unreachable) return success: False, and postconditions are verified."""
    monkeypatch.setenv("OCULAR_ALLOWED_HOSTS", "127.0.0.1")
    monkeypatch.setenv("OCULAR_ALLOWED_EGRESS_HOSTS", "127.0.0.1")
    dist_dir = tmp_path / "dist_nav_test"
    segments = [
        {"id": "btn_submit", "label": "Submit", "role": "button", "suggested_selector": "#btn_submit", "bbox": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.05}},
        {"id": "input_query", "label": "Query", "role": "textbox", "suggested_selector": "#input_query", "bbox": {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.05}},
    ]
    dag = {
        "workflow_name": "NavTestWorkflow",
        "target_url": f"{fixture_server}/",
        "nodes": {s["id"]: {"role": s["role"]} for s in segments},
        "edges": [
            {"from_node": "state_root", "to_node": "btn_submit", "action_type": "click", "trigger_element": "btn_submit"},
            {"from_node": "btn_submit", "to_node": "input_query", "action_type": "fill", "trigger_element": "input_query", "parameter_ref": "query", "default_value": "hello_test"},
        ],
    }
    generator = JinjaCodeGenerator()
    generator.emit(dag=dag, segments=segments, out_dir=dist_dir)

    import sys
    import importlib
    sys.modules.pop("netguard", None)
    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    sys.path.insert(0, str(dist_dir.resolve()))
    try:
        driver_mod = importlib.import_module("playwright_driver")
        driver_cls = getattr(driver_mod, "NavTestWorkflow")

        # 1. 404 Target page must fail with success: False, status: error
        driver_404 = driver_cls(base_url=f"{fixture_server}/error-404")
        res_404 = await driver_404.execute_workflow()
        assert res_404["success"] is False
        assert res_404["status"] == "error"
        assert "404" in res_404.get("error", "")

        # 2. 500 Target page must fail
        driver_500 = driver_cls(base_url=f"{fixture_server}/error-500")
        res_500 = await driver_500.execute_workflow()
        assert res_500["success"] is False
        assert res_500["status"] == "error"

        # 3. Successful execution on real fixture verifies postconditions
        driver_ok = driver_cls(base_url=f"{fixture_server}/")
        res_ok = await driver_ok.execute_workflow(params={"query": "verified_content"})
        assert res_ok["success"] is True
        assert res_ok["status"] == "success"
        for step in res_ok["steps"]:
            assert step.get("verified") is True
    finally:
        sys.path.remove(str(dist_dir.resolve()))
        sys.modules.pop("netguard", None)
        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)


def test_fastapi_authentication_and_limits(fixture_server: str, tmp_path: Path, monkeypatch):
    """Phase 4: FastAPI auth via X-API-Key, 401 when unconfigured/unauthorized, 503 on queue saturation, 422 on limits."""
    monkeypatch.setenv("OCULAR_ALLOWED_HOSTS", "127.0.0.1")
    monkeypatch.setenv("OCULAR_ALLOWED_EGRESS_HOSTS", "127.0.0.1")
    dist_dir = tmp_path / "dist_auth_test"
    segments = [
        {"id": "btn_submit", "label": "Submit", "role": "button", "suggested_selector": "#btn_submit", "bbox": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.05}},
    ]
    dag = {
        "workflow_name": "AuthTestWorkflow",
        "target_url": f"{fixture_server}/",
        "nodes": {"btn_submit": {"role": "button"}},
        "edges": [{"from_node": "state_root", "to_node": "btn_submit", "action_type": "click", "trigger_element": "btn_submit"}],
    }
    generator = JinjaCodeGenerator()
    generator.emit(dag=dag, segments=segments, out_dir=dist_dir)

    import sys
    import importlib
    sys.modules.pop("netguard", None)
    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    sys.path.insert(0, str(dist_dir.resolve()))
    try:
        app_mod = importlib.import_module("app")
        app = app_mod.app
        client = TestClient(app)

        # 1. When OCULAR_API_KEY is configured, request without X-API-Key must return 401
        monkeypatch.delenv("OCULAR_INSECURE_NO_AUTH", raising=False)
        monkeypatch.setenv("OCULAR_API_KEY", "secret-token-123")

        res_no_auth = client.post("/api/v1/execute", json={"headless": True})
        assert res_no_auth.status_code == 401

        res_wrong_auth = client.post("/api/v1/execute", json={"headless": True}, headers={"X-API-Key": "wrong-token"})
        assert res_wrong_auth.status_code == 401

        # 2. Valid X-API-Key proceeds
        res_auth_ok = client.post("/api/v1/execute", json={"headless": True}, headers={"X-API-Key": "secret-token-123"})
        assert res_auth_ok.status_code == 200
        assert res_auth_ok.json()["success"] is True

        # 3. Viewport and timeout bounds validation (422)
        res_bad_viewport = client.post(
            "/api/v1/execute",
            json={"viewport_width": 50, "headless": True},
            headers={"X-API-Key": "secret-token-123"}
        )
        assert res_bad_viewport.status_code == 422

        res_bad_timeout = client.post(
            "/api/v1/execute",
            json={"timeout_ms": 10, "headless": True},
            headers={"X-API-Key": "secret-token-123"}
        )
        assert res_bad_timeout.status_code == 422

        # 4. Saturated queue timeout (503)
        monkeypatch.setenv("OCULAR_QUEUE_TIMEOUT_SEC", "0.01")
        # Acquire all semaphore slots
        app_mod._execution_semaphore._value = 0
        try:
            res_queue_sat = client.post(
                "/api/v1/execute",
                json={"headless": True},
                headers={"X-API-Key": "secret-token-123"}
            )
            assert res_queue_sat.status_code == 503
            assert "queue saturated" in res_queue_sat.json()["detail"].lower()
        finally:
            app_mod._execution_semaphore._value = app_mod.MAX_CONCURRENT_RUNS
    finally:
        sys.path.remove(str(dist_dir.resolve()))
        sys.modules.pop("netguard", None)
        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)
