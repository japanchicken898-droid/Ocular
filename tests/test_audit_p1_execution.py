"""P1 Execution Truthfulness & Service Hardening Regression Tests.

Covers:
- E-01: Truthful verification (no-op button fails postcondition, working button passes).
- E-02: Required parameter enforcement and 422 validation without input reflection.
- C-02: Strict selector count enforcement (locator.count() > 1 fails as ambiguous).
- D-01: Memory-bounded telemetry load with streaming line caps and 4 KB event caps.
- D-02 to D-07: FastAPI hardening (non-ASCII auth, clean health, disabled docs, CORS safety, auth before body limit).
"""

import http.server
import importlib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient
from playwright.async_api import async_playwright

from ocular.pipeline.code_generator import JinjaCodeGenerator
from ocular.telemetry import load_telemetry_events


@pytest.fixture
def clean_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    return dist


def test_d01_memory_bounded_telemetry_load(tmp_path: Path):
    """D-01: load_telemetry_events enforces 64 KB line length, 4 KB event cap, and bounded memory."""
    telemetry_file = tmp_path / "telemetry_test.jsonl"

    valid_event_1 = {"event_type": "keydown", "key": "a", "norm_x": 0.1, "norm_y": 0.2, "elapsed_ms": 100}
    # Oversized line > 64 KB
    huge_padding = "x" * 70_000
    oversized_line = json.dumps({"event_type": "keydown", "key": huge_padding})
    valid_event_2 = {"event_type": "click", "norm_x": 0.5, "norm_y": 0.5, "elapsed_ms": 200}
    # Event within 64 KB line cap but exceeding 4 KB serialized size cap
    medium_padding = "y" * 5_000
    oversized_event = {"event_type": "keydown", "key": medium_padding}

    with open(telemetry_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(valid_event_1) + "\n")
        f.write(oversized_line + "\n")
        f.write(json.dumps(valid_event_2) + "\n")
        f.write(json.dumps(oversized_event) + "\n")

    events, dropped = load_telemetry_events(telemetry_file)
    assert len(events) == 2, f"Expected 2 valid events, got {len(events)}"
    assert dropped == 2, f"Expected 2 dropped lines (one >64KB, one >4KB), got {dropped}"
    assert events[0]["key"] == "a"
    assert events[1]["norm_x"] == 0.5


@pytest.mark.asyncio
async def test_e01_execution_truthfulness_noop_vs_working_button(clean_dist: Path):
    """E-01: No-op dummy button must fail verification (verified=False); working button must pass."""
    html_content = """<!DOCTYPE html>
    <html>
    <body>
        <button id="dummy_btn">Dummy No-op Button</button>
        <button id="working_btn" onclick="document.getElementById('result').innerText = 'Clicked!'; this.setAttribute('data-done', '1');">Working Button</button>
        <div id="result">Initial</div>
    </body>
    </html>
    """

    class FixtureServer(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), FixtureServer)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # 1. Compile workflow with dummy no-op button
    dag_dummy = {
        "workflow_name": "DummyWorkflow",
        "target_url": f"http://127.0.0.1:{port}",
        "edges": [
            {
                "from_node_id": "state_1",
                "to_node_id": "state_2",
                "action_type": "click",
                "trigger_element": "dummy_btn",
                "target_segment_id": "dummy_btn",
                "target_selector": '[id="dummy_btn"]',
            }
        ],
    }
    segments_dummy = [
        {"id": "dummy_btn", "suggested_selector": '[id="dummy_btn"]', "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05}}
    ]

    gen = JinjaCodeGenerator()
    gen.emit(dag_dummy, segments=segments_dummy, out_dir=clean_dist)

    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    sys.path.insert(0, str(clean_dist))
    try:
        driver_mod = importlib.import_module("playwright_driver")
        driver_cls = getattr(driver_mod, "DummyWorkflow")

        # Execute on dummy button: must fail verification and overall success must be False
        driver_dummy = driver_cls()
        res_dummy = await driver_dummy.execute_workflow()
        assert res_dummy["success"] is False, "No-op dummy button falsely succeeded overall!"
        assert res_dummy["steps"][0]["verified"] is False, "No-op dummy button falsely marked verified=True!"

        # 2. Compile workflow with working button
        dist_working = clean_dist.parent / "dist_working"
        dist_working.mkdir(parents=True, exist_ok=True)
        dag_working = {
            "workflow_name": "WorkingWorkflow",
            "target_url": f"http://127.0.0.1:{port}",
            "edges": [
                {
                    "from_node_id": "state_1",
                    "to_node_id": "state_2",
                    "action_type": "click",
                    "trigger_element": "working_btn",
                    "target_segment_id": "working_btn",
                    "target_selector": '[id="working_btn"]',
                }
            ],
        }
        segments_working = [
            {"id": "working_btn", "suggested_selector": '[id="working_btn"]', "bbox": {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.05}}
        ]
        gen.emit(dag_working, segments=segments_working, out_dir=dist_working)

        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)
        sys.path.insert(0, str(dist_working))
        try:
            driver_mod_w = importlib.import_module("playwright_driver")
            driver_cls_w = getattr(driver_mod_w, "WorkingWorkflow")

            driver_working = driver_cls_w()
            res_working = await driver_working.execute_workflow()
            assert res_working["success"] is True, "Working button failed execution!"
            assert res_working["steps"][0]["verified"] is True, "Working button should have verified=True"
        finally:
            if str(dist_working) in sys.path:
                sys.path.remove(str(dist_working))
            sys.modules.pop("playwright_driver", None)
            sys.modules.pop("app", None)
    finally:
        if str(clean_dist) in sys.path:
            sys.path.remove(str(clean_dist))
        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)


@pytest.mark.asyncio
async def test_c02_ambiguous_selector_strict_count(clean_dist: Path):
    """C-02: If locator.count() > 1, fail step as ambiguous rather than silently clicking .first."""
    html_content = """<!DOCTYPE html>
    <html>
    <body>
        <button class="action-btn">First Button</button>
        <button class="action-btn">Second Button</button>
    </body>
    </html>
    """

    class DupServer(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), DupServer)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    dag = {
        "workflow_name": "AmbiguousWorkflow",
        "target_url": f"http://127.0.0.1:{port}",
        "edges": [
            {
                "from_node_id": "state_1",
                "to_node_id": "state_2",
                "action_type": "click",
                "trigger_element": "ambiguous_btn",
                "target_segment_id": "ambiguous_btn",
                "target_selector": ".action-btn",
            }
        ],
    }
    segments = [
        {"id": "ambiguous_btn", "suggested_selector": ".action-btn", "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05}}
    ]

    gen = JinjaCodeGenerator()
    gen.emit(dag, segments=segments, out_dir=clean_dist)

    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    sys.path.insert(0, str(clean_dist))
    try:
        driver_mod = importlib.import_module("playwright_driver")
        driver_cls = getattr(driver_mod, "AmbiguousWorkflow")

        driver = driver_cls()
        res = await driver.execute_workflow()
        assert res["success"] is False
        assert "ambiguous" in str(res.get("error", "")).lower() or any(
            "ambiguous" in str(s.get("error", "")).lower() for s in res.get("steps", [])
        )
    finally:
        if str(clean_dist) in sys.path:
            sys.path.remove(str(clean_dist))
        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)


def test_e02_required_parameters_and_422_no_input_reflection(clean_dist: Path, monkeypatch):
    """E-02: Missing required parameter returns 422 with missing field names without echoing input."""
    dag = {
        "workflow_name": "ValidationWorkflow",
        "target_url": "http://127.0.0.1:8080",
        "edges": [
            {
                "from_node_id": "state_1",
                "to_node_id": "state_2",
                "action_type": "fill",
                "trigger_element": "secret_field",
                "target_segment_id": "secret_field",
                "target_selector": 'input[id="secret"]',
                "parameter_ref": "token",
                "input_payload": "",
                "default_value": "",
            }
        ],
    }
    segments = [
        {"id": "secret_field", "suggested_selector": 'input[id="secret"]', "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05}}
    ]

    gen = JinjaCodeGenerator()
    gen.emit(dag, segments=segments, out_dir=clean_dist)

    monkeypatch.setenv("OCULAR_INSECURE_NO_AUTH", "1")
    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    sys.path.insert(0, str(clean_dist))
    try:
        app_mod = importlib.import_module("app")
        client = TestClient(app_mod.app)

        # 1. Missing required parameter 'token' must return 422
        res_missing = client.post("/api/v1/execute", json={"headless": True})
        assert res_missing.status_code == 422
        err_detail = res_missing.json()["detail"]
        assert any("token" in str(err) for err in err_detail)

        # 2. Sending invalid type with sensitive secret must NOT reflect input in 422 detail
        secret_reflection_payload = "SUPER_SECRET_PAYLOAD_TOKEN"
        res_invalid_type = client.post(
            "/api/v1/execute",
            json={"token": {"nested": secret_reflection_payload}, "headless": True},
        )
        assert res_invalid_type.status_code == 422
        raw_response_text = res_invalid_type.text
        assert secret_reflection_payload not in raw_response_text, "Secret payload was reflected in 422 response!"
    finally:
        if str(clean_dist) in sys.path:
            sys.path.remove(str(clean_dist))
        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)


def test_d02_to_d07_fastapi_hardening(clean_dist: Path, monkeypatch):
    """D-02 to D-07: Non-ASCII auth, clean health endpoint, disabled docs, and CORS validation."""
    dag = {
        "workflow_name": "HardeningWorkflow",
        "target_url": "http://127.0.0.1:8080",
        "edges": [],
    }
    gen = JinjaCodeGenerator()
    gen.emit(dag, segments=[], out_dir=clean_dist)

    # Configure API key
    monkeypatch.delenv("OCULAR_INSECURE_NO_AUTH", raising=False)
    monkeypatch.setenv("OCULAR_API_KEY", "valid-secret-key")
    monkeypatch.delenv("OCULAR_ENABLE_DOCS", raising=False)

    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    sys.path.insert(0, str(clean_dist))
    try:
        app_mod = importlib.import_module("app")
        client = TestClient(app_mod.app)

        # D-02: Non-ASCII API key must return 401, not 500
        res_non_ascii = client.get(
            "/api/v1/schema",
            headers=[(b"x-api-key", b"non_ascii_\xff\xfe")],
        )
        assert res_non_ascii.status_code == 401

        # D-03: /health returns strictly {"status": "ok"}
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json() == {"status": "ok"}
        assert "compile_hash" not in res_health.json()
        assert "workflow" not in res_health.json()

        # D-04: Docs disabled by default
        res_docs = client.get("/docs")
        assert res_docs.status_code in (404, 401)
        res_openapi = client.get("/openapi.json")
        assert res_openapi.status_code in (404, 401)

        # D-05: CORS with ALLOWED_ORIGINS=* and credentials must fail at startup
        monkeypatch.setenv("ALLOWED_ORIGINS", "*")
        sys.modules.pop("app", None)
        with pytest.raises(RuntimeError, match="CORS"):
            importlib.import_module("app")

    finally:
        if str(clean_dist) in sys.path:
            sys.path.remove(str(clean_dist))
        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)


def test_e05_driver_cli_invalid_params_exit_code(clean_dist: Path):
    """E-05: Driver CLI entrypoint exits with non-zero code on invalid --params JSON."""
    import subprocess
    dag = {
        "workflow_name": "CliWorkflow",
        "target_url": "http://127.0.0.1:8080",
        "edges": [],
    }
    gen = JinjaCodeGenerator()
    gen.emit(dag, segments=[], out_dir=clean_dist)

    driver_path = clean_dist / "playwright_driver.py"
    # Execute driver with invalid JSON
    proc = subprocess.run(
        [sys.executable, str(driver_path), "--params", "{not_valid_json}"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, f"Expected non-zero exit code on invalid params JSON, got {proc.returncode}"
    assert "Invalid --params JSON" in proc.stderr or "Error" in proc.stderr


def test_d06_body_limit_auth_check_before_body(clean_dist: Path, monkeypatch):
    """D-06: Body-limit middleware verifies API key before reading or buffering body."""
    dag = {
        "workflow_name": "BodyLimitWorkflow",
        "target_url": "http://127.0.0.1:8080",
        "edges": [],
    }
    gen = JinjaCodeGenerator()
    gen.emit(dag, segments=[], out_dir=clean_dist)

    monkeypatch.delenv("OCULAR_INSECURE_NO_AUTH", raising=False)
    monkeypatch.setenv("OCULAR_API_KEY", "secret-test-key")

    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    sys.path.insert(0, str(clean_dist))
    try:
        app_mod = importlib.import_module("app")
        client = TestClient(app_mod.app)

        # Huge body without API key must return 401 Unauthorized immediately, NOT 413
        huge_payload = b"A" * (3 * 1024 * 1024)  # 3 MB > 2 MB cap
        res = client.post(
            "/api/v1/execute",
            content=huge_payload,
            headers={"Content-Type": "application/json"},
        )
        assert res.status_code == 401, f"Expected 401 Unauthorized before body check, got {res.status_code}"

        # Huge body with valid API key returns 413 Payload Too Large
        res_auth_huge = client.post(
            "/api/v1/execute",
            content=huge_payload,
            headers={"Content-Type": "application/json", "X-API-Key": "secret-test-key"},
        )
        assert res_auth_huge.status_code == 413
    finally:
        if str(clean_dist) in sys.path:
            sys.path.remove(str(clean_dist))
        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)
