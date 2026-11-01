"""Security audit hardening unit and integration tests."""

import os
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from ocular.pipeline.code_generator import (
    JinjaCodeGenerator,
    derive_workflow_name_from_url,
    sanitize_workflow_class_name,
)
from ocular.recorder import OcularSessionRecorder
from ocular.cli import compile_command
from samples.create_sample_video import generate_synthetic_screen_recording


def test_workflow_name_sanitization_alphabetic_prefix():
    # Digits must be prefixed with Workflow_
    name1 = sanitize_workflow_class_name("127_0_0_1")
    assert name1.startswith("Workflow_")
    assert name1.isidentifier()

    name2 = derive_workflow_name_from_url("http://127.0.0.1:8000")
    assert name2.startswith("Workflow_")
    assert name2.isidentifier()

    # Injection characters must be stripped/replaced
    name3 = sanitize_workflow_class_name("test'; import os; class Hack")
    assert name3.isidentifier()
    assert " " not in name3
    assert ";" not in name3


def test_code_generation_injection_resilience(tmp_path: Path):
    generator = JinjaCodeGenerator()
    malicious_segments = [
        {
            "id": 'btn_safe"; malicious_call(); #',
            "label": 'Click me \n""" multi-line exploit """',
            "role": "button",
            "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
            "suggested_selector": 'div[data-hack="true\']"; drop_db(); #',
        }
    ]
    dag = {
        "workflow_name": "123_exploit_workflow",
        "target_url": 'https://example.com/exploit?q="; eval()',
        "nodes": {},
        "edges": [],
    }

    out_dir = tmp_path / "hardened_dist"
    artifacts = generator.emit(dag=dag, segments=malicious_segments, out_dir=out_dir)

    driver_code = artifacts["playwright_driver"].read_text(encoding="utf-8")
    app_code = artifacts["app"].read_text(encoding="utf-8")

    # Both files must parse cleanly as valid Python without SyntaxError
    compile(driver_code, str(artifacts["playwright_driver"]), "exec")
    compile(app_code, str(artifacts["app"]), "exec")


def test_ssrf_protection_and_local_driver_isolation(tmp_path: Path):
    video_path = tmp_path / "sample.mp4"
    generate_synthetic_screen_recording(video_path, fps=30, width=1280, height=720)

    dist_dir = tmp_path / "ssrf_dist"
    compile_command(input_path=str(video_path), out_dir=str(dist_dir))

    import sys
    import importlib

    sys.path.insert(0, str(dist_dir.resolve()))
    try:
        app_mod = importlib.import_module("app")
        app = app_mod.app
        client = TestClient(app)

        # 1. Attempt SSRF to AWS metadata IP: must be blocked with HTTP 400
        res_ssrf_metadata = client.post(
            "/api/v1/execute",
            json={
                "base_url": "http://169.254.169.254/latest/meta-data",
                "headless": True,
            },
        )
        assert res_ssrf_metadata.status_code == 400
        assert "restricted" in res_ssrf_metadata.json()["detail"].lower() or "blocked" in res_ssrf_metadata.json()["detail"].lower()

        # 2. Attempt SSRF to loopback IP: must be blocked with HTTP 400
        res_ssrf_loopback = client.post(
            "/api/v1/execute",
            json={
                "base_url": "http://127.0.0.1:8000/internal",
                "headless": True,
            },
        )
        assert res_ssrf_loopback.status_code == 400

        # 3. Test that no global driver singleton is mutated across requests
        assert not hasattr(app_mod, "driver") or app_mod.driver is None

    finally:
        if str(dist_dir.resolve()) in sys.path:
            sys.path.remove(str(dist_dir.resolve()))


def test_recorder_telemetry_hardening(tmp_path: Path):
    recorder = OcularSessionRecorder(out_dir=tmp_path)

    # 1. Iframe defense present in init script
    assert "if (window !== window.top) return;" in recorder.TELEMETRY_INIT_SCRIPT

    # 2. String attribute truncation
    long_selector = "div#" + ("a" * 500)
    event = {
        "event_type": "mousedown",
        "action_type": "click",
        "target_selector": long_selector,
        "norm_x": 0.5,
        "norm_y": 0.5,
    }
    recorder.push_telemetry_event(event)
    pushed = recorder.telemetry_events[0]
    assert len(pushed["target_selector"]) <= 256

    # 3. Maximum telemetry event cap (2,500)
    recorder.telemetry_events = [{"dummy": i} for i in range(2500)]
    recorder.push_telemetry_event({"event_type": "overflow"})
    assert len(recorder.telemetry_events) == 2500


def test_credential_and_password_redaction(tmp_path: Path):
    # 1. Check client-side init script contains sensitive element detection
    recorder = OcularSessionRecorder(out_dir=tmp_path)
    assert "isSensitiveElement" in recorder.TELEMETRY_INIT_SCRIPT
    assert "password" in recorder.TELEMETRY_INIT_SCRIPT
    assert "[REDACTED]" in recorder.TELEMETRY_INIT_SCRIPT

    # 2. Server-side redaction of password events
    sensitive_event = {
        "event_type": "keydown",
        "action_type": "fill",
        "target_selector": "input#user_password",
        "key": "P@ssw0rd123!",
        "code": "KeyP",
        "input_payload": "P@ssw0rd123!",
    }
    recorder.push_telemetry_event(sensitive_event)
    stored = recorder.telemetry_events[-1]
    assert stored["key"] == "[REDACTED]"
    assert stored["input_payload"] == "[REDACTED]"

    # 3. Parameters schema in DAGBuilder redacts secrets
    from ocular.pipeline.state_graph import DAGBuilder
    builder = DAGBuilder(workflow_name="SecretWorkflow")
    nodes = {"state_login": {}}
    edges = [
        {
            "action_type": "fill",
            "trigger_element": "password_input",
            "input_payload": "secretTokenABC",
            "default_value": "secretTokenABC",
        }
    ]
    schema = builder._synthesize_parameters_schema(nodes, edges, [])
    for prop in schema["properties"].values():
        assert prop["default"] == "" or prop["default"] != "secretTokenABC"


def test_memory_optimization_normalized_segments_and_single_pass(tmp_path: Path):
    from ocular.pipeline.state_graph import DAGBuilder
    from ocular.pipeline.video_ingest import VideoIngestSampler
    import unittest.mock as mock

    builder = DAGBuilder(workflow_name="MemoryOptWorkflow")
    sample_segments = [
        {"id": "btn_1", "label": "Button 1", "role": "button", "bbox": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1}},
        {"id": "btn_2", "label": "Button 2", "role": "button", "bbox": {"x": 0.2, "y": 0.2, "width": 0.1, "height": 0.1}},
    ]
    dag = builder.build_dag(sample_segments)

    # Segments must be stored in top-level table, not duplicated in every node
    assert "segments" in dag
    assert isinstance(dag["segments"], dict)
    assert "btn_1" in dag["segments"]

    for node_data in dag["nodes"].values():
        assert "segment_ids" in node_data
        assert "segments" not in node_data or not isinstance(node_data.get("segments"), list)

    # Ingest pipeline single pass test
    sampler = VideoIngestSampler()
    with mock.patch.object(
        sampler,
        "sample_transitions",
        return_value=(
            {"fps": 30, "width": 1280, "height": 720, "resolution": "1280x720", "total_frames": 2, "sampled_count": 2},
            [],
            [],
        ),
    ) as mock_sample:
        sampler.process_video("dummy_video.mp4")
        assert mock_sample.call_count == 1


def test_safe_redirect_handler_strips_auth_header():
    import urllib.request
    from ocular.pipeline.segmenter import SafeRedirectHandler

    handler = SafeRedirectHandler()
    req = urllib.request.Request(
        "https://api.meta.ai/v1/responses",
        headers={"Authorization": "Bearer super-secret-key"},
    )

    # Redirect to same host must preserve Authorization
    same_host_req = handler.redirect_request(
        req, None, 302, "Found", {}, "https://api.meta.ai/v1/responses_redirected"
    )
    assert same_host_req is not None
    assert "Authorization" in same_host_req.headers

    # Redirect to different host must strip Authorization
    diff_host_req = handler.redirect_request(
        req, None, 302, "Found", {}, "https://attacker.example.com/steal"
    )
    assert diff_host_req is not None
    assert "Authorization" not in diff_host_req.headers
    assert "authorization" not in diff_host_req.headers


def test_baseten_api_key_isolation(monkeypatch):
    from ocular.pipeline.state_graph import DAGBuilder

    builder = DAGBuilder(workflow_name="IsolationWorkflow")
    # Set META_MODEL_API_KEY but delete BASETEN_API_KEY
    monkeypatch.setenv("META_MODEL_API_KEY", "meta-token-should-not-leak")
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)

    # Must return None without attempting to send META_MODEL_API_KEY to Baseten
    assert builder._call_muse_spark_api([]) is None


def test_cors_and_request_body_size_limit(tmp_path: Path):
    video_path = tmp_path / "sample.mp4"
    generate_synthetic_screen_recording(video_path, fps=30, width=1280, height=720)

    dist_dir = tmp_path / "cors_dist"
    compile_command(input_path=str(video_path), out_dir=str(dist_dir))

    import sys
    import importlib

    sys.path.insert(0, str(dist_dir.resolve()))
    try:
        app_mod = importlib.import_module("app")
        app = app_mod.app
        client = TestClient(app)

        # 1. Verify 2 MB request body limit returns 413
        oversized_str = "x" * (2 * 1024 * 1024 + 1024)
        res_large = client.post(
            "/api/v1/execute",
            content=oversized_str,
            headers={"Content-Type": "application/json", "Content-Length": str(len(oversized_str))},
        )
        assert res_large.status_code == 413
    finally:
        if str(dist_dir.resolve()) in sys.path:
            sys.path.remove(str(dist_dir.resolve()))


def test_mermaid_sanitization_and_symlink_defense(tmp_path: Path):
    from ocular.pipeline.state_graph import generate_mermaid_dag, export_mermaid_dag
    from ocular.pipeline.code_generator import JinjaCodeGenerator

    # 1. Mermaid syntax injection defense
    malicious_dag = {
        "workflow_name": 'Exploit\n" \t; graph LR; evil()',
        "root_node": "state_root",
        "terminal_node": "state_term",
        "nodes": {
            'state_root" \n evil': {
                "title": 'State "Quote" \\Backslash \n Control\x00',
                "preconditions": [{"type": "check<evil>"}],
            },
            "state_term": {
                "title": 'Terminal "Node"',
            },
        },
        "edges": [
            {
                "from_node": 'state_root" \n evil',
                "to_node": "state_term",
                "action": 'click" \n break',
                "trigger_element": 'btn<bad>"',
            }
        ],
    }
    diagram = generate_mermaid_dag(malicious_dag)
    assert "\x00" not in diagram
    assert "\\" not in diagram

    # 2. Symlink write defense
    out_dir = tmp_path / "symlink_test_dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_hijack = tmp_path / "hijacked.txt"
    target_hijack.write_text("original content")

    symlink_driver = out_dir / "playwright_driver.py"
    symlink_driver.symlink_to(target_hijack)

    generator = JinjaCodeGenerator()
    generator.emit(dag={"workflow_name": "Test"}, segments=[], out_dir=out_dir)

    # Symlink must be unlinked and original hijacked file untouched
    assert not symlink_driver.is_symlink()
    assert target_hijack.read_text() == "original content"


def test_cli_bracket_paths_markup_escape():
    # Calling compile_command with bracketed path should not raise rich.errors.MarkupError
    from ocular.cli import compile_command

    exit_code = compile_command(
        input_path="/tmp/[invalid_bracket_test]/video[1].mp4",
        out_dir="/tmp/[bracket_dist]",
    )
    # File not found exit code is 1, but no MarkupError exception raised
    assert exit_code == 1


@pytest.mark.asyncio
async def test_driver_live_mode_fails_without_tautological_fallback(tmp_path: Path):
    """Verify that live mode fails explicitly when connection fails, rather than falling back to DEMO_HTML."""
    video_path = tmp_path / "sample.mp4"
    generate_synthetic_screen_recording(video_path, fps=30, width=1280, height=720)

    dist_dir = tmp_path / "driver_live_test"
    compile_command(input_path=str(video_path), out_dir=str(dist_dir))

    import sys
    import importlib

    sys.path.insert(0, str(dist_dir.resolve()))
    try:
        driver_mod = importlib.import_module("playwright_driver")
        workflow_cls = getattr(driver_mod, "OcularPlaywrightDriver")

        # 1. Unreachable port in live mode must return success: False and status: 'error'
        driver = workflow_cls(
            base_url="http://127.0.0.1:59999",
            headless=True,
            live=True,
        )
        res = await driver.execute_workflow()
        assert res.get("success") is False
        assert res.get("status") == "error"
        assert "Live connection to 'http://127.0.0.1:59999' failed" in res.get("error", "")

        # 2. Verify that --base-url is honored over TARGET_URL in live mode
        driver_override = workflow_cls(
            base_url="https://override.example.com",
            live=True,
        )
        assert driver_override.base_url == "https://override.example.com"
        assert driver_override.live is True

        # 3. Test non-blocking _is_server_online probe
        is_online = await driver._is_server_online(timeout_sec=0.1)
        assert is_online is False
    finally:
        if str(dist_dir.resolve()) in sys.path:
            sys.path.remove(str(dist_dir.resolve()))


def test_pydantic_422_parameter_validation_and_async_dns(tmp_path: Path):
    """Verify that parameter type mismatches return clean HTTP 422 and validate_safe_url is non-blocking."""
    video_path = tmp_path / "sample.mp4"
    generate_synthetic_screen_recording(video_path, fps=30, width=1280, height=720)

    dist_dir = tmp_path / "pydantic_validation_dist"
    compile_command(input_path=str(video_path), out_dir=str(dist_dir))

    import sys
    import importlib

    sys.path.insert(0, str(dist_dir.resolve()))
    try:
        app_mod = importlib.import_module("app")
        app = app_mod.app
        client = TestClient(app)

        # 1. Parameter type mismatch in nested parameters dict must return 422
        res_mismatch = client.post(
            "/api/v1/execute",
            json={
                "parameters": {
                    "viewport_width": "not-a-valid-int",
                }
            },
        )
        assert res_mismatch.status_code == 422

        # 2. Parameter type mismatch in top-level request body must return 422
        res_top_mismatch = client.post(
            "/api/v1/execute",
            json={
                "viewport_width": "not-an-int",
            },
        )
        assert res_top_mismatch.status_code == 422

        # 3. Non-blocking validate_safe_url check
        import asyncio
        from unittest.mock import patch
        loop = asyncio.new_event_loop()
        try:
            # Valid domain with mocked DNS resolution to public IP
            with patch.object(
                loop, "getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 0))]
            ):
                validated = loop.run_until_complete(
                    app_mod.validate_safe_url("https://example.com")
                )
                assert validated == "https://example.com"

            # Blocked SSRF loopback
            with pytest.raises(Exception) as exc_info:
                loop.run_until_complete(
                    app_mod.validate_safe_url("http://127.0.0.1:8080")
                )
            assert "blocked loopback" in str(exc_info.value).lower()
        finally:
            loop.close()
    finally:
        if str(dist_dir.resolve()) in sys.path:
            sys.path.remove(str(dist_dir.resolve()))



