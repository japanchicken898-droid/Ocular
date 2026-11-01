"""Security audit hardening unit and integration tests."""

import json
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
    from ocular.pipeline.state_graph import CompilationError
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
    # Malicious segment IDs violating ^[A-Za-z0-9_-]{1,64}$ must be rejected by CompilationError
    with pytest.raises(CompilationError):
        generator.emit(dag=dag, segments=malicious_segments, out_dir=out_dir)

    # Tricky labels, selectors, and URLs with a valid ID emit safely via workflow.json
    safe_segments = [
        {
            "id": "btn_safe",
            "label": 'Click me \n""" multi-line exploit """',
            "role": "button",
            "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
            "suggested_selector": 'div[data-hack="true\']"; drop_db(); #',
        }
    ]
    artifacts = generator.emit(dag=dag, segments=safe_segments, out_dir=out_dir)

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
    generator.emit(dag={"workflow_name": "Test", "target_url": "https://example.com"}, segments=[], out_dir=out_dir)

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

        # 3. Verify _is_server_online probe removed to prevent pre-flight TCP probes (F4)
        assert not hasattr(driver, "_is_server_online")
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


def test_workflow_name_digit_sanitization():
    """Verify that domains/IPs starting with numbers resolve to valid Python identifiers prepended with 'Workflow_'."""
    name_ip = derive_workflow_name_from_url("http://127.0.0.1:8000")
    assert name_ip.startswith("Workflow_")
    assert name_ip.isidentifier()
    assert name_ip == "Workflow_127Workflow"

    name_domain = derive_workflow_name_from_url("https://1password.com")
    assert name_domain.startswith("Workflow_")
    assert name_domain.isidentifier()
    assert name_domain == "Workflow_1passwordWorkflow"

    name_raw = sanitize_workflow_class_name("42Workflow")
    assert name_raw.startswith("Workflow_")
    assert name_raw.isidentifier()


def test_samples_package_collection():
    """Verify samples package imports cleanly and generate_synthetic_screen_recording is callable."""
    import samples
    from samples.create_sample_video import generate_synthetic_screen_recording

    assert hasattr(samples, "__file__")
    assert callable(generate_synthetic_screen_recording)


def test_driver_respects_step_timeout(tmp_path: Path):
    """Verify that synthesized edge timeout_ms values are emitted into the driver and respected."""
    generator = JinjaCodeGenerator()
    dag = {
        "workflow_name": "TimeoutTestWorkflow",
        "target_url": "https://example.com",
        "nodes": {
            "state_root": {"title": "Root"},
            "state_done": {"title": "Done"},
        },
        "edges": [
            {
                "from_node": "state_root",
                "to_node": "state_done",
                "action": "click",
                "trigger_element": "submit_btn",
                "timeout_ms": 4500,
            },
            {
                "from_node": "state_done",
                "to_node": "state_done",
                "action": "fill",
                "trigger_element": "query_input",
                "timeout_ms": 3200,
            },
        ],
    }
    segments = [
        {"id": "submit_btn", "bbox": {"x": 0.1, "y": 0.2, "width": 0.1, "height": 0.05}},
        {"id": "query_input", "bbox": {"x": 0.3, "y": 0.4, "width": 0.2, "height": 0.05}},
    ]

    out_dir = tmp_path / "timeout_dist"
    artifacts = generator.emit(dag=dag, segments=segments, out_dir=out_dir)

    wf_json = (out_dir / "workflow.json").read_text(encoding="utf-8")
    assert '"timeout_ms": 4500' in wf_json
    assert '"timeout_ms": 3200' in wf_json

    # Verify runtime driver instance has ACTION_STEPS with timeout_ms
    import sys
    import importlib

    sys.modules.pop("playwright_driver", None)
    sys.path.insert(0, str(out_dir.resolve()))
    try:
        driver_mod = importlib.import_module("playwright_driver")
        workflow_cls = getattr(driver_mod, "TimeoutTestWorkflow", getattr(driver_mod, "OcularPlaywrightDriver"))
        steps = workflow_cls.ACTION_STEPS
        assert len(steps) == 2
        assert steps[0]["timeout_ms"] == 4500
        assert steps[1]["timeout_ms"] == 3200
    finally:
        sys.modules.pop("playwright_driver", None)
        if str(out_dir.resolve()) in sys.path:
            sys.path.remove(str(out_dir.resolve()))


def test_browser_redirect_guard_blocks_private_ip(tmp_path: Path):
    """Verify that private and loopback subnets (RFC 1918 / 3927) are blocked by driver guard."""
    generator = JinjaCodeGenerator()
    out_dir = tmp_path / "guard_dist"
    generator.emit(dag={"workflow_name": "GuardWorkflow", "target_url": "https://example.com"}, segments=[], out_dir=out_dir)

    import sys
    import importlib

    sys.modules.pop("playwright_driver", None)
    sys.path.insert(0, str(out_dir.resolve()))
    try:
        driver_mod = importlib.import_module("playwright_driver")
        assert hasattr(driver_mod, "BLOCKED_PRIVATE_NETWORKS")
        is_priv = getattr(driver_mod, "is_private_url", getattr(driver_mod, "_is_private_url", None))
        assert is_priv is not None

        # Loopback
        assert is_priv("http://127.0.0.1/admin") is True
        assert is_priv("http://localhost:3000") is True

        # Cloud metadata service (RFC 3927)
        assert is_priv("http://169.254.169.254/latest/meta-data") is True

        # RFC 1918 Private networks
        assert is_priv("http://10.0.0.1/status") is True
        assert is_priv("https://172.16.0.5:8443/internal") is True
        assert is_priv("http://192.168.1.1/router") is True

        # Public domains/IPs must not be flagged as private
        assert is_priv("https://8.8.8.8") is False
    finally:
        sys.modules.pop("playwright_driver", None)
        if str(out_dir.resolve()) in sys.path:
            sys.path.remove(str(out_dir.resolve()))


def test_keyword_sanitization_in_workflow_name():
    """Verify that Python reserved keywords are escaped in workflow class and parameter names."""
    import keyword

    # Reserved keywords must be sanitized
    kw_name = sanitize_workflow_class_name("class")
    assert kw_name.isidentifier()
    assert not keyword.iskeyword(kw_name)
    assert kw_name == "class_Workflow"

    def_name = sanitize_workflow_class_name("def")
    assert def_name.isidentifier()
    assert not keyword.iskeyword(def_name)

    from ocular.pipeline.code_generator import _sanitize_param_key
    param_kw = _sanitize_param_key("for")
    assert param_kw.isidentifier()
    assert not keyword.iskeyword(param_kw)
    assert param_kw == "for_"


def test_context_level_ssrf_route_interception(tmp_path: Path):
    """Verify that route interception is attached at the BrowserContext level."""
    generator = JinjaCodeGenerator()
    out_dir = tmp_path / "context_ssrf_dist"
    artifacts = generator.emit(dag={"workflow_name": "ContextSSRFWorkflow", "target_url": "https://example.com"}, segments=[], out_dir=out_dir)

    driver_code = artifacts["playwright_driver"].read_text(encoding="utf-8")
    assert "await context.route(" in driver_code
    assert "_ssrf_route_guard" in driver_code


def test_recorder_sensitive_field_expansion(tmp_path: Path):
    """Verify that isSensitiveElement expands to contenteditable, textarea, and extended sensitive tokens."""
    recorder = OcularSessionRecorder(out_dir=tmp_path)
    script = recorder.TELEMETRY_INIT_SCRIPT
    assert "contenteditable" in script
    assert "textarea" in script
    for token in ("otp", "pin", "ssn", "one-time-code", "auth_key", "credit_card", "cvv", "password", "token", "secret"):
        assert token in script

    # Test server-side truncation and redaction with extended patterns
    event = {
        "event_type": "keydown",
        "action_type": "fill",
        "target_selector": "input#user_one_time_code",
        "key": "123456",
        "input_payload": "123456",
    }
    recorder.push_telemetry_event(event)
    stored = recorder.telemetry_events[-1]
    assert stored["key"] == "[REDACTED]"
    assert stored["input_payload"] == "[REDACTED]"


def test_robust_telemetry_deserialization_rejects_non_dict(tmp_path: Path):
    """Verify that load_telemetry_events safely rejects non-dict JSON entries (lists, strings, nulls, numbers)."""
    telem_file = tmp_path / "telemetry_events.jsonl"
    lines = [
        json.dumps({"event_type": "mousedown", "action_type": "click", "norm_x": 0.5, "norm_y": 0.5}),
        json.dumps(["invalid", "list"]),
        json.dumps("just a string"),
        json.dumps(12345),
        json.dumps(None),
        "not valid json at all",
        json.dumps({"event_type": "keydown", "action_type": "fill", "key": "a"}),
    ]
    telem_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    events = OcularSessionRecorder.load_telemetry_events(telem_file)
    assert len(events) == 2
    assert all(isinstance(e, dict) for e in events)

    # Verify push_telemetry_event rejects non-dict inputs
    recorder = OcularSessionRecorder(out_dir=tmp_path)
    recorder.push_telemetry_event(["invalid", "payload"])
    recorder.push_telemetry_event("raw string")
    recorder.push_telemetry_event(None)
    assert len(recorder.telemetry_events) == 0

    # Verify video_ingest processes corrupted telemetry without throwing AttributeError
    from ocular.pipeline.video_ingest import VideoIngestSampler
    import numpy as np

    sampler = VideoIngestSampler(sample_rate=2.0)
    corrupted_telemetry = [{"valid": 1}, ["bad"], "string", 42, None]
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    f_meta, trans, _ = sampler.process_frames_list(frames, telemetry_events=corrupted_telemetry)
    assert len(f_meta) == 3


def test_fail_explicit_ingestion_zero_actionable_elements():
    """Verify that build_dag raises CompilationError on zero actionable elements rather than falling back to TradingView."""
    from ocular.pipeline.state_graph import DAGBuilder, CompilationError

    builder = DAGBuilder(workflow_name="EmptyWorkflow")

    with pytest.raises(CompilationError) as exc_info:
        builder.build_dag(segments=[], transitions=[], telemetry_events=[])
    assert "zero actionable elements" in str(exc_info.value).lower()

    # Generic segments must synthesize a custom DAG rather than hardcoded TradingView states
    custom_segments = [
        {"id": "custom_login_btn", "label": "Login", "role": "button", "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05}},
        {"id": "custom_search_bar", "label": "Search", "role": "input", "bbox": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.05}},
    ]
    custom_dag = builder.build_dag(segments=custom_segments)
    assert "state_root" in custom_dag["nodes"]
    # Must NOT contain hardcoded TradingView interval dropdown states
    assert "state_interval_menu_open" not in custom_dag["nodes"]
    assert "state_interval_selected" not in custom_dag["nodes"]
    # Must contain custom segment states
    assert any("custom_login_btn" in nid for nid in custom_dag["nodes"])


def test_ssrf_blocks_ipv4_mapped_ipv6_and_schemes(tmp_path: Path):
    """Verify that IPv4-mapped IPv6, non-HTTP schemes, and alternate IPv4 encodings are blocked by SSRF guards."""
    from ocular.pipeline.code_generator import JinjaCodeGenerator
    import sys
    import importlib
    import pytest
    from fastapi import HTTPException

    generator = JinjaCodeGenerator()
    out_dir = tmp_path / "ssrf_guard_dist"
    generator.emit(dag={"workflow_name": "SsrfGuardWorkflow", "target_url": "https://example.com"}, segments=[], out_dir=out_dir)

    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    sys.path.insert(0, str(out_dir.resolve()))
    try:
        driver_mod = importlib.import_module("playwright_driver")
        app_mod = importlib.import_module("app")

        is_priv_driver = getattr(driver_mod, "is_private_url")
        is_priv_app = getattr(app_mod, "is_private_url")

        blocked_test_cases = [
            # IPv4-mapped IPv6
            "[::ffff:127.0.0.1]",
            "http://[::ffff:127.0.0.1]",
            "http://[::ffff:127.0.0.1]:8080/admin",
            "http://[::ffff:10.0.0.1]",
            "http://[::ffff:169.254.169.254]",
            # Non-HTTP / dangerous URI schemes
            "file:///etc/passwd",
            "file:///C:/Windows/win.ini",
            "data:text/html,<html>test</html>",
            "blob:http://example.com/uuid",
            "about:blank",
            "gopher://127.0.0.1:70/_status",
            # Encoded IPv4 variations (octal, hex, dword int)
            "http://0x7f000001",
            "http://2130706433",
            "http://0177.0.0.1",
            "http://017700000001",
        ]

        for url in blocked_test_cases:
            assert is_priv_driver(url) is True, f"driver should block {url}"
            assert is_priv_app(url) is True, f"app should block {url}"

        # Allowed public endpoints
        allowed_test_cases = [
            "https://8.8.8.8",
            "http://[::ffff:8.8.8.8]",
            "https://1.1.1.1/dns-query",
        ]
        for url in allowed_test_cases:
            assert is_priv_driver(url) is False, f"driver should allow {url}"
            assert is_priv_app(url) is False, f"app should allow {url}"

        # Test FastAPI validate_safe_url rejecting non-http and private targets
        validate_fn = getattr(app_mod, "validate_safe_url")
        for bad_url in [
            "file:///etc/passwd",
            "data:text/plain;base64,SGVsbG8=",
            "http://[::ffff:127.0.0.1]",
            "http://0x7f000001",
        ]:
            with pytest.raises(HTTPException) as exc_info:
                import asyncio
                asyncio.run(validate_fn(bad_url))
            assert exc_info.value.status_code == 400

    finally:
        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)
        if str(out_dir.resolve()) in sys.path:
            sys.path.remove(str(out_dir.resolve()))


def test_telemetry_numeric_field_coercion(tmp_path: Path):
    """Verify that string timestamps, coordinates, and nulls in telemetry are safely coerced without TypeErrors."""
    import numpy as np
    from ocular.recorder import OcularSessionRecorder
    from ocular.pipeline.video_ingest import VideoIngestSampler

    # 1. Test OcularSessionRecorder coercion
    raw_event_strings = {
        "event_type": "mousedown",
        "action_type": "click",
        "elapsed_ms": "1250",
        "timestamp_ms": "1250.5",
        "norm_x": "0.45",
        "norm_y": "0.85",
        "target_selector": "#submit-btn",
    }
    coerced = OcularSessionRecorder.coerce_telemetry_event(raw_event_strings)
    assert isinstance(coerced["elapsed_ms"], int)
    assert coerced["elapsed_ms"] == 1250
    assert isinstance(coerced["timestamp_ms"], float)
    assert coerced["timestamp_ms"] == 1250.5
    assert isinstance(coerced["norm_x"], float)
    assert coerced["norm_x"] == 0.45
    assert isinstance(coerced["norm_y"], float)
    assert coerced["norm_y"] == 0.85

    # 2. Test nulls and malformed non-numeric values
    raw_event_nulls = {
        "event_type": "keydown",
        "elapsed_ms": None,
        "timestamp_ms": None,
        "norm_x": None,
        "norm_y": None,
    }
    coerced_nulls = OcularSessionRecorder.coerce_telemetry_event(raw_event_nulls)
    assert coerced_nulls["elapsed_ms"] == 0
    assert coerced_nulls["timestamp_ms"] == 0.0
    assert coerced_nulls["norm_x"] == 0.0
    assert coerced_nulls["norm_y"] == 0.0

    raw_event_garbage = {
        "event_type": "mousedown",
        "elapsed_ms": "invalid_number",
        "norm_x": "NaN_value",
        "norm_y": {},
    }
    coerced_garbage = OcularSessionRecorder.coerce_telemetry_event(raw_event_garbage)
    assert coerced_garbage["elapsed_ms"] == 0
    assert coerced_garbage["norm_x"] == 0.0
    assert coerced_garbage["norm_y"] == 0.0

    # 3. Test VideoIngestSampler.coerce_telemetry_event matches
    coerced_sampler = VideoIngestSampler.coerce_telemetry_event(raw_event_strings)
    assert coerced_sampler["elapsed_ms"] == 1250
    assert coerced_sampler["norm_x"] == 0.45

    # 4. Test ingestion timeline with diverse string and null types without TypeError
    sampler = VideoIngestSampler(sample_rate=2.0)
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
    mixed_telemetry = [
        {"event_type": "mousedown", "action_type": "click", "elapsed_ms": "500", "norm_x": "0.1", "norm_y": "0.2"},
        {"event_type": "mousedown", "action_type": "click", "elapsed_ms": None, "norm_x": None, "norm_y": None},
        {"event_type": "mousedown", "action_type": "click", "elapsed_ms": 1500, "norm_x": 0.5, "norm_y": 0.5},
        {"event_type": "mousedown", "action_type": "click", "elapsed_ms": "invalid", "norm_x": "bad"},
        ["non-dict-ignored"],
        None,
    ]

    f_meta, transitions, _ = sampler.process_frames_list(frames, telemetry_events=mixed_telemetry)
    assert len(f_meta) == 4

    # 5. Verify timeline sorting by elapsed_ms never raises TypeError
    sorted_events = sorted(
        [e for e in mixed_telemetry if VideoIngestSampler.coerce_telemetry_event(e)],
        key=lambda ev: VideoIngestSampler.coerce_telemetry_event(ev)["elapsed_ms"]
    )
    assert len(sorted_events) == 4
    assert sorted_events[-1]["elapsed_ms"] == 1500





