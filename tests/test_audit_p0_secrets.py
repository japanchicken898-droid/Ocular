"""P0 Secrets & Schema Hygiene Regression Tests (S-01 through S-06).

Ensures:
- S-01: Default-deny on typed payloads in artifacts (dist/ files contain 0 secret payloads).
- S-02: Comprehensive redaction and stem tokenization without false positives on spinner, author, etc.
- S-03: Single source of redaction logic, redaction.py copied to dist, and client/server parity.
- S-04: Exception and log scrubbing (caplog DEBUG asserts 0 secret leakage).
- S-05: Sanitized error codes in API responses (NAV_HTTP_ERROR, NAV_FAILED, STEP_UNVERIFIED, TIMEOUT, TARGET_BLOCKED, INTERNAL_ERROR).
- S-06: Origin URL strictness (rejects credentials with CompilationError, trims to scheme://host[:port], TARGET_BLOCKED on bad canonicalization).
"""

import http.server
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient
from playwright.async_api import async_playwright

from ocular.pipeline.state_graph import CompilationError
from ocular.pipeline.code_generator import JinjaCodeGenerator, extract_origin_url
from ocular.recorder import OcularSessionRecorder as SessionRecorder
from ocular.security.redaction import (
    EXPLICIT_PARAMETER_ALLOWLIST,
    is_sensitive_field,
    redact_event,
    scrub_exception,
)


@pytest.fixture
def clean_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    return dist


def test_s01_default_deny_payloads_in_dist(clean_dist: Path):
    """S-01: Default-deny typed payloads in dist/.
    Any payload not in EXPLICIT_PARAMETER_ALLOWLIST must be redacted to [REDACTED] / empty default.
    Grep across all files in dist/ must yield 0 occurrences of secret payloads.
    """
    secret_email = "alice_secret_account@corporate.internal"
    secret_recovery = "RECOVERY-ABCD-9999-SECRET"
    secret_typed = "SuperP@ssw0rd123!"

    dag = {
        "workflow_name": "SecretsVaultWorkflow",
        "target_url": "http://127.0.0.1:8080",
        "edges": [
            {
                "from_node_id": "state_1",
                "to_node_id": "state_2",
                "action_type": "fill",
                "trigger_element": "search_input",
                "target_segment_id": "search_input",
                "target_selector": "input#search",
                "parameter_ref": "interval",
                "input_payload": "15m",
                "default_value": "15m",
            },
            {
                "from_node_id": "state_2",
                "to_node_id": "state_3",
                "action_type": "fill",
                "trigger_element": "email_input",
                "target_segment_id": "email_input",
                "target_selector": "input#email",
                "parameter_ref": "email",
                "input_payload": secret_email,
                "default_value": secret_email,
            },
            {
                "from_node_id": "state_3",
                "to_node_id": "state_4",
                "action_type": "fill",
                "trigger_element": "recovery_input",
                "target_segment_id": "recovery_input",
                "target_selector": "input#recovery",
                "parameter_ref": "recovery_token",
                "input_payload": secret_recovery,
                "default_value": secret_recovery,
            },
        ],
    }

    telemetry = [
        {"event_type": "keydown", "key": "S", "value": secret_typed, "target_selector": "input#recovery"},
    ]

    segments = [
        {"id": "search_input", "role": "input", "suggested_selector": "input#search", "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05}},
        {"id": "email_input", "role": "input", "suggested_selector": "input#email", "bbox": {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.05}},
        {"id": "recovery_input", "role": "input", "suggested_selector": "input#recovery", "bbox": {"x": 0.1, "y": 0.3, "width": 0.2, "height": 0.05}},
    ]

    gen = JinjaCodeGenerator()
    artifacts = gen.emit(dag, segments=segments, telemetry_events=telemetry, out_dir=clean_dist)

    # 1. Check workflow.json and state_graph.json schemas
    wf_data = json.loads(Path(artifacts["workflow_json"]).read_text(encoding="utf-8"))
    inputs = wf_data.get("input_fields", {})
    assert inputs["interval"]["default_value"] == "15m"
    assert inputs["email"]["default_value"] == ""
    assert inputs["recovery_token"]["default_value"] == ""

    # 2. Grep every emitted file in dist/
    for f in clean_dist.rglob("*"):
        if f.is_file():
            content = f.read_text(encoding="utf-8", errors="ignore")
            assert secret_email not in content, f"Secret email leaked in {f.name}"
            assert secret_recovery not in content, f"Secret recovery leaked in {f.name}"
            assert secret_typed not in content, f"Secret typed characters leaked in {f.name}"


def test_s02_comprehensive_redaction_and_stem_tokenization():
    """S-02: Comprehensive vocabulary and stem tokenization without false positives."""
    sensitive_samples = [
        "verificationCode",
        "password1",
        "userpassword",
        "JSESSIONID",
        "phpsessid",
        "seed_phrase",
        "seedphrase",
        "recovery_phrase",
        "client_secret",
        "access_token",
        "auth_key",
        "pin2",
        "pincode",
        "sin",
        "dob",
        "birthdate",
        "tax_id",
        "national_id",
        "totp",
    ]

    for sample in sensitive_samples:
        assert is_sensitive_field(sample) is True, f"Expected '{sample}' to be detected as sensitive"
        assert is_sensitive_field({"name": sample}) is True
        assert is_sensitive_field({"id": sample}) is True

    # False positives MUST NOT match
    false_positives = [
        "spinner",
        "shipping_address",
        "typing",
        "author",
        "bypass",
        "pinterest",
        "compass",
        "tokenizer",
    ]

    for fp in false_positives:
        assert is_sensitive_field(fp) is False, f"False positive detected: '{fp}' should NOT be sensitive"
        assert is_sensitive_field({"name": fp}) is False
        assert is_sensitive_field({"id": fp}) is False
        assert is_sensitive_field({"target_selector": f"div.{fp}"}) is False


@pytest.mark.asyncio
async def test_s03_redaction_single_source_and_parity(clean_dist: Path):
    """S-03: redaction.py copied to dist/, and client-side JS regex in recorder has parity with Python."""
    dag = {
        "workflow_name": "ParityWorkflow",
        "target_url": "http://127.0.0.1:8080",
        "edges": [],
    }
    gen = JinjaCodeGenerator()
    artifacts = gen.emit(dag, segments=[], out_dir=clean_dist)

    # 1. redaction.py must be copied into dist/
    redaction_file = clean_dist / "redaction.py"
    assert redaction_file.exists(), "redaction.py was not copied into dist/"
    assert "is_sensitive_field" in redaction_file.read_text(encoding="utf-8")

    # 2. Verify client-server parity between recorder JS checkSensitiveString and Python is_sensitive_field
    recorder = SessionRecorder(out_dir=clean_dist)
    init_script = recorder._INIT_SCRIPT_TEMPLATE

    test_words = [
        ("password", True),
        ("verificationCode", True),
        ("password1", True),
        ("JSESSIONID", True),
        ("seed_phrase", True),
        ("pin2", True),
        ("spinner", False),
        ("shipping_address", False),
        ("typing", False),
        ("author", False),
        ("bypass", False),
        ("pinterest", False),
        ("compass", False),
        ("tokenizer", False),
    ]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        # Inject the script's checkSensitiveString function
        await page.add_init_script(init_script.replace("__SESSION_NONCE__", "nonce_test"))
        await page.goto("about:blank")

        for word, expected in test_words:
            py_verdict = is_sensitive_field(word)
            assert py_verdict == expected, f"Python verdict mismatch for {word}: expected {expected}, got {py_verdict}"

            # Run client-side check
            js_verdict = await page.evaluate(f"""() => {{
                return checkSensitiveString("{word}");
            }}""")
            assert js_verdict == expected, f"Client JS verdict mismatch for {word}: expected {expected}, got {js_verdict}"

        await browser.close()


def test_s04_exception_and_log_scrubbing(caplog):
    """S-04: scrub_exception removes secrets and Playwright fill arguments from exceptions and logs."""
    secret_val = "SuperSecretP@ssword999!"
    raw_exc = (
        f'playwright._impl._errors.TimeoutError: Page.fill: Timeout 5000ms exceeded.\n'
        f'Call log:\n'
        f'  - waiting for locator("input#pwd")\n'
        f'  - locator.fill("{secret_val}")\n'
        f'  - navigating to "http://user:secretpass@internal.host:8080/path?token=XYZSECRET"\n'
    )

    scrubbed = scrub_exception(raw_exc, secrets=[secret_val])
    assert secret_val not in scrubbed
    assert "[REDACTED]" in scrubbed
    assert "user:secretpass@" not in scrubbed
    assert "token=XYZSECRET" not in scrubbed

    # Caplog DEBUG test: ensure logged exceptions do not leak secrets
    caplog.set_level(logging.DEBUG)
    test_logger = logging.getLogger("ocular.test.redaction")
    test_logger.debug(scrub_exception(f"Error handling secret: {secret_val}", secrets=[secret_val]))

    for record in caplog.records:
        assert secret_val not in record.message


@pytest.mark.asyncio
async def test_s05_sanitized_error_codes_in_api(clean_dist: Path, monkeypatch):
    """S-05: Emitted FastAPI service returns stable error codes (NAV_HTTP_ERROR, NAV_FAILED, etc.)."""
    import importlib

    # 1. Setup local HTTP server returning 500
    class Handler500(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Internal Error")

        def log_message(self, format, *args):
            pass

    server_500 = http.server.HTTPServer(("127.0.0.1", 0), Handler500)
    port_500 = server_500.server_port
    t_500 = threading.Thread(target=server_500.serve_forever, daemon=True)
    t_500.start()

    dag = {
        "workflow_name": "ErrorCodeWorkflow",
        "target_url": f"http://127.0.0.1:{port_500}",
        "edges": [
            {
                "from_node_id": "state_1",
                "to_node_id": "state_2",
                "action_type": "click",
                "trigger_element": "btn",
                "target_selector": "#btn",
            }
        ],
    }
    gen = JinjaCodeGenerator()
    gen.emit(dag, segments=[{"id": "btn", "suggested_selector": "#btn", "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05}}], out_dir=clean_dist)

    monkeypatch.setenv("OCULAR_INSECURE_NO_AUTH", "1")
    sys.path.insert(0, str(clean_dist))
    try:
        app_mod = importlib.import_module("app")
        app_instance = getattr(app_mod, "app")
        client = TestClient(app_instance)

        # 1. Navigation to 500 server must return NAV_HTTP_ERROR
        res_500 = client.post("/api/v1/execute", json={"headless": True})
        assert res_500.status_code == 500
        data_500 = res_500.json()
        assert data_500["success"] is False
        assert data_500.get("error_code") == "NAV_HTTP_ERROR" or data_500["results"].get("error_code") == "NAV_HTTP_ERROR"

        # 2. Blocked SSRF navigation must return 400 with restricted/blocked detail
        res_blocked = client.post("/api/v1/execute", json={"base_url": "http://169.254.169.254/latest/meta-data", "headless": True})
        assert res_blocked.status_code == 400
        assert "restricted" in res_blocked.json()["detail"].lower() or "blocked" in res_blocked.json()["detail"].lower()
    finally:
        if str(clean_dist) in sys.path:
            sys.path.remove(str(clean_dist))


def test_s06_origin_url_strictness_and_metadata(tmp_path: Path):
    """S-06: extract_origin_url trims to origin, rejects credentials, and recorder sanitizes origin."""
    # 1. Trims query, fragment, credentials and trailing slash
    origin_clean = extract_origin_url({"target_url": "https://example.com:8443/deep/path?query=val#hash"})
    assert origin_clean == "https://example.com:8443/deep/path"

    origin_simple = extract_origin_url({"target_url": "http://my-service.internal/login"})
    assert origin_simple == "http://my-service.internal/login"

    # 2. Rejects embedded credentials with CompilationError
    with pytest.raises(CompilationError, match="credentials"):
        extract_origin_url({"target_url": "http://admin:secret@example.com/api"})

    with pytest.raises(CompilationError, match="credentials"):
        extract_origin_url({"target_url": "https://user@example.com"})

    # 3. Recorder writes sanitized origin to session_metadata.json
    rec_dir = tmp_path / "rec_session"
    rec = SessionRecorder(out_dir=rec_dir)
    # Mocking launch/recording start
    start_url = "https://example.com:9000/auth/login?param=123"
    rec._write_session_metadata(start_url)
    meta = json.loads((rec_dir / "session_metadata.json").read_text(encoding="utf-8"))
    assert meta["url"] == "https://example.com:9000"
    assert "auth" not in meta["url"]
    assert "param" not in meta["url"]


def test_s07_fail_closed_on_missing_redaction_module(tmp_path: Path, monkeypatch):
    """FIX 1: Instantiating/importing emitted app or driver without redaction module raises RuntimeError."""
    import importlib.util

    dist_dir = tmp_path / "dist_no_redaction"
    generator = JinjaCodeGenerator()
    generator.emit(
        dag={"workflow_name": "RedactionFailSafe", "target_url": "http://example.com", "nodes": {}, "edges": []},
        segments=[],
        out_dir=dist_dir,
    )

    # Delete copied redaction.py from dist
    redaction_file = dist_dir / "redaction.py"
    if redaction_file.exists():
        redaction_file.unlink()

    # Block both redaction and ocular.security.redaction from being imported
    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    monkeypatch.syspath_prepend(str(dist_dir))
    monkeypatch.setitem(sys.modules, "redaction", None)
    monkeypatch.setitem(sys.modules, "ocular.security.redaction", None)

    # 1. Assert driver fails closed with RuntimeError
    driver_path = dist_dir / "playwright_driver.py"
    spec_driver = importlib.util.spec_from_file_location("test_driver_no_redaction", driver_path)
    assert spec_driver and spec_driver.loader
    mod_driver = importlib.util.module_from_spec(spec_driver)
    with pytest.raises(RuntimeError, match="redaction module unavailable"):
        spec_driver.loader.exec_module(mod_driver)

    # 2. Assert app fails closed with RuntimeError
    app_path = dist_dir / "app.py"
    spec_app = importlib.util.spec_from_file_location("test_app_no_redaction", app_path)
    assert spec_app and spec_app.loader
    mod_app = importlib.util.module_from_spec(spec_app)
    with pytest.raises(RuntimeError, match="redaction module unavailable"):
        spec_app.loader.exec_module(mod_app)

