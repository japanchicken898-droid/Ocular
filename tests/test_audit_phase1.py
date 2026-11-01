"""Regression tests for Phase 1 Secrets Hardening (F3, F12, F11).

Tests:
- Token-matching word boundaries: negative tests (spinner, shipping_address, typing, author)
  and positive tests (pwd, cvc, cardnumber, iban, jwt, api_key, etc.)
- Non-string values (lists/dicts) on sensitive targets are replaced wholesale with "[REDACTED]"
- 100KB nested list / oversized payloads rejected/truncated by recorder
- Typed secret text never appears in any emitted file in dist
- Parameter keys are never derived from input_payload/typed text
- Shadow-DOM password in real Chromium is redacted
- Service and driver execution responses redact secrets in results.parameters and steps[].value
- /api/v1/schema does not expose default values for sensitive parameters
"""

import asyncio
import importlib
import json
from pathlib import Path
import sys
import pytest
from starlette.testclient import TestClient

from ocular.pipeline.code_generator import JinjaCodeGenerator
from ocular.pipeline.state_graph import DAGBuilder
from ocular.recorder import OcularSessionRecorder


def test_token_matching_boundaries():
    from ocular.security.redaction import is_sensitive_field

    # Negative tests: MUST NOT be flagged as sensitive
    negative_cases = [
        {"name": "spinner"},
        {"id": "shipping_address"},
        {"target_selector": "#typing_indicator"},
        {"label": "book_author"},
        {"name": "footprint_tracker"},
        {"label": "Search Author Name"},
    ]
    for case in negative_cases:
        assert not is_sensitive_field(case), f"False positive on non-sensitive field: {case}"

    # Positive tests: MUST be flagged as sensitive
    positive_cases = [
        {"type": "password"},
        {"autocomplete": "current-password"},
        {"autocomplete": "cc-number"},
        {"autocomplete": "one-time-code"},
        {"name": "user_password"},
        {"id": "user_pin"},
        {"target_selector": "input#pwd"},
        {"label": "Enter CVC"},
        {"name": "cardnumber"},
        {"name": "iban_code"},
        {"name": "jwt_token"},
        {"placeholder": "Your API Key"},
        {"aria_label": "Enter passcode"},
        {"target_selector": "input[name='secret_key']"},
    ]
    for case in positive_cases:
        assert is_sensitive_field(case), f"False negative on sensitive field: {case}"


def test_redact_event_non_string_values():
    from ocular.security.redaction import redact_event

    # Sensitive event with list and dict values
    event_list = {
        "event_type": "keydown",
        "action_type": "fill",
        "target_selector": "input[type='password']",
        "key": ["s", "3", "c"],
        "code": ["KeyS", "Digit3", "KeyC"],
        "input_payload": {"nested": "secret_data"},
    }
    redacted = redact_event(event_list)
    assert redacted["key"] == "[REDACTED]"
    assert redacted["code"] == "[REDACTED]"
    assert redacted["input_payload"] == "[REDACTED]"

    # Non-sensitive event should retain normal values (subject to type constraints)
    safe_event = {
        "event_type": "keydown",
        "action_type": "fill",
        "target_selector": "#author_name",
        "key": "a",
        "code": "KeyA",
    }
    safe_redacted = redact_event(safe_event)
    assert safe_redacted["key"] == "a"
    assert safe_redacted["code"] == "KeyA"


def test_oversized_payload_rejected(tmp_path: Path):
    rec = OcularSessionRecorder(out_dir=tmp_path, headless=True)
    # 100KB nested list
    huge_payload = {
        "event_type": "keydown",
        "action_type": "fill",
        "target_selector": "#search",
        "key": "a",
        "nested_junk": [["nested" * 50] * 50] * 50,
        "_nonce": getattr(rec, "session_nonce", "test_nonce"),
    }
    rec.push_telemetry_event(huge_payload)
    # The event should be rejected or truncated such that recorded events do not exceed 4KB
    if rec.telemetry_events:
        event_str = json.dumps(rec.telemetry_events[-1])
        assert len(event_str.encode("utf-8")) <= 4096
    else:
        assert len(rec.telemetry_events) == 0


def test_secrets_never_appear_in_any_emitted_file(tmp_path: Path):
    out_dir = tmp_path / "dist_secrets"

    super_secret_pw = "SuperSecretP@ssw0rd999!"
    secret_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secretpayload"
    secret_card = "4111222233334444"
    secret_iban = "DE89370400440532013000"
    secret_cvc = "888"

    # Telemetry events with non-obvious IDs but sensitive attributes/selectors
    telemetry_events = [
        {
            "event_type": "keydown",
            "action_type": "fill",
            "target_selector": "input#field_98124[type='password']",
            "target_tag": "input",
            "key": super_secret_pw,
            "input_payload": super_secret_pw,
            "norm_x": 0.2,
            "norm_y": 0.3,
            "elapsed_ms": 100,
            "timestamp_ms": 1000,
        },
        {
            "event_type": "keydown",
            "action_type": "fill",
            "target_selector": "#field_jwt",
            "target_tag": "input",
            "key": secret_jwt,
            "input_payload": secret_jwt,
            "norm_x": 0.2,
            "norm_y": 0.4,
            "elapsed_ms": 200,
            "timestamp_ms": 1100,
        },
        {
            "event_type": "keydown",
            "action_type": "fill",
            "target_selector": "#user_cardnumber",
            "target_tag": "input",
            "key": secret_card,
            "input_payload": secret_card,
            "norm_x": 0.2,
            "norm_y": 0.5,
            "elapsed_ms": 300,
            "timestamp_ms": 1200,
        },
        {
            "event_type": "keydown",
            "action_type": "fill",
            "target_selector": "#field_iban",
            "target_tag": "input",
            "key": secret_iban,
            "input_payload": secret_iban,
            "norm_x": 0.2,
            "norm_y": 0.6,
            "elapsed_ms": 400,
            "timestamp_ms": 1300,
        },
        {
            "event_type": "keydown",
            "action_type": "fill",
            "target_selector": "#field_cvc",
            "target_tag": "input",
            "key": secret_cvc,
            "input_payload": secret_cvc,
            "norm_x": 0.2,
            "norm_y": 0.7,
            "elapsed_ms": 500,
            "timestamp_ms": 1400,
        },
    ]

    segments = [
        {"id": "field_98124", "label": "Secret Field", "role": "input", "suggested_selector": "input#field_98124[type='password']", "bbox": {"x": 0.2, "y": 0.3, "width": 0.2, "height": 0.05}},
        {"id": "field_jwt", "label": "JWT Field", "role": "input", "suggested_selector": "#field_jwt", "bbox": {"x": 0.2, "y": 0.4, "width": 0.2, "height": 0.05}},
        {"id": "user_cardnumber", "label": "Card Number", "role": "input", "suggested_selector": "#user_cardnumber", "bbox": {"x": 0.2, "y": 0.5, "width": 0.2, "height": 0.05}},
        {"id": "field_iban", "label": "IBAN Field", "role": "input", "suggested_selector": "#field_iban", "bbox": {"x": 0.2, "y": 0.6, "width": 0.2, "height": 0.05}},
        {"id": "field_cvc", "label": "CVC Field", "role": "input", "suggested_selector": "#field_cvc", "bbox": {"x": 0.2, "y": 0.7, "width": 0.2, "height": 0.05}},
    ]

    dag = {
        "workflow_name": "SecretsTestWorkflow",
        "target_url": "https://example.com",
        "nodes": {s["id"]: {"role": s["role"]} for s in segments},
        "edges": [
            {
                "from_state": "start",
                "to_state": s["id"],
                "action": "fill",
                "action_type": "fill",
                "trigger_element": s["id"],
                "target_segment_id": s["id"],
                "target_selector": s["suggested_selector"],
                "input_payload": super_secret_pw if idx == 0 else "some_secret",
            }
            for idx, s in enumerate(segments)
        ],
    }

    builder = DAGBuilder(workflow_name="SecretsTestWorkflow")
    schema = builder._synthesize_parameters_schema(dag["nodes"], dag["edges"], segments)
    dag["parameters_schema"] = schema

    generator = JinjaCodeGenerator()
    generator.emit(
        dag=dag,
        segments=segments,
        out_dir=out_dir,
        telemetry_events=telemetry_events,
    )

    # Grep EVERY file in dist directory
    all_secrets = [super_secret_pw, secret_jwt, secret_card, secret_iban, secret_cvc]
    emitted_files = list(out_dir.rglob("*"))
    assert len(emitted_files) > 0, "No files emitted"

    for file_path in emitted_files:
        if file_path.is_file():
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for secret in all_secrets:
                assert secret not in content, f"Secret leaked in {file_path.name}: {secret}"

    # Also verify parameter keys are not derived from the secret payload
    for key in schema.get("properties", {}):
        for secret in all_secrets:
            assert secret not in key
            assert "SuperSecret" not in key

    # Test importing and executing emitted driver
    sys.path.insert(0, str(out_dir))
    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    try:
        driver_module = importlib.import_module("playwright_driver")
        driver_cls = getattr(driver_module, "SecretsTestWorkflow")
        driver = driver_cls(headless=True)
        res = asyncio.run(driver.execute_workflow(params={"field_98124": super_secret_pw}))
        assert res["success"] is True
        # Secrets should not be echoed in driver results
        for secret in all_secrets:
            assert secret not in json.dumps(res)
    finally:
        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)
        sys.path.remove(str(out_dir))


def test_fastapi_schema_and_echo_redaction(tmp_path: Path):
    out_dir = tmp_path / "dist_api"

    segments = [
        {"id": "pwd_input", "label": "Password Input", "role": "input", "suggested_selector": "input#pwd", "bbox": {"x": 0.2, "y": 0.3, "width": 0.2, "height": 0.05}},
    ]
    dag = {
        "workflow_name": "ApiSecretsWorkflow",
        "target_url": "https://example.com",
        "nodes": {"pwd_input": {"role": "input"}},
        "edges": [
            {
                "from_state": "start",
                "to_state": "pwd_input",
                "action": "fill",
                "action_type": "fill",
                "trigger_element": "pwd_input",
                "target_segment_id": "pwd_input",
                "target_selector": "input#pwd",
                "input_payload": "my_secret_password",
                "default_value": "",
            }
        ],
    }

    builder = DAGBuilder(workflow_name="ApiSecretsWorkflow")
    schema = builder._synthesize_parameters_schema(dag["nodes"], dag["edges"], segments)
    dag["parameters_schema"] = schema

    generator = JinjaCodeGenerator()
    generator.emit(dag=dag, segments=segments, out_dir=out_dir)

    sys.path.insert(0, str(out_dir))
    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    try:
        app_module = importlib.import_module("app")
        app = getattr(app_module, "app")
        client = TestClient(app)

        # 1. /api/v1/schema should NOT expose defaults for sensitive parameters
        resp_schema = client.get("/api/v1/schema")
        assert resp_schema.status_code == 200
        schema_data = resp_schema.json()
        props = schema_data.get("parameters_schema", {}).get("properties", {})
        for p_name, p_info in props.items():
            if "pwd" in p_name or "password" in p_name:
                assert "default" not in p_info or not p_info["default"]

        # 2. POST /api/v1/execute must not echo secret parameter values
        resp_exec = client.post("/api/v1/execute", json={"pwd_input": "VerySecretPassword123!"})
        assert resp_exec.status_code == 200
        body = resp_exec.json()
        body_str = json.dumps(body)
        assert "VerySecretPassword123!" not in body_str
        assert body["results"]["parameters"].get("pwd_input") == "[REDACTED]"
    finally:
        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)
        sys.path.remove(str(out_dir))


@pytest.mark.asyncio
async def test_shadow_dom_password_redacted_real_chromium(tmp_path: Path):
    from playwright.async_api import async_playwright

    rec = OcularSessionRecorder(out_dir=tmp_path, headless=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        # Wire up binding and init script
        await context.expose_binding("__ocular_push_telemetry", rec._handle_telemetry_binding)
        await context.add_init_script(rec.get_init_script() if hasattr(rec, "get_init_script") else rec.TELEMETRY_INIT_SCRIPT)

        page = await context.new_page()

        # HTML with Shadow DOM containing password input
        html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <div id="host"></div>
            <script>
                const host = document.getElementById('host');
                const shadow = host.attachShadow({mode: 'open'});
                const input = document.createElement('input');
                input.type = 'password';
                input.id = 'shadow_secret';
                shadow.appendChild(input);
            </script>
        </body>
        </html>
        """
        import urllib.parse
        await page.goto(f"data:text/html,{urllib.parse.quote(html_content)}")
        await page.wait_for_timeout(200)

        # Focus shadow input and type
        shadow_input = page.locator("#host").locator("input")
        await shadow_input.click()
        await page.keyboard.type("SecretShadow123!")
        await page.wait_for_timeout(300)

        await browser.close()

    # Verify that telemetry events captured the typing but redacted the secret
    assert len(rec.telemetry_events) > 0, "No telemetry events captured"
    for ev in rec.telemetry_events:
        if ev.get("action_type") == "fill" or ev.get("event_type") == "keydown":
            assert ev.get("key") == "[REDACTED]"
            assert ev.get("code") == "[REDACTED]"
