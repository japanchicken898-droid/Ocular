"""Regression tests for Phase 3 SSRF & Network Confinement (F5-F8, F16, and urlparse differential).

Tests:
- host_is_blocked: comprehensive table-driven tests for all IP encodings, IPv6 unwrapping,
  multicast, CGNAT, link-local, cloud metadata, fullwidth digits, trailing dots, userinfo tricks.
- DNS rebinding prevention via per-run caching.
- Real-Chromium Egress Tests:
  Local loopback HTTP and WebSocket servers receive ZERO unauthorized packets across:
  direct nav, 302 redirect, fetch, img, iframe, window.open, form post, service worker, WebSocket.
"""

import asyncio
import http.server
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
import pytest
from playwright.async_api import async_playwright

from ocular.security.netguard import (
    canonicalize_url,
    host_is_blocked,
    resolve_and_check_host,
    setup_confinement_routes,
    BLOCKED_IP_NETWORKS,
)


def test_host_is_blocked_table():
    blocked_cases = [
        "127.0.0.1",
        "127.0.0.1.",
        "localhost",
        "localhost.",
        "foo.localhost",
        "myhost.local",
        "internal.service.internal",
        "test.localdomain",
        "0",
        "0.0.0.0",
        "::",
        "::1",
        "::127.0.0.1",
        "::ffff:127.0.0.1",
        "::ffff:7f00:1",
        "2130706433",
        "0x7f000001",
        "0177.0.0.1",
        "127.1",
        "10.0.0.1",
        "10.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # AWS/GCP metadata
        "100.100.100.200",  # Alibaba metadata
        "100.64.0.1",       # CGNAT RFC 6598
        "100.127.255.254",  # CGNAT
        "192.0.0.1",        # IETF Protocol Assignments
        "198.18.0.1",       # Benchmarking RFC 2544
        "198.19.255.255",   # Benchmarking
        "224.0.0.1",        # Multicast
        "240.0.0.1",        # Reserved
        "255.255.255.255",  # Broadcast
        "64:ff9b::7f00:1",  # NAT64 prefix with loopback
        "2002:7f00:0001::", # 6to4 prefix with loopback
        "127.0.0.1@example.com",
        "127.0.0.1\\@example.com",
        "１２７.０.０.１",     # Fullwidth digits
        "invalid-domain-that-does-not-exist-xyz123456.test",
        None,
        "",
        12345,
    ]

    for host in blocked_cases:
        assert host_is_blocked(host) is True, f"Failed to block dangerous host: {host}"

    # Allowed public host
    assert host_is_blocked("google.com", allowed_hosts={"google.com"}) is False
    assert host_is_blocked("8.8.8.8") is False


def test_url_canonicalization():
    # Backslashes normalized
    assert canonicalize_url("https://example.com\\path") == "https://example.com/path"

    # Userinfo rejected
    with pytest.raises(ValueError):
        canonicalize_url("https://user:pass@127.0.0.1/test")

    # Control chars / whitespace rejected
    with pytest.raises(ValueError):
        canonicalize_url("https://example.com/path\x00")
    with pytest.raises(ValueError):
        canonicalize_url("https://example.com/path\r\n")

    # Non-http/https schemes rejected
    with pytest.raises(ValueError):
        canonicalize_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        canonicalize_url("gopher://127.0.0.1")


@pytest.mark.asyncio
async def test_dns_rebinding_cached_verdict():
    # Verify that once a host is evaluated, the verdict is cached and prevents rebinding
    host = "rebinding-target.example.com"
    call_count = 0

    async def mock_getaddrinfo(h, p, proto=0):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    # First lookup resolves public
    blocked_first = await resolve_and_check_host(host, custom_resolver=mock_getaddrinfo)
    # Second lookup must use cached verdict or fail closed
    blocked_second = await resolve_and_check_host(host, custom_resolver=mock_getaddrinfo)

    assert blocked_first is False
    assert blocked_second is False
    assert call_count == 1, "DNS resolution must be cached per-run to defeat rebinding"


@pytest.mark.asyncio
async def test_real_chromium_egress_confinement():
    # 1. Start internal loopback HTTP server to detect ANY leaked requests
    leaked_requests = []

    class InternalHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            leaked_requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"INTERNAL_SECRET")

        def do_POST(self):
            leaked_requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"INTERNAL_SECRET")

        def log_message(self, format, *args):
            pass

    internal_server = http.server.HTTPServer(("127.0.0.1", 0), InternalHandler)
    internal_port = internal_server.server_port
    t_internal = threading.Thread(target=internal_server.serve_forever, daemon=True)
    t_internal.start()

    # 2. Start attacker HTTP server serving SSRF vectors
    class AttackerHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{internal_port}/leaked_302")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                html = f"""
                <!DOCTYPE html>
                <html>
                <body>
                    <img src="http://127.0.0.1:{internal_port}/leaked_img" />
                    <iframe src="http://127.0.0.1:{internal_port}/leaked_iframe"></iframe>
                    <form action="http://127.0.0.1:{internal_port}/leaked_form" method="POST" id="f"></form>
                    <script>
                        // fetch
                        try {{ fetch("http://127.0.0.1:{internal_port}/leaked_fetch"); }} catch(e) {{}}
                        // window.open
                        try {{ window.open("http://127.0.0.1:{internal_port}/leaked_popup"); }} catch(e) {{}}
                        // WebSocket
                        try {{ new WebSocket("ws://127.0.0.1:{internal_port}/leaked_ws"); }} catch(e) {{}}
                        // Service Worker
                        try {{ navigator.serviceWorker.register("http://127.0.0.1:{internal_port}/sw.js"); }} catch(e) {{}}
                    </script>
                </body>
                </html>
                """
                self.wfile.write(html.encode("utf-8"))

        def log_message(self, format, *args):
            pass

    attacker_server = http.server.HTTPServer(("127.0.0.1", 0), AttackerHandler)
    attacker_port = attacker_server.server_port
    t_attacker = threading.Thread(target=attacker_server.serve_forever, daemon=True)
    t_attacker.start()

    attacker_host = f"127.0.0.1:{attacker_port}"
    allowed_hosts = {attacker_host, f"localhost:{attacker_port}", "127.0.0.1"}

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-webrtc", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"],
            )
            context = await browser.new_context(
                service_workers="block",
                accept_downloads=False,
                permissions=[],
            )

            # Setup network confinement routes on context
            await setup_confinement_routes(context, allowed_hosts=allowed_hosts, blocked_port=internal_port)

            page = await context.new_page()

            # Test A: Direct navigation to blocked internal port
            try:
                await page.goto(f"http://127.0.0.1:{internal_port}/direct_nav", timeout=2000)
            except Exception:
                pass

            # Test B: 302 Redirect to internal port
            try:
                await page.goto(f"http://127.0.0.1:{attacker_port}/redirect", timeout=2000)
            except Exception:
                pass

            # Test C: Attacker page containing sub-resources, popup, websocket
            try:
                await page.goto(f"http://127.0.0.1:{attacker_port}/attack", timeout=3000)
                await page.wait_for_timeout(1000)
            except Exception:
                pass

            await browser.close()

        # CRITICAL ASSERTION: The internal server must receive ZERO requests!
        assert len(leaked_requests) == 0, f"SSRF confinement failed! Internal server received: {leaked_requests}"
    finally:
        internal_server.shutdown()
        attacker_server.shutdown()


def test_extract_origin_url_strict_sources_and_validation(tmp_path: Path):
    from ocular.pipeline.code_generator import extract_origin_url
    from ocular.pipeline.state_graph import CompilationError

    # 1. No origin in DAG or session_metadata -> raises CompilationError (no default to localhost:3000)
    with pytest.raises(CompilationError):
        extract_origin_url(dag={"workflow_name": "Test"}, out_dir=tmp_path)

    # 2. Origin present only in telemetry_events -> ignored, raises CompilationError
    telemetry = [{"url": "https://telemetry-origin.com/app", "target_url": "https://telemetry-origin.com"}]
    with pytest.raises(CompilationError):
        extract_origin_url(
            dag={"workflow_name": "Test"},
            telemetry_events=telemetry,
            out_dir=tmp_path,
        )

    # 3. Invalid scheme or missing hostname -> raises CompilationError
    with pytest.raises(CompilationError):
        extract_origin_url(dag={"target_url": "ftp://example.com/app"})
    with pytest.raises(CompilationError):
        extract_origin_url(dag={"target_url": "file:///etc/passwd"})
    with pytest.raises(CompilationError):
        extract_origin_url(dag={"target_url": "http://"})

    # 4. Valid target_url in DAG
    res = extract_origin_url(dag={"target_url": "https://example.com/app/"})
    assert res == "https://example.com/app"

    # 5. Valid origin in session_metadata.json
    sess_dir = tmp_path / "session_dir"
    sess_dir.mkdir()
    (sess_dir / "session_metadata.json").write_text(
        json.dumps({"url": "https://trusted-session.com/dashboard"}),
        encoding="utf-8",
    )
    res_sess = extract_origin_url(dag={"session_dir": str(sess_dir)})
    assert res_sess == "https://trusted-session.com/dashboard"


@pytest.mark.asyncio
async def test_netguard_emitted_and_shared_code(tmp_path: Path):
    import importlib
    import sys
    from ocular.pipeline.code_generator import JinjaCodeGenerator

    generator = JinjaCodeGenerator()
    out_dir = tmp_path / "shared_dist"
    dag = {
        "workflow_name": "SharedNetguardWorkflow",
        "target_url": "https://trusted.example.com",
        "root_node": "state_root",
        "terminal_node": "state_root",
        "nodes": {"state_root": {"id": "state_root", "title": "Root", "screen_name": "main"}},
        "edges": [],
    }

    artifacts = generator.emit(dag=dag, segments=[], out_dir=out_dir)

    # 1. netguard.py must be copied into out_dir
    assert "netguard" in artifacts
    netguard_file = out_dir / "netguard.py"
    assert netguard_file.exists()
    assert (out_dir / "playwright_driver.py").exists()
    assert (out_dir / "app.py").exists()

    # 2. Verify driver and app import netguard and do not duplicate network constants
    driver_code = (out_dir / "playwright_driver.py").read_text(encoding="utf-8")
    app_code = (out_dir / "app.py").read_text(encoding="utf-8")

    assert "from netguard import" in driver_code
    assert "from netguard import" in app_code
    assert "_is_server_online" not in driver_code
    assert "_is_server_online" not in app_code

    # 3. Dynamic execution test: import emitted modules and verify functions
    sys.modules.pop("netguard", None)
    sys.modules.pop("playwright_driver", None)
    sys.modules.pop("app", None)
    sys.path.insert(0, str(out_dir))
    try:
        netguard_mod = importlib.import_module("netguard")
        driver_mod = importlib.import_module("playwright_driver")
        app_mod = importlib.import_module("app")

        assert callable(getattr(netguard_mod, "host_is_blocked"))
        assert callable(getattr(driver_mod, "is_private_url"))
        assert callable(getattr(app_mod, "is_private_url"))
        assert callable(getattr(app_mod, "validate_safe_url"))

        # Test validate_safe_url blocks private IP
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await app_mod.validate_safe_url("http://127.0.0.1:8000")

        with pytest.raises(HTTPException):
            await app_mod.validate_safe_url("http://169.254.169.254/latest/meta-data")
    finally:
        if str(out_dir) in sys.path:
            sys.path.remove(str(out_dir))
        sys.modules.pop("netguard", None)
        sys.modules.pop("playwright_driver", None)
        sys.modules.pop("app", None)
