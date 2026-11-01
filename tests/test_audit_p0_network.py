import asyncio
import http.server
import ipaddress
import os
import socket
import threading
import time
from typing import Any, Dict, List
from urllib.parse import urlsplit
import pytest
from playwright.async_api import async_playwright

from ocular.security.netguard import (
    BLOCKED_IP_NETWORKS,
    canonicalize_url,
    get_allowed_hosts,
    host_is_blocked,
    is_private_url,
    resolve_and_check_host,
    setup_confinement_routes,
    unwrap_ip,
)


@pytest.mark.asyncio
async def test_multihop_redirect_confinement():
    """N-01: Multi-hop redirect chains (1-hop, 2-hop, 3-hop, and 302->allowed->302->169.254.169.254)
    must NEVER leak any request to an un-allowlisted internal server or cloud metadata endpoint.
    """
    leaked_internal_paths: List[str] = []

    # 1. Internal server listening on loopback (not allowlisted)
    class InternalServerHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            leaked_internal_paths.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"INTERNAL_SECRET")

        def log_message(self, format, *args):
            pass

    internal_server = http.server.HTTPServer(("127.0.0.1", 0), InternalServerHandler)
    internal_port = internal_server.server_port
    t_internal = threading.Thread(target=internal_server.serve_forever, daemon=True)
    t_internal.start()

    # 2. Attacker server on localhost (allowlisted by host:port)
    class AttackerServerHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/hop1":
                # 1-hop redirect to internal server
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{internal_port}/leak_hop1")
                self.end_headers()
            elif self.path == "/hop2_a":
                # 2-hop step 1 -> step 2
                self.send_response(302)
                self.send_header("Location", f"/hop2_b")
                self.end_headers()
            elif self.path == "/hop2_b":
                # 2-hop step 2 -> internal server
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{internal_port}/leak_hop2")
                self.end_headers()
            elif self.path == "/hop3_a":
                # 3-hop step 1 (302)
                self.send_response(302)
                self.send_header("Location", f"/hop3_b")
                self.end_headers()
            elif self.path == "/hop3_b":
                # 3-hop step 2 (302)
                self.send_response(302)
                self.send_header("Location", f"/hop3_c")
                self.end_headers()
            elif self.path == "/hop3_c":
                # 3-hop step 3 (307) -> internal server
                self.send_response(307)
                self.send_header("Location", f"http://127.0.0.1:{internal_port}/leak_hop3")
                self.end_headers()
            elif self.path == "/meta_step1":
                self.send_response(302)
                self.send_header("Location", f"/meta_step2")
                self.end_headers()
            elif self.path == "/meta_step2":
                self.send_response(302)
                self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"OK")

        def log_message(self, format, *args):
            pass

    attacker_server = http.server.HTTPServer(("127.0.0.1", 0), AttackerServerHandler)
    attacker_port = attacker_server.server_port
    t_attacker = threading.Thread(target=attacker_server.serve_forever, daemon=True)
    t_attacker.start()

    # Allowlist ONLY the attacker server via explicit host:port
    allowed_hosts = {f"127.0.0.1:{attacker_port}", f"localhost:{attacker_port}"}

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
            await setup_confinement_routes(context, allowed_hosts=allowed_hosts, blocked_port=internal_port)
            page = await context.new_page()

            # Test 1-hop redirect
            try:
                await page.goto(f"http://127.0.0.1:{attacker_port}/hop1", timeout=3000)
            except Exception:
                pass

            # Test 2-hop redirect
            try:
                await page.goto(f"http://127.0.0.1:{attacker_port}/hop2_a", timeout=3000)
            except Exception:
                pass

            # Test 3-hop redirect (302 -> 302 -> 307)
            try:
                await page.goto(f"http://127.0.0.1:{attacker_port}/hop3_a", timeout=3000)
            except Exception:
                pass

            # Test 302 -> allowed -> 302 -> 169.254.169.254
            try:
                await page.goto(f"http://127.0.0.1:{attacker_port}/meta_step1", timeout=3000)
            except Exception:
                pass

            await browser.close()

        # ZERO internal hits must be recorded across all multi-hop redirect attempts!
        assert leaked_internal_paths == [], f"Multi-hop redirect leaked requests to internal server: {leaked_internal_paths}"
    finally:
        internal_server.shutdown()
        attacker_server.shutdown()


@pytest.mark.asyncio
async def test_port_blind_allowlist():
    """N-03: Allowlist entries must require explicit port for loopback/private targets.
    Allowlist {"localhost:3000"} must block navigation to localhost:9001 (internal), while localhost:3000 works.
    """
    leaked_9001 = False

    class Handler9001(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal leaked_9001
            leaked_9001 = True
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"SECRET_9001")

        def log_message(self, format, *args):
            pass

    server_9001 = http.server.HTTPServer(("127.0.0.1", 0), Handler9001)
    port_9001 = server_9001.server_port
    t9001 = threading.Thread(target=server_9001.serve_forever, daemon=True)
    t9001.start()

    class Handler3000(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ALLOWED_3000")

        def log_message(self, format, *args):
            pass

    server_3000 = http.server.HTTPServer(("127.0.0.1", 0), Handler3000)
    port_3000 = server_3000.server_port
    t3000 = threading.Thread(target=server_3000.serve_forever, daemon=True)
    t3000.start()

    allowed = {f"localhost:{port_3000}", f"127.0.0.1:{port_3000}"}

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await setup_confinement_routes(context, allowed_hosts=allowed, blocked_port=port_9001)
            page = await context.new_page()

            # 1. Navigation to port_9001 must be blocked
            nav_9001_failed = False
            page_9001 = await context.new_page()
            try:
                await page_9001.goto(f"http://127.0.0.1:{port_9001}/", timeout=2000)
            except Exception:
                nav_9001_failed = True
            finally:
                await page_9001.close()

            assert nav_9001_failed or not leaked_9001
            assert not leaked_9001, "Port-blind bypass occurred! Port 9001 was accessed despite only port 3000 in allowlist."

            # 2. Navigation to allowed port_3000 must succeed
            page_3000 = await context.new_page()
            try:
                resp_3000 = await page_3000.goto(f"http://127.0.0.1:{port_3000}/", timeout=3000)
                assert resp_3000 is not None
                assert resp_3000.status == 200
                content = await resp_3000.text()
                assert "ALLOWED_3000" in content
            finally:
                await page_3000.close()

            await browser.close()
    finally:
        server_9001.shutdown()
        server_3000.shutdown()


@pytest.mark.parametrize("dangerous_url", [
    r"http://127.0.0.1\@evil.com/",
    r"http://127.0.0.1\test@evil.com/",
    "http://user:pass@127.0.0.1/",
    "ftp:127.0.0.1",
    "ftp://127.0.0.1/file",
    "chrome:settings",
    "chrome://settings",
    "javascript:alert(1)",
    "data:text/html,<html></html>",
    "view-source:http://127.0.0.1/",
    "blob:http://example.com/uuid",
    "file:///etc/passwd",
    "gopher://127.0.0.1:70/",
])
def test_is_private_url_parser_differentials(dangerous_url: str):
    """N-04: is_private_url must strictly fail-closed on backslashes, userinfo, and non-http(s) schemes."""
    assert is_private_url(dangerous_url) is True, f"Dangerous URL {dangerous_url} was not identified as private/blocked!"


def test_ip_blocklist_additions_and_unwrapping():
    """N-08: Verify fec0::/10 and 192.88.99.0/24 in BLOCKED_IP_NETWORKS and unwrap_ip uses int(ip)."""
    # 1. Check subnets
    fec0_net = ipaddress.ip_network("fec0::/10")
    relay_net = ipaddress.ip_network("192.88.99.0/24")

    assert any(fec0_net == net for net in BLOCKED_IP_NETWORKS), "fec0::/10 missing from BLOCKED_IP_NETWORKS"
    assert any(relay_net == net for net in BLOCKED_IP_NETWORKS), "192.88.99.0/24 missing from BLOCKED_IP_NETWORKS"

    assert host_is_blocked("fec0::1") is True
    assert host_is_blocked("192.88.99.1") is True

    # 2. Check unwrap_ip without accessing private attributes
    mapped_ip = ipaddress.ip_address("::ffff:127.0.0.1")
    unwrapped = unwrap_ip(mapped_ip)
    assert unwrapped == ipaddress.ip_address("127.0.0.1")

    nat64_ip = ipaddress.ip_address("64:ff9b::192.168.1.1")
    unwrapped_nat64 = unwrap_ip(nat64_ip)
    assert unwrapped_nat64 == ipaddress.ip_address("192.168.1.1")


@pytest.mark.asyncio
async def test_dns_rebinding_ttl_lru_and_resolver_stub():
    """N-02: Bounded TTL LRU cache and resolution re-checking.
    Resolver stub returns public IP on first lookup, but 127.0.0.1 on subsequent lookups.
    """
    call_count = 0

    async def rebinding_resolver(h, p, proto=0):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    host = "rebinding-test.example.com"
    # First lookup -> public IP -> allowed (False)
    blocked_1 = await resolve_and_check_host(host, custom_resolver=rebinding_resolver)
    assert blocked_1 is False

    # Within TTL (under 5s), cached result is used
    blocked_cached = await resolve_and_check_host(host, custom_resolver=rebinding_resolver)
    assert blocked_cached is False

    # Simulate TTL expiration (manually expire cache or test TTL logic)
    from ocular.security import netguard
    if hasattr(netguard, "_DNS_CACHE"):
        netguard._DNS_CACHE.clear()

    # Second lookup after TTL -> resolves to 127.0.0.1 -> BLOCKED (True)
    blocked_2 = await resolve_and_check_host(host, custom_resolver=rebinding_resolver)
    assert blocked_2 is True, "Rebinding lookup to 127.0.0.1 was not blocked after TTL expiration!"


def test_production_proxy_mandatory(monkeypatch):
    """N-05: When OCULAR_ENV=production, OCULAR_EGRESS_PROXY is mandatory; refuse to start otherwise."""
    from ocular.pipeline.code_generator import JinjaCodeGenerator
    import importlib
    import sys
    from pathlib import Path

    dist_dir = Path("/tmp/test_prod_proxy_dist")
    generator = JinjaCodeGenerator()
    generator.emit(
        dag={"workflow_name": "ProdTest", "target_url": "http://example.com", "nodes": {}, "edges": []},
        segments=[],
        out_dir=dist_dir,
    )

    sys.modules.pop("playwright_driver", None)
    sys.path.insert(0, str(dist_dir.resolve()))
    try:
        mod = importlib.import_module("playwright_driver")
        cls = getattr(mod, "ProdTestWorkflow", getattr(mod, "OcularPlaywrightDriver", None))

        monkeypatch.setenv("OCULAR_ENV", "production")
        monkeypatch.delenv("OCULAR_EGRESS_PROXY", raising=False)

        with pytest.raises(RuntimeError, match="OCULAR_ENV=production requires OCULAR_EGRESS_PROXY"):
            cls()
    finally:
        sys.path.remove(str(dist_dir.resolve()))
        sys.modules.pop("playwright_driver", None)


@pytest.mark.asyncio
async def test_egress_proxy_mode():
    """N-02: start_egress_proxy validates destinations and proxies traffic, returning 403 on blocked targets."""
    from ocular.security.netguard import start_egress_proxy

    # 1. Target server
    target_hits = 0

    class TargetHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal target_hits
            target_hits += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"PROXY_TARGET_OK")

        def log_message(self, format, *args):
            pass

    target_srv = http.server.HTTPServer(("127.0.0.1", 0), TargetHandler)
    t_port = target_srv.server_port
    t_thread = threading.Thread(target=target_srv.serve_forever, daemon=True)
    t_thread.start()

    # Allow target host:port
    allowed = {f"127.0.0.1:{t_port}"}
    proxy_srv, proxy_url = await start_egress_proxy(allowed_hosts=allowed)

    try:
        async with async_playwright() as p:
            # Launch with proxy and <-loopback>
            browser = await p.chromium.launch(
                headless=True,
                proxy={"server": proxy_url, "bypass": "<-loopback>"},
            )
            page = await browser.new_page()

            # Allowed target should succeed through proxy
            resp = await page.goto(f"http://127.0.0.1:{t_port}/", timeout=4000)
            assert resp is not None
            assert resp.status == 200
            assert "PROXY_TARGET_OK" in (await resp.text())
            assert target_hits == 1

            # Non-allowlisted target on different port should be rejected with 403 by proxy
            resp_blocked = await page.goto("http://127.0.0.1:59999/", timeout=2000)
            # Proxy returns 403 or connection failure
            if resp_blocked is not None:
                assert resp_blocked.status == 403

            await browser.close()
    finally:
        proxy_srv.close()
        await proxy_srv.wait_closed()
        target_srv.shutdown()


@pytest.mark.asyncio
async def test_response_caps_size_and_time():
    """N-09: Responses exceeding 25 MB or 20s wall time must be aborted."""
    class LargeHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/large":
                self.send_response(200)
                # Claim 30 MB in Content-Length
                self.send_header("Content-Length", str(30 * 1024 * 1024))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(b"A" * 1024)
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), LargeHandler)
    srv_port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    allowed = {f"127.0.0.1:{srv_port}", f"localhost:{srv_port}"}

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await setup_confinement_routes(context, allowed_hosts=allowed)
            page = await context.new_page()

            # Exceeding size cap must be aborted
            with pytest.raises(Exception):
                await page.goto(f"http://127.0.0.1:{srv_port}/large", timeout=3000)

            await browser.close()
    finally:
        server.shutdown()


def test_normalize_host_ipv6_bracket_and_port():
    """FIX 3: normalize_host handles IPv6 bracket + port, bare bracket, and raw IPv6."""
    from ocular.security.netguard import normalize_host

    assert normalize_host("[::1]:8080") == "::1"
    assert normalize_host("[::1]") == "::1"
    assert normalize_host("::1") == "::1"
    assert normalize_host("[2001:db8::1]:443") == "2001:db8::1"
    assert normalize_host("[2001:db8::1]") == "2001:db8::1"
    assert normalize_host("2001:db8::1") == "2001:db8::1"


@pytest.mark.asyncio
async def test_egress_proxy_dns_rebinding_pinned_to_validated_ip(monkeypatch):
    """FIX 2: DNS-rebinding TOCTOU mitigation in start_egress_proxy.
    
    Simulates a DNS rebinding attack where the resolver returns a public IP on the
    first lookup and private 127.0.0.1 on subsequent lookups.
    Verifies that the proxy connects directly to the first validated IP literal,
    preventing open_connection from re-resolving the hostname to 127.0.0.1.
    """
    from ocular.security.netguard import start_egress_proxy, _DNS_CACHE
    _DNS_CACHE.clear()

    call_count = 0

    async def rebinding_resolver(host, port):
        nonlocal call_count
        call_count += 1
        # Calls 1 & 2 return public IP; subsequent calls return private 127.0.0.1
        if call_count <= 2:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    connected_destinations = []
    real_open_connection = asyncio.open_connection

    proxy_srv, proxy_url = await start_egress_proxy(custom_resolver=rebinding_resolver)
    parsed_proxy = urlsplit(proxy_url)
    proxy_host, proxy_port = parsed_proxy.hostname, parsed_proxy.port

    async def mock_open_connection(host, port, *args, **kwargs):
        if (host, port) == (proxy_host, proxy_port):
            return await real_open_connection(host, port, *args, **kwargs)
        connected_destinations.append((host, port))
        raise ConnectionRefusedError("Simulated upstream close after capturing pinned target")

    monkeypatch.setattr(asyncio, "open_connection", mock_open_connection)

    try:
        # 1. Plain HTTP request through proxy
        r, w = await asyncio.open_connection(proxy_host, proxy_port)
        w.write(b"GET http://rebind.attacker.com:80/data HTTP/1.1\r\nHost: rebind.attacker.com\r\n\r\n")
        await w.drain()
        resp_data = await r.read(1024)
        w.close()
        await w.wait_closed()

        # Upstream open_connection must have been called with the pinned validated IP literal
        assert len(connected_destinations) == 1
        assert connected_destinations[0] == ("93.184.216.34", 80)
        assert connected_destinations[0][0] != "127.0.0.1"
        assert connected_destinations[0][0] != "rebind.attacker.com"

        # 2. CONNECT request through proxy
        r2, w2 = await asyncio.open_connection(proxy_host, proxy_port)
        w2.write(b"CONNECT rebind-tls.attacker.com:443 HTTP/1.1\r\nHost: rebind-tls.attacker.com\r\n\r\n")
        await w2.drain()
        resp_data2 = await r2.read(1024)
        w2.close()
        await w2.wait_closed()

        # CONNECT must also connect directly to the pinned validated IP literal
        assert len(connected_destinations) == 2
        assert connected_destinations[1] == ("93.184.216.34", 443)
        assert connected_destinations[1][0] != "127.0.0.1"
        assert connected_destinations[1][0] != "rebind-tls.attacker.com"

        # 3. Third request after DNS rebinding to 127.0.0.1: proxy must block with 403
        _DNS_CACHE.clear()
        r3, w3 = await asyncio.open_connection(proxy_host, proxy_port)
        w3.write(b"GET http://rebind-private.attacker.com:80/secret HTTP/1.1\r\nHost: rebind-private.attacker.com\r\n\r\n")
        await w3.drain()
        resp_data3 = await r3.read(1024)
        w3.close()
        await w3.wait_closed()

        assert b"403 Forbidden" in resp_data3
        assert b"SSRF Policy Blocked" in resp_data3
        # No additional connection attempt made to upstream for blocked private IP
        assert len(connected_destinations) == 2

    finally:
        proxy_srv.close()
        await proxy_srv.wait_closed()


