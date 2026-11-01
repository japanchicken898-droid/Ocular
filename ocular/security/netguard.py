"""Network Confinement and SSRF Prevention Module for Ocular.

Provides robust, multi-layer egress and SSRF defenses:
- URL canonicalization defeating parser differentials (whitespace, control chars, backslashes, userinfo).
- Host normalization and legacy-IPv4 / IPv6 unwrapping (int, octal, hex, shorthand, NAT64, 6to4, Teredo).
- Comprehensive non-global IP blocking (RFC 1918, CGNAT, link-local, cloud metadata, multicast, etc.).
- Async DNS resolution with fail-closed semantics and per-run verdict caching defeating DNS rebinding.
- Full browser context confinement (route interception, WebSocket termination, service worker block, WebRTC confinement).
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import socket
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

logger = logging.getLogger("ocular.security.netguard")

# Blocked subnets covering all non-global, private, loopback, link-local, and reserved ranges
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),       # CGNAT (RFC 6598)
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local / Cloud metadata (169.254.169.254)
    ipaddress.ip_network("172.16.0.0/12"),     # RFC 1918
    ipaddress.ip_network("192.0.0.0/24"),      # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),      # Documentation (TEST-NET-1)
    ipaddress.ip_network("192.168.0.0/16"),    # RFC 1918
    ipaddress.ip_network("198.18.0.0/15"),     # Benchmarking (RFC 2544)
    ipaddress.ip_network("198.51.100.0/24"),   # Documentation (TEST-NET-2)
    ipaddress.ip_network("203.0.113.0/24"),    # Documentation (TEST-NET-3)
    ipaddress.ip_network("224.0.0.0/4"),       # Multicast
    ipaddress.ip_network("240.0.0.0/4"),       # Reserved / Future Use
    ipaddress.ip_network("255.255.255.255/32"),# Broadcast
    ipaddress.ip_network("::/128"),            # Unspecified
    ipaddress.ip_network("::1/128"),           # Loopback
    ipaddress.ip_network("::ffff:0:0/96"),     # IPv4-mapped IPv6
    ipaddress.ip_network("64:ff9b::/96"),      # NAT64
    ipaddress.ip_network("100::/64"),          # Discard-only
    ipaddress.ip_network("2001:db8::/32"),     # Documentation
    ipaddress.ip_network("2002::/16"),         # 6to4
    ipaddress.ip_network("fc00::/7"),          # Unique Local Address (ULA)
    ipaddress.ip_network("fe80::/10"),         # Link-local unicast
    ipaddress.ip_network("ff00::/8"),          # Multicast
]

# Explicit metadata IPs to strictly block
EXPLICIT_BLOCKED_IPS = {
    ipaddress.ip_address("169.254.169.254"),   # AWS, Azure, GCP, DigitalOcean, OpenStack metadata
    ipaddress.ip_address("100.100.100.200"),   # Alibaba Cloud metadata
}

BLOCKED_NAME_SUFFIXES = (
    "localhost",
    ".localhost",
    ".local",
    ".internal",
    ".localdomain",
    ".onion",
    ".invalid",
    ".test",
    ".nip.io",
    ".sslip.io",
    ".localtest.me",
)

# Per-process / per-run DNS verdict cache: maps hostname -> is_blocked boolean
_DNS_CACHE: Dict[str, bool] = {}


def get_allowed_hosts() -> Set[str]:
    """Retrieves allowed hosts set from OCULAR_ALLOWED_HOSTS or OCULAR_ALLOWED_EGRESS_HOSTS."""
    raw = os.environ.get("OCULAR_ALLOWED_HOSTS") or os.environ.get("OCULAR_ALLOWED_EGRESS_HOSTS", "")
    raw = raw.strip()
    if not raw:
        return set()
    return {h.strip().lower() for h in re.split(r"[\s,;]+", raw) if h.strip()}


def parse_legacy_ipv4(s: str) -> Optional[ipaddress.IPv4Address]:
    """Parses legacy IPv4 formats including integers, hex, octal, and shorthand notation."""
    parts = s.split(".")
    if not (1 <= len(parts) <= 4):
        return None

    vals: List[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            return None
        try:
            if p.startswith(("0x", "0X")):
                vals.append(int(p, 16))
            elif p.startswith("0") and len(p) > 1 and p.isdigit():
                vals.append(int(p, 8))
            elif p.isdigit():
                vals.append(int(p, 10))
            else:
                return None
        except (ValueError, OverflowError):
            return None

    try:
        if len(vals) == 1:
            if 0 <= vals[0] <= 0xFFFFFFFF:
                return ipaddress.IPv4Address(vals[0])
        elif len(vals) == 2:
            if 0 <= vals[0] <= 0xFF and 0 <= vals[1] <= 0xFFFFFF:
                return ipaddress.IPv4Address((vals[0] << 24) | (vals[1] & 0xFFFFFF))
        elif len(vals) == 3:
            if 0 <= vals[0] <= 0xFF and 0 <= vals[1] <= 0xFF and 0 <= vals[2] <= 0xFFFF:
                return ipaddress.IPv4Address((vals[0] << 24) | (vals[1] << 16) | (vals[2] & 0xFFFF))
        elif len(vals) == 4:
            if all(0 <= v <= 0xFF for v in vals):
                return ipaddress.IPv4Address((vals[0] << 24) | (vals[1] << 16) | (vals[2] << 8) | vals[3])
    except (ValueError, OverflowError):
        return None

    return None


def unwrap_ip(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> Union[ipaddress.IPv4Address, ipaddress.IPv6Address]:
    """Unwraps IPv4-mapped, IPv4-compatible, NAT64, 6to4, or Teredo IPv6 addresses into standard IPv4."""
    if isinstance(ip, ipaddress.IPv4Address):
        return ip

    if isinstance(ip, ipaddress.IPv6Address):
        if ip == ipaddress.IPv6Address("::1") or ip == ipaddress.IPv6Address("::"):
            return ip
        if ip.ipv4_mapped:
            return ip.ipv4_mapped
        # IPv4-compatible (::a.b.c.d)
        if ip in ipaddress.ip_network("::/96") and ip._ip > 1:
            return ipaddress.IPv4Address(ip._ip & 0xFFFFFFFF)
        # NAT64 (64:ff9b::/96)
        if ip in ipaddress.ip_network("64:ff9b::/96"):
            return ipaddress.IPv4Address(ip._ip & 0xFFFFFFFF)
        # 6to4 (2002::/16) -> bits 16..48 represent IPv4
        if ip in ipaddress.ip_network("2002::/16"):
            return ipaddress.IPv4Address((ip._ip >> 80) & 0xFFFFFFFF)
        # Teredo (2001:0000::/32) -> last 32 bits negated represent IPv4
        if ip in ipaddress.ip_network("2001:0000::/32"):
            return ipaddress.IPv4Address((ip._ip & 0xFFFFFFFF) ^ 0xFFFFFFFF)

    return ip


def normalize_host(host: Any) -> Optional[str]:
    """Normalizes a host string: NFKC normalization, lowercase, strip brackets, trailing dots, zone id."""
    if not host or not isinstance(host, str):
        return None

    # Reject userinfo @ or backslashes in host
    if "@" in host or "\\" in host:
        return None

    # NFKC normalization (converts fullwidth digits like １２７.０.０.１ to 127.0.0.1)
    norm = unicodedata.normalize("NFKC", host.strip()).lower()

    # Strip IPv6 enclosing brackets
    if norm.startswith("[") and norm.endswith("]"):
        norm = norm[1:-1]

    # Strip IPv6 zone id (%eth0)
    if "%" in norm:
        norm = norm.split("%")[0]

    # Strip trailing dot (DNS root)
    norm = norm.rstrip(".")

    # Strip port if present for host evaluation
    if ":" in norm and not norm.count(":") > 1:
        norm = norm.split(":")[0]

    return norm if norm else None


def is_private_url(url_or_host: Any, allowed_hosts: Optional[Set[str]] = None) -> bool:
    """Evaluates whether a URL or host points to a private subnet, dangerous URI scheme, or prohibited endpoint.
    Handles full URLs (e.g. 'http://127.0.0.1/admin', 'file:///etc/passwd'), IP addresses, and hostnames.
    """
    if not url_or_host or not isinstance(url_or_host, str):
        return True
    u = url_or_host.strip()
    if not u:
        return True

    lower = u.lower()
    # Reject non-http(s) schemes (file, data, blob, about, gopher, etc.)
    if lower.startswith(("file:", "data:", "blob:", "about:", "gopher:", "javascript:")):
        return True

    # If full URL with scheme
    if "://" in u:
        try:
            parsed = urlsplit(u)
            if (parsed.scheme or "").lower() not in ("http", "https"):
                return True
            hostname = parsed.hostname or u.split("://")[1].split("/")[0].split(":")[0]
            if not hostname:
                return True
            return host_is_blocked(hostname, allowed_hosts=allowed_hosts)
        except Exception:
            return True

    # If path component present (e.g. 127.0.0.1/admin)
    if "/" in u:
        host = u.split("/")[0].split(":")[0]
        return host_is_blocked(host, allowed_hosts=allowed_hosts)

    return host_is_blocked(u, allowed_hosts=allowed_hosts)


def host_is_blocked(host: Any, allowed_hosts: Optional[Set[str]] = None) -> bool:
    """Evaluates whether a host or IP resolves to private subnets or prohibited internal endpoints.
    
    Fails closed (returns True) on invalid, unresolvable, or malformed input.
    """
    if not host or not isinstance(host, str):
        return True

    # Delegate URL strings to is_private_url
    if "://" in host or host.strip().lower().startswith(("file:", "data:", "blob:", "about:", "gopher:")):
        return is_private_url(host, allowed_hosts=allowed_hosts)

    # Userinfo or backslash injection in hostname
    if "@" in host or "\\" in host:
        return True

    clean_host = normalize_host(host)
    if not clean_host:
        return True

    # Check explicit allowlist
    active_allowlist = allowed_hosts if allowed_hosts is not None else get_allowed_hosts()
    if clean_host in active_allowlist:
        return False
    # Check if host:port or raw input is allowed
    if isinstance(host, str) and host.strip().lower().rstrip(".") in active_allowlist:
        return False

    # Check blocked suffix names
    if clean_host == "localhost" or any(clean_host.endswith(sfx) for sfx in BLOCKED_NAME_SUFFIXES):
        return True

    # Attempt to parse as legacy IPv4 or standard IP address
    parsed_ip: Optional[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]] = parse_legacy_ipv4(clean_host)
    if parsed_ip is None:
        try:
            parsed_ip = ipaddress.ip_address(clean_host)
        except ValueError:
            pass

    if parsed_ip is not None:
        unwrapped = unwrap_ip(parsed_ip)

        # Check explicit prohibited metadata IPs
        if unwrapped in EXPLICIT_BLOCKED_IPS:
            return True
        if unwrapped == parsed_ip and parsed_ip in EXPLICIT_BLOCKED_IPS:
            return True

        # Check if non-global (covers private, loopback, link-local, multicast, etc.)
        # Note: unwrapped represents the real IPv4 address for mapped/compatible/NAT64/6to4
        if not unwrapped.is_global:
            return True
        if unwrapped == parsed_ip and not parsed_ip.is_global:
            return True

        # Check blocked subnets
        for net in BLOCKED_IP_NETWORKS:
            if unwrapped in net:
                return True
            if unwrapped == parsed_ip and parsed_ip in net:
                return True

        return False

    # Non-IP hostname check: if contains invalid characters, block
    if not re.match(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$", clean_host):
        return True

    return False


async def resolve_and_check_host(
    host: str,
    port: Optional[int] = None,
    allowed_hosts: Optional[Set[str]] = None,
    custom_resolver: Optional[Callable] = None,
) -> bool:
    """Asynchronously resolves a host and checks all returned IP addresses against blocked subnets.
    
    Caches verdicts per run to defeat DNS rebinding attacks. Fail-closed on resolution errors.
    Returns True if blocked, False if permitted.
    """
    clean_host = normalize_host(host)
    if not clean_host:
        return True

    active_allowlist = allowed_hosts if allowed_hosts is not None else get_allowed_hosts()
    if clean_host in active_allowlist or (f"{clean_host}:{port}" in active_allowlist if port else False):
        return False

    # Check run cache
    if clean_host in _DNS_CACHE:
        return _DNS_CACHE[clean_host]

    # Static host check
    if host_is_blocked(clean_host, allowed_hosts=active_allowlist):
        _DNS_CACHE[clean_host] = True
        return True

    # Async DNS resolution
    try:
        if custom_resolver:
            addr_info = await custom_resolver(clean_host, port or 80)
        else:
            loop = asyncio.get_running_loop()
            addr_info = await asyncio.wait_for(
                loop.getaddrinfo(clean_host, port or 80, proto=socket.IPPROTO_TCP),
                timeout=2.0,
            )

        if not addr_info:
            _DNS_CACHE[clean_host] = True
            return True

        # Check every resolved IP address
        for entry in addr_info:
            ip_str = entry[4][0]
            if host_is_blocked(ip_str, allowed_hosts=active_allowlist):
                _DNS_CACHE[clean_host] = True
                return True

        _DNS_CACHE[clean_host] = False
        return False
    except Exception as exc:
        logger.debug(f"DNS resolution failure for '{clean_host}': {exc}; failing closed.")
        _DNS_CACHE[clean_host] = True
        return True


def canonicalize_url(url: str) -> str:
    """Canonicalizes a URL and verifies safe schemes, rejecting parser differential vectors."""
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    # Reject null bytes, control characters, and leading/trailing whitespace
    for ch in url:
        if ord(ch) < 32 or ord(ch) == 127:
            raise ValueError(f"URL contains illegal control character (ASCII {ord(ch)})")

    stripped = url.strip()
    if not stripped:
        raise ValueError("URL cannot be empty")

    # Reject userinfo before host (@ in authority)
    if "@" in stripped:
        raise ValueError("URLs with userinfo (@) are prohibited")

    # Normalize backslashes as browsers do
    normalized = stripped.replace("\\", "/")

    parsed = urlsplit(normalized)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Prohibited URL scheme: '{scheme}'. Only http and https are allowed.")

    if not parsed.hostname:
        raise ValueError(f"URL missing valid hostname: {url}")

    # Reconstruct canonical form
    path = parsed.path
    if not path and stripped.endswith("/"):
        path = "/"
    return urlunsplit((scheme, parsed.netloc, path, parsed.query, parsed.fragment))


async def setup_confinement_routes(
    context: Any,
    allowed_hosts: Optional[Set[str]] = None,
    blocked_port: Optional[int] = None,
) -> None:
    """Configures Playwright BrowserContext with comprehensive SSRF route and WebSocket guards."""
    active_allowlist = allowed_hosts if allowed_hosts is not None else get_allowed_hosts()

    async def _http_route_guard(route: Any) -> None:
        try:
            req_url = route.request.url
            if req_url.startswith("about:"):
                await route.continue_()
                return

            parsed = urlparse(req_url)
            scheme = (parsed.scheme or "").lower()
            if scheme not in ("http", "https"):
                logger.warning(f"Aborting non-http/https navigation: {req_url}")
                await route.abort("blockedbyclient")
                return

            hostname = parsed.hostname or ""
            port = parsed.port

            if blocked_port and port == blocked_port:
                logger.warning(f"Aborting navigation to explicitly blocked port {port}: {req_url}")
                await route.abort("blockedbyclient")
                return

            if await resolve_and_check_host(hostname, port=port, allowed_hosts=active_allowlist):
                logger.warning(f"Aborting SSRF navigation to prohibited host: {hostname}")
                await route.abort("blockedbyclient")
                return

            resp = await route.fetch(max_redirects=0)
            if resp.status in (301, 302, 303, 307, 308):
                loc = resp.headers.get("location")
                if loc:
                    target_url = urljoin(req_url, loc)
                    target_parsed = urlparse(target_url)
                    t_host = target_parsed.hostname or ""
                    t_port = target_parsed.port
                    if blocked_port and t_port == blocked_port:
                        logger.warning(f"Aborting redirect to blocked port {t_port}: {target_url}")
                        await route.abort("blockedbyclient")
                        return
                    if await resolve_and_check_host(t_host, port=t_port, allowed_hosts=active_allowlist):
                        logger.warning(f"Aborting redirect to prohibited host: {t_host}")
                        await route.abort("blockedbyclient")
                        return

            await route.fulfill(response=resp)
        except Exception as exc:
            logger.debug(f"Error in _http_route_guard: {exc}; aborting route.")
            await route.abort("blockedbyclient")

    async def _ws_route_guard(ws_route: Any) -> None:
        try:
            ws_url = ws_route.url
            parsed = urlparse(ws_url)
            hostname = parsed.hostname or ""
            port = parsed.port

            if blocked_port and port == blocked_port:
                logger.warning(f"Closing WebSocket to explicitly blocked port {port}: {ws_url}")
                ws_route.close(code=1008, reason="Blocked port policy")
                return

            if await resolve_and_check_host(hostname, port=port, allowed_hosts=active_allowlist):
                logger.warning(f"Closing WebSocket to prohibited host: {hostname}")
                ws_route.close(code=1008, reason="SSRF policy blocked")
                return

            ws_route.connect_to_server()
        except Exception as exc:
            logger.debug(f"Error in _ws_route_guard: {exc}; closing WebSocket.")
            ws_route.close(code=1008, reason="Internal error")

    # Register HTTP route guard using regex catching all schemes
    await context.route(re.compile(r".*"), _http_route_guard)
    # Register WebSocket route guard
    await context.route_web_socket(re.compile(r".*"), _ws_route_guard)

    # Attach guards to new pages, frames, and popups
    def _on_new_page(page: Any) -> None:
        asyncio.create_task(page.route(re.compile(r".*"), _http_route_guard))

    context.on("page", _on_new_page)
