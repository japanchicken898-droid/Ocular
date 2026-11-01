"""Network Confinement and SSRF Prevention Module for Ocular.

Provides robust, multi-layer egress and SSRF defenses:
- URL canonicalization defeating parser differentials (whitespace, control chars, backslashes, userinfo).
- Host normalization and legacy-IPv4 / IPv6 unwrapping (int, octal, hex, shorthand, NAT64, 6to4, Teredo).
- Comprehensive non-global IP blocking (RFC 1918, CGNAT, link-local, cloud metadata, multicast, 6to4 relay, site-local).
- Async DNS resolution with fail-closed semantics, bounded concurrency, and TTL LRU cache defeating DNS rebinding.
- Full browser context confinement (route interception, multi-hop redirect resolution, WebSocket termination, size/time caps).
- Local validating egress proxy mode.

==============================================================================================
PRODUCTION HARDENING REQUIREMENT: CONTAINER / NETWORK NAMESPACE FIREWALL RULES
==============================================================================================
In production deployments, software-level SSRF guards MUST be reinforced by kernel-level network
namespace isolation and firewall egress filtering (e.g. iptables / nftables).

Minimum required production egress policy:
  1. Drop loopback egress:
     iptables -A OUTPUT -d 127.0.0.0/8 -j REJECT
     ip6tables -A OUTPUT -d ::1/128 -j REJECT
  2. Drop RFC 1918 private subnets:
     iptables -A OUTPUT -d 10.0.0.0/8 -j REJECT
     iptables -A OUTPUT -d 172.16.0.0/12 -j REJECT
     iptables -A OUTPUT -d 192.168.0.0/16 -j REJECT
  3. Drop link-local, carrier-grade NAT, and cloud metadata:
     iptables -A OUTPUT -d 169.254.0.0/16 -j REJECT
     iptables -A OUTPUT -d 169.254.169.254 -j REJECT
     iptables -A OUTPUT -d 100.100.100.200 -j REJECT
     iptables -A OUTPUT -d 100.64.0.0/10 -j REJECT
==============================================================================================
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import ipaddress
import logging
import os
import re
import socket
import time
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

logger = logging.getLogger("ocular.security.netguard")

# Blocked subnets covering all non-global, private, loopback, link-local, and reserved ranges
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),       # CGNAT (RFC 6598)
    ipaddress.ip_network("127.0.0.0/8"),         # Loopback
    ipaddress.ip_network("169.254.0.0/16"),       # Link-local / Cloud metadata (169.254.169.254)
    ipaddress.ip_network("172.16.0.0/12"),       # RFC 1918
    ipaddress.ip_network("192.0.0.0/24"),        # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),        # Documentation (TEST-NET-1)
    ipaddress.ip_network("192.88.99.0/24"),      # 6to4 relay anycast (RFC 7526)
    ipaddress.ip_network("192.168.0.0/16"),      # RFC 1918
    ipaddress.ip_network("198.18.0.0/15"),       # Benchmarking (RFC 2544)
    ipaddress.ip_network("198.51.100.0/24"),     # Documentation (TEST-NET-2)
    ipaddress.ip_network("203.0.113.0/24"),      # Documentation (TEST-NET-3)
    ipaddress.ip_network("224.0.0.0/4"),         # Multicast
    ipaddress.ip_network("240.0.0.0/4"),         # Reserved / Future Use
    ipaddress.ip_network("255.255.255.255/32"),  # Broadcast
    ipaddress.ip_network("::/128"),              # Unspecified
    ipaddress.ip_network("::1/128"),             # Loopback
    ipaddress.ip_network("::ffff:0:0/96"),       # IPv4-mapped IPv6
    ipaddress.ip_network("64:ff9b::/96"),        # NAT64
    ipaddress.ip_network("100::/64"),            # Discard-only
    ipaddress.ip_network("2001:db8::/32"),       # Documentation
    ipaddress.ip_network("2002::/16"),           # 6to4
    ipaddress.ip_network("fc00::/7"),            # Unique Local Address (ULA)
    ipaddress.ip_network("fe80::/10"),           # Link-local unicast
    ipaddress.ip_network("fec0::/10"),           # Site-local unicast (RFC 3879 deprecated)
    ipaddress.ip_network("ff00::/8"),            # Multicast
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


class TTLLRUCache:
    """Bounded LRU cache with per-entry time-to-live (TTL)."""

    def __init__(self, maxsize: int = 1024, default_ttl: float = 5.0) -> None:
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Tuple[bool, float]] = OrderedDict()

    def get(self, key: str) -> Optional[bool]:
        if key not in self._cache:
            return None
        verdict, expires_at = self._cache[key]
        if time.monotonic() > expires_at:
            try:
                del self._cache[key]
            except KeyError:
                pass
            return None
        self._cache.move_to_end(key)
        return verdict

    def set(self, key: str, value: bool, ttl: Optional[float] = None) -> None:
        if ttl is None:
            ttl = self.default_ttl
        expires_at = time.monotonic() + ttl
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, expires_at)
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: str) -> bool:
        res = self.get(key)
        if res is None:
            raise KeyError(key)
        return res

    def __setitem__(self, key: str, value: bool) -> None:
        # Default TTL: 5s if allowed (False), 2s if failure/transient error (True)
        ttl = 2.0 if value is True else 5.0
        self.set(key, value, ttl=ttl)


# Per-process bounded TTL LRU DNS cache
_DNS_CACHE = TTLLRUCache(maxsize=1024, default_ttl=5.0)

# Dedicated bounded thread executor and concurrency semaphore for DNS resolutions
_DNS_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="netguard_dns")
_DNS_SEMAPHORE = asyncio.Semaphore(8)


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

        ip_int = int(ip)
        # IPv4-compatible (::a.b.c.d)
        if ip in ipaddress.ip_network("::/96") and ip_int > 1:
            return ipaddress.IPv4Address(ip_int & 0xFFFFFFFF)
        # NAT64 (64:ff9b::/96)
        if ip in ipaddress.ip_network("64:ff9b::/96"):
            return ipaddress.IPv4Address(ip_int & 0xFFFFFFFF)
        # 6to4 (2002::/16) -> bits 16..48 represent IPv4
        if ip in ipaddress.ip_network("2002::/16"):
            return ipaddress.IPv4Address((ip_int >> 80) & 0xFFFFFFFF)
        # Teredo (2001:0000::/32) -> last 32 bits negated represent IPv4
        if ip in ipaddress.ip_network("2001:0000::/32"):
            return ipaddress.IPv4Address((ip_int & 0xFFFFFFFF) ^ 0xFFFFFFFF)

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

    # Strip IPv6 enclosing brackets and optional port (e.g. [::1]:8080, [::1])
    if norm.startswith("["):
        close_bracket = norm.find("]")
        if close_bracket != -1:
            rest = norm[close_bracket + 1:]
            if rest and not rest.startswith(":"):
                return None
            norm = norm[1:close_bracket]

    # Strip IPv6 zone id (%eth0)
    if "%" in norm:
        norm = norm.split("%")[0]

    # Strip trailing dot (DNS root)
    norm = norm.rstrip(".")

    # Strip port if present for host evaluation
    if ":" in norm and not norm.count(":") > 1:
        norm = norm.split(":")[0]

    return norm if norm else None


def is_loopback_or_private_host(host: str) -> bool:
    """Checks whether a normalized host string represents loopback, internal naming, or a private IP."""
    h = (normalize_host(host) or host).strip().lower()
    if h in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    if any(h == sfx.lstrip(".") or h.endswith(sfx) for sfx in BLOCKED_NAME_SUFFIXES):
        return True

    parsed_ip = parse_legacy_ipv4(h)
    if parsed_ip is None:
        try:
            parsed_ip = ipaddress.ip_address(h)
        except ValueError:
            pass

    if parsed_ip is not None:
        unwrapped = unwrap_ip(parsed_ip)
        if unwrapped in EXPLICIT_BLOCKED_IPS or not unwrapped.is_global:
            return True
        for net in BLOCKED_IP_NETWORKS:
            if unwrapped in net:
                return True
        return False

    return False


def get_allowed_hosts() -> Set[str]:
    """Retrieves allowed hosts set from OCULAR_ALLOWED_HOSTS or OCULAR_ALLOWED_EGRESS_HOSTS.
    
    Allowlist entries become scheme-optional host:port (e.g. 'localhost:3000').
    A bare host entry is accepted only if it is a non-loopback hostname.
    Loopback/private targets require an explicit port.
    """
    raw = os.environ.get("OCULAR_ALLOWED_HOSTS") or os.environ.get("OCULAR_ALLOWED_EGRESS_HOSTS", "")
    raw = raw.strip()
    if not raw:
        return set()

    allowed: Set[str] = set()
    for item in re.split(r"[\s,;]+", raw):
        item = item.strip().lower()
        if not item:
            continue
        # Strip scheme if present
        if "://" in item:
            item = item.split("://", 1)[1]
        # Strip trailing path/slash
        item = item.split("/", 1)[0]
        if not item:
            continue

        if ":" in item and not (item.startswith("[") and item.endswith("]") and item.count(":") > 1):
            # Explicit host:port entry
            allowed.add(item)
            h_part = item.split(":", 1)[0]
            if not is_loopback_or_private_host(h_part):
                allowed.add(h_part)
        else:
            # Bare host entry: accepted only if it is a non-loopback hostname
            if not is_loopback_or_private_host(item):
                allowed.add(item)
            else:
                logger.warning(
                    f"Ignored bare loopback/private entry '{item}' in allowlist. "
                    "Loopback/private targets strictly require an explicit host:port (e.g. 'localhost:3000')."
                )

    return allowed


def is_host_allowed(clean_host: str, port: Optional[int], allowed_hosts: Set[str]) -> bool:
    """Checks whether host and optional port are explicitly permitted in the allowlist."""
    if not clean_host or not allowed_hosts:
        return False
    # If explicit host:port matches
    if port and f"{clean_host}:{port}" in allowed_hosts:
        return True
    # If bare host matches AND host is non-loopback
    if clean_host in allowed_hosts and not is_loopback_or_private_host(clean_host):
        return True
    return False


def host_is_blocked(host: Any, port: Optional[int] = None, allowed_hosts: Optional[Set[str]] = None) -> bool:
    """Evaluates whether a host or IP resolves to private subnets or prohibited internal endpoints.
    
    Fails closed (returns True) on invalid, unresolvable, or malformed input.
    """
    if not host or not isinstance(host, str):
        return True

    # Userinfo or backslash injection in hostname
    if "@" in host or "\\" in host:
        return True

    clean_host = normalize_host(host)
    if not clean_host:
        return True

    # Check explicit allowlist
    active_allowlist = allowed_hosts if allowed_hosts is not None else get_allowed_hosts()
    if is_host_allowed(clean_host, port, active_allowlist):
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

    # Browser-style backslash normalization (convert \ to /)
    normalized = stripped.replace("\\", "/")

    parsed = urlsplit(normalized)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Prohibited URL scheme: '{scheme}'. Only http and https are allowed.")

    if not parsed.hostname:
        raise ValueError(f"URL missing valid hostname: {url}")

    path = parsed.path
    if not path and stripped.endswith("/"):
        path = "/"
    return urlunsplit((scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def is_private_url(url_or_host: Any, allowed_hosts: Optional[Set[str]] = None) -> bool:
    """Evaluates whether a URL or host points to a private subnet, dangerous URI scheme, or prohibited endpoint.
    
    Fails closed (returns True) on any invalid URL, non-http(s) scheme, userinfo, backslashes, or private IP/host.
    """
    if not url_or_host or not isinstance(url_or_host, str):
        return True
    u = url_or_host.strip()
    if not u:
        return True

    # Immediate rejection of non-http(s) schemes even without '://' (e.g. ftp:..., chrome:..., javascript:...)
    lower = u.lower()
    if lower.startswith(("file:", "data:", "blob:", "about:", "gopher:", "javascript:", "chrome:", "ftp:", "view-source:", "ws:", "wss:")):
        return True

    if "@" in u:
        return True

    try:
        url_to_eval = u if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", u) else f"http://{u}"
        canon = canonicalize_url(url_to_eval)
        parsed = urlsplit(canon)
        if (parsed.scheme or "").lower() not in ("http", "https"):
            return True
        hostname = parsed.hostname
        if not hostname:
            return True
        port = parsed.port
        return host_is_blocked(hostname, port=port, allowed_hosts=allowed_hosts)
    except Exception:
        return True


async def resolve_safe_ip(
    host: str,
    port: Optional[int] = None,
    allowed_hosts: Optional[Set[str]] = None,
    custom_resolver: Optional[Callable] = None,
) -> Optional[str]:
    """Asynchronously resolves a host and validates all returned IP addresses against blocked subnets.
    
    If permitted, returns the first validated safe IP literal string (e.g. '93.184.216.34').
    If blocked or on resolution error, returns None (fails closed).
    
    This function eliminates DNS rebinding TOCTOU vulnerabilities by resolving and validating
    the IP address once, allowing callers to connect directly to the returned IP literal.
    """
    clean_host = normalize_host(host)
    if not clean_host:
        return None

    active_allowlist = allowed_hosts if allowed_hosts is not None else get_allowed_hosts()

    # If host:port is explicitly allowlisted
    if is_host_allowed(clean_host, port, active_allowlist):
        parsed_ip = parse_legacy_ipv4(clean_host)
        if parsed_ip is None:
            try:
                parsed_ip = ipaddress.ip_address(clean_host)
            except ValueError:
                pass
        if parsed_ip is not None:
            return str(parsed_ip)

    # Check if clean_host is already an IP address
    parsed_ip = parse_legacy_ipv4(clean_host)
    if parsed_ip is None:
        try:
            parsed_ip = ipaddress.ip_address(clean_host)
        except ValueError:
            pass

    if parsed_ip is not None:
        if host_is_blocked(str(parsed_ip), port=port, allowed_hosts=active_allowlist):
            return None
        return str(parsed_ip)

    cache_key = f"{clean_host}:{port}" if port else clean_host
    cached_val = _DNS_CACHE.get(cache_key)
    if cached_val is not None:
        if cached_val is True or cached_val == "":
            return None
        if isinstance(cached_val, str) and cached_val:
            return cached_val

    # Static host check
    if host_is_blocked(clean_host, port=port, allowed_hosts=active_allowlist):
        _DNS_CACHE.set(cache_key, "", ttl=5.0)
        return None

    # Async DNS resolution bounded by semaphore and thread pool
    try:
        if custom_resolver:
            addr_info = await custom_resolver(clean_host, port or 80)
        else:
            async with _DNS_SEMAPHORE:
                loop = asyncio.get_running_loop()
                addr_info = await asyncio.wait_for(
                    loop.run_in_executor(
                        _DNS_EXECUTOR,
                        socket.getaddrinfo,
                        clean_host,
                        port or 80,
                        0,
                        socket.SOCK_STREAM,
                        socket.IPPROTO_TCP,
                    ),
                    timeout=2.0,
                )

        if not addr_info:
            _DNS_CACHE.set(cache_key, "", ttl=2.0)
            return None

        # Check every resolved IP address
        safe_ip: Optional[str] = None
        for entry in addr_info:
            ip_str = entry[4][0]
            if host_is_blocked(ip_str, port=port, allowed_hosts=active_allowlist):
                _DNS_CACHE.set(cache_key, "", ttl=5.0)
                return None
            if safe_ip is None:
                safe_ip = ip_str

        if safe_ip:
            # Narrow allowed cache TTL to 2.0s to minimize DNS rebinding exposure window
            _DNS_CACHE.set(cache_key, safe_ip, ttl=2.0)
            return safe_ip

        _DNS_CACHE.set(cache_key, "", ttl=2.0)
        return None
    except Exception as exc:
        logger.debug(f"DNS resolution failure for '{clean_host}': {exc}; failing closed with 2s TTL.")
        _DNS_CACHE.set(cache_key, "", ttl=2.0)
        return None


async def resolve_and_check_host(
    host: str,
    port: Optional[int] = None,
    allowed_hosts: Optional[Set[str]] = None,
    custom_resolver: Optional[Callable] = None,
) -> bool:
    """Asynchronously resolves a host and checks all returned IP addresses against blocked subnets.
    
    Fail-closed on resolution errors. Returns True if blocked, False if permitted.
    """
    safe_ip = await resolve_safe_ip(
        host=host,
        port=port,
        allowed_hosts=allowed_hosts,
        custom_resolver=custom_resolver,
    )
    return safe_ip is None


MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_WALL_TIME_SEC = 20.0


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

            start_time = time.monotonic()
            current_url = req_url
            visited_urls = {current_url}
            current_method = route.request.method
            current_headers = dict(route.request.headers)
            current_post_data = route.request.post_data_buffer

            for hop in range(MAX_REDIRECTS + 1):
                # 1. Wall-time cap check
                if time.monotonic() - start_time > MAX_WALL_TIME_SEC:
                    logger.warning(f"Aborting navigation exceeding wall time cap ({MAX_WALL_TIME_SEC}s): {current_url}")
                    await route.abort("timedout")
                    return

                # 2. Validate URL scheme
                parsed = urlparse(current_url)
                scheme = (parsed.scheme or "").lower()
                if scheme not in ("http", "https"):
                    logger.warning(f"Aborting non-http/https navigation: {current_url}")
                    await route.abort("blockedbyclient")
                    return

                # 3. Validate port
                hostname = parsed.hostname or ""
                port = parsed.port or (443 if scheme == "https" else 80)
                if blocked_port and port == blocked_port:
                    logger.warning(f"Aborting navigation to explicitly blocked port {port}: {current_url}")
                    await route.abort("blockedbyclient")
                    return

                # 4. Validate host and resolved IP addresses
                # NOTE on Playwright route.fetch():
                # In Playwright Python (>= 1.48.0), route.fetch() delegates to Chromium's internal network
                # stack and does not expose an API to pin destination socket IP literals while preserving
                # Host/SNI. To mitigate TOCTOU rebinding in route.fetch(), resolve_safe_ip() validates
                # the host with a narrowed DNS cache TTL (2.0s). In production environments, kernel-level
                # egress filtering (iptables / network namespaces) and the Ocular egress proxy serve as
                # the authoritative IP confinement boundaries.
                safe_ip = await resolve_safe_ip(hostname, port=port, allowed_hosts=active_allowlist)
                if not safe_ip:
                    logger.warning(f"Aborting SSRF navigation to prohibited host/IP: {hostname}:{port}")
                    await route.abort("blockedbyclient")
                    return

                # 5. Fetch with max_redirects=0
                fetch_kwargs: Dict[str, Any] = {"max_redirects": 0}
                if hop == 0:
                    resp = await route.fetch(**fetch_kwargs)
                else:
                    resp = await route.fetch(
                        url=current_url,
                        method=current_method,
                        headers=current_headers,
                        data=current_post_data,
                        **fetch_kwargs,
                    )

                # 6. Check for 3xx redirect status
                if resp.status in (301, 302, 303, 307, 308):
                    if hop >= MAX_REDIRECTS:
                        logger.warning(f"Aborting navigation exceeding max redirect limit ({MAX_REDIRECTS}): {current_url}")
                        await route.abort("blockedbyclient")
                        return

                    loc = resp.headers.get("location")
                    if not loc:
                        logger.warning(f"Aborting redirect with missing Location header from {current_url}")
                        await route.abort("blockedbyclient")
                        return

                    next_url = urljoin(current_url, loc.strip())
                    if next_url in visited_urls:
                        logger.warning(f"Aborting redirect loop detected ({next_url})")
                        await route.abort("blockedbyclient")
                        return

                    visited_urls.add(next_url)

                    # Update method per RFC 7231
                    if resp.status in (301, 302, 303) and current_method == "POST":
                        current_method = "GET"
                        current_post_data = None
                    elif resp.status == 303 and current_method != "HEAD":
                        current_method = "GET"
                        current_post_data = None

                    current_url = next_url
                    continue

                # 7. Non-3xx final response: check response size and fulfill
                cl_header = resp.headers.get("content-length")
                if cl_header:
                    try:
                        if int(cl_header) > MAX_RESPONSE_BYTES:
                            logger.warning(f"Aborting response exceeding size cap ({cl_header} bytes): {current_url}")
                            await route.abort("blockedbyclient")
                            return
                    except ValueError:
                        pass

                body = await resp.body()
                if len(body) > MAX_RESPONSE_BYTES:
                    logger.warning(f"Aborting response body exceeding size cap ({len(body)} bytes): {current_url}")
                    await route.abort("blockedbyclient")
                    return

                # Fulfill ONLY the final non-3xx response
                await route.fulfill(status=resp.status, headers=resp.headers, body=body)
                return

            await route.abort("blockedbyclient")
        except Exception as exc:
            err_msg = str(exc).lower()
            if "econnrefused" in err_msg or "connection refused" in err_msg:
                abort_code = "connectionrefused"
            elif "econnreset" in err_msg or "connection reset" in err_msg:
                abort_code = "connectionreset"
            elif "timed out" in err_msg or "timeout" in err_msg or "etimedout" in err_msg:
                abort_code = "timedout"
            elif "enotfound" in err_msg or "name not resolved" in err_msg:
                abort_code = "namenotresolved"
            elif "blockedbyclient" in err_msg or "prohibited" in err_msg:
                abort_code = "blockedbyclient"
            else:
                abort_code = "connectionfailed"
            logger.debug(f"Error in _http_route_guard: {exc}; aborting route with {abort_code}.")
            await route.abort(abort_code)

    async def _ws_route_guard(ws_route: Any) -> None:
        try:
            ws_url = ws_route.url
            parsed = urlparse(ws_url)
            hostname = parsed.hostname or ""
            port = parsed.port

            if blocked_port and port == blocked_port:
                logger.warning(f"Closing WebSocket to explicitly blocked port {port}: {ws_url}")
                res = ws_route.close(code=1008, reason="Blocked port policy")
                if asyncio.iscoroutine(res):
                    await res
                return

            safe_ip = await resolve_safe_ip(hostname, port=port, allowed_hosts=active_allowlist)
            if not safe_ip:
                logger.warning(f"Closing WebSocket to prohibited host: {hostname}")
                res = ws_route.close(code=1008, reason="SSRF policy blocked")
                if asyncio.iscoroutine(res):
                    await res
                return

            ws_route.connect_to_server()
        except Exception as exc:
            logger.debug(f"Error in _ws_route_guard: {exc}; closing WebSocket.")
            try:
                res = ws_route.close(code=1008, reason="Internal error")
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

    # Register HTTP route guard using regex catching all schemes
    await context.route(re.compile(r".*"), _http_route_guard)
    # Register WebSocket route guard
    await context.route_web_socket(re.compile(r".*"), _ws_route_guard)

    # Attach guards to new pages, frames, and popups
    def _on_new_page(page: Any) -> None:
        asyncio.create_task(page.route(re.compile(r".*"), _http_route_guard))

    context.on("page", _on_new_page)


async def start_egress_proxy(
    host: str = "127.0.0.1",
    port: int = 0,
    allowed_hosts: Optional[Set[str]] = None,
    custom_resolver: Optional[Callable] = None,
) -> Tuple[asyncio.Server, str]:
    """Starts a lightweight asyncio forward / CONNECT HTTP proxy validating all outbound IPs.
    
    Tunnels HTTPS via CONNECT and forwards HTTP GET/POST, strictly blocking private IPs and unauthorized egress.
    Returns (server, proxy_url).
    """
    active_allowlist = allowed_hosts if allowed_hosts is not None else get_allowed_hosts()

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not request_line:
                writer.close()
                await writer.wait_closed()
                return

            parts = request_line.decode("latin1", errors="replace").strip().split()
            if len(parts) < 2:
                writer.close()
                await writer.wait_closed()
                return

            method, target = parts[0].upper(), parts[1]

            if method == "CONNECT":
                # CONNECT host:port HTTP/1.1
                if ":" in target:
                    target_host, target_port_s = target.split(":", 1)
                    target_port = int(target_port_s)
                else:
                    target_host = target
                    target_port = 443

                # Consume remaining headers
                while True:
                    line = await reader.readline()
                    if not line or line in (b"\r\n", b"\n"):
                        break

                resolved_ip = await resolve_safe_ip(
                    target_host,
                    port=target_port,
                    allowed_hosts=active_allowlist,
                    custom_resolver=custom_resolver,
                )
                if not resolved_ip:
                    writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\n\r\nSSRF Policy Blocked\r\n")
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()
                    return

                try:
                    # Pin connection to validated IP literal to prevent DNS-rebinding TOCTOU
                    upstream_reader, upstream_writer = await asyncio.wait_for(
                        asyncio.open_connection(resolved_ip, target_port), timeout=5.0
                    )
                except Exception:
                    writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()
                    return

                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()

                async def pipe(r, w):
                    try:
                        while not r.at_eof():
                            data = await r.read(65536)
                            if not data:
                                break
                            w.write(data)
                            await w.drain()
                    except Exception:
                        pass
                    finally:
                        try:
                            w.close()
                        except Exception:
                            pass

                await asyncio.gather(pipe(reader, upstream_writer), pipe(upstream_reader, writer))

            else:
                # Standard HTTP GET/POST/etc.
                parsed = urlsplit(target)
                target_host = parsed.hostname or ""
                target_port = parsed.port or 80

                headers = [request_line]
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    headers.append(line)
                    if line in (b"\r\n", b"\n"):
                        break

                resolved_ip = await resolve_safe_ip(
                    target_host,
                    port=target_port,
                    allowed_hosts=active_allowlist,
                    custom_resolver=custom_resolver,
                )
                if not resolved_ip:
                    writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\n\r\nSSRF Policy Blocked\r\n")
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()
                    return

                try:
                    # Pin connection to validated IP literal to prevent DNS-rebinding TOCTOU
                    upstream_reader, upstream_writer = await asyncio.wait_for(
                        asyncio.open_connection(resolved_ip, target_port), timeout=5.0
                    )
                except Exception:
                    writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()
                    return

                upstream_writer.write(b"".join(headers))
                await upstream_writer.drain()

                async def pipe_http(r, w):
                    try:
                        while not r.at_eof():
                            data = await r.read(65536)
                            if not data:
                                break
                            w.write(data)
                            await w.drain()
                    except Exception:
                        pass
                    finally:
                        try:
                            w.close()
                        except Exception:
                            pass

                await asyncio.gather(pipe_http(reader, upstream_writer), pipe_http(upstream_reader, writer))
        except Exception:
            try:
                writer.close()
            except Exception:
                pass

    server = await asyncio.start_server(handle_client, host, port)
    assigned_port = server.sockets[0].getsockname()[1]
    proxy_url = f"http://{host}:{assigned_port}"
    return server, proxy_url
