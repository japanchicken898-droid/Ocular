"""Playwright Interactive Session Recorder & Telemetry Probe.

Captures live browser sessions using Playwright's async API, streaming interaction
events (mousedown, keydown) with normalized viewport coordinates and relative timestamps
into telemetry_events.jsonl alongside captured browser video recordings.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import re
import secrets
import time
from typing import Any, Callable, Dict, List, Optional, Union

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from ocular.security.redaction import get_client_js_sensitive_check, is_sensitive_field, redact_event
from ocular.telemetry import coerce_telemetry_event

logger = logging.getLogger("ocular.recorder")


class OcularSessionRecorder:
    """Records an interactive browser session with video and telemetry events."""

    _INIT_SCRIPT_TEMPLATE = r"""
    (() => {
        if (window !== window.top) return;
        if (window.self !== window.top) return;
        const startTime = performance.now();
        const sessionEpoch = Date.now();
        const SESSION_NONCE = "__SESSION_NONCE__";

        // Private closure binding encapsulation: prevent untrusted page scripts from tampering
        const _pushTelemetry = window.__ocular_push_telemetry;
        try {
            Object.defineProperty(window, '__ocular_push_telemetry', {
                value: undefined,
                writable: false,
                configurable: false
            });
        } catch (e) {
            try { delete window.__ocular_push_telemetry; } catch (e2) {}
        }

        function getElapsedMs() {
            return Math.round(performance.now() - startTime);
        }

        function getElementSelector(el) {
            if (!el || el === document.body || el === document.documentElement) return null;
            if (el.id) return '#' + el.id;
            let selector = (el.tagName || '').toLowerCase();
            if (el.className && typeof el.className === 'string') {
                const classes = el.className.trim().split(/\s+/).slice(0, 2).filter(Boolean);
                if (classes.length > 0) {
                    selector += '.' + classes.join('.');
                }
            }
            if (el.getAttribute && el.getAttribute('data-role')) {
                selector += `[data-role="${el.getAttribute('data-role')}"]`;
            }
            return selector;
        }

        __CLIENT_JS_SENSITIVE_CHECK__
        window.checkSensitiveString = checkSensitiveString;

        function isSensitiveElement(el) {
            if (!el) return false;
            let curr = el;
            while (curr && curr !== document && curr !== window) {
                const tag = (curr.tagName || '').toLowerCase();
                const inputType = (curr.type || '').toLowerCase();
                if (tag === 'input' && inputType === 'password') {
                    return true;
                }
                const isContentEditable = Boolean(curr.isContentEditable || (curr.getAttribute && (curr.getAttribute('contenteditable') === 'true' || curr.getAttribute('contenteditable') === '')));
                const isRoleTextbox = Boolean(curr.getAttribute && curr.getAttribute('role') === 'textbox');
                const isTextarea = tag === 'textarea';

                const ac = (curr.getAttribute ? curr.getAttribute('autocomplete') || '' : '').toLowerCase();
                if (ac && (ac.indexOf('password') !== -1 || ac.indexOf('one-time-code') !== -1 || ac.indexOf('cc-') !== -1)) {
                    return true;
                }

                if (isTextarea || isContentEditable || isRoleTextbox) {
                    if (checkSensitiveString(curr.id) || checkSensitiveString(curr.className) || (curr.getAttribute && checkSensitiveString(curr.getAttribute('placeholder')))) {
                        return true;
                    }
                }
                const attrs = ['name', 'id', 'autocomplete', 'placeholder', 'aria-label', 'data-role', 'type', 'aria-description'];
                for (let i = 0; i < attrs.length; i++) {
                    const val = curr.getAttribute ? curr.getAttribute(attrs[i]) : null;
                    if (val && checkSensitiveString(val)) {
                        return true;
                    }
                }
                curr = curr.parentElement || (curr.getRootNode && curr.getRootNode().host);
            }
            return false;
        }

        window.addEventListener('mousedown', (e) => {
            if (e.isTrusted === false) return;
            const target = (e.composedPath && e.composedPath().length > 0) ? e.composedPath()[0] : e.target;
            const vw = window.innerWidth || 1;
            const vh = window.innerHeight || 1;
            const norm_x = Math.max(0.0, Math.min(1.0, e.clientX / vw));
            const norm_y = Math.max(0.0, Math.min(1.0, e.clientY / vh));
            const payload = {
                event_type: 'mousedown',
                action_type: 'click',
                norm_x: Number(norm_x.toFixed(4)),
                norm_y: Number(norm_y.toFixed(4)),
                client_x: e.clientX,
                client_y: e.clientY,
                button: e.button,
                target_tag: target && target.tagName ? target.tagName.toLowerCase() : null,
                target_selector: getElementSelector(target),
                elapsed_ms: getElapsedMs(),
                timestamp_ms: sessionEpoch + getElapsedMs(),
                _nonce: SESSION_NONCE,
            };
            try {
                if (_pushTelemetry) {
                    _pushTelemetry(payload);
                }
            } catch (err) {
                console.error('[Ocular Telemetry] Push failed:', err);
            }
        }, true);

        window.addEventListener('keydown', (e) => {
            if (e.isTrusted === false) return;
            const target = (e.composedPath && e.composedPath().length > 0) ? e.composedPath()[0] : e.target;
            const vw = window.innerWidth || 1;
            const vh = window.innerHeight || 1;
            let norm_x = 0.5;
            let norm_y = 0.5;
            if (target && target.getBoundingClientRect) {
                const rect = target.getBoundingClientRect();
                norm_x = (rect.left + rect.width / 2) / vw;
                norm_y = (rect.top + rect.height / 2) / vh;
            }
            const isSensitive = isSensitiveElement(target);
            const emittedKey = isSensitive ? '[REDACTED]' : e.key;
            const emittedCode = isSensitive ? '[REDACTED]' : e.code;
            const payload = {
                event_type: 'keydown',
                action_type: 'fill',
                key: emittedKey,
                code: emittedCode,
                norm_x: Number(Math.max(0.0, Math.min(1.0, norm_x)).toFixed(4)),
                norm_y: Number(Math.max(0.0, Math.min(1.0, norm_y)).toFixed(4)),
                target_tag: target && target.tagName ? target.tagName.toLowerCase() : null,
                target_selector: getElementSelector(target),
                elapsed_ms: getElapsedMs(),
                timestamp_ms: sessionEpoch + getElapsedMs(),
                _nonce: SESSION_NONCE,
            };
            try {
                if (_pushTelemetry) {
                    _pushTelemetry(payload);
                }
            } catch (err) {
                console.error('[Ocular Telemetry] Push failed:', err);
            }
        }, true);
    })();
    """
    _INIT_SCRIPT_TEMPLATE = _INIT_SCRIPT_TEMPLATE.replace(
        "__CLIENT_JS_SENSITIVE_CHECK__", get_client_js_sensitive_check()
    )

    TELEMETRY_INIT_SCRIPT = _INIT_SCRIPT_TEMPLATE.replace("__SESSION_NONCE__", secrets.token_urlsafe(32))

    @property
    def init_script(self) -> str:
        return self.get_init_script()

    def __getattribute__(self, name: str) -> Any:
        if name == "TELEMETRY_INIT_SCRIPT":
            try:
                nonce = object.__getattribute__(self, "session_nonce")
                template = object.__getattribute__(self, "_INIT_SCRIPT_TEMPLATE")
                return template.replace("__SESSION_NONCE__", nonce)
            except AttributeError:
                pass
        return super().__getattribute__(name)

    MAX_TELEMETRY_EVENTS = 2500
    MAX_STRING_ATTR_LEN = 256
    MAX_EVENT_BYTES = 4096

    ALLOWED_EVENT_TYPES = {"mousedown", "keydown", "click", "fill", "input", "keypress", "navigation"}
    ALLOWED_ACTION_TYPES = {"click", "fill", "type", "navigate", "wait"}
    ALLOWED_EVENT_KEYS = {
        "event_type",
        "action_type",
        "key",
        "code",
        "value",
        "input_payload",
        "default_value",
        "norm_x",
        "norm_y",
        "client_x",
        "client_y",
        "button",
        "target_tag",
        "target_selector",
        "elapsed_ms",
        "timestamp_ms",
        "_nonce",
    }
    STRING_FIELDS = {"key", "code", "value", "input_payload", "target_tag", "target_selector"}

    def __init__(
        self,
        out_dir: Union[str, Path] = "./recordings",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        headless: bool = False,
        slow_mo: int = 0,
        on_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.headless = headless
        self.slow_mo = slow_mo
        self.on_event_callback = on_event_callback
        self.session_nonce = secrets.token_urlsafe(32)
        self.telemetry_events: List[Dict[str, Any]] = []
        self.telemetry_path = self.out_dir / "telemetry_events.jsonl"
        self.video_path: Optional[Path] = None
        self.truncated: bool = False
        self._max_warning_logged: bool = False

    def get_init_script(self, nonce: Optional[str] = None) -> str:
        """Returns initialization script injected with the per-session random nonce."""
        active_nonce = nonce or self.session_nonce
        return self._INIT_SCRIPT_TEMPLATE.replace("__SESSION_NONCE__", active_nonce)

    def _truncate_attributes(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Truncates arbitrary string attributes and redacts sensitive inputs via unified module."""
        # Whitelist allowed keys
        filtered = {k: v for k, v in event_data.items() if k in self.ALLOWED_EVENT_KEYS}

        # Check and bound coordinates and button
        if "client_x" in filtered:
            cx = filtered["client_x"]
            if not isinstance(cx, (int, float)) or cx < 0:
                return {}
        if "client_y" in filtered:
            cy = filtered["client_y"]
            if not isinstance(cy, (int, float)) or cy < 0:
                return {}
        if "button" in filtered:
            btn = filtered["button"]
            if not isinstance(btn, int) or btn < 0 or btn > 10:
                return {}

        # Check for nested containers in string fields
        is_sensitive = is_sensitive_field(filtered)
        for k in self.STRING_FIELDS:
            if k in filtered:
                val = filtered[k]
                if isinstance(val, (list, dict)) and not is_sensitive:
                    return {}

        redacted = redact_event(filtered)
        cleaned = {}
        for k, v in redacted.items():
            if isinstance(v, str) and len(v) > self.MAX_STRING_ATTR_LEN:
                cleaned[k] = v[:self.MAX_STRING_ATTR_LEN]
            else:
                cleaned[k] = v
        cleaned.pop("_nonce", None)
        return cleaned

    @staticmethod
    def coerce_telemetry_event(event: Any) -> Optional[Dict[str, Any]]:
        """Safely coerces numeric fields in a telemetry event dictionary using shared telemetry module."""
        return coerce_telemetry_event(event)

    def push_telemetry_event(self, event_data: Dict[str, Any]) -> None:
        """Appends an event to in-memory history and streams to telemetry_events.jsonl."""
        if not isinstance(event_data, dict):
            logger.warning(f"Ignoring non-dictionary telemetry event: {type(event_data)}")
            return

        # Cap total event size to 4KB
        try:
            raw_bytes = json.dumps(event_data).encode("utf-8")
            if len(raw_bytes) > self.MAX_EVENT_BYTES:
                logger.warning(
                    f"Telemetry event size {len(raw_bytes)} bytes exceeds limit {self.MAX_EVENT_BYTES}. Dropping."
                )
                return
        except Exception:
            return

        # Whitelist event_type and action_type
        ev_type = str(event_data.get("event_type") or "").strip().lower()
        if ev_type and ev_type not in self.ALLOWED_EVENT_TYPES:
            logger.warning(f"Ignoring unwhitelisted event_type: {ev_type}")
            return

        action_type = str(event_data.get("action_type") or "").strip().lower()
        if action_type and action_type not in self.ALLOWED_ACTION_TYPES:
            logger.warning(f"Ignoring unwhitelisted action_type: {action_type}")
            return

        if len(self.telemetry_events) >= self.MAX_TELEMETRY_EVENTS:
            if not self._max_warning_logged:
                logger.warning(
                    f"Telemetry event limit reached ({self.MAX_TELEMETRY_EVENTS}). Truncating events."
                )
                self._max_warning_logged = True
            self.truncated = True
            return

        sanitized_event = self._truncate_attributes(event_data)
        if not sanitized_event:
            return

        coerced_event = self.coerce_telemetry_event(sanitized_event)
        if coerced_event is None:
            return

        self.telemetry_events.append(coerced_event)
        import os
        os.makedirs(self.out_dir, mode=0o700, exist_ok=True)
        try:
            os.chmod(self.out_dir, 0o700)
        except OSError:
            pass

        try:
            fd = os.open(self.telemetry_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.chmod(self.telemetry_path, 0o600)
            except OSError:
                pass
            with open(fd, "a", encoding="utf-8", closefd=True) as f:
                f.write(json.dumps(coerced_event) + "\n")
        except Exception as write_err:
            logger.debug(f"Error writing telemetry event: {write_err}")

        if self.on_event_callback:
            try:
                self.on_event_callback(coerced_event)
            except Exception as e:
                logger.debug(f"Error in on_event_callback: {e}")

    async def _handle_telemetry_binding(self, source: Any, event_data: Dict[str, Any]) -> None:
        """Playwright binding handler called by client-side probe."""
        if not isinstance(event_data, dict):
            return

        # Top-frame check: reject events originating from subframes
        if isinstance(source, dict):
            frame = source.get("frame")
            page = source.get("page")
            if frame is not None and page is not None:
                try:
                    if frame != page.main_frame:
                        logger.warning("Rejected telemetry event originating from subframe.")
                        return
                except Exception:
                    pass

        # Strict per-session nonce validation (no default_nonce fallback)
        nonce = event_data.get("_nonce")
        if not self.session_nonce or nonce != self.session_nonce:
            logger.warning("Rejected telemetry event with missing or invalid session nonce.")
            return

        # Validate event coordinates and mouse button
        if "client_x" in event_data:
            cx = event_data["client_x"]
            if not isinstance(cx, (int, float)) or cx < 0:
                logger.warning(f"Rejected telemetry event with invalid client_x: {cx}")
                return
        if "client_y" in event_data:
            cy = event_data["client_y"]
            if not isinstance(cy, (int, float)) or cy < 0:
                logger.warning(f"Rejected telemetry event with invalid client_y: {cy}")
                return
        if "button" in event_data:
            btn = event_data["button"]
            if not isinstance(btn, int) or btn < 0 or btn > 10:
                logger.warning(f"Rejected telemetry event with invalid button: {btn}")
                return

        self.push_telemetry_event(event_data)

    @staticmethod
    def _sanitize_origin(url: str) -> str:
        if not url:
            return ""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url.strip())
            if parsed.scheme in ("http", "https") and parsed.hostname:
                host = parsed.hostname.lower()
                port = f":{parsed.port}" if parsed.port and parsed.port not in (80, 443) else ""
                return f"{parsed.scheme.lower()}://{host}{port}"
        except Exception:
            pass
        return url

    def _write_session_metadata(self, url: str) -> None:
        start_epoch_ms = int(time.time() * 1000)
        sanitized = self._sanitize_origin(url)
        import os
        os.makedirs(self.out_dir, mode=0o700, exist_ok=True)
        try:
            os.chmod(self.out_dir, 0o700)
        except OSError:
            pass
        meta_path = self.out_dir / "session_metadata.json"
        try:
            fd = os.open(meta_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.chmod(meta_path, 0o600)
            except OSError:
                pass
            with open(fd, "w", encoding="utf-8", closefd=True) as f:
                f.write(json.dumps({
                    "url": sanitized,
                    "origin": sanitized,
                    "timestamp": time.time(),
                    "start_timestamp_ms": start_epoch_ms,
                }))
        except Exception:
            pass

    async def record_session(
        self,
        url: str,
        duration: Optional[float] = None,
        wait_for_close: bool = True,
    ) -> Dict[str, Any]:
        """Launches browser, sets up telemetry probe, and captures video + events."""
        import os
        os.makedirs(self.out_dir, mode=0o700, exist_ok=True)
        try:
            os.chmod(self.out_dir, 0o700)
        except OSError:
            pass

        # Initialize empty telemetry file with 0o600 permissions
        try:
            fd = os.open(self.telemetry_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.close(fd)
            try:
                os.chmod(self.telemetry_path, 0o600)
            except OSError:
                pass
        except Exception:
            self.telemetry_path.write_text("", encoding="utf-8")

        self.telemetry_events.clear()
        self.truncated = False
        self._max_warning_logged = False
        self._write_session_metadata(url)

        start_time = time.perf_counter()

        async with async_playwright() as p:
            browser: Browser = await p.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
            )

            context: BrowserContext = await browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height},
                record_video_dir=str(self.out_dir),
                record_video_size={"width": self.viewport_width, "height": self.viewport_height},
            )

            # Apply netguard confinement route guard to recording session
            from ocular.security.netguard import get_allowed_hosts, setup_confinement_routes

            active_allowed = set(get_allowed_hosts())
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                if parsed.hostname:
                    if parsed.port:
                        active_allowed.add(f"{parsed.hostname.lower()}:{parsed.port}")
                    else:
                        active_allowed.add(parsed.hostname.lower())
            except Exception:
                pass

            await setup_confinement_routes(context, allowed_hosts=active_allowed)

            # Expose telemetry binding and inject listeners
            await context.expose_binding("__ocular_push_telemetry", self._handle_telemetry_binding)
            await context.add_init_script(self.get_init_script())

            page: Page = await context.new_page()

            close_event = asyncio.Event()

            def _on_close(_: Any) -> None:
                close_event.set()

            page.on("close", _on_close)
            context.on("close", _on_close)

            # Navigate to target URL
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as exc:
                logger.warning(f"Navigation warning for {url}: {exc}")

            # Keep session open until closed or duration reached
            if wait_for_close and not self.headless:
                try:
                    if duration:
                        await asyncio.wait_for(close_event.wait(), timeout=duration)
                    else:
                        await close_event.wait()
                except asyncio.TimeoutError:
                    pass
            elif duration:
                await asyncio.sleep(duration)
            else:
                # Default short settle in headless mode
                await page.wait_for_timeout(500)

            video = page.video
            raw_video_path = None
            if video:
                # Need to close context before video path is flushed
                await context.close()
                raw_video_path = await video.path()
                self.video_path = Path(raw_video_path)
            else:
                await context.close()

            await browser.close()

        elapsed_total = round((time.perf_counter() - start_time) * 1000.0, 2)
        return {
            "video_path": str(self.video_path) if self.video_path else None,
            "telemetry_path": str(self.telemetry_path),
            "event_count": len(self.telemetry_events),
            "events": self.telemetry_events,
            "url": url,
            "elapsed_ms": elapsed_total,
            "truncated": getattr(self, "truncated", False),
        }

    def record(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """Synchronous helper for record_session."""
        return asyncio.run(self.record_session(url, **kwargs))

    @staticmethod
    def load_telemetry_events(path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Parses telemetry_events.jsonl into a list of event dictionaries, safely rejecting non-dict payloads."""
        from ocular.telemetry import load_telemetry_events as _load
        events, _ = _load(path)
        return events
        file_path = Path(path)
        if not file_path.exists():
            return []
        events: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            coerced = OcularSessionRecorder.coerce_telemetry_event(data)
                            if coerced is not None:
                                events.append(coerced)
                        else:
                            logger.debug(f"Ignoring non-dictionary telemetry payload: {type(data)}")
                    except (json.JSONDecodeError, ValueError):
                        continue
        return events


# Backward-compatibility alias
SessionRecorder = OcularSessionRecorder
