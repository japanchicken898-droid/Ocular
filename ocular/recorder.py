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
import time
from typing import Any, Callable, Dict, List, Optional, Union

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

logger = logging.getLogger("ocular.recorder")


class OcularSessionRecorder:
    """Records an interactive browser session with video and telemetry events."""

    TELEMETRY_INIT_SCRIPT = """
    (() => {
        if (window !== window.top) return;
        const startTime = performance.now();
        const sessionEpoch = Date.now();

        function getElapsedMs() {
            return Math.round(performance.now() - startTime);
        }

        function getElementSelector(el) {
            if (!el || el === document.body || el === document.documentElement) return null;
            if (el.id) return '#' + el.id;
            let selector = el.tagName.toLowerCase();
            if (el.className && typeof el.className === 'string') {
                const classes = el.className.trim().split(/\\s+/).slice(0, 2).filter(Boolean);
                if (classes.length > 0) {
                    selector += '.' + classes.join('.');
                }
            }
            if (el.getAttribute && el.getAttribute('data-role')) {
                selector += `[data-role="${el.getAttribute('data-role')}"]`;
            }
            return selector;
        }

        window.addEventListener('mousedown', (e) => {
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
                target_tag: e.target && e.target.tagName ? e.target.tagName.toLowerCase() : null,
                target_selector: getElementSelector(e.target),
                elapsed_ms: getElapsedMs(),
                timestamp_ms: sessionEpoch + getElapsedMs(),
            };
            try {
                if (window.__ocular_push_telemetry) {
                    window.__ocular_push_telemetry(payload);
                }
            } catch (err) {
                console.error('[Ocular Telemetry] Push failed:', err);
            }
        }, true);

        function isSensitiveElement(el) {
            if (!el) return false;
            const tag = (el.tagName || '').toLowerCase();
            const inputType = (el.type || '').toLowerCase();
            if (tag === 'input' && inputType === 'password') {
                return true;
            }
            const isContentEditable = Boolean(el.isContentEditable || (el.getAttribute && (el.getAttribute('contenteditable') === 'true' || el.getAttribute('contenteditable') === '')));
            const isTextarea = tag === 'textarea';
            const sensitiveRegex = /password|token|secret|credit_card|credit-card|cvv|one-time-code|one_time_code|otp|pin|ssn|auth_key|auth-key/i;

            if (isTextarea || isContentEditable) {
                if (sensitiveRegex.test(el.id || '') || sensitiveRegex.test(el.className || '') || sensitiveRegex.test(el.getAttribute ? el.getAttribute('placeholder') || '' : '')) {
                    return true;
                }
            }
            const attrs = ['name', 'id', 'autocomplete', 'placeholder', 'aria-label', 'data-role', 'type', 'aria-description'];
            for (let i = 0; i < attrs.length; i++) {
                const val = el.getAttribute ? el.getAttribute(attrs[i]) : null;
                if (val && sensitiveRegex.test(val)) {
                    return true;
                }
            }
            return false;
        }

        window.addEventListener('keydown', (e) => {
            const vw = window.innerWidth || 1;
            const vh = window.innerHeight || 1;
            let norm_x = 0.5;
            let norm_y = 0.5;
            if (e.target && e.target.getBoundingClientRect) {
                const rect = e.target.getBoundingClientRect();
                norm_x = (rect.left + rect.width / 2) / vw;
                norm_y = (rect.top + rect.height / 2) / vh;
            }
            const isSensitive = isSensitiveElement(e.target);
            const emittedKey = isSensitive ? '[REDACTED]' : e.key;
            const emittedCode = isSensitive ? '[REDACTED]' : e.code;
            const payload = {
                event_type: 'keydown',
                action_type: 'fill',
                key: emittedKey,
                code: emittedCode,
                norm_x: Number(Math.max(0.0, Math.min(1.0, norm_x)).toFixed(4)),
                norm_y: Number(Math.max(0.0, Math.min(1.0, norm_y)).toFixed(4)),
                target_tag: e.target && e.target.tagName ? e.target.tagName.toLowerCase() : null,
                target_selector: getElementSelector(e.target),
                elapsed_ms: getElapsedMs(),
                timestamp_ms: sessionEpoch + getElapsedMs(),
            };
            try {
                if (window.__ocular_push_telemetry) {
                    window.__ocular_push_telemetry(payload);
                }
            } catch (err) {
                console.error('[Ocular Telemetry] Push failed:', err);
            }
        }, true);
    })();
    """

    MAX_TELEMETRY_EVENTS = 2500
    MAX_STRING_ATTR_LEN = 256

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
        self.telemetry_events: List[Dict[str, Any]] = []
        self.telemetry_path = self.out_dir / "telemetry_events.jsonl"
        self.video_path: Optional[Path] = None

    SENSITIVE_ATTR_PATTERN = re.compile(
        r"password|token|secret|credit_card|credit-card|cvv|one-time-code|one_time_code|otp|pin|ssn|auth_key|auth-key|api_key|auth", re.IGNORECASE
    )

    def _truncate_attributes(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Truncates arbitrary string attributes and redacts sensitive inputs."""
        cleaned = {}
        target_str = str(event_data.get("target_selector") or "") + " " + str(event_data.get("target_tag") or "")
        is_sensitive = bool(self.SENSITIVE_ATTR_PATTERN.search(target_str))

        for k, v in event_data.items():
            if isinstance(v, str):
                if is_sensitive and k in ("key", "code", "value", "input_payload"):
                    cleaned[k] = "[REDACTED]"
                elif len(v) > self.MAX_STRING_ATTR_LEN:
                    cleaned[k] = v[:self.MAX_STRING_ATTR_LEN]
                else:
                    cleaned[k] = v
            else:
                cleaned[k] = v

        if is_sensitive and "key" in cleaned:
            cleaned["key"] = "[REDACTED]"
        return cleaned

    def push_telemetry_event(self, event_data: Dict[str, Any]) -> None:
        """Appends an event to in-memory history and streams to telemetry_events.jsonl."""
        if not isinstance(event_data, dict):
            logger.warning(f"Ignoring non-dictionary telemetry event: {type(event_data)}")
            return

        if len(self.telemetry_events) >= self.MAX_TELEMETRY_EVENTS:
            logger.warning(
                f"Telemetry event limit reached ({self.MAX_TELEMETRY_EVENTS}). Dropping event."
            )
            return

        sanitized_event = self._truncate_attributes(event_data)
        self.telemetry_events.append(sanitized_event)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.telemetry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sanitized_event) + "\n")
        if self.on_event_callback:
            try:
                self.on_event_callback(sanitized_event)
            except Exception as e:
                logger.debug(f"Error in on_event_callback: {e}")

    async def _handle_telemetry_binding(self, source: Any, event_data: Dict[str, Any]) -> None:
        """Playwright binding handler called by client-side probe."""
        self.push_telemetry_event(event_data)

    async def record_session(
        self,
        url: str,
        duration: Optional[float] = None,
        wait_for_close: bool = True,
    ) -> Dict[str, Any]:
        """Launches browser, sets up telemetry probe, and captures video + events."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Initialize empty telemetry file
        self.telemetry_path.write_text("", encoding="utf-8")
        self.telemetry_events.clear()
        try:
            (self.out_dir / "session_metadata.json").write_text(
                json.dumps({"url": url, "timestamp": time.time()}),
                encoding="utf-8"
            )
        except Exception:
            pass

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

            # Expose telemetry binding and inject listeners
            await context.expose_binding("__ocular_push_telemetry", self._handle_telemetry_binding)
            await context.add_init_script(self.TELEMETRY_INIT_SCRIPT)

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
        }

    def record(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """Synchronous helper for record_session."""
        return asyncio.run(self.record_session(url, **kwargs))

    @staticmethod
    def load_telemetry_events(path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Parses telemetry_events.jsonl into a list of event dictionaries, safely rejecting non-dict payloads."""
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
                            events.append(data)
                        else:
                            logger.debug(f"Ignoring non-dictionary telemetry payload: {type(data)}")
                    except (json.JSONDecodeError, ValueError):
                        continue
        return events
