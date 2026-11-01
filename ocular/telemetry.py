"""Unified Telemetry Ingestion, Coercion, and Deserialization Module.

Provides robust, memory-bounded parsing of telemetry streams, clamping numeric fields,
preventing recursion/parser denial-of-service, and eliminating duplicate coercion logic.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("ocular.telemetry")

MAX_NUMERIC_VALUE = 1e15
MAX_ELAPSED_MS = 86_400_000  # 24 hours in milliseconds
MAX_LINE_LENGTH = 65_536  # 64 KB per line cap
MAX_EVENT_SIZE = 4_096   # 4 KB per serialized event cap
MAX_CUMULATIVE_BYTES = 50 * 1024 * 1024  # 50 MB total file budget
DISCARD_CHUNK_SIZE = 16_384  # 16 KB chunks for discarding oversized lines


def coerce_telemetry_event(event: Any) -> Optional[Dict[str, Any]]:
    """Safely coerces numeric fields in a telemetry event dictionary, rejecting non-dict objects.
    
    Never raises exceptions. Enforces finite float bounds and clamps coordinates and timestamps.
    """
    if not isinstance(event, dict):
        return None

    try:
        e = dict(event)

        # 1. Coerce elapsed_ms: clamped to [0, 86400000] (24h)
        val_elapsed = e.get("elapsed_ms")
        if val_elapsed is None:
            raw_ts = e.get("timestamp_ms")
            val_elapsed = raw_ts if raw_ts is not None else 0

        try:
            num_elapsed = float(val_elapsed)
            if not math.isfinite(num_elapsed) or num_elapsed > MAX_NUMERIC_VALUE:
                elapsed_int = 0
            else:
                elapsed_int = int(max(0, min(MAX_ELAPSED_MS, num_elapsed)))
        except (ValueError, TypeError, OverflowError):
            elapsed_int = 0
        e["elapsed_ms"] = elapsed_int

        # 2. Coerce timestamp_ms
        val_ts = e.get("timestamp_ms")
        try:
            num_ts = float(val_ts) if val_ts is not None else float(elapsed_int)
            if not math.isfinite(num_ts) or num_ts > MAX_NUMERIC_VALUE or num_ts < 0:
                ts_float = float(elapsed_int)
            else:
                ts_float = num_ts
        except (ValueError, TypeError, OverflowError):
            ts_float = float(elapsed_int)
        e["timestamp_ms"] = ts_float

        # 3. Coerce norm_x: clamped to [0.0, 1.0]
        val_x = e.get("norm_x")
        try:
            num_x = float(val_x) if val_x is not None else 0.0
            if not math.isfinite(num_x):
                x_float = 0.0
            else:
                x_float = float(max(0.0, min(1.0, num_x)))
        except (ValueError, TypeError, OverflowError):
            x_float = 0.0
        e["norm_x"] = round(x_float, 4)

        # 4. Coerce norm_y: clamped to [0.0, 1.0]
        val_y = e.get("norm_y")
        try:
            num_y = float(val_y) if val_y is not None else 0.0
            if not math.isfinite(num_y):
                y_float = 0.0
            else:
                y_float = float(max(0.0, min(1.0, num_y)))
        except (ValueError, TypeError, OverflowError):
            y_float = 0.0
        e["norm_y"] = round(y_float, 4)

        return e
    except Exception as exc:
        logger.debug(f"Unexpected error coercing telemetry event: {exc}")
        return None


def load_telemetry_events(
    path: Union[str, Path],
    max_events: int = 20_000,
) -> Tuple[List[Dict[str, Any]], int]:
    """Safely loads telemetry events from a JSONL file with memory and line size bounds.

    Returns:
        A tuple of (sanitized_events, dropped_count).
    """
    file_path = Path(path)
    if not file_path.exists():
        return [], 0

    events: List[Dict[str, Any]] = []
    dropped_count = 0
    total_bytes_read = 0

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            while len(events) < max_events:
                if total_bytes_read >= MAX_CUMULATIVE_BYTES:
                    logger.warning(
                        f"Telemetry file {file_path.name} exceeded cumulative byte budget ({MAX_CUMULATIVE_BYTES} bytes)."
                    )
                    break

                chunk = f.readline(MAX_LINE_LENGTH + 1)
                if not chunk:
                    break

                chunk_len = len(chunk)
                total_bytes_read += chunk_len

                # If chunk length exceeds MAX_LINE_LENGTH, or line was cut off without newline before EOF
                if chunk_len > MAX_LINE_LENGTH or (not chunk.endswith("\n") and not chunk.endswith("\r")):
                    # Check if at EOF: if reading 1 more char gives EOF, then chunk is the last line
                    next_char = f.readline(1)
                    if next_char:
                        total_bytes_read += 1
                        # It's an oversized line! Discard the rest of the line in bounded chunks without buffering
                        while True:
                            discard = f.readline(DISCARD_CHUNK_SIZE)
                            if not discard:
                                break
                            total_bytes_read += len(discard)
                            if discard.endswith("\n") or discard.endswith("\r"):
                                break
                        dropped_count += 1
                        continue

                stripped = chunk.strip()
                if not stripped:
                    continue

                # Enforce per-event serialized size cap (4 KB)
                if len(stripped.encode("utf-8")) > MAX_EVENT_SIZE:
                    dropped_count += 1
                    continue

                try:
                    data = json.loads(stripped)
                    coerced = coerce_telemetry_event(data)
                    if coerced is not None and isinstance(coerced, dict):
                        events.append(coerced)
                    else:
                        dropped_count += 1
                except (Exception, RecursionError):
                    dropped_count += 1

        if dropped_count > 0:
            logger.warning(
                f"Loaded {len(events)} telemetry events from {file_path.name}; dropped {dropped_count} corrupted/oversized lines."
            )

        return events, dropped_count
    except Exception as exc:
        logger.warning(f"Failed to read telemetry events file {file_path}: {exc}")
        return events, dropped_count

