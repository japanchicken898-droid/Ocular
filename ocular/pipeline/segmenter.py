"""Meta SAM 3.1 Segmentation Stage (Mocked/Abstracted).

Provides MockSAMSegmenter returning normalized relative coordinates (0.0 -> 1.0),
element IDs, and visual anchor labels (e.g., interval_dropdown, chart_canvas, timeframe_15m).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from ocular.pipeline.base import BaseSegmenter


class MockSAMSegmenter(BaseSegmenter):
    """Mock implementation of Meta SAM 3.1 segmentation engine.

    Emits normalized relative coordinates (0.0 -> 1.0) for canvas interactive masks,
    visual anchor labels, and element IDs for offline compile-time resolution.
    """

    def __init__(self, fixtures_path: Optional[str | Path] = None) -> None:
        self.fixtures_path = (
            Path(fixtures_path)
            if fixtures_path
            else Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_masks.json"
        )
        self._cached_segments: Optional[List[Dict[str, Any]]] = None

    def _load_fixtures(self) -> List[Dict[str, Any]]:
        if self.fixtures_path.exists():
            try:
                data = json.loads(self.fixtures_path.read_text(encoding="utf-8"))
                return data.get("segments", [])
            except Exception:
                pass
        return self._default_segments()

    def _default_segments(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "chart_canvas",
                "label": "chart_canvas",
                "role": "canvas",
                "bbox": {"x": 0.05, "y": 0.12, "width": 0.90, "height": 0.80},
                "confidence": 0.995,
                "metadata": {"tag": "canvas.tradingview-chart"},
            },
            {
                "id": "interval_dropdown",
                "label": "interval_dropdown",
                "role": "button",
                "bbox": {"x": 0.12, "y": 0.04, "width": 0.06, "height": 0.04},
                "confidence": 0.982,
                "metadata": {"tag": "div[data-role='interval-menu']"},
            },
            {
                "id": "timeframe_15m",
                "label": "timeframe_15m",
                "role": "menuitem",
                "bbox": {"x": 0.12, "y": 0.18, "width": 0.08, "height": 0.035},
                "confidence": 0.974,
                "metadata": {"value": "15m", "text": "15 minutes"},
            },
            {
                "id": "timeframe_1h",
                "label": "timeframe_1h",
                "role": "menuitem",
                "bbox": {"x": 0.12, "y": 0.22, "width": 0.08, "height": 0.035},
                "confidence": 0.968,
                "metadata": {"value": "1h", "text": "1 hour"},
            },
            {
                "id": "timeframe_1d",
                "label": "timeframe_1d",
                "role": "menuitem",
                "bbox": {"x": 0.12, "y": 0.26, "width": 0.08, "height": 0.035},
                "confidence": 0.971,
                "metadata": {"value": "1d", "text": "1 day"},
            },
        ]

    def segment(
        self,
        frame: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Segments canvas interactive masks, returning normalized relative bounding boxes."""
        if self._cached_segments is None:
            self._cached_segments = self._load_fixtures()

        # Return deep copy so callers cannot mutate internal state
        return [dict(s) for s in self._cached_segments]
