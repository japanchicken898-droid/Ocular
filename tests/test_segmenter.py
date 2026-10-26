"""Tests for Meta SAM 3.1 Segmentation stage."""

import pytest
from ocular.pipeline.segmenter import MockSAMSegmenter
from ocular.pipeline.base import BaseSegmenter


def test_mock_sam_segmenter_interface():
    segmenter = MockSAMSegmenter()
    assert isinstance(segmenter, BaseSegmenter)

    segments = segmenter.segment()
    assert len(segments) > 0

    seg_ids = [s["id"] for s in segments]
    assert "chart_canvas" in seg_ids
    assert "interval_dropdown" in seg_ids
    assert "timeframe_15m" in seg_ids

    # Validate normalized scale-invariant coordinates (0.0 -> 1.0)
    for seg in segments:
        bbox = seg["bbox"]
        assert 0.0 <= bbox["x"] <= 1.0
        assert 0.0 <= bbox["y"] <= 1.0
        assert 0.0 <= bbox["width"] <= 1.0
        assert 0.0 <= bbox["height"] <= 1.0
        assert 0.0 <= seg["confidence"] <= 1.0
