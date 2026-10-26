"""Tests for Meta SAM 3.1 & Deterministic OpenCV Segmentation Stage."""

import numpy as np
import pytest
from ocular.models.types import BoundingBox
from ocular.pipeline.base import BaseSegmenter
from ocular.pipeline.segmenter import (
    MetaSAMSegmenter,
    MockSAMSegmenter,
    OpenCVHeuristicSegmenter,
    denormalize_coordinates,
    normalize_coordinates,
)


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


def test_normalize_and_denormalize_coordinates():
    # 1. Pixel tuple normalization
    norm = normalize_coordinates((192, 108, 384, 216), (1920, 1080))
    assert norm["x"] == 0.1
    assert norm["y"] == 0.1
    assert norm["width"] == 0.2
    assert norm["height"] == 0.2

    # 2. BoundingBox normalization
    bbox_obj = BoundingBox(x=960, y=540, width=480, height=270)
    norm2 = normalize_coordinates(bbox_obj, {"width": 1920, "height": 1080})
    assert norm2["x"] == 0.5
    assert norm2["y"] == 0.5
    assert norm2["width"] == 0.25
    assert norm2["height"] == 0.25

    # 3. Denormalize translation back to target resolution
    denorm = denormalize_coordinates(norm, (1920, 1080))
    assert denorm["pixel_x"] == 192.0
    assert denorm["pixel_y"] == 108.0
    assert denorm["center_x"] == 384.0  # (0.1 + 0.1) * 1920
    assert denorm["center_y"] == 216.0  # (0.1 + 0.1) * 1080


def test_opencv_heuristic_segmenter():
    segmenter = OpenCVHeuristicSegmenter(min_area=50)

    # Create synthetic frame with distinct button and input shapes
    frame = np.full((720, 1280, 3), 240, dtype=np.uint8)
    # Draw button (rectangle)
    frame[100:160, 200:450] = [50, 50, 200]
    # Draw input field (wide rectangle)
    frame[250:300, 200:700] = [20, 20, 20]

    segments = segmenter.segment(frame)
    assert len(segments) >= 2

    for s in segments:
        bbox = s["bbox"]
        assert 0.0 <= bbox["x"] <= 1.0
        assert 0.0 <= bbox["y"] <= 1.0
        assert 0.0 <= bbox["width"] <= 1.0
        assert 0.0 <= bbox["height"] <= 1.0
        assert "suggested_selector" in s


def test_meta_sam_offline_fallback():
    # MetaSAMSegmenter should gracefully fall back without active checkpoints
    segmenter = MetaSAMSegmenter(checkpoint_path=None)
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    segments = segmenter.segment(frame)
    assert isinstance(segments, list)


def test_mock_sam_segment_batch():
    segmenter = MockSAMSegmenter()
    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    f2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

    batch = segmenter.segment_batch([f1, f2])
    assert 0 in batch
    assert 1 in batch
    assert len(batch[0]) > 0
    assert hasattr(batch[0][0], "bbox")


def test_export_visual_overlays(tmp_path):
    from pathlib import Path
    from ocular.pipeline.segmenter import export_visual_overlays

    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    segments = [
        {"id": "btn_test", "label": "Test Button", "role": "button", "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1}, "confidence": 0.98},
        {"id": "input_test", "label": "Username", "role": "input", "bbox": {"x": 0.1, "y": 0.3, "width": 0.4, "height": 0.08}, "confidence": 0.95},
    ]

    annotated_dir = tmp_path / "annotated"
    exported = export_visual_overlays([frame], segments, output_dir=annotated_dir)

    assert len(exported) == 1
    assert exported[0].exists()
    assert exported[0].name == "annotated_frame_0000.jpg"

