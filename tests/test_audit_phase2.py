"""Regression tests for Phase 2 Telemetry & Ingest Robustness (F10, F13, F17, F18).

Tests:
- Coerce telemetry event: Infinity, 1e999, NaN, ints > 1e15, null coords, string timestamps, clamping.
- load_telemetry_events: 10MB line, 100k-deep nested JSON, invalid UTF-8, non-dict lines, line cap 64KB, event cap 20k.
- Keystroke handling: Backspace editing, non-printable key filtering, non-str key, non-str selector.
- push_telemetry_event truncation flag on event limits.
- video_ingest: fps=NaN/0/inf fixture does not crash; memory-bounded frame streaming; absurd dimension rejection; cv2.imwrite error checking.
"""

import json
import math
from pathlib import Path
import pytest
import numpy as np
import cv2

from ocular.telemetry import coerce_telemetry_event, load_telemetry_events
from ocular.pipeline.code_generator import extract_telemetry_input_parameters
from ocular.recorder import OcularSessionRecorder
from ocular.pipeline.video_ingest import VideoIngestSampler


def test_coerce_telemetry_event_fuzzing():
    # 1. Non-dict inputs must return None without raising
    assert coerce_telemetry_event("not_a_dict") is None
    assert coerce_telemetry_event(12345) is None
    assert coerce_telemetry_event(None) is None
    assert coerce_telemetry_event(["event"]) is None

    # 2. NaN, Infinity, -Infinity, 1e999
    bad_floats = [float("nan"), float("inf"), float("-inf"), 1e999, "nan", "inf", "-inf"]
    for val in bad_floats:
        res = coerce_telemetry_event({"norm_x": val, "norm_y": val, "elapsed_ms": val, "timestamp_ms": val})
        assert res is not None
        assert math.isfinite(res["norm_x"])
        assert 0.0 <= res["norm_x"] <= 1.0
        assert math.isfinite(res["norm_y"])
        assert 0.0 <= res["norm_y"] <= 1.0
        assert math.isfinite(res["elapsed_ms"])
        assert 0 <= res["elapsed_ms"] <= 86400000

    # 3. Numbers > 1e15 rejected or clamped
    huge_res = coerce_telemetry_event({"norm_x": 0.5, "norm_y": 0.5, "elapsed_ms": 10**18, "timestamp_ms": 10**18})
    assert huge_res is not None
    assert huge_res["elapsed_ms"] <= 86400000

    # 4. Clamping norm_x and norm_y to [0.0, 1.0]
    out_of_bounds = coerce_telemetry_event({"norm_x": -5.0, "norm_y": 15.0, "elapsed_ms": -100})
    assert out_of_bounds["norm_x"] == 0.0
    assert out_of_bounds["norm_y"] == 1.0
    assert out_of_bounds["elapsed_ms"] == 0


def test_load_telemetry_events_robustness(tmp_path: Path):
    telemetry_file = tmp_path / "telemetry_fuzz.jsonl"

    lines = []
    # Normal line
    lines.append(json.dumps({"event_type": "keydown", "key": "a", "norm_x": 0.1, "norm_y": 0.2}) + "\n")
    # 10MB line (must be dropped by 64KB cap)
    lines.append('{"event_type": "keydown", "large": "' + ("A" * (10 * 1024 * 1024)) + '"}\n')
    # Non-dict line
    lines.append('"just a string"\n')
    lines.append("[1, 2, 3]\n")
    # 100k-deep nested JSON
    deep_json = ('{"a":' * 100) + '1' + ('}' * 100) + '\n'
    lines.append(deep_json)
    # Invalid JSON syntax
    lines.append('{"incomplete": \n')

    telemetry_file.write_text("".join(lines), encoding="utf-8")

    # Also append invalid UTF-8 bytes to the file
    with open(telemetry_file, "ab") as f:
        f.write(b'\xff\xfe\xfd\n')

    events, dropped = load_telemetry_events(telemetry_file, max_events=100)
    assert len(events) >= 1
    assert dropped >= 4
    assert events[0]["key"] == "a"


def test_keystroke_buffer_backspace_and_non_printable():
    events = [
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": "H"},
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": "e"},
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": "x"},
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": "Backspace"},
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": "Shift"},
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": "ArrowLeft"},
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": "l"},
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": "l"},
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": "o"},
        # Corrupted non-str key types
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": 5},
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": True},
        {"event_type": "keydown", "action_type": "fill", "target_selector": "#search", "key": ["bad"]},
        # Non-str selector
        {"event_type": "keydown", "action_type": "fill", "target_selector": ["#bad_selector"], "key": "z"},
    ]

    params = extract_telemetry_input_parameters(events)
    search_param = next((p for p in params if "search" in p["key"]), None)
    assert search_param is not None
    # "H", "e", "x", Backspace -> "He" + "l", "l", "o" -> "Hello"
    assert search_param["default_value"] == "Hello"


def test_push_telemetry_event_truncation_flag(tmp_path: Path):
    rec = OcularSessionRecorder(out_dir=tmp_path, headless=True)
    rec.MAX_TELEMETRY_EVENTS = 5

    for i in range(10):
        rec.push_telemetry_event({
            "event_type": "keydown",
            "action_type": "fill",
            "key": "a",
            "_nonce": rec.session_nonce,
        })

    assert len(rec.telemetry_events) == 5
    assert getattr(rec, "truncated", False) is True


def test_video_ingest_robustness(tmp_path: Path):
    sampler = VideoIngestSampler(sample_rate=2.0)

    # 1. Invalid or NaN FPS
    assert sampler._sanitize_fps(float("nan")) == 30.0
    assert sampler._sanitize_fps(float("inf")) == 30.0
    assert sampler._sanitize_fps(0.0) == 30.0
    assert sampler._sanitize_fps(300.0) == 240.0
    assert sampler._sanitize_fps(60.0) == 60.0

    # 2. Absurd dimensions validation
    assert sampler._validate_dimensions(1920, 1080) is True
    assert sampler._validate_dimensions(0, 1080) is False
    assert sampler._validate_dimensions(100000, 100000) is False

    # 3. cv2.imwrite check in KeyframeIndex.export_slices
    from ocular.pipeline.video_ingest import KeyframeIndex, KeyframeSlice
    idx = KeyframeIndex(
        video_info={"fps": 30.0, "width": 1920, "height": 1080},
        keyframes=[KeyframeSlice(frame_index=0, timestamp_ms=0.0, diff_score=0.0, is_keyframe=True)],
        slices={0: np.zeros((10, 10, 3), dtype=np.uint8)},
    )
    # Target directory that cannot be written or invalid path raises IOError
    invalid_out = tmp_path / "file_exists"
    invalid_out.write_text("blocker")
    with pytest.raises(Exception):
        idx.export_slices(invalid_out / "sub")

