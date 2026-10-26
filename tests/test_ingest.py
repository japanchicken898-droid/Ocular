"""Tests for Frame and Transition Ingest Stage."""

from pathlib import Path
import numpy as np
import pytest
from ocular.models.types import ActionType
from ocular.pipeline.video_ingest import IngestPipeline, KeyframeIndex, VideoIngestSampler
from samples.create_sample_video import generate_synthetic_screen_recording


def test_ingest_frames_list():
    pipeline = IngestPipeline(diff_threshold=0.05)

    # Create 3 synthetic frames: 2 identical, 1 different
    f1 = np.zeros((480, 640, 3), dtype=np.uint8)
    f2 = np.zeros((480, 640, 3), dtype=np.uint8)
    f3 = np.ones((480, 640, 3), dtype=np.uint8) * 200

    frames_meta, transitions, raw_frames = pipeline.process_frames_list([f1, f2, f3])

    assert len(frames_meta) == 3
    assert len(raw_frames) == 3
    # First frame is always keyframe
    assert frames_meta[0].is_keyframe is True
    # Second frame is identical, diff ~ 0 -> not keyframe
    assert frames_meta[1].is_keyframe is False
    # Third frame is significantly different -> keyframe
    assert frames_meta[2].is_keyframe is True

    # One transition between frame 1 and frame 2
    assert len(transitions) >= 1
    assert transitions[0].to_frame_index == 2
    assert transitions[0].action_type == ActionType.CLICK


def test_video_ingest_timeline_and_slices(tmp_path: Path):
    video_path = tmp_path / "test_ingest.mp4"
    generate_synthetic_screen_recording(video_path, fps=30, width=640, height=360)

    sampler = VideoIngestSampler(sample_rate=5.0, diff_threshold=0.04)
    timeline = sampler.ingest_timeline(video_path)

    assert isinstance(timeline, KeyframeIndex)
    assert timeline.video_info["width"] == 640
    assert timeline.video_info["height"] == 360
    assert len(timeline.keyframes) >= 1

    # Test keyframe slice export
    slices_dir = tmp_path / "exported_slices"
    exported = timeline.export_slices(slices_dir)
    assert len(exported) == len(timeline.keyframes)
    for p in exported:
        assert p.exists()
