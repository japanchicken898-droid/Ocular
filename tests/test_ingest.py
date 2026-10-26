"""Tests for Frame and Transition Ingest Stage."""

import numpy as np
import pytest
from ocular.pipeline.ingest import IngestPipeline
from ocular.models.types import ActionType


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
