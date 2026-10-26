"""Keyframe transition sampler stub for Ocular video ingest stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


class VideoIngestSampler:
    """Keyframe transition sampler stub for screen recordings (.mp4)."""

    def __init__(self, sample_rate: float = 2.0, diff_threshold: float = 0.08) -> None:
        self.sample_rate = sample_rate
        self.diff_threshold = diff_threshold

    def sample_transitions(
        self, video_path: str | Path
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[np.ndarray]]:
        """Samples keyframes and detects UI state transitions from an MP4 file.

        Returns:
            - video_info: dict with resolution, fps, frame count
            - transitions: list of transition event records
            - sampled_frames: list of raw frame arrays
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Screen recording not found at: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        step = max(1, int(round(fps / self.sample_rate)))

        sampled_frames: List[np.ndarray] = []
        transitions: List[Dict[str, Any]] = []
        prev_gray: np.ndarray | None = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = 0.0
                if prev_gray is not None:
                    diff = float(np.mean(cv2.absdiff(prev_gray, gray)) / 255.0)

                is_keyframe = (diff >= self.diff_threshold) or (len(sampled_frames) == 0)
                sampled_frames.append(frame)

                if is_keyframe and len(sampled_frames) > 1:
                    transitions.append(
                        {
                            "from_frame": len(sampled_frames) - 2,
                            "to_frame": len(sampled_frames) - 1,
                            "timestamp_ms": round((frame_idx / fps) * 1000.0, 2),
                            "diff_score": round(diff, 4),
                            "action": "click",
                        }
                    )
                prev_gray = gray

            frame_idx += 1

        cap.release()

        video_info = {
            "path": str(video_path),
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "fps": int(round(fps)),
            "total_frames": total_frames,
            "sampled_count": len(sampled_frames),
        }

        return video_info, transitions, sampled_frames
