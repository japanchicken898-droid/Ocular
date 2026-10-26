"""Frame and Transition Ingestion Stage for Ocular Compiler."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

from ocular.models.types import FrameMetadata, TransitionEvent, ActionType


class IngestPipeline:
    """Ingests video recordings (.mp4) and extracts keyframes and transition events."""

    def __init__(
        self,
        diff_threshold: float = 0.12,
        target_fps: float = 2.0,
        output_frames_dir: Optional[Path] = None,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.target_fps = target_fps
        self.output_frames_dir = output_frames_dir
        if self.output_frames_dir:
            self.output_frames_dir.mkdir(parents=True, exist_ok=True)

    def process_video(
        self, video_path: str | Path
    ) -> Tuple[List[FrameMetadata], List[TransitionEvent], List[np.ndarray]]:
        """Processes an .mp4 screen recording and returns frames and transitions."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video recording not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_step = max(1, int(round(fps / self.target_fps)))

        frames_meta: List[FrameMetadata] = []
        raw_frames: List[np.ndarray] = []
        transitions: List[TransitionEvent] = []

        prev_gray: Optional[np.ndarray] = None
        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                h, w = frame.shape[:2]
                timestamp_ms = (frame_idx / fps) * 1000.0

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff_score = 0.0

                if prev_gray is not None:
                    # Normalized Mean Absolute Difference between consecutive sampled frames
                    diff = cv2.absdiff(prev_gray, gray)
                    diff_score = float(np.mean(diff) / 255.0)

                is_keyframe = (diff_score >= self.diff_threshold) or (saved_idx == 0)

                frame_path = None
                if self.output_frames_dir:
                    frame_filename = f"frame_{saved_idx:04d}_{int(timestamp_ms)}ms.jpg"
                    full_path = self.output_frames_dir / frame_filename
                    cv2.imwrite(str(full_path), frame)
                    frame_path = str(full_path)

                meta = FrameMetadata(
                    frame_index=saved_idx,
                    timestamp_ms=timestamp_ms,
                    image_path=frame_path,
                    width=w,
                    height=h,
                    is_keyframe=is_keyframe,
                    diff_score=diff_score,
                )
                frames_meta.append(meta)
                raw_frames.append(frame)

                if is_keyframe and saved_idx > 0:
                    transitions.append(
                        TransitionEvent(
                            from_frame_index=saved_idx - 1,
                            to_frame_index=saved_idx,
                            timestamp_ms=timestamp_ms,
                            action_type=ActionType.CLICK,
                            diff_score=diff_score,
                            confidence=min(1.0, 0.7 + diff_score),
                        )
                    )

                prev_gray = gray
                saved_idx += 1

            frame_idx += 1

        cap.release()

        # Always ensure at least the initial frame is marked as keyframe
        if frames_meta:
            frames_meta[0].is_keyframe = True

        return frames_meta, transitions, raw_frames

    def process_frames_list(
        self, frames: List[np.ndarray], timestamps_ms: Optional[List[float]] = None
    ) -> Tuple[List[FrameMetadata], List[TransitionEvent], List[np.ndarray]]:
        """Processes an in-memory list of image arrays (useful for testing and synthetic pipelines)."""
        frames_meta: List[FrameMetadata] = []
        transitions: List[TransitionEvent] = []

        prev_gray: Optional[np.ndarray] = None

        for idx, frame in enumerate(frames):
            h, w = frame.shape[:2]
            ts = timestamps_ms[idx] if timestamps_ms else float(idx * 500.0)

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            diff_score = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                diff_score = float(np.mean(diff) / 255.0)

            is_keyframe = (diff_score >= self.diff_threshold) or (idx == 0)

            frame_path = None
            if self.output_frames_dir:
                frame_filename = f"frame_{idx:04d}_{int(ts)}ms.jpg"
                full_path = self.output_frames_dir / frame_filename
                cv2.imwrite(str(full_path), frame)
                frame_path = str(full_path)

            meta = FrameMetadata(
                frame_index=idx,
                timestamp_ms=ts,
                image_path=frame_path,
                width=w,
                height=h,
                is_keyframe=is_keyframe,
                diff_score=diff_score,
            )
            frames_meta.append(meta)

            if is_keyframe and idx > 0:
                transitions.append(
                    TransitionEvent(
                        from_frame_index=idx - 1,
                        to_frame_index=idx,
                        timestamp_ms=ts,
                        action_type=ActionType.CLICK,
                        diff_score=diff_score,
                        confidence=min(1.0, 0.75 + diff_score),
                    )
                )

            prev_gray = gray

        if frames_meta:
            frames_meta[0].is_keyframe = True

        return frames_meta, transitions, frames
