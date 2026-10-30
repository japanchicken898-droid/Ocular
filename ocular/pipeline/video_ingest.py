"""Video and Frame Ingestion Stage for Ocular Compiler.

Extracts frames using OpenCV, computes temporal frame differencing to identify
key interaction moments (scene change thresholding, mouse idle detection, and
action boundaries), and emits timestamped keyframe image slices into an
indexed intermediate data structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from ocular.models.types import ActionType, FrameMetadata, TransitionEvent

logger = logging.getLogger("ocular.pipeline.video_ingest")


@dataclass
class KeyframeSlice:
    """Timestamped keyframe image slice with region of interest."""

    frame_index: int
    timestamp_ms: float
    diff_score: float
    is_keyframe: bool
    image: Optional[np.ndarray] = None
    image_path: Optional[str] = None
    roi_bbox: Optional[Dict[str, float]] = None  # Normalized changed region [0.0, 1.0]
    motion_energy: float = 0.0


@dataclass
class KeyframeIndex:
    """Indexed intermediate data structure holding timeline keyframes and slices."""

    video_info: Dict[str, Any]
    frames_meta: List[FrameMetadata] = field(default_factory=list)
    keyframes: List[KeyframeSlice] = field(default_factory=list)
    transitions: List[TransitionEvent] = field(default_factory=list)
    slices: Dict[int, np.ndarray] = field(default_factory=dict)

    def get_slice(self, frame_index: int) -> Optional[np.ndarray]:
        return self.slices.get(frame_index)

    def export_slices(self, output_dir: Union[str, Path]) -> List[Path]:
        """Saves all keyframe image slices to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []
        for kf in self.keyframes:
            frame_arr = self.slices.get(kf.frame_index)
            if frame_arr is not None:
                fname = f"keyframe_{kf.frame_index:04d}_{int(kf.timestamp_ms)}ms.jpg"
                fpath = out_path / fname
                cv2.imwrite(str(fpath), frame_arr)
                kf.image_path = str(fpath)
                saved_paths.append(fpath)
        return saved_paths


class VideoIngestSampler:
    """Extracts frames, computes temporal differencing, and indexes keyframe moments."""

    def __init__(
        self,
        sample_rate: float = 2.0,
        diff_threshold: float = 0.08,
        idle_threshold: float = 0.015,
        output_frames_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the video ingest sampler.

        Args:
            sample_rate: Target sampling rate in Hz (frames per second).
            diff_threshold: Normalized mean absolute pixel difference threshold [0.0, 1.0].
            idle_threshold: Difference threshold below which screen is considered idle/settled.
            output_frames_dir: Optional directory to persist extracted keyframes.
        """
        self.sample_rate = sample_rate
        self.diff_threshold = diff_threshold
        self.idle_threshold = idle_threshold
        self.output_frames_dir = Path(output_frames_dir) if output_frames_dir else None
        if self.output_frames_dir:
            self.output_frames_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _compute_frame_diff(
        prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> Tuple[float, Optional[Dict[str, float]], float]:
        """Computes normalized diff score, changed ROI bounding box, and motion energy."""
        # Mean absolute difference
        abs_diff = cv2.absdiff(prev_gray, curr_gray)
        diff_score = float(np.mean(abs_diff) / 255.0)

        # Motion energy (standard deviation of changes)
        motion_energy = float(np.std(abs_diff) / 255.0)

        # Find changed region bounding box via thresholding & contours
        h, w = curr_gray.shape[:2]
        _, thresh = cv2.threshold(abs_diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_bbox: Optional[Dict[str, float]] = None
        if contours:
            # Combine major changed contours
            min_x, min_y, max_x, max_y = w, h, 0, 0
            has_valid_contour = False
            for c in contours:
                if cv2.contourArea(c) > 50:  # Filter minor noise
                    x, y, cw, ch = cv2.boundingRect(c)
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + cw)
                    max_y = max(max_y, y + ch)
                    has_valid_contour = True

            if has_valid_contour and w > 0 and h > 0:
                roi_bbox = {
                    "x": round(max(0.0, min_x / w), 4),
                    "y": round(max(0.0, min_y / h), 4),
                    "width": round(min(1.0, (max_x - min_x) / w), 4),
                    "height": round(min(1.0, (max_y - min_y) / h), 4),
                }

        return diff_score, roi_bbox, motion_energy

    def sample_transitions(
        self,
        video_path: Union[str, Path],
        telemetry_events: Optional[List[Dict[str, Any]]] = None,
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

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        step = max(1, int(round(fps / self.sample_rate)))

        sampled_frames: List[np.ndarray] = []
        transitions: List[Dict[str, Any]] = []
        prev_gray: Optional[np.ndarray] = None
        frame_idx = 0

        # Filter telemetry clicks as point-prompt priors
        click_events = (
            [
                e
                for e in telemetry_events
                if e.get("event_type") == "mousedown" or e.get("action_type") == "click"
            ]
            if telemetry_events
            else []
        )

        while True:
            if frame_idx % step == 0:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = 0.0
                roi_bbox = None
                motion_energy = 0.0

                if prev_gray is not None:
                    diff, roi_bbox, motion_energy = self._compute_frame_diff(prev_gray, gray)

                curr_ts_ms = round((frame_idx / fps) * 1000.0, 2)
                prev_ts_ms = round(((frame_idx - step) / fps) * 1000.0, 2) if frame_idx >= step else 0.0

                # Match against telemetry click events
                matched_clicks = [
                    c for c in click_events
                    if prev_ts_ms <= c.get("elapsed_ms", c.get("timestamp_ms", -1)) <= curr_ts_ms + 100.0
                ]

                is_keyframe = (diff >= self.diff_threshold) or (len(sampled_frames) == 0) or bool(matched_clicks)
                sampled_frames.append(frame)

                if is_keyframe and len(sampled_frames) > 1:
                    matched_click = matched_clicks[0] if matched_clicks else None
                    if matched_click and not roi_bbox:
                        roi_bbox = {
                            "x": matched_click.get("norm_x", 0.5),
                            "y": matched_click.get("norm_y", 0.5),
                            "width": 0.08,
                            "height": 0.04,
                        }
                    transitions.append(
                        {
                            "from_frame": len(sampled_frames) - 2,
                            "to_frame": len(sampled_frames) - 1,
                            "timestamp_ms": matched_click.get("elapsed_ms", curr_ts_ms) if matched_click else curr_ts_ms,
                            "diff_score": round(max(diff, 0.20 if matched_click else diff), 4),
                            "action": "click",
                            "roi_bbox": roi_bbox,
                            "motion_energy": round(motion_energy, 4),
                            "target_selector": matched_click.get("target_selector") if matched_click else None,
                            "telemetry_event": matched_click,
                        }
                    )
                prev_gray = gray
            else:
                ret = cap.grab()
                if not ret:
                    break

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

    def ingest_timeline(
        self,
        video_path: Union[str, Path],
        telemetry_events: Optional[List[Dict[str, Any]]] = None,
    ) -> KeyframeIndex:
        """Full extraction and temporal differencing emitting an indexed KeyframeIndex structure."""
        video_info, raw_transitions, raw_frames = self.sample_transitions(
            video_path, telemetry_events=telemetry_events
        )

        frames_meta: List[FrameMetadata] = []
        keyframe_slices: List[KeyframeSlice] = []
        transition_events: List[TransitionEvent] = []
        slices_map: Dict[int, np.ndarray] = {}

        fps = video_info["fps"] or 30.0
        step = max(1, int(round(fps / self.sample_rate)))
        w = video_info["width"]
        h = video_info["height"]

        # Map transition info by to_frame
        trans_by_to = {t["to_frame"]: t for t in raw_transitions}

        for idx, frame in enumerate(raw_frames):
            timestamp_ms = round((idx * step / fps) * 1000.0, 2)
            trans_info = trans_by_to.get(idx)
            is_keyframe = (idx == 0) or (trans_info is not None)
            diff_score = trans_info["diff_score"] if trans_info else 0.0
            roi_bbox = trans_info.get("roi_bbox") if trans_info else None
            motion_energy = trans_info.get("motion_energy", 0.0) if trans_info else 0.0

            frame_path: Optional[str] = None
            if self.output_frames_dir and is_keyframe:
                fname = f"keyframe_{idx:04d}_{int(timestamp_ms)}ms.jpg"
                fpath = self.output_frames_dir / fname
                cv2.imwrite(str(fpath), frame)
                frame_path = str(fpath)

            meta = FrameMetadata(
                frame_index=idx,
                timestamp_ms=timestamp_ms,
                image_path=frame_path,
                width=w,
                height=h,
                is_keyframe=is_keyframe,
                diff_score=diff_score,
            )
            frames_meta.append(meta)

            if is_keyframe:
                slices_map[idx] = frame
                keyframe_slices.append(
                    KeyframeSlice(
                        frame_index=idx,
                        timestamp_ms=timestamp_ms,
                        diff_score=diff_score,
                        is_keyframe=True,
                        image=frame,
                        image_path=frame_path,
                        roi_bbox=roi_bbox,
                        motion_energy=motion_energy,
                    )
                )

        for t in raw_transitions:
            transition_events.append(
                TransitionEvent(
                    from_frame_index=t["from_frame"],
                    to_frame_index=t["to_frame"],
                    timestamp_ms=t["timestamp_ms"],
                    action_type=ActionType.CLICK,
                    target_selector=t.get("target_selector"),
                    diff_score=t["diff_score"],
                    confidence=min(1.0, 0.75 + t["diff_score"]),
                )
            )

        return KeyframeIndex(
            video_info=video_info,
            frames_meta=frames_meta,
            keyframes=keyframe_slices,
            transitions=transition_events,
            slices=slices_map,
        )

    def process_video(
        self,
        video_path: Union[str, Path],
        telemetry_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[FrameMetadata], List[TransitionEvent], List[np.ndarray]]:
        """Processes video recording and returns typed frames metadata and transitions."""
        timeline = self.ingest_timeline(video_path, telemetry_events=telemetry_events)
        _, _, all_sampled = self.sample_transitions(video_path, telemetry_events=telemetry_events)
        return timeline.frames_meta, timeline.transitions, all_sampled

    def process_frames_list(
        self,
        frames: List[np.ndarray],
        timestamps_ms: Optional[List[float]] = None,
        telemetry_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[FrameMetadata], List[TransitionEvent], List[np.ndarray]]:
        """Processes in-memory synthetic frame list (for tests & programmatic invocation)."""
        frames_meta: List[FrameMetadata] = []
        transitions: List[TransitionEvent] = []
        prev_gray: Optional[np.ndarray] = None

        click_events = (
            [
                e
                for e in telemetry_events
                if e.get("event_type") == "mousedown" or e.get("action_type") == "click"
            ]
            if telemetry_events
            else []
        )

        for idx, frame in enumerate(frames):
            h, w = frame.shape[:2]
            ts = timestamps_ms[idx] if timestamps_ms else float(idx * 500.0)

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            diff_score = 0.0
            if prev_gray is not None:
                abs_diff = cv2.absdiff(prev_gray, gray)
                diff_score = float(np.mean(abs_diff) / 255.0)

            # Match against telemetry clicks
            matched_clicks = [
                c for c in click_events
                if abs(c.get("elapsed_ms", c.get("timestamp_ms", -1000.0)) - ts) <= 300.0
            ]

            is_keyframe = (diff_score >= self.diff_threshold) or (idx == 0) or bool(matched_clicks)

            frame_path = None
            if self.output_frames_dir and is_keyframe:
                fname = f"frame_{idx:04d}_{int(ts)}ms.jpg"
                fpath = self.output_frames_dir / fname
                cv2.imwrite(str(fpath), frame)
                frame_path = str(fpath)

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
                matched_click = matched_clicks[0] if matched_clicks else None
                transitions.append(
                    TransitionEvent(
                        from_frame_index=idx - 1,
                        to_frame_index=idx,
                        timestamp_ms=ts,
                        action_type=ActionType.CLICK,
                        target_selector=matched_click.get("target_selector") if matched_click else None,
                        diff_score=diff_score,
                        confidence=min(1.0, (0.95 if matched_click else 0.75) + diff_score),
                    )
                )

            prev_gray = gray

        if frames_meta:
            frames_meta[0].is_keyframe = True

        return frames_meta, transitions, frames


# IngestPipeline alias for full backward compatibility
IngestPipeline = VideoIngestSampler
