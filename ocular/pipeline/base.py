"""Abstract Pipeline Interfaces for Ocular Compiler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class BaseSegmenter(ABC):
    """Abstract base class for vision/element segmenters (e.g. Meta SAM 3.1, OpenCV Heuristics)."""

    @abstractmethod
    def segment(
        self,
        frame: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Segments interactive elements on screen into normalized bounding boxes & labels.

        Returns list of segment dicts with keys:
          - id: str
          - label: str
          - role: str
          - bbox: dict with normalized x, y, width, height (0.0 -> 1.0)
          - confidence: float
          - metadata: dict
        """
        pass

    def segment_batch(
        self,
        frames: List[np.ndarray],
        frames_meta: Optional[List[Any]] = None,
    ) -> Dict[int, List[Any]]:
        """Segments a batch of frames, keyed by frame index."""
        result: Dict[int, List[Any]] = {}
        for idx, frame in enumerate(frames):
            meta = frames_meta[idx] if frames_meta and idx < len(frames_meta) else None
            meta_dict = meta.model_dump() if hasattr(meta, "model_dump") else {}
            result[idx] = self.segment(frame, idx, meta_dict)
        return result


class BaseSynthesizer(ABC):
    """Abstract base class for state graph / DAG synthesizers (e.g. Muse Spark)."""

    @abstractmethod
    def build_dag(
        self,
        segments: List[Dict[str, Any]],
        transitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Synthesizes interactive segments & transitions into a route & dependency DAG."""
        pass


class BaseGenerator(ABC):
    """Abstract base class for code artifact emission (e.g. Muse Code)."""

    @abstractmethod
    def emit(
        self,
        dag: Dict[str, Any],
        segments: List[Dict[str, Any]],
        out_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """Emits executable automation driver and service wrapper to out_dir."""
        pass
