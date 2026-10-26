"""Frame and Transition Ingestion Stage for Ocular Compiler.

Re-exports VideoIngestSampler, IngestPipeline, KeyframeIndex, and KeyframeSlice
from ocular.pipeline.video_ingest.
"""

from __future__ import annotations

from ocular.pipeline.video_ingest import (
    IngestPipeline,
    KeyframeIndex,
    KeyframeSlice,
    VideoIngestSampler,
)

__all__ = [
    "IngestPipeline",
    "VideoIngestSampler",
    "KeyframeIndex",
    "KeyframeSlice",
]
