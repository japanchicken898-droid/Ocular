"""Ocular Compiler Orchestrator.

Moves multimodal vision inference from runtime to compile time by executing
the modular stages: Ingest -> SAM 3.1 Segmentation -> Muse DAG Synthesis -> Code Generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple, Union
import numpy as np

from ocular.models.types import (
    CompiledArtifacts,
    FrameMetadata,
    TransitionEvent,
    VisualSegment,
    StateGraph,
)
from ocular.pipeline.ingest import IngestPipeline
from ocular.pipeline.segmenter import BaseSegmenter, MockSAMSegmenter
from ocular.pipeline.state_graph import BaseSynthesizer, MockMuseSynthesizer
from ocular.pipeline.code_generator import CodeGenerator, derive_workflow_name_from_url
from ocular.recorder import OcularSessionRecorder


def discover_recording_assets(
    input_path: Union[str, Path],
    telemetry_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Optional[Path]]:
    """Resolves video file and telemetry stream from an exact file path or session directory.

    If input_path is a directory:
      - Automatically searches for .webm and .mp4 video recordings (including Playwright page@*.webm patterns).
      - If multiple video files are found, selects the newest file by modification timestamp (st_mtime).
      - Automatically discovers sibling telemetry_events.jsonl within that directory unless telemetry_path is explicitly provided.

    If input_path is an exact file:
      - Retains existing behavior.
      - Discovers sibling telemetry_events.jsonl unless telemetry_path is explicitly provided.

    Returns:
      (resolved_video_file_path, resolved_telemetry_file_path)

    Raises:
      FileNotFoundError: If input path does not exist or target directory contains no video recordings.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file():
        video_file = path
        telem_file = (
            Path(telemetry_path)
            if telemetry_path
            else (path.parent / "telemetry_events.jsonl")
        )
        return video_file, (telem_file if telem_file.exists() else None)

    if path.is_dir():
        # Search direct children first
        videos = [
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in (".webm", ".mp4")
        ]
        # If not found directly, search recursively for nested session directories
        if not videos:
            videos = [
                f for f in path.rglob("*")
                if f.is_file() and f.suffix.lower() in (".webm", ".mp4")
            ]

        if not videos:
            raise FileNotFoundError(
                f"No video recording (.webm or .mp4) found in directory: {path}"
            )

        # Sort by modification time descending (newest first)
        videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        video_file = videos[0]

        if telemetry_path:
            telem_candidate = Path(telemetry_path)
            telem_file = telem_candidate if telem_candidate.exists() else None
        else:
            adjacent_telem = video_file.parent / "telemetry_events.jsonl"
            dir_telem = path / "telemetry_events.jsonl"
            if adjacent_telem.exists():
                telem_file = adjacent_telem
            elif dir_telem.exists():
                telem_file = dir_telem
            else:
                telem_file = None

        return video_file, telem_file

    raise FileNotFoundError(f"Input path is neither a file nor a directory: {path}")


class OcularCompiler:
    """Orchestrates the offline visual compilation pipeline."""

    def __init__(
        self,
        segmenter: Optional[BaseSegmenter] = None,
        synthesizer: Optional[BaseSynthesizer] = None,
        ingest: Optional[IngestPipeline] = None,
        generator: Optional[CodeGenerator] = None,
        on_stage_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.segmenter = segmenter or MockSAMSegmenter()
        self.synthesizer = synthesizer or MockMuseSynthesizer()
        self.ingest = ingest or IngestPipeline()
        self.generator = generator or CodeGenerator()
        self.on_stage_callback = on_stage_callback

    def _notify(self, stage: str, data: Dict[str, Any]) -> None:
        if self.on_stage_callback:
            self.on_stage_callback(stage, data)

    def compile_video(
        self,
        video_path: str | Path,
        output_dir: str | Path = "dist",
        workflow_name: str = "OcularAutomatedWorkflow",
        telemetry_path: Optional[str | Path] = None,
        telemetry_events: Optional[List[Dict[str, Any]]] = None,
        target_url: Optional[str] = None,
    ) -> CompiledArtifacts:
        """Compiles an MP4 video recording into executable Playwright + FastAPI artifacts."""
        output_dir = Path(output_dir)
        video_file, resolved_telem = discover_recording_assets(video_path, telemetry_path)
        video_path = video_file

        # Ingest telemetry if specified or co-located with video file
        loaded_events: List[Dict[str, Any]] = list(telemetry_events or [])
        if not loaded_events and resolved_telem and resolved_telem.exists():
            loaded_events = OcularSessionRecorder.load_telemetry_events(resolved_telem)

        if loaded_events:
            self._notify(
                "telemetry_ingested",
                {
                    "event_count": len(loaded_events),
                    "clicks_count": sum(
                        1
                        for e in loaded_events
                        if e.get("event_type") == "mousedown" or e.get("action_type") == "click"
                    ),
                },
            )

        # Stage 1: Ingest
        self._notify("ingest_start", {"source": str(video_path)})
        frames_meta, transitions, raw_frames = self.ingest.process_video(
            video_path, telemetry_events=loaded_events
        )
        self._notify(
            "ingest_complete",
            {
                "frames_count": len(frames_meta),
                "keyframes_count": sum(1 for f in frames_meta if f.is_keyframe),
                "transitions_count": len(transitions),
            },
        )

        # Stage 2: SAM 3.1 Segmentation (Mocked / Pluggable)
        self._notify("segmentation_start", {"target_frames": len(raw_frames)})
        segments: Dict[int, List[VisualSegment]] = self.segmenter.segment_batch(
            raw_frames, frames_meta, telemetry_events=loaded_events
        )
        total_segments = sum(len(segs) for segs in segments.values())
        self._notify(
            "segmentation_complete",
            {"total_segments": total_segments, "frames_analyzed": len(segments)},
        )

        # Stage 3: Muse DAG State Synthesis
        meta_file = video_path.parent / "session_metadata.json"
        origin_url = target_url
        if not origin_url and meta_file.exists():
            try:
                import json
                meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                origin_url = meta_data.get("url") or meta_data.get("target_url")
                if not workflow_name:
                    workflow_name = origin_url or meta_data.get("workflow_name")
            except Exception:
                pass

        if not workflow_name or workflow_name in ("OcularAutomatedWorkflow", "TradingViewChartWorkflow"):
            if origin_url:
                workflow_name = origin_url

        workflow_name = derive_workflow_name_from_url(workflow_name)

        self._notify("synthesis_start", {"workflow_name": workflow_name})
        state_graph: StateGraph = self.synthesizer.synthesize(
            frames=frames_meta,
            segments=segments,
            transitions=transitions,
            workflow_name=workflow_name,
            telemetry_events=loaded_events,
        )
        if origin_url:
            state_graph.target_url = origin_url
        self._notify(
            "synthesis_complete",
            {
                "states": len(state_graph.nodes),
                "edges": len(state_graph.edges),
                "initial_state": state_graph.initial_node_id,
            },
        )

        # Stage 4: Code Generation (Playwright + FastAPI)
        self._notify("codegen_start", {"dist_dir": str(output_dir)})
        artifacts: CompiledArtifacts = self.generator.generate(
            graph=state_graph, dist_dir=output_dir, telemetry_events=loaded_events
        )
        self._notify(
            "codegen_complete",
            {
                "artifacts": artifacts.model_dump(),
            },
        )

        return artifacts

    def compile_frames(
        self,
        frames: List[np.ndarray],
        output_dir: str | Path = "dist",
        workflow_name: str = "OcularAutomatedWorkflow",
        timestamps_ms: Optional[List[float]] = None,
        telemetry_path: Optional[str | Path] = None,
        telemetry_events: Optional[List[Dict[str, Any]]] = None,
        target_url: Optional[str] = "https://example.com/app",
    ) -> CompiledArtifacts:
        """Compiles in-memory frame sequences (useful for headless / test suites)."""
        output_dir = Path(output_dir)

        loaded_events: List[Dict[str, Any]] = list(telemetry_events or [])
        if not loaded_events and telemetry_path:
            telem_p = Path(telemetry_path)
            if telem_p.exists():
                loaded_events = OcularSessionRecorder.load_telemetry_events(telem_p)

        if loaded_events:
            self._notify(
                "telemetry_ingested",
                {
                    "event_count": len(loaded_events),
                    "clicks_count": sum(
                        1
                        for e in loaded_events
                        if e.get("event_type") == "mousedown" or e.get("action_type") == "click"
                    ),
                },
            )

        self._notify("ingest_start", {"source": "in_memory_frames"})
        frames_meta, transitions, raw_frames = self.ingest.process_frames_list(
            frames, timestamps_ms, telemetry_events=loaded_events
        )
        self._notify(
            "ingest_complete",
            {
                "frames_count": len(frames_meta),
                "keyframes_count": sum(1 for f in frames_meta if f.is_keyframe),
                "transitions_count": len(transitions),
            },
        )

        self._notify("segmentation_start", {"target_frames": len(raw_frames)})
        segments = self.segmenter.segment_batch(
            raw_frames, frames_meta, telemetry_events=loaded_events
        )
        total_segments = sum(len(segs) for segs in segments.values())
        self._notify(
            "segmentation_complete",
            {"total_segments": total_segments, "frames_analyzed": len(segments)},
        )

        self._notify("synthesis_start", {"workflow_name": workflow_name})
        state_graph = self.synthesizer.synthesize(
            frames=frames_meta,
            segments=segments,
            transitions=transitions,
            workflow_name=workflow_name,
            telemetry_events=loaded_events,
        )
        if target_url:
            state_graph.target_url = target_url
        self._notify(
            "synthesis_complete",
            {
                "states": len(state_graph.nodes),
                "edges": len(state_graph.edges),
            },
        )

        self._notify("codegen_start", {"dist_dir": str(output_dir)})
        artifacts = self.generator.generate(graph=state_graph, dist_dir=output_dir)
        self._notify("codegen_complete", {"artifacts": artifacts.model_dump()})

        return artifacts
