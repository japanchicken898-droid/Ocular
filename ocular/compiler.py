"""Ocular Compiler Orchestrator.

Moves multimodal vision inference from runtime to compile time by executing
the modular stages: Ingest -> SAM 3.1 Segmentation -> Muse DAG Synthesis -> Code Generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
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
    ) -> CompiledArtifacts:
        """Compiles an MP4 video recording into executable Playwright + FastAPI artifacts."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        # Ingest telemetry if specified or co-located with video file
        loaded_events: List[Dict[str, Any]] = list(telemetry_events or [])
        if not loaded_events:
            effective_telem_path = (
                Path(telemetry_path)
                if telemetry_path
                else video_path.parent / "telemetry_events.jsonl"
            )
            if effective_telem_path.exists():
                loaded_events = OcularSessionRecorder.load_telemetry_events(effective_telem_path)

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
        if not workflow_name or workflow_name in ("OcularAutomatedWorkflow", "TradingViewChartWorkflow"):
            meta_file = video_path.parent / "session_metadata.json"
            if meta_file.exists():
                try:
                    import json
                    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                    workflow_name = meta_data.get("url") or meta_data.get("workflow_name") or workflow_name
                except Exception:
                    pass
            elif "excalidraw" in video_path.parent.name.lower():
                workflow_name = "https://excalidraw.com"
            elif video_path.parent.name.startswith("session_"):
                workflow_name = video_path.parent.name

        workflow_name = derive_workflow_name_from_url(workflow_name)

        self._notify("synthesis_start", {"workflow_name": workflow_name})
        state_graph: StateGraph = self.synthesizer.synthesize(
            frames=frames_meta,
            segments=segments,
            transitions=transitions,
            workflow_name=workflow_name,
            telemetry_events=loaded_events,
        )
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
