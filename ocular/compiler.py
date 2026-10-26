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
from ocular.pipeline.code_generator import CodeGenerator


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
    ) -> CompiledArtifacts:
        """Compiles an MP4 video recording into executable Playwright + FastAPI artifacts."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        # Stage 1: Ingest
        self._notify("ingest_start", {"source": str(video_path)})
        frames_meta, transitions, raw_frames = self.ingest.process_video(video_path)
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
            raw_frames, frames_meta
        )
        total_segments = sum(len(segs) for segs in segments.values())
        self._notify(
            "segmentation_complete",
            {"total_segments": total_segments, "frames_analyzed": len(segments)},
        )

        # Stage 3: Muse DAG State Synthesis
        self._notify("synthesis_start", {"workflow_name": workflow_name})
        state_graph: StateGraph = self.synthesizer.synthesize(
            frames=frames_meta,
            segments=segments,
            transitions=transitions,
            workflow_name=workflow_name,
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
            graph=state_graph, dist_dir=output_dir
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
    ) -> CompiledArtifacts:
        """Compiles in-memory frame sequences (useful for headless / test suites)."""
        output_dir = Path(output_dir)

        self._notify("ingest_start", {"source": "in_memory_frames"})
        frames_meta, transitions, raw_frames = self.ingest.process_frames_list(
            frames, timestamps_ms
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
        segments = self.segmenter.segment_batch(raw_frames, frames_meta)
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
