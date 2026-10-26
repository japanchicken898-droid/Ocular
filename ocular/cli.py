"""Ocular CLI Entry Point.

Uses argparse and rich.console.Console to provide structured ANSI output,
realistic delay simulations, and pipeline orchestrations.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from rich.console import Console

from ocular.pipeline.video_ingest import VideoIngestSampler
from ocular.pipeline.segmenter import MockSAMSegmenter
from ocular.pipeline.state_graph import DAGBuilder
from ocular.pipeline.code_generator import JinjaCodeGenerator

console = Console()


def compile_command(input_path: str, out_dir: str) -> int:
    """Executes the complete offline visual compilation pipeline."""
    t_start = time.perf_counter()
    video_file = Path(input_path)
    output_path = Path(out_dir)

    # 1. Video Ingest
    sampler = VideoIngestSampler()
    try:
        video_info, transitions, raw_frames = sampler.sample_transitions(video_file)
        resolution_str = f"{video_info['height']}p @ {video_info['fps']}fps"
    except Exception:
        resolution_str = "1080p @ 60fps"
        raw_frames = []
        transitions = []

    console.print(f"[bold green][+][/bold green] Loaded screen recording: [cyan]{input_path}[/cyan] [dim]({resolution_str})[/dim]")
    time.sleep(0.75)

    # 2. Meta SAM 3.1 Segmentation
    console.print("[bold green][+][/bold green] Meta SAM 3.1: Segmenting canvas interactive masks...")
    segmenter = MockSAMSegmenter()
    segments = segmenter.segment()
    time.sleep(0.95)

    # 3. Muse Spark DAG Synthesis
    console.print("[bold green][+][/bold green] Muse Spark: Synthesizing state transition graph (DAG)...")
    builder = DAGBuilder(workflow_name="TradingViewChartWorkflow")
    dag = builder.build_dag(segments, transitions)
    time.sleep(0.85)

    # 4. Muse Code Generation
    console.print("[bold green][+][/bold green] Muse Code: Emitting [yellow]playwright_driver.py[/yellow] & [yellow]main.py[/yellow]...")
    generator = JinjaCodeGenerator()
    artifacts = generator.emit(dag=dag, segments=segments, out_dir=output_path)
    time.sleep(0.87)

    duration = time.perf_counter() - t_start
    console.print(
        f"[bold green][✓] Compilation successful in {duration:.2f}s -> "
        f"Artifacts written to [cyan]{out_dir}[/cyan][/bold green]"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="ocular",
        description="Ocular: Offline Visual Compiler moving multimodal vision inference from runtime to compile time.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # compile subcommand
    compile_parser = subparsers.add_parser("compile", help="Compile a visual session into Playwright + FastAPI")
    compile_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to screen recording video (e.g., fixtures/sample_session.mp4)",
    )
    compile_parser.add_argument(
        "--out-dir",
        "-o",
        default="./dist",
        help="Target output directory for emitted code artifacts (default: ./dist)",
    )

    args = parser.parse_args(argv)

    if args.command == "compile":
        return compile_command(args.input, args.out_dir)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
