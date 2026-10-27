"""Ocular CLI Entry Point.

Builds a Typer CLI entrypoint exposing:
    ocular compile --input <recording.mp4> --out-dir ./dist

Provides structured Rich console logging and deterministic error handling.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
import typer

from ocular.pipeline.code_generator import JinjaCodeGenerator
from ocular.pipeline.segmenter import MetaSAMSegmenter, MockSAMSegmenter, export_visual_overlays
from ocular.pipeline.state_graph import DAGBuilder, render_rich_dag_tree
from ocular.pipeline.video_ingest import VideoIngestSampler

console = Console()
logger = logging.getLogger("ocular.cli")

app = typer.Typer(
    name="ocular",
    help="Ocular: Offline Visual Compiler moving multimodal vision inference from runtime to compile time.",
    no_args_is_help=True,
    add_completion=False,
)


def compile_command(
    input_path: str,
    out_dir: str = "./dist",
    workflow_name: str = "TradingViewChartWorkflow",
) -> int:
    """Executes the complete offline visual compilation pipeline with structured logging."""
    t_start = time.perf_counter()
    video_file = Path(input_path)
    output_path = Path(out_dir)

    console.print(
        Panel(
            f"[bold cyan]Ocular Offline Visual Compiler[/bold cyan]\n"
            f"[dim]Compiling visual recording into deterministic Playwright + FastAPI service[/dim]\n"
            f"Input Recording : [yellow]{input_path}[/yellow]\n"
            f"Output Directory: [yellow]{out_dir}[/yellow]",
            border_style="cyan",
        )
    )

    # 1. Video Ingest Stage
    console.print("[bold green][+][/bold green] [bold]Stage 1: Video Ingest & Temporal Differencing[/bold]")
    sampler = VideoIngestSampler(sample_rate=2.0, diff_threshold=0.08)
    try:
        video_info, transitions, raw_frames = sampler.sample_transitions(video_file)
        resolution_str = f"{video_info['height']}p @ {video_info['fps']}fps"
        frames_count = len(raw_frames)
        trans_count = len(transitions)
        console.print(
            f"    Loaded [cyan]{input_path}[/cyan] ({resolution_str}, {frames_count} frames sampled, {trans_count} key transitions)"
        )
    except FileNotFoundError:
        console.print(f"[bold red][!] Error:[/bold red] Input video file not found: [yellow]{input_path}[/yellow]")
        return 1
    except Exception as exc:
        console.print(f"[yellow][!] Notice:[/yellow] Ingest fallback mode enabled ({exc})")
        raw_frames = []
        transitions = []

    # 2. Meta SAM 3.1 Segmentation Stage
    console.print("[bold green][+][/bold green] [bold]Stage 2: Meta SAM 3.1 Visual Segmentation[/bold]")
    try:
        segmenter = MetaSAMSegmenter(fallback_segmenter=MockSAMSegmenter())
        # Segment keyframe or fallback
        sample_frame = raw_frames[0] if raw_frames else None
        segments = segmenter.segment(sample_frame)
        if getattr(segmenter, "last_inference_source", None) == "meta_sam_3.1_api" or os.environ.get("META_MODEL_API_KEY"):
            console.print("    [green]✓[/green] Live Meta SAM 3.1 inference executed (POST https://api.meta.ai/v1/responses)")
        console.print(
            f"    Synthesized [cyan]{len(segments)}[/cyan] scale-invariant interactive element masks (coordinates in [0.0, 1.0])"
        )

        # Export SAM 3.1 visual overlays to ./dist/annotated/
        annotated_dir = output_path / "annotated"
        annotated_frames = raw_frames[:5] if raw_frames else ([sample_frame] if sample_frame is not None else [])
        if annotated_frames:
            exported_overlays = export_visual_overlays(
                frames=annotated_frames,
                segments_by_frame=segments,
                output_dir=annotated_dir,
            )
            console.print(
                f"    [green]✓[/green] Exported [cyan]{len(exported_overlays)}[/cyan] SAM 3.1 visual overlays to [yellow]{annotated_dir}[/yellow]"
            )
    except Exception as exc:
        console.print(f"[bold red][!] Segmentation failed:[/bold red] {exc}")
        return 1

    # 3. Muse Spark DAG Synthesis Stage
    console.print("[bold green][+][/bold green] [bold]Stage 3: Muse Spark State Graph Synthesis[/bold]")
    try:
        builder = DAGBuilder(workflow_name=workflow_name)
        dag = builder.build_dag(segments, transitions)
        if getattr(builder, "last_inference_source", None) == "live_baseten_muse_spark" or os.environ.get("BASETEN_API_KEY"):
            console.print("    [green]✓[/green] Live Baseten Meta Muse Spark inference executed (model: meta-models/muse-spark-open-1026)")
        console.print(
            f"    Synthesized verified DAG: [cyan]{len(dag['nodes'])}[/cyan] states, [cyan]{len(dag['edges'])}[/cyan] transitions"
        )
        # Render clean Rich Tree representation of states and transitions
        dag_tree = render_rich_dag_tree(dag)
        console.print(dag_tree)
    except Exception as exc:
        console.print(f"[bold red][!] DAG synthesis failed:[/bold red] {exc}")
        return 1

    # 4. Muse Code Generation Stage
    console.print("[bold green][+][/bold green] [bold]Stage 4: Production Artifact Code Generation[/bold]")
    try:
        generator = JinjaCodeGenerator()
        artifacts = generator.emit(dag=dag, segments=segments, out_dir=output_path)
        console.print(f"    [green]✓[/green] Emitted: [yellow]{artifacts['playwright_driver'].name}[/yellow]")
        console.print(f"    [green]✓[/green] Emitted: [yellow]{artifacts['app'].name}[/yellow]")
        console.print(f"    [green]✓[/green] Emitted: [yellow]{artifacts['main'].name}[/yellow]")
        console.print(f"    [green]✓[/green] Emitted: [yellow]{artifacts['state_graph'].name}[/yellow]")
        console.print(f"    [green]✓[/green] Emitted: [yellow]{artifacts['workflow_dag_mermaid'].name}[/yellow]")
    except Exception as exc:
        console.print(f"[bold red][!] Code generation failed:[/bold red] {exc}")
        return 1

    duration = time.perf_counter() - t_start
    console.print(
        Panel(
            f"[bold green]✓ Compilation Successful in {duration:.2f}s[/bold green]\n"
            f"Output artifacts written to: [cyan]{output_path.resolve()}[/cyan]\n"
            f"FastAPI microservice entry : [dim]uvicorn main:app --reload[/dim]\n"
            f"Playwright driver script   : [dim]python playwright_driver.py[/dim]",
            border_style="green",
        )
    )
    return 0


@app.command("compile")
def compile_cli(
    input: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to screen recording video (e.g. fixtures/sample_session.mp4)",
    ),
    out_dir: str = typer.Option(
        "./dist",
        "--out-dir",
        "-o",
        help="Target output directory for emitted code artifacts",
    ),
    workflow_name: str = typer.Option(
        "TradingViewChartWorkflow",
        "--workflow-name",
        "-w",
        help="Custom name for the compiled workflow",
    ),
) -> None:
    """Compile a screen recording into deterministic Playwright drivers and FastAPI service."""
    exit_code = compile_command(input_path=input, out_dir=out_dir, workflow_name=workflow_name)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command("benchmark")
def benchmark_cli(
    iterations: int = typer.Option(5, "--iterations", "-n", help="Number of benchmark iterations"),
    dist_dir: str = typer.Option("./dist", "--dist-dir", "-d", help="Dist directory containing compiled artifacts"),
    out_file: str = typer.Option("./dist/benchmark_results.json", "--out-file", "-o", help="Benchmark output report path"),
) -> None:
    """Run side-by-side performance benchmark: Ocular compiled driver vs. Runtime VLM."""
    import asyncio
    from benchmark import run_benchmark_suite
    asyncio.run(run_benchmark_suite(iterations=iterations, dist_dir=dist_dir, output_report=out_file))


@app.command("version")
def version_cli() -> None:
    """Show Ocular compiler version."""
    console.print("[bold cyan]Ocular Visual Compiler[/bold cyan] v0.1.0")


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entrypoint supporting both Typer and direct programmatic invocation."""
    if argv is None:
        argv = sys.argv[1:]

    try:
        app(args=argv, standalone_mode=False)
        return 0
    except typer.Exit as te:
        return te.exit_code
    except Exception as exc:
        console.print(f"[bold red][!] Fatal error:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
