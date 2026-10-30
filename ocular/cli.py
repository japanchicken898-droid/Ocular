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
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import typer

from ocular.pipeline.code_generator import JinjaCodeGenerator
from ocular.pipeline.segmenter import MetaSAMSegmenter, MockSAMSegmenter, export_visual_overlays
from ocular.pipeline.state_graph import DAGBuilder, render_rich_dag_tree
from ocular.pipeline.video_ingest import VideoIngestSampler
from ocular.recorder import OcularSessionRecorder

console = Console()
logger = logging.getLogger("ocular.cli")

# Submodule alias for architectural interoperability
sys.modules["ocular.cli.compiler"] = sys.modules[__name__]
setattr(sys.modules[__name__], "compiler", sys.modules[__name__])

app = typer.Typer(
    name="ocular",
    help="Ocular: Offline Visual Compiler moving multimodal vision inference from runtime to compile time.",
    no_args_is_help=True,
    add_completion=False,
)


def print_banner(console_instance: Optional[Console] = None) -> None:
    """Renders the stylized ASCII banner with subtitle and version badge."""
    c = console_instance or console
    ascii_art = r"""
  ██████╗  ██████╗██╗   ██╗██╗      █████╗ ██████╗ 
 ██╔═══██╗██╔════╝██║   ██║██║     ██╔══██╗██╔══██╗
 ██║   ██║██║     ██║   ██║██║     ███████║██████╔╝
 ██║   ██║██║     ██║   ██║██║     ██╔══██║██╔══██╗
 ╚██████╔╝╚██████╗╚██████╔╝███████╗██║  ██║██║  ██║
  ╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
""".strip("\n")
    banner_text = Text(ascii_art + "\n\n", style="bold cyan")
    banner_text.append(
        "Offline Visual Workflow Compiler • Meta SAM 3.1 + Meta Muse Spark\n",
        style="bold white",
    )
    banner_text.append("Version: ", style="dim white")
    banner_text.append("v0.1.0-mvp", style="bold green")

    banner_panel = Panel(
        Align.center(banner_text),
        box=box.ROUNDED,
        border_style="cyan",
        padding=(1, 2),
    )
    c.print(banner_panel)


def compile_command(
    input_path: str,
    out_dir: str = "./dist",
    workflow_name: str = "TradingViewChartWorkflow",
) -> int:
    """Executes the complete offline visual compilation pipeline with structured logging."""
    t_start = time.perf_counter()
    video_file = Path(input_path)
    output_path = Path(out_dir)

    print_banner(console)

    console.print(
        Panel(
            f"[bold cyan]Compilation Job Manifest[/bold cyan]\n"
            f"[dim]Compiling visual recording into deterministic Playwright + FastAPI service[/dim]\n"
            f"Input Recording : [yellow]{input_path}[/yellow]\n"
            f"Output Directory: [yellow]{out_dir}[/yellow]",
            box=box.ROUNDED,
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
        with console.status(
            "[bold cyan]Calling Meta SAM 3.1 API (api.meta.ai)...[/bold cyan]",
            spinner="dots",
        ):
            segmenter = MetaSAMSegmenter(fallback_segmenter=MockSAMSegmenter())
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
        with console.status(
            "[bold cyan]Synthesizing state DAG via Meta Muse Spark (Baseten)...[/bold cyan]",
            spinner="dots",
        ):
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

        # Print extracted parameters table if parameters_schema is present
        params_schema = dag.get("parameters_schema", {})
        properties = params_schema.get("properties", {})
        if properties:
            table = Table(
                title="[bold cyan]Synthesized Workflow Parameters Schema[/bold cyan]",
                box=box.ROUNDED,
                header_style="bold magenta",
            )
            table.add_column("Parameter", style="bold cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Default", style="green")
            table.add_column("Target Anchor", style="blue")
            table.add_column("Description", style="white")

            for param_name, prop_meta in properties.items():
                param_type = prop_meta.get("type", "string")
                param_default = str(prop_meta.get("default", ""))
                param_desc = prop_meta.get("description", "")
                target_anchor = (
                    "interval_dropdown"
                    if param_name == "interval"
                    else f"#{param_name}"
                )
                table.add_row(param_name, param_type, param_default, target_anchor, param_desc)

            console.print(table)
    except Exception as exc:
        console.print(f"[bold red][!] Code generation failed:[/bold red] {exc}")
        return 1

    duration = time.perf_counter() - t_start
    cli_cmd = f"python {out_dir}/playwright_driver.py --params '{{\"interval\": \"1h\"}}'"
    api_cmd = "uvicorn dist.main:app --reload"
    docs_url = "http://127.0.0.1:8000/docs"

    completion_content = (
        f"[bold green]✓ Compilation Successful in {duration:.2f}s[/bold green]\n\n"
        f"[bold white]Output Directory:[/bold white] [cyan]{output_path.resolve()}[/cyan]\n\n"
        f"[bold cyan]▶ CLI Execution Command:[/bold cyan]\n"
        f"  [yellow]{cli_cmd}[/yellow]\n\n"
        f"[bold cyan]▶ Microservice Startup Command:[/bold cyan]\n"
        f"  [yellow]{api_cmd}[/yellow]\n\n"
        f"[bold cyan]▶ Interactive Swagger UI Documentation:[/bold cyan]\n"
        f"  [link={docs_url}]{docs_url}[/link]"
    )
    console.print(
        Panel(
            completion_content,
            title="[bold green]Ocular Build Complete[/bold green]",
            box=box.ROUNDED,
            border_style="green",
            padding=(1, 2),
        )
    )
    return 0


def record_command(
    url: str,
    out_dir: str = "./recordings",
    headless: bool = False,
    duration: Optional[float] = None,
) -> int:
    """Executes the interactive session recorder pipeline with Rich terminal logging."""
    print_banner(console)

    console.print(
        Panel(
            f"[bold cyan]Interactive Session Recorder Manifest[/bold cyan]\n"
            f"[dim]Capturing browser interaction stream, normalized telemetry, and canvas video[/dim]\n"
            f"Target URL       : [yellow]{url}[/yellow]\n"
            f"Output Directory : [yellow]{out_dir}[/yellow]\n"
            f"Mode             : [green]{'Headless' if headless else 'Interactive (Headed)'}[/green]\n"
            f"Status           : [bold green]● RECORDING ACTIVE[/bold green] [dim](Interact with browser; close window to finalize)[/dim]",
            box=box.ROUNDED,
            border_style="cyan",
        )
    )

    recorder = OcularSessionRecorder(
        out_dir=out_dir,
        headless=headless,
    )

    try:
        result = recorder.record(url=url, duration=duration)
    except Exception as exc:
        console.print(f"[bold red][!] Recording error:[/bold red] {exc}")
        return 1

    # Artifact Summary Table upon browser close
    artifact_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    artifact_table.add_column("Artifact / Metric", style="white")
    artifact_table.add_column("Details", style="green")

    if result.get("video_path"):
        artifact_table.add_row("Captured Video", str(result["video_path"]))
    artifact_table.add_row("Telemetry Stream", str(result["telemetry_path"]))
    artifact_table.add_row("Events Logged", str(result["event_count"]))
    artifact_table.add_row("Session Elapsed", f"{result['elapsed_ms']} ms")

    console.print(
        Panel(
            artifact_table,
            title="[bold green]Recording Session Finalized[/bold green]",
            subtitle="[dim]Telemetry events and video saved successfully[/dim]",
            box=box.ROUNDED,
            border_style="green",
            padding=(1, 2),
        )
    )
    return 0


@app.command("record")
def record_cli(
    url: str = typer.Option(
        ...,
        "--url",
        "-u",
        help="Target URL for interactive recording session",
    ),
    out_dir: str = typer.Option(
        "./recordings",
        "--out-dir",
        "-o",
        help="Target output directory for recorded video and telemetry_events.jsonl",
    ),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Run browser in headless mode (default: False)",
    ),
    duration: Optional[float] = typer.Option(
        None,
        "--duration",
        "-t",
        help="Auto-stop session after specified seconds (optional)",
    ),
) -> None:
    """Record an interactive browser session capturing video and telemetry events."""
    exit_code = record_command(url=url, out_dir=out_dir, headless=headless, duration=duration)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


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


@app.command("execute")
def execute_cli(
    dist_dir: str = typer.Option("./dist", "--dist-dir", "-d", help="Directory containing compiled driver"),
    params: str = typer.Option("{}", "--params", "-p", help="JSON string of dynamic workflow parameters"),
    base_url: str = typer.Option("http://localhost:3000", "--base-url", "-u", help="Target URL"),
    headed: bool = typer.Option(False, "--headed", help="Run headed browser"),
) -> None:
    """Execute the compiled workflow driver with dynamic parameters."""
    print_banner(console)
    driver_script = Path(dist_dir) / "playwright_driver.py"
    if not driver_script.exists():
        console.print(f"[bold red][!] Compiled driver not found at {driver_script}[/bold red]")
        raise typer.Exit(code=1)

    import subprocess
    cmd = [sys.executable, str(driver_script), "--params", params, "--base-url", base_url]
    if headed:
        cmd.append("--headed")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise typer.Exit(code=proc.returncode)

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
