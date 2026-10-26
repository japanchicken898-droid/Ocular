"""End-to-End Compiler Pipeline Tests."""

from pathlib import Path
import pytest
from ocular.cli import compile_command, main
from samples.create_sample_video import generate_synthetic_screen_recording


def test_cli_compile_end_to_end(tmp_path: Path):
    video_path = tmp_path / "test_session.mp4"
    generate_synthetic_screen_recording(video_path, fps=30, width=1280, height=720)

    out_dir = tmp_path / "dist"
    code = compile_command(input_path=str(video_path), out_dir=str(out_dir))

    assert code == 0
    assert (out_dir / "playwright_driver.py").exists()
    assert (out_dir / "app.py").exists()
    assert (out_dir / "main.py").exists()
    assert (out_dir / "state_graph.json").exists()


def test_cli_main_argv_invocation(tmp_path: Path):
    video_path = tmp_path / "test_session2.mp4"
    generate_synthetic_screen_recording(video_path, fps=30, width=640, height=360)

    out_dir = tmp_path / "dist2"
    exit_code = main(["compile", "--input", str(video_path), "--out-dir", str(out_dir)])

    assert exit_code == 0
    assert (out_dir / "app.py").exists()
    assert (out_dir / "playwright_driver.py").exists()
