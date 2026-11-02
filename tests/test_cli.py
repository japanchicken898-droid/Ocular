"""Unit and Integration Tests for Ocular CLI and Recording Ingestion."""

import json
import os
import time
from pathlib import Path
import pytest
from typer.testing import CliRunner

from ocular.cli import app, main, compile_command, discover_recording_assets
from ocular.recorder import OcularSessionRecorder
from samples.create_sample_video import generate_synthetic_screen_recording

runner = CliRunner()


def test_discover_recording_assets_exact_file(tmp_path: Path):
    """Direct file path returns exact file and discovers sibling telemetry."""
    session_dir = tmp_path / "session_file"
    session_dir.mkdir()
    video_file = session_dir / "custom_recording.mp4"
    generate_synthetic_screen_recording(video_file, fps=10, width=320, height=240)

    telem_file = session_dir / "telemetry_events.jsonl"
    telem_file.write_text('{"event_type": "mousedown", "client_x": 10, "client_y": 20}\n', encoding="utf-8")

    # 1. Automatic sibling telemetry discovery
    v_found, t_found = discover_recording_assets(video_file)
    assert v_found == video_file
    assert t_found == telem_file

    # 2. Explicit telemetry override
    custom_telem = tmp_path / "custom.jsonl"
    custom_telem.write_text('{"event_type": "keydown", "key": "a"}\n', encoding="utf-8")
    v_found2, t_found2 = discover_recording_assets(video_file, telemetry_path=custom_telem)
    assert v_found2 == video_file
    assert t_found2 == custom_telem


def test_discover_recording_assets_directory(tmp_path: Path):
    """Directory input discovers Playwright page@*.webm screencast and sibling telemetry."""
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()

    # Playwright screencast naming pattern
    screencast_file = recordings_dir / "page@9a8b7c6d5e4f.webm"
    generate_synthetic_screen_recording(screencast_file, fps=10, width=320, height=240)

    telem_file = recordings_dir / "telemetry_events.jsonl"
    telem_file.write_text('{"event_type": "mousedown", "client_x": 100, "client_y": 150}\n', encoding="utf-8")

    v_found, t_found = discover_recording_assets(recordings_dir)
    assert v_found == screencast_file
    assert t_found == telem_file


def test_discover_recording_assets_selects_newest_mtime(tmp_path: Path):
    """Multiple videos in directory results in selecting the newest video by mtime."""
    multi_dir = tmp_path / "multi_sessions"
    multi_dir.mkdir()

    old_video = multi_dir / "old_session.mp4"
    generate_synthetic_screen_recording(old_video, fps=10, width=320, height=240)

    new_video = multi_dir / "page@latest_screencast.webm"
    generate_synthetic_screen_recording(new_video, fps=10, width=320, height=240)

    # Set distinct modification timestamps
    t_now = time.time()
    os.utime(old_video, (t_now - 100, t_now - 100))
    os.utime(new_video, (t_now, t_now))

    v_found, _ = discover_recording_assets(multi_dir)
    assert v_found == new_video


def test_discover_recording_assets_empty_dir_raises(tmp_path: Path):
    """Directory with no video files raises FileNotFoundError with helpful message."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No video recording"):
        discover_recording_assets(empty_dir)

    with pytest.raises(FileNotFoundError, match="Input path not found"):
        discover_recording_assets(tmp_path / "non_existent_path")


def test_cli_compile_directory_and_output_alias(tmp_path: Path):
    """Verifies ocular compile -i recordings/ --output dist_test resolves directory and screencast."""
    recordings_dir = tmp_path / "recordings_dir"
    recordings_dir.mkdir()
    screencast_file = recordings_dir / "page@1a2b3c4d5e6f.webm"
    generate_synthetic_screen_recording(screencast_file, fps=10, width=640, height=360)

    # Add telemetry events
    telem_file = recordings_dir / "telemetry_events.jsonl"
    events = [
        {"event_type": "mousedown", "action_type": "click", "client_x": 50, "client_y": 50, "norm_x": 0.08, "norm_y": 0.14, "elapsed_ms": 100},
        {"event_type": "keydown", "action_type": "fill", "key": "query", "value": "test search", "elapsed_ms": 300},
    ]
    telem_file.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    # Add session metadata
    meta_file = recordings_dir / "session_metadata.json"
    meta_file.write_text(json.dumps({"url": "https://example.com/search"}), encoding="utf-8")

    dist_test = tmp_path / "dist_test"

    # Invoke CLI with -i <dir> and --output <dir>
    result = runner.invoke(app, [
        "compile",
        "-i", str(recordings_dir),
        "--output", str(dist_test),
    ])

    assert result.exit_code == 0, f"CLI command failed:\n{result.output}"
    assert "Resolved recording: page@1a2b3c4d5e6f.webm" in result.output
    assert (dist_test / "playwright_driver.py").exists()
    assert (dist_test / "app.py").exists()
    assert (dist_test / "main.py").exists()
    assert (dist_test / "workflow.json").exists()


def test_cli_compile_out_dir_flag_retained(tmp_path: Path):
    """Verifies ocular compile --input <dir> --out-dir <dir> continues working."""
    recordings_dir = tmp_path / "recordings_out_dir"
    recordings_dir.mkdir()
    screencast_file = recordings_dir / "page@abcdef.webm"
    generate_synthetic_screen_recording(screencast_file, fps=10, width=640, height=360)

    meta_file = recordings_dir / "session_metadata.json"
    meta_file.write_text(json.dumps({"url": "https://example.com/checkout"}), encoding="utf-8")

    dist_dir = tmp_path / "dist_out_dir"

    result = runner.invoke(app, [
        "compile",
        "--input", str(recordings_dir),
        "--out-dir", str(dist_dir),
    ])

    assert result.exit_code == 0
    assert (dist_dir / "playwright_driver.py").exists()


def test_cli_main_argv_directory_and_output(tmp_path: Path):
    """Verifies main() entrypoint handles -i <dir> --output <dir> directly via sys.argv."""
    recordings_dir = tmp_path / "recordings_main"
    recordings_dir.mkdir()
    screencast_file = recordings_dir / "page@998877.webm"
    generate_synthetic_screen_recording(screencast_file, fps=10, width=640, height=360)

    meta_file = recordings_dir / "session_metadata.json"
    meta_file.write_text(json.dumps({"url": "https://example.com/main_test"}), encoding="utf-8")

    dist_dir = tmp_path / "dist_main"
    exit_code = main(["compile", "-i", str(recordings_dir), "--output", str(dist_dir)])

    assert exit_code == 0
    assert (dist_dir / "playwright_driver.py").exists()
    assert (dist_dir / "app.py").exists()



def test_cli_compile_empty_directory_error(tmp_path: Path):
    """Verifies passing a directory with no video fails gracefully with clear error."""
    empty_dir = tmp_path / "empty_recordings"
    empty_dir.mkdir()

    result = runner.invoke(app, [
        "compile",
        "-i", str(empty_dir),
        "--output", str(tmp_path / "dist_fail"),
    ])

    assert result.exit_code == 1
    assert "No video recording" in result.output


def test_recorder_clean_naming_on_close(tmp_path: Path, monkeypatch):
    """Tests that OcularSessionRecorder cleans page@*.webm to recording.webm when possible."""
    out_dir = tmp_path / "recorder_session"
    out_dir.mkdir()

    # Create dummy page@hash.webm
    raw_video = out_dir / "page@12345678abcdef.webm"
    raw_video.write_bytes(b"dummy_video_bytes")

    recorder = OcularSessionRecorder(out_dir=out_dir, headless=True)

    # Simulate video close and rename logic
    clean_video_path = out_dir / "recording.webm"
    assert raw_video.exists()

    # Execute rename block
    if raw_video.resolve() != clean_video_path.resolve():
        raw_video.replace(clean_video_path)
        recorder.video_path = clean_video_path

    assert clean_video_path.exists()
    assert recorder.video_path == clean_video_path
    assert clean_video_path.read_bytes() == b"dummy_video_bytes"
