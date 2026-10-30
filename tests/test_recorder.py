"""Unit and integration tests for OcularSessionRecorder and telemetry probe."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pytest
import numpy as np
from playwright.async_api import async_playwright

from ocular.cli import record_command, main as cli_main
from ocular.compiler import OcularCompiler
from ocular.models.types import ActionType
from ocular.pipeline.segmenter import MockSAMSegmenter
from ocular.pipeline.state_graph import DAGBuilder
from ocular.recorder import OcularSessionRecorder


def test_recorder_initialization(tmp_path: Path):
    """Verifies default and custom initialization attributes of OcularSessionRecorder."""
    rec = OcularSessionRecorder(out_dir=tmp_path / "rec_test")
    assert rec.out_dir == tmp_path / "rec_test"
    assert rec.viewport_width == 1280
    assert rec.viewport_height == 720
    assert rec.headless is False
    assert rec.telemetry_path == tmp_path / "rec_test" / "telemetry_events.jsonl"
    assert len(rec.telemetry_events) == 0

    custom_rec = OcularSessionRecorder(
        out_dir=tmp_path / "custom",
        viewport_width=1920,
        viewport_height=1080,
        headless=True,
    )
    assert custom_rec.viewport_width == 1920
    assert custom_rec.viewport_height == 1080
    assert custom_rec.headless is True


def test_mock_telemetry_ingestion(tmp_path: Path):
    """Tests manual event pushing, callback propagation, and JSONL persistence."""
    captured_callback_events = []
    rec = OcularSessionRecorder(
        out_dir=tmp_path / "telemetry_test",
        on_event_callback=lambda ev: captured_callback_events.append(ev),
    )

    ev1 = {
        "event_type": "mousedown",
        "action_type": "click",
        "norm_x": 0.245,
        "norm_y": 0.582,
        "elapsed_ms": 120,
        "target_selector": "#submit_btn",
    }
    ev2 = {
        "event_type": "keydown",
        "action_type": "fill",
        "key": "BTC",
        "norm_x": 0.5,
        "norm_y": 0.2,
        "elapsed_ms": 250,
        "target_selector": "#search_input",
    }

    rec.push_telemetry_event(ev1)
    rec.push_telemetry_event(ev2)

    assert len(rec.telemetry_events) == 2
    assert len(captured_callback_events) == 2
    assert rec.telemetry_path.exists()

    loaded = OcularSessionRecorder.load_telemetry_events(rec.telemetry_path)
    assert len(loaded) == 2
    assert loaded[0]["norm_x"] == 0.245
    assert loaded[0]["action_type"] == "click"
    assert loaded[1]["key"] == "BTC"


@pytest.mark.asyncio
async def test_session_recording_live_probe(tmp_path: Path):
    """End-to-end integration test verifying video capture and client-side telemetry probe."""
    rec_dir = tmp_path / "live_session"
    recorder = OcularSessionRecorder(
        out_dir=rec_dir,
        viewport_width=800,
        viewport_height=600,
        headless=True,
    )

    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <button id="test_button" style="position: absolute; left: 100px; top: 100px; width: 120px; height: 40px;">
            Target Action
        </button>
        <input id="test_input" style="position: absolute; left: 100px; top: 200px; width: 120px; height: 40px;" />
    </body>
    </html>
    """

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": recorder.viewport_width, "height": recorder.viewport_height},
            record_video_dir=str(recorder.out_dir),
            record_video_size={"width": recorder.viewport_width, "height": recorder.viewport_height},
        )
        await context.expose_binding("__ocular_push_telemetry", recorder._handle_telemetry_binding)
        await context.add_init_script(recorder.TELEMETRY_INIT_SCRIPT)

        page = await context.new_page()
        await page.goto(f"data:text/html,{html_content}")

        # Trigger mousedown
        await page.click("#test_button")
        await page.wait_for_timeout(150)

        # Trigger keydown
        await page.focus("#test_input")
        await page.keyboard.press("x")
        await page.wait_for_timeout(200)

        video = page.video
        await context.close()
        video_path = await video.path() if video else None
        await browser.close()

    assert recorder.telemetry_path.exists()
    events = OcularSessionRecorder.load_telemetry_events(recorder.telemetry_path)
    assert len(events) >= 2

    click_ev = next((e for e in events if e.get("event_type") == "mousedown"), None)
    assert click_ev is not None
    assert "norm_x" in click_ev and "norm_y" in click_ev and "elapsed_ms" in click_ev
    assert click_ev["norm_x"] > 0.0 and click_ev["norm_y"] > 0.0
    assert click_ev["target_selector"] == "#test_button"

    key_ev = next((e for e in events if e.get("event_type") == "keydown"), None)
    assert key_ev is not None
    assert key_ev["key"] == "x"
    assert key_ev["target_selector"] == "#test_input"

    if video_path:
        assert Path(video_path).exists()


def test_cli_record_command_execution(tmp_path: Path):
    """Verifies CLI execution of record command and artifact output."""
    out_dir = tmp_path / "cli_rec_out"
    code = record_command(
        url="data:text/html,<html><body><h1>Ocular Recording CLI</h1></body></html>",
        out_dir=str(out_dir),
        headless=True,
        duration=0.2,
    )
    assert code == 0
    assert (out_dir / "telemetry_events.jsonl").exists()

    cli_exit = cli_main([
        "record",
        "--url",
        "data:text/html,<html><body><h1>CLI Main</h1></body></html>",
        "--out-dir",
        str(out_dir),
        "--headless",
        "--duration",
        "0.2",
    ])
    assert cli_exit == 0


def test_compiler_ingestion_of_recorded_telemetry(tmp_path: Path):
    """Verifies that compiler ingest pipeline incorporates recorded telemetry as priors."""
    rec_dir = tmp_path / "telemetry_ingest_test"
    rec_dir.mkdir(parents=True)
    telem_file = rec_dir / "telemetry_events.jsonl"

    events = [
        {
            "event_type": "mousedown",
            "action_type": "click",
            "norm_x": 0.42,
            "norm_y": 0.68,
            "elapsed_ms": 300,
            "target_selector": "#custom_point_anchor",
        }
    ]
    with open(telem_file, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    # Verify SAM segmenter extracts the point prompt anchor
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    segmenter = MockSAMSegmenter()
    segs = segmenter.segment(frame, point_prompts=events)
    prior_anchors = [s for s in segs if s.get("metadata", {}).get("point_prompt_prior")]
    assert len(prior_anchors) >= 1
    assert prior_anchors[0]["id"] == "custom_point_anchor"

    # Verify DAG builder transitions connect through telemetry anchor
    builder = DAGBuilder(workflow_name="TelemetryTestWorkflow")
    dag = builder.build_dag(segs, telemetry_events=events)
    assert len(dag["edges"]) >= 1
    edge = dag["edges"][0]
    assert edge["action_type"] == "click"

    # Verify OcularCompiler end-to-end ingestion
    compiler = OcularCompiler()
    dist_dir = tmp_path / "dist"
    artifacts = compiler.compile_frames(
        frames=[frame, frame],
        output_dir=dist_dir,
        telemetry_events=events,
    )
    assert artifacts is not None
    assert (dist_dir / "playwright_driver.py").exists()
