"""Tests for JinjaCodeGenerator stage."""

from pathlib import Path
import pytest
from ocular.pipeline.code_generator import JinjaCodeGenerator
from ocular.pipeline.segmenter import MockSAMSegmenter
from ocular.pipeline.state_graph import DAGBuilder
from ocular.pipeline.base import BaseGenerator


def test_jinja_code_generator(tmp_path: Path):
    generator = JinjaCodeGenerator()
    assert isinstance(generator, BaseGenerator)

    segmenter = MockSAMSegmenter()
    segments = segmenter.segment()

    builder = DAGBuilder()
    dag = builder.build_dag(segments)

    out_dir = tmp_path / "dist"
    artifacts = generator.emit(dag=dag, segments=segments, out_dir=out_dir)

    assert "playwright_driver" in artifacts
    assert "app" in artifacts
    assert "main" in artifacts

    driver_file = artifacts["playwright_driver"]
    app_file = artifacts["app"]
    main_file = artifacts["main"]

    assert driver_file.exists()
    assert app_file.exists()
    assert main_file.exists()

    assert "workflow_dag_mermaid" in artifacts
    assert (out_dir / "workflow_dag.mermaid").exists()

    driver_code = driver_file.read_text(encoding="utf-8")
    assert "class OcularPlaywrightDriver:" in driver_code
    assert "def denormalize_target(" in driver_code
    assert "click_with_retry" in driver_code
    assert "execute_workflow" in driver_code
    assert "--headed" in driver_code

    app_code = app_file.read_text(encoding="utf-8")
    assert '@app.post("/api/v1/execute"' in app_code
    assert "ExecuteWorkflowRequest" in app_code
    assert "ExecuteWorkflowResponse" in app_code
    assert '@app.post("/api/v1/chart/interval"' in app_code
    assert "X-Ocular-Compile-Hash" in app_code
