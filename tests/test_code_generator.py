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
    assert "main" in artifacts

    driver_file = artifacts["playwright_driver"]
    main_file = artifacts["main"]

    assert driver_file.exists()
    assert main_file.exists()

    driver_code = driver_file.read_text(encoding="utf-8")
    assert "class OcularPlaywrightDriver:" in driver_code
    assert "def denormalize_target(" in driver_code

    main_code = main_file.read_text(encoding="utf-8")
    assert "@app.post(\"/api/v1/chart/interval\"" in main_code
    assert "X-Ocular-Compile-Hash" in main_code
