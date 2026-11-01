"""Regression tests for Audit Phase 0: Architecture Change."""

from __future__ import annotations

import copy
import importlib
import json
import math
from pathlib import Path
import re
import sys
import pytest

from ocular.pipeline.code_generator import JinjaCodeGenerator
from ocular.pipeline.state_graph import CompilationError


def test_phase0_non_dict_edges_and_segments_raise_compilation_error(tmp_path: Path):
    """Non-dict edges ('click', 5, None) and segments must raise CompilationError."""
    generator = JinjaCodeGenerator()
    out_dir = tmp_path / "dist_non_dict"

    valid_dag = {
        "workflow_name": "ValidWorkflow",
        "root_node": "state_root",
        "terminal_node": "state_root",
        "nodes": {
            "state_root": {
                "id": "state_root",
                "title": "Root",
                "screen_name": "screen_main",
            }
        },
        "edges": ["click"],  # non-dict edge!
    }

    with pytest.raises(CompilationError):
        generator.emit(dag=valid_dag, segments=[], out_dir=out_dir)

    for bad_edge in (5, None, "bad_edge"):
        dag = copy.deepcopy(valid_dag)
        dag["edges"] = [bad_edge]
        with pytest.raises(CompilationError):
            generator.emit(dag=dag, segments=[], out_dir=out_dir)

    # Non-dict segment
    dag_ok = copy.deepcopy(valid_dag)
    dag_ok["edges"] = []
    for bad_seg in ("seg_1", 123, None, ["not_a_dict"]):
        with pytest.raises(CompilationError):
            generator.emit(dag=dag_ok, segments=[bad_seg], out_dir=out_dir)


def test_phase0_nan_and_infinite_coordinates_raise_compilation_error(tmp_path: Path):
    """NaN / 'nan' / '1e999' coordinates and bbox values raise CompilationError."""
    generator = JinjaCodeGenerator()
    out_dir = tmp_path / "dist_nan"

    base_dag = {
        "workflow_name": "NanWorkflow",
        "root_node": "state_root",
        "terminal_node": "state_root",
        "nodes": {
            "state_root": {
                "id": "state_root",
                "title": "Root",
                "screen_name": "screen_main",
            }
        },
        "edges": [],
    }

    nan_bboxes = [
        {"x": float("nan"), "y": 0.1, "width": 0.2, "height": 0.3},
        {"x": 0.1, "y": float("inf"), "width": 0.2, "height": 0.3},
        {"x": 0.1, "y": 0.1, "width": "nan", "height": 0.3},
        {"x": 0.1, "y": 0.1, "width": 0.2, "height": "1e999"},
    ]

    for bbox in nan_bboxes:
        seg = {
            "id": "seg_test",
            "label": "Test",
            "bbox": bbox,
        }
        with pytest.raises(CompilationError):
            generator.emit(dag=base_dag, segments=[seg], out_dir=out_dir)


def test_phase0_injection_payloads_in_ids_and_metadata(tmp_path: Path):
    """Injection payloads in ids, timeout_ms, labels, selectors, and urls must either be rejected
    by strict validation (CompilationError) or safely emitted via workflow.json without executing code."""
    sentinel = tmp_path / "PWNED"
    if sentinel.exists():
        sentinel.unlink()

    generator = JinjaCodeGenerator()

    # 1. Injection in segment id
    bad_id = 'x""" + str(__import__("os").system("touch ' + str(sentinel) + '")) + """y'
    bad_seg = {
        "id": bad_id,
        "label": "Inject",
        "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
    }
    dag = {
        "workflow_name": "InjectWorkflow",
        "root_node": "state_root",
        "nodes": {"state_root": {"id": "state_root", "title": "Root", "screen_name": "s"}},
        "edges": [],
    }
    with pytest.raises(CompilationError):
        generator.emit(dag=dag, segments=[bad_seg], out_dir=tmp_path / "dist_bad_id")
    assert not sentinel.exists(), "Malicious id executed code!"

    # 2. Injection in edge timeout_ms
    bad_edge = {
        "from_node_id": "state_root",
        "to_node_id": "state_root",
        "target_selector": "#btn",
        "action_type": "click",
        "timeout_ms": '1, "z": __import__("os").system("touch ' + str(sentinel) + '")',
    }
    dag_bad_edge = copy.deepcopy(dag)
    dag_bad_edge["edges"] = [bad_edge]
    with pytest.raises(CompilationError):
        generator.emit(dag=dag_bad_edge, segments=[], out_dir=tmp_path / "dist_bad_edge")
    assert not sentinel.exists(), "Malicious timeout_ms executed code!"

    # 3. Tricky characters in labels/selectors/urls: quotes, backslashes, newlines, {{ }}, {% %}, U+2028
    tricky_label = 'Hello "world" \n \r \\ \' {{ 7*7 }} {% if True %}test{% endif %} \u2028'
    valid_seg = {
        "id": "seg_valid",
        "label": tricky_label,
        "suggested_selector": '#main > div[data-test="foo\\bar"]',
        "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
    }
    valid_edge = {
        "from_node_id": "state_root",
        "to_node_id": "state_root",
        "target_selector": '#btn[title="click\'me"]',
        "action_type": "click",
        "timeout_ms": 5000,
        "description": "Step with quotes and {{ 49 }}",
    }
    dag_tricky = copy.deepcopy(dag)
    dag_tricky["edges"] = [valid_edge]

    dist_tricky = tmp_path / "dist_tricky"
    generator.emit(dag=dag_tricky, segments=[valid_seg], out_dir=dist_tricky)
    assert not sentinel.exists()

    # Verify workflow.json round-trips exactly
    wf_json = dist_tricky / "workflow.json"
    assert wf_json.exists()
    data = json.loads(wf_json.read_text(encoding="utf-8"))
    assert data["anchors"]["seg_valid"]["label"] == tricky_label

    # Verify the driver imports and executes without running injected code
    sys.modules.pop("playwright_driver", None)
    sys.path.insert(0, str(dist_tricky.resolve()))
    try:
        driver_mod = importlib.import_module("playwright_driver")
        driver_cls = getattr(driver_mod, "InjectWorkflow")
        driver_inst = driver_cls()
        assert driver_inst.ANCHORS["seg_valid"]["label"] == tricky_label
        assert not sentinel.exists()
    finally:
        sys.modules.pop("playwright_driver", None)
        if str(dist_tricky.resolve()) in sys.path:
            sys.path.remove(str(dist_tricky.resolve()))


def test_phase0_determinism_and_immutability(tmp_path: Path):
    """emit() must deep-copy inputs, never mutate caller's dag/segments, and yield deterministic compile hash."""
    generator = JinjaCodeGenerator()

    original_dag = {
        "workflow_name": "ImmutableWorkflow",
        "root_node": "state_root",
        "terminal_node": "state_root",
        "nodes": {
            "state_root": {
                "id": "state_root",
                "title": "Root",
                "screen_name": "screen_main",
            }
        },
        "edges": [
            {
                "from_node_id": "state_root",
                "to_node_id": "state_root",
                "target_selector": "#btn",
                "action_type": "click",
                "timeout_ms": 5000,
            }
        ],
    }

    original_segments = [
        {
            "id": "seg_btn",
            "label": "Click Me",
            "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
        }
    ]

    dag_copy = copy.deepcopy(original_dag)
    segments_copy = copy.deepcopy(original_segments)

    out1 = tmp_path / "emit_run1"
    out2 = tmp_path / "emit_run2"

    res1 = generator.emit(dag=original_dag, segments=original_segments, out_dir=out1)
    # Caller's inputs must remain completely unmutated
    assert original_dag == dag_copy, "Caller dag was mutated by emit()!"
    assert original_segments == segments_copy, "Caller segments were mutated by emit()!"

    res2 = generator.emit(dag=original_dag, segments=original_segments, out_dir=out2)

    data1 = json.loads((out1 / "workflow.json").read_text(encoding="utf-8"))
    data2 = json.loads((out2 / "workflow.json").read_text(encoding="utf-8"))

    assert data1["compile_hash"] == data2["compile_hash"], "Compile hash is not deterministic!"


def test_phase0_templates_no_raw_jinja_and_no_demo_html(tmp_path: Path):
    """Generated playwright_driver.py and app.py must have zero data interpolations and no DEMO_HTML."""
    generator = JinjaCodeGenerator()
    out_dir = tmp_path / "dist_tmpl_check"
    dag = {
        "workflow_name": "TmplWorkflow",
        "root_node": "state_root",
        "terminal_node": "state_root",
        "nodes": {"state_root": {"id": "state_root", "title": "Root", "screen_name": "s"}},
        "edges": [],
    }
    generator.emit(dag=dag, segments=[], out_dir=out_dir)

    driver_code = (out_dir / "playwright_driver.py").read_text(encoding="utf-8")
    app_code = (out_dir / "app.py").read_text(encoding="utf-8")

    # DEMO_HTML literal must be removed
    assert "DEMO_HTML =" not in driver_code

    # Check template sources for unescaped data interpolations
    driver_tmpl = (Path(generator.templates_dir) / "playwright_driver.py.j2").read_text(encoding="utf-8")
    app_tmpl = (Path(generator.templates_dir) / "app.py.j2").read_text(encoding="utf-8")

    # Only {{ workflow_class_name }} and {{ compile_hash }} are allowed in templates
    allowed_patterns = {"workflow_class_name", "compile_hash"}
    for tmpl_name, tmpl_text in [("playwright_driver.py.j2", driver_tmpl), ("app.py.j2", app_tmpl)]:
        matches = re.findall(r"\{\{\s*([^}]+?)\s*\}\}", tmpl_text)
        for m in matches:
            clean_m = m.strip()
            assert clean_m in allowed_patterns, f"Unexpected data interpolation in {tmpl_name}: {m}"

    # main.py must be a 3-line shim from app import app
    main_code = (out_dir / "main.py").read_text(encoding="utf-8")
    assert "from app import app" in main_code

    # Stale templates must not exist
    templates_dir = Path(generator.templates_dir)
    assert not (templates_dir / "fastapi_app.py.j2").exists()
    assert not list(templates_dir.glob("*.jinja2"))

