"""Muse Code: Jinja2 Template Compiler targeting output directory."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
from jinja2 import Environment, FileSystemLoader

from ocular.models.types import CompiledArtifacts, StateGraph
from ocular.pipeline.base import BaseGenerator

logger = logging.getLogger("ocular.pipeline.code_generator")


def _sanitize_param_key(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"param_{cleaned}"
    return cleaned


class JinjaCodeGenerator(BaseGenerator):
    """Compiles state graphs and visual segments into runnable Python artifacts."""

    def __init__(self, templates_dir: Optional[Union[str, Path]] = None) -> None:
        if templates_dir is None:
            self.templates_dir = Path(__file__).resolve().parent.parent / "templates"
        else:
            self.templates_dir = Path(templates_dir)

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _compute_hash(self, dag: Dict[str, Any], segments: List[Dict[str, Any]]) -> str:
        payload = json.dumps({"dag": dag, "segments": segments}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _extract_dynamic_inputs(
        self, dag: Dict[str, Any], segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extracts and parameterizes dynamic form input fields from DAG transitions and segments."""
        inputs: List[Dict[str, Any]] = []
        seen_keys = set()
        seg_by_id = {s.get("id"): s for s in segments if isinstance(s, dict)}

        # 1. Inspect DAG edges for fill actions or payload keys
        edges = dag.get("edges", [])
        for e in edges:
            action = e.get("action_type") or e.get("action")
            if action == "fill" or e.get("input_payload"):
                target_id = e.get("trigger_element") or e.get("target_segment_id") or "field"
                raw_key = e.get("input_payload") or e.get("parameter_ref") or target_id
                key = _sanitize_param_key(raw_key)
                if key not in seen_keys:
                    seen_keys.add(key)
                    inputs.append(
                        {
                            "key": key,
                            "parameter_ref": key,
                            "target_anchor": target_id,
                            "default_value": e.get("default_value", "sample_input"),
                            "label": e.get("target_label", key),
                            "action": "fill",
                            "selector": e.get("target_selector"),
                        }
                    )

        # 2. Inspect segments for interactive input fields
        for s in segments:
            if isinstance(s, dict) and s.get("role") in ("input", "form"):
                raw_key = s.get("id", "input")
                key = _sanitize_param_key(raw_key)
                if key not in seen_keys:
                    seen_keys.add(key)
                    inputs.append(
                        {
                            "key": key,
                            "parameter_ref": key,
                            "target_anchor": s.get("id"),
                            "default_value": s.get("metadata", {}).get("value", "sample_input"),
                            "label": s.get("label", key),
                            "action": "fill",
                            "selector": s.get("suggested_selector"),
                        }
                    )

        # 3. Always ensure 'interval' is registered for TradingView or default workflows
        if "interval" not in seen_keys:
            inputs.append(
                {
                    "key": "interval",
                    "parameter_ref": "interval",
                    "target_anchor": "interval_dropdown",
                    "default_value": "15m",
                    "label": "Timeframe Interval",
                    "action": "click",
                    "selector": "div[data-role='interval-menu']",
                }
            )

        return inputs

    def emit(
        self,
        dag: Dict[str, Any],
        segments: List[Dict[str, Any]],
        out_dir: Union[str, Path] = "./dist",
    ) -> Dict[str, Path]:
        """Renders playwright_driver.py, app.py, and main.py into out_dir."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        compile_hash = self._compute_hash(dag, segments)
        input_fields = self._extract_dynamic_inputs(dag, segments)
        for inp in input_fields:
            if "parameter_ref" not in inp:
                inp["parameter_ref"] = inp["key"]
        input_keys = [inp["key"] for inp in input_fields]

        parameters_schema = dag.get("parameters_schema")
        if not parameters_schema or not isinstance(parameters_schema, dict) or "properties" not in parameters_schema:
            properties = {}
            for inp in input_fields:
                properties[inp["key"]] = {
                    "type": "string",
                    "default": inp["default_value"],
                    "description": inp.get("label", f"{inp['key']} parameter"),
                }
            parameters_schema = {
                "title": "WorkflowExecutionRequest",
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
            }
        dag["parameters_schema"] = parameters_schema

        context = {
            "dag": dag,
            "segments": segments,
            "compile_hash": compile_hash,
            "input_fields": input_fields,
            "input_keys": input_keys,
            "parameters_schema": parameters_schema,
        }

        # 1. Render playwright_driver.py
        driver_tmpl = self.env.get_template("playwright_driver.py.j2")
        driver_code = driver_tmpl.render(context)
        driver_file = out_path / "playwright_driver.py"
        driver_file.write_text(driver_code, encoding="utf-8")

        # 2. Render app.py (FastAPI microservice with /api/v1/execute)
        tmpl_name = "app.py.j2" if (self.templates_dir / "app.py.j2").exists() else "fastapi_app.py.j2"
        app_tmpl = self.env.get_template(tmpl_name)
        app_code = app_tmpl.render(context)
        app_file = out_path / "app.py"
        app_file.write_text(app_code, encoding="utf-8")

        # 3. Also write main.py (identical copy for backward compatibility)
        main_file = out_path / "main.py"
        main_file.write_text(app_code, encoding="utf-8")

        # 4. Save state_graph.json
        state_graph_file = out_path / "state_graph.json"
        state_graph_file.write_text(
            json.dumps(
                {
                    "dag": dag,
                    "compile_hash": compile_hash,
                    "input_fields": input_fields,
                    "parameters_schema": parameters_schema,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # 5. Save workflow_dag.mermaid
        from ocular.pipeline.state_graph import export_mermaid_dag
        mermaid_file = out_path / "workflow_dag.mermaid"
        export_mermaid_dag(dag, mermaid_file)

        return {
            "playwright_driver": driver_file.resolve(),
            "app": app_file.resolve(),
            "main": main_file.resolve(),
            "state_graph": state_graph_file.resolve(),
            "workflow_dag_mermaid": mermaid_file.resolve(),
        }

    def generate(
        self,
        graph: StateGraph,
        dist_dir: Union[str, Path] = "./dist",
    ) -> CompiledArtifacts:
        """Generates compiled artifacts from a typed StateGraph model."""
        out_path = Path(dist_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Convert StateGraph to dag dictionary
        dag_dict = {
            "workflow_name": graph.name,
            "root_node": graph.initial_node_id,
            "terminal_node": graph.terminal_node_ids[0]
            if graph.terminal_node_ids
            else graph.initial_node_id,
            "nodes": {nid: n.model_dump() for nid, n in graph.nodes.items()},
            "edges": [e.model_dump() for e in graph.edges],
            "topological_order": list(graph.nodes.keys()),
        }

        # Extract segments from initial node
        segments_list = []
        if graph.initial_node_id in graph.nodes:
            for s in graph.nodes[graph.initial_node_id].segments:
                segments_list.append(s.model_dump())

        # Render artifacts
        rendered = self.emit(dag=dag_dict, segments=segments_list, out_dir=out_path)

        manifest_file = out_path / "manifest.json"
        manifest_data = {
            "workflow": graph.name,
            "states_count": len(graph.nodes),
            "transitions_count": len(graph.edges),
            "runtime_vision_inference": False,
        }
        manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        return CompiledArtifacts(
            playwright_script_path=str(rendered["playwright_driver"]),
            fastapi_app_path=str(rendered["app"]),
            state_graph_path=str(rendered["state_graph"]),
            manifest_path=str(manifest_file.resolve()),
            dist_dir=str(out_path.resolve()),
            steps_count=len(graph.edges),
            states_count=len(graph.nodes),
        )


# Architectural alias
CodeGenerator = JinjaCodeGenerator
