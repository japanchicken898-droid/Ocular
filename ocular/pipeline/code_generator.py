"""Muse Code: Jinja2 Template Compiler targeting output directory."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from jinja2 import Environment, FileSystemLoader

from ocular.models.types import CompiledArtifacts, StateGraph
from ocular.pipeline.base import BaseGenerator

logger = logging.getLogger("ocular.pipeline.code_generator")


def _sanitize_param_key(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"param_{cleaned}"
    return cleaned


def derive_workflow_name_from_url(url_or_name: Optional[str], default: str = "OcularWorkflow") -> str:
    """Derives a clean PascalCase workflow class name dynamically from a target URL's domain.

    Examples:
        'https://excalidraw.com' -> 'ExcalidrawWorkflow'
        'excalidraw.com' -> 'ExcalidrawWorkflow'
        'https://example.com/chart' -> 'ExampleWorkflow'
        'http://localhost:3000' -> 'LocalhostWorkflow'
        'session_excalidraw' -> 'ExcalidrawWorkflow'
    """
    if not url_or_name:
        return default

    text = str(url_or_name).strip()
    if not text:
        return default

    # If already a valid PascalCase identifier ending with Workflow
    if re.match(r"^[A-Z][a-zA-Z0-9]*Workflow$", text):
        return text

    # Extract hostname or path
    if "://" in text or text.startswith("http"):
        parsed = urlparse(text)
        host = parsed.netloc or parsed.path
    else:
        host = text

    host = host.split(":")[0].strip()
    if host.startswith("www."):
        host = host[4:]

    # Remove domain extension (.com, .org, .io, etc.)
    parts = host.split(".")
    if len(parts) >= 2 and parts[-1].lower() in ("com", "org", "net", "io", "co", "app", "dev", "ai"):
        base_name = parts[-2]
    else:
        base_name = parts[0]

    # Clean path prefixes like session_
    base_name = Path(base_name).name
    if base_name.startswith("session_"):
        base_name = base_name[len("session_"):]

    # Convert to PascalCase
    tokens = re.split(r"[^a-zA-Z0-9]+", base_name)
    pascal = "".join(t.capitalize() for t in tokens if t)
    if not pascal:
        pascal = "Ocular"

    if not pascal.endswith("Workflow"):
        pascal = f"{pascal}Workflow"

    return pascal


def extract_telemetry_input_parameters(telemetry_events: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Extracts and groups keypress/input text events from telemetry stream into dynamic parameters."""
    inputs: List[Dict[str, Any]] = []
    if not telemetry_events:
        return inputs

    by_target: Dict[str, List[Dict[str, Any]]] = {}
    for ev in telemetry_events:
        ev_type = str(ev.get("event_type", "")).lower()
        action = str(ev.get("action_type", "")).lower()
        if ev_type in ("keydown", "input", "keypress") or action in ("fill", "type"):
            target = ev.get("target_selector") or ev.get("target_tag") or "text_input"
            by_target.setdefault(target, []).append(ev)

    for target, ev_list in by_target.items():
        chars = []
        for e in ev_list:
            k = e.get("key") or e.get("value") or e.get("input_payload") or ""
            if len(k) == 1:
                chars.append(k)
            elif k and k not in ("Enter", "Tab", "Shift", "Control", "Alt", "Meta", "Escape"):
                chars.append(k)

        default_val = "".join(chars) if chars else "sample_input"
        target_name = target
        if "#" in target_name:
            target_name = target_name.split("#")[-1]
        elif "." in target_name:
            target_name = target_name.split(".")[-1]
        target_name = target_name.split("[")[0].strip()
        clean_name = re.sub(r"[^a-zA-Z0-9_]+", "_", target_name).strip("_")
        key = _sanitize_param_key(clean_name or "input_text")

        inputs.append({
            "key": key,
            "parameter_ref": key,
            "target_anchor": clean_name,
            "default_value": default_val,
            "label": f"Input value for {target}",
            "action": "fill",
            "selector": target,
        })

    return inputs


class JinjaCodeGenerator(BaseGenerator):
    """Compiles state graphs and visual segments into runnable domain-agnostic Python artifacts."""

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
        self,
        dag: Dict[str, Any],
        segments: List[Dict[str, Any]],
        telemetry_events: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Extracts dynamic input fields from telemetry keypresses and DAG edges."""
        inputs: List[Dict[str, Any]] = []
        seen_keys = set()

        # 1. Telemetry keypress / input events parameterization
        telemetry_inputs = extract_telemetry_input_parameters(telemetry_events)
        for inp in telemetry_inputs:
            if inp["key"] not in seen_keys:
                seen_keys.add(inp["key"])
                inputs.append(inp)

        # 2. Inspect DAG edges for fill actions or payload keys
        edges = dag.get("edges", [])
        for e in edges:
            action = str(e.get("action_type") or e.get("action", "")).lower()
            if action in ("fill", "type") or e.get("input_payload"):
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
                            "default_value": e.get("default_value", e.get("input_payload", "sample_input")),
                            "label": e.get("target_label", key),
                            "action": "fill",
                            "selector": e.get("target_selector"),
                        }
                    )

        return inputs

    def _extract_action_steps(
        self,
        dag: Dict[str, Any],
        segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extracts generic action execution steps from state graph edges and nodes."""
        steps: List[Dict[str, Any]] = []
        edges = dag.get("edges", [])
        seg_by_id = {s.get("id"): s for s in segments if isinstance(s, dict)}

        for idx, e in enumerate(edges):
            action = str(e.get("action_type") or e.get("action") or "click").lower()
            target_id = e.get("target_segment_id") or e.get("trigger_element") or f"step_{idx}"
            seg = seg_by_id.get(target_id, {})
            bbox = seg.get("bbox", {})

            edge_coords = e.get("coordinates") or {}
            norm_x = edge_coords.get("x")
            norm_y = edge_coords.get("y")
            if norm_x is None:
                norm_x = bbox.get("x", 0.5) + bbox.get("width", 0.0) / 2.0
            if norm_y is None:
                norm_y = bbox.get("y", 0.5) + bbox.get("height", 0.0) / 2.0

            param_ref = e.get("parameter_ref")
            default_val = e.get("input_payload") or e.get("default_value") or ""

            steps.append({
                "step_index": idx + 1,
                "action": action,
                "target_anchor": target_id,
                "norm_x": round(float(norm_x), 4),
                "norm_y": round(float(norm_y), 4),
                "selector": e.get("target_selector") or seg.get("suggested_selector"),
                "label": e.get("target_label") or seg.get("label", target_id),
                "parameter_ref": param_ref,
                "default_value": default_val,
                "description": e.get("description", f"Step {idx + 1}: {action} on {target_id}"),
            })

        # Fallback if no edges in DAG: create steps from segments
        if not steps and segments:
            for idx, s in enumerate(segments):
                bbox = s.get("bbox", {})
                norm_x = bbox.get("x", 0.5) + bbox.get("width", 0.0) / 2.0
                norm_y = bbox.get("y", 0.5) + bbox.get("height", 0.0) / 2.0
                role = s.get("role", "button")
                action = "fill" if role in ("input", "textbox") else "click"
                steps.append({
                    "step_index": idx + 1,
                    "action": action,
                    "target_anchor": s.get("id", f"anchor_{idx}"),
                    "norm_x": round(float(norm_x), 4),
                    "norm_y": round(float(norm_y), 4),
                    "selector": s.get("suggested_selector"),
                    "label": s.get("label", s.get("id")),
                    "parameter_ref": s.get("id") if action == "fill" else None,
                    "default_value": "",
                    "description": f"Step {idx + 1}: {action} on {s.get('id')}",
                })

        return steps

    def emit(
        self,
        dag: Dict[str, Any],
        segments: List[Dict[str, Any]],
        out_dir: Union[str, Path] = "./dist",
        telemetry_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Path]:
        """Renders playwright_driver.py, app.py, and main.py into out_dir."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Derive workflow class name dynamically from URL domain or dag name
        raw_name = dag.get("target_url") or dag.get("url") or dag.get("workflow_name")
        workflow_class_name = derive_workflow_name_from_url(raw_name)
        dag["workflow_name"] = workflow_class_name

        compile_hash = self._compute_hash(dag, segments)

        # Extract dynamic inputs and generic action steps
        input_fields = self._extract_dynamic_inputs(dag, segments, telemetry_events=telemetry_events)
        for inp in input_fields:
            if "parameter_ref" not in inp:
                inp["parameter_ref"] = inp["key"]
        input_keys = [inp["key"] for inp in input_fields]

        action_steps = self._extract_action_steps(dag, segments)

        has_parameterized_inputs = len(input_fields) > 0
        if has_parameterized_inputs:
            properties = {}
            for inp in input_fields:
                properties[inp["key"]] = {
                    "type": "string",
                    "default": inp["default_value"],
                    "description": inp.get("label", f"{inp['key']} parameter"),
                }
            parameters_schema = {
                "title": f"{workflow_class_name}ExecutionRequest",
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
            }
        else:
            # Generic execution options schema
            parameters_schema = {
                "title": f"{workflow_class_name}ExecutionRequest",
                "type": "object",
                "properties": {
                    "viewport_width": {
                        "type": "integer",
                        "default": 1280,
                        "description": "Browser viewport width in pixels",
                    },
                    "viewport_height": {
                        "type": "integer",
                        "default": 720,
                        "description": "Browser viewport height in pixels",
                    },
                    "headless": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to run browser in headless mode",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "default": 10000,
                        "description": "Execution timeout in milliseconds",
                    },
                },
                "required": [],
            }
        dag["parameters_schema"] = parameters_schema

        target_url = dag.get("target_url") or dag.get("url")
        if not target_url:
            if raw_name and ("." in str(raw_name) or "://" in str(raw_name)):
                target_url = str(raw_name) if "://" in str(raw_name) else f"https://{raw_name}"
            else:
                target_url = "http://localhost:3000"

        anchors = {}
        for seg in segments:
            bbox = seg.get("bbox", {})
            norm_x = bbox.get("x", 0.0) + bbox.get("width", 0.0) / 2.0
            norm_y = bbox.get("y", 0.0) + bbox.get("height", 0.0) / 2.0
            anchors[seg.get("id")] = {
                "id": seg.get("id"),
                "label": seg.get("label", seg.get("id")),
                "role": seg.get("role", "element"),
                "norm_x": round(float(norm_x), 4),
                "norm_y": round(float(norm_y), 4),
                "bbox": bbox,
                "suggested_selector": seg.get("suggested_selector"),
            }

        context = {
            "dag": dag,
            "workflow_class_name": workflow_class_name,
            "target_url": target_url,
            "segments": segments,
            "anchors": anchors,
            "compile_hash": compile_hash,
            "input_fields": input_fields,
            "input_keys": input_keys,
            "action_steps": action_steps,
            "parameters_schema": parameters_schema,
            "has_parameterized_inputs": has_parameterized_inputs,
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

        # 3. Write main.py
        main_file = out_path / "main.py"
        main_file.write_text(app_code, encoding="utf-8")

        # 4. Save state_graph.json
        state_graph_file = out_path / "state_graph.json"
        state_graph_file.write_text(
            json.dumps(
                {
                    "dag": dag,
                    "workflow_class_name": workflow_class_name,
                    "compile_hash": compile_hash,
                    "action_steps": action_steps,
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
        telemetry_events: Optional[List[Dict[str, Any]]] = None,
    ) -> CompiledArtifacts:
        """Generates compiled artifacts from a typed StateGraph model."""
        out_path = Path(dist_dir)
        out_path.mkdir(parents=True, exist_ok=True)

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

        segments_list = []
        if graph.initial_node_id in graph.nodes:
            for s in graph.nodes[graph.initial_node_id].segments:
                segments_list.append(s.model_dump())

        rendered = self.emit(
            dag=dag_dict,
            segments=segments_list,
            out_dir=out_path,
            telemetry_events=telemetry_events,
        )

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
