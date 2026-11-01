"""Muse Code: Jinja2 Template Compiler targeting output directory."""

from __future__ import annotations

import hashlib
import json
import keyword
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from jinja2 import Environment, FileSystemLoader

from ocular.models.types import CompiledArtifacts, StateGraph
from ocular.pipeline.base import BaseGenerator

logger = logging.getLogger("ocular.pipeline.code_generator")


SECRET_PATTERN = re.compile(
    r"password|token|secret|credit_card|cvv|one-time-code|otp|pin|ssn|auth_key|api_key|auth|bearer", re.IGNORECASE
)


def _sanitize_param_key(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"param_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def sanitize_workflow_class_name(name: Optional[str]) -> str:
    """Sanitizes a string to be a valid Python identifier for workflow class names.

    Guarantees an alphabetic prefix, strips invalid characters, and avoids reserved keywords.
    """
    if not name:
        return "OcularWorkflow"

    text = str(name).strip()
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "Ocular"

    if not cleaned[0].isalpha():
        cleaned = f"Workflow_{cleaned}"

    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"

    if not cleaned.endswith("Workflow"):
        cleaned = f"{cleaned}Workflow"

    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"

    return cleaned


def derive_workflow_name_from_url(url_or_name: Optional[str], default: str = "OcularWorkflow") -> str:
    """Derives a clean PascalCase workflow class name dynamically from a target URL's domain.

    Examples:
        'https://excalidraw.com' -> 'ExcalidrawWorkflow'
        'excalidraw.com' -> 'ExcalidrawWorkflow'
        'https://example.com/chart' -> 'ExampleWorkflow'
        'http://localhost:3000' -> 'LocalhostWorkflow'
        'http://127.0.0.1:8000' -> 'Workflow_127Workflow'
        'session_excalidraw' -> 'ExcalidrawWorkflow'
    """
    if not url_or_name:
        return default

    text = str(url_or_name).strip()
    if not text:
        return default

    # If already a valid PascalCase identifier ending with Workflow
    if re.match(r"^[A-Za-z][a-zA-Z0-9_]*Workflow$", text):
        return sanitize_workflow_class_name(text)

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

    return sanitize_workflow_class_name(pascal)


def extract_telemetry_input_parameters(telemetry_events: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Extracts and groups keypress/input text events from telemetry stream into dynamic parameters."""
    inputs: List[Dict[str, Any]] = []
    if not telemetry_events:
        return inputs

    by_target: Dict[str, List[Dict[str, Any]]] = {}
    for ev in telemetry_events:
        if not isinstance(ev, dict):
            continue
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

        is_sensitive = bool(
            SECRET_PATTERN.search(key)
            or SECRET_PATTERN.search(target)
            or "[REDACTED]" in default_val
        )
        if is_sensitive:
            default_val = ""

        inputs.append({
            "key": key,
            "parameter_ref": key,
            "target_anchor": clean_name,
            "default_value": default_val,
            "label": f"Input value for {target}" if not is_sensitive else f"Credential parameter {key}",
            "action": "fill",
            "selector": target,
        })

    return inputs


def extract_origin_url(
    dag: Optional[Dict[str, Any]] = None,
    telemetry_events: Optional[List[Dict[str, Any]]] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Extracts the session recording's origin URL from session_metadata.json or telemetry events."""
    dag = dag or {}

    # 1. Direct DAG metadata
    for k in ("target_url", "url", "origin_url", "origin"):
        val = dag.get(k)
        if val and isinstance(val, str) and ("http://" in val or "https://" in val):
            return val.rstrip("/")

    # 2. Check telemetry events (first event or any event)
    if telemetry_events:
        for ev in telemetry_events:
            if not isinstance(ev, dict):
                continue
            for k in ("url", "page_url", "origin", "target_url"):
                val = ev.get(k)
                if val and isinstance(val, str) and ("http://" in val or "https://" in val):
                    return val.rstrip("/")

    # 3. Check session_metadata.json in known recording paths
    search_dirs: List[Path] = []
    if out_dir:
        out_p = Path(out_dir)
        search_dirs.append(out_p)
        name = out_p.name
        if name.startswith("dist_"):
            clean = name[len("dist_"):]
            search_dirs.append(Path(f"recordings/session_{clean}"))
            search_dirs.append(Path(f"recordings/{clean}"))

    for key in ("recording_dir", "session_dir", "video_path", "input_path"):
        p_val = dag.get(key)
        if p_val:
            p = Path(p_val)
            search_dirs.append(p if p.is_dir() else p.parent)

    search_dirs.extend([
        Path("recordings/session_excalidraw"),
        Path("recordings"),
    ])

    for sdir in search_dirs:
        meta_file = sdir / "session_metadata.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                for k in ("url", "target_url", "origin"):
                    val = data.get(k)
                    if val and isinstance(val, str) and ("http://" in val or "https://" in val):
                        return val.rstrip("/")
            except Exception:
                pass

    # 4. Check workflow name or raw name hints
    raw_name = str(dag.get("workflow_name") or dag.get("url") or "")
    if "excalidraw" in raw_name.lower():
        return "https://excalidraw.com"

    if "." in raw_name and not raw_name.endswith(".py"):
        if "://" in raw_name:
            return raw_name.rstrip("/")
        return f"https://{raw_name}".rstrip("/")

    return "http://localhost:3000"


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

    @staticmethod
    def _safe_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
        """Safely writes content to target path, rejecting or unlinking existing symlinks to prevent symlink traversal attacks."""
        if path.is_symlink():
            logger.warning(f"Rejecting existing symlink at {path}; unlinking before write.")
            path.unlink()
        path.write_text(content, encoding=encoding)

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
                    default_val = e.get("default_value", e.get("input_payload", "sample_input"))
                    is_sensitive = bool(
                        SECRET_PATTERN.search(key)
                        or SECRET_PATTERN.search(str(target_id))
                        or SECRET_PATTERN.search(str(raw_key))
                        or "[REDACTED]" in str(default_val)
                    )
                    if is_sensitive:
                        default_val = ""

                    inputs.append(
                        {
                            "key": key,
                            "parameter_ref": key,
                            "target_anchor": target_id,
                            "default_value": default_val,
                            "label": e.get("target_label", key) if not is_sensitive else f"Credential parameter {key}",
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
                "timeout_ms": e.get("timeout_ms"),
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
                    "timeout_ms": s.get("timeout_ms"),
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
        workflow_class_name = sanitize_workflow_class_name(workflow_class_name)
        dag["workflow_name"] = workflow_class_name

        compile_hash = self._compute_hash(dag, segments)
        compile_hash_repr = json.dumps(compile_hash)
        workflow_name_repr = json.dumps(workflow_class_name)

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

        target_url = extract_origin_url(dag, telemetry_events=telemetry_events, out_dir=out_path)
        dag["target_url"] = target_url
        target_url_repr = json.dumps(target_url)

        # Pre-serialize string attributes in segments, anchors, inputs, and action steps
        for seg in segments:
            seg["id_repr"] = json.dumps(str(seg.get("id", "")))
            seg["label_repr"] = json.dumps(str(seg.get("label", seg.get("id", ""))))
            seg["role_repr"] = json.dumps(str(seg.get("role", "element")))
            seg["suggested_selector_repr"] = json.dumps(str(seg.get("suggested_selector") or ""))

        anchors = {}
        for seg in segments:
            bbox = seg.get("bbox", {})
            norm_x = bbox.get("x", 0.0) + bbox.get("width", 0.0) / 2.0
            norm_y = bbox.get("y", 0.0) + bbox.get("height", 0.0) / 2.0
            anchors[seg.get("id")] = {
                "id": seg.get("id"),
                "id_repr": seg["id_repr"],
                "label": seg.get("label", seg.get("id")),
                "label_repr": seg["label_repr"],
                "role": seg.get("role", "element"),
                "role_repr": seg["role_repr"],
                "norm_x": round(float(norm_x), 4),
                "norm_y": round(float(norm_y), 4),
                "bbox": bbox,
                "suggested_selector": seg.get("suggested_selector"),
                "suggested_selector_repr": seg["suggested_selector_repr"],
            }

        for inp in input_fields:
            inp["key_repr"] = json.dumps(str(inp.get("key") or ""))
            inp["target_anchor_repr"] = json.dumps(str(inp.get("target_anchor") or ""))
            inp["default_value_repr"] = json.dumps(str(inp.get("default_value") if inp.get("default_value") is not None else ""))
            inp["label_repr"] = json.dumps(str(inp.get("label") or ""))
            inp["action_repr"] = json.dumps(str(inp.get("action") or "fill"))
            inp["selector_repr"] = json.dumps(str(inp.get("selector") or ""))
            inp["parameter_ref_repr"] = json.dumps(str(inp.get("parameter_ref") or ""))

        for step in action_steps:
            step["action_repr"] = json.dumps(str(step.get("action") or "click"))
            step["target_anchor_repr"] = json.dumps(str(step.get("target_anchor") or ""))
            step["selector_repr"] = json.dumps(str(step.get("selector"))) if step.get("selector") is not None else "None"
            step["label_repr"] = json.dumps(str(step.get("label"))) if step.get("label") is not None else "None"
            step["parameter_ref_repr"] = json.dumps(str(step.get("parameter_ref"))) if step.get("parameter_ref") is not None else "None"
            step["default_value_repr"] = json.dumps(str(step.get("default_value") if step.get("default_value") is not None else ""))
            step["description_repr"] = json.dumps(str(step.get("description") or ""))

        context = {
            "dag": dag,
            "workflow_class_name": workflow_class_name,
            "workflow_name_repr": workflow_name_repr,
            "target_url": target_url,
            "target_url_repr": target_url_repr,
            "segments": segments,
            "anchors": anchors,
            "compile_hash": compile_hash,
            "compile_hash_repr": compile_hash_repr,
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
        self._safe_write_text(driver_file, driver_code)

        # 2. Render app.py (FastAPI microservice with /api/v1/execute)
        tmpl_name = "app.py.j2" if (self.templates_dir / "app.py.j2").exists() else "fastapi_app.py.j2"
        app_tmpl = self.env.get_template(tmpl_name)
        app_code = app_tmpl.render(context)
        app_file = out_path / "app.py"
        self._safe_write_text(app_file, app_code)

        # 3. Write main.py
        main_file = out_path / "main.py"
        self._safe_write_text(main_file, app_code)

        # 4. Save state_graph.json
        state_graph_file = out_path / "state_graph.json"
        self._safe_write_text(
            state_graph_file,
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

        meta_url = getattr(graph, "target_url", None)
        if not meta_url and hasattr(graph, "metadata") and isinstance(graph.metadata, dict):
            meta_url = graph.metadata.get("target_url")

        dag_dict = {
            "workflow_name": graph.name,
            "target_url": meta_url,
            "root_node": graph.initial_node_id,
            "terminal_node": graph.terminal_node_ids[0]
            if graph.terminal_node_ids
            else graph.initial_node_id,
            "nodes": {nid: n.model_dump() for nid, n in graph.nodes.items()},
            "edges": [e.model_dump() for e in graph.edges],
            "topological_order": list(graph.nodes.keys()),
        }

        segments_list = []
        if hasattr(graph, "segments") and graph.segments:
            for s in graph.segments.values():
                segments_list.append(s.model_dump() if hasattr(s, "model_dump") else s)
        elif graph.initial_node_id in graph.nodes and graph.nodes[graph.initial_node_id].segments:
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
        self._safe_write_text(manifest_file, json.dumps(manifest_data, indent=2))

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
