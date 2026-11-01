"""Muse Code: Jinja2 Template Compiler targeting output directory."""

from __future__ import annotations

import copy
import hashlib
import json
import keyword
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError

from ocular.models.types import (
    CompiledArtifacts,
    StateGraph,
    VisualSegment,
    DAGEdge,
    DAGNode,
    ID_REGEX,
)
from ocular.pipeline.base import BaseGenerator
from ocular.pipeline.state_graph import CompilationError

logger = logging.getLogger("ocular.pipeline.code_generator")


def validate_compilation_inputs(dag: Any, segments: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Strictly validates compilation dag and segments using Pydantic models.

    Raises CompilationError if validation fails or inputs contain non-dict / invalid elements.
    """
    if not isinstance(dag, dict):
        raise CompilationError(f"DAG must be a dictionary, got {type(dag).__name__}")
    if not isinstance(segments, list):
        raise CompilationError(f"Segments must be a list, got {type(segments).__name__}")

    # Validate each segment is a dict and satisfies VisualSegment
    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise CompilationError(f"Segment at index {idx} must be a dictionary, got {type(seg).__name__}")
        seg_norm = dict(seg)
        if "label" not in seg_norm or not seg_norm["label"]:
            seg_norm["label"] = str(seg_norm.get("id", ""))
        try:
            VisualSegment.model_validate(seg_norm)
        except (ValidationError, ValueError) as val_err:
            raise CompilationError(f"Invalid segment at index {idx}: {val_err}") from val_err

    # Validate root_node and initial_node_id if present
    for node_key in ("root_node", "initial_node_id"):
        if node_key in dag and dag[node_key]:
            val = str(dag[node_key])
            if not ID_REGEX.match(val):
                raise CompilationError(f"Invalid {node_key} '{val}': does not match ^[A-Za-z0-9_-]{{1,64}}$")

    # Validate edges
    edges = dag.get("edges", [])
    if not isinstance(edges, list):
        raise CompilationError(f"DAG edges must be a list, got {type(edges).__name__}")

    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise CompilationError(f"Edge at index {idx} must be a dictionary, got {type(edge).__name__}")
        edge_norm = dict(edge)
        raw_action = edge_norm.get("action_type") or edge_norm.get("action") or "click"
        if hasattr(raw_action, "value"):
            action = str(raw_action.value).lower()
        else:
            action = str(raw_action).lower()
            if "." in action:
                action = action.split(".")[-1]
        edge_norm["action_type"] = action
        if "from_node" in edge_norm and "from_node_id" not in edge_norm:
            edge_norm["from_node_id"] = edge_norm["from_node"]
        elif "from_state" in edge_norm and "from_node_id" not in edge_norm:
            edge_norm["from_node_id"] = edge_norm["from_state"]
        if "to_node" in edge_norm and "to_node_id" not in edge_norm:
            edge_norm["to_node_id"] = edge_norm["to_node"]
        elif "to_state" in edge_norm and "to_node_id" not in edge_norm:
            edge_norm["to_node_id"] = edge_norm["to_state"]
        if "target_selector" not in edge_norm or not edge_norm["target_selector"]:
            target_id = edge_norm.get("target_segment_id") or edge_norm.get("trigger_element") or "element"
            edge_norm["target_selector"] = f"#{target_id}"
        try:
            DAGEdge.model_validate(edge_norm)
        except (ValidationError, ValueError) as val_err:
            raise CompilationError(f"Invalid edge at index {idx}: {val_err}") from val_err

    # Validate nodes
    nodes = dag.get("nodes", {})
    if not isinstance(nodes, dict):
        raise CompilationError(f"DAG nodes must be a dictionary, got {type(nodes).__name__}")
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            raise CompilationError(f"DAG node '{nid}' must be a dictionary, got {type(node).__name__}")
        node_norm = dict(node)
        if "id" not in node_norm or not node_norm["id"]:
            node_norm["id"] = nid
        if "title" not in node_norm or not node_norm["title"]:
            node_norm["title"] = str(node_norm.get("label") or nid)
        try:
            DAGNode.model_validate(node_norm)
        except (ValidationError, ValueError) as val_err:
            raise CompilationError(f"Invalid node '{nid}': {val_err}") from val_err

    return dag, segments


from ocular.security.redaction import EXPLICIT_PARAMETER_ALLOWLIST, is_sensitive_field, redact_event


def _sanitize_param_key(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"param_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned[:64]


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


IGNORED_NAMED_KEYS = {
    "shift", "control", "alt", "meta", "capslock",
    "arrowup", "arrowdown", "arrowleft", "arrowright",
    "enter", "tab", "escape", "dead", "process",
    "home", "end", "pageup", "pagedown", "insert",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
}


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
            # Hashing/processing of selector must tolerate non-str by skipping
            if not isinstance(target, str):
                continue
            by_target.setdefault(target, []).append(ev)

    seen_keys: Set[str] = set()
    for target, ev_list in by_target.items():
        # NEVER derive parameter key from typed characters or payload!
        target_name = target
        if "#" in target_name:
            target_name = target_name.split("#")[-1]
        elif "." in target_name:
            target_name = target_name.split(".")[-1]
        target_name = target_name.split("[")[0].strip()
        clean_name = re.sub(r"[^a-zA-Z0-9_]+", "_", target_name).strip("_")
        base_key = _sanitize_param_key(clean_name or "input_text")

        key = base_key
        suffix = 2
        while key in seen_keys:
            key = f"{base_key[:60]}_{suffix}"
            suffix += 1
        seen_keys.add(key)

        is_sensitive = is_sensitive_field({"key": key, "name": key, "selector": target, "target_selector": target})
        if not is_sensitive:
            for e in ev_list:
                if is_sensitive_field(e):
                    is_sensitive = True
                    break

        # Default-deny for typed text:
        # Sensitive fields or unknown sensitivity not in explicit allowlist get NO default value
        if is_sensitive or key not in EXPLICIT_PARAMETER_ALLOWLIST:
            default_val = ""
        else:
            chars: List[str] = []
            for e in ev_list:
                raw_k = e.get("key") if "key" in e else e.get("value")
                if not isinstance(raw_k, str):
                    continue
                k_lower = raw_k.lower()
                if k_lower == "backspace":
                    if chars:
                        chars.pop()
                elif k_lower in IGNORED_NAMED_KEYS:
                    continue
                elif len(raw_k) == 1:
                    chars.append(raw_k)
            default_val = "".join(chars) if chars else "sample_input"

        inputs.append({
            "key": key,
            "parameter_ref": key,
            "target_anchor": clean_name or key,
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
    session_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Extracts the session recording's origin URL from trusted sources only:
    1. Direct DAG metadata (passed from CLI --target-url / -u or target_url/url key)
    2. session_metadata.json written by recorder in the session dir or out dir.

    NEVER from telemetry events or CWD-relative search paths.
    Never silently defaults to localhost:3000.
    Raises CompilationError if no valid origin is found.
    """
    dag = dag or {}

    def _validate_origin(val: Any) -> Optional[str]:
        if not val or not isinstance(val, str):
            return None
        val_str = val.strip()
        if "@" in val_str:
            raise CompilationError(f"Target URL must not contain embedded user credentials: {val_str}")
        try:
            parsed = urlparse(val_str)
            if parsed.scheme.lower() in ("http", "https") and parsed.hostname:
                host = parsed.hostname.lower()
                port_str = f":{parsed.port}" if parsed.port and parsed.port not in (80, 443) else ""
                path = parsed.path.rstrip("/")
                return f"{parsed.scheme.lower()}://{host}{port_str}{path}"
        except CompilationError:
            raise
        except Exception:
            pass
        return None

    # 1. Direct DAG metadata
    for k in ("target_url", "url", "origin_url", "origin"):
        validated = _validate_origin(dag.get(k))
        if validated:
            return validated

    # 2. Check session_metadata.json in trusted directories
    candidate_dirs: List[Path] = []
    if session_dir:
        candidate_dirs.append(Path(session_dir))
    for key in ("recording_dir", "session_dir", "video_path", "input_path"):
        p_val = dag.get(key)
        if p_val:
            p = Path(p_val)
            candidate_dirs.append(p if p.is_dir() else p.parent)
    if out_dir:
        candidate_dirs.append(Path(out_dir))

    for sdir in candidate_dirs:
        meta_file = sdir / "session_metadata.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                for k in ("url", "target_url", "origin", "origin_url"):
                    validated = _validate_origin(data.get(k))
                    if validated:
                        return validated
            except Exception:
                pass

    raise CompilationError(
        "No target URL specified. Provide target_url in DAG/CLI (--target-url / -u) "
        "or ensure session_metadata.json contains a valid http(s) origin."
    )


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
        """Safely writes content to target path atomically with strict permissions, refusing symlinks."""
        import os
        import uuid

        if path.is_symlink() or os.path.islink(path):
            raise IOError(f"Refusing to write to symlink at {path}")

        target_dir = path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(target_dir, 0o700)
        except Exception:
            pass

        tmp_path = target_dir / f".tmp_{path.name}_{uuid.uuid4().hex}"
        try:
            tmp_path.write_text(content, encoding=encoding)
            try:
                os.chmod(tmp_path, 0o600)
            except Exception:
                pass
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

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
                # NEVER derive parameter key from input_payload
                raw_key = e.get("parameter_ref") or target_id
                base_key = _sanitize_param_key(raw_key)
                key = base_key
                suffix = 2
                while key in seen_keys:
                    key = f"{base_key[:60]}_{suffix}"
                    suffix += 1

                seen_keys.add(key)
                is_sensitive = is_sensitive_field({
                    "key": key,
                    "name": key,
                    "target": target_id,
                    "target_selector": e.get("target_selector"),
                    "selector": e.get("target_selector"),
                })
                if is_sensitive or key not in EXPLICIT_PARAMETER_ALLOWLIST:
                    default_val = ""
                    e["input_payload"] = "[REDACTED]"
                    e["default_value"] = ""
                else:
                    default_val = e.get("default_value", "sample_input")

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
            target_selector = e.get("target_selector") or seg.get("suggested_selector")
            is_sensitive = is_sensitive_field({
                "target": target_id,
                "selector": target_selector,
                "parameter_ref": param_ref,
                "label": e.get("target_label"),
            })

            if is_sensitive or (not param_ref) or (param_ref not in EXPLICIT_PARAMETER_ALLOWLIST):
                default_val = ""
                e["input_payload"] = "[REDACTED]"
                e["default_value"] = ""
            else:
                default_val = e.get("default_value") or ""

            steps.append({
                "step_index": idx + 1,
                "action": action,
                "target_anchor": target_id,
                "norm_x": round(float(norm_x), 4),
                "norm_y": round(float(norm_y), 4),
                "selector": target_selector,
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
        # 1. Strictly validate compilation inputs before processing
        validate_compilation_inputs(dag, segments)

        # 2. Deep copy inputs so caller's objects are never mutated
        dag = copy.deepcopy(dag)
        segments = copy.deepcopy(segments)
        telemetry_events = copy.deepcopy(telemetry_events) if telemetry_events else None

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Derive workflow class name dynamically from dag name or URL domain
        raw_name = dag.get("workflow_name") or dag.get("target_url") or dag.get("url")
        workflow_class_name = derive_workflow_name_from_url(raw_name)
        workflow_class_name = sanitize_workflow_class_name(workflow_class_name)
        dag["workflow_name"] = workflow_class_name

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
            required_keys = []
            for inp in input_fields:
                k = inp["key"]
                properties[k] = {
                    "type": "string",
                    "default": inp["default_value"],
                    "description": inp.get("label", f"{k} parameter"),
                }
                if inp.get("required") or not inp.get("default_value"):
                    required_keys.append(k)

            parameters_schema = {
                "title": f"{workflow_class_name}ExecutionRequest",
                "type": "object",
                "properties": properties,
                "required": required_keys,
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

        anchors = {}
        for seg in segments:
            bbox = seg.get("bbox", {})
            norm_x = float(bbox.get("x", 0.0) + bbox.get("width", 0.0) / 2.0)
            norm_y = float(bbox.get("y", 0.0) + bbox.get("height", 0.0) / 2.0)
            anchors[seg.get("id")] = {
                "id": seg.get("id"),
                "label": seg.get("label", seg.get("id")),
                "role": seg.get("role", "element"),
                "norm_x": round(norm_x, 4),
                "norm_y": round(norm_y, 4),
                "bbox": {
                    "x": float(bbox.get("x", 0.0)),
                    "y": float(bbox.get("y", 0.0)),
                    "width": float(bbox.get("width", 0.0)),
                    "height": float(bbox.get("height", 0.0)),
                },
                "suggested_selector": seg.get("suggested_selector"),
            }

        input_fields_dict = {
            inp["key"]: {
                "key": inp.get("key"),
                "target_anchor": inp.get("target_anchor", ""),
                "default_value": inp.get("default_value", ""),
                "label": inp.get("label", ""),
                "action": inp.get("action", "fill"),
                "selector": inp.get("selector", ""),
                "parameter_ref": inp.get("parameter_ref", inp["key"]),
            }
            for inp in input_fields
        }

        cleaned_steps = []
        for step in action_steps:
            cleaned_steps.append({
                "step_index": step.get("step_index", 0),
                "action": step.get("action", "click"),
                "target_anchor": step.get("target_anchor", ""),
                "norm_x": float(step.get("norm_x", 0.5)),
                "norm_y": float(step.get("norm_y", 0.5)),
                "selector": step.get("selector"),
                "label": step.get("label"),
                "parameter_ref": step.get("parameter_ref"),
                "default_value": step.get("default_value", ""),
                "description": step.get("description", ""),
                "timeout_ms": step.get("timeout_ms"),
            })

        # Deterministic compile hash based on normalized workflow payload
        hash_payload = {
            "workflow_name": workflow_class_name,
            "target_url": target_url,
            "anchors": anchors,
            "input_fields": input_fields_dict,
            "action_steps": cleaned_steps,
            "parameters_schema": parameters_schema,
        }
        compile_hash = hashlib.sha256(
            json.dumps(hash_payload, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()[:16]

        # Save workflow.json (primary source of truth loaded by driver)
        workflow_json_data = {
            "workflow_name": workflow_class_name,
            "workflow_class_name": workflow_class_name,
            "target_url": target_url,
            "compile_hash": compile_hash,
            "anchors": anchors,
            "input_fields": input_fields_dict,
            "action_steps": cleaned_steps,
            "parameters_schema": parameters_schema,
            "has_parameterized_inputs": has_parameterized_inputs,
        }
        workflow_json_file = out_path / "workflow.json"
        self._safe_write_text(
            workflow_json_file,
            json.dumps(workflow_json_data, indent=2, allow_nan=False),
        )

        context = {
            "workflow_class_name": workflow_class_name,
            "compile_hash": compile_hash,
        }

        # 1. Render playwright_driver.py
        driver_tmpl = self.env.get_template("playwright_driver.py.j2")
        driver_code = driver_tmpl.render(context)
        driver_file = out_path / "playwright_driver.py"
        self._safe_write_text(driver_file, driver_code)

        # 2. Render app.py (FastAPI microservice with /api/v1/execute)
        app_tmpl = self.env.get_template("app.py.j2")
        app_code = app_tmpl.render(context)
        app_file = out_path / "app.py"
        self._safe_write_text(app_file, app_code)

        # 3. Write main.py (3-line entrypoint shim)
        main_file = out_path / "main.py"
        self._safe_write_text(
            main_file,
            '"""Ocular application entrypoint shim."""\nfrom app import app\n\n__all__ = ["app"]\n',
        )

        # 4. Save state_graph.json
        clean_dag = copy.deepcopy(dag)
        for edge in clean_dag.get("edges", []):
            p_ref = edge.get("parameter_ref")
            is_sens = is_sensitive_field(edge)
            if is_sens or (not p_ref) or (p_ref not in EXPLICIT_PARAMETER_ALLOWLIST):
                if "input_payload" in edge:
                    edge["input_payload"] = "[REDACTED]"
                if "default_value" in edge:
                    edge["default_value"] = ""
            for ev_list_key in ("events", "telemetry_events", "keypresses"):
                if ev_list_key in edge and isinstance(edge[ev_list_key], list):
                    edge[ev_list_key] = [redact_event(ev) for ev in edge[ev_list_key] if isinstance(ev, dict)]

        state_graph_file = out_path / "state_graph.json"
        self._safe_write_text(
            state_graph_file,
            json.dumps(
                {
                    "dag": clean_dag,
                    "workflow_class_name": workflow_class_name,
                    "compile_hash": compile_hash,
                    "action_steps": cleaned_steps,
                    "input_fields": input_fields,
                    "parameters_schema": parameters_schema,
                },
                indent=2,
                allow_nan=False,
            ),
        )

        # 5. Save workflow_dag.mermaid
        from ocular.pipeline.state_graph import export_mermaid_dag
        mermaid_file = out_path / "workflow_dag.mermaid"
        export_mermaid_dag(dag, mermaid_file)

        # 6. Copy netguard.py (shared SSRF and network confinement engine)
        import ocular.security.netguard as netguard_module
        netguard_src = Path(netguard_module.__file__)
        netguard_dst = out_path / "netguard.py"
        self._safe_write_text(netguard_dst, netguard_src.read_text(encoding="utf-8"))

        # 7. Copy redaction.py (unified secrets redaction and exception scrubbing engine)
        import ocular.security.redaction as redaction_module
        redaction_src = Path(redaction_module.__file__)
        redaction_dst = out_path / "redaction.py"
        self._safe_write_text(redaction_dst, redaction_src.read_text(encoding="utf-8"))

        return {
            "playwright_driver": driver_file.resolve(),
            "app": app_file.resolve(),
            "main": main_file.resolve(),
            "state_graph": state_graph_file.resolve(),
            "workflow_dag_mermaid": mermaid_file.resolve(),
            "workflow_json": workflow_json_file.resolve(),
            "netguard": netguard_dst.resolve(),
            "redaction": redaction_dst.resolve(),
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
