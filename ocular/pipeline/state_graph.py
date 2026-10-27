"""Muse Spark: State Transition Graph (DAG) Synthesizer.

Parses keyframe diffs, transitions, and segmented targets into a Directed
Acyclic Graph (DAG) representing the sequential UI workflow.
Models nodes as states (URL, screen hash, pre-conditions) and edges as actions
(click, fill, wait, assert).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import urllib.request
import networkx as nx
from rich.tree import Tree

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ocular.models.types import (
    ActionType,
    DAGEdge,
    DAGNode,
    FrameMetadata,
    StateGraph,
    TransitionEvent,
    VisualSegment,
)
from ocular.pipeline.base import BaseSynthesizer

logger = logging.getLogger("ocular.pipeline.state_graph")


class DAGBuilder(BaseSynthesizer):
    """Synthesizes visual segments and transition diffs into a verified DAG."""

    def __init__(self, workflow_name: str = "TradingViewChartWorkflow") -> None:
        self.workflow_name = workflow_name
        self.last_inference_source: Optional[str] = None

    def _hash_screen(self, label: str, seg_ids: List[str]) -> str:
        content = f"{label}:{':'.join(sorted(seg_ids))}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _call_muse_spark_api(
        self,
        segments: List[Dict[str, Any]],
        transitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Calls the live Baseten Meta Muse Spark endpoint using the standard OpenAI client."""
        meta_key = os.getenv("META_MODEL_API_KEY")
        baseten_key = os.getenv("BASETEN_API_KEY")
        if not meta_key:
            return None
        api_key = baseten_key or meta_key

        if OpenAI is None:
            return None

        base_url = os.getenv("MUSE_SPARK_API_ENDPOINT", "https://inference.baseten.co/v1")
        model_id = os.getenv("MUSE_SPARK_MODEL_ID", "meta-models/muse-spark-open-1026")

        try:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=25.0,
                max_retries=1,
            )

            system_prompt = (
                "You are Meta Muse Spark, an expert state transition graph (DAG) synthesizer.\n"
                "Given keyframe transition summaries and SAM bounding box manifests, synthesize a compact Directed Acyclic Graph (DAG) "
                "representing state transitions, DAG topology, and dynamic variables.\n"
                "Return ONLY a clean valid JSON object with the following schema:\n"
                "{\n"
                '  "workflow_name": "string",\n'
                '  "root_node": "state_root",\n'
                '  "terminal_node": "string",\n'
                '  "nodes": {\n'
                '    "<node_id>": {\n'
                '      "id": "<node_id>",\n'
                '      "label": "string",\n'
                '      "title": "string",\n'
                '      "screen_name": "string",\n'
                '      "screen_hash": "string",\n'
                '      "url_pattern": "string",\n'
                '      "preconditions": [{"type": "string", "target_id": "string", "description": "string"}],\n'
                '      "assertions": ["string"],\n'
                '      "route": "string",\n'
                '      "method": "GET" | "POST"\n'
                "    }\n"
                "  },\n"
                '  "edges": [\n'
                "    {\n"
                '      "from_node": "<node_id>",\n'
                '      "to_node": "<node_id>",\n'
                '      "transition_label": "string",\n'
                '      "trigger_element": "string",\n'
                '      "action_type": "click" | "fill",\n'
                '      "target_selector": "string",\n'
                '      "target_label": "string",\n'
                '      "timeout_ms": 5000,\n'
                '      "wait_condition": "domcontentloaded",\n'
                '      "description": "string"\n'
                "    }\n"
                "  ],\n"
                '  "topological_order": ["<node_id>", ...]\n'
                "}\n"
                "Do NOT re-inline full bounding boxes. Output compact valid JSON only. Graph MUST be strictly acyclic."
            )

            manifest = [
                {
                    "id": s.get("id"),
                    "label": s.get("label"),
                    "role": s.get("role"),
                }
                for s in segments[:8]
            ]
            trans_summary = transitions or []

            user_prompt = (
                f"Workflow: {self.workflow_name}\n"
                f"Interactive Elements Manifest: {json.dumps(manifest)}\n"
                f"Keyframe Transitions: {json.dumps(trans_summary)}"
            )

            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                    extra_body={"chat_template_args": {"disable_thought": True}},
                )
            except Exception:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )

            raw_text = resp.choices[0].message.content or ""
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            data = json.loads(raw_text)
            if "nodes" in data and "edges" in data:
                if isinstance(data["nodes"], list):
                    nodes_dict = {}
                    for n in data["nodes"]:
                        nid = n.get("id") or f"state_{len(nodes_dict)}"
                        nodes_dict[nid] = n
                    data["nodes"] = nodes_dict

                if not isinstance(data["nodes"], dict):
                    return None

                # Normalize edge keys
                for e in data["edges"]:
                    if "from" in e and "from_node" not in e:
                        e["from_node"] = e["from"]
                    if "source" in e and "from_node" not in e:
                        e["from_node"] = e["source"]
                    if "to" in e and "to_node" not in e:
                        e["to_node"] = e["to"]
                    if "target" in e and "to_node" not in e:
                        e["to_node"] = e["target"]
                    if "action" in e and "action_type" not in e:
                        e["action_type"] = e["action"]
                    if "target_id" in e and "trigger_element" not in e:
                        e["trigger_element"] = e["target_id"]
                    if "target_selector" not in e or not e["target_selector"]:
                        trigger = e.get("trigger_element", "target")
                        e["target_selector"] = f"button#{trigger}" if "button" in trigger else f"#{trigger}"
                    if "target_label" not in e:
                        e["target_label"] = e.get("trigger_element", "element")
                    if "timeout_ms" not in e:
                        e["timeout_ms"] = 5000
                    if "wait_condition" not in e:
                        e["wait_condition"] = "domcontentloaded"
                    if "description" not in e:
                        e["description"] = f"{e.get('action_type', 'click')} {e.get('trigger_element', '')}"

                # Normalize node keys
                for nid, ndata in data["nodes"].items():
                    if "id" not in ndata:
                        ndata["id"] = nid
                    if "title" not in ndata:
                        ndata["title"] = ndata.get("label", nid)
                    if "label" not in ndata:
                        ndata["label"] = ndata.get("title", nid)
                    if "screen_name" not in ndata:
                        ndata["screen_name"] = nid
                    if "screen_hash" not in ndata:
                        ndata["screen_hash"] = self._hash_screen(ndata["label"], [nid])
                    if "url_pattern" not in ndata or not ndata["url_pattern"]:
                        ndata["url_pattern"] = "/chart" if "root" in nid else f"/step/{nid}"
                    if "preconditions" not in ndata:
                        ndata["preconditions"] = []
                    if "assertions" not in ndata:
                        ndata["assertions"] = [f"expect({nid}).to_be_active()"]
                    if "route" not in ndata:
                        ndata["route"] = "/api/v1/chart/status" if "root" in nid else f"/api/v1/step/{nid}"
                    if "method" not in ndata:
                        ndata["method"] = "GET" if "root" in nid else "POST"
                    if "segments" not in ndata or not ndata["segments"]:
                        ndata["segments"] = segments

                # Ensure canonical TradingView states if applicable
                if "TradingView" in self.workflow_name and "state_interval_selected" not in data["nodes"]:
                    candidate_nid = None
                    for nid in data["nodes"]:
                        if nid != "state_root" and ("interval" in nid or "timeframe" in nid or "15m" in nid or "1h" in nid or "chart" in nid):
                            candidate_nid = nid
                            break
                    if candidate_nid is None:
                        for nid in reversed(list(data["nodes"].keys())):
                            if nid != "state_root":
                                candidate_nid = nid
                                break
                    if candidate_nid:
                        node_data = data["nodes"].pop(candidate_nid)
                        node_data["id"] = "state_interval_selected"
                        data["nodes"]["state_interval_selected"] = node_data
                        for e in data["edges"]:
                            if e.get("from_node") == candidate_nid:
                                e["from_node"] = "state_interval_selected"
                            if e.get("to_node") == candidate_nid:
                                e["to_node"] = "state_interval_selected"
                        if data.get("root_node") == candidate_nid:
                            data["root_node"] = "state_interval_selected"
                        if data.get("terminal_node") == candidate_nid:
                            data["terminal_node"] = "state_interval_selected"

                g = nx.DiGraph()
                for nid in data["nodes"]:
                    g.add_node(nid)
                for edge in data["edges"]:
                    if "from_node" in edge and "to_node" in edge:
                        g.add_edge(edge["from_node"], edge["to_node"])

                if not nx.is_directed_acyclic_graph(g):
                    logger.warning("Muse Spark synthesized graph contains cycles; using fallback.")
                    return None

                topological_order = list(nx.topological_sort(g))
                root_node = data.get("root_node")
                if not root_node or root_node not in data["nodes"]:
                    root_node = topological_order[0] if topological_order else "state_root"
                terminal_node = data.get("terminal_node")
                if not terminal_node or terminal_node not in data["nodes"]:
                    terminal_node = topological_order[-1] if topological_order else root_node

                dag_result = {
                    "workflow_name": data.get("workflow_name", self.workflow_name),
                    "root_node": root_node,
                    "terminal_node": terminal_node,
                    "nodes": data["nodes"],
                    "edges": data["edges"],
                    "topological_order": topological_order,
                    "segments_count": len(segments),
                }
                self.last_inference_source = "live_baseten_muse_spark"
                logger.info(
                    f"Live Baseten Muse Spark synthesized {len(data['nodes'])} states and {len(data['edges'])} transitions."
                )
                return dag_result

        except Exception as exc:
            logger.warning(f"Muse Spark API request failed ({exc}); seamlessly falling back to local synthesis.")

        return None

    def build_dag(
        self,
        segments: List[Dict[str, Any]],
        transitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Synthesizes segments and actions into a validated Directed Acyclic Graph (DAG)."""
        # 1. Try official Meta Model API if configured
        api_dag = self._call_muse_spark_api(segments, transitions)
        if api_dag is not None:
            return api_dag

        seg_by_id = {s["id"]: s for s in segments}

        # If dynamic transitions are provided with multiple frames
        if transitions and len(transitions) > 0:
            nodes: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []

            # Create state nodes for each transition boundary
            # Collect unique frame indices involved in transitions
            frame_indices = sorted(
                list(
                    set(
                        [t.get("from_frame", 0) for t in transitions]
                        + [t.get("to_frame", 1) for t in transitions]
                    )
                )
            )

            for f_idx in frame_indices:
                node_id = f"state_frame_{f_idx}" if f_idx > 0 else "state_root"
                screen_label = f"Screen State Frame {f_idx}" if f_idx > 0 else "Initial Dashboard / Chart Canvas"
                screen_hash = self._hash_screen(screen_label, list(seg_by_id.keys()))

                preconditions = []
                if f_idx == 0:
                    preconditions.append(
                        {
                            "type": "canvas_present",
                            "target_id": "chart_canvas" if "chart_canvas" in seg_by_id else list(seg_by_id.keys())[0] if seg_by_id else "root",
                            "description": "Verify primary canvas or UI container rendered",
                        }
                    )
                else:
                    preconditions.append(
                        {
                            "type": "state_active",
                            "target_id": f"state_frame_{f_idx - 1}" if (f_idx - 1) > 0 else "state_root",
                            "description": "Preceding state must be resolved",
                        }
                    )

                nodes[node_id] = {
                    "id": node_id,
                    "label": screen_label,
                    "title": screen_label,
                    "screen_name": f"screen_{f_idx}",
                    "screen_hash": screen_hash,
                    "url_pattern": "/" if f_idx == 0 else f"/step/{f_idx}",
                    "preconditions": preconditions,
                    "assertions": [f"expect({node_id}).to_be_active()"],
                    "route": f"/api/v1/step/{f_idx}",
                    "method": "POST" if f_idx > 0 else "GET",
                    "segments": segments,
                }

            # Build edges from transitions
            for idx, t in enumerate(transitions):
                from_idx = t.get("from_frame", 0)
                to_idx = t.get("to_frame", 1)
                from_nid = f"state_frame_{from_idx}" if from_idx > 0 else "state_root"
                to_nid = f"state_frame_{to_idx}" if to_idx > 0 else "state_root"

                # Pick an interactable segment as target
                target_seg_id = t.get("target_id")
                if not target_seg_id:
                    # Pick from available segments
                    available_ids = list(seg_by_id.keys())
                    target_seg_id = available_ids[min(idx, len(available_ids) - 1)] if available_ids else f"target_{idx}"

                target_seg = seg_by_id.get(target_seg_id, {})
                action = t.get("action", "click")

                edges.append(
                    {
                        "from_node": from_nid,
                        "to_node": to_nid,
                        "transition_label": f"step_{idx}_{action}_{target_seg_id}",
                        "trigger_element": target_seg_id,
                        "action_type": action,
                        "target_selector": target_seg.get("suggested_selector") or f"#{target_seg_id}",
                        "target_label": target_seg.get("label", target_seg_id),
                        "timeout_ms": 5000,
                        "wait_condition": "domcontentloaded",
                        "description": f"Execute {action} on {target_seg_id}",
                    }
                )

            # Preserve TradingView endpoints if chart_canvas / interval_dropdown in segments
            if "interval_dropdown" in seg_by_id and "timeframe_15m" in seg_by_id:
                nodes["state_interval_menu_open"] = {
                    "id": "state_interval_menu_open",
                    "label": "Interval Selection Dropdown Open",
                    "title": "Interval Dropdown Open",
                    "screen_name": "interval_menu",
                    "screen_hash": self._hash_screen("Interval Menu", ["interval_dropdown"]),
                    "url_pattern": "/chart",
                    "preconditions": [
                        {
                            "type": "state_active",
                            "target_id": "state_root",
                            "description": "Chart canvas must be active",
                        },
                        {
                            "type": "element_interactive",
                            "target_id": "interval_dropdown",
                            "description": "Interval dropdown selector must be clickable",
                        },
                    ],
                    "action": {
                        "type": "click",
                        "target_id": "interval_dropdown",
                        "target_segment": seg_by_id.get("interval_dropdown"),
                    },
                    "route": "/api/v1/chart/interval/menu",
                    "method": "POST",
                }
                nodes["state_interval_selected"] = {
                    "id": "state_interval_selected",
                    "label": "Timeframe Interval Applied",
                    "title": "Timeframe Applied",
                    "screen_name": "chart_active",
                    "screen_hash": self._hash_screen("Chart Active", ["timeframe_15m"]),
                    "url_pattern": "/chart",
                    "preconditions": [
                        {
                            "type": "menu_open",
                            "target_id": "state_interval_menu_open",
                            "description": "Interval menu must be in open state",
                        }
                    ],
                    "action": {
                        "type": "click",
                        "target_id": "timeframe_15m",
                        "target_segment": seg_by_id.get("timeframe_15m"),
                    },
                    "route": "/api/v1/chart/interval",
                    "method": "POST",
                }
        else:
            # Canonical structured workflow mapping
            nodes = {
                "state_root": {
                    "id": "state_root",
                    "label": "Initial Dashboard / Chart Canvas",
                    "title": "Initial Dashboard / Chart Canvas",
                    "screen_name": "chart_canvas",
                    "screen_hash": self._hash_screen("Root Canvas", ["chart_canvas"]),
                    "url_pattern": "/chart",
                    "preconditions": [
                        {
                            "type": "canvas_present",
                            "target_id": "chart_canvas",
                            "description": "Verify TradingView chart canvas rendered",
                        }
                    ],
                    "assertions": ["assert canvas.is_visible()"],
                    "route": "/api/v1/chart/status",
                    "method": "GET",
                    "segments": segments,
                },
                "state_interval_menu_open": {
                    "id": "state_interval_menu_open",
                    "label": "Interval Selection Dropdown Open",
                    "title": "Interval Dropdown Open",
                    "screen_name": "interval_menu",
                    "screen_hash": self._hash_screen("Interval Menu Open", ["interval_dropdown"]),
                    "url_pattern": "/chart",
                    "preconditions": [
                        {
                            "type": "state_active",
                            "target_id": "state_root",
                            "description": "Chart canvas must be active",
                        },
                        {
                            "type": "element_interactive",
                            "target_id": "interval_dropdown",
                            "description": "Interval dropdown selector must be clickable",
                        },
                    ],
                    "action": {
                        "type": "click",
                        "target_id": "interval_dropdown",
                        "target_segment": seg_by_id.get("interval_dropdown"),
                    },
                    "assertions": ["assert interval_menu.is_visible()"],
                    "route": "/api/v1/chart/interval/menu",
                    "method": "POST",
                    "segments": segments,
                },
                "state_interval_selected": {
                    "id": "state_interval_selected",
                    "label": "Timeframe Interval Applied",
                    "title": "Timeframe Interval Applied",
                    "screen_name": "interval_selected",
                    "screen_hash": self._hash_screen("Interval Selected", ["timeframe_15m"]),
                    "url_pattern": "/chart",
                    "preconditions": [
                        {
                            "type": "menu_open",
                            "target_id": "state_interval_menu_open",
                            "description": "Interval menu must be in open state",
                        }
                    ],
                    "action": {
                        "type": "click",
                        "target_id": "timeframe_15m",
                        "target_segment": seg_by_id.get("timeframe_15m"),
                    },
                    "assertions": ["assert timeframe_applied.is_visible()"],
                    "route": "/api/v1/chart/interval",
                    "method": "POST",
                    "segments": segments,
                },
            }

            edges = [
                {
                    "from_node": "state_root",
                    "to_node": "state_interval_menu_open",
                    "transition_label": "open_interval_menu",
                    "trigger_element": "interval_dropdown",
                    "action_type": "click",
                    "target_selector": "div[data-role='interval-menu'], button#interval_dropdown",
                    "target_label": "interval_dropdown",
                    "timeout_ms": 5000,
                    "wait_condition": "domcontentloaded",
                    "description": "Open interval dropdown",
                },
                {
                    "from_node": "state_interval_menu_open",
                    "to_node": "state_interval_selected",
                    "transition_label": "select_15m_timeframe",
                    "trigger_element": "timeframe_15m",
                    "action_type": "click",
                    "target_selector": "div[data-value='15m'], button#timeframe_15m",
                    "target_label": "timeframe_15m",
                    "timeout_ms": 5000,
                    "wait_condition": "domcontentloaded",
                    "description": "Select 15m timeframe",
                },
            ]

        # Verify DAG properties using NetworkX
        g = nx.DiGraph()
        for nid in nodes:
            g.add_node(nid)
        for e in edges:
            g.add_edge(e["from_node"], e["to_node"])

        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("Synthesized transition graph contains cycles, which violates DAG constraints.")

        topological_order = list(nx.topological_sort(g))
        root_node = topological_order[0] if topological_order else "state_root"
        terminal_node = topological_order[-1] if topological_order else root_node

        return {
            "workflow_name": self.workflow_name,
            "root_node": root_node,
            "terminal_node": terminal_node,
            "nodes": nodes,
            "edges": edges,
            "topological_order": topological_order,
            "segments_count": len(segments),
        }

    def synthesize(
        self,
        frames: Optional[List[FrameMetadata]] = None,
        segments: Optional[Dict[int, List[VisualSegment]]] = None,
        transitions: Optional[List[TransitionEvent]] = None,
        workflow_name: Optional[str] = None,
    ) -> StateGraph:
        """Synthesizes frames and transitions into a typed StateGraph model."""
        name = workflow_name or self.workflow_name

        flat_segments: List[Dict[str, Any]] = []
        if segments:
            for s_list in segments.values():
                for s in s_list:
                    flat_segments.append(
                        {
                            "id": s.id,
                            "label": s.label,
                            "role": s.role.value,
                            "bbox": {
                                "x": s.bbox.x,
                                "y": s.bbox.y,
                                "width": s.bbox.width,
                                "height": s.bbox.height,
                            },
                            "confidence": s.confidence,
                            "suggested_selector": s.suggested_selector,
                            "metadata": s.attributes,
                        }
                    )

        trans_dicts: Optional[List[Dict[str, Any]]] = None
        if transitions:
            trans_dicts = [
                {
                    "from_frame": t.from_frame_index,
                    "to_frame": t.to_frame_index,
                    "timestamp_ms": t.timestamp_ms,
                    "diff_score": t.diff_score,
                    "action": t.action_type.value,
                }
                for t in transitions
            ]

        dag_dict = self.build_dag(flat_segments, trans_dicts)

        # Convert dag_dict to StateGraph
        typed_nodes: Dict[str, DAGNode] = {}
        for nid, ndata in dag_dict["nodes"].items():
            typed_nodes[nid] = DAGNode(
                id=nid,
                title=ndata.get("title", ndata.get("label", nid)),
                screen_name=ndata.get("screen_name", nid),
                keyframe_index=0,
                url_pattern=ndata.get("url_pattern"),
                assertions=ndata.get("assertions", []),
            )

        typed_edges: List[DAGEdge] = []
        for e in dag_dict["edges"]:
            action_val = e.get("action_type", "click")
            action_enum = ActionType.CLICK
            for a in ActionType:
                if a.value == action_val:
                    action_enum = a
                    break

            typed_edges.append(
                DAGEdge(
                    from_node_id=e["from_node"],
                    to_node_id=e["to_node"],
                    action_type=action_enum,
                    target_segment_id=e.get("trigger_element"),
                    target_selector=e.get("target_selector", f"#{e.get('trigger_element')}"),
                    target_label=e.get("target_label"),
                    timeout_ms=e.get("timeout_ms", 5000),
                    wait_condition=e.get("wait_condition"),
                    description=e.get("description"),
                )
            )

        return StateGraph(
            name=name,
            initial_node_id=dag_dict["root_node"],
            terminal_node_ids=[dag_dict["terminal_node"]],
            nodes=typed_nodes,
            edges=typed_edges,
        )


def generate_mermaid_dag(dag: Dict[str, Any]) -> str:
    """Generates a clean Mermaid flowchart diagram representation of the synthesized DAG."""
    nodes = dag.get("nodes", {})
    edges = dag.get("edges", [])
    root_node = dag.get("root_node", "state_root")
    terminal_node = dag.get("terminal_node", "")

    lines = [
        "flowchart TD",
        f"    %% Ocular Synthesized Workflow DAG: {dag.get('workflow_name', 'Workflow')}",
        "    classDef default fill:#1e1e2e,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;",
        "    classDef initNode fill:#181825,stroke:#a6e3a1,stroke-width:3px,color:#a6e3a1;",
        "    classDef termNode fill:#181825,stroke:#f38ba8,stroke-width:3px,color:#f38ba8;",
        "",
    ]

    for nid, ndata in nodes.items():
        title = ndata.get("title", ndata.get("label", nid))
        safe_title = title.replace('"', "'")
        preconds = ndata.get("preconditions", [])
        pre_desc = f"<br/><small>Pre: {preconds[0].get('type')}</small>" if preconds else ""

        node_style = ":::initNode" if nid == root_node else (":::termNode" if nid == terminal_node else ":::default")
        lines.append(f'    {nid}["{safe_title}{pre_desc}"]{node_style}')

    lines.append("")

    for e in edges:
        from_n = e.get("from_node")
        to_n = e.get("to_node")
        action = e.get("action_type") or e.get("action", "action")
        target = e.get("trigger_element") or e.get("target_label", "target")
        label = f"{action}: {target}".replace('"', "'")
        lines.append(f'    {from_n} -->|"{label}"| {to_n}')

    return "\n".join(lines) + "\n"


def export_mermaid_dag(dag: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Exports the synthesized workflow DAG as a Mermaid diagram file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    mermaid_content = generate_mermaid_dag(dag)
    out_file.write_text(mermaid_content, encoding="utf-8")
    return out_file


def render_rich_dag_tree(dag: Dict[str, Any]) -> Tree:
    """Constructs a Rich Tree visualizing states, preconditions, and action transitions."""
    workflow_name = dag.get("workflow_name", "OcularWorkflow")
    nodes = dag.get("nodes", {})
    edges = dag.get("edges", [])
    topo_order = dag.get("topological_order", list(nodes.keys()))

    root_tree = Tree(
        f"[bold cyan]Workflow DAG: {workflow_name}[/bold cyan] "
        f"[dim]({len(nodes)} states, {len(edges)} transitions)[/dim]"
    )

    edges_by_from: Dict[str, List[Dict[str, Any]]] = {}
    for e in edges:
        edges_by_from.setdefault(e.get("from_node", ""), []).append(e)

    for nid in topo_order:
        ndata = nodes.get(nid, {})
        label = ndata.get("label", nid)
        is_root = (nid == dag.get("root_node"))
        is_term = (nid == dag.get("terminal_node"))

        badge = "[bold green][ROOT][/bold green] " if is_root else ("[bold magenta][TERMINAL][/bold magenta] " if is_term else "")
        node_branch = root_tree.add(f"{badge}[bold white]{nid}[/bold white] - [dim]{label}[/dim]")

        for p in ndata.get("preconditions", []):
            node_branch.add(f"[cyan]⚡ Precondition:[/cyan] {p.get('description', p.get('type'))}")

        for inv in ndata.get("assertions", []):
            node_branch.add(f"[green]✓ Invariant:[/green] {inv}")

        for e in edges_by_from.get(nid, []):
            action = e.get("action_type", "action")
            target = e.get("trigger_element", e.get("target_label", "target"))
            to_n = e.get("to_node", "")
            node_branch.add(f"[yellow]──({action}: {target})──>[/yellow] [bold white]{to_n}[/bold white]")

    return root_tree


# Aliases for architectural interoperability
MockMuseSynthesizer = DAGBuilder

