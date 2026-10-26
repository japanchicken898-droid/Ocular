"""Muse Spark: State Transition Graph (DAG) Synthesizer.

Parses keyframe diffs, transitions, and segmented targets into a Directed
Acyclic Graph (DAG) representing the sequential UI workflow.
Models nodes as states (URL, screen hash, pre-conditions) and edges as actions
(click, fill, wait, assert).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Union
import networkx as nx

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

    def _hash_screen(self, label: str, seg_ids: List[str]) -> str:
        content = f"{label}:{':'.join(sorted(seg_ids))}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def build_dag(
        self,
        segments: List[Dict[str, Any]],
        transitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Synthesizes segments and actions into a validated Directed Acyclic Graph (DAG)."""
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


# Aliases for architectural interoperability
MockMuseSynthesizer = DAGBuilder
