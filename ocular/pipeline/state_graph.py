"""Muse Spark: State Transition Graph (DAG) Synthesizer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import networkx as nx

from ocular.pipeline.base import BaseSynthesizer


class DAGBuilder(BaseSynthesizer):
    """Establishes pre-condition checks and route dependency graphs for UI automation."""

    def __init__(self, workflow_name: str = "TradingViewChartWorkflow") -> None:
        self.workflow_name = workflow_name

    def build_dag(
        self,
        segments: List[Dict[str, Any]],
        transitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Synthesizes segments and actions into a validated Directed Acyclic Graph (DAG)."""
        seg_by_id = {s["id"]: s for s in segments}

        # Define route endpoints mapped to visual states and actions
        nodes: Dict[str, Dict[str, Any]] = {
            "state_root": {
                "id": "state_root",
                "label": "Initial Dashboard / Chart Canvas",
                "preconditions": [
                    {
                        "type": "canvas_present",
                        "target_id": "chart_canvas",
                        "description": "Verify TradingView chart canvas rendered",
                    }
                ],
                "route": "/api/v1/chart/status",
                "method": "GET",
            },
            "state_interval_menu_open": {
                "id": "state_interval_menu_open",
                "label": "Interval Selection Dropdown Open",
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
            },
            "state_interval_selected": {
                "id": "state_interval_selected",
                "label": "Timeframe Interval Applied",
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
            },
        }

        # Define directed dependency edges
        edges = [
            {
                "from_node": "state_root",
                "to_node": "state_interval_menu_open",
                "transition_label": "open_interval_menu",
                "trigger_element": "interval_dropdown",
            },
            {
                "from_node": "state_interval_menu_open",
                "to_node": "state_interval_selected",
                "transition_label": "select_15m_timeframe",
                "trigger_element": "timeframe_15m",
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

        return {
            "workflow_name": self.workflow_name,
            "root_node": "state_root",
            "terminal_node": "state_interval_selected",
            "nodes": nodes,
            "edges": edges,
            "topological_order": topological_order,
            "segments_count": len(segments),
        }
