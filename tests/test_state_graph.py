"""Tests for Muse Spark State Transition Graph (DAG) stage."""

import networkx as nx
import numpy as np
import pytest
from ocular.models.types import StateGraph
from ocular.pipeline.base import BaseSynthesizer
from ocular.pipeline.segmenter import MockSAMSegmenter
from ocular.pipeline.state_graph import DAGBuilder, MockMuseSynthesizer


def test_dag_builder():
    builder = DAGBuilder(workflow_name="TradingViewChartWorkflow")
    assert isinstance(builder, BaseSynthesizer)

    segmenter = MockSAMSegmenter()
    segments = segmenter.segment()

    dag = builder.build_dag(segments=segments)

    assert dag["workflow_name"] == "TradingViewChartWorkflow"
    assert dag["root_node"] == "state_root"
    assert "state_interval_selected" in dag["nodes"]

    # Verify NetworkX DAG integrity
    g = nx.DiGraph()
    for nid in dag["nodes"]:
        g.add_node(nid)
    for edge in dag["edges"]:
        g.add_edge(edge["from_node"], edge["to_node"])

    assert nx.is_directed_acyclic_graph(g)
    assert len(dag["topological_order"]) == len(dag["nodes"])


def test_dag_builder_with_dynamic_transitions():
    builder = DAGBuilder(workflow_name="DynamicFormWorkflow")
    segments = [
        {"id": "username_input", "label": "username", "role": "input", "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.05}},
        {"id": "submit_button", "label": "submit", "role": "button", "bbox": {"x": 0.1, "y": 0.3, "width": 0.2, "height": 0.05}},
    ]
    transitions = [
        {"from_frame": 0, "to_frame": 1, "timestamp_ms": 500.0, "diff_score": 0.15, "action": "fill", "target_id": "username_input"},
        {"from_frame": 1, "to_frame": 2, "timestamp_ms": 1200.0, "diff_score": 0.25, "action": "click", "target_id": "submit_button"},
    ]

    dag = builder.build_dag(segments=segments, transitions=transitions)
    assert len(dag["nodes"]) >= 3
    assert len(dag["edges"]) >= 2
    assert "screen_hash" in dag["nodes"]["state_root"]
    assert len(dag["topological_order"]) == len(dag["nodes"])


def test_mock_muse_synthesizer_typed_model():
    synthesizer = MockMuseSynthesizer(workflow_name="TypedWorkflow")
    segmenter = MockSAMSegmenter()
    segments_batch = segmenter.segment_batch([np.zeros((50, 50, 3), dtype=np.uint8)])

    graph = synthesizer.synthesize(segments=segments_batch, workflow_name="TypedWorkflow")
    assert isinstance(graph, StateGraph)
    assert graph.name == "TypedWorkflow"
    assert graph.initial_node_id in graph.nodes
    linear_path = graph.get_linear_path()
    assert isinstance(linear_path, list)
