"""Tests for Muse Spark State Transition Graph (DAG) stage."""

import pytest
import networkx as nx
from ocular.pipeline.state_graph import DAGBuilder
from ocular.pipeline.segmenter import MockSAMSegmenter
from ocular.pipeline.base import BaseSynthesizer


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
