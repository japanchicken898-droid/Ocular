import json
from pathlib import Path
import pytest
from ocular.telemetry import coerce_telemetry_event, load_telemetry_events
from ocular.security.netguard import canonicalize_url, is_private_url, host_is_blocked
from ocular.pipeline.state_graph import DAGBuilder, CompilationError
from ocular.models.types import DAGNode, DAGEdge

CORPUS_DIR = Path(__file__).parent / "fuzz" / "corpus"


def test_fuzz_telemetry_coercion():
    corpus_file = CORPUS_DIR / "malformed_telemetry.json"
    with open(corpus_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    for item in events:
        # coerce_telemetry_event must never raise an unhandled exception
        result = coerce_telemetry_event(item)
        if result is not None:
            assert isinstance(result, dict)
            assert 0.0 <= result.get("norm_x", 0.0) <= 1.0
            assert 0.0 <= result.get("norm_y", 0.0) <= 1.0
            assert result.get("elapsed_ms", 0.0) >= 0.0


@pytest.mark.parametrize("bad_input", [
    {"norm_x": float("nan"), "norm_y": float("inf")},
    {"norm_x": -99999.0, "norm_y": 99999.0},
    {"elapsed_ms": float("nan")},
    {"elapsed_ms": -1000},
    {"key": "A" * 50000},
    None,
    12345,
    "string_event",
    [],
])
def test_fuzz_telemetry_mutations(bad_input):
    result = coerce_telemetry_event(bad_input)
    if result is not None:
        assert isinstance(result, dict)
        assert 0.0 <= result.get("norm_x", 0.0) <= 1.0
        assert 0.0 <= result.get("norm_y", 0.0) <= 1.0


def test_fuzz_hostile_urls():
    corpus_file = CORPUS_DIR / "hostile_urls.txt"
    with open(corpus_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    for raw_url in urls:
        # Every hostile URL must either fail canonicalization (ValueError)
        # or be identified as private/blocked by netguard.
        try:
            clean_url = canonicalize_url(raw_url)
            # If canonicalization passed, netguard must detect it as private/blocked
            assert is_private_url(clean_url) is True, f"Hostile URL was not blocked: {raw_url} -> {clean_url}"
        except ValueError:
            # Expected rejection by canonicalize_url
            pass


def test_fuzz_hostile_dag():
    corpus_file = CORPUS_DIR / "hostile_dag.json"
    with open(corpus_file, "r", encoding="utf-8") as f:
        dags = json.load(f)

    for hostile_dag in dags:
        # Hostile DAGs must fail validation when attempting to validate nodes or edges
        has_error = False
        try:
            for nid, ndata in hostile_dag.get("nodes", {}).items():
                DAGNode.model_validate(ndata)
            for edge in hostile_dag.get("edges", []):
                DAGEdge.model_validate(edge)
                t = edge.get("timeout_ms", 5000)
                if not (0 < t <= 120000):
                    raise CompilationError("Invalid timeout")
        except Exception:
            has_error = True

        assert has_error is True, f"Hostile DAG unexpectedly passed validation: {hostile_dag}"
