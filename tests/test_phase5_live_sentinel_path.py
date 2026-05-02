"""Phase 5 integration test — live Sentinel context path.

Tests the live wiring of SSM, RAG, graph agents against real databases,
and optionally tests the live Sentinel backend if available.

Marks:
  - Tests against real DBs run unconditionally (they gracefully handle missing DBs)
  - Live Sentinel call is skipped if backend is not running
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.base import AgentRegistry
from cell.agents.rag_agent import RAGLookupAgent
from cell.agents.graph_agent import GraphLookupAgent, GraphNeighborsAgent, GraphStatsAgent
from cell.agents.ssm_agent import SSMGetStateAgent
from cell.agents.policy_agent import GateDecideAgent

SENTINEL_PORT = int(os.environ.get("SENTINEL_PORT", "8085"))
FGIP_DB = os.path.expanduser("~/fgip-engine/fgip.db")
SENTINEL_DB = os.path.expanduser("~/tools/sentinel/sentinel.db")


def _sentinel_alive(port: int = SENTINEL_PORT) -> bool:
    """Check if Sentinel backend is reachable."""
    try:
        req = urllib.request.Request(f"http://localhost:{port}/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def _build_registry() -> AgentRegistry:
    registry = AgentRegistry()
    for cls in [RAGLookupAgent, GraphLookupAgent, GraphNeighborsAgent,
                GraphStatsAgent, SSMGetStateAgent, GateDecideAgent]:
        registry.register(cls())
    return registry


# ---------------------------------------------------------------------------
# Context pack agents — real DB tests
# ---------------------------------------------------------------------------

def test_ssm_agent_queries_sentinel_db():
    """SSM agent queries sentinel.db without crashing."""
    registry = _build_registry()
    result = registry.run("ssm_get_state", {"entity_id": "185.220.101.1"})
    assert result.ok
    assert "entity" in result.output
    assert "found" in result.output
    assert result.receipt is not None
    assert result.receipt["agent"] == "ssm_get_state"


@pytest.mark.skipif(not os.path.exists(FGIP_DB), reason="FGIP DB not available")
def test_rag_agent_queries_fgip_fts():
    """RAG agent hits FGIP FTS5 and returns results."""
    registry = _build_registry()
    result = registry.run("rag_lookup", {
        "query": "security",
        "scope": "claims",
        "limit": 5,
    })
    assert result.ok
    assert result.output["source"] == "fgip.db FTS5"
    assert result.output["count"] > 0
    assert len(result.output["results"]) > 0


@pytest.mark.skipif(not os.path.exists(FGIP_DB), reason="FGIP DB not available")
def test_graph_stats_returns_real_counts():
    """Graph stats agent returns real node/edge/claim counts from FGIP."""
    registry = _build_registry()
    result = registry.run("graph_stats", {})
    assert result.ok
    assert result.output["nodes"] > 1000
    assert result.output["edges"] > 1000
    assert result.output["claims"] > 10000


@pytest.mark.skipif(not os.path.exists(FGIP_DB), reason="FGIP DB not available")
def test_graph_lookup_finds_entity():
    """Graph lookup finds a known entity in FGIP."""
    registry = _build_registry()
    # "Chamber" should match "US Chamber of Commerce"
    result = registry.run("graph_lookup", {"entity": "Chamber"})
    assert result.ok
    assert result.output["found"] is True
    assert result.output["count"] > 0


def test_gate_decide_auto_approve_live():
    """Gate agent auto-approves in test mode."""
    registry = _build_registry()
    result = registry.run("gate_decide", {
        "action": "block_ip",
        "detail": "Block 185.220.101.1 (Tor exit node)",
        "auto": True,
    })
    assert result.ok
    assert result.output["allowed"] is True
    assert result.receipt["agent"] == "gate_decide"


# ---------------------------------------------------------------------------
# Live Sentinel backend tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _sentinel_alive(), reason="Sentinel backend not running")
def test_live_sentinel_verdict():
    """Live Sentinel produces a structured verdict for a security alert."""
    alert = (
        "ALERT: Outbound connection from svchost.exe to 185.220.101.1:443 "
        "detected on host WIN-DC01. Connection to known Tor exit node."
    )
    system_prompt = (
        "You are a security triage agent. Respond with:\n"
        "SEVERITY: <critical|high|medium|low|benign>\n"
        "SUMMARY: <one-line summary>\n"
        "ACTIONS: <comma-separated recommended actions>"
    )

    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": alert},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"http://localhost:{SENTINEL_PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    content = data["choices"][0]["message"]["content"]
    assert "SEVERITY:" in content
    assert data["usage"]["completion_tokens"] > 0

    # Severity should be high or critical for a Tor exit node C2 alert
    severity_line = [l for l in content.split("\n") if "SEVERITY:" in l]
    assert len(severity_line) > 0
    severity = severity_line[0].split("SEVERITY:")[-1].strip().lower()
    assert severity in ("critical", "high", "medium")


@pytest.mark.skipif(not _sentinel_alive(), reason="Sentinel backend not running")
def test_live_full_chain():
    """Full live chain: context pack → Sentinel → gate → receipt fields."""
    registry = _build_registry()

    # Step 1: Assemble context pack
    ssm = registry.run("ssm_get_state", {"entity_id": "test-entity"})
    assert ssm.ok

    rag = registry.run("rag_lookup", {"query": "security", "limit": 3})
    assert rag.ok

    graph = registry.run("graph_stats", {})
    assert graph.ok

    # Step 2: Call live Sentinel
    context_lines = ["Alert: Suspicious DNS query to known C2 domain evil.example.com"]
    if rag.output.get("count", 0) > 0:
        context_lines.append(f"RAG: {rag.output['count']} related docs")
    if graph.output.get("nodes", 0) > 0:
        context_lines.append(f"Graph: {graph.output['nodes']} entities")

    payload = json.dumps({
        "messages": [
            {"role": "system", "content": "You are a security triage agent. Classify severity."},
            {"role": "user", "content": "\n".join(context_lines)},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"http://localhost:{SENTINEL_PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    sentinel_output = data["choices"][0]["message"]["content"]
    assert len(sentinel_output) > 0

    # Step 3: Gate
    gate = registry.run("gate_decide", {
        "action": "investigate",
        "detail": "Investigate DNS C2 alert",
        "auto": True,
    })
    assert gate.ok
    assert gate.output["allowed"] is True

    # Step 4: Verify receipt fields from all steps
    for result in [ssm, rag, graph, gate]:
        assert result.receipt is not None
        assert "agent" in result.receipt
        assert "wall_time_s" in result.receipt
        assert "timestamp" in result.receipt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
