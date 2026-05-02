"""Context pack assembly tests — RAG + graph + SSM → context pack.

Contract:
  - Context pack has: alert, ssm_state, rag_context, graph_links
  - Agents return AgentResult with output dict
  - Context pack assembly degrades gracefully when backends unavailable
  - Sentinel triage agent assembles pack from sub-agents
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.base import AgentRegistry, AgentResult
from cell.agents.rag_agent import RAGLookupAgent, RAGSearchAgent
from cell.agents.graph_agent import GraphLookupAgent, GraphNeighborsAgent, GraphStatsAgent
from cell.agents.ssm_agent import SSMGetStateAgent
from cell.agents.sentinel_agent import SentinelTriageAgent


def _make_registry():
    """Create a registry with all read agents."""
    reg = AgentRegistry()
    reg.register(RAGLookupAgent())
    reg.register(RAGSearchAgent())
    reg.register(GraphLookupAgent())
    reg.register(GraphNeighborsAgent())
    reg.register(GraphStatsAgent())
    reg.register(SSMGetStateAgent())
    return reg


def test_context_pack_structure():
    """Context pack has the required four keys."""
    sentinel = SentinelTriageAgent()
    sentinel._agent_registry = _make_registry()

    pack = sentinel._assemble_context("test alert", entity_id="IP:1.2.3.4")
    assert "alert" in pack
    assert "ssm_state" in pack
    assert "rag_context" in pack
    assert "graph_links" in pack
    assert pack["alert"]["text"] == "test alert"


def test_context_pack_without_entity():
    """Context pack works without entity_id (SSM/graph skip)."""
    sentinel = SentinelTriageAgent()
    sentinel._agent_registry = _make_registry()

    pack = sentinel._assemble_context("generic alert")
    assert "alert" in pack
    assert pack["ssm_state"] == {}  # No entity → no SSM lookup
    assert pack["graph_links"] == []  # No entity → no graph lookup


def test_context_pack_without_registry():
    """Context pack degrades gracefully without agent registry."""
    sentinel = SentinelTriageAgent()
    sentinel._agent_registry = None

    pack = sentinel._assemble_context("test alert", entity_id="IP:1.2.3.4")
    assert "alert" in pack
    assert pack["ssm_state"] == {}
    assert pack["rag_context"] == []
    assert pack["graph_links"] == []


def test_rag_agent_returns_structured_output():
    """RAG agent returns output with results/count/source keys."""
    agent = RAGLookupAgent()
    # This will use file_search fallback if FGIP DB is not available
    result = agent.run({"query": "sentinel"})
    assert result.ok or result.error  # Either works or fails cleanly
    if result.ok:
        assert "results" in result.output
        assert "count" in result.output
        assert "source" in result.output


def test_graph_agent_graceful_on_missing_db():
    """Graph agent returns error when DB is missing, doesn't crash."""
    agent = GraphLookupAgent()
    # If FGIP DB doesn't exist, should return error
    if not os.path.exists(os.path.expanduser("~/fgip-engine/fgip.db")):
        result = agent.run({"entity": "test"})
        assert not result.ok
        assert "not available" in result.error
    else:
        result = agent.run({"entity": "test"})
        assert result.ok
        assert "found" in result.output


def test_ssm_agent_graceful_on_missing_db():
    """SSM agent returns empty state when DB is missing, doesn't crash."""
    agent = SSMGetStateAgent()
    result = agent.run({"entity_id": "IP:1.2.3.4"})
    assert result.ok  # Should succeed even without DB (returns empty state)
    assert "entity" in result.output


def test_graph_stats_returns_counts():
    """Graph stats agent returns node/edge/claim counts."""
    agent = GraphStatsAgent()
    result = agent.run({})
    if os.path.exists(os.path.expanduser("~/fgip-engine/fgip.db")):
        assert result.ok
        assert "nodes" in result.output
        assert "edges" in result.output
        assert "claims" in result.output
    else:
        assert not result.ok  # Expected: DB not available


def test_receipt_in_agent_result():
    """Every agent run attaches a receipt."""
    agent = RAGLookupAgent()
    result = agent.run({"query": "test"})
    assert result.receipt is not None
    assert "agent" in result.receipt
    assert result.receipt["agent"] == "rag_lookup"
    assert "wall_time_s" in result.receipt
    assert "timestamp" in result.receipt


def test_sentinel_triage_without_orchestrator():
    """Sentinel triage returns 'unknown' verdict without orchestrator."""
    sentinel = SentinelTriageAgent()
    sentinel._agent_registry = _make_registry()
    sentinel._orchestrator = None

    result = sentinel.run({"alert_text": "test alert"})
    assert result.ok
    assert result.output["verdict"]["severity"] == "unknown"
    assert "context_pack" in result.output


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
