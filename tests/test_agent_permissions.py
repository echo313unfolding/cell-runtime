"""Agent permission tests — enforce the permission model.

Contract:
  - Read agents: auto-approve, no side effects
  - Write agents: auto-approve with log
  - Privileged actions: require ask-pass
  - Every agent has name, permission, input_schema
  - Unknown agents return error
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.base import AgentBase, AgentResult, AgentRegistry, Permission
from cell.agents.rag_agent import RAGLookupAgent, RAGSearchAgent
from cell.agents.graph_agent import GraphLookupAgent, GraphNeighborsAgent, GraphStatsAgent
from cell.agents.ssm_agent import SSMGetStateAgent, SSMUpdateEventAgent
from cell.agents.sentinel_agent import SentinelTriageAgent
from cell.agents.policy_agent import GateDecideAgent
from cell.agents.receipt_agent import ReceiptLookupAgent, ReceiptWriteAgent


# All agent classes
ALL_AGENTS = [
    RAGLookupAgent, RAGSearchAgent,
    GraphLookupAgent, GraphNeighborsAgent, GraphStatsAgent,
    SSMGetStateAgent, SSMUpdateEventAgent,
    SentinelTriageAgent,
    GateDecideAgent,
    ReceiptLookupAgent, ReceiptWriteAgent,
]

READ_AGENTS = [
    RAGLookupAgent, RAGSearchAgent,
    GraphLookupAgent, GraphNeighborsAgent, GraphStatsAgent,
    SSMGetStateAgent,
    SentinelTriageAgent,
    GateDecideAgent,
    ReceiptLookupAgent,
]

WRITE_AGENTS = [
    SSMUpdateEventAgent,
    ReceiptWriteAgent,
]


def test_all_agents_have_names():
    """Every agent has a non-empty name."""
    for cls in ALL_AGENTS:
        agent = cls()
        assert agent.name, f"{cls.__name__} has no name"


def test_all_agents_have_permission():
    """Every agent has a valid permission level."""
    for cls in ALL_AGENTS:
        agent = cls()
        assert isinstance(agent.permission, Permission), f"{cls.__name__} has invalid permission"


def test_read_agents_are_read():
    """Read agents have Permission.READ."""
    for cls in READ_AGENTS:
        agent = cls()
        assert agent.permission == Permission.READ, (
            f"{cls.__name__} should be READ, got {agent.permission}")


def test_write_agents_are_write():
    """Write agents have Permission.WRITE."""
    for cls in WRITE_AGENTS:
        agent = cls()
        assert agent.permission == Permission.WRITE, (
            f"{cls.__name__} should be WRITE, got {agent.permission}")


def test_no_privileged_agents():
    """No agents are PRIVILEGED (those stay in tool_registry.py)."""
    for cls in ALL_AGENTS:
        agent = cls()
        assert agent.permission != Permission.PRIVILEGED, (
            f"{cls.__name__} is PRIVILEGED — privileged actions belong in tool_registry.py")


def test_all_agents_have_input_schema():
    """Every agent defines an input schema."""
    for cls in ALL_AGENTS:
        agent = cls()
        assert isinstance(agent.input_schema, dict), f"{cls.__name__} has no input_schema"


def test_registry_unknown_agent_returns_error():
    """Unknown agent name returns error, doesn't crash."""
    registry = AgentRegistry()
    result = registry.run("nonexistent_agent", {})
    assert not result.ok
    assert "Unknown agent" in result.error


def test_registry_validates_input():
    """Registry validates required fields before executing."""
    registry = AgentRegistry()
    registry.register(RAGLookupAgent())
    # Missing required "query" field
    result = registry.run("rag_lookup", {})
    assert not result.ok
    assert "Missing required" in result.error


def test_registry_list_agents():
    """Registry lists all registered agents with metadata."""
    registry = AgentRegistry()
    for cls in ALL_AGENTS:
        registry.register(cls())
    listing = registry.list_agents()
    assert len(listing) == len(ALL_AGENTS)
    for entry in listing:
        assert "name" in entry
        assert "permission" in entry
        assert "description" in entry


def test_agent_names_are_unique():
    """No two agents share a name."""
    names = [cls().name for cls in ALL_AGENTS]
    assert len(names) == len(set(names)), f"Duplicate agent names: {names}"


def test_agent_result_ok():
    """AgentResult.ok is True when no error."""
    assert AgentResult(output={"x": 1}).ok
    assert not AgentResult(error="fail").ok


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
