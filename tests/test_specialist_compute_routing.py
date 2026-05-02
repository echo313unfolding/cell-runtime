"""Tests for specialist compute routing — cartridge first, shard second.

Contract:
  - Routes to cartridge when cartridge matches intent
  - Falls through to shard when no cartridge matches
  - Falls back to Sentinel when neither matches
  - prefer_shard overrides cartridge-first
  - All agents have READ permission
  - Registered in tool registry
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.base import Permission
from cell.agents.specialist_compute_agent import (
    SpecialistComputeRouteAgent, ShardListAgent, ShardResourceCheckAgent,
    get_shard_pool, reset_shard_pool,
)
from cell.agents.cartridge_agent import reset_pool as reset_cartridge_pool
from cell.tool_registry import create_default_registry


def setup_function():
    """Reset pools before each test."""
    reset_shard_pool()
    reset_cartridge_pool()


def test_all_compute_agents_read():
    """All specialist compute agents are READ."""
    for cls in [SpecialistComputeRouteAgent, ShardListAgent, ShardResourceCheckAgent]:
        assert cls().permission == Permission.READ


def test_compute_agents_registered():
    """Compute agents are in tool registry."""
    reg = create_default_registry()
    assert reg.has("specialist_compute_route")
    assert reg.has("shard_list")
    assert reg.has("shard_resource_check")


def test_route_to_cartridge_first():
    """When cartridge matches intent, route to cartridge (not shard)."""
    agent = SpecialistComputeRouteAgent()
    result = agent.run({"intent": "code_repair", "task": "Fix parser error"})
    assert result.ok
    assert result.output["route"] == "cartridge"
    assert "cartridge_id" in result.output


def test_route_to_shard_when_no_cartridge():
    """When no cartridge matches, fall through to shard pool."""
    agent = SpecialistComputeRouteAgent()
    # deep_code_analysis is a shard intent, not a cartridge intent
    result = agent.run({"intent": "deep_code_analysis", "task": "Analyze codebase"})
    assert result.ok
    # If shard exists and is active, route=shard. If candidate, route=fallback.
    assert result.output["route"] in ("shard", "fallback")


def test_fallback_when_neither_matches():
    """When neither cartridge nor shard matches, fall back to Sentinel."""
    agent = SpecialistComputeRouteAgent()
    result = agent.run({"intent": "completely_unknown_intent", "task": "Do something"})
    assert result.ok
    assert result.output["route"] == "fallback"
    assert "sentinel" in result.output["model_id"].lower() or "sentinel" in result.output.get("reason", "").lower()


def test_shard_list_returns_shards():
    """shard_list returns metadata."""
    agent = ShardListAgent()
    result = agent.run({})
    assert result.ok
    assert "shards" in result.output
    assert "count" in result.output


def test_shard_resource_check_validates():
    """shard_resource_check requires model_id."""
    agent = ShardResourceCheckAgent()
    err = agent.validate_input({})
    assert err is not None
    assert "model_id" in err.lower() or "missing" in err.lower()


def test_shard_resource_check_unknown():
    """shard_resource_check returns error for unknown model."""
    agent = ShardResourceCheckAgent()
    result = agent.run({"model_id": "nonexistent"})
    assert result.ok  # agent itself succeeds
    assert result.output["fits"] is False


def test_compute_route_emits_receipt():
    """Specialist compute route emits receipt."""
    agent = SpecialistComputeRouteAgent()
    result = agent.run({"intent": "code_repair", "task": "Fix bug"})
    assert result.receipt is not None
    assert result.receipt["agent"] == "specialist_compute_route"
    assert result.receipt["permission"] == "read"


def test_tool_prompt_includes_compute_agents():
    """format_for_prompt() includes specialist compute agents."""
    reg = create_default_registry()
    prompt = reg.format_for_prompt()
    assert "specialist_compute_route" in prompt
    assert "shard_list" in prompt
    assert "shard_resource_check" in prompt


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
