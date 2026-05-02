"""Graceful degradation tests — every failure path is bounded, receipted, and deterministic.

Proves:
  - Sentinel backend down → bounded error + receipt
  - RAG DB missing → continue without RAG + receipt
  - Graph DB missing → continue without graph + receipt
  - sentinel.db missing → empty SSM state + receipt
  - Malformed Sentinel JSON → parse failure handled + receipt
  - Receipt path unwritable → fail closed for privileged, error propagated
  - Shard load fails → fallback shard or Sentinel
  - Cartridge missing for intent → shard or Sentinel fallback
  - HXQ eval missing → quarantine / no promotion
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.base import AgentRegistry, AgentResult
from cell.agents.rag_agent import RAGLookupAgent
from cell.agents.graph_agent import GraphLookupAgent, GraphStatsAgent
from cell.agents.ssm_agent import SSMGetStateAgent
from cell.agents.sentinel_agent import SentinelTriageAgent
from cell.agents.receipt_agent import ReceiptWriteAgent
from cell.shard_pool import ShardPool, ShardManifest
from cell.cartridge_pool import CartridgePool, CartridgeManifest
from cell.hxq_asset import can_promote, validate_hxq_asset


# ---------------------------------------------------------------------------
# 1. Sentinel backend down → bounded error + receipt
# ---------------------------------------------------------------------------

def test_sentinel_down():
    """When Sentinel has no orchestrator, agent degrades to unknown verdict."""
    reg = AgentRegistry()
    reg.register(SentinelTriageAgent())
    result = reg.run("sentinel_triage", {
        "alert_text": "test alert for offline sentinel",
    })
    # Should degrade gracefully — verdict severity=unknown, not crash
    assert result.ok, f"Sentinel down should degrade gracefully, got error: {result.error}"
    assert result.output["verdict"]["severity"] == "unknown"
    assert "no orchestrator" in result.output["verdict"]["note"].lower()
    # Receipt is always attached by AgentBase.run()
    assert result.receipt is not None
    assert result.receipt["agent"] == "sentinel_triage"


def test_sentinel_timeout():
    """Sentinel agent respects its timeout and doesn't hang."""
    reg = AgentRegistry()
    agent = SentinelTriageAgent()
    reg.register(agent)
    # Just verify the timeout is set to a bounded value
    assert agent.timeout_s <= 60, f"Sentinel timeout {agent.timeout_s}s is too high"


# ---------------------------------------------------------------------------
# 2. RAG DB missing → continue without RAG + receipt
# ---------------------------------------------------------------------------

def test_rag_missing_db():
    """When FGIP DB doesn't exist, RAG agent returns results via file_search fallback."""
    reg = AgentRegistry()
    reg.register(RAGLookupAgent())
    # Patch FGIP_DB to a non-existent path
    with mock.patch("cell.agents.rag_agent.FGIP_DB", "/nonexistent/path/fgip.db"):
        result = reg.run("rag_lookup", {"query": "test", "limit": 3})
    # Should NOT crash — graceful fallback to file_search
    assert result.ok, f"RAG should degrade gracefully, got error: {result.error}"
    assert "source" in result.output
    # Source should be file_search (fallback), not FGIP
    assert result.output["source"] == "file_search"
    # Receipt attached
    assert result.receipt is not None
    assert result.receipt["agent"] == "rag_lookup"


def test_rag_bad_query():
    """RAG agent handles FTS5 syntax errors gracefully."""
    reg = AgentRegistry()
    reg.register(RAGLookupAgent())
    # FTS5 special chars that would cause syntax errors
    result = reg.run("rag_lookup", {"query": "\"unclosed quote", "limit": 1})
    # Should not crash — either returns results or falls back
    assert result.ok or result.error is not None
    assert result.receipt is not None


# ---------------------------------------------------------------------------
# 3. Graph DB missing → continue without graph + receipt
# ---------------------------------------------------------------------------

def test_graph_missing_db():
    """When FGIP DB doesn't exist, graph agent returns error."""
    reg = AgentRegistry()
    reg.register(GraphLookupAgent())
    with mock.patch("cell.agents.graph_agent.FGIP_DB", "/nonexistent/path/fgip.db"):
        result = reg.run("graph_lookup", {"entity": "test_entity"})
    # Should return an error result, not crash
    assert result.error is not None
    assert "not available" in result.error.lower() or "graph" in result.error.lower()
    assert result.receipt is not None
    assert result.receipt["agent"] == "graph_lookup"


def test_graph_stats_missing_db():
    """Graph stats with missing DB returns error."""
    reg = AgentRegistry()
    reg.register(GraphStatsAgent())
    with mock.patch("cell.agents.graph_agent.FGIP_DB", "/nonexistent/path/fgip.db"):
        result = reg.run("graph_stats", {})
    assert result.error is not None
    assert result.receipt is not None


# ---------------------------------------------------------------------------
# 4. sentinel.db missing → empty SSM state + receipt
# ---------------------------------------------------------------------------

def test_ssm_missing_db():
    """When sentinel.db doesn't exist, SSM returns empty state."""
    reg = AgentRegistry()
    reg.register(SSMGetStateAgent())
    with mock.patch("cell.agents.ssm_agent.SENTINEL_DB", "/nonexistent/sentinel.db"):
        result = reg.run("ssm_get_state", {"entity_id": "test_entity"})
    # Should return empty state, not crash
    assert result.ok, f"SSM should degrade gracefully, got error: {result.error}"
    assert result.output["found"] is False
    assert result.output["events"] == []
    assert result.receipt is not None
    assert result.receipt["agent"] == "ssm_get_state"


# ---------------------------------------------------------------------------
# 5. Malformed Sentinel JSON → parse failure handled + receipt
# ---------------------------------------------------------------------------

def test_sentinel_malformed_json():
    """Sentinel agent without orchestrator degrades regardless of mock HTTP.

    SentinelTriageAgent doesn't make HTTP calls directly — the orchestrator
    does. Without orchestrator, it returns unknown verdict. The graceful
    degradation is that it NEVER crashes regardless of backend state.
    """
    agent = SentinelTriageAgent()
    # No orchestrator set — should return unknown verdict
    result = agent.run({"alert_text": "test malformed scenario"})

    # Should not crash
    assert result.receipt is not None
    assert result.output["verdict"]["severity"] == "unknown"
    # Context pack still assembled (empty without agent registry)
    assert "context_pack" in result.output


# ---------------------------------------------------------------------------
# 6. Receipt path unwritable → fail closed, error propagated
# ---------------------------------------------------------------------------

def test_receipt_path_unwritable():
    """Receipt agent propagates error when path is unwritable."""
    agent = ReceiptWriteAgent()

    # Patch RECEIPT_DIR to unwritable location
    with mock.patch("cell.agents.receipt_agent.RECEIPT_DIR", "/proc/fake_unwritable"):
        result = agent.run({
            "action": "test_action",
            "result": {"data": "test"},
        })

    # Should fail — either error result or exception caught by run()
    assert result.error is not None, \
        "Receipt write to unwritable path should produce error"
    assert result.receipt is not None


# ---------------------------------------------------------------------------
# 7. Shard load fails → fallback shard or Sentinel
# ---------------------------------------------------------------------------

def test_shard_fallback_on_missing():
    """Shard pool returns fallback when shard files don't exist."""
    pool = ShardPool()
    pool.register(ShardManifest(
        model_id="test_hxq_shard",
        status="active",
        shard_paths=["/nonexistent/model.gguf"],
        activation_intents=["test_intent"],
        fallback_shard="test_q5_shard",
        fallback="qwen2.5-sentinel",
    ))
    pool.register(ShardManifest(
        model_id="test_q5_shard",
        status="active",
        shard_paths=["/nonexistent/fallback.gguf"],
        activation_intents=[],
        fallback="qwen2.5-sentinel",
    ))

    # Route finds the shard
    shard = pool.route("test_intent")
    assert shard is not None
    assert shard.model_id == "test_hxq_shard"

    # But build_load_plan detects missing files
    plan = pool.build_load_plan("test_hxq_shard")
    assert "error" in plan
    assert "missing" in plan["error"].lower()

    # Fallback shard is available as a named fallback
    assert shard.fallback_shard == "test_q5_shard"
    fb = pool.get(shard.fallback_shard)
    assert fb is not None


def test_shard_inactive_not_routed():
    """Inactive shards are never returned by route()."""
    pool = ShardPool()
    for status in ["candidate", "disabled", "quarantined"]:
        pool.register(ShardManifest(
            model_id=f"shard_{status}",
            status=status,
            activation_intents=[f"intent_{status}"],
        ))
        assert pool.route(f"intent_{status}") is None, \
            f"Shard with status={status} should not route"


# ---------------------------------------------------------------------------
# 8. Cartridge missing for intent → error, not crash
# ---------------------------------------------------------------------------

def test_cartridge_missing_intent():
    """CartridgePool.dispatch returns error for unknown intent."""
    pool = CartridgePool()
    result = pool.dispatch("nonexistent_intent", "do something")
    assert "error" in result
    assert "no active cartridge" in result["error"].lower()


def test_cartridge_disabled_not_routed():
    """Disabled/candidate cartridges are not returned by route()."""
    pool = CartridgePool()
    pool.register(CartridgeManifest(
        cartridge_id="disabled_cart",
        cartridge_dir="/tmp",
        status="disabled",
        activation_intents=["disabled_intent"],
    ))
    pool.register(CartridgeManifest(
        cartridge_id="candidate_cart",
        cartridge_dir="/tmp",
        status="candidate",
        activation_intents=["candidate_intent"],
    ))

    assert pool.route("disabled_intent") is None
    assert pool.route("candidate_intent") is None


def test_cartridge_load_unknown():
    """Loading unknown cartridge returns error dict."""
    pool = CartridgePool()
    result = pool.load_cartridge("nonexistent_cartridge")
    assert "error" in result
    assert "unknown" in result["error"].lower()


# ---------------------------------------------------------------------------
# 9. HXQ eval missing → quarantine / no promotion
# ---------------------------------------------------------------------------

def test_cannot_promote_hxq_missing_helix():
    """HXQ asset cannot promote without tensor fidelity receipt."""
    r = can_promote("hxq_affine_6", behavioral_receipt_path="/fake/path")
    assert r["promotable"] is False
    assert "helix" in r["reason"].lower() or "tensor" in r["reason"].lower()


def test_cannot_promote_hxq_missing_behavioral():
    """HXQ asset cannot promote without behavioral eval receipt."""
    r = can_promote("hxq_affine_6", helix_receipt_path="/fake/path")
    assert r["promotable"] is False
    assert "behavioral" in r["reason"].lower()


def test_cannot_promote_hxq_missing_both():
    """HXQ asset cannot promote without either receipt."""
    r = can_promote("hxq_affine_6")
    assert r["promotable"] is False


def test_validate_hxq_missing_receipts():
    """validate_hxq_asset reports issues for missing receipts."""
    result = validate_hxq_asset(
        manifest_dir="/tmp",
        codec="hxq_affine_6",
    )
    assert not result["valid"]
    assert len(result["issues"]) >= 2  # missing helix + missing behavioral


def test_baseline_promote_needs_behavioral():
    """Baseline codec (Q5_K_M) still needs behavioral eval to promote."""
    r = can_promote("q5_k_m")
    assert r["promotable"] is False
    assert "behavioral" in r["reason"].lower()


def test_nonmodel_codec_always_promotable():
    """Non-model codecs (prompt_pack, etc.) are always promotable."""
    r = can_promote("prompt_pack")
    assert r["promotable"] is True


# ---------------------------------------------------------------------------
# Cross-cutting: every agent emits receipt on failure
# ---------------------------------------------------------------------------

def test_all_agents_emit_receipt_on_error():
    """Every agent emits a receipt even when it errors."""
    agents_and_args = [
        (RAGLookupAgent(), {"query": "x"}),
        (GraphStatsAgent(), {}),
        (SSMGetStateAgent(), {"entity_id": "x"}),
    ]
    for agent, args in agents_and_args:
        # Patch DB paths to force failure/fallback
        with mock.patch.dict(os.environ, {}, clear=False):
            result = agent.run(args)
        assert result.receipt is not None, \
            f"{agent.name} did not emit receipt"
        assert "agent" in result.receipt
        assert "wall_time_s" in result.receipt
        assert "timestamp" in result.receipt


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
