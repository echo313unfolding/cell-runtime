"""Phase 6 integration test — regulated asset adapter on the security substrate.

Tests deterministic risk policy, context pack assembly, gate integration,
and receipt structure. Live Sentinel tests are skipped if backend is down.
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
from cell.agents.graph_agent import GraphStatsAgent
from cell.agents.ssm_agent import SSMGetStateAgent
from cell.agents.policy_agent import GateDecideAgent
from cell.regulated_asset_adapter import (
    TransferEvent, KYCAttestation, OracleSignal,
    evaluate_risk_policy, build_sentinel_prompt,
    POLICY_VERSION, SANCTIONED_JURISDICTIONS,
)

SENTINEL_PORT = int(os.environ.get("SENTINEL_PORT", "8085"))
FGIP_DB = os.path.expanduser("~/fgip-engine/fgip.db")


def _sentinel_alive(port: int = SENTINEL_PORT) -> bool:
    try:
        req = urllib.request.Request(f"http://localhost:{port}/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return json.loads(resp.read()).get("status") == "ok"
    except Exception:
        return False


def _build_registry() -> AgentRegistry:
    registry = AgentRegistry()
    for cls in [RAGLookupAgent, GraphStatsAgent, SSMGetStateAgent, GateDecideAgent]:
        registry.register(cls())
    return registry


# ---------------------------------------------------------------------------
# TransferEvent parsing
# ---------------------------------------------------------------------------

def test_transfer_event_from_dict():
    """TransferEvent parses all required fields."""
    e = TransferEvent.from_dict({
        "event_id": "RA-001",
        "event_type": "transfer",
        "wallet_from": "alice",
        "wallet_to": "bob",
        "asset_type": "stablecoin",
        "amount": 1000,
        "jurisdiction": "US",
    })
    assert e.event_id == "RA-001"
    assert e.asset_type == "stablecoin"
    assert e.amount == 1000
    assert e.amount_usd == 1000  # defaults to amount


def test_transfer_event_wallet_hash():
    """Wallet hash is deterministic and anonymized."""
    e = TransferEvent.from_dict({
        "event_id": "RA-002",
        "wallet_from": "wallet-alice",
        "wallet_to": "wallet-bob",
        "asset_type": "stablecoin",
        "amount": 100,
        "jurisdiction": "US",
    })
    assert len(e.wallet_from_hash) == 16
    assert e.wallet_from_hash != e.wallet_to_hash
    # Deterministic
    e2 = TransferEvent.from_dict({
        "event_id": "RA-003",
        "wallet_from": "wallet-alice",
        "wallet_to": "wallet-bob",
        "asset_type": "stablecoin",
        "amount": 200,
        "jurisdiction": "US",
    })
    assert e.wallet_from_hash == e2.wallet_from_hash


def test_transfer_event_cross_border():
    """Cross-border detection."""
    domestic = TransferEvent.from_dict({
        "event_id": "RA-D", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "stablecoin", "amount": 100, "jurisdiction": "US",
    })
    assert not domestic.is_cross_border()

    cross = TransferEvent.from_dict({
        "event_id": "RA-X", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "stablecoin", "amount": 100,
        "jurisdiction": "US", "counterparty_jurisdiction": "SG",
    })
    assert cross.is_cross_border()


# ---------------------------------------------------------------------------
# Risk policy — deterministic rules
# ---------------------------------------------------------------------------

def test_policy_clean_low_value():
    """Clean low-value domestic transfer → allow."""
    event = TransferEvent.from_dict({
        "event_id": "RA-CLN", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "stablecoin", "amount": 500,
        "jurisdiction": "US", "velocity_24h": 3, "cumulative_24h_usd": 1500,
    })
    kyc = KYCAttestation(
        attestation_id="K1", wallet_id_hash="x", level="enhanced",
        jurisdiction="US",
    )
    result = evaluate_risk_policy(event, kyc=kyc)
    assert result["decision"] == "allow"
    assert result["risk_level"] == "low"
    assert "RC-CLEAN" in result["reason_codes"]
    assert result["policy_version"] == POLICY_VERSION


def test_policy_sanctioned_jurisdiction():
    """Transfer from sanctioned jurisdiction → reject."""
    event = TransferEvent.from_dict({
        "event_id": "RA-SAN", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "stablecoin", "amount": 100,
        "jurisdiction": "KP",
    })
    result = evaluate_risk_policy(event)
    assert result["decision"] == "reject"
    assert result["risk_level"] == "critical"
    assert "RC-SANCTION-ORIGIN" in result["reason_codes"]


def test_policy_sanctioned_counterparty():
    """Transfer to sanctioned counterparty → reject."""
    event = TransferEvent.from_dict({
        "event_id": "RA-SCP", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "stablecoin", "amount": 100,
        "jurisdiction": "US", "counterparty_jurisdiction": "IR",
    })
    result = evaluate_risk_policy(event)
    assert result["decision"] == "reject"
    assert "RC-SANCTION-COUNTERPARTY" in result["reason_codes"]


def test_policy_no_kyc_adds_risk():
    """No KYC attestation adds risk score."""
    event = TransferEvent.from_dict({
        "event_id": "RA-NK", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "stablecoin", "amount": 500, "jurisdiction": "EU",
    })
    result = evaluate_risk_policy(event, kyc=None)
    assert "RC-KYC-NONE" in result["reason_codes"]
    assert result["risk_score"] >= 30


def test_policy_high_value_basic_kyc():
    """High-value transfer with basic KYC → review or higher."""
    event = TransferEvent.from_dict({
        "event_id": "RA-HV", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "security_token", "amount": 50000, "amount_usd": 50000,
        "jurisdiction": "US", "cumulative_24h_usd": 50000,
    })
    kyc = KYCAttestation(
        attestation_id="K2", wallet_id_hash="y", level="basic", jurisdiction="US",
    )
    result = evaluate_risk_policy(event, kyc=kyc)
    assert result["decision"] in ("review", "hold")
    assert "RC-KYC-INSUFFICIENT" in result["reason_codes"]
    assert "RC-HIGH-VALUE" in result["reason_codes"]


def test_policy_high_velocity():
    """High transaction velocity triggers risk."""
    event = TransferEvent.from_dict({
        "event_id": "RA-VEL", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "utility_token", "amount": 100,
        "jurisdiction": "US", "velocity_24h": 50,
    })
    result = evaluate_risk_policy(event)
    assert "RC-HIGH-VELOCITY" in result["reason_codes"]


def test_policy_cross_border_kyc_gap():
    """Cross-border with insufficient KYC triggers extra risk."""
    event = TransferEvent.from_dict({
        "event_id": "RA-XB", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "stablecoin", "amount": 5000,
        "jurisdiction": "US", "counterparty_jurisdiction": "SG",
    })
    kyc = KYCAttestation(
        attestation_id="K3", wallet_id_hash="z", level="basic", jurisdiction="US",
    )
    result = evaluate_risk_policy(event, kyc=kyc)
    assert "RC-CROSS-BORDER" in result["reason_codes"]
    assert "RC-CROSS-BORDER-KYC-GAP" in result["reason_codes"]


def test_policy_sanctions_screen_fail():
    """Sanctions screening failure → high risk."""
    event = TransferEvent.from_dict({
        "event_id": "RA-SF", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "stablecoin", "amount": 100, "jurisdiction": "US",
    })
    kyc = KYCAttestation(
        attestation_id="K4", wallet_id_hash="w", level="enhanced",
        jurisdiction="US", sanctions_screen_pass=False,
    )
    result = evaluate_risk_policy(event, kyc=kyc)
    assert "RC-SANCTIONS-FAIL" in result["reason_codes"]
    assert result["decision"] in ("hold", "reject")


def test_policy_oracle_high_risk():
    """Oracle high risk signal adds risk."""
    event = TransferEvent.from_dict({
        "event_id": "RA-OR", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "stablecoin", "amount": 5000, "jurisdiction": "US",
    })
    oracle = OracleSignal(signal_id="O1", signal_type="risk_score", value=80)
    result = evaluate_risk_policy(event, oracle=oracle)
    assert "RC-ORACLE-HIGH-RISK" in result["reason_codes"]


def test_policy_pep_fail():
    """PEP screening failure increases risk."""
    event = TransferEvent.from_dict({
        "event_id": "RA-PEP", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "stablecoin", "amount": 100, "jurisdiction": "US",
    })
    kyc = KYCAttestation(
        attestation_id="K5", wallet_id_hash="v", level="enhanced",
        jurisdiction="US", pep_screen_pass=False,
    )
    result = evaluate_risk_policy(event, kyc=kyc)
    assert "RC-PEP-FAIL" in result["reason_codes"]


def test_policy_score_capped_at_100():
    """Risk score is capped at 100 even with multiple triggers."""
    event = TransferEvent.from_dict({
        "event_id": "RA-CAP", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "stablecoin", "amount": 100000, "amount_usd": 100000,
        "jurisdiction": "KP", "counterparty_jurisdiction": "IR",
        "velocity_24h": 100, "cumulative_24h_usd": 500000,
    })
    result = evaluate_risk_policy(event)
    assert result["risk_score"] == 100
    assert result["decision"] == "reject"


# ---------------------------------------------------------------------------
# Sentinel prompt building
# ---------------------------------------------------------------------------

def test_build_sentinel_prompt():
    """Prompt includes event details, policy result, and context."""
    event = TransferEvent.from_dict({
        "event_id": "RA-PR", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "security_token", "amount": 25000, "amount_usd": 25000,
        "jurisdiction": "US", "counterparty_jurisdiction": "SG",
        "velocity_24h": 5,
    })
    policy = {"decision": "review", "risk_level": "medium", "risk_score": 35,
              "reason_codes": ["RC-HIGH-VALUE", "RC-CROSS-BORDER"]}
    context = {
        "ssm": {"found": True, "event_count": 3, "trend": "active"},
        "rag": {"count": 5},
        "graph": {"nodes": 1910},
    }
    prompt = build_sentinel_prompt(event, context, policy)
    assert "security_token" in prompt
    assert "$25,000.00" in prompt
    assert "REVIEW" in prompt
    assert "RC-HIGH-VALUE" in prompt
    assert "3 prior events" in prompt
    assert "5 related policy documents" in prompt
    assert "1910 entities" in prompt


# ---------------------------------------------------------------------------
# Gate integration
# ---------------------------------------------------------------------------

def test_gate_allow_decision():
    """Gate auto-approves allow decision."""
    registry = _build_registry()
    result = registry.run("gate_decide", {
        "action": "asset_allow",
        "detail": "Allow RA-001: 500 USD stablecoin",
        "auto": True,
    })
    assert result.ok
    assert result.output["allowed"] is True


def test_gate_reject_decision():
    """Gate auto-approves reject decision (action is recording the rejection)."""
    registry = _build_registry()
    result = registry.run("gate_decide", {
        "action": "asset_reject",
        "detail": "Reject RA-003: sanctioned jurisdiction",
        "auto": True,
    })
    assert result.ok
    assert result.output["allowed"] is True


# ---------------------------------------------------------------------------
# Context pack assembly against real DBs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(FGIP_DB), reason="FGIP DB not available")
def test_rag_policy_lookup():
    """RAG returns results for compliance/regulation queries."""
    registry = _build_registry()
    result = registry.run("rag_lookup", {
        "query": "compliance OR regulation OR sanctions",
        "scope": "claims",
        "limit": 5,
    })
    assert result.ok
    assert result.output["count"] > 0


@pytest.mark.skipif(not os.path.exists(FGIP_DB), reason="FGIP DB not available")
def test_graph_available_for_context():
    """Graph DB is available and returns entity/edge counts."""
    registry = _build_registry()
    result = registry.run("graph_stats", {})
    assert result.ok
    assert result.output["nodes"] > 0


# ---------------------------------------------------------------------------
# Live Sentinel (skipped if backend not running)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _sentinel_alive(), reason="Sentinel backend not running")
def test_live_sentinel_regulated_asset_verdict():
    """Live Sentinel produces a risk assessment for a regulated asset event."""
    event = TransferEvent.from_dict({
        "event_id": "RA-LIVE", "wallet_from": "a", "wallet_to": "b",
        "asset_type": "security_token", "amount": 50000, "amount_usd": 50000,
        "jurisdiction": "US", "counterparty_jurisdiction": "SG",
    })
    kyc = KYCAttestation(
        attestation_id="K-LIVE", wallet_id_hash="x", level="basic", jurisdiction="US",
    )
    policy = evaluate_risk_policy(event, kyc=kyc)
    context = {"ssm": {}, "rag": {"count": 3}, "graph": {"nodes": 1910}}
    prompt = build_sentinel_prompt(event, context, policy)

    payload = json.dumps({
        "messages": [
            {"role": "system", "content": "You are a regulated asset risk assessment agent."},
            {"role": "user", "content": prompt},
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
    assert len(content) > 20
    assert data["usage"]["completion_tokens"] > 0


@pytest.mark.skipif(not _sentinel_alive(), reason="Sentinel backend not running")
def test_live_full_regulated_asset_chain():
    """Full live chain: event → policy → context → Sentinel → gate → receipt fields."""
    registry = _build_registry()

    # Event
    event = TransferEvent.from_dict({
        "event_id": "RA-CHAIN", "wallet_from": "chain-alice", "wallet_to": "chain-bob",
        "asset_type": "stablecoin", "amount": 15000, "amount_usd": 15000,
        "jurisdiction": "US", "velocity_24h": 8, "cumulative_24h_usd": 30000,
    })
    kyc = KYCAttestation(
        attestation_id="K-CHAIN", wallet_id_hash="ch", level="enhanced", jurisdiction="US",
    )

    # Policy
    policy = evaluate_risk_policy(event, kyc=kyc)
    assert policy["decision"] in ("allow", "review", "hold", "reject")

    # Context
    ssm = registry.run("ssm_get_state", {"entity_id": event.wallet_from})
    rag = registry.run("rag_lookup", {"query": "regulation", "limit": 3})
    graph = registry.run("graph_stats", {})
    context = {
        "ssm": ssm.output if ssm.ok else {},
        "rag": {"count": rag.output.get("count", 0)} if rag.ok else {},
        "graph": graph.output if graph.ok else {},
    }

    # Sentinel
    prompt = build_sentinel_prompt(event, context, policy)
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": "You are a regulated asset risk assessment agent."},
            {"role": "user", "content": prompt},
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
    model_output = data["choices"][0]["message"]["content"]
    assert len(model_output) > 0

    # Gate
    gate = registry.run("gate_decide", {
        "action": f"asset_{policy['decision']}",
        "detail": f"{policy['decision']} {event.event_id}",
        "auto": True,
    })
    assert gate.ok
    assert gate.output["allowed"] is True

    # Verify all receipt-required fields are available
    assert event.event_id is not None
    assert event.wallet_from_hash is not None
    assert policy["policy_version"] == POLICY_VERSION
    assert len(policy["reason_codes"]) > 0
    assert gate.receipt["agent"] == "gate_decide"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
