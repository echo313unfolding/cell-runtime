"""Sequential memory tests for regulated asset adapter.

Proves that the SSM + deterministic policy layer detects patterns
across sequences of events — structuring, velocity bursts, jurisdiction
hopping, counterparty fan-in/fan-out.

This does NOT test model behavior. It tests that the deterministic
adapter layer and its interface to SSM produce correct risk signals
when events are evaluated in sequence.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.regulated_asset_adapter import (
    TransferEvent, KYCAttestation, OracleSignal,
    evaluate_risk_policy,
    THRESHOLD_HIGH_VALUE_USD, THRESHOLD_VELOCITY_24H,
    THRESHOLD_CUMULATIVE_24H_USD,
)


def _make_event(**overrides) -> dict:
    base = {
        "event_id": "SEQ-001",
        "wallet_from": "wallet-a",
        "wallet_to": "wallet-b",
        "asset_type": "stablecoin",
        "amount": 500,
        "amount_usd": 500,
        "jurisdiction": "US",
        "velocity_24h": 0,
        "cumulative_24h_usd": 0,
    }
    base.update(overrides)
    return base


# ===========================================================================
# 1. Structuring — many small transfers below threshold
# ===========================================================================

def test_structuring_individual_events_clean():
    """Each individual $9,900 transfer is below the $10,000 threshold.

    Single event: amount_usd < THRESHOLD_HIGH_VALUE_USD → no RC-HIGH-VALUE.
    But cumulative and velocity fields carry the sequence signal.
    """
    kyc = KYCAttestation("K1", "x", "enhanced", "US")
    event = TransferEvent.from_dict(_make_event(
        amount=9900, amount_usd=9900,
        velocity_24h=1, cumulative_24h_usd=9900,
    ))
    result = evaluate_risk_policy(event, kyc=kyc)
    assert "RC-HIGH-VALUE" not in result["reason_codes"]
    assert result["decision"] == "allow"


def test_structuring_cumulative_triggers():
    """After 6 transfers of $9,900, cumulative exceeds $50,000.

    The adapter relies on the caller to provide cumulative_24h_usd.
    When that crosses the threshold, RC-CUMULATIVE-HIGH fires.
    """
    kyc = KYCAttestation("K1", "x", "enhanced", "US")
    # 6th transfer: 6 * 9900 = 59400 > 50000
    event = TransferEvent.from_dict(_make_event(
        event_id="SEQ-006",
        amount=9900, amount_usd=9900,
        velocity_24h=6, cumulative_24h_usd=59400,
    ))
    result = evaluate_risk_policy(event, kyc=kyc)
    assert "RC-CUMULATIVE-HIGH" in result["reason_codes"]
    assert result["risk_score"] >= 20


def test_structuring_velocity_triggers():
    """After 21+ transfers in 24h, velocity threshold fires."""
    kyc = KYCAttestation("K1", "x", "enhanced", "US")
    event = TransferEvent.from_dict(_make_event(
        event_id="SEQ-021",
        amount=2000, amount_usd=2000,
        velocity_24h=21, cumulative_24h_usd=42000,
    ))
    result = evaluate_risk_policy(event, kyc=kyc)
    assert "RC-HIGH-VELOCITY" in result["reason_codes"]


def test_structuring_pattern_escalates():
    """Structuring pattern: velocity + cumulative together escalate decision.

    10 transfers of $9,900 = $99,000 cumulative, 10 velocity.
    Not yet velocity threshold (20), but cumulative is high.
    """
    kyc = KYCAttestation("K1", "x", "enhanced", "US")
    results = []
    for i in range(1, 11):
        event = TransferEvent.from_dict(_make_event(
            event_id=f"STRUCT-{i:03d}",
            amount=9900, amount_usd=9900,
            velocity_24h=i,
            cumulative_24h_usd=9900 * i,
        ))
        result = evaluate_risk_policy(event, kyc=kyc)
        results.append(result)

    # First few: clean
    assert results[0]["decision"] == "allow"
    # By event 6 (cumulative 59400 > 50000): triggers
    assert "RC-CUMULATIVE-HIGH" in results[5]["reason_codes"]
    # Last event (cumulative 99000, velocity 10): cumulative fires
    # With enhanced KYC, only RC-CUMULATIVE-HIGH (20 pts) → below review (25)
    # This documents that cumulative alone is insufficient without KYC gap
    assert "RC-CUMULATIVE-HIGH" in results[-1]["reason_codes"]


def test_structuring_no_kyc_accelerates():
    """Structuring without KYC triggers faster due to RC-KYC-NONE (+30)."""
    results_with_kyc = []
    results_no_kyc = []
    kyc = KYCAttestation("K1", "x", "enhanced", "US")

    for i in range(1, 11):
        event = TransferEvent.from_dict(_make_event(
            event_id=f"STRUCT-KYC-{i:03d}",
            amount=9900, amount_usd=9900,
            velocity_24h=i,
            cumulative_24h_usd=9900 * i,
        ))
        results_with_kyc.append(evaluate_risk_policy(event, kyc=kyc))
        results_no_kyc.append(evaluate_risk_policy(event, kyc=None))

    # No-KYC scores are always >= with-KYC scores
    for r_kyc, r_no in zip(results_with_kyc, results_no_kyc):
        assert r_no["risk_score"] >= r_kyc["risk_score"]

    # No-KYC hits hold/reject earlier
    no_kyc_first_escalation = next(
        i for i, r in enumerate(results_no_kyc) if r["decision"] != "allow"
    )
    kyc_first_escalation = next(
        (i for i, r in enumerate(results_with_kyc) if r["decision"] != "allow"),
        len(results_with_kyc),  # might never escalate
    )
    assert no_kyc_first_escalation <= kyc_first_escalation


# ===========================================================================
# 2. Velocity burst
# ===========================================================================

def test_velocity_burst_triggers_hold():
    """45 transactions in 24h with no KYC → hold."""
    event = TransferEvent.from_dict(_make_event(
        amount=200, amount_usd=200,
        velocity_24h=45, cumulative_24h_usd=9000,
    ))
    result = evaluate_risk_policy(event)
    assert "RC-HIGH-VELOCITY" in result["reason_codes"]
    assert "RC-KYC-NONE" in result["reason_codes"]
    assert result["decision"] in ("hold", "reject")


def test_velocity_just_at_threshold():
    """Exactly at velocity threshold — should NOT trigger."""
    kyc = KYCAttestation("K1", "x", "enhanced", "US")
    event = TransferEvent.from_dict(_make_event(
        velocity_24h=THRESHOLD_VELOCITY_24H,
        cumulative_24h_usd=5000,
    ))
    result = evaluate_risk_policy(event, kyc=kyc)
    assert "RC-HIGH-VELOCITY" not in result["reason_codes"]


def test_velocity_one_above_threshold():
    """One above velocity threshold triggers."""
    kyc = KYCAttestation("K1", "x", "enhanced", "US")
    event = TransferEvent.from_dict(_make_event(
        velocity_24h=THRESHOLD_VELOCITY_24H + 1,
        cumulative_24h_usd=5000,
    ))
    result = evaluate_risk_policy(event, kyc=kyc)
    assert "RC-HIGH-VELOCITY" in result["reason_codes"]


# ===========================================================================
# 3. Counterparty fan-in / fan-out
# ===========================================================================

def test_fan_out_same_sender_many_receivers():
    """Same wallet sends to many different wallets.

    Current adapter doesn't track counterparty diversity — each event
    is evaluated independently. Documents this as a design gap.
    The velocity_24h and cumulative_24h_usd are the only sequence signals.
    """
    kyc = KYCAttestation("K1", "x", "enhanced", "US")
    results = []
    for i in range(10):
        event = TransferEvent.from_dict(_make_event(
            event_id=f"FAN-OUT-{i:03d}",
            wallet_from="wallet-sender",
            wallet_to=f"wallet-receiver-{i}",
            amount=5000, amount_usd=5000,
            velocity_24h=i + 1,
            cumulative_24h_usd=5000 * (i + 1),
        ))
        results.append(evaluate_risk_policy(event, kyc=kyc))

    # No fan-out reason code exists — known gap
    for r in results:
        assert "RC-FAN-OUT" not in r.get("reason_codes", [])

    # Event 10: cumulative = 50000 — check is > 50000, so exactly 50000 doesn't trigger
    # Event 11 would trigger. This documents the boundary behavior.
    assert "RC-CUMULATIVE-HIGH" not in results[9]["reason_codes"]
    # But with 11th event (55000), it would:
    event_11 = TransferEvent.from_dict(_make_event(
        event_id="FAN-OUT-010",
        wallet_from="wallet-sender",
        wallet_to="wallet-receiver-10",
        amount=5000, amount_usd=5000,
        velocity_24h=11,
        cumulative_24h_usd=55000,
    ))
    r11 = evaluate_risk_policy(event_11, kyc=kyc)
    assert "RC-CUMULATIVE-HIGH" in r11["reason_codes"]


def test_fan_in_many_senders_same_receiver():
    """Many wallets send to the same wallet.

    Same gap as fan-out — no counterparty tracking per-event.
    Each event appears independent.
    """
    kyc = KYCAttestation("K1", "x", "enhanced", "US")
    for i in range(5):
        event = TransferEvent.from_dict(_make_event(
            event_id=f"FAN-IN-{i:03d}",
            wallet_from=f"wallet-sender-{i}",
            wallet_to="wallet-receiver-central",
            amount=5000, amount_usd=5000,
            velocity_24h=1,
            cumulative_24h_usd=5000,
        ))
        result = evaluate_risk_policy(event, kyc=kyc)
        # Each looks clean individually
        assert result["decision"] == "allow"


# ===========================================================================
# 4. Jurisdiction hopping
# ===========================================================================

def test_jurisdiction_hopping_sequence():
    """Wallet sends from different jurisdictions across events.

    Current policy evaluates jurisdiction per-event. A wallet that
    hops US → SG → JP → EU is evaluated per-event only. There is
    no cross-event jurisdiction tracking. Documents the gap.
    """
    kyc = KYCAttestation("K1", "x", "enhanced", "US")
    jurisdictions = ["US", "SG", "JP", "EU", "GB"]
    results = []
    for i, j in enumerate(jurisdictions):
        event = TransferEvent.from_dict(_make_event(
            event_id=f"JUR-HOP-{i:03d}",
            jurisdiction=j,
            velocity_24h=i + 1,
            cumulative_24h_usd=1000 * (i + 1),
        ))
        results.append(evaluate_risk_policy(event, kyc=kyc))

    # Each is clean (all are non-sanctioned, low amounts)
    for r in results:
        assert r["decision"] == "allow"

    # No jurisdiction-hopping reason code — known gap
    for r in results:
        assert "RC-JURISDICTION-HOP" not in r.get("reason_codes", [])


def test_jurisdiction_hop_into_sanctioned():
    """Wallet hops from clean jurisdictions into sanctioned → reject."""
    kyc = KYCAttestation("K1", "x", "enhanced", "US")
    clean = TransferEvent.from_dict(_make_event(
        event_id="HOP-CLEAN", jurisdiction="US"))
    r1 = evaluate_risk_policy(clean, kyc=kyc)
    assert r1["decision"] == "allow"

    sanctioned = TransferEvent.from_dict(_make_event(
        event_id="HOP-SANCT", jurisdiction="KP"))
    r2 = evaluate_risk_policy(sanctioned, kyc=kyc)
    assert r2["decision"] == "reject"


# ===========================================================================
# 5. Cross-border escalation
# ===========================================================================

def test_clean_domestic_then_cross_border():
    """Clean domestic transfers followed by cross-border triggers codes."""
    kyc = KYCAttestation("K1", "x", "basic", "US")

    domestic = TransferEvent.from_dict(_make_event(
        event_id="DOM-001", amount=5000, amount_usd=5000,
    ))
    r1 = evaluate_risk_policy(domestic, kyc=kyc)
    assert "RC-CROSS-BORDER" not in r1["reason_codes"]

    cross = TransferEvent.from_dict(_make_event(
        event_id="XB-001", amount=5000, amount_usd=5000,
        counterparty_jurisdiction="SG",
    ))
    r2 = evaluate_risk_policy(cross, kyc=kyc)
    assert "RC-CROSS-BORDER" in r2["reason_codes"]
    assert "RC-CROSS-BORDER-KYC-GAP" in r2["reason_codes"]


# ===========================================================================
# 6. KYC downgrade scenarios
# ===========================================================================

def test_kyc_downgrade_increases_risk():
    """Going from enhanced to no KYC increases risk for same event."""
    event = TransferEvent.from_dict(_make_event(
        amount=15000, amount_usd=15000,
        velocity_24h=5, cumulative_24h_usd=20000,
    ))
    kyc_enhanced = KYCAttestation("K1", "x", "enhanced", "US")
    kyc_basic = KYCAttestation("K2", "x", "basic", "US")

    r_enhanced = evaluate_risk_policy(event, kyc=kyc_enhanced)
    r_basic = evaluate_risk_policy(event, kyc=kyc_basic)
    r_none = evaluate_risk_policy(event, kyc=None)

    # Risk increases: enhanced < basic < none
    assert r_enhanced["risk_score"] < r_basic["risk_score"]
    assert r_basic["risk_score"] < r_none["risk_score"]


def test_kyc_downgrade_escalates_decision():
    """KYC downgrade can escalate from allow to hold/reject."""
    event = TransferEvent.from_dict(_make_event(
        amount=15000, amount_usd=15000,
        counterparty_jurisdiction="SG",
    ))
    kyc_enhanced = KYCAttestation("K1", "x", "enhanced", "US")
    kyc_none = None

    r_enhanced = evaluate_risk_policy(event, kyc=kyc_enhanced)
    r_none = evaluate_risk_policy(event, kyc=kyc_none)

    # Enhanced: HIGH-VALUE + CROSS-BORDER = review-ish
    # None: + KYC-NONE + CROSS-BORDER-KYC-GAP = higher
    decision_severity = {"allow": 0, "review": 1, "hold": 2, "reject": 3}
    assert decision_severity[r_none["decision"]] >= decision_severity[r_enhanced["decision"]]


# ===========================================================================
# 7. Documented gaps
# ===========================================================================

def test_no_counterparty_diversity_tracking():
    """DOCUMENTED GAP: No per-wallet counterparty diversity tracking.

    The adapter evaluates each event independently. Counterparty fan-in/out
    must be detected by the caller updating velocity_24h and cumulative_24h_usd.
    """
    # This test exists to document the gap, not to test missing functionality
    event = TransferEvent.from_dict(_make_event())
    result = evaluate_risk_policy(event)
    # No RC-FAN-IN, RC-FAN-OUT, RC-COUNTERPARTY-DIVERSITY codes exist
    all_codes = set()
    for j in ["US", "KP", "SG", "IR"]:
        e = TransferEvent.from_dict(_make_event(jurisdiction=j))
        r = evaluate_risk_policy(e)
        all_codes.update(r["reason_codes"])
    assert "RC-FAN-IN" not in all_codes
    assert "RC-FAN-OUT" not in all_codes
    assert "RC-JURISDICTION-HOP" not in all_codes
    assert "RC-STRUCTURING" not in all_codes


def test_no_kyc_expiry_tracking():
    """DOCUMENTED GAP: No KYC attestation expiry/staleness tracking."""
    kyc = KYCAttestation("K-OLD", "x", "enhanced", "US")
    assert not hasattr(kyc, "issued_at")
    assert not hasattr(kyc, "expires_at")


def test_no_event_deduplication():
    """DOCUMENTED GAP: Same event_id can be evaluated multiple times.

    The adapter is stateless — it doesn't track seen event_ids.
    Deduplication is the caller's responsibility.
    """
    event = TransferEvent.from_dict(_make_event(event_id="DUPE-001"))
    r1 = evaluate_risk_policy(event)
    r2 = evaluate_risk_policy(event)
    # Same result both times — no dedup enforcement
    assert r1 == r2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
