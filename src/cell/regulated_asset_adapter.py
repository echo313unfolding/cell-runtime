"""Regulated asset event adapter — risk evaluation on top of the security substrate.

Not a chain. Not custody. Not KYC. Not settlement.

This adapter takes a synthetic transfer event, extracts risk-relevant fields,
queries the same SSM/RAG/graph context pack used by Sentinel, runs a local
risk decision, and emits a receipt.

Decision outputs: allow, hold, review, reject.

The adapter consumes mock attestations (KYC, oracle signals) — it does not
perform identity verification or price discovery.
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional


# Risk policy — deterministic rules applied BEFORE model call
POLICY_VERSION = "regulated_asset_v0.2"

# Thresholds
THRESHOLD_HIGH_VALUE_USD = 10_000
THRESHOLD_VELOCITY_24H = 20
THRESHOLD_CUMULATIVE_24H_USD = 50_000
THRESHOLD_COUNTERPARTY_DIVERSITY_24H = 10
THRESHOLD_SUBTHRESHOLD_REPEAT_24H = 5

# Sanctioned jurisdictions (synthetic — for demo only)
SANCTIONED_JURISDICTIONS = {"KP", "IR", "SY", "CU"}

# Jurisdictions requiring enhanced KYC for transfers > threshold
ENHANCED_KYC_JURISDICTIONS = {"US", "EU", "GB", "SG", "JP", "AU", "CA"}


@dataclass
class TransferEvent:
    """Parsed regulated asset transfer event."""
    event_id: str
    event_type: str
    timestamp: str
    wallet_from: str
    wallet_to: str
    asset_type: str
    amount: float
    jurisdiction: str
    amount_usd: float = 0.0
    counterparty_jurisdiction: str = ""
    velocity_24h: int = 0
    cumulative_24h_usd: float = 0.0
    unique_counterparties_24h: int = 0
    subthreshold_repeat_count_24h: int = 0
    kyc_attestation_id: str = ""
    oracle_signal_id: str = ""
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "TransferEvent":
        return cls(
            event_id=d["event_id"],
            event_type=d.get("event_type", "transfer"),
            timestamp=d.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            wallet_from=d["wallet_from"],
            wallet_to=d["wallet_to"],
            asset_type=d.get("asset_type", "stablecoin"),
            amount=d.get("amount", 0),
            jurisdiction=d.get("jurisdiction", ""),
            amount_usd=d.get("amount_usd", d.get("amount", 0)),
            counterparty_jurisdiction=d.get("counterparty_jurisdiction", ""),
            velocity_24h=d.get("velocity_24h", 0),
            cumulative_24h_usd=d.get("cumulative_24h_usd", 0),
            unique_counterparties_24h=d.get("unique_counterparties_24h", 0),
            subthreshold_repeat_count_24h=d.get("subthreshold_repeat_count_24h", 0),
            kyc_attestation_id=d.get("kyc_attestation_id", ""),
            oracle_signal_id=d.get("oracle_signal_id", ""),
            metadata=d.get("metadata", {}),
        )

    @property
    def wallet_from_hash(self) -> str:
        return hashlib.sha256(self.wallet_from.encode()).hexdigest()[:16]

    @property
    def wallet_to_hash(self) -> str:
        return hashlib.sha256(self.wallet_to.encode()).hexdigest()[:16]

    def is_cross_border(self) -> bool:
        return bool(self.counterparty_jurisdiction
                    and self.counterparty_jurisdiction != self.jurisdiction)

    def is_self_transfer(self) -> bool:
        return self.wallet_from == self.wallet_to


@dataclass
class KYCAttestation:
    """Parsed KYC attestation (mock)."""
    attestation_id: str
    wallet_id_hash: str
    level: str  # none, basic, enhanced, institutional
    jurisdiction: str
    sanctions_screen_pass: bool = True
    pep_screen_pass: bool = True
    issued_at: str = ""  # ISO8601 timestamp
    max_age_days: int = 365  # KYC considered stale after this many days

    def is_stale(self, reference_time: str = "") -> bool:
        """Check if KYC attestation is older than max_age_days."""
        if not self.issued_at:
            return False  # no issue date = can't determine staleness
        try:
            from datetime import datetime, timedelta
            issued = datetime.fromisoformat(self.issued_at.replace("Z", "+00:00"))
            if reference_time:
                now = datetime.fromisoformat(reference_time.replace("Z", "+00:00"))
            else:
                now = datetime.utcnow().replace(tzinfo=issued.tzinfo)
            return (now - issued).days > self.max_age_days
        except (ValueError, TypeError):
            return False  # unparseable date = can't determine staleness

    @classmethod
    def from_dict(cls, d: dict) -> "KYCAttestation":
        return cls(
            attestation_id=d["attestation_id"],
            wallet_id_hash=d["wallet_id_hash"],
            level=d.get("level", "none"),
            jurisdiction=d.get("jurisdiction", ""),
            sanctions_screen_pass=d.get("sanctions_screen_pass", True),
            pep_screen_pass=d.get("pep_screen_pass", True),
            issued_at=d.get("issued_at", ""),
            max_age_days=d.get("max_age_days", 365),
        )


@dataclass
class OracleSignal:
    """Parsed oracle signal (mock)."""
    signal_id: str
    signal_type: str
    value: float
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "OracleSignal":
        return cls(
            signal_id=d["signal_id"],
            signal_type=d.get("signal_type", "risk_score"),
            value=d.get("value", 0),
            confidence=d.get("confidence", 1.0),
        )


def evaluate_risk_policy(
    event: TransferEvent,
    kyc: Optional[KYCAttestation] = None,
    oracle: Optional[OracleSignal] = None,
) -> dict:
    """Apply deterministic risk policy to a transfer event.

    Returns:
        {
            "risk_level": "low|medium|high|critical",
            "decision": "allow|hold|review|reject",
            "reason_codes": ["RC-001", ...],
            "policy_version": "...",
        }
    """
    reason_codes = []
    risk_score = 0  # 0-100

    # --- Input validation (fail closed) ---
    try:
        amount_val = float(event.amount_usd)
    except (TypeError, ValueError):
        amount_val = 0
    if amount_val < 0:
        reason_codes.append("RC-INVALID-AMOUNT")
        risk_score += 50

    # --- Self-transfer detection ---
    if event.is_self_transfer():
        reason_codes.append("RC-SELF-TRANSFER")
        risk_score += 25

    # --- Sanctions check ---
    if event.jurisdiction in SANCTIONED_JURISDICTIONS:
        reason_codes.append("RC-SANCTION-ORIGIN")
        risk_score += 80
    if event.counterparty_jurisdiction in SANCTIONED_JURISDICTIONS:
        reason_codes.append("RC-SANCTION-COUNTERPARTY")
        risk_score += 80

    # --- KYC level check ---
    kyc_level = kyc.level if kyc else "none"
    if kyc_level == "none":
        reason_codes.append("RC-KYC-NONE")
        risk_score += 30
    elif kyc_level == "basic" and event.amount_usd > THRESHOLD_HIGH_VALUE_USD:
        reason_codes.append("RC-KYC-INSUFFICIENT")
        risk_score += 20

    # --- Sanctions/PEP screening ---
    if kyc and not kyc.sanctions_screen_pass:
        reason_codes.append("RC-SANCTIONS-FAIL")
        risk_score += 60
    if kyc and not kyc.pep_screen_pass:
        reason_codes.append("RC-PEP-FAIL")
        risk_score += 40

    # --- Value thresholds ---
    if event.amount_usd > THRESHOLD_HIGH_VALUE_USD:
        reason_codes.append("RC-HIGH-VALUE")
        risk_score += 15
    if event.cumulative_24h_usd > THRESHOLD_CUMULATIVE_24H_USD:
        reason_codes.append("RC-CUMULATIVE-HIGH")
        risk_score += 20

    # --- Velocity check ---
    if event.velocity_24h > THRESHOLD_VELOCITY_24H:
        reason_codes.append("RC-HIGH-VELOCITY")
        risk_score += 25

    # --- Cross-border ---
    if event.is_cross_border():
        reason_codes.append("RC-CROSS-BORDER")
        risk_score += 10
        # Enhanced KYC required for cross-border in regulated jurisdictions
        if (event.jurisdiction in ENHANCED_KYC_JURISDICTIONS
                and kyc_level not in ("enhanced", "institutional")):
            reason_codes.append("RC-CROSS-BORDER-KYC-GAP")
            risk_score += 15

    # --- Oracle risk signal ---
    if oracle and oracle.signal_type == "risk_score":
        if oracle.value > 70:
            reason_codes.append("RC-ORACLE-HIGH-RISK")
            risk_score += 25
        elif oracle.value > 40:
            reason_codes.append("RC-ORACLE-MEDIUM-RISK")
            risk_score += 10

    # --- KYC staleness ---
    if kyc and kyc.is_stale(event.timestamp):
        reason_codes.append("RC-KYC-STALE")
        risk_score += 15

    # --- KYC jurisdiction mismatch ---
    if kyc and kyc.jurisdiction and event.jurisdiction:
        if kyc.jurisdiction != event.jurisdiction:
            reason_codes.append("RC-KYC-JURISDICTION-MISMATCH")
            risk_score += 10

    # --- Counterparty diversity ---
    if event.unique_counterparties_24h > THRESHOLD_COUNTERPARTY_DIVERSITY_24H:
        reason_codes.append("RC-COUNTERPARTY-DIVERSITY")
        risk_score += 20

    # --- Structuring sequence ---
    if (event.subthreshold_repeat_count_24h > THRESHOLD_SUBTHRESHOLD_REPEAT_24H
            and event.cumulative_24h_usd > THRESHOLD_CUMULATIVE_24H_USD):
        reason_codes.append("RC-STRUCTURING-SEQUENCE")
        risk_score += 30

    # --- Map score to level and decision ---
    risk_score = min(risk_score, 100)

    if risk_score >= 80:
        risk_level = "critical"
        decision = "reject"
    elif risk_score >= 50:
        risk_level = "high"
        decision = "hold"
    elif risk_score >= 25:
        risk_level = "medium"
        decision = "review"
    else:
        risk_level = "low"
        decision = "allow"

    if not reason_codes:
        reason_codes.append("RC-CLEAN")

    return {
        "risk_level": risk_level,
        "decision": decision,
        "reason_codes": reason_codes,
        "risk_score": risk_score,
        "policy_version": POLICY_VERSION,
    }


def build_sentinel_prompt(event: TransferEvent, context_pack: dict,
                          policy_result: dict) -> str:
    """Build a prompt for Sentinel to assess a regulated asset event.

    The model provides a natural-language assessment. The deterministic policy
    has already made the decision — the model adds explanation and may flag
    additional concerns.
    """
    lines = [
        f"Regulated asset transfer event for risk assessment.",
        f"",
        f"Event: {event.event_type} of {event.amount} {event.asset_type}",
        f"Amount (USD): ${event.amount_usd:,.2f}",
        f"Jurisdiction: {event.jurisdiction}",
    ]
    if event.counterparty_jurisdiction:
        lines.append(f"Counterparty jurisdiction: {event.counterparty_jurisdiction}")
    lines.append(f"Velocity (24h): {event.velocity_24h} transactions")
    lines.append(f"Cumulative (24h USD): ${event.cumulative_24h_usd:,.2f}")
    lines.append(f"")
    lines.append(f"Policy decision: {policy_result['decision'].upper()}")
    lines.append(f"Risk level: {policy_result['risk_level']}")
    lines.append(f"Risk score: {policy_result['risk_score']}/100")
    lines.append(f"Reason codes: {', '.join(policy_result['reason_codes'])}")
    lines.append(f"")

    # Context
    ssm = context_pack.get("ssm", {})
    if ssm.get("found"):
        lines.append(f"SSM: Entity has {ssm.get('event_count', 0)} prior events, trend={ssm.get('trend', 'unknown')}")
    rag = context_pack.get("rag", {})
    if rag.get("count", 0) > 0:
        lines.append(f"RAG: {rag['count']} related policy documents found")
    graph = context_pack.get("graph", {})
    if graph.get("nodes", 0) > 0:
        lines.append(f"Graph: {graph['nodes']} entities in knowledge base")

    lines.append(f"")
    lines.append(f"Provide a brief risk assessment. Note any additional concerns not captured by the policy rules.")

    return "\n".join(lines)
