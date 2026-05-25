"""Gate 8: Medical/Insurance Claim State Machine

Domain-specific state machine for medical artifact bundles.
Extends the base 3-state artifact lifecycle (Candidate/Active/Quarantined)
with a 14-state claim workflow overlay.

Architecture:
  - Base layer: artifact provenance (Candidate → Active → Quarantined)
  - Domain layer: claim workflow (Draft → Submitted → ... → Closed)
  - Consent layer: patient consent receipt gates artifact consumption
  - PHI boundary: only hashes touch the chain; content stays off-chain

The two layers are orthogonal but connected:
  - A Quarantined artifact blocks claim state advancement
  - An Active artifact with a Denied claim is still a valid artifact
  - Consent revocation quarantines the artifact (not just the claim)

State transitions emit receipts. Invalid transitions are rejected with
reason codes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ── Claim States ─────────────────────────────────────────────────────────────

class ClaimState(str, Enum):
    """14-state medical claim lifecycle."""
    DRAFT = "Draft"
    SUBMITTED = "Submitted"
    ACKNOWLEDGED = "Acknowledged"
    PENDED = "Pended"
    INFO_REQUESTED = "Information_Requested"
    INFO_RECEIVED = "Information_Received"
    IN_REVIEW = "In_Review"
    PRE_APPROVED = "Pre_Approved"
    APPROVED = "Approved"
    DENIED = "Denied"
    APPEALED = "Appealed"
    APPEAL_IN_REVIEW = "Appeal_In_Review"
    APPEAL_DECIDED = "Appeal_Decided"
    QUARANTINED = "Quarantined"


# ── Valid Transitions ────────────────────────────────────────────────────────

# Each state maps to the set of states it can transition to.
VALID_TRANSITIONS: Dict[ClaimState, set] = {
    ClaimState.DRAFT: {ClaimState.SUBMITTED, ClaimState.QUARANTINED},
    ClaimState.SUBMITTED: {ClaimState.ACKNOWLEDGED, ClaimState.QUARANTINED},
    ClaimState.ACKNOWLEDGED: {ClaimState.PENDED, ClaimState.IN_REVIEW, ClaimState.QUARANTINED},
    ClaimState.PENDED: {ClaimState.INFO_REQUESTED, ClaimState.IN_REVIEW, ClaimState.QUARANTINED},
    ClaimState.INFO_REQUESTED: {ClaimState.INFO_RECEIVED, ClaimState.DENIED, ClaimState.QUARANTINED},
    ClaimState.INFO_RECEIVED: {ClaimState.IN_REVIEW, ClaimState.QUARANTINED},
    ClaimState.IN_REVIEW: {ClaimState.PRE_APPROVED, ClaimState.APPROVED, ClaimState.DENIED, ClaimState.QUARANTINED},
    ClaimState.PRE_APPROVED: {ClaimState.APPROVED, ClaimState.DENIED, ClaimState.QUARANTINED},
    ClaimState.APPROVED: {ClaimState.QUARANTINED},  # terminal except quarantine
    ClaimState.DENIED: {ClaimState.APPEALED, ClaimState.QUARANTINED},
    ClaimState.APPEALED: {ClaimState.APPEAL_IN_REVIEW, ClaimState.QUARANTINED},
    ClaimState.APPEAL_IN_REVIEW: {ClaimState.APPEAL_DECIDED, ClaimState.QUARANTINED},
    ClaimState.APPEAL_DECIDED: {ClaimState.QUARANTINED},  # terminal except quarantine
    ClaimState.QUARANTINED: set(),  # absorbing state
}


# ── Artifact State Mapping ───────────────────────────────────────────────────

def claim_to_artifact_state(claim_state: ClaimState) -> str:
    """Map claim workflow state to base artifact lifecycle state.

    Candidate: claim is in-progress (not yet decided)
    Active: claim has a positive terminal decision
    Quarantined: fraud/integrity issue
    """
    if claim_state == ClaimState.QUARANTINED:
        return "Quarantined"
    if claim_state in (ClaimState.APPROVED, ClaimState.APPEAL_DECIDED):
        return "Active"
    return "Candidate"


# ── Consent Receipt ──────────────────────────────────────────────────────────

@dataclass
class ConsentReceipt:
    """Patient consent receipt. Gates artifact consumption."""
    consent_id: str
    patient_hash: str           # SHA-256 of patient identifier (NOT the identifier itself)
    scope: str                  # e.g. "prior_auth_imaging", "claim_review"
    granted_utc: str
    expires_utc: Optional[str] = None
    revoked: bool = False
    revoked_utc: Optional[str] = None

    def is_valid(self, at_time: Optional[str] = None) -> bool:
        """Check if consent is currently valid."""
        if self.revoked:
            return False
        if self.expires_utc and at_time:
            return at_time <= self.expires_utc
        if self.expires_utc:
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return now <= self.expires_utc
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consent_id": self.consent_id,
            "patient_hash": self.patient_hash,
            "scope": self.scope,
            "granted_utc": self.granted_utc,
            "expires_utc": self.expires_utc,
            "revoked": self.revoked,
            "revoked_utc": self.revoked_utc,
        }


# ── Transition Result ────────────────────────────────────────────────────────

@dataclass
class TransitionResult:
    """Result of a state transition attempt."""
    allowed: bool
    from_state: str
    to_state: str
    artifact_state: str         # base lifecycle state after transition
    reason_codes: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "artifact_state": self.artifact_state,
            "reason_codes": self.reason_codes,
            "details": self.details,
        }


# ── Medical Artifact Bundle ─────────────────────────────────────────────────

@dataclass
class MedicalArtifactBundle:
    """A bundle of medical artifacts sharing one claim lifecycle.

    A bundle can contain:
      - Structured records (FHIR JSON) → gzip/zstd codec
      - Numeric imaging arrays (DICOM-like) → VQ/HXQ codec
      - Each artifact has its own content_hash
      - The bundle has a composite hash (sorted artifact hashes)
    """
    bundle_id: str
    claim_state: ClaimState = ClaimState.DRAFT
    consent: Optional[ConsentReceipt] = None
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    transition_log: List[Dict[str, Any]] = field(default_factory=list)

    def add_artifact(self, artifact_id: str, artifact_type: str,
                     codec: str, content_hash: str,
                     phi_fields_hashed: bool = True) -> None:
        """Register an artifact in the bundle."""
        self.artifacts[artifact_id] = {
            "artifact_type": artifact_type,
            "codec": codec,
            "content_hash": content_hash,
            "phi_fields_hashed": phi_fields_hashed,
            "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def bundle_hash(self) -> str:
        """Composite hash of all artifact content hashes."""
        if not self.artifacts:
            return hashlib.sha256(b"empty_bundle").hexdigest()
        sorted_hashes = sorted(
            a["content_hash"] for a in self.artifacts.values()
        )
        return hashlib.sha256(
            json.dumps(sorted_hashes).encode("utf-8")
        ).hexdigest()

    def transition(self, to_state: ClaimState,
                   reason: str = "",
                   actor: str = "system") -> TransitionResult:
        """Attempt a state transition."""
        from_state = self.claim_state
        valid_targets = VALID_TRANSITIONS.get(from_state, set())

        # Check consent for non-quarantine transitions
        if to_state != ClaimState.QUARANTINED and self.consent is not None:
            if not self.consent.is_valid():
                result = TransitionResult(
                    allowed=False,
                    from_state=from_state.value,
                    to_state=to_state.value,
                    artifact_state=claim_to_artifact_state(from_state),
                    reason_codes=["MS-CONSENT-REVOKED"],
                    details={"consent_id": self.consent.consent_id},
                )
                self.transition_log.append({
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    **result.to_dict(),
                })
                return result

        if to_state not in valid_targets:
            result = TransitionResult(
                allowed=False,
                from_state=from_state.value,
                to_state=to_state.value,
                artifact_state=claim_to_artifact_state(from_state),
                reason_codes=["MS-INVALID-TRANSITION"],
                details={
                    "valid_targets": [s.value for s in valid_targets],
                    "reason": reason,
                },
            )
            self.transition_log.append({
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **result.to_dict(),
            })
            return result

        # Valid transition
        self.claim_state = to_state
        new_artifact_state = claim_to_artifact_state(to_state)

        result = TransitionResult(
            allowed=True,
            from_state=from_state.value,
            to_state=to_state.value,
            artifact_state=new_artifact_state,
            reason_codes=["MS-TRANSITION-OK"],
            details={"reason": reason, "actor": actor},
        )
        self.transition_log.append({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **result.to_dict(),
        })
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "claim_state": self.claim_state.value,
            "artifact_state": claim_to_artifact_state(self.claim_state),
            "bundle_hash": self.bundle_hash(),
            "consent": self.consent.to_dict() if self.consent else None,
            "artifacts": self.artifacts,
            "transition_count": len(self.transition_log),
            "transition_log": self.transition_log,
        }


# ── Medical Preflight Extension ─────────────────────────────────────────────

def medical_preflight(bundle: MedicalArtifactBundle) -> Dict[str, Any]:
    """Extended preflight for medical bundles.

    Checks:
      1. Consent exists and is valid
      2. Claim state maps to Active artifact state
      3. All artifacts have content hashes
      4. PHI boundary maintained (phi_fields_hashed == True)
      5. Bundle is not empty

    Returns dict with decision and reason codes.
    """
    decision = "ALLOW"
    reason_codes = []
    details = {}

    # 1. Consent check
    if bundle.consent is None:
        decision = "REJECT"
        reason_codes.append("MPF-NO-CONSENT")
    elif not bundle.consent.is_valid():
        decision = "REJECT"
        reason_codes.append("MPF-CONSENT-INVALID")
        details["consent_id"] = bundle.consent.consent_id

    # 2. Claim state check
    artifact_state = claim_to_artifact_state(bundle.claim_state)
    if artifact_state == "Quarantined":
        decision = "REJECT"
        reason_codes.append("MPF-QUARANTINED")
    elif artifact_state == "Candidate":
        if decision != "REJECT":
            decision = "HOLD"
        reason_codes.append("MPF-NOT-ACTIVE")
        details["claim_state"] = bundle.claim_state.value

    # 3. Empty bundle
    if not bundle.artifacts:
        decision = "REJECT"
        reason_codes.append("MPF-EMPTY-BUNDLE")

    # 4. Content hash check
    missing_hashes = [
        aid for aid, a in bundle.artifacts.items()
        if not a.get("content_hash")
    ]
    if missing_hashes:
        decision = "REJECT"
        reason_codes.append("MPF-MISSING-HASH")
        details["missing_hash_artifacts"] = missing_hashes

    # 5. PHI boundary check
    phi_violations = [
        aid for aid, a in bundle.artifacts.items()
        if not a.get("phi_fields_hashed", True)
    ]
    if phi_violations:
        decision = "REJECT"
        reason_codes.append("MPF-PHI-BOUNDARY")
        details["phi_violation_artifacts"] = phi_violations

    if not reason_codes:
        reason_codes.append("MPF-CLEAN")

    return {
        "decision": decision,
        "reason_codes": reason_codes,
        "bundle_id": bundle.bundle_id,
        "claim_state": bundle.claim_state.value,
        "artifact_state": artifact_state,
        "bundle_hash": bundle.bundle_hash(),
        "artifact_count": len(bundle.artifacts),
        "details": details,
    }
