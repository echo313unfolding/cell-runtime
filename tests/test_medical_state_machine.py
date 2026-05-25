"""Tests for medical_state_machine.py — Gate 8D/8E coverage."""

import hashlib
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cell.medical_state_machine import (
    ClaimState,
    ConsentReceipt,
    MedicalArtifactBundle,
    VALID_TRANSITIONS,
    claim_to_artifact_state,
    medical_preflight,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── State mapping ────────────────────────────────────────────────────────────

class TestClaimToArtifactState:
    def test_approved_is_active(self):
        assert claim_to_artifact_state(ClaimState.APPROVED) == "Active"

    def test_appeal_decided_is_active(self):
        assert claim_to_artifact_state(ClaimState.APPEAL_DECIDED) == "Active"

    def test_quarantined_is_quarantined(self):
        assert claim_to_artifact_state(ClaimState.QUARANTINED) == "Quarantined"

    def test_draft_is_candidate(self):
        assert claim_to_artifact_state(ClaimState.DRAFT) == "Candidate"

    def test_in_review_is_candidate(self):
        assert claim_to_artifact_state(ClaimState.IN_REVIEW) == "Candidate"

    def test_denied_is_candidate(self):
        assert claim_to_artifact_state(ClaimState.DENIED) == "Candidate"


# ── Transition graph ─────────────────────────────────────────────────────────

class TestTransitionGraph:
    def test_all_states_reachable(self):
        reachable = set()
        to_visit = [ClaimState.DRAFT]
        while to_visit:
            s = to_visit.pop()
            if s in reachable:
                continue
            reachable.add(s)
            for t in VALID_TRANSITIONS.get(s, set()):
                if t not in reachable:
                    to_visit.append(t)
        assert reachable == set(ClaimState)

    def test_quarantine_absorbing(self):
        assert VALID_TRANSITIONS[ClaimState.QUARANTINED] == set()

    def test_every_state_can_quarantine(self):
        for state in ClaimState:
            if state == ClaimState.QUARANTINED:
                continue
            assert ClaimState.QUARANTINED in VALID_TRANSITIONS[state], \
                f"{state.value} cannot transition to Quarantined"

    def test_fourteen_states(self):
        assert len(ClaimState) == 14


# ── Bundle transitions ───────────────────────────────────────────────────────

class TestBundleTransitions:
    def _make_bundle(self, bundle_id="test", with_consent=True):
        consent = None
        if with_consent:
            consent = ConsentReceipt(
                consent_id="c1",
                patient_hash=sha256(b"p1"),
                scope="test",
                granted_utc="2026-01-01T00:00:00Z",
            )
        b = MedicalArtifactBundle(bundle_id=bundle_id, consent=consent)
        b.add_artifact("a1", "StructuredRecord", "gzip", sha256(b"a1"), True)
        return b

    def test_happy_path(self):
        b = self._make_bundle()
        for target in [ClaimState.SUBMITTED, ClaimState.ACKNOWLEDGED,
                       ClaimState.IN_REVIEW, ClaimState.APPROVED]:
            r = b.transition(target)
            assert r.allowed, f"Failed at {target.value}: {r.reason_codes}"
        assert b.claim_state == ClaimState.APPROVED
        assert claim_to_artifact_state(b.claim_state) == "Active"

    def test_invalid_skip(self):
        b = self._make_bundle()
        r = b.transition(ClaimState.APPROVED)
        assert not r.allowed
        assert "MS-INVALID-TRANSITION" in r.reason_codes

    def test_quarantine_blocks_escape(self):
        b = self._make_bundle()
        b.transition(ClaimState.SUBMITTED)
        b.transition(ClaimState.QUARANTINED)
        r = b.transition(ClaimState.ACKNOWLEDGED)
        assert not r.allowed
        assert "MS-INVALID-TRANSITION" in r.reason_codes

    def test_revoked_consent_blocks_transition(self):
        b = self._make_bundle()
        b.consent.revoked = True
        b.consent.revoked_utc = "2026-01-02T00:00:00Z"
        r = b.transition(ClaimState.SUBMITTED)
        assert not r.allowed
        assert "MS-CONSENT-REVOKED" in r.reason_codes

    def test_revoked_consent_allows_quarantine(self):
        b = self._make_bundle()
        b.consent.revoked = True
        r = b.transition(ClaimState.QUARANTINED)
        assert r.allowed

    def test_transition_log_grows(self):
        b = self._make_bundle()
        assert len(b.transition_log) == 0
        b.transition(ClaimState.SUBMITTED)
        assert len(b.transition_log) == 1
        b.transition(ClaimState.ACKNOWLEDGED)
        assert len(b.transition_log) == 2


# ── Bundle hashing ───────────────────────────────────────────────────────────

class TestBundleHash:
    def test_composite_hash_deterministic(self):
        b = MedicalArtifactBundle(bundle_id="h1")
        b.add_artifact("a1", "T", "gzip", "aaa", True)
        b.add_artifact("a2", "T", "gzip", "bbb", True)
        h1 = b.bundle_hash()
        h2 = b.bundle_hash()
        assert h1 == h2

    def test_composite_hash_order_independent(self):
        b1 = MedicalArtifactBundle(bundle_id="h2")
        b1.add_artifact("a1", "T", "gzip", "aaa", True)
        b1.add_artifact("a2", "T", "gzip", "bbb", True)

        b2 = MedicalArtifactBundle(bundle_id="h3")
        b2.add_artifact("a2", "T", "gzip", "bbb", True)
        b2.add_artifact("a1", "T", "gzip", "aaa", True)

        assert b1.bundle_hash() == b2.bundle_hash()

    def test_empty_bundle_hash(self):
        b = MedicalArtifactBundle(bundle_id="h4")
        h = b.bundle_hash()
        assert len(h) == 64  # SHA-256 hex


# ── Consent ──────────────────────────────────────────────────────────────────

class TestConsent:
    def test_valid_consent(self):
        c = ConsentReceipt(
            consent_id="c1",
            patient_hash=sha256(b"p"),
            scope="test",
            granted_utc="2026-01-01T00:00:00Z",
        )
        assert c.is_valid()

    def test_revoked_consent(self):
        c = ConsentReceipt(
            consent_id="c2",
            patient_hash=sha256(b"p"),
            scope="test",
            granted_utc="2026-01-01T00:00:00Z",
            revoked=True,
            revoked_utc="2026-06-01T00:00:00Z",
        )
        assert not c.is_valid()

    def test_expired_consent(self):
        c = ConsentReceipt(
            consent_id="c3",
            patient_hash=sha256(b"p"),
            scope="test",
            granted_utc="2020-01-01T00:00:00Z",
            expires_utc="2020-12-31T00:00:00Z",
        )
        assert not c.is_valid()


# ── Medical preflight ────────────────────────────────────────────────────────

class TestMedicalPreflight:
    def test_allow_active_with_consent(self):
        b = MedicalArtifactBundle(
            bundle_id="pf1",
            claim_state=ClaimState.APPROVED,
            consent=ConsentReceipt("c1", sha256(b"p"), "test", "2026-01-01T00:00:00Z"),
        )
        b.add_artifact("a1", "T", "gzip", sha256(b"a1"), True)
        pf = medical_preflight(b)
        assert pf["decision"] == "ALLOW"

    def test_reject_no_consent(self):
        b = MedicalArtifactBundle(bundle_id="pf2", claim_state=ClaimState.APPROVED)
        b.add_artifact("a1", "T", "gzip", sha256(b"a1"), True)
        pf = medical_preflight(b)
        assert pf["decision"] == "REJECT"
        assert "MPF-NO-CONSENT" in pf["reason_codes"]

    def test_hold_candidate(self):
        b = MedicalArtifactBundle(
            bundle_id="pf3",
            claim_state=ClaimState.IN_REVIEW,
            consent=ConsentReceipt("c1", sha256(b"p"), "test", "2026-01-01T00:00:00Z"),
        )
        b.add_artifact("a1", "T", "gzip", sha256(b"a1"), True)
        pf = medical_preflight(b)
        assert pf["decision"] == "HOLD"

    def test_reject_quarantined(self):
        b = MedicalArtifactBundle(
            bundle_id="pf4",
            claim_state=ClaimState.QUARANTINED,
            consent=ConsentReceipt("c1", sha256(b"p"), "test", "2026-01-01T00:00:00Z"),
        )
        b.add_artifact("a1", "T", "gzip", sha256(b"a1"), True)
        pf = medical_preflight(b)
        assert pf["decision"] == "REJECT"
        assert "MPF-QUARANTINED" in pf["reason_codes"]

    def test_reject_empty_bundle(self):
        b = MedicalArtifactBundle(
            bundle_id="pf5",
            claim_state=ClaimState.APPROVED,
            consent=ConsentReceipt("c1", sha256(b"p"), "test", "2026-01-01T00:00:00Z"),
        )
        pf = medical_preflight(b)
        assert pf["decision"] == "REJECT"
        assert "MPF-EMPTY-BUNDLE" in pf["reason_codes"]

    def test_reject_phi_boundary(self):
        b = MedicalArtifactBundle(
            bundle_id="pf6",
            claim_state=ClaimState.APPROVED,
            consent=ConsentReceipt("c1", sha256(b"p"), "test", "2026-01-01T00:00:00Z"),
        )
        b.add_artifact("a1", "T", "gzip", sha256(b"a1"), phi_fields_hashed=False)
        pf = medical_preflight(b)
        assert pf["decision"] == "REJECT"
        assert "MPF-PHI-BOUNDARY" in pf["reason_codes"]

    def test_reject_missing_hash(self):
        b = MedicalArtifactBundle(
            bundle_id="pf7",
            claim_state=ClaimState.APPROVED,
            consent=ConsentReceipt("c1", sha256(b"p"), "test", "2026-01-01T00:00:00Z"),
        )
        b.add_artifact("a1", "T", "gzip", "", True)  # empty hash
        pf = medical_preflight(b)
        assert pf["decision"] == "REJECT"
        assert "MPF-MISSING-HASH" in pf["reason_codes"]
