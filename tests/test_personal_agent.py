"""Tests for personal_agent.py — Gate 9 personal encryption lifecycle."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cell.personal_agent import PersonalEncryptionAgent, PersonalEnvelope
from cell.device_fingerprint import generate_salt, derive_device_key
from cell.medical_crypto import generate_key, sha256_bytes
from cryptography.exceptions import InvalidTag


# Fixed hardware IDs for reproducible tests
HW_DEVICE_A = {"machine_id": "device-A-001", "cpu_model": "test-cpu-A"}
HW_DEVICE_B = {"machine_id": "device-B-002", "cpu_model": "test-cpu-B"}
PASSPHRASE = "test-passphrase-gate9"


class TestEncryptDecrypt:
    def test_roundtrip(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        data = b"sensitive medical record content"
        envelope = agent.encrypt_artifact(data, "art-001", codec="gzip")
        recovered = agent.decrypt_artifact(envelope)
        assert recovered == data

    def test_roundtrip_no_compression(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        data = b"already compressed VQ bytes"
        envelope = agent.encrypt_artifact(data, "art-002", codec="none")
        recovered = agent.decrypt_artifact(envelope)
        assert recovered == data

    def test_ciphertext_differs_from_plaintext(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        data = b"test data for ciphertext check"
        envelope = agent.encrypt_artifact(data, "art-003", codec="none")
        assert envelope.ciphertext != data

    def test_content_hash_is_ciphertext_hash(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        envelope = agent.encrypt_artifact(b"data", "art-004", codec="none")
        assert envelope.content_hash == sha256_bytes(envelope.ciphertext)

    def test_codebook_hash_as_aad(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        cb_hash = sha256_bytes(b"codebook-content")
        envelope = agent.encrypt_artifact(
            b"artifact data", "art-005", codec="none",
            codebook_hash=cb_hash,
        )
        assert envelope.codebook_hash_aad == cb_hash
        recovered = agent.decrypt_artifact(envelope)
        assert recovered == b"artifact data"


class TestDeviceBinding:
    def test_wrong_device_fails(self):
        agent_a = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        envelope = agent_a.encrypt_artifact(b"bound data", "art-dev-1", codec="none")

        # Different device, same passphrase
        agent_b = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_B)
        with pytest.raises(InvalidTag):
            agent_b.decrypt_artifact(envelope)

    def test_wrong_passphrase_fails(self):
        agent1 = PersonalEncryptionAgent("correct-pass", hw_ids=HW_DEVICE_A)
        envelope = agent1.encrypt_artifact(b"secret", "art-pass-1", codec="none")

        # Same device, wrong passphrase
        agent2 = PersonalEncryptionAgent("wrong-pass", hw_ids=HW_DEVICE_A)
        with pytest.raises(InvalidTag):
            agent2.decrypt_artifact(envelope)

    def test_tampered_aad_fails(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        envelope = agent.encrypt_artifact(
            b"aad test", "art-aad-1", codec="none",
            codebook_hash="original_codebook_hash",
        )
        # Tamper the codebook hash
        envelope.codebook_hash_aad = "tampered_codebook_hash"
        with pytest.raises(InvalidTag):
            agent.decrypt_artifact(envelope)


class TestConsent:
    def test_consent_gated_decrypt(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        envelope = agent.encrypt_artifact(
            b"consent data", "art-c-1", codec="none", consent_id="consent-001",
        )
        # Decrypt with matching consent
        recovered = agent.decrypt_artifact(envelope, consent_id="consent-001")
        assert recovered == b"consent data"

    def test_revoke_consent_blocks_decrypt(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        envelope = agent.encrypt_artifact(
            b"revocable", "art-c-2", codec="none", consent_id="consent-rev",
        )
        agent.revoke_consent("consent-rev")
        with pytest.raises(PermissionError, match="revoked"):
            agent.decrypt_artifact(envelope, consent_id="consent-rev")

    def test_revoke_consent_destroys_wrapped_dek(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        envelope = agent.encrypt_artifact(
            b"destroy test", "art-c-3", codec="none", consent_id="consent-destroy",
        )
        dek_before = envelope.wrapped_dek
        agent.revoke_consent("consent-destroy")
        # Wrapped DEK should be zeroed
        assert envelope.wrapped_dek == b'\x00' * len(dek_before)

    def test_consent_mismatch_denied(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        envelope = agent.encrypt_artifact(
            b"data", "art-c-4", codec="none", consent_id="consent-A",
        )
        with pytest.raises(PermissionError, match="does not match"):
            agent.decrypt_artifact(envelope, consent_id="consent-B")


class TestGrantRevoke:
    def test_grant_and_revoke(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        agent.encrypt_artifact(b"shared data", "art-g-1", codec="none")

        requester_key = generate_key()
        grant = agent.grant_access("art-g-1", "dr-smith", "read", requester_key)
        assert grant.requester_id == "dr-smith"
        assert grant.scope == "read"
        assert not grant.revoked

        count = agent.revoke_access("art-g-1", "dr-smith")
        assert count == 1
        assert grant.revoked
        assert grant.wrapped_dek == b'\x00' * len(grant.wrapped_dek)

    def test_revoke_all(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        agent.encrypt_artifact(b"multi-grant", "art-g-2", codec="none")

        agent.grant_access("art-g-2", "user-1", "read", generate_key())
        agent.grant_access("art-g-2", "user-2", "full", generate_key())

        count = agent.revoke_access("art-g-2")
        assert count == 2

    def test_grant_nonexistent_raises(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        with pytest.raises(KeyError):
            agent.grant_access("no-such-art", "user", "read", generate_key())


class TestAuditLog:
    def test_audit_records_operations(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        agent.encrypt_artifact(b"data", "art-a-1", codec="none")
        envelope = agent._envelopes["art-a-1"]
        agent.decrypt_artifact(envelope)

        log = agent.audit_log()
        assert len(log) == 2
        assert log[0]["action"] == "encrypt"
        assert log[1]["action"] == "decrypt"
        assert log[0]["artifact_id"] == "art-a-1"

    def test_audit_records_deny(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        agent.encrypt_artifact(
            b"data", "art-a-2", codec="none", consent_id="c-deny",
        )
        agent.revoke_consent("c-deny")
        envelope = agent._envelopes["art-a-2"]
        try:
            agent.decrypt_artifact(envelope, consent_id="c-deny")
        except PermissionError:
            pass
        log = agent.audit_log()
        actions = [e["action"] for e in log]
        assert "deny" in actions


class TestExplain:
    def test_explain_active(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        agent.encrypt_artifact(b"data", "art-e-1", codec="gzip", consent_id="c-e-1")
        text = agent.explain("art-e-1")
        assert "art-e-1" in text
        assert "AES-256-GCM" in text
        assert "ACTIVE" in text

    def test_explain_revoked(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        agent.encrypt_artifact(b"data", "art-e-2", codec="none", consent_id="c-e-2")
        agent.revoke_consent("c-e-2")
        text = agent.explain("art-e-2")
        assert "REVOKED" in text
        assert "DESTROYED" in text

    def test_explain_missing(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        text = agent.explain("nonexistent")
        assert "No envelope found" in text


class TestReceiptSafety:
    def test_no_raw_serials_in_receipt(self):
        agent = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
        envelope = agent.encrypt_artifact(b"data", "art-r-1", codec="none")
        fields = envelope.to_receipt_fields()

        # Must contain fingerprint hash
        assert "hardware_fingerprint_hash" in fields
        assert len(fields["hardware_fingerprint_hash"]) == 64

        # Must NOT contain raw hardware values
        field_values = str(fields)
        for raw_val in HW_DEVICE_A.values():
            assert raw_val not in field_values

        # Must NOT contain key material
        assert "wrapped_dek" not in fields
        assert "device_key" not in fields
        assert "passphrase" not in fields
        assert "salt" not in fields
