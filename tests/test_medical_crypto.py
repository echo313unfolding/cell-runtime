"""Tests for medical_crypto.py — Gate 8F encryption coverage."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cell.medical_crypto import (
    generate_key,
    encrypt_artifact,
    decrypt_artifact,
    sha256_bytes,
    KeyVault,
    EncryptedEnvelope,
    encrypt_compressed_artifact,
)
from cryptography.exceptions import InvalidTag


# ── Primitives ───────────────────────────────────────────────────────────────

class TestPrimitives:
    def test_generate_key_length(self):
        key = generate_key()
        assert len(key) == 32  # 256 bits

    def test_generate_key_unique(self):
        k1 = generate_key()
        k2 = generate_key()
        assert k1 != k2

    def test_encrypt_decrypt_roundtrip(self):
        key = generate_key()
        plaintext = b"FHIR prior-auth resource content"
        ciphertext, nonce = encrypt_artifact(plaintext, key)
        recovered = decrypt_artifact(ciphertext, key, nonce)
        assert recovered == plaintext

    def test_ciphertext_differs_from_plaintext(self):
        key = generate_key()
        plaintext = b"sensitive medical data"
        ciphertext, _ = encrypt_artifact(plaintext, key)
        assert ciphertext != plaintext

    def test_wrong_key_fails(self):
        key1 = generate_key()
        key2 = generate_key()
        plaintext = b"test data"
        ciphertext, nonce = encrypt_artifact(plaintext, key1)
        with pytest.raises(InvalidTag):
            decrypt_artifact(ciphertext, key2, nonce)

    def test_tampered_ciphertext_fails(self):
        key = generate_key()
        plaintext = b"integrity test"
        ciphertext, nonce = encrypt_artifact(plaintext, key)
        tampered = bytearray(ciphertext)
        tampered[0] ^= 0xFF
        with pytest.raises(InvalidTag):
            decrypt_artifact(bytes(tampered), key, nonce)

    def test_nonce_is_12_bytes(self):
        key = generate_key()
        _, nonce = encrypt_artifact(b"test", key)
        assert len(nonce) == 12


# ── KeyVault ─────────────────────────────────────────────────────────────────

class TestKeyVault:
    def test_store_and_release(self):
        vault = KeyVault()
        key = generate_key()
        vault.store_key("art1", "consent1", key)
        released = vault.release_key("art1", "consent1")
        assert released == key

    def test_wrong_consent_blocks_release(self):
        vault = KeyVault()
        vault.store_key("art1", "consent1", generate_key())
        assert vault.release_key("art1", "wrong_consent") is None

    def test_missing_artifact_returns_none(self):
        vault = KeyVault()
        assert vault.release_key("nonexistent", "c1") is None

    def test_destroy_key(self):
        vault = KeyVault()
        key = generate_key()
        vault.store_key("art1", "consent1", key)
        assert vault.destroy_key("art1")
        assert vault.release_key("art1", "consent1") is None
        assert vault.is_destroyed("art1") is True

    def test_destroy_zeros_key(self):
        vault = KeyVault()
        vault.store_key("art1", "consent1", generate_key())
        vault.destroy_key("art1")
        entry = vault._keys["art1"]
        assert entry.key == b'\x00' * 32

    def test_revoke_consent_destroys_all_linked_keys(self):
        vault = KeyVault()
        vault.store_key("art1", "consent_shared", generate_key())
        vault.store_key("art2", "consent_shared", generate_key())
        vault.store_key("art3", "consent_other", generate_key())
        count = vault.revoke_consent("consent_shared")
        assert count == 2
        assert vault.is_destroyed("art1") is True
        assert vault.is_destroyed("art2") is True
        assert vault.is_destroyed("art3") is False

    def test_double_destroy_is_idempotent(self):
        vault = KeyVault()
        vault.store_key("art1", "c1", generate_key())
        assert vault.destroy_key("art1")
        assert vault.destroy_key("art1")  # second call still True

    def test_is_destroyed_none_for_missing(self):
        vault = KeyVault()
        assert vault.is_destroyed("nonexistent") is None


# ── Encrypted Envelope ───────────────────────────────────────────────────────

class TestEncryptedEnvelope:
    def test_receipt_fields_exclude_plaintext_hash(self):
        env = EncryptedEnvelope(
            artifact_id="a1",
            ciphertext=b"encrypted",
            nonce=b'\x00' * 12,
            content_hash="abc123",
            plaintext_hash="secret_local_hash",
            codec="gzip",
            consent_id="c1",
        )
        fields = env.to_receipt_fields()
        assert "content_hash" in fields
        assert "plaintext_hash" not in fields
        assert fields["cipher"] == "AES-256-GCM"

    def test_receipt_fields_contain_nonce_hex(self):
        nonce = b'\xab\xcd' + b'\x00' * 10
        env = EncryptedEnvelope(
            artifact_id="a1",
            ciphertext=b"x",
            nonce=nonce,
            content_hash="h",
            plaintext_hash="p",
            codec="gzip",
        )
        fields = env.to_receipt_fields()
        assert fields["nonce_hex"] == nonce.hex()


# ── Full Pipeline ────────────────────────────────────────────────────────────

class TestEncryptCompressedArtifact:
    def test_full_pipeline(self):
        vault = KeyVault()
        compressed = b"compressed FHIR data bytes"
        original = b"original FHIR JSON"

        env = encrypt_compressed_artifact(
            compressed, "fhir-001", "gzip", "consent-001", vault,
            plaintext_for_local_hash=original,
        )

        assert env.content_hash == sha256_bytes(env.ciphertext)
        assert env.plaintext_hash == sha256_bytes(original)
        assert env.codec == "gzip"
        assert env.cipher == "AES-256-GCM"
        assert env.consent_id == "consent-001"

        # Decrypt via vault
        key = vault.release_key("fhir-001", "consent-001")
        assert key is not None
        recovered = decrypt_artifact(env.ciphertext, key, env.nonce)
        assert recovered == compressed

    def test_consent_revocation_blocks_decrypt(self):
        vault = KeyVault()
        env = encrypt_compressed_artifact(
            b"data", "art1", "gzip", "consent-revoke", vault,
        )

        # Before revocation — key available
        assert vault.release_key("art1", "consent-revoke") is not None

        # Revoke
        vault.revoke_consent("consent-revoke")

        # After revocation — key gone
        assert vault.release_key("art1", "consent-revoke") is None
        assert vault.is_destroyed("art1") is True

    def test_destroyed_key_makes_decrypt_fail(self):
        vault = KeyVault()
        env = encrypt_compressed_artifact(
            b"sensitive", "art2", "VQ_k256", "consent-destroy", vault,
        )

        vault.destroy_key("art2")

        # Cannot get key
        key = vault.release_key("art2", "consent-destroy")
        assert key is None

        # Even if someone had the zeroed key, GCM auth would fail
        zeroed_key = b'\x00' * 32
        with pytest.raises(InvalidTag):
            decrypt_artifact(env.ciphertext, zeroed_key, env.nonce)

    def test_content_hash_is_ciphertext_hash(self):
        vault = KeyVault()
        env = encrypt_compressed_artifact(
            b"data", "art3", "gzip", "c1", vault,
        )
        assert env.content_hash == sha256_bytes(env.ciphertext)
        # Content hash is NOT the plaintext hash
        assert env.content_hash != sha256_bytes(b"data")
