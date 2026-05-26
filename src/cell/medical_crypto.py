"""Gate 8F: Consent-Gated Medical Artifact Encryption

HIPAA-aligned technical safeguard prototype for encryption at rest,
integrity, audit, and consent-gated key release.

This is NOT HIPAA-compliant by itself. HIPAA compliance requires
administrative, physical, and technical safeguards plus a BAA.
This module implements one technical safeguard: encryption at rest
with consent-linked key lifecycle.

Design:
  artifact → compress → encrypt(AES-256-GCM) → hash(ciphertext) → receipt
  content_hash = SHA-256 of ciphertext (never plaintext)
  plaintext_hash = local/private, never chain-facing
  consent revocation → key destroyed → ciphertext unrecoverable

Key lifecycle:
  1. Key generated per-artifact, linked to consent_id
  2. Key stored in KeyVault (in-memory for prototype)
  3. Key released only if consent is valid
  4. Consent revocation destroys key (zeroed + removed)
  5. Destroyed key makes ciphertext permanently unrecoverable
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ── Encryption Primitives ────────────────────────────────────────────────────

def generate_key() -> bytes:
    """Generate a 256-bit AES key."""
    return AESGCM.generate_key(bit_length=256)


def encrypt_artifact(plaintext: bytes, key: bytes) -> Tuple[bytes, bytes]:
    """Encrypt artifact content with AES-256-GCM.

    Returns (ciphertext, nonce). The nonce is 96 bits (12 bytes),
    generated from a CSPRNG.
    """
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return ciphertext, nonce


def decrypt_artifact(ciphertext: bytes, key: bytes, nonce: bytes) -> bytes:
    """Decrypt artifact content. Raises on tamper or wrong key."""
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)


# ── AAD-Aware Encryption (Gate 9) ───────────────────────────────────────────

def encrypt_with_aad(plaintext: bytes, key: bytes, aad: bytes) -> Tuple[bytes, bytes]:
    """AES-256-GCM encrypt with additional authenticated data.

    AAD is verified during decryption but never encrypted.
    Use codebook hash as AAD to bind ciphertext to a specific artifact.
    Returns (ciphertext, nonce).
    """
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return ciphertext, nonce


def decrypt_with_aad(ciphertext: bytes, key: bytes, nonce: bytes, aad: bytes) -> bytes:
    """Decrypt and verify AAD. Fails if AAD doesn't match (wrong artifact)."""
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, aad)


def wrap_key(dek: bytes, master_key: bytes) -> Tuple[bytes, bytes]:
    """AES-256-GCM key wrapping. Returns (wrapped_dek, wrap_nonce)."""
    nonce = os.urandom(12)
    aesgcm = AESGCM(master_key)
    wrapped = aesgcm.encrypt(nonce, dek, None)
    return wrapped, nonce


def unwrap_key(wrapped_dek: bytes, master_key: bytes, wrap_nonce: bytes) -> bytes:
    """Unwrap DEK. Raises InvalidTag if wrong master key."""
    aesgcm = AESGCM(master_key)
    return aesgcm.decrypt(wrap_nonce, wrapped_dek, None)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Key Vault ────────────────────────────────────────────────────────────────

@dataclass
class KeyEntry:
    """A key entry linked to a consent_id."""
    consent_id: str
    key: bytes
    created_utc: str
    destroyed: bool = False
    destroyed_utc: Optional[str] = None


class KeyVault:
    """In-memory key vault for prototype. Production would use HSM/KMS.

    Keys are linked to consent_ids. Consent revocation destroys the key
    by zeroing the bytes and marking it destroyed.
    """

    def __init__(self):
        self._keys: Dict[str, KeyEntry] = {}

    def store_key(self, artifact_id: str, consent_id: str, key: bytes) -> None:
        """Store a key linked to a consent."""
        self._keys[artifact_id] = KeyEntry(
            consent_id=consent_id,
            key=bytes(key),  # copy
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def release_key(self, artifact_id: str, consent_id: str) -> Optional[bytes]:
        """Release key only if consent matches and key is not destroyed.

        Returns None if:
          - artifact_id not found
          - consent_id doesn't match
          - key has been destroyed
        """
        entry = self._keys.get(artifact_id)
        if entry is None:
            return None
        if entry.destroyed:
            return None
        if entry.consent_id != consent_id:
            return None
        return bytes(entry.key)  # copy

    def destroy_key(self, artifact_id: str) -> bool:
        """Destroy a key by zeroing and marking destroyed.

        Returns True if key was found and destroyed.
        """
        entry = self._keys.get(artifact_id)
        if entry is None:
            return False
        if entry.destroyed:
            return True  # already destroyed

        # Zero the key bytes
        entry.key = b'\x00' * len(entry.key)
        entry.destroyed = True
        entry.destroyed_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return True

    def revoke_consent(self, consent_id: str) -> int:
        """Destroy all keys linked to a consent_id.

        Returns count of keys destroyed.
        """
        count = 0
        for artifact_id, entry in self._keys.items():
            if entry.consent_id == consent_id and not entry.destroyed:
                self.destroy_key(artifact_id)
                count += 1
        return count

    def is_destroyed(self, artifact_id: str) -> Optional[bool]:
        """Check if a key is destroyed. None if not found."""
        entry = self._keys.get(artifact_id)
        if entry is None:
            return None
        return entry.destroyed


# ── Encrypted Artifact Envelope ──────────────────────────────────────────────

@dataclass
class EncryptedEnvelope:
    """An encrypted artifact with its metadata.

    content_hash is SHA-256 of ciphertext (chain-facing).
    plaintext_hash is SHA-256 of plaintext (local/private, never on-chain).
    """
    artifact_id: str
    ciphertext: bytes
    nonce: bytes
    content_hash: str          # SHA-256 of ciphertext — this goes on-chain
    plaintext_hash: str        # SHA-256 of plaintext — local only, never chain
    codec: str                 # compression codec used before encryption
    cipher: str = "AES-256-GCM"
    consent_id: str = ""
    encrypted_utc: str = ""

    def to_receipt_fields(self) -> Dict[str, Any]:
        """Fields safe to include in a receipt (no PHI, no key material)."""
        return {
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,  # ciphertext hash
            "codec": self.codec,
            "cipher": self.cipher,
            "ciphertext_bytes": len(self.ciphertext),
            "nonce_hex": self.nonce.hex(),
            "consent_id": self.consent_id,
            "encrypted_utc": self.encrypted_utc,
            # plaintext_hash intentionally excluded from receipt
        }


def encrypt_compressed_artifact(
    compressed_data: bytes,
    artifact_id: str,
    codec: str,
    consent_id: str,
    vault: KeyVault,
    plaintext_for_local_hash: Optional[bytes] = None,
) -> EncryptedEnvelope:
    """Compress → Encrypt → Hash(ciphertext) → Store key in vault.

    Args:
        compressed_data: Already-compressed artifact bytes.
        artifact_id: Unique artifact identifier.
        codec: Compression codec used (e.g. "gzip", "VQ_k256").
        consent_id: Consent receipt ID that gates key release.
        vault: KeyVault to store the encryption key.
        plaintext_for_local_hash: Original uncompressed bytes for local hash
            (optional, never included in receipt).

    Returns:
        EncryptedEnvelope with ciphertext and metadata.
    """
    key = generate_key()
    ciphertext, nonce = encrypt_artifact(compressed_data, key)

    vault.store_key(artifact_id, consent_id, key)

    plaintext_hash = ""
    if plaintext_for_local_hash is not None:
        plaintext_hash = sha256_bytes(plaintext_for_local_hash)

    return EncryptedEnvelope(
        artifact_id=artifact_id,
        ciphertext=ciphertext,
        nonce=nonce,
        content_hash=sha256_bytes(ciphertext),
        plaintext_hash=plaintext_hash,
        codec=codec,
        cipher="AES-256-GCM",
        consent_id=consent_id,
        encrypted_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
