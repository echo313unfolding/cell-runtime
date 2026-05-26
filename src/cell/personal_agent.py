"""Gate 9: Personal Encryption Agent — device-bound key orchestration.

User-controlled local encryption with envelope encryption:
  DEK (data encryption key) wraps the artifact.
  Device master key wraps the DEK.
  Codebook hash binds ciphertext to a specific artifact via AAD.

Three-property key model:
  something you know  = passphrase
  something you have  = device fingerprint (hardware IDs)
  something it is     = artifact codebook hash (AAD in AES-GCM)

Recovery warning: hardware changes or lost passphrase make artifacts
permanently unrecoverable. This is by design.
"""

from __future__ import annotations

import gzip
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cell.device_fingerprint import (
    collect_hardware_ids,
    derive_device_key,
    generate_salt,
    hardware_fingerprint_hash,
)
from cell.medical_crypto import (
    decrypt_with_aad,
    encrypt_with_aad,
    generate_key,
    sha256_bytes,
    unwrap_key,
    wrap_key,
)


# ── Audit Log ───────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    timestamp_utc: str
    action: str          # encrypt, decrypt, grant, revoke, deny
    artifact_id: str
    detail: str = ""


# ── Encrypted Envelope (Gate 9) ─────────────────────────────────────────────

@dataclass
class PersonalEnvelope:
    """Encrypted artifact envelope with device-bound key wrapping."""
    artifact_id: str
    ciphertext: bytes
    nonce: bytes
    wrapped_dek: bytes
    wrap_nonce: bytes
    salt: bytes
    hardware_fingerprint_hash: str
    codebook_hash_aad: str       # codebook hash used as AAD
    content_hash: str            # SHA-256 of ciphertext
    codec: str
    cipher: str = "AES-256-GCM"
    consent_id: str = ""
    encrypted_utc: str = ""

    def to_receipt_fields(self) -> Dict[str, Any]:
        """Fields safe to include in a receipt (no secrets, no raw serials)."""
        return {
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "codebook_hash_aad": self.codebook_hash_aad,
            "hardware_fingerprint_hash": self.hardware_fingerprint_hash,
            "codec": self.codec,
            "cipher": self.cipher,
            "ciphertext_bytes": len(self.ciphertext),
            "nonce_hex": self.nonce.hex(),
            "consent_id": self.consent_id,
            "encrypted_utc": self.encrypted_utc,
        }


# ── Access Grants ───────────────────────────────────────────────────────────

@dataclass
class AccessGrant:
    artifact_id: str
    requester_id: str
    scope: str             # e.g. "read", "read_summary", "full"
    wrapped_dek: bytes     # DEK wrapped with requester-specific key
    wrap_nonce: bytes
    granted_utc: str
    revoked: bool = False
    revoked_utc: Optional[str] = None


# ── Personal Encryption Agent ───────────────────────────────────────────────

class PersonalEncryptionAgent:
    """User-facing encryption agent with device-bound keys.

    Lifecycle:
      1. Agent init: collect hardware IDs, derive device master key
      2. encrypt_artifact: compress → DEK → encrypt(AAD) → wrap(DEK)
      3. decrypt_artifact: check consent → unwrap DEK → decrypt(AAD)
      4. grant_access / revoke_access: scoped key sharing
      5. audit_log: all decisions with timestamps
    """

    def __init__(self, passphrase: str, hw_ids: Optional[Dict[str, str]] = None):
        """Initialize with passphrase and hardware IDs.

        Args:
            passphrase: User passphrase (something you know).
            hw_ids: Hardware identifiers. If None, collected from this device.
        """
        self._hw_ids = hw_ids if hw_ids is not None else collect_hardware_ids()
        self._salt = generate_salt()
        self._hw_hash = hardware_fingerprint_hash(self._hw_ids)
        self._device_key = derive_device_key(passphrase, self._salt, self._hw_ids)

        # Storage
        self._envelopes: Dict[str, PersonalEnvelope] = {}
        self._grants: Dict[str, List[AccessGrant]] = {}  # artifact_id → grants
        self._consents: Dict[str, bool] = {}  # consent_id → active
        self._audit: List[AuditEntry] = []

    @property
    def salt(self) -> bytes:
        return self._salt

    @property
    def hardware_hash(self) -> str:
        return self._hw_hash

    def _log(self, action: str, artifact_id: str, detail: str = "") -> None:
        self._audit.append(AuditEntry(
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            action=action,
            artifact_id=artifact_id,
            detail=detail,
        ))

    def encrypt_artifact(
        self,
        data: bytes,
        artifact_id: str,
        codec: str = "gzip",
        consent_id: str = "",
        codebook_hash: Optional[str] = None,
    ) -> PersonalEnvelope:
        """Compress, encrypt, and wrap with device-bound key.

        Args:
            data: Raw artifact bytes.
            artifact_id: Unique artifact identifier.
            codec: Compression codec ("gzip", "none", or VQ variant).
            consent_id: Consent ID that gates access.
            codebook_hash: VQ codebook hash for AAD binding.
                If None, uses SHA-256 of compressed data as AAD.
        """
        # Compress
        if codec == "gzip":
            compressed = gzip.compress(data, compresslevel=9)
        elif codec == "none":
            compressed = data
        else:
            # For VQ or other codecs, data is assumed already compressed
            compressed = data

        # AAD: codebook hash binds ciphertext to specific artifact identity
        aad = (codebook_hash or sha256_bytes(compressed)).encode("utf-8")

        # Generate DEK, encrypt artifact
        dek = generate_key()
        ciphertext, nonce = encrypt_with_aad(compressed, dek, aad)

        # Wrap DEK with device master key
        wrapped_dek, wrap_nonce = wrap_key(dek, self._device_key)

        # Track consent
        if consent_id:
            self._consents[consent_id] = True

        envelope = PersonalEnvelope(
            artifact_id=artifact_id,
            ciphertext=ciphertext,
            nonce=nonce,
            wrapped_dek=wrapped_dek,
            wrap_nonce=wrap_nonce,
            salt=self._salt,
            hardware_fingerprint_hash=self._hw_hash,
            codebook_hash_aad=codebook_hash or sha256_bytes(compressed),
            content_hash=sha256_bytes(ciphertext),
            codec=codec,
            consent_id=consent_id,
            encrypted_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        self._envelopes[artifact_id] = envelope
        self._log("encrypt", artifact_id, f"codec={codec}, consent={consent_id}")

        return envelope

    def decrypt_artifact(
        self,
        envelope: PersonalEnvelope,
        consent_id: str = "",
    ) -> bytes:
        """Decrypt artifact: check consent, unwrap DEK, decrypt with AAD.

        Raises:
            PermissionError: If consent is revoked or doesn't match.
            cryptography.exceptions.InvalidTag: If wrong key, wrong device,
                or tampered codebook hash.
        """
        # Check consent
        if consent_id:
            if consent_id in self._consents and not self._consents[consent_id]:
                self._log("deny", envelope.artifact_id, "consent revoked")
                raise PermissionError(
                    f"Consent {consent_id} has been revoked"
                )
            if envelope.consent_id and envelope.consent_id != consent_id:
                self._log("deny", envelope.artifact_id, "consent mismatch")
                raise PermissionError(
                    f"Consent {consent_id} does not match envelope consent {envelope.consent_id}"
                )

        # Unwrap DEK
        dek = unwrap_key(envelope.wrapped_dek, self._device_key, envelope.wrap_nonce)

        # Decrypt with AAD
        aad = envelope.codebook_hash_aad.encode("utf-8")
        compressed = decrypt_with_aad(envelope.ciphertext, dek, envelope.nonce, aad)

        # Decompress
        if envelope.codec == "gzip":
            plaintext = gzip.decompress(compressed)
        else:
            plaintext = compressed

        self._log("decrypt", envelope.artifact_id, f"consent={consent_id}")
        return plaintext

    def grant_access(
        self,
        artifact_id: str,
        requester_id: str,
        scope: str,
        requester_key: bytes,
    ) -> AccessGrant:
        """Create scoped access grant for a specific requester.

        Wraps the artifact's DEK with the requester's key so they can decrypt.
        """
        envelope = self._envelopes.get(artifact_id)
        if envelope is None:
            raise KeyError(f"No envelope for artifact {artifact_id}")

        # Unwrap our DEK first
        dek = unwrap_key(envelope.wrapped_dek, self._device_key, envelope.wrap_nonce)

        # Re-wrap for requester
        req_wrapped_dek, req_wrap_nonce = wrap_key(dek, requester_key)

        grant = AccessGrant(
            artifact_id=artifact_id,
            requester_id=requester_id,
            scope=scope,
            wrapped_dek=req_wrapped_dek,
            wrap_nonce=req_wrap_nonce,
            granted_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        self._grants.setdefault(artifact_id, []).append(grant)
        self._log("grant", artifact_id, f"requester={requester_id}, scope={scope}")

        return grant

    def revoke_access(
        self,
        artifact_id: str,
        requester_id: Optional[str] = None,
    ) -> int:
        """Revoke access by destroying wrapped DEKs.

        If requester_id is None, revokes all grants for the artifact.
        Returns count of grants revoked.
        """
        grants = self._grants.get(artifact_id, [])
        count = 0
        for grant in grants:
            if grant.revoked:
                continue
            if requester_id is not None and grant.requester_id != requester_id:
                continue
            grant.wrapped_dek = b'\x00' * len(grant.wrapped_dek)
            grant.revoked = True
            grant.revoked_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            count += 1

        target = requester_id or "all"
        self._log("revoke", artifact_id, f"requester={target}, count={count}")
        return count

    def revoke_consent(self, consent_id: str) -> int:
        """Revoke consent: mark inactive and destroy all wrapped DEKs for linked artifacts.

        Returns count of envelopes affected.
        """
        self._consents[consent_id] = False
        count = 0
        for artifact_id, envelope in self._envelopes.items():
            if envelope.consent_id == consent_id:
                # Zero the wrapped DEK in the envelope
                envelope.wrapped_dek = b'\x00' * len(envelope.wrapped_dek)
                # Also revoke all grants
                self.revoke_access(artifact_id)
                count += 1
        self._log("revoke_consent", consent_id, f"artifacts_affected={count}")
        return count

    def audit_log(self) -> List[Dict[str, str]]:
        """All encrypt/decrypt/grant/revoke decisions with timestamps."""
        return [
            {
                "timestamp_utc": e.timestamp_utc,
                "action": e.action,
                "artifact_id": e.artifact_id,
                "detail": e.detail,
            }
            for e in self._audit
        ]

    def explain(self, artifact_id: str) -> str:
        """Human-readable explanation of current artifact state."""
        envelope = self._envelopes.get(artifact_id)
        if envelope is None:
            return f"No envelope found for artifact '{artifact_id}'."

        lines = [
            f"Artifact: {artifact_id}",
            f"  Cipher: {envelope.cipher}",
            f"  Codec: {envelope.codec}",
            f"  Ciphertext hash: {envelope.content_hash[:16]}...",
            f"  Codebook AAD: {envelope.codebook_hash_aad[:16]}...",
            f"  Device bound: {envelope.hardware_fingerprint_hash[:16]}...",
            f"  Encrypted: {envelope.encrypted_utc}",
        ]

        if envelope.consent_id:
            active = self._consents.get(envelope.consent_id, False)
            lines.append(f"  Consent: {envelope.consent_id} ({'ACTIVE' if active else 'REVOKED'})")

        # Check if wrapped DEK is zeroed (revoked)
        if envelope.wrapped_dek == b'\x00' * len(envelope.wrapped_dek):
            lines.append("  DEK: DESTROYED (consent revoked, artifact unrecoverable)")
        else:
            lines.append("  DEK: wrapped (device-bound, requires passphrase + hardware)")

        grants = self._grants.get(artifact_id, [])
        if grants:
            active_grants = [g for g in grants if not g.revoked]
            revoked_grants = [g for g in grants if g.revoked]
            lines.append(f"  Grants: {len(active_grants)} active, {len(revoked_grants)} revoked")
            for g in active_grants:
                lines.append(f"    - {g.requester_id} ({g.scope}) granted {g.granted_utc}")

        return "\n".join(lines)
