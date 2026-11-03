"""Vault bridge — minimal store/load/verify for cell evidence records.

Uses the vendored crypto_v2 primitives directly. No dependency on the
full almanac_core Vault container — just encrypted files with receipts.

Directory layout:
    vault_root/
    ├── vault_salt.key       # 32-byte per-vault salt (generated once)
    ├── scrypt_salt.key      # 32-byte KDF salt (generated once)
    ├── evidence/            # Encrypted .v2.json envelopes + .blob
    ├── receipts/            # Plaintext discovery receipts (no PII)
    └── chain/               # Chain hash files

File permissions: 0600 (owner read/write only).
"""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from cell.vault.crypto_v2 import (
    encrypt_evidence, decrypt_evidence, validate_secret,
    generate_salt, EncryptedEnvelope, sha256_hex,
)
from cell.vault.structure_key import build_structure_context

_FILE_MODE = 0o600
_OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC


def _secure_write(path: Path, content: str) -> None:
    """Write text with owner-only permissions from birth."""
    tmp = path.parent / (path.name + ".tmp")
    fd = os.open(str(tmp), _OPEN_FLAGS, _FILE_MODE)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.rename(path)


def _secure_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes with owner-only permissions from birth."""
    tmp = path.parent / (path.name + ".tmp")
    fd = os.open(str(tmp), _OPEN_FLAGS, _FILE_MODE)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.rename(path)


def _canonical(obj) -> str:
    """Deterministic JSON: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _receipt_hash(receipt: dict) -> str:
    """SHA-256 of canonical receipt JSON."""
    return hashlib.sha256(_canonical(receipt).encode()).hexdigest()


def _discovery_receipt(
    user_commit: str,
    evidence_hash: str,
    source: str,
    previous_hash: str = "",
    notes: str = "",
) -> dict:
    """Create a discovery receipt for vaulted evidence."""
    return {
        "schema": "almanac.record_discovery.v1",
        "receipt_id": str(uuid.uuid4()),
        "user_commitment": user_commit,
        "service_id": "cell_runtime_local",
        "broker_id": source,
        "record_category": "other",
        "confidence": 0.0,
        "evidence_hash": evidence_hash,
        "discovered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_pii_stored": False,
        "previous_receipt_hash": previous_hash,
        "notes": notes or "Local import via cell-runtime vault bridge.",
    }


class VaultBridge:
    """Minimal vault operations for cell-runtime evidence records.

    Handles salt generation, encryption, file storage, and receipts.
    Thread-safe (stateless per operation — salts read fresh each call).
    """

    def __init__(self, vault_dir: str | Path):
        self.root = Path(vault_dir)

    def init(self) -> None:
        """Initialize vault directory and salts if not present."""
        for d in ("evidence", "receipts", "chain"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

        vault_salt_path = self.root / "vault_salt.key"
        scrypt_salt_path = self.root / "scrypt_salt.key"

        if not vault_salt_path.exists():
            _secure_write_bytes(vault_salt_path, generate_salt())
        if not scrypt_salt_path.exists():
            _secure_write_bytes(scrypt_salt_path, generate_salt())

    def _load_salts(self) -> tuple[bytes, bytes]:
        """Load vault and scrypt salts."""
        vault_salt = (self.root / "vault_salt.key").read_bytes()
        scrypt_salt = (self.root / "scrypt_salt.key").read_bytes()
        if len(vault_salt) != 32 or len(scrypt_salt) != 32:
            raise ValueError("Salt files corrupted (expected 32 bytes each)")
        return vault_salt, scrypt_salt

    def _chain_position(self) -> int:
        """Count existing receipts for chain position."""
        receipts_dir = self.root / "receipts"
        return len(list(receipts_dir.glob("*.json"))) if receipts_dir.exists() else 0

    def _previous_hash(self) -> str:
        """Get the hash of the most recent receipt (for chain linking)."""
        chain_file = self.root / "chain" / "latest_hash.txt"
        if chain_file.exists():
            return chain_file.read_text().strip()
        return ""

    def _update_chain(self, rh: str) -> None:
        """Update the chain head."""
        _secure_write(self.root / "chain" / "latest_hash.txt", rh)

    def store(
        self,
        evidence: bytes,
        commitment: str,
        secret: str,
        source: str = "cell_runtime_local",
        notes: str = "",
    ) -> dict:
        """Encrypt evidence and store with receipt.

        Returns dict with evidence_hash, receipt_hash, receipt_id, paths.
        Raises on validation/crypto failure.
        """
        errors = validate_secret(secret)
        if errors:
            raise ValueError(f"Vault secret validation failed: {'; '.join(errors)}")

        self.init()
        vault_salt, scrypt_salt = self._load_salts()
        evidence_hash = sha256_hex(evidence)
        chain_pos = self._chain_position()
        prev_hash = self._previous_hash()

        # Build structure context for this evidence blob
        receipt_id = str(uuid.uuid4())
        ctx = build_structure_context(
            user_commitment=commitment,
            receipt_schema="almanac.record_discovery.v1",
            receipt_id=receipt_id,
            previous_receipt_hash=prev_hash,
            capsule_type="evidence",
            chain_position=chain_pos,
            vault_id=sha256_hex(vault_salt),
        )

        # Encrypt
        envelope = encrypt_evidence(
            plaintext=evidence,
            vault_secret=secret,
            user_commitment=commitment,
            vault_salt=vault_salt,
            scrypt_salt=scrypt_salt,
            structure_ctx=ctx,
        )

        # Store encrypted blob
        blob_path = self.root / "evidence" / f"{evidence_hash}.blob"
        _secure_write_bytes(blob_path, envelope.ciphertext)

        # Store envelope metadata (no ciphertext — that's in .blob)
        meta_path = self.root / "evidence" / f"{evidence_hash}.v2.json"
        meta = envelope.to_dict()
        meta["structure_context"] = ctx
        _secure_write(meta_path, json.dumps(meta, indent=2))

        # Create and store receipt
        receipt = _discovery_receipt(
            user_commit=commitment,
            evidence_hash=evidence_hash,
            source=source,
            previous_hash=prev_hash,
            notes=notes,
        )
        receipt["receipt_id"] = receipt_id  # bind to structure context
        rh = _receipt_hash(receipt)

        receipt_path = self.root / "receipts" / f"{rh[:16]}.json"
        _secure_write(receipt_path, json.dumps(receipt, indent=2))

        # Update chain
        self._update_chain(rh)

        # Verify round-trip: decrypt and compare
        decrypted = decrypt_evidence(
            envelope=envelope,
            vault_secret=secret,
            user_commitment=commitment,
            vault_salt=vault_salt,
            scrypt_salt=scrypt_salt,
            structure_ctx=ctx,
        )
        if decrypted != evidence:
            raise ValueError("Vault round-trip integrity check FAILED")

        return {
            "evidence_hash": evidence_hash,
            "receipt_hash": rh,
            "receipt_id": receipt_id,
            "chain_position": chain_pos,
            "blob_path": str(blob_path),
            "receipt_path": str(receipt_path),
        }

    def load(
        self,
        evidence_hash: str,
        expected_receipt_hash: str,
        commitment: str,
        secret: str,
    ) -> tuple[bytes, dict]:
        """Load and verify a vault record.

        Returns (plaintext_evidence, receipt).
        Raises on any integrity failure.
        """
        errors = validate_secret(secret)
        if errors:
            raise ValueError(f"Vault secret validation failed: {'; '.join(errors)}")

        vault_salt, scrypt_salt = self._load_salts()

        # Load envelope metadata
        meta_path = self.root / "evidence" / f"{evidence_hash}.v2.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No vault record for evidence_hash: {evidence_hash}")
        meta = json.loads(meta_path.read_text())

        # Load ciphertext
        blob_path = self.root / "evidence" / f"{evidence_hash}.blob"
        if not blob_path.exists():
            raise FileNotFoundError(f"Ciphertext blob missing: {evidence_hash}")
        ciphertext = blob_path.read_bytes()

        # Reconstruct envelope
        envelope = EncryptedEnvelope.from_stored(meta, ciphertext)
        ctx = meta["structure_context"]

        # Decrypt
        plaintext = decrypt_evidence(
            envelope=envelope,
            vault_secret=secret,
            user_commitment=commitment,
            vault_salt=vault_salt,
            scrypt_salt=scrypt_salt,
            structure_ctx=ctx,
        )

        # Verify evidence hash
        if sha256_hex(plaintext) != evidence_hash:
            raise ValueError("Evidence hash mismatch after decryption")

        # Find and verify receipt
        receipt = self._find_receipt(expected_receipt_hash)
        if receipt is None:
            raise ValueError(f"Receipt not found: {expected_receipt_hash}")
        if receipt.get("evidence_hash") != evidence_hash:
            raise ValueError("Receipt evidence_hash does not match")

        return plaintext, receipt

    def _find_receipt(self, expected_hash: str) -> dict | None:
        """Find a receipt by its hash."""
        receipts_dir = self.root / "receipts"
        if not receipts_dir.exists():
            return None
        for p in receipts_dir.glob("*.json"):
            receipt = json.loads(p.read_text())
            if _receipt_hash(receipt) == expected_hash:
                return receipt
        return None

    def status(self) -> dict:
        """Report vault state."""
        evidence_dir = self.root / "evidence"
        receipts_dir = self.root / "receipts"
        return {
            "vault_dir": str(self.root),
            "initialized": (self.root / "vault_salt.key").exists(),
            "evidence_blobs": len(list(evidence_dir.glob("*.blob")))
                if evidence_dir.exists() else 0,
            "receipts": len(list(receipts_dir.glob("*.json")))
                if receipts_dir.exists() else 0,
            "chain_head": self._previous_hash() or None,
        }
