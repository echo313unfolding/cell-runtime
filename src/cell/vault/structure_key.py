"""Almanac Core — Structure context for key binding.

Bio-inspired context binding: the structure of a receipt, its position in
the chain, its policy context, and its capsule type all contribute to the
cryptographic context that binds the encryption key.

The structure does NOT replace the secret. It binds the key to its context
so that the same secret in a different structural context produces a
different key, and tampered context fails decryption.

    structure_context → canonical JSON → SHA-256 hash
    hash used as: HKDF info field + AES-GCM AAD
"""

import hashlib
import json


def build_structure_context(
    user_commitment: str,
    receipt_schema: str = "",
    receipt_id: str = "",
    previous_receipt_hash: str = "",
    policy_hash: str = "",
    capsule_type: str = "evidence",
    chain_position: int = 0,
    vault_id: str = "",
) -> dict:
    """Build a structure context dict from receipt/vault metadata.

    All fields are non-PII by design. user_commitment is already a hash.
    No raw names, emails, addresses, or locations appear here.
    """
    return {
        "user_commitment": user_commitment,
        "receipt_schema": receipt_schema,
        "receipt_id": receipt_id,
        "previous_receipt_hash": previous_receipt_hash,
        "policy_hash": policy_hash,
        "capsule_type": capsule_type,
        "chain_position": chain_position,
        "vault_id": vault_id,
    }


def canonicalize_context(ctx: dict) -> str:
    """Produce a deterministic canonical JSON string from a context dict.

    Sorted keys, no whitespace, no trailing newline.
    """
    return json.dumps(ctx, sort_keys=True, separators=(",", ":"))


def structure_context_hash(ctx: dict) -> str:
    """SHA-256 of the canonical context. Used as HKDF info and AES-GCM AAD."""
    canonical = canonicalize_context(ctx)
    return hashlib.sha256(canonical.encode()).hexdigest()
