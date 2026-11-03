"""Almanac Core — Structure-bound vault encryption v2.

Upgrade from Fernet MVP to AES-256-GCM with structure-bound context binding.

Key hierarchy:
    vault_secret (private, never stored)
      → scrypt(vault_secret, scrypt_salt) → passphrase_key
      → HKDF(passphrase_key, vault_salt, info=structure_context_hash) → KEK
    random DEK per evidence blob
      → AES-256-GCM(DEK, nonce, plaintext, aad=structure_context_canonical)
    KEK wraps DEK via AES-256-GCM key wrapping

Security properties:
    - vault_secret provides entropy (the actual secret)
    - user_commitment binds key to identity (public, appears in receipts)
    - vault_salt binds key to vault instance
    - structure_context binds ciphertext to its receipt/chain position
    - AAD ensures tampered context fails decryption
    - Per-blob DEK limits exposure from any single key compromise

Does not invent new cryptography. Uses NIST-approved primitives.
PQ-ready: interfaces designed for future ML-KEM/ML-DSA/SLH-DSA upgrade.

Doctrine:
    The structure binds the key.
    The secret unlocks it.
    The receipt gates access.
    Revocation destroys it.
"""

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hashes

try:
    from .structure_key import canonicalize_context, structure_context_hash
except ImportError:
    from cell.vault.structure_key import canonicalize_context, structure_context_hash


VAULT_SALT_FILE = "vault_salt_v2.key"
SCRYPT_SALT_FILE = "scrypt_salt_v2.key"
HKDF_INFO_PREFIX = b"almanac-core-vault-v2|"
HKDF_HMAC_INFO = b"almanac-core-vault-hmac-v1"
MIN_SECRET_LENGTH = 12
SCRYPT_N = 2**17  # OWASP recommended for file encryption


# ── Passphrase validation ─────────────────────────────────────────────────────

def validate_secret(vault_secret: str) -> list[str]:
    """Check passphrase strength. Returns list of failure reasons (empty = OK)."""
    errors = []
    if len(vault_secret) < MIN_SECRET_LENGTH:
        errors.append(
            f"vault_secret must be at least {MIN_SECRET_LENGTH} characters "
            f"(got {len(vault_secret)})"
        )
    # Reject all-same-character secrets
    if len(set(vault_secret)) < 4:
        errors.append(
            "vault_secret has too few unique characters (need at least 4)"
        )
    return errors


# ── Salt generation ──────────────────────────────────────────────────────────

def generate_salt() -> bytes:
    """Generate a 32-byte random salt."""
    return os.urandom(32)


# ── Signing key derivation ────────────────────────────────────────────────

def derive_signing_key(
    vault_secret: str,
    user_commitment: str,
    vault_salt: bytes,
    scrypt_salt: bytes,
) -> bytes:
    """Derive a 32-byte HMAC signing key, independent of any KEK.

    Uses the same scrypt hardening but a different HKDF info field,
    so compromise of a KEK does not compromise receipt signing.
    """
    kdf = Scrypt(salt=scrypt_salt, length=32, n=SCRYPT_N, r=8, p=1)
    passphrase_key = kdf.derive(
        f"{vault_secret}|{user_commitment}".encode()
    )
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=vault_salt,
        info=HKDF_HMAC_INFO,
    )
    return hkdf.derive(passphrase_key)


def compute_hmac(data: bytes, signing_key: bytes) -> str:
    """HMAC-SHA256 of data. Returns hex digest."""
    return hmac.new(signing_key, data, hashlib.sha256).hexdigest()


def verify_hmac(data: bytes, signing_key: bytes, expected_hex: str) -> bool:
    """Constant-time HMAC verification."""
    actual = hmac.new(signing_key, data, hashlib.sha256).hexdigest()
    return hmac.compare_digest(actual, expected_hex)


# ── Key derivation ───────────────────────────────────────────────────────────

def derive_kek(
    vault_secret: str,
    user_commitment: str,
    vault_salt: bytes,
    scrypt_salt: bytes,
    structure_ctx: dict,
) -> bytes:
    """Derive a Key Encryption Key (KEK) from secret + context.

    1. scrypt(vault_secret, scrypt_salt) → passphrase_key (32 bytes)
    2. HKDF(passphrase_key, vault_salt, info=prefix+structure_hash) → KEK (32 bytes)

    The structure_context hash in HKDF info means the same secret under
    different receipt context produces a different KEK.
    """
    # Step 1: Harden the passphrase with scrypt
    kdf = Scrypt(salt=scrypt_salt, length=32, n=SCRYPT_N, r=8, p=1)
    passphrase_key = kdf.derive(
        f"{vault_secret}|{user_commitment}".encode()
    )

    # Step 2: Context-bind via HKDF
    ctx_hash = structure_context_hash(structure_ctx)
    info = HKDF_INFO_PREFIX + ctx_hash.encode()

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=vault_salt,
        info=info,
    )
    return hkdf.derive(passphrase_key)


# ── AES-256-GCM operations ──────────────────────────────────────────────────

def generate_dek() -> bytes:
    """Generate a random 256-bit Data Encryption Key."""
    return AESGCM.generate_key(bit_length=256)


def encrypt_blob(plaintext: bytes, dek: bytes, aad: bytes) -> Tuple[bytes, bytes]:
    """AES-256-GCM encrypt with AAD. Returns (ciphertext, nonce)."""
    nonce = os.urandom(12)
    aesgcm = AESGCM(dek)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return ciphertext, nonce


def decrypt_blob(ciphertext: bytes, dek: bytes, nonce: bytes, aad: bytes) -> bytes:
    """AES-256-GCM decrypt. Fails if AAD doesn't match or ciphertext tampered."""
    aesgcm = AESGCM(dek)
    return aesgcm.decrypt(nonce, ciphertext, aad)


def wrap_dek(dek: bytes, kek: bytes, aad: bytes | None = None) -> Tuple[bytes, bytes]:
    """Wrap DEK with KEK using AES-256-GCM. Returns (wrapped_dek, wrap_nonce).

    When aad is provided, the wrapped DEK is bound to that context —
    unwrapping with different AAD fails GCM authentication.
    """
    nonce = os.urandom(12)
    aesgcm = AESGCM(kek)
    wrapped = aesgcm.encrypt(nonce, dek, aad)
    return wrapped, nonce


def unwrap_dek(wrapped_dek: bytes, kek: bytes, wrap_nonce: bytes, aad: bytes | None = None) -> bytes:
    """Unwrap DEK. Raises InvalidTag if wrong KEK or wrong AAD."""
    aesgcm = AESGCM(kek)
    return aesgcm.decrypt(wrap_nonce, wrapped_dek, aad)


# ── Encrypted envelope ───────────────────────────────────────────────────────

@dataclass
class EncryptedEnvelope:
    """An encrypted evidence blob with its metadata.

    All fields are safe for storage/receipts. No vault_secret, no DEK,
    no plaintext, no raw PII.
    """
    schema: str
    algorithm: str
    kdf: str
    vault_salt_hash: str
    structure_context_hash: str
    aad_hash: str
    wrapped_dek: bytes
    wrap_nonce: bytes
    nonce: bytes
    ciphertext: bytes
    ciphertext_hash: str
    created_at: str

    def to_dict(self) -> dict:
        """Serialize to dict for storage. Binary fields are hex-encoded."""
        return {
            "schema": self.schema,
            "algorithm": self.algorithm,
            "kdf": self.kdf,
            "vault_salt_hash": self.vault_salt_hash,
            "structure_context_hash": self.structure_context_hash,
            "aad_hash": self.aad_hash,
            "wrapped_dek_hex": self.wrapped_dek.hex(),
            "wrap_nonce_hex": self.wrap_nonce.hex(),
            "nonce_hex": self.nonce.hex(),
            "ciphertext_hash": self.ciphertext_hash,
            "ciphertext_bytes": len(self.ciphertext),
            "created_at": self.created_at,
        }

    @staticmethod
    def from_stored(d: dict, ciphertext: bytes) -> "EncryptedEnvelope":
        """Reconstruct from stored dict + ciphertext blob."""
        return EncryptedEnvelope(
            schema=d["schema"],
            algorithm=d["algorithm"],
            kdf=d["kdf"],
            vault_salt_hash=d["vault_salt_hash"],
            structure_context_hash=d["structure_context_hash"],
            aad_hash=d["aad_hash"],
            wrapped_dek=bytes.fromhex(d["wrapped_dek_hex"]),
            wrap_nonce=bytes.fromhex(d["wrap_nonce_hex"]),
            nonce=bytes.fromhex(d["nonce_hex"]),
            ciphertext=ciphertext,
            ciphertext_hash=d["ciphertext_hash"],
            created_at=d["created_at"],
        )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encrypt_evidence(
    plaintext: bytes,
    vault_secret: str,
    user_commitment: str,
    vault_salt: bytes,
    scrypt_salt: bytes,
    structure_ctx: dict,
) -> EncryptedEnvelope:
    """Encrypt evidence with structure-bound envelope.

    1. Derive KEK from secret + commitment + salts + structure context
    2. Generate random DEK
    3. Encrypt plaintext with DEK + structure canonical JSON as AAD
    4. Wrap DEK with KEK
    5. Return envelope (no secret, no DEK, no plaintext)
    """
    kek = derive_kek(vault_secret, user_commitment, vault_salt, scrypt_salt, structure_ctx)
    dek = generate_dek()

    aad = canonicalize_context(structure_ctx).encode()
    ciphertext, nonce = encrypt_blob(plaintext, dek, aad)
    # Bind wrapped DEK to structure context — prevents cross-envelope DEK swap
    wrap_aad = b"almanac-dek-wrap|" + structure_context_hash(structure_ctx).encode()
    wrapped_dek, wrap_nonce = wrap_dek(dek, kek, aad=wrap_aad)

    return EncryptedEnvelope(
        schema="almanac.encrypted_envelope.v2",
        algorithm="AES-256-GCM",
        kdf="scrypt-n17-r8-p1+HKDF-SHA256",
        vault_salt_hash=sha256_hex(vault_salt),
        structure_context_hash=structure_context_hash(structure_ctx),
        aad_hash=sha256_hex(aad),
        wrapped_dek=wrapped_dek,
        wrap_nonce=wrap_nonce,
        nonce=nonce,
        ciphertext=ciphertext,
        ciphertext_hash=sha256_hex(ciphertext),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def decrypt_evidence(
    envelope: EncryptedEnvelope,
    vault_secret: str,
    user_commitment: str,
    vault_salt: bytes,
    scrypt_salt: bytes,
    structure_ctx: dict,
) -> bytes:
    """Decrypt evidence from envelope.

    Re-derives KEK from the same inputs, unwraps DEK, decrypts with AAD.
    Fails if any input is wrong: secret, commitment, salt, or structure.
    """
    kek = derive_kek(vault_secret, user_commitment, vault_salt, scrypt_salt, structure_ctx)
    wrap_aad = b"almanac-dek-wrap|" + structure_context_hash(structure_ctx).encode()
    dek = unwrap_dek(envelope.wrapped_dek, kek, envelope.wrap_nonce, aad=wrap_aad)

    aad = canonicalize_context(structure_ctx).encode()
    return decrypt_blob(envelope.ciphertext, dek, envelope.nonce, aad)
