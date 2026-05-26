"""Device-bound key derivation for personal encryption agent.

Collects stable hardware identifiers and derives device-bound keys
using standard cryptographic primitives (scrypt + HKDF-SHA256).

Three-property key model:
  - something you know  = passphrase
  - something you have  = device fingerprint
  - something it is     = artifact/codebook hash (AAD, handled elsewhere)

Raw hardware identifiers are NEVER written to logs or receipts.
Only the SHA-256 hash of the combined fingerprint is stored.
"""

from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path
from typing import Dict

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hashes


# ── Hardware ID Collection ──────────────────────────────────────────────────

_DMI_PATHS = {
    "product_uuid": "/sys/class/dmi/id/product_uuid",
    "board_serial": "/sys/class/dmi/id/board_serial",
    "product_serial": "/sys/class/dmi/id/product_serial",
}


def collect_hardware_ids() -> Dict[str, str]:
    """Collect stable hardware identifiers (best-effort, graceful fallback).

    Each source is read in a try/except — missing fields produce empty string.
    """
    ids: Dict[str, str] = {}

    # DMI identifiers (Linux)
    for name, path in _DMI_PATHS.items():
        try:
            ids[name] = Path(path).read_text().strip()
        except (OSError, PermissionError):
            ids[name] = ""

    # machine-id (survives reboots)
    try:
        ids["machine_id"] = Path("/etc/machine-id").read_text().strip()
    except (OSError, PermissionError):
        ids["machine_id"] = ""

    # CPU model string
    try:
        ids["cpu_model"] = platform.processor() or platform.machine()
    except Exception:
        ids["cpu_model"] = ""

    return ids


def hardware_fingerprint_hash(hw_ids: Dict[str, str]) -> str:
    """SHA-256 of sorted, normalized hardware IDs. Safe for receipts."""
    # Sort by key for determinism, join as "key=value" lines
    normalized = "\n".join(
        f"{k}={v}" for k, v in sorted(hw_ids.items())
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ── Key Derivation ──────────────────────────────────────────────────────────

def generate_salt() -> bytes:
    """Generate a 32-byte random salt. Stored alongside wrapped keys."""
    return os.urandom(32)


def derive_passphrase_key(passphrase: str, salt: bytes) -> bytes:
    """Derive 32-byte key from passphrase using scrypt.

    Parameters match OWASP recommendations for interactive login:
      n=2^14 (16384), r=8, p=1
    """
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=2**14,
        r=8,
        p=1,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def derive_device_key(passphrase: str, salt: bytes, hw_ids: Dict[str, str]) -> bytes:
    """Derive device-bound master key (two-factor: passphrase + hardware).

    1. passphrase_key = scrypt(passphrase, salt)
    2. hw_hash = hardware_fingerprint_hash(hw_ids)
    3. device_key = HKDF-SHA256(ikm=passphrase_key, salt=salt, info=hw_hash)

    Hardware fingerprint is a binding factor in HKDF info, not the sole secret.
    """
    passphrase_key = derive_passphrase_key(passphrase, salt)
    hw_hash = hardware_fingerprint_hash(hw_ids)

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=hw_hash.encode("utf-8"),
    )
    return hkdf.derive(passphrase_key)
