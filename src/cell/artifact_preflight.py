"""Gate 6: Artifact Preflight Checker

Runtime-level verification that an artifact is safe to consume.
Checks receipt existence, hash integrity, on-chain state, and
policy gates before allowing an agent to load or use an artifact.

Decisions: ALLOW, HOLD, REVIEW, REJECT — every decision receipted.

This is the client-side complement to the chain-side Transfer Hook.
The Transfer Hook blocks token transfers of non-Active artifacts.
The preflight checker blocks agent consumption of non-Active artifacts.

Reason codes:
  PF-CLEAN              All checks pass
  PF-NO-RECEIPT         No receipt JSON found
  PF-HASH-MISMATCH      Content hash doesn't match artifact
  PF-SPEC-UNKNOWN       Validation spec hash not in trusted list
  PF-NOT-ACTIVE         On-chain status is not Active
  PF-QUARANTINED        On-chain status is Quarantined
  PF-CHAIN-UNAVAILABLE  Could not query on-chain state (graceful degrade)
  PF-SENTINEL-REJECT    Sentinel policy rejected
"""

import hashlib
import json
import platform
import resource
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ── Decision enum ─────────────────────────────────────────────────────────────

ALLOW = "ALLOW"
HOLD = "HOLD"
REVIEW = "REVIEW"
REJECT = "REJECT"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PreflightResult:
    """Result of a preflight check on an artifact."""
    decision: str                          # ALLOW / HOLD / REVIEW / REJECT
    reason_codes: List[str] = field(default_factory=list)
    artifact_path: str = ""
    receipt_path: str = ""
    receipt_hash: str = ""
    content_hash_expected: str = ""
    content_hash_actual: str = ""
    on_chain_status: Optional[str] = None  # Active / Quarantined / Candidate / None
    chain_queried: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


# ── Trusted validation specs ──────────────────────────────────────────────────

# Known-good validation spec hashes. An artifact whose validation_spec_hash
# is not in this set gets PF-SPEC-UNKNOWN (HOLD, not REJECT — unknown is
# not the same as bad).
TRUSTED_VALIDATION_SPECS: Dict[str, str] = {
    # livecell_scientific_artifact_v1
    "29bf0afc32a24049f90c65002e06e97cbce7083c2b081ddb73e55dbb33b09b79": "livecell_scientific_artifact_v1",
}


# ── Hash helpers ──────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    """SHA-256 of file contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── On-chain query interface ──────────────────────────────────────────────────

# Type for on-chain query function:
#   (content_hash_hex: str) -> Optional[dict]
#   Returns {"status": "Active"|"Quarantined"|"Candidate", ...} or None
OnChainQuery = Callable[[str], Optional[Dict[str, Any]]]


# ── Core preflight ────────────────────────────────────────────────────────────

def preflight_check(
    artifact_path: str,
    receipt_path: str,
    on_chain_query: Optional[OnChainQuery] = None,
    trusted_specs: Optional[Dict[str, str]] = None,
) -> PreflightResult:
    """Run preflight checks on an artifact before consumption.

    Args:
        artifact_path: Path to the artifact file or directory.
        receipt_path: Path to the receipt JSON (gate receipt or register params).
        on_chain_query: Optional callable that takes content_hash (hex) and
            returns on-chain state dict, or None if chain unavailable.
        trusted_specs: Optional dict of trusted validation_spec_hash → name.
            Defaults to TRUSTED_VALIDATION_SPECS.

    Returns:
        PreflightResult with decision and reason codes.
    """
    if trusted_specs is None:
        trusted_specs = TRUSTED_VALIDATION_SPECS

    result = PreflightResult(
        decision=ALLOW,
        artifact_path=str(artifact_path),
        receipt_path=str(receipt_path),
    )

    # ── 1. Receipt exists and parses ──────────────────────────────────────

    rp = Path(receipt_path)
    if not rp.exists():
        result.decision = REJECT
        result.reason_codes.append("PF-NO-RECEIPT")
        return result

    try:
        receipt = json.loads(rp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        result.decision = REJECT
        result.reason_codes.append("PF-NO-RECEIPT")
        result.details["parse_error"] = True
        return result

    result.receipt_hash = sha256_file(rp)

    # ── 2. Content hash verification ──────────────────────────────────────

    # Look for content_hash in receipt (Gate 4 register params or gate receipt)
    content_hash_expected = (
        receipt.get("content_hash")
        or receipt.get("hashes", {}).get("content_hash")
    )

    ap = Path(artifact_path)
    if content_hash_expected and ap.exists():
        if ap.is_file():
            content_hash_actual = sha256_file(ap)
        elif ap.is_dir():
            # For directories, hash the sorted list of file hashes
            file_hashes = {}
            for f in sorted(ap.rglob("*")):
                if f.is_file():
                    file_hashes[str(f.relative_to(ap))] = sha256_file(f)
            content_hash_actual = sha256_bytes(
                json.dumps(file_hashes, sort_keys=True).encode("utf-8")
            )
        else:
            content_hash_actual = ""

        result.content_hash_expected = content_hash_expected
        result.content_hash_actual = content_hash_actual

        if content_hash_actual != content_hash_expected:
            result.decision = REJECT
            result.reason_codes.append("PF-HASH-MISMATCH")
            return result

    # ── 3. Validation spec check ──────────────────────────────────────────

    spec_hash = (
        receipt.get("validation_spec_hash")
        or receipt.get("hashes", {}).get("validation_spec_hash")
    )
    if spec_hash and spec_hash not in trusted_specs:
        # Unknown spec → HOLD, not REJECT
        if result.decision == ALLOW:
            result.decision = HOLD
        result.reason_codes.append("PF-SPEC-UNKNOWN")
        result.details["unknown_spec_hash"] = spec_hash

    # ── 4. On-chain state check ───────────────────────────────────────────

    if on_chain_query is not None:
        query_hash = content_hash_expected or ""
        if query_hash:
            try:
                chain_state = on_chain_query(query_hash)
                if chain_state is not None:
                    result.chain_queried = True
                    status = chain_state.get("status", "unknown")
                    result.on_chain_status = status

                    if status == "Quarantined":
                        result.decision = REJECT
                        result.reason_codes.append("PF-QUARANTINED")
                        return result
                    elif status == "Candidate":
                        result.decision = REJECT
                        result.reason_codes.append("PF-NOT-ACTIVE")
                        return result
                    elif status != "Active":
                        result.decision = REJECT
                        result.reason_codes.append("PF-NOT-ACTIVE")
                        return result
                    # Active → proceed
                else:
                    # Query returned None — chain available but no account found
                    result.chain_queried = True
                    result.on_chain_status = None
                    result.details["chain_note"] = "No on-chain account found for content_hash"
            except Exception as e:
                # Chain query failed — degrade gracefully
                result.reason_codes.append("PF-CHAIN-UNAVAILABLE")
                result.details["chain_error"] = str(e)[:200]

    # ── 5. Final decision ─────────────────────────────────────────────────

    if not result.reason_codes:
        result.reason_codes.append("PF-CLEAN")

    return result


# ── Receipt emitter ───────────────────────────────────────────────────────────

def emit_preflight_receipt(
    result: PreflightResult,
    output_path: str,
    wall_time_s: float = 0.0,
    cpu_time_s: float = 0.0,
) -> Dict[str, Any]:
    """Write a preflight receipt JSON with cost block."""
    receipt = {
        "schema_version": "gate6_preflight_receipt:v1",
        "gate": "Gate 6: Artifact Preflight",
        "decision": result.decision,
        "reason_codes": result.reason_codes,
        "artifact_path": result.artifact_path,
        "receipt_path": result.receipt_path,
        "receipt_hash": result.receipt_hash,
        "content_hash_expected": result.content_hash_expected,
        "content_hash_actual": result.content_hash_actual,
        "on_chain_status": result.on_chain_status,
        "chain_queried": result.chain_queried,
        "details": result.details,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cost": {
            "wall_time_s": round(wall_time_s, 3),
            "cpu_time_s": round(cpu_time_s, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(receipt, indent=2))
    return receipt
