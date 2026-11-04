"""Vault approval — member-authenticated authorization records.

An approval binds an authenticated member to an exact evidence hash
and authorization scope. The model may reference an approval_id but
cannot create one — creation requires interactive member confirmation
or direct CLI invocation.

Approval lifecycle:
  1. Created by member (CLI or interactive prompt, never by model)
  2. Validated by commit (hash, member, scope, HMAC, expiry, not consumed)
  3. Consumed after successful vault write (single-use)

Approval file: vault_dir/approvals/pending_<nonce>.json
Consumed file: vault_dir/approvals/consumed_<nonce>.json

HMAC prevents tampering: signed with vault secret, which the model
may access — but the approval creation path is not exposed as a tool.
The defense is the creation channel (interactive stdin / CLI), not
the signing key.
"""

import hashlib
import hmac as _hmac
import json
import os
import sys
import time
import uuid
from pathlib import Path

APPROVAL_TTL_S = 300  # 5 minutes
APPROVAL_SCHEMA = "cell.vault_approval.v1"

_FILE_MODE = 0o600
_OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC


def _secure_write(path: Path, content: str) -> None:
    """Atomic write with 0600 permissions."""
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


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sign_approval(approval: dict, vault_secret: str) -> str:
    """HMAC-SHA256 of canonical approval fields (excluding the hmac field)."""
    signable = {k: v for k, v in approval.items() if k != "hmac"}
    data = _canonical(signable).encode()
    key = vault_secret.encode()
    return _hmac.new(key, data, hashlib.sha256).hexdigest()


def _verify_signature(approval: dict, vault_secret: str) -> bool:
    """Constant-time HMAC verification."""
    expected = _sign_approval(approval, vault_secret)
    return _hmac.compare_digest(expected, approval.get("hmac", ""))


def create_approval(
    approval_dir: Path,
    evidence_hash: str,
    member_hash: str,
    scope: str,
    vault_secret: str,
    ttl_s: int = APPROVAL_TTL_S,
) -> dict:
    """Create a signed approval file. NOT exposed as a model-callable tool.

    Called by prompt_member_approval() after interactive confirmation,
    or by tests directly.

    Returns the approval dict.
    """
    approval_dir.mkdir(parents=True, exist_ok=True)
    nonce = os.urandom(16).hex()
    now = time.time()

    approval = {
        "schema": APPROVAL_SCHEMA,
        "approval_id": str(uuid.uuid4()),
        "nonce": nonce,
        "member_hash": member_hash,
        "evidence_hash": evidence_hash,
        "scope": scope,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "expires_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + ttl_s)),
        "expires_epoch": now + ttl_s,
        "consumed": False,
    }
    approval["hmac"] = _sign_approval(approval, vault_secret)

    path = approval_dir / f"pending_{nonce}.json"
    _secure_write(path, json.dumps(approval, indent=2))

    return approval


def prompt_member_approval(
    approval_dir: Path,
    evidence_hash: str,
    member_hash: str,
    scope: str,
    vault_secret: str,
    evidence_summary: dict | None = None,
    ttl_s: int = APPROVAL_TTL_S,
) -> dict | None:
    """Interactive member approval. Always reads from stdin.

    Cannot be bypassed by CELL_ASK_PASS, session flags, or env vars.
    The member must type: APPROVE <first 8 chars of evidence_hash>

    Returns approval dict on success, None on denial.
    Raises EOFError/KeyboardInterrupt on no input.
    """
    hash_prefix = evidence_hash[:8]

    print("\n--- vault approval required ---", file=sys.stderr)
    print(f"  Evidence hash: {evidence_hash}", file=sys.stderr)
    print(f"  Member:        {member_hash[:16]}...", file=sys.stderr)
    print(f"  Scope:         {scope}", file=sys.stderr)
    if evidence_summary:
        records = evidence_summary.get("total_records", "?")
        version = evidence_summary.get("transform_version", "?")
        print(f"  Records:       {records}", file=sys.stderr)
        print(f"  Version:       {version}", file=sys.stderr)
        attr_types = evidence_summary.get("attribute_types", {})
        if attr_types:
            print(f"  Attributes:    {attr_types}", file=sys.stderr)
    print(f"  Expires:       {ttl_s}s after approval", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  To approve, type: APPROVE {hash_prefix}", file=sys.stderr)
    print(f"  To deny, type:    DENY", file=sys.stderr)

    try:
        response = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("  Denied (no input).", file=sys.stderr)
        return None

    expected = f"APPROVE {hash_prefix}"
    if response == expected:
        approval = create_approval(
            approval_dir=approval_dir,
            evidence_hash=evidence_hash,
            member_hash=member_hash,
            scope=scope,
            vault_secret=vault_secret,
            ttl_s=ttl_s,
        )
        print(f"  Approved. ID: {approval['approval_id']}", file=sys.stderr)
        return approval
    else:
        print("  Denied.", file=sys.stderr)
        return None


def validate_approval(
    approval_dir: Path,
    evidence_hash: str,
    member_hash: str,
    scope: str,
    vault_secret: str,
) -> dict | None:
    """Find and validate a pending approval for this evidence.

    Returns the approval dict if valid, None otherwise.
    Checks: exists, HMAC valid, not expired, not consumed, hash+member+scope match.
    """
    if not approval_dir.exists():
        return None

    now = time.time()

    for path in sorted(approval_dir.glob("pending_*.json")):
        try:
            approval = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Schema check
        if approval.get("schema") != APPROVAL_SCHEMA:
            continue

        # Match check
        if approval.get("evidence_hash") != evidence_hash:
            continue
        if approval.get("member_hash") != member_hash:
            continue
        if approval.get("scope") != scope:
            continue

        # Consumed check
        if approval.get("consumed"):
            continue

        # Expiry check
        expires = approval.get("expires_epoch", 0)
        if now > expires:
            continue

        # HMAC check (last — most expensive)
        if not _verify_signature(approval, vault_secret):
            continue

        return approval

    return None


def consume_approval(approval_dir: Path, approval_id: str) -> bool:
    """Atomically consume an approval. Returns True if this call consumed it.

    Uses os.rename (atomic on POSIX) to move pending → consumed. If two
    processes race, only one rename succeeds; the other gets FileNotFoundError
    and returns False. This prevents double-use of a single approval.
    """
    if not approval_dir.exists():
        return False

    for path in approval_dir.glob("pending_*.json"):
        try:
            approval = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        if approval.get("approval_id") == approval_id:
            consumed_path = (
                approval_dir
                / path.name.replace("pending_", "consumed_")
            )
            # Atomic rename — only one process wins this race
            try:
                os.rename(str(path), str(consumed_path))
            except FileNotFoundError:
                # Another process already consumed this approval
                return False

            # Update content (post-rename, non-critical for race safety)
            approval["consumed"] = True
            approval["consumed_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _secure_write(consumed_path, json.dumps(approval, indent=2))
            return True

    return False
