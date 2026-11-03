"""Vault agent — member review gate for evidence records.

Two-phase approval protocol:
  1. store  → computes evidence hash, returns review summary (no write)
  2. commit → requires exact evidence_hash + approval_scope="storage_only",
              verifies hash matches current evidence, then encrypts + stores

The hash-binding check in commit is in the agent code. It cannot be
bypassed by CELL_ASK_PASS=0, session "allow all", or any other
orchestrator-level gate. The member must approve the exact bytes
that will be encrypted.

Approval scope is explicit: "storage_only" authorizes encrypted local
storage. Licensing and release require separate authorization.

Usage:
    # Phase 1: Review evidence (returns hash + summary, writes nothing)
    review = agent.execute({"action": "store"})
    evidence_hash = review.output["evidence_hash"]

    # Phase 2: Commit with hash binding (encrypts + stores)
    result = agent.execute({
        "action": "commit",
        "evidence_hash": evidence_hash,
        "approval_scope": "storage_only",
    })

    # Verify a vault record
    result = agent.execute({
        "action": "verify",
        "evidence_hash": "abc123...",
        "receipt_hash": "def456...",
    })

    # Check vault status
    result = agent.execute({"action": "status"})
"""

import hashlib
import json
import os
import time
from pathlib import Path

from cell.agents.base import AgentBase, AgentResult, Permission
from cell.vault.bridge import VaultBridge

# Default paths
VAULT_DIR = Path(os.path.expanduser("~/.almanac/cell_vault"))
EVIDENCE_DIR = Path(os.path.expanduser("~/receipts/almanac"))
TRANSACTION_DIR = Path(os.path.expanduser("~/receipts/almanac/transactions"))

VALID_SCOPES = {"storage_only"}


def _get_vault_secret() -> str | None:
    """Get vault secret from env or local keyfile. Never logs the value."""
    secret = os.environ.get("ALMANAC_VAULT_SECRET")
    if secret:
        return secret
    keyfile = Path(os.path.expanduser("~/.almanac/vault_secret"))
    if keyfile.exists():
        return keyfile.read_text().strip()
    return None


def _load_evidence(evidence_path: str | None) -> tuple[Path, dict, bytes, str]:
    """Load evidence file, build canonical bundle, compute hash.

    Returns (path, evidence_data, evidence_bytes, evidence_hash).
    Raises ValueError on any problem.
    """
    if evidence_path:
        path = Path(os.path.expanduser(evidence_path))
    else:
        path = EVIDENCE_DIR / "latest_evidence_table.json"

    if not path.exists():
        raise ValueError(f"Evidence not found: {path}")

    try:
        evidence_data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Cannot read evidence: {e}")

    member_hash = evidence_data.get("member_hash")
    if not member_hash:
        raise ValueError(
            "Evidence table has no member_hash. Run almanac_normalize first.")

    bundle = {
        "schema": "cell.evidence_table.v1",
        "source_sha256": evidence_data.get("source_sha256", ""),
        "total_records": evidence_data.get("total_records", 0),
        "mapped_records": evidence_data.get("mapped_records", 0),
        "transform_version": evidence_data.get("transform_version", ""),
        "member_hash": member_hash,
        "vaulted_from": str(path),
        "attributes": evidence_data.get("attributes", []),
    }
    evidence_bytes = json.dumps(
        bundle, sort_keys=True, separators=(",", ":")).encode()
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest()

    return path, evidence_data, evidence_bytes, evidence_hash


class VaultReviewAgent(AgentBase):
    """Two-phase vault agent: review then commit with hash binding.

    The orchestrator's ask_pass is the outer gate. The agent's own
    evidence_hash verification is the inner gate — it binds the
    approval to the exact evidence bytes. No env var or session flag
    can bypass the inner gate.

    The vault uses AES-256-GCM with structure-bound context binding
    (vendored from almanac_core). No raw PII in receipts. The vault_secret
    never leaves local.
    """
    name = "vault_review"
    description = (
        "Review and vault evidence records. "
        "Actions: 'store' (review evidence, returns hash + summary), "
        "'commit' (encrypt + store, requires evidence_hash + approval_scope), "
        "'verify' (load and verify a vault record), "
        "'status' (show vault state)."
    )
    permission = Permission.PRIVILEGED
    timeout_s = 60

    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "Operation: 'store' (review evidence), "
                    "'commit' (encrypt+store with approval), "
                    "'verify' (check vault record), 'status' (vault info)"
                ),
            },
            "evidence_path": {
                "type": "string",
                "description": (
                    "Path to evidence file (for store/commit). "
                    "Default: latest_evidence_table.json"
                ),
            },
            "evidence_hash": {
                "type": "string",
                "description": (
                    "Evidence hash from store phase (required for commit). "
                    "Also used for verify."
                ),
            },
            "approval_scope": {
                "type": "string",
                "description": (
                    "Authorization scope (required for commit). "
                    "Must be 'storage_only'. Licensing and release "
                    "require separate authorization."
                ),
            },
            "receipt_hash": {
                "type": "string",
                "description": "Receipt hash (for verify).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, vault_dir: str | Path | None = None):
        self._vault_dir = Path(vault_dir) if vault_dir else VAULT_DIR

    def execute(self, args: dict) -> AgentResult:
        action = args.get("action", "").strip().lower()
        if action == "store":
            return self._store(args)
        elif action == "commit":
            return self._commit(args)
        elif action == "verify":
            return self._verify(args)
        elif action == "status":
            return self._status()
        return AgentResult(
            error=f"Unknown action: {action!r}. "
                  f"Use: store, commit, verify, status")

    def _store(self, args: dict) -> AgentResult:
        """Phase 1: Compute evidence hash and return review summary.

        Writes nothing. The member reviews the evidence_hash, source,
        record count, and attribute summary. If satisfied, calls commit
        with the evidence_hash.
        """
        try:
            path, evidence_data, _, evidence_hash = _load_evidence(
                args.get("evidence_path"))
        except ValueError as e:
            return AgentResult(error=str(e))

        # Build review summary (no sensitive data — just metadata)
        attrs = evidence_data.get("attributes", [])
        attr_types = {}
        for a in attrs:
            name = a.get("attribute_name", "unknown")
            attr_types[name] = attr_types.get(name, 0) + 1

        return AgentResult(
            output={
                "status": "review_pending",
                "evidence_hash": evidence_hash,
                "evidence_source": str(path),
                "member_hash": evidence_data.get("member_hash"),
                "source_sha256": evidence_data.get("source_sha256", "")[:16] + "...",
                "total_records": evidence_data.get("total_records", 0),
                "mapped_records": evidence_data.get("mapped_records", 0),
                "transform_version": evidence_data.get("transform_version", ""),
                "attribute_types": attr_types,
                "instruction": (
                    "Review the evidence summary above. To approve vault "
                    "storage, call vault_review with action='commit', "
                    "evidence_hash='" + evidence_hash + "', "
                    "approval_scope='storage_only'. "
                    "This authorizes encrypted local storage only — "
                    "licensing and release require separate authorization."
                ),
            },
        )

    def _commit(self, args: dict) -> AgentResult:
        """Phase 2: Verify hash binding and store in vault.

        Requirements (enforced in agent code, not bypassable):
          - evidence_hash must match current evidence on disk
          - approval_scope must be "storage_only"
          - vault secret must be available
        """
        # ── Validate approval scope ──
        approval_scope = args.get("approval_scope", "")
        if approval_scope not in VALID_SCOPES:
            return AgentResult(
                error=f"approval_scope must be one of {VALID_SCOPES}, "
                      f"got {approval_scope!r}. Licensing and release "
                      f"require separate authorization.")

        # ── Validate evidence hash binding ──
        approved_hash = args.get("evidence_hash", "")
        if not approved_hash:
            return AgentResult(
                error="commit requires evidence_hash from the store phase.")

        try:
            path, evidence_data, evidence_bytes, current_hash = _load_evidence(
                args.get("evidence_path"))
        except ValueError as e:
            return AgentResult(error=str(e))

        if current_hash != approved_hash:
            return AgentResult(
                error=f"Evidence changed since review. "
                      f"Approved hash: {approved_hash[:16]}..., "
                      f"current hash: {current_hash[:16]}.... "
                      f"Run store again to review the updated evidence.")

        # ── Validate vault secret ──
        secret = _get_vault_secret()
        if not secret:
            return AgentResult(
                error="No vault secret. Set ALMANAC_VAULT_SECRET env var "
                      "or create ~/.almanac/vault_secret")

        member_hash = evidence_data["member_hash"]

        # ── Encrypt and store ──
        # Add vaulted_at timestamp to the bundle (after hash verification,
        # so the hash was computed on the pre-timestamp bundle — the exact
        # bytes the member reviewed).
        # NOTE: We store evidence_bytes as-is (the hash-verified bytes).
        # The vaulted_at appears only in the transaction record, not in
        # the encrypted evidence.
        bridge = VaultBridge(self._vault_dir)
        t_start = time.time()
        try:
            result = bridge.store(
                evidence=evidence_bytes,
                commitment=member_hash,
                secret=secret,
                source="cell_runtime_local",
                notes=(
                    f"Evidence from {path.name}, "
                    f"{evidence_data.get('total_records', '?')} records. "
                    f"Approval scope: {approval_scope}."
                ),
            )
        except Exception as e:
            return AgentResult(error=f"Vault store failed: {e}")

        wall_time = round(time.time() - t_start, 3)

        # ── Write transaction record ──
        TRANSACTION_DIR.mkdir(parents=True, exist_ok=True)
        transaction = {
            "schema": "cell.vault_transaction.v1",
            "action": "store",
            "authorization_scope": approval_scope,
            "member_hash": member_hash,
            "evidence_hash": result["evidence_hash"],
            "receipt_hash": result["receipt_hash"],
            "receipt_id": result["receipt_id"],
            "chain_position": result["chain_position"],
            "source": "cell_runtime_local",
            "vault_dir": str(self._vault_dir),
            "evidence_source_path": str(path),
            "approved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cost": {"wall_time_s": wall_time},
        }
        tx_hash = hashlib.sha256(
            json.dumps(transaction, sort_keys=True,
                       separators=(",", ":")).encode()
        ).hexdigest()
        tx_path = (
            TRANSACTION_DIR
            / f"tx_{time.strftime('%Y%m%dT%H%M%SZ')}_{tx_hash[:8]}.json"
        )
        tx_path.write_text(json.dumps(transaction, indent=2))

        return AgentResult(
            output={
                "status": "stored",
                "evidence_hash": result["evidence_hash"],
                "receipt_hash": result["receipt_hash"],
                "receipt_id": result["receipt_id"],
                "chain_position": result["chain_position"],
                "authorization_scope": approval_scope,
                "vault_dir": str(self._vault_dir),
                "transaction_path": str(tx_path),
                "member_hash": member_hash,
                "wall_time_s": wall_time,
            },
            receipt=transaction,
        )

    def _verify(self, args: dict) -> AgentResult:
        """Load and verify a vault record."""
        evidence_hash = args.get("evidence_hash")
        receipt_hash = args.get("receipt_hash")
        if not evidence_hash or not receipt_hash:
            return AgentResult(
                error="verify requires both evidence_hash and receipt_hash")

        secret = _get_vault_secret()
        if not secret:
            return AgentResult(
                error="No vault secret. Set ALMANAC_VAULT_SECRET or "
                      "~/.almanac/vault_secret")

        # Get member commitment from latest evidence table
        evidence_table_path = EVIDENCE_DIR / "latest_evidence_table.json"
        if not evidence_table_path.exists():
            return AgentResult(
                error="No evidence table to derive member commitment.")
        try:
            member_hash = json.loads(
                evidence_table_path.read_text()).get("member_hash")
        except Exception:
            member_hash = None
        if not member_hash:
            return AgentResult(error="Evidence table has no member_hash.")

        bridge = VaultBridge(self._vault_dir)
        t_start = time.time()
        try:
            plaintext, receipt = bridge.load(
                evidence_hash=evidence_hash,
                expected_receipt_hash=receipt_hash,
                commitment=member_hash,
                secret=secret,
            )
        except Exception as e:
            return AgentResult(error=f"Vault verify failed: {e}")

        wall_time = round(time.time() - t_start, 3)

        # Parse the decrypted bundle for summary
        try:
            bundle = json.loads(plaintext.decode())
            bundle_summary = {
                "schema": bundle.get("schema"),
                "total_records": bundle.get("total_records"),
                "mapped_records": bundle.get("mapped_records"),
                "attribute_count": len(bundle.get("attributes", [])),
            }
        except Exception:
            bundle_summary = {"raw_bytes": len(plaintext)}

        return AgentResult(
            output={
                "status": "verified",
                "evidence_hash": evidence_hash,
                "receipt_hash": receipt_hash,
                "receipt_id": receipt.get("receipt_id"),
                "member_hash": member_hash,
                "bundle_summary": bundle_summary,
                "verification_time_s": wall_time,
            },
        )

    def _status(self) -> AgentResult:
        """Report vault state."""
        bridge = VaultBridge(self._vault_dir)
        vault_info = bridge.status()

        # Transaction count
        tx_count = 0
        if TRANSACTION_DIR.exists():
            tx_count = len(list(TRANSACTION_DIR.glob("tx_*.json")))

        # Latest evidence table
        evidence_table = EVIDENCE_DIR / "latest_evidence_table.json"
        has_evidence = evidence_table.exists()
        member_hash = None
        if has_evidence:
            try:
                member_hash = json.loads(
                    evidence_table.read_text()).get("member_hash")
            except Exception:
                pass

        has_secret = _get_vault_secret() is not None

        return AgentResult(
            output={
                **vault_info,
                "has_secret": has_secret,
                "transactions": tx_count,
                "has_evidence_table": has_evidence,
                "member_hash": member_hash,
            },
        )
