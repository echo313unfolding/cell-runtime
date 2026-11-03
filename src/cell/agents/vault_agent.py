"""Vault agent — member review gate for evidence records.

Pipeline: evidence → member review → vault encrypt → transaction record.

The vault_review agent is PRIVILEGED — the orchestrator gates it through
ask_pass, which is the member approval step. The member sees what's being
vaulted and explicitly approves before encryption happens.

Usage:
    # Store evidence in vault (requires ask_pass approval)
    result = agent.execute({"action": "store"})

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


def _get_vault_secret() -> str | None:
    """Get vault secret from env or local keyfile. Never logs the value."""
    secret = os.environ.get("ALMANAC_VAULT_SECRET")
    if secret:
        return secret
    keyfile = Path(os.path.expanduser("~/.almanac/vault_secret"))
    if keyfile.exists():
        return keyfile.read_text().strip()
    return None


class VaultReviewAgent(AgentBase):
    """Review evidence and write approved records to the member vault.

    This agent implements the member review gate. The orchestrator calls
    ask_pass before executing any PRIVILEGED tool — that prompt IS the
    member review. The member sees the action + detail, and approves or
    denies. Only approved calls reach execute().

    The vault uses AES-256-GCM with structure-bound context binding
    (vendored from almanac_core). No raw PII in receipts. The vault_secret
    never leaves local.
    """
    name = "vault_review"
    description = (
        "Review and vault evidence records. "
        "Actions: 'store' (encrypt evidence into vault with member approval), "
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
                    "Operation: 'store' (vault evidence), "
                    "'verify' (check vault record), 'status' (vault info)"
                ),
            },
            "evidence_path": {
                "type": "string",
                "description": (
                    "Path to evidence file (for store). "
                    "Default: latest_evidence_table.json"
                ),
            },
            "evidence_hash": {
                "type": "string",
                "description": "Evidence hash (for verify).",
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
        elif action == "verify":
            return self._verify(args)
        elif action == "status":
            return self._status()
        return AgentResult(
            error=f"Unknown action: {action!r}. Use: store, verify, status")

    def _store(self, args: dict) -> AgentResult:
        """Encrypt evidence and store in vault with receipt + transaction."""
        # Load evidence
        evidence_path = args.get("evidence_path")
        if evidence_path:
            path = Path(os.path.expanduser(evidence_path))
        else:
            path = EVIDENCE_DIR / "latest_evidence_table.json"

        if not path.exists():
            return AgentResult(error=f"Evidence not found: {path}")

        try:
            evidence_data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            return AgentResult(error=f"Cannot read evidence: {e}")

        member_hash = evidence_data.get("member_hash")
        if not member_hash:
            return AgentResult(
                error="Evidence table has no member_hash. Run almanac_normalize first.")

        secret = _get_vault_secret()
        if not secret:
            return AgentResult(
                error="No vault secret. Set ALMANAC_VAULT_SECRET env var "
                      "or create ~/.almanac/vault_secret")

        # Wrap evidence in a bundle with provenance
        bundle = {
            "schema": "cell.evidence_table.v1",
            "source_sha256": evidence_data.get("source_sha256", ""),
            "total_records": evidence_data.get("total_records", 0),
            "mapped_records": evidence_data.get("mapped_records", 0),
            "transform_version": evidence_data.get("transform_version", ""),
            "member_hash": member_hash,
            "vaulted_from": str(path),
            "vaulted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "attributes": evidence_data.get("attributes", []),
        }
        evidence_bytes = json.dumps(
            bundle, sort_keys=True, separators=(",", ":")).encode()

        # Store
        bridge = VaultBridge(self._vault_dir)
        t_start = time.time()
        try:
            result = bridge.store(
                evidence=evidence_bytes,
                commitment=member_hash,
                secret=secret,
                source="cell_runtime_local",
                notes=f"Evidence from {path.name}, {evidence_data.get('total_records', '?')} records",
            )
        except Exception as e:
            return AgentResult(error=f"Vault store failed: {e}")

        wall_time = round(time.time() - t_start, 3)

        # Write transaction record
        TRANSACTION_DIR.mkdir(parents=True, exist_ok=True)
        transaction = {
            "schema": "cell.vault_transaction.v1",
            "action": "store",
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
            json.dumps(transaction, sort_keys=True, separators=(",", ":")).encode()
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
