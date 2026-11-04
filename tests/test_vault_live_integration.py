#!/usr/bin/env python3
"""Live integration test: real Takeout evidence through vault agent.

Exercises the full workflow against the actual evidence table at
~/receipts/almanac/latest_evidence_table.json using an isolated vault.

Workflow:
  1. store  → review evidence, get hash + summary
  2. approve → member creates approval (create_approval, not model-callable)
  3. commit → checks approval → encrypts → stores → consumes approval
  4. verify → decrypts → compares plaintext hash to approved hash

Saved artifacts: commit ID, source hash, approved bundle hash,
approval ID, task receipt, vault receipt, decrypted-plaintext hash.

Usage:
    python3 tests/test_vault_live_integration.py
    python3 -m pytest tests/test_vault_live_integration.py -v
"""

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.vault_agent import VaultReviewAgent
from cell.vault.approval import create_approval
from cell.vault.bridge import VaultBridge
from cell.vault.crypto_v2 import sha256_hex

REAL_EVIDENCE = Path(os.path.expanduser(
    "~/receipts/almanac/latest_evidence_table.json"))
RECEIPT_DIR = Path(os.path.expanduser(
    "~/receipts/almanac/vault_integration"))


def run_live_integration():
    """Full store → approve → commit → verify against real evidence."""

    # ── Prerequisites ──
    if not REAL_EVIDENCE.exists():
        print(f"SKIP: No real evidence at {REAL_EVIDENCE}")
        return None

    evidence_data = json.loads(REAL_EVIDENCE.read_text())
    member_hash = evidence_data.get("member_hash")
    if not member_hash:
        print("SKIP: Evidence table has no member_hash")
        return None

    print(f"  Evidence: {REAL_EVIDENCE}")
    print(f"  Member:   {member_hash}")
    print(f"  Records:  {evidence_data.get('total_records')}")
    print(f"  Source:   {evidence_data.get('source_sha256', '')[:24]}...")

    results = {}
    t_start = time.time()

    with tempfile.TemporaryDirectory(prefix="vault_live_") as tmp:
        vault_dir = Path(tmp) / "vault"
        tx_dir = Path(tmp) / "transactions"
        tx_dir.mkdir()

        # Redirect transaction dir for isolation
        import cell.agents.vault_agent as mod
        orig_tx = mod.TRANSACTION_DIR
        mod.TRANSACTION_DIR = tx_dir

        # Use a test secret for the isolated vault
        test_secret = "live-integration-test-secret-2026!"
        orig_env = os.environ.get("ALMANAC_VAULT_SECRET")
        os.environ["ALMANAC_VAULT_SECRET"] = test_secret

        try:
            agent = VaultReviewAgent(vault_dir=vault_dir)
            approval_dir = vault_dir / "approvals"

            # ── Phase 1: store (review) ──
            print("\n  Phase 1: store (review)...")
            review = agent.execute({
                "action": "store",
                "evidence_path": str(REAL_EVIDENCE),
            })
            assert review.ok, f"store failed: {review.error}"
            assert review.output["status"] == "review_pending"

            evidence_hash = review.output["evidence_hash"]
            results["evidence_hash"] = evidence_hash
            results["evidence_source"] = review.output["evidence_source"]
            results["member_hash"] = review.output["member_hash"]
            results["source_sha256"] = evidence_data["source_sha256"]
            results["total_records"] = review.output["total_records"]
            results["mapped_records"] = review.output.get("mapped_records")
            results["transform_version"] = review.output["transform_version"]
            results["attribute_types"] = review.output["attribute_types"]

            print(f"    evidence_hash:     {evidence_hash}")
            print(f"    total_records:     {results['total_records']}")
            print(f"    transform_version: {results['transform_version']}")
            print(f"    attribute_types:   {results['attribute_types']}")

            # ── Phase 2: approve (member creates approval) ──
            print("\n  Phase 2: approve (member-created, not model-callable)...")
            approval = create_approval(
                approval_dir=approval_dir,
                evidence_hash=evidence_hash,
                member_hash=member_hash,
                scope="storage_only",
                vault_secret=test_secret,
            )
            results["approval_id"] = approval["approval_id"]
            results["approval_scope"] = approval["scope"]
            results["approval_nonce"] = approval["nonce"]

            print(f"    approval_id:    {approval['approval_id']}")
            print(f"    scope:          {approval['scope']}")
            print(f"    expires_at:     {approval['expires_at']}")

            # ── Phase 3: commit (approval check → encrypt → store) ──
            print("\n  Phase 3: commit (approval → encrypt → store)...")
            commit = agent.execute({
                "action": "commit",
                "evidence_hash": evidence_hash,
                "approval_scope": "storage_only",
                "evidence_path": str(REAL_EVIDENCE),
            })
            assert commit.ok, f"commit failed: {commit.error}"
            assert commit.output["status"] == "stored"

            results["receipt_hash"] = commit.output["receipt_hash"]
            results["receipt_id"] = commit.output["receipt_id"]
            results["chain_position"] = commit.output["chain_position"]
            results["transaction_path"] = commit.output["transaction_path"]
            results["wall_time_s"] = commit.output["wall_time_s"]

            print(f"    receipt_hash:   {results['receipt_hash']}")
            print(f"    receipt_id:     {results['receipt_id']}")
            print(f"    chain_position: {results['chain_position']}")
            print(f"    wall_time_s:    {results['wall_time_s']}")

            # ── Verify approval was consumed ──
            pending = list(approval_dir.glob("pending_*.json"))
            consumed = list(approval_dir.glob("consumed_*.json"))
            assert len(pending) == 0, f"Approval not consumed: {len(pending)} pending"
            assert len(consumed) == 1, f"Expected 1 consumed: {len(consumed)}"
            results["approval_consumed"] = True
            print(f"    approval consumed: True")

            # ── Phase 4: verify (decrypt → hash check) ──
            print("\n  Phase 4: verify (decrypt → plaintext hash check)...")
            verify = agent.execute({
                "action": "verify",
                "evidence_hash": evidence_hash,
                "receipt_hash": results["receipt_hash"],
            })
            assert verify.ok, f"verify failed: {verify.error}"
            assert verify.output["status"] == "verified"

            print(f"    status:         {verify.output['status']}")
            print(f"    bundle schema:  {verify.output['bundle_summary'].get('schema')}")
            print(f"    bundle records: {verify.output['bundle_summary'].get('total_records')}")

            # ── Independent plaintext hash verification ──
            # Decrypt directly via VaultBridge and hash the raw bytes.
            # Must equal the approved evidence_hash.
            print("\n  Independent verification: decrypt + hash raw bytes...")
            bridge = VaultBridge(vault_dir)
            vault_salt, scrypt_salt = bridge._load_salts()

            plaintext_bytes, receipt = bridge.load(
                evidence_hash=evidence_hash,
                expected_receipt_hash=results["receipt_hash"],
                commitment=member_hash,
                secret=test_secret,
            )
            independent_hash = hashlib.sha256(plaintext_bytes).hexdigest()
            results["independent_plaintext_hash"] = independent_hash

            assert independent_hash == evidence_hash, (
                f"Plaintext hash mismatch: {independent_hash[:16]}... "
                f"!= {evidence_hash[:16]}...")
            print(f"    plaintext hash: {independent_hash}")
            print(f"    matches approved hash: True")

            # Parse the plaintext and verify record content
            bundle = json.loads(plaintext_bytes.decode())
            assert bundle["schema"] == "cell.evidence_table.v1"
            assert bundle["total_records"] == evidence_data["total_records"]
            assert bundle["member_hash"] == member_hash
            assert bundle["source_sha256"] == evidence_data["source_sha256"]
            results["plaintext_schema"] = bundle["schema"]
            results["plaintext_records_match"] = True
            results["plaintext_member_match"] = True
            results["plaintext_source_match"] = True

            print(f"    schema:         {bundle['schema']}")
            print(f"    records match:  True ({bundle['total_records']})")
            print(f"    member match:   True")
            print(f"    source match:   True")

            # ── Read transaction record ──
            tx_files = sorted(tx_dir.glob("tx_*.json"))
            assert len(tx_files) == 1, f"Expected 1 tx: {len(tx_files)}"
            tx = json.loads(tx_files[0].read_text())
            results["transaction"] = tx
            assert tx["authorization_scope"] == "storage_only"
            assert tx["member_hash"] == member_hash
            assert tx["evidence_hash"] == evidence_hash

            print(f"\n  Transaction record:")
            print(f"    schema:         {tx['schema']}")
            print(f"    scope:          {tx['authorization_scope']}")
            print(f"    member_hash:    {tx['member_hash']}")
            print(f"    evidence_hash:  {tx['evidence_hash'][:24]}...")
            print(f"    receipt_hash:   {tx['receipt_hash'][:24]}...")

        finally:
            mod.TRANSACTION_DIR = orig_tx
            if orig_env is not None:
                os.environ["ALMANAC_VAULT_SECRET"] = orig_env
            elif "ALMANAC_VAULT_SECRET" in os.environ:
                del os.environ["ALMANAC_VAULT_SECRET"]

    total_time = round(time.time() - t_start, 3)
    results["total_wall_time_s"] = total_time
    results["test_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ── Save receipt ──
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = (
        RECEIPT_DIR
        / f"live_integration_{time.strftime('%Y%m%dT%H%M%SZ')}_{evidence_hash[:8]}.json"
    )

    receipt = {
        "schema": "cell.vault_live_integration.v1",
        "test": "store_approve_commit_verify",
        "evidence_source": str(REAL_EVIDENCE),
        "evidence_hash": results["evidence_hash"],
        "independent_plaintext_hash": results["independent_plaintext_hash"],
        "hashes_match": results["independent_plaintext_hash"] == results["evidence_hash"],
        "member_hash": results["member_hash"],
        "source_sha256": results["source_sha256"],
        "total_records": results["total_records"],
        "transform_version": results["transform_version"],
        "approval_id": results["approval_id"],
        "approval_scope": results["approval_scope"],
        "approval_consumed": results["approval_consumed"],
        "receipt_hash": results["receipt_hash"],
        "receipt_id": results["receipt_id"],
        "chain_position": results["chain_position"],
        "plaintext_schema": results["plaintext_schema"],
        "plaintext_records_match": results["plaintext_records_match"],
        "plaintext_member_match": results["plaintext_member_match"],
        "plaintext_source_match": results["plaintext_source_match"],
        "cost": {
            "wall_time_s": total_time,
            "encrypt_wall_time_s": results["wall_time_s"],
            "hostname": __import__("platform").node(),
            "python_version": __import__("platform").python_version(),
            "timestamp": results["test_timestamp"],
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt saved: {receipt_path}")
    print(f"  Total time:    {total_time}s")

    return receipt


def test_live_integration():
    """Pytest entry point."""
    receipt = run_live_integration()
    if receipt is None:
        import pytest
        pytest.skip("No real evidence available")
    assert receipt["hashes_match"], "Plaintext hash does not match approved hash"
    assert receipt["approval_consumed"], "Approval was not consumed"
    assert receipt["plaintext_records_match"], "Plaintext records do not match"


if __name__ == "__main__":
    receipt = run_live_integration()
    if receipt is None:
        sys.exit(0)
    if receipt["hashes_match"]:
        print("\n  PASS: Live integration — all phases verified.")
        sys.exit(0)
    else:
        print("\n  FAIL: Plaintext hash does not match approved hash.")
        sys.exit(1)
