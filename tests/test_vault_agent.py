"""Vault agent approval-binding tests.

Acceptance criteria (from user review of 99d5c96):
  1. Member approves exact evidence hash, source/version, storage scope
  2. Encryption stores precisely those approved bytes
  3. Denial creates no evidence blob or transaction
  4. Changed evidence cannot reuse an earlier approval
  5. Record explicitly authorizes storage; licensing/release separate

Also tests: status, verify round-trip, error paths.

Usage:
    python3 -m pytest tests/test_vault_agent.py -v
    python3 tests/test_vault_agent.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.vault_agent import VaultReviewAgent, EVIDENCE_DIR, TRANSACTION_DIR


MOCK_EVIDENCE = {
    "member_hash": "test_member_abc123",
    "source_sha256": "deadbeef" * 8,
    "total_records": 100,
    "mapped_records": 100,
    "transform_version": "test_v1",
    "attributes": [
        {
            "attribute_name": "interest_channel_affinity",
            "typed_value": {
                "channel": "TestChannel",
                "watch_count": 42,
                "quality_weighted_score": 38.5,
                "avg_signal_quality": 0.917,
            },
        },
        {
            "attribute_name": "signal_quality_distribution",
            "typed_value": {
                "deliberate": 0.55,
                "music": 0.15,
                "quick_view": 0.20,
                "shorts": 0.10,
            },
        },
    ],
}

MODIFIED_EVIDENCE = {
    **MOCK_EVIDENCE,
    "total_records": 200,
    "mapped_records": 150,
}

TEST_SECRET = "test-vault-secret-2026!"


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.details = []

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            self.details.append(f"  FAIL: {name}: {detail}")

    def report(self):
        for d in self.details:
            print(d)
        total = self.passed + self.failed
        print(f"\n  {self.passed}/{total} PASS, {self.failed}/{total} FAIL")
        return 0 if self.failed == 0 else 1


def _count_blobs(vault_dir):
    """Count .blob files in vault evidence directory."""
    d = vault_dir / "evidence"
    return len(list(d.glob("*.blob"))) if d.exists() else 0


def _count_tx(tx_dir):
    """Count transaction files."""
    return len(list(tx_dir.glob("tx_*.json"))) if tx_dir.exists() else 0


def run_tests():
    r = TestResults()

    with tempfile.TemporaryDirectory() as tmp:
        vault_dir = Path(tmp) / "vault"
        evidence_dir = Path(tmp) / "evidence"
        tx_dir = Path(tmp) / "transactions"
        evidence_dir.mkdir()
        tx_dir.mkdir()

        evidence_path = evidence_dir / "latest_evidence_table.json"
        evidence_path.write_text(json.dumps(MOCK_EVIDENCE, indent=2))

        import cell.agents.vault_agent as mod
        orig_evidence = mod.EVIDENCE_DIR
        orig_tx = mod.TRANSACTION_DIR
        mod.EVIDENCE_DIR = evidence_dir
        mod.TRANSACTION_DIR = tx_dir

        orig_env = os.environ.get("ALMANAC_VAULT_SECRET")
        os.environ["ALMANAC_VAULT_SECRET"] = TEST_SECRET

        try:
            agent = VaultReviewAgent(vault_dir=vault_dir)

            # ══════════════════════════════════════════════════════════
            # Acceptance Test 1: Member approves exact evidence hash,
            #   source/version, and storage scope.
            # ══════════════════════════════════════════════════════════

            # Phase 1: store returns review_pending with hash + summary
            review = agent.execute({
                "action": "store",
                "evidence_path": str(evidence_path),
            })
            r.check("review_ok", review.ok, f"error={review.error}")
            r.check("review_pending",
                     review.output.get("status") == "review_pending",
                     f"status={review.output.get('status')}")
            r.check("review_has_hash",
                     len(review.output.get("evidence_hash", "")) == 64,
                     f"hash={review.output.get('evidence_hash')}")
            r.check("review_has_source",
                     "evidence_source" in review.output,
                     f"keys={list(review.output.keys())}")
            r.check("review_has_member",
                     review.output.get("member_hash") == "test_member_abc123",
                     f"member={review.output.get('member_hash')}")
            r.check("review_has_records",
                     review.output.get("total_records") == 100,
                     f"records={review.output.get('total_records')}")
            r.check("review_has_version",
                     review.output.get("transform_version") == "test_v1",
                     f"version={review.output.get('transform_version')}")
            r.check("review_has_attrs",
                     review.output.get("attribute_types") == {
                         "interest_channel_affinity": 1,
                         "signal_quality_distribution": 1,
                     },
                     f"attrs={review.output.get('attribute_types')}")
            r.check("review_has_instruction",
                     "approval_scope" in review.output.get("instruction", ""),
                     f"instruction={review.output.get('instruction', '')[:80]}")

            # No blob or transaction created by store phase
            r.check("store_no_blob",
                     _count_blobs(vault_dir) == 0,
                     f"blobs={_count_blobs(vault_dir)}")
            r.check("store_no_tx",
                     _count_tx(tx_dir) == 0,
                     f"txs={_count_tx(tx_dir)}")

            evidence_hash = review.output["evidence_hash"]

            # ══════════════════════════════════════════════════════════
            # Acceptance Test 2: Encryption stores precisely approved bytes.
            # ══════════════════════════════════════════════════════════

            # Phase 2: commit with matching hash + scope
            commit = agent.execute({
                "action": "commit",
                "evidence_hash": evidence_hash,
                "approval_scope": "storage_only",
                "evidence_path": str(evidence_path),
            })
            r.check("commit_ok", commit.ok, f"error={commit.error}")
            r.check("commit_stored",
                     commit.output.get("status") == "stored",
                     f"status={commit.output.get('status')}")
            r.check("commit_hash_matches",
                     commit.output.get("evidence_hash") == evidence_hash,
                     f"hash={commit.output.get('evidence_hash')}")
            r.check("commit_has_receipt",
                     bool(commit.output.get("receipt_hash")),
                     f"receipt={commit.output.get('receipt_hash')}")
            r.check("commit_blob_created",
                     _count_blobs(vault_dir) == 1,
                     f"blobs={_count_blobs(vault_dir)}")
            r.check("commit_tx_created",
                     _count_tx(tx_dir) == 1,
                     f"txs={_count_tx(tx_dir)}")

            # Verify the committed record round-trips correctly
            stored_receipt_hash = commit.output["receipt_hash"]
            verify = agent.execute({
                "action": "verify",
                "evidence_hash": evidence_hash,
                "receipt_hash": stored_receipt_hash,
            })
            r.check("verify_roundtrip_ok", verify.ok, f"error={verify.error}")
            r.check("verify_status_verified",
                     verify.output.get("status") == "verified",
                     f"status={verify.output.get('status')}")
            r.check("verify_bundle_schema",
                     verify.output.get("bundle_summary", {}).get("schema")
                     == "cell.evidence_table.v1",
                     f"summary={verify.output.get('bundle_summary')}")
            r.check("verify_bundle_records",
                     verify.output.get("bundle_summary", {}).get("total_records") == 100,
                     f"records={verify.output.get('bundle_summary', {}).get('total_records')}")

            # ══════════════════════════════════════════════════════════
            # Acceptance Test 3: Denial creates no blob or transaction.
            # ══════════════════════════════════════════════════════════

            # store (review phase) → member decides not to commit
            blobs_before = _count_blobs(vault_dir)
            tx_before = _count_tx(tx_dir)

            denial_review = agent.execute({"action": "store"})
            r.check("denial_review_ok", denial_review.ok,
                     f"error={denial_review.error}")

            # Member never calls commit. No new artifacts.
            r.check("denial_no_new_blob",
                     _count_blobs(vault_dir) == blobs_before,
                     f"before={blobs_before}, after={_count_blobs(vault_dir)}")
            r.check("denial_no_new_tx",
                     _count_tx(tx_dir) == tx_before,
                     f"before={tx_before}, after={_count_tx(tx_dir)}")

            # ══════════════════════════════════════════════════════════
            # Acceptance Test 4: Changed evidence cannot reuse approval.
            # ══════════════════════════════════════════════════════════

            # Get approval hash for current evidence
            pre_change_review = agent.execute({
                "action": "store",
                "evidence_path": str(evidence_path),
            })
            old_hash = pre_change_review.output["evidence_hash"]

            # Modify the evidence file
            evidence_path.write_text(json.dumps(MODIFIED_EVIDENCE, indent=2))

            # Try to commit with the old hash — must fail
            blobs_before = _count_blobs(vault_dir)
            stale_commit = agent.execute({
                "action": "commit",
                "evidence_hash": old_hash,
                "approval_scope": "storage_only",
                "evidence_path": str(evidence_path),
            })
            r.check("stale_hash_rejected",
                     not stale_commit.ok,
                     f"ok={stale_commit.ok}, error={stale_commit.error}")
            r.check("stale_hash_error_msg",
                     "changed" in (stale_commit.error or "").lower(),
                     f"error={stale_commit.error}")
            r.check("stale_hash_no_blob",
                     _count_blobs(vault_dir) == blobs_before,
                     f"before={blobs_before}, after={_count_blobs(vault_dir)}")

            # Get new hash and commit correctly
            new_review = agent.execute({
                "action": "store",
                "evidence_path": str(evidence_path),
            })
            new_hash = new_review.output["evidence_hash"]
            r.check("new_hash_different",
                     new_hash != old_hash,
                     f"old={old_hash[:16]}, new={new_hash[:16]}")

            new_commit = agent.execute({
                "action": "commit",
                "evidence_hash": new_hash,
                "approval_scope": "storage_only",
                "evidence_path": str(evidence_path),
            })
            r.check("new_hash_commit_ok", new_commit.ok,
                     f"error={new_commit.error}")

            # Restore original evidence
            evidence_path.write_text(json.dumps(MOCK_EVIDENCE, indent=2))

            # ══════════════════════════════════════════════════════════
            # Acceptance Test 5: Record explicitly authorizes storage;
            #   licensing and release receive separate authorization.
            # ══════════════════════════════════════════════════════════

            # Transaction record must have authorization_scope
            tx_files = sorted(tx_dir.glob("tx_*.json"))
            r.check("tx_files_exist", len(tx_files) >= 1,
                     f"count={len(tx_files)}")
            if tx_files:
                tx = json.loads(tx_files[-1].read_text())
                r.check("tx_has_scope",
                         tx.get("authorization_scope") == "storage_only",
                         f"scope={tx.get('authorization_scope')}")
                r.check("tx_schema",
                         tx.get("schema") == "cell.vault_transaction.v1",
                         f"schema={tx.get('schema')}")

            # Commit with wrong scope fails
            review_for_scope = agent.execute({"action": "store"})
            scope_hash = review_for_scope.output["evidence_hash"]

            bad_scope = agent.execute({
                "action": "commit",
                "evidence_hash": scope_hash,
                "approval_scope": "licensing",
            })
            r.check("wrong_scope_rejected",
                     not bad_scope.ok,
                     f"ok={bad_scope.ok}, error={bad_scope.error}")
            r.check("wrong_scope_error_msg",
                     "storage_only" in (bad_scope.error or ""),
                     f"error={bad_scope.error}")

            # Commit with empty scope fails
            no_scope = agent.execute({
                "action": "commit",
                "evidence_hash": scope_hash,
            })
            r.check("no_scope_rejected",
                     not no_scope.ok,
                     f"ok={no_scope.ok}, error={no_scope.error}")

            # Commit without evidence_hash fails
            no_hash = agent.execute({
                "action": "commit",
                "approval_scope": "storage_only",
            })
            r.check("no_hash_rejected",
                     not no_hash.ok,
                     f"ok={no_hash.ok}, error={no_hash.error}")

            # ══════════════════════════════════════════════════════════
            # Additional: status, error paths
            # ══════════════════════════════════════════════════════════

            # Status
            status = agent.execute({"action": "status"})
            r.check("status_ok", status.ok, f"error={status.error}")
            r.check("status_initialized",
                     status.output.get("initialized") is True,
                     f"init={status.output.get('initialized')}")
            r.check("status_has_blobs",
                     status.output.get("evidence_blobs", 0) >= 2,
                     f"blobs={status.output.get('evidence_blobs')}")

            # No secret
            del os.environ["ALMANAC_VAULT_SECRET"]
            no_secret = agent.execute({
                "action": "commit",
                "evidence_hash": scope_hash,
                "approval_scope": "storage_only",
            })
            r.check("no_secret_rejected",
                     not no_secret.ok and "secret" in (no_secret.error or "").lower(),
                     f"ok={no_secret.ok}, error={no_secret.error}")
            os.environ["ALMANAC_VAULT_SECRET"] = TEST_SECRET

            # Missing evidence
            missing = agent.execute({
                "action": "store",
                "evidence_path": "/nonexistent/path.json",
            })
            r.check("missing_evidence_rejected",
                     not missing.ok,
                     f"ok={missing.ok}, error={missing.error}")

            # Unknown action
            unknown = agent.execute({"action": "unknown"})
            r.check("unknown_action_rejected",
                     not unknown.ok,
                     f"ok={unknown.ok}, error={unknown.error}")

            # Wrong receipt hash in verify
            bad_verify = agent.execute({
                "action": "verify",
                "evidence_hash": evidence_hash,
                "receipt_hash": "0" * 64,
            })
            r.check("bad_receipt_verify_rejected",
                     not bad_verify.ok,
                     f"ok={bad_verify.ok}, error={bad_verify.error}")

        finally:
            mod.EVIDENCE_DIR = orig_evidence
            mod.TRANSACTION_DIR = orig_tx
            if orig_env is not None:
                os.environ["ALMANAC_VAULT_SECRET"] = orig_env
            elif "ALMANAC_VAULT_SECRET" in os.environ:
                del os.environ["ALMANAC_VAULT_SECRET"]

    return r


def test_all():
    """Pytest entry point."""
    r = run_tests()
    if r.failed:
        raise AssertionError(
            f"{r.failed}/{r.passed + r.failed} failed:\n"
            + "\n".join(r.details))


def main():
    r = run_tests()
    return r.report()


if __name__ == "__main__":
    sys.exit(main())
