"""Vault agent round-trip tests.

Tests the member review → vault store → verify pipeline end-to-end:
  1. Store evidence in vault (encrypted, receipted)
  2. Verify round-trip (decrypt, check hashes)
  3. Status reporting
  4. Error cases (missing secret, missing evidence, bad hash)

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


# Minimal evidence table for testing
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

# Test vault secret (meets 12-char + 4-unique-char requirements)
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


def run_tests():
    r = TestResults()

    with tempfile.TemporaryDirectory() as tmp:
        vault_dir = Path(tmp) / "vault"
        evidence_dir = Path(tmp) / "evidence"
        tx_dir = Path(tmp) / "transactions"
        evidence_dir.mkdir()
        tx_dir.mkdir()

        # Write mock evidence table
        evidence_path = evidence_dir / "latest_evidence_table.json"
        evidence_path.write_text(json.dumps(MOCK_EVIDENCE, indent=2))

        # Patch module-level paths
        import cell.agents.vault_agent as mod
        orig_evidence = mod.EVIDENCE_DIR
        orig_tx = mod.TRANSACTION_DIR
        mod.EVIDENCE_DIR = evidence_dir
        mod.TRANSACTION_DIR = tx_dir

        # Set secret via env
        orig_env = os.environ.get("ALMANAC_VAULT_SECRET")
        os.environ["ALMANAC_VAULT_SECRET"] = TEST_SECRET

        try:
            agent = VaultReviewAgent(vault_dir=vault_dir)

            # ── Test 1: Status before any vault operations ──
            r1 = agent.execute({"action": "status"})
            r.check("status_before_ok", r1.ok, f"error={r1.error}")
            r.check("status_before_no_vault",
                     not r1.output.get("initialized"),
                     f"initialized={r1.output.get('initialized')}")
            r.check("status_has_secret",
                     r1.output.get("has_secret") is True,
                     f"has_secret={r1.output.get('has_secret')}")
            r.check("status_has_evidence",
                     r1.output.get("has_evidence_table") is True,
                     f"has_evidence={r1.output.get('has_evidence_table')}")

            # ── Test 2: Store evidence in vault ──
            r2 = agent.execute({
                "action": "store",
                "evidence_path": str(evidence_path),
            })
            r.check("store_ok", r2.ok, f"error={r2.error}")
            r.check("store_has_evidence_hash",
                     bool(r2.output.get("evidence_hash")),
                     f"output={r2.output}")
            r.check("store_has_receipt_hash",
                     bool(r2.output.get("receipt_hash")),
                     f"output={r2.output}")
            r.check("store_has_receipt_id",
                     bool(r2.output.get("receipt_id")),
                     f"output={r2.output}")
            r.check("store_member_hash",
                     r2.output.get("member_hash") == "test_member_abc123",
                     f"member_hash={r2.output.get('member_hash')}")
            r.check("store_has_transaction",
                     bool(r2.output.get("transaction_path")),
                     f"output={r2.output}")
            r.check("store_receipt_present",
                     r2.receipt is not None,
                     "receipt missing")

            evidence_hash = r2.output.get("evidence_hash", "")
            receipt_hash = r2.output.get("receipt_hash", "")

            # Transaction file exists
            if r2.output.get("transaction_path"):
                tx_path = Path(r2.output["transaction_path"])
                r.check("transaction_file_exists",
                         tx_path.exists(),
                         f"path={tx_path}")
                if tx_path.exists():
                    tx = json.loads(tx_path.read_text())
                    r.check("transaction_schema",
                             tx.get("schema") == "cell.vault_transaction.v1",
                             f"schema={tx.get('schema')}")
                    r.check("transaction_evidence_hash",
                             tx.get("evidence_hash") == evidence_hash,
                             f"hash={tx.get('evidence_hash')}")

            # ── Test 3: Status after store ──
            r3 = agent.execute({"action": "status"})
            r.check("status_after_ok", r3.ok, f"error={r3.error}")
            r.check("status_after_initialized",
                     r3.output.get("initialized") is True,
                     f"initialized={r3.output.get('initialized')}")
            r.check("status_after_has_blob",
                     r3.output.get("evidence_blobs", 0) >= 1,
                     f"blobs={r3.output.get('evidence_blobs')}")
            r.check("status_after_has_receipt",
                     r3.output.get("receipts", 0) >= 1,
                     f"receipts={r3.output.get('receipts')}")
            r.check("status_after_has_tx",
                     r3.output.get("transactions", 0) >= 1,
                     f"transactions={r3.output.get('transactions')}")

            # ── Test 4: Verify the stored record ──
            r4 = agent.execute({
                "action": "verify",
                "evidence_hash": evidence_hash,
                "receipt_hash": receipt_hash,
            })
            r.check("verify_ok", r4.ok, f"error={r4.error}")
            r.check("verify_status",
                     r4.output.get("status") == "verified",
                     f"status={r4.output.get('status')}")
            r.check("verify_evidence_hash",
                     r4.output.get("evidence_hash") == evidence_hash,
                     f"hash={r4.output.get('evidence_hash')}")
            r.check("verify_bundle_summary",
                     r4.output.get("bundle_summary", {}).get("schema") == "cell.evidence_table.v1",
                     f"summary={r4.output.get('bundle_summary')}")
            r.check("verify_record_count",
                     r4.output.get("bundle_summary", {}).get("total_records") == 100,
                     f"records={r4.output.get('bundle_summary', {}).get('total_records')}")

            # ── Test 5: Verify with wrong receipt hash fails ──
            r5 = agent.execute({
                "action": "verify",
                "evidence_hash": evidence_hash,
                "receipt_hash": "0000" * 16,
            })
            r.check("wrong_receipt_fails",
                     not r5.ok,
                     f"ok={r5.ok}, error={r5.error}")

            # ── Test 6: Verify with wrong evidence hash fails ──
            r6 = agent.execute({
                "action": "verify",
                "evidence_hash": "0000" * 16,
                "receipt_hash": receipt_hash,
            })
            r.check("wrong_evidence_fails",
                     not r6.ok,
                     f"ok={r6.ok}, error={r6.error}")

            # ── Test 7: Store without secret fails ──
            del os.environ["ALMANAC_VAULT_SECRET"]
            r7 = agent.execute({"action": "store"})
            r.check("no_secret_fails",
                     not r7.ok and "secret" in (r7.error or "").lower(),
                     f"ok={r7.ok}, error={r7.error}")
            os.environ["ALMANAC_VAULT_SECRET"] = TEST_SECRET

            # ── Test 8: Store with missing evidence fails ──
            r8 = agent.execute({
                "action": "store",
                "evidence_path": "/nonexistent/path.json",
            })
            r.check("missing_evidence_fails",
                     not r8.ok,
                     f"ok={r8.ok}, error={r8.error}")

            # ── Test 9: Unknown action fails ──
            r9 = agent.execute({"action": "unknown"})
            r.check("unknown_action_fails",
                     not r9.ok,
                     f"ok={r9.ok}, error={r9.error}")

            # ── Test 10: Second store creates chain link ──
            r10 = agent.execute({
                "action": "store",
                "evidence_path": str(evidence_path),
            })
            r.check("second_store_ok", r10.ok, f"error={r10.error}")
            r.check("second_store_chain_position",
                     r10.output.get("chain_position", -1) >= 1,
                     f"chain_pos={r10.output.get('chain_position')}")

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
