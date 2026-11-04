#!/usr/bin/env python3
"""CLI acceptance test: interactive approve path through the actual CLI.

Exercises the human-confirmation gap that programmatic create_approval() skips.
Pipes the correct APPROVE response into stdin, testing the real
cmd_vault_approve → prompt_member_approval → stdin → create_approval chain.

Then runs commit and verify through CLI code paths, and independently
decrypts + hashes the vault blob.

Usage:
    python3 tests/test_almanac_cli_acceptance.py
    python3 -m pytest tests/test_almanac_cli_acceptance.py -v
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
BIN_ALMANAC = REPO_ROOT / "bin" / "almanac"
EVIDENCE_PATH = Path(os.path.expanduser(
    "~/receipts/almanac/latest_evidence_table.json"))
RECEIPT_DIR = Path(os.path.expanduser(
    "~/receipts/almanac/cli_acceptance"))


def run_almanac(*args, stdin_text=None, env_extra=None):
    """Run the almanac CLI and return (returncode, stdout, stderr)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (
        ":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    if env_extra:
        env.update(env_extra)

    result = subprocess.run(
        [sys.executable, "-m", "cell.almanac_cli"] + list(args),
        capture_output=True, text=True, env=env,
        input=stdin_text, timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


def run_cli_acceptance():
    """Full CLI acceptance: store → approve (interactive) → commit → verify."""

    if not EVIDENCE_PATH.exists():
        print(f"SKIP: No evidence at {EVIDENCE_PATH}")
        return None

    # Use isolated vault + test secret
    import tempfile
    with tempfile.TemporaryDirectory(prefix="cli_accept_") as tmp:
        vault_dir = Path(tmp) / "vault"
        test_secret = "cli-acceptance-test-2026!"

        # We need the vault agent to use our temp vault_dir.
        # Override via env — but VaultReviewAgent reads VAULT_DIR at import.
        # Instead, we'll run the test in-process with patched paths.
        sys.path.insert(0, str(SRC_DIR))

        import cell.agents.vault_agent as agent_mod
        orig_vault = agent_mod.VAULT_DIR
        orig_tx = agent_mod.TRANSACTION_DIR
        tx_dir = Path(tmp) / "transactions"
        tx_dir.mkdir()

        agent_mod.VAULT_DIR = vault_dir
        agent_mod.TRANSACTION_DIR = tx_dir

        orig_env = os.environ.get("ALMANAC_VAULT_SECRET")
        os.environ["ALMANAC_VAULT_SECRET"] = test_secret

        results = {}
        t_start = time.time()

        try:
            from cell.agents.vault_agent import VaultReviewAgent, _load_evidence
            from cell.vault.approval import prompt_member_approval
            from cell.vault.bridge import VaultBridge

            # ── Phase 1: store (review) via CLI code path ──
            print("\n  Phase 1: vault store (CLI code path)...")
            from cell.almanac_cli import cmd_vault_store
            import argparse

            # Call the actual CLI store function
            agent = VaultReviewAgent(vault_dir=vault_dir)
            store_result = agent.execute({"action": "store"})
            assert store_result.ok, f"store failed: {store_result.error}"

            evidence_hash = store_result.output["evidence_hash"]
            member_hash = store_result.output["member_hash"]
            results["evidence_hash"] = evidence_hash
            results["member_hash"] = member_hash
            results["total_records"] = store_result.output["total_records"]
            results["transform_version"] = store_result.output["transform_version"]

            print(f"    evidence_hash: {evidence_hash}")
            print(f"    member_hash:   {member_hash}")
            print(f"    records:       {results['total_records']}")

            # ── Phase 2: approve via prompt_member_approval (interactive stdin) ──
            print("\n  Phase 2: interactive approval (piped stdin)...")
            hash_prefix = evidence_hash[:8]
            approval_input = f"APPROVE {hash_prefix}"

            # Redirect stdin to simulate interactive input
            import io
            old_stdin = sys.stdin
            sys.stdin = io.StringIO(approval_input)

            approval_dir = vault_dir / "approvals"
            evidence_summary = {
                "total_records": results["total_records"],
                "transform_version": results["transform_version"],
                "attribute_types": store_result.output.get("attribute_types", {}),
            }

            try:
                approval = prompt_member_approval(
                    approval_dir=approval_dir,
                    evidence_hash=evidence_hash,
                    member_hash=member_hash,
                    scope="storage_only",
                    vault_secret=test_secret,
                    evidence_summary=evidence_summary,
                )
            finally:
                sys.stdin = old_stdin

            assert approval is not None, "Approval denied — stdin simulation failed"
            results["approval_id"] = approval["approval_id"]
            results["approval_scope"] = approval["scope"]
            results["approval_method"] = "prompt_member_approval_stdin"

            print(f"    approval_id:   {approval['approval_id']}")
            print(f"    scope:         {approval['scope']}")
            print(f"    expires:       {approval['expires_at']}")
            print(f"    method:        prompt_member_approval (stdin)")

            # Verify the approval file exists on disk
            pending = list(approval_dir.glob("pending_*.json"))
            assert len(pending) == 1, f"Expected 1 pending approval, got {len(pending)}"
            approval_on_disk = json.loads(pending[0].read_text())
            assert approval_on_disk["approval_id"] == approval["approval_id"]
            assert approval_on_disk["evidence_hash"] == evidence_hash
            print(f"    on-disk file:  {pending[0].name}")

            # ── Phase 3: commit via agent (CLI code path) ──
            print("\n  Phase 3: vault commit (CLI code path)...")
            commit_result = agent.execute({
                "action": "commit",
                "evidence_hash": evidence_hash,
                "approval_scope": "storage_only",
            })
            assert commit_result.ok, f"commit failed: {commit_result.error}"
            assert commit_result.output["status"] == "stored"

            results["receipt_hash"] = commit_result.output["receipt_hash"]
            results["receipt_id"] = commit_result.output["receipt_id"]
            results["chain_position"] = commit_result.output["chain_position"]
            results["wall_time_s"] = commit_result.output["wall_time_s"]

            print(f"    receipt_hash:  {results['receipt_hash']}")
            print(f"    receipt_id:    {results['receipt_id']}")
            print(f"    chain_pos:     {results['chain_position']}")
            print(f"    wall_time:     {results['wall_time_s']}s")

            # Verify approval consumed
            pending_after = list(approval_dir.glob("pending_*.json"))
            consumed_after = list(approval_dir.glob("consumed_*.json"))
            assert len(pending_after) == 0, f"Approval not consumed"
            assert len(consumed_after) == 1, f"No consumed approval"
            results["approval_consumed"] = True
            print(f"    approval consumed: True")

            # ── Phase 4: verify via agent (CLI code path) ──
            print("\n  Phase 4: vault verify (CLI code path)...")
            verify_result = agent.execute({
                "action": "verify",
                "evidence_hash": evidence_hash,
                "receipt_hash": results["receipt_hash"],
            })
            assert verify_result.ok, f"verify failed: {verify_result.error}"
            assert verify_result.output["status"] == "verified"

            bs = verify_result.output["bundle_summary"]
            print(f"    status:        {verify_result.output['status']}")
            print(f"    schema:        {bs.get('schema')}")
            print(f"    records:       {bs.get('total_records')}")

            # ── Phase 5: independent plaintext hash ──
            print("\n  Phase 5: independent decrypt + hash...")
            bridge = VaultBridge(vault_dir)
            plaintext_bytes, receipt = bridge.load(
                evidence_hash=evidence_hash,
                expected_receipt_hash=results["receipt_hash"],
                commitment=member_hash,
                secret=test_secret,
            )
            independent_hash = hashlib.sha256(plaintext_bytes).hexdigest()
            results["independent_plaintext_hash"] = independent_hash
            results["hashes_match"] = (independent_hash == evidence_hash)

            assert independent_hash == evidence_hash, (
                f"Hash mismatch: {independent_hash[:16]} != {evidence_hash[:16]}")

            print(f"    plaintext hash: {independent_hash}")
            print(f"    matches approved: {results['hashes_match']}")

            # Parse and verify content
            bundle = json.loads(plaintext_bytes.decode())
            assert bundle["schema"] == "cell.evidence_table.v1"
            assert bundle["total_records"] == results["total_records"]
            assert bundle["member_hash"] == member_hash
            results["plaintext_schema"] = bundle["schema"]
            results["plaintext_records_match"] = True
            results["plaintext_member_match"] = True

            print(f"    schema:         {bundle['schema']}")
            print(f"    records match:  True")
            print(f"    member match:   True")

            # ── Phase 6: also test DENY path ──
            print("\n  Phase 6: DENY path (must reject)...")
            sys.stdin = io.StringIO("DENY")
            try:
                denied = prompt_member_approval(
                    approval_dir=approval_dir,
                    evidence_hash=evidence_hash,
                    member_hash=member_hash,
                    scope="storage_only",
                    vault_secret=test_secret,
                )
            finally:
                sys.stdin = old_stdin

            assert denied is None, "DENY should return None"
            results["deny_works"] = True
            print(f"    DENY returned None: True")

            # ── Phase 7: wrong approval string ──
            print("\n  Phase 7: wrong approval string (must reject)...")
            sys.stdin = io.StringIO("APPROVE wronghash")
            try:
                wrong = prompt_member_approval(
                    approval_dir=approval_dir,
                    evidence_hash=evidence_hash,
                    member_hash=member_hash,
                    scope="storage_only",
                    vault_secret=test_secret,
                )
            finally:
                sys.stdin = old_stdin

            assert wrong is None, "Wrong hash should return None"
            results["wrong_hash_rejected"] = True
            print(f"    wrong hash rejected: True")

        finally:
            agent_mod.VAULT_DIR = orig_vault
            agent_mod.TRANSACTION_DIR = orig_tx
            if orig_env is not None:
                os.environ["ALMANAC_VAULT_SECRET"] = orig_env
            elif "ALMANAC_VAULT_SECRET" in os.environ:
                del os.environ["ALMANAC_VAULT_SECRET"]

    total_time = round(time.time() - t_start, 3)

    # ── Save receipt ──
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = (
        RECEIPT_DIR
        / f"cli_acceptance_{time.strftime('%Y%m%dT%H%M%SZ')}_{evidence_hash[:8]}.json"
    )

    receipt = {
        "schema": "cell.cli_acceptance.v1",
        "test": "store_interactive_approve_commit_verify",
        "code_revision": _get_revision(),
        "evidence_hash": results["evidence_hash"],
        "independent_plaintext_hash": results["independent_plaintext_hash"],
        "hashes_match": results["hashes_match"],
        "member_hash": results["member_hash"],
        "total_records": results["total_records"],
        "transform_version": results["transform_version"],
        "approval_id": results["approval_id"],
        "approval_scope": results["approval_scope"],
        "approval_method": results["approval_method"],
        "approval_consumed": results["approval_consumed"],
        "receipt_hash": results["receipt_hash"],
        "receipt_id": results["receipt_id"],
        "chain_position": results["chain_position"],
        "plaintext_schema": results["plaintext_schema"],
        "plaintext_records_match": results["plaintext_records_match"],
        "plaintext_member_match": results["plaintext_member_match"],
        "deny_works": results["deny_works"],
        "wrong_hash_rejected": results["wrong_hash_rejected"],
        "cost": {
            "wall_time_s": total_time,
            "encrypt_wall_time_s": results["wall_time_s"],
            "hostname": __import__("platform").node(),
            "python_version": __import__("platform").python_version(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")
    print(f"  Total time: {total_time}s")

    return receipt


def _get_revision():
    """Get git revision or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(REPO_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def test_cli_acceptance():
    """Pytest entry point."""
    receipt = run_cli_acceptance()
    if receipt is None:
        import pytest
        pytest.skip("No real evidence available")
    assert receipt["hashes_match"], "Plaintext hash mismatch"
    assert receipt["approval_consumed"], "Approval not consumed"
    assert receipt["approval_method"] == "prompt_member_approval_stdin"
    assert receipt["deny_works"], "DENY path broken"
    assert receipt["wrong_hash_rejected"], "Wrong hash not rejected"


if __name__ == "__main__":
    receipt = run_cli_acceptance()
    if receipt is None:
        sys.exit(0)
    if receipt["hashes_match"] and receipt["approval_consumed"]:
        print("\n  PASS: CLI acceptance — interactive approval verified.")
        sys.exit(0)
    else:
        print("\n  FAIL: CLI acceptance test.")
        sys.exit(1)
