"""Gate 6: Artifact Preflight tests — 7 pass conditions from spec.

Pass conditions:
  1. Active artifact with valid receipt → ALLOW
  2. Quarantined artifact → REJECT with PF-QUARANTINED
  3. Missing receipt → REJECT with PF-NO-RECEIPT
  4. Hash mismatch → REJECT with PF-HASH-MISMATCH
  5. Chain unavailable → degrade to local-receipt-only (ALLOW with warning)
  6. FGIP low conviction on claim artifact → HOLD (mapped to PF-SPEC-UNKNOWN)
  7. Every decision produces a receipt with cost block
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.artifact_preflight import (
    ALLOW, HOLD, REJECT,
    PreflightResult,
    preflight_check,
    emit_preflight_receipt,
    sha256_file,
    sha256_bytes,
    TRUSTED_VALIDATION_SPECS,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_artifact(tmpdir: str, content: bytes = b"test artifact content") -> str:
    """Create a temp artifact file, return its path."""
    p = os.path.join(tmpdir, "artifact.bin")
    with open(p, "wb") as f:
        f.write(content)
    return p


def _make_receipt(tmpdir: str, receipt_dict: dict) -> str:
    """Write a receipt JSON, return its path."""
    p = os.path.join(tmpdir, "receipt.json")
    with open(p, "w") as f:
        json.dump(receipt_dict, f)
    return p


def _content_hash(path: str) -> str:
    return sha256_file(Path(path))


# ── Pass condition 1: Active artifact with valid receipt → ALLOW ─────────────

def test_active_artifact_allow():
    """Active on-chain + valid receipt + matching hash → ALLOW, PF-CLEAN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)

        # Use the real trusted validation spec hash
        spec_hash = list(TRUSTED_VALIDATION_SPECS.keys())[0]

        receipt = _make_receipt(tmpdir, {
            "content_hash": content_hash,
            "validation_spec_hash": spec_hash,
        })

        def chain_active(h):
            return {"status": "Active"}

        result = preflight_check(artifact, receipt, on_chain_query=chain_active)

        assert result.decision == ALLOW, f"Expected ALLOW, got {result.decision}"
        assert "PF-CLEAN" in result.reason_codes
        assert result.chain_queried is True
        assert result.on_chain_status == "Active"
        assert result.content_hash_expected == content_hash
        assert result.content_hash_actual == content_hash


def test_active_artifact_no_chain_query():
    """Valid receipt, no chain query configured → ALLOW (local-only)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)
        spec_hash = list(TRUSTED_VALIDATION_SPECS.keys())[0]

        receipt = _make_receipt(tmpdir, {
            "content_hash": content_hash,
            "validation_spec_hash": spec_hash,
        })

        result = preflight_check(artifact, receipt)

        assert result.decision == ALLOW
        assert "PF-CLEAN" in result.reason_codes
        assert result.chain_queried is False


# ── Pass condition 2: Quarantined artifact → REJECT with PF-QUARANTINED ─────

def test_quarantined_artifact_reject():
    """On-chain status Quarantined → REJECT, PF-QUARANTINED."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)

        receipt = _make_receipt(tmpdir, {
            "content_hash": content_hash,
            "validation_spec_hash": list(TRUSTED_VALIDATION_SPECS.keys())[0],
        })

        def chain_quarantined(h):
            return {"status": "Quarantined"}

        result = preflight_check(artifact, receipt, on_chain_query=chain_quarantined)

        assert result.decision == REJECT, f"Expected REJECT, got {result.decision}"
        assert "PF-QUARANTINED" in result.reason_codes
        assert result.on_chain_status == "Quarantined"


def test_candidate_artifact_reject():
    """On-chain status Candidate (not yet promoted) → REJECT, PF-NOT-ACTIVE."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)

        receipt = _make_receipt(tmpdir, {
            "content_hash": content_hash,
            "validation_spec_hash": list(TRUSTED_VALIDATION_SPECS.keys())[0],
        })

        def chain_candidate(h):
            return {"status": "Candidate"}

        result = preflight_check(artifact, receipt, on_chain_query=chain_candidate)

        assert result.decision == REJECT
        assert "PF-NOT-ACTIVE" in result.reason_codes


# ── Pass condition 3: Missing receipt → REJECT with PF-NO-RECEIPT ────────────

def test_missing_receipt_reject():
    """Receipt file does not exist → REJECT, PF-NO-RECEIPT."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        fake_receipt = os.path.join(tmpdir, "nonexistent_receipt.json")

        result = preflight_check(artifact, fake_receipt)

        assert result.decision == REJECT
        assert "PF-NO-RECEIPT" in result.reason_codes


def test_corrupt_receipt_reject():
    """Receipt file exists but is not valid JSON → REJECT, PF-NO-RECEIPT."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        receipt_path = os.path.join(tmpdir, "receipt.json")
        with open(receipt_path, "w") as f:
            f.write("NOT VALID JSON {{{")

        result = preflight_check(artifact, receipt_path)

        assert result.decision == REJECT
        assert "PF-NO-RECEIPT" in result.reason_codes
        assert result.details.get("parse_error") is True


# ── Pass condition 4: Hash mismatch → REJECT with PF-HASH-MISMATCH ──────────

def test_hash_mismatch_reject():
    """Content hash in receipt doesn't match artifact → REJECT, PF-HASH-MISMATCH."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir, content=b"real content")

        receipt = _make_receipt(tmpdir, {
            "content_hash": "0000000000000000000000000000000000000000000000000000000000000000",
        })

        result = preflight_check(artifact, receipt)

        assert result.decision == REJECT
        assert "PF-HASH-MISMATCH" in result.reason_codes
        assert result.content_hash_expected == "0" * 64
        assert result.content_hash_actual != "0" * 64


def test_hash_mismatch_directory():
    """Directory content hash mismatch → REJECT."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create artifact directory with files
        artifact_dir = os.path.join(tmpdir, "artifact_dir")
        os.makedirs(artifact_dir)
        with open(os.path.join(artifact_dir, "data.txt"), "w") as f:
            f.write("some data")

        receipt = _make_receipt(tmpdir, {
            "content_hash": "0" * 64,
        })

        result = preflight_check(artifact_dir, receipt)

        assert result.decision == REJECT
        assert "PF-HASH-MISMATCH" in result.reason_codes


# ── Pass condition 5: Chain unavailable → degrade (ALLOW with warning) ───────

def test_chain_unavailable_degrade():
    """Chain query throws exception → ALLOW with PF-CHAIN-UNAVAILABLE warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)
        spec_hash = list(TRUSTED_VALIDATION_SPECS.keys())[0]

        receipt = _make_receipt(tmpdir, {
            "content_hash": content_hash,
            "validation_spec_hash": spec_hash,
        })

        def chain_fails(h):
            raise ConnectionError("devnet unreachable")

        result = preflight_check(artifact, receipt, on_chain_query=chain_fails)

        assert result.decision == ALLOW, f"Expected ALLOW (degraded), got {result.decision}"
        assert "PF-CHAIN-UNAVAILABLE" in result.reason_codes
        assert "chain_error" in result.details


def test_chain_returns_none():
    """Chain query returns None (no account found) → ALLOW, no error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)
        spec_hash = list(TRUSTED_VALIDATION_SPECS.keys())[0]

        receipt = _make_receipt(tmpdir, {
            "content_hash": content_hash,
            "validation_spec_hash": spec_hash,
        })

        def chain_not_found(h):
            return None

        result = preflight_check(artifact, receipt, on_chain_query=chain_not_found)

        assert result.decision == ALLOW
        assert result.chain_queried is True
        assert result.on_chain_status is None
        assert "chain_note" in result.details


# ── Pass condition 6: Unknown spec → HOLD (proxy for FGIP low conviction) ───

def test_unknown_spec_hold():
    """Unknown validation_spec_hash → HOLD with PF-SPEC-UNKNOWN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)

        receipt = _make_receipt(tmpdir, {
            "content_hash": content_hash,
            "validation_spec_hash": "deadbeef" * 8,  # Not in trusted list
        })

        result = preflight_check(artifact, receipt)

        assert result.decision == HOLD, f"Expected HOLD, got {result.decision}"
        assert "PF-SPEC-UNKNOWN" in result.reason_codes
        assert result.details["unknown_spec_hash"] == "deadbeef" * 8


def test_known_spec_allow():
    """Known validation_spec_hash → no PF-SPEC-UNKNOWN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)
        spec_hash = list(TRUSTED_VALIDATION_SPECS.keys())[0]

        receipt = _make_receipt(tmpdir, {
            "content_hash": content_hash,
            "validation_spec_hash": spec_hash,
        })

        result = preflight_check(artifact, receipt)

        assert result.decision == ALLOW
        assert "PF-SPEC-UNKNOWN" not in result.reason_codes
        assert "PF-CLEAN" in result.reason_codes


# ── Pass condition 7: Every decision produces a receipt with cost block ──────

def test_receipt_emitted_allow():
    """ALLOW decision produces receipt with cost block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)
        spec_hash = list(TRUSTED_VALIDATION_SPECS.keys())[0]

        receipt = _make_receipt(tmpdir, {
            "content_hash": content_hash,
            "validation_spec_hash": spec_hash,
        })

        result = preflight_check(artifact, receipt)
        assert result.decision == ALLOW

        receipt_out = os.path.join(tmpdir, "preflight_receipt.json")
        emitted = emit_preflight_receipt(result, receipt_out, wall_time_s=0.05, cpu_time_s=0.04)

        assert os.path.exists(receipt_out)
        assert emitted["decision"] == ALLOW
        assert emitted["schema_version"] == "gate6_preflight_receipt:v1"
        assert "cost" in emitted
        assert emitted["cost"]["wall_time_s"] == 0.05
        assert emitted["cost"]["cpu_time_s"] == 0.04
        assert "peak_memory_mb" in emitted["cost"]
        assert "python_version" in emitted["cost"]
        assert "hostname" in emitted["cost"]
        assert "timestamp_utc" in emitted

        # Verify the file on disk matches
        on_disk = json.loads(Path(receipt_out).read_text())
        assert on_disk == emitted


def test_receipt_emitted_reject():
    """REJECT decision also produces a receipt with cost block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)

        result = preflight_check(artifact, os.path.join(tmpdir, "nope.json"))
        assert result.decision == REJECT

        receipt_out = os.path.join(tmpdir, "preflight_receipt.json")
        emitted = emit_preflight_receipt(result, receipt_out, wall_time_s=0.01, cpu_time_s=0.01)

        assert os.path.exists(receipt_out)
        assert emitted["decision"] == REJECT
        assert "PF-NO-RECEIPT" in emitted["reason_codes"]
        assert "cost" in emitted


def test_receipt_emitted_hold():
    """HOLD decision also produces a receipt with cost block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)

        receipt = _make_receipt(tmpdir, {
            "content_hash": content_hash,
            "validation_spec_hash": "unknown" * 8,
        })

        result = preflight_check(artifact, receipt)
        assert result.decision == HOLD

        receipt_out = os.path.join(tmpdir, "preflight_receipt.json")
        emitted = emit_preflight_receipt(result, receipt_out, wall_time_s=0.02)

        assert os.path.exists(receipt_out)
        assert emitted["decision"] == HOLD
        assert "cost" in emitted


# ── Edge cases ───────────────────────────────────────────────────────────────

def test_receipt_with_nested_hashes():
    """Receipt with content_hash under 'hashes' key (Gate 4 receipt format)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)
        spec_hash = list(TRUSTED_VALIDATION_SPECS.keys())[0]

        receipt = _make_receipt(tmpdir, {
            "hashes": {
                "content_hash": content_hash,
                "validation_spec_hash": spec_hash,
            },
        })

        result = preflight_check(artifact, receipt)

        assert result.decision == ALLOW
        assert "PF-CLEAN" in result.reason_codes


def test_no_content_hash_in_receipt():
    """Receipt without content_hash → skip hash check, still ALLOW."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)

        receipt = _make_receipt(tmpdir, {
            "tool": "some_gate",
            "status": "PASS",
        })

        result = preflight_check(artifact, receipt)

        assert result.decision == ALLOW
        assert "PF-CLEAN" in result.reason_codes


def test_custom_trusted_specs():
    """Custom trusted_specs dict overrides default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _make_artifact(tmpdir)
        content_hash = _content_hash(artifact)
        custom_hash = "custom" + "0" * 58

        receipt = _make_receipt(tmpdir, {
            "content_hash": content_hash,
            "validation_spec_hash": custom_hash,
        })

        # Without custom specs → HOLD
        result1 = preflight_check(artifact, receipt)
        assert result1.decision == HOLD

        # With custom specs → ALLOW
        result2 = preflight_check(
            artifact, receipt,
            trusted_specs={custom_hash: "custom_spec"},
        )
        assert result2.decision == ALLOW


# ── Integration: real Gate 4 receipt ─────────────────────────────────────────

def test_real_gate4_receipt():
    """Preflight against real Gate 4 receipt (if available)."""
    gate4_receipt = Path.home() / "EchoLivingSystem" / "receipts" / "gate4_solana_scientific_artifact" / "gate4_receipt.json"
    gate1_ledger = Path.home() / "EchoLivingSystem" / "receipts" / "gate1_ctc_real_data" / "ctc_experiment_ledger.jsonl.gz"

    if not gate4_receipt.exists():
        return  # Skip if not available

    result = preflight_check(
        str(gate1_ledger) if gate1_ledger.exists() else str(gate4_receipt),
        str(gate4_receipt),
    )

    # Gate 4 receipt has content_hash under "hashes" key.
    # The content_hash is a composite of gate1+gate2 ledger hashes,
    # not the hash of gate1_ledger alone, so hash check may not match
    # a single file. But receipt should parse and spec should be known.
    assert result.decision in (ALLOW, REJECT)  # REJECT if hash mismatch (expected)
    assert result.receipt_hash  # Receipt was hashed


if __name__ == "__main__":
    tests = [
        ("active_artifact_allow", test_active_artifact_allow),
        ("active_artifact_no_chain", test_active_artifact_no_chain_query),
        ("quarantined_artifact_reject", test_quarantined_artifact_reject),
        ("candidate_artifact_reject", test_candidate_artifact_reject),
        ("missing_receipt_reject", test_missing_receipt_reject),
        ("corrupt_receipt_reject", test_corrupt_receipt_reject),
        ("hash_mismatch_reject", test_hash_mismatch_reject),
        ("hash_mismatch_directory", test_hash_mismatch_directory),
        ("chain_unavailable_degrade", test_chain_unavailable_degrade),
        ("chain_returns_none", test_chain_returns_none),
        ("unknown_spec_hold", test_unknown_spec_hold),
        ("known_spec_allow", test_known_spec_allow),
        ("receipt_emitted_allow", test_receipt_emitted_allow),
        ("receipt_emitted_reject", test_receipt_emitted_reject),
        ("receipt_emitted_hold", test_receipt_emitted_hold),
        ("receipt_with_nested_hashes", test_receipt_with_nested_hashes),
        ("no_content_hash_in_receipt", test_no_content_hash_in_receipt),
        ("custom_trusted_specs", test_custom_trusted_specs),
        ("real_gate4_receipt", test_real_gate4_receipt),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print()
    print(f"Gate 6 preflight tests: {passed}/{passed + failed} PASS")
    if failed:
        raise SystemExit(1)
