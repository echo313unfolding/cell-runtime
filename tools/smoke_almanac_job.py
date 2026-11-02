#!/usr/bin/env python3
"""Smoke test — Almanac job wiring through Cell Runtime.

Tests the full path:
  1. Router classifies almanac intents correctly
  2. AlmanacNormalizeAgent runs on real Takeout data + produces receipt
  3. AlmanacEvidenceAgent queries pre-computed evidence table
  4. Tools are registered and callable through tool dispatch
  5. Context compiler finds almanac receipts

Usage:
    python3 tools/smoke_almanac_job.py
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ALMANAC JOB WIRING SMOKE TEST")
    print("=" * 60)

    # ── 1. Router ────────────────────────────────────────────────
    print("\n--- Router Classification ---")
    from cell.router import classify

    almanac_queries = [
        "what does my youtube data say about me",
        "normalize my takeout export",
        "show my top channels by signal quality",
        "what does this source claim about me",
        "almanac evidence table",
    ]
    non_almanac = [
        ("write a python sort function", "coding"),
        ("explain quicksort", "reasoning"),
    ]

    for q in almanac_queries:
        intent = classify(q)
        check(f"route '{q[:40]}...'", intent == "almanac", f"got={intent}")

    for q, expected in non_almanac:
        intent = classify(q)
        check(f"non-almanac '{q[:30]}...'", intent == expected, f"got={intent}")

    # ── 2. Normalize Agent ───────────────────────────────────────
    print("\n--- AlmanacNormalizeAgent ---")
    from cell.agents.almanac_agent import AlmanacNormalizeAgent, EVIDENCE_DIR

    agent = AlmanacNormalizeAgent()
    result = agent.run({})

    check("normalize runs without error", result.ok, result.error or "")
    if result.ok:
        out = result.output
        check("total_records > 0", out["total_records"] > 0,
              f"got {out['total_records']}")
        check("mapped == total (no unmapped)",
              out["mapped_records"] == out["total_records"])
        check("provenance coverage 100%",
              out["provenance_coverage"] == "100%",
              f"got {out['provenance_coverage']}")
        check("has top_channels", len(out["top_channels"]) > 0)
        check("has signal_distribution",
              out["signal_distribution"] is not None)
        check("receipt file exists",
              Path(out["receipt_path"]).exists())

        # Verify receipt has cost block
        receipt = json.loads(Path(out["receipt_path"]).read_text())
        check("receipt has cost.wall_time_s",
              "wall_time_s" in receipt.get("cost", {}))
        check("receipt has source_sha256",
              len(receipt.get("source_sha256", "")) == 64)

    # ── 3. Evidence Agent ────────────────────────────────────────
    print("\n--- AlmanacEvidenceAgent ---")
    from cell.agents.almanac_agent import AlmanacEvidenceAgent

    ev_agent = AlmanacEvidenceAgent()

    r1 = ev_agent.run({"query": "top_channels"})
    check("top_channels query", r1.ok and len(r1.output.get("results", [])) > 0)

    r2 = ev_agent.run({"query": "signal_quality"})
    check("signal_quality query",
          r2.ok and "distribution" in r2.output)

    r3 = ev_agent.run({"query": "recency"})
    check("recency query",
          r3.ok and "year_distribution" in r3.output)

    r4 = ev_agent.run({"query": "channel:zeihan"})
    check("channel lookup",
          r4.ok and len(r4.output.get("results", [])) > 0)

    r5 = ev_agent.run({"query": "channel:nonexistent_xyz_123"})
    check("channel lookup miss returns empty",
          r5.ok and len(r5.output.get("results", [])) == 0)

    # ── 4. Tool Registry ─────────────────────────────────────────
    print("\n--- Tool Registry ---")
    from cell.tool_registry import create_default_registry

    reg = create_default_registry()
    check("almanac_normalize registered", reg.has("almanac_normalize"))
    check("almanac_evidence registered", reg.has("almanac_evidence"))

    # Tool dispatch
    dispatch_result = reg.execute("almanac_evidence",
                                  {"query": "top_channels"})
    try:
        data = json.loads(dispatch_result)
        check("tool dispatch returns valid JSON", True)
        check("dispatch has results",
              len(data.get("results", [])) > 0)
    except (json.JSONDecodeError, TypeError) as e:
        check("tool dispatch returns valid JSON", False, str(e))

    # ── 5. Evidence table on disk ────────────────────────────────
    print("\n--- Evidence Table Persistence ---")
    evidence_path = EVIDENCE_DIR / "latest_evidence_table.json"
    check("latest_evidence_table.json exists", evidence_path.exists())
    if evidence_path.exists():
        table = json.loads(evidence_path.read_text())
        check("table has member_hash", "member_hash" in table)
        check("table has source_sha256", "source_sha256" in table)
        check("table has attributes", len(table.get("attributes", [])) > 0)

        # Verify all attributes have provenance
        attrs = table["attributes"]
        has_locator = sum(1 for a in attrs if a.get("source_locator"))
        check("all attributes have source_locator",
              has_locator == len(attrs),
              f"{has_locator}/{len(attrs)}")

    # ── Summary ──────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 2)
    total = PASS + FAIL
    print(f"\n{'=' * 60}")
    print(f"  RESULT: {PASS}/{total} PASS, {FAIL}/{total} FAIL")
    print(f"  Wall time: {elapsed}s")
    print(f"{'=' * 60}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
