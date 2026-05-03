#!/usr/bin/env python3
"""Smoke test: prove the full context pack path returns populated results.

Runs each context agent (RAG, graph, SSM) against real backends,
assembles a populated context pack, feeds it through the regulated-asset
policy pipeline, and writes a receipt proving every source returned data.

This is NOT a unit test (those mock backends). This proves the live
backends return non-empty results and the pipeline works end-to-end.

Usage:
    python3 tools/smoke_live_populated_context_pack.py
"""

import hashlib
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.base import AgentRegistry
from cell.agents.rag_agent import RAGLookupAgent
from cell.agents.graph_agent import GraphLookupAgent, GraphNeighborsAgent, GraphStatsAgent
from cell.agents.ssm_agent import SSMGetStateAgent
from cell.regulated_asset_adapter import (
    TransferEvent, KYCAttestation, OracleSignal,
    evaluate_risk_policy, build_sentinel_prompt,
)


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 60)
    print("LIVE POPULATED CONTEXT PACK SMOKE TEST")
    print("=" * 60)

    # --- 1. Register agents ---
    reg = AgentRegistry()
    reg.register(RAGLookupAgent())
    reg.register(GraphLookupAgent())
    reg.register(GraphNeighborsAgent())
    reg.register(GraphStatsAgent())
    reg.register(SSMGetStateAgent())

    results = {}
    all_ok = True

    # --- 2. RAG: FTS5 search ---
    print("\n[1/5] RAG lookup (FTS5)...")
    rag = reg.run("rag_lookup", {"query": "tariff", "scope": "claims", "limit": 5})
    rag_count = rag.output.get("count", 0) if rag.ok else 0
    rag_source = rag.output.get("source", "") if rag.ok else ""
    rag_ok = rag.ok and rag_count > 0
    results["rag"] = {
        "ok": rag.ok,
        "populated": rag_count > 0,
        "count": rag_count,
        "source": rag_source,
        "receipt": rag.receipt,
    }
    print(f"  ok={rag.ok}, hits={rag_count}, source={rag_source}")
    if not rag_ok:
        all_ok = False

    # --- 3. Graph: stats ---
    print("\n[2/5] Graph stats...")
    gstats = reg.run("graph_stats", {})
    graph_nodes = gstats.output.get("nodes", 0) if gstats.ok else 0
    graph_edges = gstats.output.get("edges", 0) if gstats.ok else 0
    graph_ok = gstats.ok and graph_nodes > 0
    results["graph_stats"] = {
        "ok": gstats.ok,
        "populated": graph_nodes > 0,
        "nodes": graph_nodes,
        "edges": graph_edges,
        "receipt": gstats.receipt,
    }
    print(f"  ok={gstats.ok}, nodes={graph_nodes}, edges={graph_edges}")
    if not graph_ok:
        all_ok = False

    # --- 4. Graph: neighbors ---
    print("\n[3/5] Graph neighbors lookup...")
    gneigh = reg.run("graph_neighbors", {"entity_id": "ADVANCE Act", "limit": 5})
    neighbor_count = gneigh.output.get("count", 0) if gneigh.ok else 0
    neighbor_ok = gneigh.ok and neighbor_count > 0
    results["graph_neighbors"] = {
        "ok": gneigh.ok,
        "populated": neighbor_count > 0,
        "count": neighbor_count,
        "receipt": gneigh.receipt,
    }
    print(f"  ok={gneigh.ok}, neighbors={neighbor_count}")
    if not neighbor_ok:
        all_ok = False

    # --- 5. SSM: state query ---
    print("\n[4/5] SSM state query...")
    # Use content from actual alerts in sentinel.db
    ssm = reg.run("ssm_get_state", {"entity_id": "FILE_CREATED", "limit": 10})
    ssm_found = ssm.output.get("found", False) if ssm.ok else False
    ssm_events = ssm.output.get("event_count", 0) if ssm.ok else 0
    ssm_verdicts = len(ssm.output.get("recent_verdicts", [])) if ssm.ok else 0
    ssm_ok = ssm.ok and (ssm_events > 0 or ssm_verdicts > 0)
    results["ssm"] = {
        "ok": ssm.ok,
        "populated": ssm_found,
        "event_count": ssm_events,
        "verdict_count": ssm_verdicts,
        "trend": ssm.output.get("trend", "") if ssm.ok else "",
        "receipt": ssm.receipt,
    }
    print(f"  ok={ssm.ok}, found={ssm_found}, events={ssm_events}, verdicts={ssm_verdicts}")
    if not ssm_ok:
        all_ok = False

    # --- 6. Assemble context pack and run policy ---
    print("\n[5/5] Regulated-asset policy with populated context pack...")

    event = TransferEvent.from_dict({
        "event_id": "smoke-test-001",
        "wallet_from": "0xSMOKE_SENDER",
        "wallet_to": "0xSMOKE_RECEIVER",
        "amount": 15000,
        "amount_usd": 15000,
        "jurisdiction": "US",
        "counterparty_jurisdiction": "GB",
        "velocity_24h": 5,
        "cumulative_24h_usd": 25000,
    })

    kyc = KYCAttestation(
        attestation_id="KYC-SMOKE-001",
        wallet_id_hash=event.wallet_from_hash,
        level="enhanced",
        jurisdiction="US",
        issued_at="2026-04-01T00:00:00Z",
    )

    oracle = OracleSignal(
        signal_id="ORC-SMOKE-001",
        signal_type="risk_score",
        value=45.0,
        confidence=0.9,
    )

    # Policy evaluation
    policy_result = evaluate_risk_policy(event, kyc=kyc, oracle=oracle)

    # Build context pack from live agent results
    context_pack = {
        "ssm": {
            "found": ssm_found,
            "event_count": ssm_events,
            "trend": ssm.output.get("trend", "") if ssm.ok else "unknown",
        },
        "rag": {
            "count": rag_count,
        },
        "graph": {
            "nodes": graph_nodes,
        },
    }

    # Build Sentinel prompt (proves context pack flows into prompt)
    prompt = build_sentinel_prompt(event, context_pack, policy_result)

    # Verify prompt contains context
    prompt_has_ssm = "SSM:" in prompt
    prompt_has_rag = "RAG:" in prompt
    prompt_has_graph = "Graph:" in prompt

    policy_ok = (
        policy_result["policy_version"] == "regulated_asset_v0.2"
        and len(policy_result["reason_codes"]) > 0
        and 0 <= policy_result["risk_score"] <= 100
    )

    results["policy"] = {
        "ok": policy_ok,
        "decision": policy_result["decision"],
        "risk_score": policy_result["risk_score"],
        "risk_level": policy_result["risk_level"],
        "reason_codes": policy_result["reason_codes"],
        "policy_version": policy_result["policy_version"],
    }

    results["prompt"] = {
        "has_ssm": prompt_has_ssm,
        "has_rag": prompt_has_rag,
        "has_graph": prompt_has_graph,
        "length": len(prompt),
    }

    print(f"  policy: {policy_result['decision']} (score={policy_result['risk_score']})")
    print(f"  reason_codes: {policy_result['reason_codes']}")
    print(f"  prompt includes SSM={prompt_has_ssm}, RAG={prompt_has_rag}, Graph={prompt_has_graph}")

    if not policy_ok:
        all_ok = False

    # --- Summary ---
    print("\n" + "=" * 60)
    populated_sources = sum(1 for k in ("rag", "graph_stats", "graph_neighbors", "ssm")
                           if results[k]["populated"])
    total_sources = 4
    print(f"Context sources populated: {populated_sources}/{total_sources}")
    print(f"Policy evaluation: {policy_result['decision']}")
    print(f"Overall: {'ALL POPULATED' if all_ok else 'SOME EMPTY'}")
    print("=" * 60)

    # --- Receipt ---
    receipt = {
        "receipt_id": f"live_populated_context_pack_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "title": "Live Populated Context Pack Smoke Test",
        "description": "Proves the generic Sentinel path with populated SSM/RAG/graph context, not just graceful degradation.",
        "status": "PASS" if all_ok else "PARTIAL",
        "populated_sources": populated_sources,
        "total_sources": total_sources,
        "all_populated": all_ok,
        "results": results,
        "prompt_preview": prompt[:500],
        "prompt_hash": sha256_str(prompt),
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
        },
    }

    receipt_dir = Path(os.path.expanduser("~/receipts"))
    receipt_path = receipt_dir / f"live_populated_context_pack_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\nReceipt: {receipt_path}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
