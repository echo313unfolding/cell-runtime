#!/usr/bin/env python3
"""Phase 6 smoke test — regulated asset event through the security substrate.

Proves:
  1. Synthetic transfer event parsed and validated
  2. Deterministic risk policy applied (reason codes, risk level)
  3. SSM/RAG/graph context pack assembled
  4. Live Sentinel produces risk assessment (if backend available)
  5. Gate controls decision execution
  6. Receipt written with all required fields

Usage:
    python3 tools/smoke_regulated_asset_event_path.py
    python3 tools/smoke_regulated_asset_event_path.py --port 8085
    python3 tools/smoke_regulated_asset_event_path.py --no-sentinel  # skip live model call
"""
import json
import os
import platform
import resource
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.base import AgentRegistry
from cell.agents.rag_agent import RAGLookupAgent
from cell.agents.graph_agent import GraphStatsAgent
from cell.agents.ssm_agent import SSMGetStateAgent
from cell.agents.policy_agent import GateDecideAgent
from cell.regulated_asset_adapter import (
    TransferEvent, KYCAttestation, OracleSignal,
    evaluate_risk_policy, build_sentinel_prompt,
)


SCENARIOS = [
    {
        "name": "Clean low-value domestic transfer",
        "event": {
            "event_id": "RA-SMOKE-001",
            "event_type": "transfer",
            "wallet_from": "wallet-alice-001",
            "wallet_to": "wallet-bob-002",
            "asset_type": "stablecoin",
            "amount": 500,
            "amount_usd": 500,
            "jurisdiction": "US",
            "velocity_24h": 3,
            "cumulative_24h_usd": 1500,
        },
        "kyc": {
            "attestation_id": "KYC-ALICE-001",
            "wallet_id_hash": "abc123",
            "level": "enhanced",
            "jurisdiction": "US",
            "sanctions_screen_pass": True,
            "pep_screen_pass": True,
        },
        "expected_decision": "allow",
    },
    {
        "name": "High-value cross-border with basic KYC",
        "event": {
            "event_id": "RA-SMOKE-002",
            "event_type": "transfer",
            "wallet_from": "wallet-charlie-003",
            "wallet_to": "wallet-delta-004",
            "asset_type": "security_token",
            "amount": 75000,
            "amount_usd": 75000,
            "jurisdiction": "US",
            "counterparty_jurisdiction": "SG",
            "velocity_24h": 2,
            "cumulative_24h_usd": 75000,
        },
        "kyc": {
            "attestation_id": "KYC-CHARLIE-001",
            "wallet_id_hash": "def456",
            "level": "basic",
            "jurisdiction": "US",
        },
        "expected_decision": "reject",
    },
    {
        "name": "Sanctioned jurisdiction transfer",
        "event": {
            "event_id": "RA-SMOKE-003",
            "event_type": "transfer",
            "wallet_from": "wallet-echo-005",
            "wallet_to": "wallet-foxtrot-006",
            "asset_type": "stablecoin",
            "amount": 1000,
            "amount_usd": 1000,
            "jurisdiction": "KP",
            "velocity_24h": 1,
            "cumulative_24h_usd": 1000,
        },
        "kyc": None,
        "expected_decision": "reject",
    },
    {
        "name": "High velocity no-KYC burst",
        "event": {
            "event_id": "RA-SMOKE-004",
            "event_type": "transfer",
            "wallet_from": "wallet-golf-007",
            "wallet_to": "wallet-hotel-008",
            "asset_type": "utility_token",
            "amount": 200,
            "amount_usd": 200,
            "jurisdiction": "EU",
            "velocity_24h": 45,
            "cumulative_24h_usd": 9000,
        },
        "kyc": None,
        "expected_decision": "hold",
    },
]


def _sentinel_alive(port: int) -> bool:
    try:
        req = urllib.request.Request(f"http://localhost:{port}/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return json.loads(resp.read()).get("status") == "ok"
    except Exception:
        return False


def _call_sentinel(port: int, prompt: str) -> dict:
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": "You are a regulated asset risk assessment agent. "
             "Evaluate the transfer event and provide a brief risk assessment. "
             "Focus on compliance concerns, not technical details."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return {
        "content": data["choices"][0]["message"]["content"],
        "model": data.get("model", ""),
        "tokens": data.get("usage", {}).get("completion_tokens", 0),
        "tok_s": round(data.get("timings", {}).get("predicted_per_second", 0), 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6: Regulated Asset Event Path")
    parser.add_argument("--port", type=int, default=8085, help="Sentinel port")
    parser.add_argument("--no-sentinel", action="store_true", help="Skip live model call")
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 60)
    print("  Phase 6: Regulated Asset Event Path Smoke Test")
    print("=" * 60)

    use_sentinel = not args.no_sentinel and _sentinel_alive(args.port)
    if not use_sentinel and not args.no_sentinel:
        print("  WARNING: Sentinel backend not available, skipping model calls")

    # Build agent registry
    registry = AgentRegistry()
    for cls in [RAGLookupAgent, GraphStatsAgent, SSMGetStateAgent, GateDecideAgent]:
        registry.register(cls())

    # Assemble shared context pack once
    print("\n  Assembling context pack...")
    ssm_r = registry.run("ssm_get_state", {"entity_id": "regulated-asset-demo"})
    rag_r = registry.run("rag_lookup", {"query": "compliance OR regulation OR sanctions", "limit": 3})
    graph_r = registry.run("graph_stats", {})

    context_pack = {
        "ssm": ssm_r.output if ssm_r.ok else {},
        "rag": {"count": rag_r.output.get("count", 0), "source": rag_r.output.get("source", "")} if rag_r.ok else {},
        "graph": graph_r.output if graph_r.ok else {},
    }
    print(f"  SSM: {'OK' if ssm_r.ok else 'FAIL'}")
    print(f"  RAG: {'OK' if rag_r.ok else 'FAIL'} ({context_pack['rag'].get('count', 0)} hits)")
    print(f"  Graph: {'OK' if graph_r.ok else 'FAIL'}")

    scenario_results = []
    all_pass = True

    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n  --- Scenario {i}/{len(SCENARIOS)}: {scenario['name']} ---")

        # Parse event
        event = TransferEvent.from_dict(scenario["event"])
        kyc = KYCAttestation.from_dict(scenario["kyc"]) if scenario.get("kyc") else None
        oracle = None  # No oracle signals in demo scenarios

        # Apply deterministic policy
        policy = evaluate_risk_policy(event, kyc=kyc, oracle=oracle)
        print(f"  Policy: decision={policy['decision']}, risk={policy['risk_level']}, "
              f"score={policy['risk_score']}, codes={policy['reason_codes']}")

        # Check expected decision
        if policy["decision"] != scenario["expected_decision"]:
            print(f"  FAIL: expected {scenario['expected_decision']}, got {policy['decision']}")
            all_pass = False

        # Optional: Sentinel assessment
        sentinel_output = ""
        sentinel_meta = {}
        if use_sentinel:
            prompt = build_sentinel_prompt(event, context_pack, policy)
            try:
                sentinel_resp = _call_sentinel(args.port, prompt)
                sentinel_output = sentinel_resp["content"]
                sentinel_meta = sentinel_resp
                print(f"  Sentinel: {sentinel_resp['tokens']} tokens, {sentinel_resp['tok_s']} tok/s")
                # Show first line of output
                first_line = sentinel_output.strip().split("\n")[0][:100]
                print(f"  Output: {first_line}...")
            except Exception as e:
                print(f"  Sentinel: ERROR ({e})")

        # Gate check
        gate_r = registry.run("gate_decide", {
            "action": f"asset_{policy['decision']}",
            "detail": f"{policy['decision']} {event.event_id}: "
                      f"{event.amount_usd} USD {event.asset_type} "
                      f"({', '.join(policy['reason_codes'])})",
            "auto": True,
        })

        scenario_results.append({
            "scenario": scenario["name"],
            "event_id": event.event_id,
            "wallet_from_hash": event.wallet_from_hash,
            "asset_type": event.asset_type,
            "amount_usd": event.amount_usd,
            "jurisdiction": event.jurisdiction,
            "kyc_level": kyc.level if kyc else "none",
            "policy": policy,
            "sentinel_output": sentinel_output,
            "sentinel_meta": sentinel_meta,
            "gate_allowed": gate_r.output.get("allowed", False) if gate_r.ok else False,
            "expected": scenario["expected_decision"],
            "pass": policy["decision"] == scenario["expected_decision"],
        })

    # Write receipt
    wall_time = round(time.time() - t_start, 3)
    cpu_time = round(time.process_time() - cpu_start, 3)

    receipt = {
        "receipt_id": f"phase6_regulated_asset_adapter_{time.strftime('%Y%m%dT%H%M%SZ')}",
        "title": "Phase 6: Regulated Asset Adapter Demo",
        "date": time.strftime("%Y-%m-%d"),
        "status": "PASS" if all_pass else "FAIL",
        "scenarios_run": len(SCENARIOS),
        "scenarios_pass": sum(1 for r in scenario_results if r["pass"]),
        "sentinel_used": use_sentinel,
        "context_pack": {
            "ssm_available": ssm_r.ok,
            "rag_hits": context_pack["rag"].get("count", 0),
            "graph_nodes": context_pack["graph"].get("nodes", 0),
        },
        "scenarios": scenario_results,
        "cost": {
            "wall_time_s": wall_time,
            "cpu_time_s": cpu_time,
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    receipt_dir = Path(os.path.expanduser("~/receipts"))
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{receipt['receipt_id']}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    # Summary
    passed = sum(1 for r in scenario_results if r["pass"])
    print(f"\n{'=' * 60}")
    print(f"  Result: {'PASS' if all_pass else 'FAIL'} ({passed}/{len(SCENARIOS)} scenarios)")
    print(f"  Sentinel: {'live' if use_sentinel else 'skipped'}")
    print(f"  Wall time: {wall_time}s")
    print(f"  Receipt: {receipt_path}")
    print(f"{'=' * 60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
