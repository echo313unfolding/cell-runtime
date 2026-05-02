#!/usr/bin/env python3
"""Phase 5 smoke test — live Sentinel pass through full context/gate/receipt path.

Proves:
  1. SSM agent queries sentinel.db for entity history
  2. RAG agent queries FGIP FTS5 for document context
  3. Graph agent queries FGIP for entity relationships
  4. Live Qwen Sentinel on port 8085 produces a verdict
  5. Gate agent controls execution (auto=True)
  6. Receipt written with all fields

Usage:
    python3 tools/smoke_live_sentinel_context_path.py
    python3 tools/smoke_live_sentinel_context_path.py --port 8085

Requires:
    - llama-server running on --port (default 8085) with Sentinel model
    - ~/tools/sentinel/sentinel.db
    - ~/fgip-engine/fgip.db
"""
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

# Ensure cell-runtime src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.base import AgentRegistry
from cell.agents.rag_agent import RAGLookupAgent
from cell.agents.graph_agent import GraphLookupAgent, GraphNeighborsAgent, GraphStatsAgent
from cell.agents.ssm_agent import SSMGetStateAgent
from cell.agents.policy_agent import GateDecideAgent


def _call_sentinel(port: int, alert_text: str, system_prompt: str) -> dict:
    """Call live Sentinel backend via OpenAI-compatible chat API."""
    import urllib.request

    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": alert_text},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    usage = data.get("usage", {})
    timings = data.get("timings", {})

    return {
        "content": content,
        "model": data.get("model", ""),
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "predicted_per_second": timings.get("predicted_per_second", 0),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5: Live Sentinel Context Path")
    parser.add_argument("--port", type=int, default=8085, help="Sentinel llama-server port")
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 60)
    print("  Phase 5: Live Sentinel Context Path Smoke Test")
    print("=" * 60)

    # Build agent registry
    registry = AgentRegistry()
    for cls in [RAGLookupAgent, GraphLookupAgent, GraphNeighborsAgent,
                GraphStatsAgent, SSMGetStateAgent, GateDecideAgent]:
        registry.register(cls())

    # Synthetic alert
    alert_text = (
        "ALERT: Outbound connection from svchost.exe to 185.220.101.1:443 "
        "detected on host WIN-DC01. Process hash: a3f2b7... "
        "Connection to known Tor exit node. Possible C2 channel."
    )
    entity_id = "185.220.101.1"
    alert_id = f"SMOKE-{int(time.time())}"

    print(f"\n  Alert ID: {alert_id}")
    print(f"  Entity:   {entity_id}")
    print(f"  Alert:    {alert_text[:80]}...")

    results = {}
    errors = []

    # --- Step 1: SSM state ---
    print("\n  [1/6] SSM state query...", end=" ", flush=True)
    ssm_result = registry.run("ssm_get_state", {
        "entity_id": entity_id,
        "limit": 5,
    })
    if ssm_result.ok:
        results["ssm"] = {
            "found": ssm_result.output.get("found", False),
            "event_count": ssm_result.output.get("event_count", 0),
            "trend": ssm_result.output.get("trend", "unknown"),
            "wall_time_s": ssm_result.receipt["wall_time_s"],
        }
        print(f"OK (found={results['ssm']['found']}, "
              f"events={results['ssm']['event_count']}, "
              f"{results['ssm']['wall_time_s']}s)")
    else:
        errors.append(f"SSM: {ssm_result.error}")
        results["ssm"] = {"error": ssm_result.error}
        print(f"ERROR: {ssm_result.error}")

    # --- Step 2: RAG context ---
    print("  [2/6] RAG document search...", end=" ", flush=True)
    rag_result = registry.run("rag_lookup", {
        "query": "security",
        "scope": "claims",
        "limit": 5,
    })
    if rag_result.ok:
        results["rag"] = {
            "source": rag_result.output.get("source", ""),
            "count": rag_result.output.get("count", 0),
            "wall_time_s": rag_result.receipt["wall_time_s"],
        }
        print(f"OK (source={results['rag']['source']}, "
              f"hits={results['rag']['count']}, "
              f"{results['rag']['wall_time_s']}s)")
    else:
        errors.append(f"RAG: {rag_result.error}")
        results["rag"] = {"error": rag_result.error}
        print(f"ERROR: {rag_result.error}")

    # --- Step 3: Graph context ---
    print("  [3/6] Graph stats...", end=" ", flush=True)
    graph_result = registry.run("graph_stats", {})
    if graph_result.ok:
        results["graph"] = {
            "nodes": graph_result.output.get("nodes", 0),
            "edges": graph_result.output.get("edges", 0),
            "claims": graph_result.output.get("claims", 0),
            "wall_time_s": graph_result.receipt["wall_time_s"],
        }
        print(f"OK (nodes={results['graph']['nodes']}, "
              f"edges={results['graph']['edges']}, "
              f"claims={results['graph']['claims']}, "
              f"{results['graph']['wall_time_s']}s)")
    else:
        errors.append(f"Graph: {graph_result.error}")
        results["graph"] = {"error": graph_result.error}
        print(f"ERROR: {graph_result.error}")

    # --- Step 4: Live Sentinel call ---
    print("  [4/6] Live Sentinel call...", end=" ", flush=True)
    system_prompt = (
        "You are a security triage agent. Analyze the following alert. "
        "Respond with:\n"
        "SEVERITY: <critical|high|medium|low|benign>\n"
        "SUMMARY: <one-line summary>\n"
        "ACTIONS: <comma-separated recommended actions>\n"
        "TOOL: <tool call if needed, or NONE>"
    )

    # Build context pack for the prompt
    context_lines = [f"Alert: {alert_text}"]
    if results.get("ssm", {}).get("found"):
        context_lines.append(f"SSM: Entity has {results['ssm']['event_count']} prior events, trend={results['ssm']['trend']}")
    if results.get("rag", {}).get("count", 0) > 0:
        context_lines.append(f"RAG: {results['rag']['count']} related documents found in evidence base")
    if results.get("graph", {}).get("nodes", 0) > 0:
        context_lines.append(f"Graph: {results['graph']['nodes']} entities, {results['graph']['edges']} relationships in knowledge graph")

    augmented_alert = "\n".join(context_lines)

    try:
        sentinel_t0 = time.time()
        sentinel_resp = _call_sentinel(args.port, augmented_alert, system_prompt)
        sentinel_wall = round(time.time() - sentinel_t0, 3)

        results["sentinel"] = {
            "model": sentinel_resp["model"],
            "output": sentinel_resp["content"],
            "completion_tokens": sentinel_resp["completion_tokens"],
            "prompt_tokens": sentinel_resp["prompt_tokens"],
            "tok_s": round(sentinel_resp["predicted_per_second"], 1),
            "wall_time_s": sentinel_wall,
        }
        print(f"OK ({sentinel_resp['completion_tokens']} tokens, "
              f"{results['sentinel']['tok_s']} tok/s, "
              f"{sentinel_wall}s)")
        print(f"\n  --- Sentinel Output ---")
        for line in sentinel_resp["content"].strip().split("\n"):
            print(f"  {line}")
        print(f"  --- End ---\n")
    except Exception as e:
        errors.append(f"Sentinel: {e}")
        results["sentinel"] = {"error": str(e)}
        print(f"ERROR: {e}")

    # --- Step 5: Gate check ---
    print("  [5/6] Gate check (auto=True)...", end=" ", flush=True)
    # Determine proposed action from sentinel output
    proposed_action = "log_alert"
    sentinel_output = results.get("sentinel", {}).get("output", "")
    if "critical" in sentinel_output.lower() or "high" in sentinel_output.lower():
        proposed_action = "block_ip"
    elif "medium" in sentinel_output.lower():
        proposed_action = "investigate"

    gate_result = registry.run("gate_decide", {
        "action": proposed_action,
        "detail": f"Sentinel recommends {proposed_action} for {entity_id}",
        "auto": True,
    })
    if gate_result.ok:
        results["gate"] = {
            "action": proposed_action,
            "allowed": gate_result.output["allowed"],
            "wall_time_s": gate_result.receipt["wall_time_s"],
        }
        print(f"OK (action={proposed_action}, "
              f"allowed={gate_result.output['allowed']}, "
              f"{gate_result.receipt['wall_time_s']}s)")
    else:
        errors.append(f"Gate: {gate_result.error}")
        results["gate"] = {"error": gate_result.error}
        print(f"ERROR: {gate_result.error}")

    # --- Step 6: Write receipt ---
    print("  [6/6] Writing receipt...", end=" ", flush=True)
    wall_time = round(time.time() - t_start, 3)
    cpu_time = round(time.process_time() - cpu_start, 3)

    receipt = {
        "receipt_id": f"phase5_live_sentinel_context_path_{time.strftime('%Y%m%dT%H%M%SZ')}",
        "title": "Phase 5: Live Sentinel Context Path",
        "date": time.strftime("%Y-%m-%d"),
        "status": "PASS" if not errors else "PARTIAL",
        "alert_id": alert_id,
        "entity_id": entity_id,
        "alert_text": alert_text,
        "context_pack": {
            "ssm": results.get("ssm", {}),
            "rag": results.get("rag", {}),
            "graph": results.get("graph", {}),
        },
        "sentinel": {
            "model_id": results.get("sentinel", {}).get("model", ""),
            "output": results.get("sentinel", {}).get("output", ""),
            "completion_tokens": results.get("sentinel", {}).get("completion_tokens", 0),
            "tok_s": results.get("sentinel", {}).get("tok_s", 0),
            "wall_time_s": results.get("sentinel", {}).get("wall_time_s", 0),
        },
        "gate": {
            "action": results.get("gate", {}).get("action", ""),
            "allowed": results.get("gate", {}).get("allowed", False),
        },
        "errors": errors,
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

    print(f"OK ({receipt_path})")

    # Summary
    print(f"\n{'=' * 60}")
    status = "PASS" if not errors else "PARTIAL"
    print(f"  Result: {status}")
    print(f"  Wall time: {wall_time}s")
    print(f"  Steps completed: {6 - len(errors)}/6")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors:
            print(f"    - {e}")
    print(f"  Receipt: {receipt_path}")
    print(f"{'=' * 60}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
