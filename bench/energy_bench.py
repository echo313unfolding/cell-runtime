#!/usr/bin/env python3
"""Energy cost benchmarking for Capsule edge runtime.

Measures GPU power draw during inference queries and reports:
- Joules per query (energy)
- Watts average during generation
- Watt-hours per 1000 queries (projected edge cost)

Usage:
    python3 energy_bench.py --gateway http://localhost:8800 --queries 3
    python3 energy_bench.py --gateway http://localhost:8800 --all-lanes
"""

import argparse
import json
import subprocess
import threading
import time
import urllib.request

def sample_power(samples: list, stop_event: threading.Event, interval_s=0.1):
    """Background thread: samples GPU power draw via nvidia-smi."""
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                timeout=2
            ).decode().strip()
            watts = float(out)
            samples.append((time.time(), watts))
        except (subprocess.TimeoutExpired, ValueError):
            pass
        time.sleep(interval_s)


def query_gateway(url, model, prompt):
    """Send a chat completion request, return (response_json, wall_time_s)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    wall = time.time() - t0
    return data, wall


def measure_query(url, model, prompt):
    """Run one query with power sampling, return energy metrics."""
    samples = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_power, args=(samples, stop), daemon=True)
    sampler.start()

    # Small delay to get idle baseline
    time.sleep(0.3)
    idle_samples = len(samples)

    data, wall_s = query_gateway(url, model, prompt)

    # Let power settle
    time.sleep(0.2)
    stop.set()
    sampler.join(timeout=2)

    # Compute energy
    if len(samples) < 2:
        return None

    # Idle power (pre-query samples)
    idle_watts = sum(w for _, w in samples[:idle_samples]) / max(idle_samples, 1) if idle_samples > 0 else 0

    # Active power (during query)
    active_samples = samples[idle_samples:]
    if not active_samples:
        active_samples = samples

    avg_watts = sum(w for _, w in active_samples) / len(active_samples)
    peak_watts = max(w for _, w in active_samples)

    # Energy = average power × time
    joules = avg_watts * wall_s
    delta_joules = (avg_watts - idle_watts) * wall_s  # incremental over idle

    capsule = data.get("capsule", {})
    usage = data.get("usage", {})

    return {
        "model": data.get("model"),
        "intent": capsule.get("intent"),
        "completion_tokens": usage.get("completion_tokens", 0),
        "tok_s": capsule.get("tok_s"),
        "wall_s": round(wall_s, 2),
        "swapped": capsule.get("swapped"),
        "swap_time_s": capsule.get("swap_time_s"),
        "energy": {
            "joules_total": round(joules, 2),
            "joules_incremental": round(delta_joules, 2),
            "watts_avg": round(avg_watts, 2),
            "watts_peak": round(peak_watts, 2),
            "watts_idle": round(idle_watts, 2),
            "wh_per_1000_queries": round((joules / 3600) * 1000, 3),
            "joules_per_token": round(joules / max(usage.get("completion_tokens", 1), 1), 4),
            "samples_collected": len(active_samples)
        }
    }


LANE_QUERIES = {
    "reasoning": ("capsule-auto", "What are the first 8 prime numbers? List them."),
    "coding": ("capsule-auto", "Write a Python function that reverses a linked list."),
    "sentinel": ("capsule-sentinel", "Triage: port scan detected from 198.51.100.7 hitting ports 22,80,443,8080,3306 in 4 seconds"),
}


def main():
    parser = argparse.ArgumentParser(description="Capsule energy benchmark")
    parser.add_argument("--gateway", default="http://localhost:8800")
    parser.add_argument("--all-lanes", action="store_true", help="Test all 3 lanes")
    parser.add_argument("--model", default="capsule-auto")
    parser.add_argument("--prompt", default="What is 2+2?")
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--json-out", help="Write results to JSON file")
    args = parser.parse_args()

    results = []

    if args.all_lanes:
        for lane_name, (model, prompt) in LANE_QUERIES.items():
            print(f"\n{'='*60}")
            print(f"Lane: {lane_name}")
            print(f"{'='*60}")
            r = measure_query(args.gateway, model, prompt)
            if r:
                r["lane"] = lane_name
                results.append(r)
                e = r["energy"]
                print(f"  Model:  {r['model']}")
                print(f"  Tokens: {r['completion_tokens']} @ {r['tok_s']} tok/s")
                print(f"  Wall:   {r['wall_s']}s (swap: {r['swap_time_s']}s)")
                print(f"  Power:  {e['watts_avg']}W avg, {e['watts_peak']}W peak, {e['watts_idle']}W idle")
                print(f"  Energy: {e['joules_total']}J total, {e['joules_incremental']}J incremental")
                print(f"  Per-token: {e['joules_per_token']} J/tok")
                print(f"  Projected: {e['wh_per_1000_queries']} Wh/1000 queries")
    else:
        for i in range(args.queries):
            r = measure_query(args.gateway, args.model, args.prompt)
            if r:
                results.append(r)
                e = r["energy"]
                print(f"Query {i+1}: {r['model']} | {r['completion_tokens']} tok @ {e['watts_avg']}W = {e['joules_total']}J ({e['joules_per_token']} J/tok)")

    if results:
        total_j = sum(r["energy"]["joules_total"] for r in results)
        total_tok = sum(r["completion_tokens"] for r in results)
        print(f"\n{'='*60}")
        print(f"TOTAL: {total_j:.1f}J for {total_tok} tokens across {len(results)} queries")
        print(f"AVERAGE: {total_j/len(results):.1f}J per query, {total_j/max(total_tok,1):.4f} J/tok")
        print(f"{'='*60}")

    if args.json_out and results:
        out = {
            "benchmark": "capsule_energy",
            "date": time.strftime("%Y-%m-%d"),
            "gateway": args.gateway,
            "results": results,
            "totals": {
                "queries": len(results),
                "total_joules": round(total_j, 2),
                "total_tokens": total_tok,
                "avg_joules_per_query": round(total_j / len(results), 2),
                "avg_joules_per_token": round(total_j / max(total_tok, 1), 4)
            }
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to: {args.json_out}")


if __name__ == "__main__":
    main()
