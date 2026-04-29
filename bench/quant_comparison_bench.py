#!/usr/bin/env python3
"""Head-to-head quant comparison: Q4_K_M vs HXQ_AFFINE_6 vs Q8_0.

Same model, same query, same hardware. Measures:
- tok/s (decode)
- energy (joules per token)
- quality (response text)

Usage:
    python3 quant_comparison_bench.py --json-out ~/receipts/quant_comparison.json
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

MODELS = [
    ("Q4_K_M", "/home/voidstr3m33/models/qwen2.5-sentinel-q4_k_m.gguf"),
    ("HXQ_AFFINE_6", "/home/voidstr3m33/models/qwen2.5-sentinel-hxq-affine6.gguf"),
    ("Q8_0", "/home/voidstr3m33/models/qwen2.5-sentinel-q8_0.gguf"),
]

SYSTEM_PROMPT = "You are a security triage agent. Classify severity and recommend actions."
USER_PROMPT = "Alert: 500 failed SSH logins from 203.0.113.42 targeting root in 2 minutes. Classify severity and recommend immediate action."

LLAMA_SERVER = "/home/voidstr3m33/llama.cpp/build/bin/llama-server"
PORT = 8083


def sample_power(samples, stop_event, interval=0.1):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                timeout=2
            ).decode().strip()
            samples.append((time.time(), float(out)))
        except (subprocess.TimeoutExpired, ValueError):
            pass
        time.sleep(interval)


def wait_for_server(port, timeout=30):
    for _ in range(timeout * 2):
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def run_query(port):
    payload = json.dumps({
        "model": "test",
        "temperature": 0,
        "max_tokens": 200,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    wall = time.time() - t0
    return data, wall


def bench_one_model(quant_name, model_path):
    print(f"\n{'='*60}")
    print(f"  {quant_name}: {os.path.basename(model_path)}")
    print(f"{'='*60}")

    # Start server
    proc = subprocess.Popen(
        [LLAMA_SERVER, "-m", model_path, "-ngl", "99", "-c", "2048", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

    if not wait_for_server(PORT):
        proc.kill()
        print("  FAILED: server did not start")
        return None

    # Warm up
    try:
        run_query(PORT)
    except Exception:
        pass

    # Power sampling
    samples = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_power, args=(samples, stop), daemon=True)
    sampler.start()
    time.sleep(0.3)
    idle_count = len(samples)

    # Actual timed query
    try:
        data, wall_s = run_query(PORT)
    except Exception as e:
        stop.set()
        proc.kill()
        print(f"  FAILED: {e}")
        return None

    time.sleep(0.2)
    stop.set()
    sampler.join(timeout=2)

    # Kill server
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    # Parse server stderr for timing
    stderr_text = proc.stderr.read().decode(errors='replace')
    tok_s = None
    for line in stderr_text.split('\n'):
        if 'eval time' in line and 'prompt' not in line:
            # "eval time =    2023.67 ms /    59 tokens (   34.30 ms per token,    29.15 tokens per second)"
            if 'tokens per second' in line:
                tok_s = float(line.split('tokens per second')[0].split(',')[-1].strip())

    # Energy calc
    active_samples = samples[idle_count:] if idle_count < len(samples) else samples
    idle_watts = sum(w for _, w in samples[:idle_count]) / max(idle_count, 1) if idle_count > 0 else 0
    avg_watts = sum(w for _, w in active_samples) / max(len(active_samples), 1)
    peak_watts = max((w for _, w in active_samples), default=0)

    tokens = data.get("usage", {}).get("completion_tokens", 0)
    content = data["choices"][0]["message"]["content"]
    joules = avg_watts * wall_s
    j_per_tok = joules / max(tokens, 1)

    result = {
        "quant": quant_name,
        "file": os.path.basename(model_path),
        "size_gb": round(os.path.getsize(model_path) / 1e9, 2),
        "tokens": tokens,
        "tok_s": tok_s,
        "wall_s": round(wall_s, 2),
        "watts_avg": round(avg_watts, 2),
        "watts_peak": round(peak_watts, 2),
        "watts_idle": round(idle_watts, 2),
        "joules_total": round(joules, 2),
        "joules_per_token": round(j_per_tok, 4),
        "response": content
    }

    print(f"  Tokens: {tokens} @ {tok_s} tok/s")
    print(f"  Energy: {joules:.1f}J total, {j_per_tok:.3f} J/tok")
    print(f"  Power:  {avg_watts:.1f}W avg, {peak_watts:.1f}W peak")
    print(f"  Response: {content[:120]}...")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default="/home/voidstr3m33/receipts/quant_energy_comparison.json")
    args = parser.parse_args()

    # Kill any existing server on our port
    subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)
    time.sleep(1)

    results = []
    for quant_name, model_path in MODELS:
        r = bench_one_model(quant_name, model_path)
        if r:
            results.append(r)
        # Ensure port is free
        subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)
        time.sleep(2)

    if results:
        print(f"\n{'='*60}")
        print(f"  COMPARISON TABLE")
        print(f"{'='*60}")
        print(f"  {'Quant':<15} {'Size':>6} {'tok/s':>7} {'J/tok':>7} {'W avg':>6}")
        print(f"  {'-'*15} {'-'*6} {'-'*7} {'-'*7} {'-'*6}")
        for r in results:
            print(f"  {r['quant']:<15} {r['size_gb']:>5.1f}G {r['tok_s'] or 0:>6.1f} {r['joules_per_token']:>7.3f} {r['watts_avg']:>5.1f}")

        out = {
            "benchmark": "quant_energy_comparison",
            "date": time.strftime("%Y-%m-%d"),
            "model_base": "Qwen2.5-Sentinel-Merged-3B",
            "hardware": "Quadro T2000 4GB",
            "task": "Security triage (SSH brute force alert)",
            "results": results
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Written to: {args.json_out}")


if __name__ == "__main__":
    main()
