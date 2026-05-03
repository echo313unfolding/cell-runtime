#!/usr/bin/env python3
"""Sweep n_gpu_layers for a GGUF model on llama-server.

Tests different GPU layer counts to find the optimal split point for
CPU/GPU offload on T2000 (4GB VRAM) + 64GB RAM.

For each split point:
  - Start llama-server with --n-gpu-layers N
  - Measure load time, peak VRAM, tok/s
  - Run a simple prompt for sanity check
  - Kill server, wait for GPU clear

Usage:
    python3 tools/sweep_gpu_layers.py ~/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
    python3 tools/sweep_gpu_layers.py ~/models/model.gguf --layers 0,8,16,24,99
"""
import argparse
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time


LLAMA_SERVER = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
LIB_DIR = os.path.expanduser("~/llama.cpp/build/bin")

TEST_PROMPTS = [
    {"role": "system", "content": "You are a helpful coding assistant. Be concise."},
    {"role": "user", "content": "Write a Python function that checks if a number is prime. Keep it short."},
]


def get_vram_mb():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(result.stdout.strip())
    except Exception:
        return -1


def start_server(gguf_path, n_gpu_layers, port=8090):
    """Start llama-server and wait for it to be ready."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB_DIR + ":" + env.get("LD_LIBRARY_PATH", "")

    log_path = os.path.expanduser(f"~/receipts/cell/sweep_layer_{n_gpu_layers}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    proc = subprocess.Popen(
        [LLAMA_SERVER,
         "--model", gguf_path,
         "--port", str(port),
         "--ctx-size", "2048",
         "--n-gpu-layers", str(n_gpu_layers),
         "--host", "0.0.0.0"],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )

    # Wait for server to be ready
    import urllib.request
    for i in range(120):
        time.sleep(1)
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            return None

    # Timeout
    stop_server(proc)
    return None


def stop_server(proc):
    """Stop llama-server."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    # Also kill orphans
    try:
        subprocess.run(["pkill", "-f", "llama-server"], timeout=5,
                       capture_output=True)
    except Exception:
        pass
    time.sleep(2)


def run_prompt(port=8090):
    """Send test prompt and measure tok/s."""
    import urllib.request
    payload = json.dumps({
        "messages": TEST_PROMPTS,
        "max_tokens": 128,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - t0

    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    tok_s = completion_tokens / elapsed if elapsed > 0 else 0

    return {
        "content": content[:200],
        "completion_tokens": completion_tokens,
        "elapsed_s": round(elapsed, 2),
        "tok_s": round(tok_s, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep n_gpu_layers")
    parser.add_argument("gguf_path", help="GGUF model path")
    parser.add_argument("--layers", type=str,
                        default="0,4,8,12,16,20,24,99",
                        help="Comma-separated layer counts to test")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    gguf_path = os.path.expanduser(args.gguf_path)
    if not os.path.exists(gguf_path):
        print(f"GGUF not found: {gguf_path}")
        sys.exit(1)

    layer_counts = [int(x) for x in args.layers.split(",")]
    model_name = os.path.basename(gguf_path)
    model_size_gb = os.path.getsize(gguf_path) / 1024**3

    t_start = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"Model: {model_name} ({model_size_gb:.2f} GB)")
    print(f"Sweep: {layer_counts}")
    print(f"Port: {args.port}")
    print()

    results = []

    for ngl in layer_counts:
        print(f"=== n_gpu_layers={ngl} ===")

        # Clear GPU
        stop_server(None)
        time.sleep(1)
        vram_before = get_vram_mb()

        # Start server
        t_load = time.time()
        proc = start_server(gguf_path, ngl, port=args.port)
        load_time = round(time.time() - t_load, 1)

        if proc is None:
            print(f"  FAIL: server did not start")
            results.append({
                "n_gpu_layers": ngl,
                "status": "FAIL",
                "reason": "server did not start",
            })
            print()
            continue

        vram_after = get_vram_mb()
        vram_used = vram_after - vram_before

        print(f"  Loaded in {load_time}s")
        print(f"  VRAM: {vram_after} MB (delta: +{vram_used} MB)")

        # Run prompt
        try:
            gen = run_prompt(port=args.port)
            print(f"  tok/s: {gen['tok_s']}")
            print(f"  Tokens: {gen['completion_tokens']} in {gen['elapsed_s']}s")
            print(f"  Output: {gen['content'][:100]}...")
        except Exception as e:
            gen = {"tok_s": 0, "error": str(e)}
            print(f"  Generation FAILED: {e}")

        result = {
            "n_gpu_layers": ngl,
            "status": "PASS",
            "load_time_s": load_time,
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "vram_delta_mb": vram_used,
            **gen,
        }
        results.append(result)

        # Stop server
        stop_server(proc)
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'ngl':>4} | {'VRAM MB':>8} | {'tok/s':>7} | {'load_s':>6} | status")
    print("-" * 50)
    for r in results:
        if r["status"] == "FAIL":
            print(f"{r['n_gpu_layers']:>4} | {'---':>8} | {'---':>7} | {'---':>6} | FAIL")
        else:
            print(f"{r['n_gpu_layers']:>4} | {r.get('vram_delta_mb', 0):>8} | "
                  f"{r.get('tok_s', 0):>7.1f} | {r.get('load_time_s', 0):>6.1f} | PASS")

    # Write receipt
    wall_time = round(time.time() - t_start, 1)
    receipt = {
        "receipt_id": f"gpu_layer_sweep_{time.strftime('%Y%m%dT%H%M%SZ')}",
        "title": f"GPU Layer Sweep: {model_name}",
        "model": model_name,
        "model_size_gb": round(model_size_gb, 2),
        "layer_counts_tested": layer_counts,
        "results": results,
        "cost": {
            "wall_time_s": wall_time,
            "cpu_time_s": round(time.process_time(), 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    receipt_path = os.path.expanduser(
        f"~/receipts/gpu_layer_sweep_{time.strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {receipt_path}")


if __name__ == "__main__":
    main()
