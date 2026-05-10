#!/usr/bin/env python3
"""Tiny behavior eval: compare Q4_K_M, Q5_K_M, HXQ_AFFINE_6 on fixed prompts.

Tests whether HXQ preserves model behavior (not speed).
25 prompts across 5 categories. Records exact outputs + pass/fail.

Usage:
    python3 tools/tiny_behavior_eval.py
"""
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import urllib.request

LLAMA_SERVER = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
LIB_DIR = os.path.expanduser("~/llama.cpp/build/bin")
PORT = 8091

MODELS = [
    {
        "name": "Q4_K_M",
        "path": os.path.expanduser("~/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"),
        "ngl": 20,
        "bpw": 4.5,
    },
    {
        "name": "Q5_K_M",
        "path": os.path.expanduser("~/models/Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf"),
        "ngl": 20,
        "bpw": 5.5,
    },
    {
        "name": "HXQ_AFFINE_6",
        "path": os.path.expanduser("~/models/qwen2.5-coder-7b-hxq-affine6-native.gguf"),
        "ngl": 17,
        "bpw": 6.27,
    },
]

PROMPTS = [
    # === Coding (5) ===
    {
        "category": "coding",
        "id": "c1",
        "prompt": "Write a Python function `is_prime(n)` that returns True if n is prime. No explanation, just code.",
        "check": lambda s: "def is_prime" in s and "return" in s,
        "check_desc": "defines is_prime with return",
    },
    {
        "category": "coding",
        "id": "c2",
        "prompt": "This code has a bug. Fix it:\n```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) / 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return -1\n```\nReturn only the fixed code.",
        "check": lambda s: "//" in s or "mid = (lo + hi) // 2" in s or "int(" in s,
        "check_desc": "fixes integer division",
    },
    {
        "category": "coding",
        "id": "c3",
        "prompt": "Write a Python function that takes a list and returns it sorted using merge sort. Code only.",
        "check": lambda s: "def " in s and ("merge" in s.lower()) and "return" in s,
        "check_desc": "defines merge sort function",
    },
    {
        "category": "coding",
        "id": "c4",
        "prompt": "Write a Python decorator `@cache` that memoizes function results. Code only.",
        "check": lambda s: "def cache" in s or "def wrapper" in s or "functools" in s,
        "check_desc": "defines cache/memoize decorator",
    },
    {
        "category": "coding",
        "id": "c5",
        "prompt": "Write a Python function `flatten(lst)` that recursively flattens nested lists. Example: flatten([1,[2,[3]],4]) -> [1,2,3,4]. Code only.",
        "check": lambda s: "def flatten" in s and ("isinstance" in s or "type(" in s or "extend" in s or "yield" in s),
        "check_desc": "defines recursive flatten",
    },
    # === Math (5) ===
    {
        "category": "math",
        "id": "m1",
        "prompt": "What is 17 * 23? Give just the number.",
        "check": lambda s: "391" in s,
        "check_desc": "answer is 391",
    },
    {
        "category": "math",
        "id": "m2",
        "prompt": "Solve for x: 2x + 5 = 17. Give just the value of x.",
        "check": lambda s: "6" in s,
        "check_desc": "x = 6",
    },
    {
        "category": "math",
        "id": "m3",
        "prompt": "What is the derivative of f(x) = x^3 + 2x? Answer briefly.",
        "check": lambda s: "3x" in s.lower().replace(" ", "") or "3x²" in s or "3x^2" in s,
        "check_desc": "derivative contains 3x^2",
    },
    {
        "category": "math",
        "id": "m4",
        "prompt": "What is the probability of rolling two sixes with two fair dice? Give the fraction.",
        "check": lambda s: "1/36" in s.replace(" ", ""),
        "check_desc": "probability is 1/36",
    },
    {
        "category": "math",
        "id": "m5",
        "prompt": "What is the sum of the first 100 natural numbers (1+2+...+100)? Give just the number.",
        "check": lambda s: "5050" in s,
        "check_desc": "sum is 5050",
    },
    # === ML/Tensor (5) ===
    {
        "category": "ml_tensor",
        "id": "t1",
        "prompt": "If A has shape (32, 768) and B has shape (768, 3072), what is the shape of A @ B? Answer: (rows, cols)",
        "check": lambda s: "32" in s and "3072" in s,
        "check_desc": "shape (32, 3072)",
    },
    {
        "category": "ml_tensor",
        "id": "t2",
        "prompt": "What does the softmax function do? One sentence.",
        "check": lambda s: "probab" in s.lower() or "sum" in s.lower() and ("1" in s or "one" in s.lower()),
        "check_desc": "mentions probabilities or sum to 1",
    },
    {
        "category": "ml_tensor",
        "id": "t3",
        "prompt": "What shape does Conv2d(3, 64, kernel_size=3, padding=1) produce given input shape (1, 3, 224, 224)? Just the output shape.",
        "check": lambda s: "64" in s and "224" in s,
        "check_desc": "output (1, 64, 224, 224)",
    },
    {
        "category": "ml_tensor",
        "id": "t4",
        "prompt": "Why is dropout used during training but not inference? One sentence.",
        "check": lambda s: "overfit" in s.lower() or "regulariz" in s.lower() or "generali" in s.lower(),
        "check_desc": "mentions overfitting/regularization",
    },
    {
        "category": "ml_tensor",
        "id": "t5",
        "prompt": "What is the purpose of layer normalization? One sentence.",
        "check": lambda s: "normaliz" in s.lower() or "stabil" in s.lower() or "mean" in s.lower(),
        "check_desc": "mentions normalization/stability",
    },
    # === Short Science (5) ===
    {
        "category": "science",
        "id": "s1",
        "prompt": "Why is the sky blue? One sentence.",
        "check": lambda s: "scatter" in s.lower() or "rayleigh" in s.lower(),
        "check_desc": "mentions scattering/Rayleigh",
    },
    {
        "category": "science",
        "id": "s2",
        "prompt": "What causes ocean tides? One sentence.",
        "check": lambda s: "moon" in s.lower() or "gravit" in s.lower(),
        "check_desc": "mentions moon/gravity",
    },
    {
        "category": "science",
        "id": "s3",
        "prompt": "What is the speed of light in vacuum in m/s? Just the number.",
        "check": lambda s: "3" in s and ("10" in s or "×" in s or "e8" in s or "00000000" in s or "299" in s),
        "check_desc": "approximately 3e8 or 299792458",
    },
    {
        "category": "science",
        "id": "s4",
        "prompt": "Why does ice float on water? One sentence.",
        "check": lambda s: "dens" in s.lower() or "expand" in s.lower() or "lighter" in s.lower() or "less dense" in s.lower(),
        "check_desc": "mentions density/expansion",
    },
    {
        "category": "science",
        "id": "s5",
        "prompt": "What is entropy in thermodynamics? One sentence.",
        "check": lambda s: "disorder" in s.lower() or "energy" in s.lower() or "microstate" in s.lower() or "unavailab" in s.lower(),
        "check_desc": "mentions disorder/energy/microstates",
    },
    # === Tool/Reasoning (5) ===
    {
        "category": "reasoning",
        "id": "r1",
        "prompt": 'What HTTP status code means "Not Found"? Just the number.',
        "check": lambda s: "404" in s,
        "check_desc": "404",
    },
    {
        "category": "reasoning",
        "id": "r2",
        "prompt": 'Given this JSON:\n```json\n{"name": "test", "cost": {"wall_time_s": 12.5, "peak_memory_mb": 245}}\n```\nWhat is the wall_time_s value? Just the number.',
        "check": lambda s: "12.5" in s,
        "check_desc": "12.5",
    },
    {
        "category": "reasoning",
        "id": "r3",
        "prompt": "Convert Unix timestamp 1704067200 to a human-readable date (UTC). Just the date.",
        "check": lambda s: "2024" in s and ("jan" in s.lower() or "01" in s or "1" in s),
        "check_desc": "2024-01-01 or January 1, 2024",
    },
    {
        "category": "reasoning",
        "id": "r4",
        "prompt": 'In YAML:\n```yaml\nserver:\n  host: 0.0.0.0\n  port: 8080\n  debug: true\n```\nWhat port does the server use? Just the number.',
        "check": lambda s: "8080" in s,
        "check_desc": "8080",
    },
    {
        "category": "reasoning",
        "id": "r5",
        "prompt": 'This log line: `[ERROR] 2024-03-15 14:23:01 ConnectionRefusedError: port 5432`\nWhat type of error is it? One phrase.',
        "check": lambda s: "connection" in s.lower() or "refused" in s.lower(),
        "check_desc": "ConnectionRefused",
    },
]


def start_server(model_path, ngl):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB_DIR + ":" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.Popen(
        [LLAMA_SERVER,
         "--model", model_path,
         "--port", str(PORT),
         "--ctx-size", "1024",
         "--n-gpu-layers", str(ngl),
         "--host", "0.0.0.0",
         "--n-predict", "128"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        preexec_fn=os.setsid,
    )
    for i in range(120):
        time.sleep(1)
        try:
            req = urllib.request.Request(f"http://localhost:{PORT}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            return None
    stop_server(proc)
    return None


def stop_server(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    try:
        subprocess.run(["pkill", "-f", f"llama-server.*{PORT}"],
                       timeout=5, capture_output=True)
    except Exception:
        pass
    time.sleep(2)


def run_prompt(prompt_text, max_tokens=128):
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": "Be concise and precise. Answer exactly what is asked."},
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - t0
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    tok_s = completion_tokens / elapsed if elapsed > 0 else 0
    return content, completion_tokens, round(elapsed, 2), round(tok_s, 2)


def main():
    t_start = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    all_results = {}

    for model in MODELS:
        name = model["name"]
        print(f"\n{'='*60}")
        print(f"MODEL: {name} (ngl={model['ngl']}, bpw={model['bpw']})")
        print(f"{'='*60}")

        if not os.path.exists(model["path"]):
            print(f"  SKIP: {model['path']} not found")
            all_results[name] = {"status": "SKIP", "reason": "file not found"}
            continue

        stop_server(None)
        proc = start_server(model["path"], model["ngl"])
        if proc is None:
            print(f"  FAIL: server did not start")
            all_results[name] = {"status": "FAIL", "reason": "server did not start"}
            continue

        model_results = []
        pass_count = 0
        total_count = 0

        for p in PROMPTS:
            total_count += 1
            pid = p["id"]
            cat = p["category"]
            print(f"  [{pid}] {cat}: ", end="", flush=True)
            try:
                content, tokens, elapsed, tok_s = run_prompt(p["prompt"])
                passed = p["check"](content)
                if passed:
                    pass_count += 1
                status = "PASS" if passed else "FAIL"
                print(f"{status} ({tok_s} tok/s, {tokens} tok)")
                model_results.append({
                    "id": pid,
                    "category": cat,
                    "status": status,
                    "check": p["check_desc"],
                    "output": content[:500],
                    "completion_tokens": tokens,
                    "elapsed_s": elapsed,
                    "tok_s": tok_s,
                })
            except Exception as e:
                print(f"ERROR: {e}")
                model_results.append({
                    "id": pid,
                    "category": cat,
                    "status": "ERROR",
                    "error": str(e),
                })

        stop_server(proc)

        score = f"{pass_count}/{total_count}"
        print(f"\n  Score: {score}")
        all_results[name] = {
            "status": "OK",
            "ngl": model["ngl"],
            "bpw": model["bpw"],
            "score": score,
            "pass_count": pass_count,
            "total_count": total_count,
            "results": model_results,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Score':<10} {'Pass%':<10}")
    print("-" * 40)
    for name, data in all_results.items():
        if data.get("status") == "OK":
            pct = round(100 * data["pass_count"] / data["total_count"], 1)
            print(f"{name:<20} {data['score']:<10} {pct}%")
        else:
            print(f"{name:<20} {data.get('status','?'):<10}")

    # Category breakdown
    print(f"\n{'='*60}")
    print("PER-CATEGORY BREAKDOWN")
    print(f"{'='*60}")
    categories = sorted(set(p["category"] for p in PROMPTS))
    header = f"{'Category':<15}"
    for name in all_results:
        if all_results[name].get("status") == "OK":
            header += f" {name:<16}"
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = f"{cat:<15}"
        for name, data in all_results.items():
            if data.get("status") != "OK":
                continue
            cat_results = [r for r in data["results"] if r["category"] == cat]
            cat_pass = sum(1 for r in cat_results if r["status"] == "PASS")
            row += f" {cat_pass}/{len(cat_results):<14}"
        print(row)

    # Write receipt
    wall_time = round(time.time() - t_start, 1)
    receipt = {
        "receipt_id": f"hxq_tiny_behavior_eval_{time.strftime('%Y%m%dT%H%M%SZ')}",
        "title": "HXQ Tiny Behavior Eval V0: Q4_K_M vs Q5_K_M vs HXQ_AFFINE_6",
        "model_base": "Qwen2.5-Coder-7B-Instruct",
        "prompt_count": len(PROMPTS),
        "categories": categories,
        "results": all_results,
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
        f"~/receipts/hxq_tiny_behavior_eval_{time.strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {receipt_path}")


if __name__ == "__main__":
    main()
