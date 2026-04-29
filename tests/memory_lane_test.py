#!/usr/bin/env python3
"""
Memory Lane A/B Test — Zamba2 vs Qwen on rolling state tracking.

Uses the LLM-as-RNN pattern: feed events one at a time, ask the model
to maintain a structured JSON state capsule. Score accuracy at checkpoints.

Real data: the Docker capsule build saga (20 events, actual errors/fixes).

Usage:
    # Start llama-server with a model, then:
    python3 memory_lane_test.py --model-name zamba2-1.2b --port 8080
    python3 memory_lane_test.py --model-name qwen2.5-1.5b --port 8080
"""

import argparse
import json
import time
import sys
import os
import resource
import platform
import requests

# ---------------------------------------------------------------------------
# Real events from the Docker capsule build saga (2026-04-29)
# ---------------------------------------------------------------------------
EVENTS = [
    {
        "id": 1,
        "type": "action",
        "content": "Created gateway.py — OpenAI-compatible API gateway for capsule orchestrator, 297 lines. Smoke test passed: cold start, warm reuse, lane swap, force-model all working."
    },
    {
        "id": 2,
        "type": "error",
        "content": "Gateway import failed: _orchestrator was None during import-time check. This was a test-path bug — the orchestrator object isn't available at import time, only at runtime."
    },
    {
        "id": 3,
        "type": "fix",
        "content": "Fixed gateway import by deferring orchestrator initialization to runtime instead of import time. Smoke test passes now."
    },
    {
        "id": 4,
        "type": "error",
        "content": "config.native.json pointed at container paths like /receipts for local runs. Local startup failed because those paths don't exist on the host."
    },
    {
        "id": 5,
        "type": "fix",
        "content": "Created config.local.json with host-appropriate paths. Local capsule startup now works."
    },
    {
        "id": 6,
        "type": "action",
        "content": "Created Dockerfile for echo-capsule image. Multi-stage build: nvidia/cuda:12.4.1-devel for building, nvidia/cuda:12.4.1-runtime for final image."
    },
    {
        "id": 7,
        "type": "error",
        "content": "Docker build 1 failed: DNS resolution failure. Container could not resolve Ubuntu package hosts. The systemd-resolved stub at 127.0.0.53 is not accessible from inside containers."
    },
    {
        "id": 8,
        "type": "fix",
        "content": "Fixed DNS by running docker build with --network=host flag. Package installation now resolves correctly."
    },
    {
        "id": 9,
        "type": "error",
        "content": "Docker build 2 failed: Python import errors. Flat-copying files into /capsule/ broke Python package imports. orchestrator.py couldn't find router.py or capsule.model_pool."
    },
    {
        "id": 10,
        "type": "fix",
        "content": "Restructured Dockerfile to preserve /app/tools/capsule/ directory layout. orchestrator.py uses sys.path.insert to find sibling modules. Imports now resolve."
    },
    {
        "id": 11,
        "type": "error",
        "content": "Docker build 3 failed at link stage: undefined reference to cuMemCreate, cuMemAddressReserve, cuDeviceGet and other CUDA driver API symbols. libggml-cuda.so needs libcuda.so.1 but it doesn't exist in the build container."
    },
    {
        "id": 12,
        "type": "fix_attempt",
        "content": "Attempted fix: added CMAKE_EXE_LINKER_FLAGS and CMAKE_SHARED_LINKER_FLAGS pointing to CUDA stubs directory. Build 3 still failed — flags didn't reach the shared library link step for libggml-cuda.so."
    },
    {
        "id": 13,
        "type": "error",
        "content": "Docker build 4 failed: tried ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs but cmake doesn't respect this environment variable. Same linker error."
    },
    {
        "id": 14,
        "type": "fix_attempt",
        "content": "Docker build 5: created symlink ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so before cmake. Build compiled all CUDA kernels successfully (16 min) but failed at final executable link — linker needs libcuda.so.1 not libcuda.so."
    },
    {
        "id": 15,
        "type": "action",
        "content": "Examined official llama.cpp cuda.Dockerfile at .devops/cuda.Dockerfile. Found they use -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined to let the linker tolerate unresolved CUDA driver symbols at build time."
    },
    {
        "id": 16,
        "type": "fix",
        "content": "Applied the official llama.cpp fix: -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined. Docker build 6 started. CUDA kernels recompiling because the RUN command changed (cache invalidated)."
    },
    {
        "id": 17,
        "type": "status",
        "content": "Docker build 6 passed link stage. All four binaries (llama-server, llama-quantize, llama-cli, llama-perplexity) linked successfully. Runtime stage copied binaries and capsule Python code."
    },
    {
        "id": 18,
        "type": "action",
        "content": "Docker image echo-capsule:latest built successfully. Size: 2.39 GB. SHA256: 35a5f0ee9498dc8a."
    },
    {
        "id": 19,
        "type": "action",
        "content": "Smoke test passed: docker run --rm echo-capsule --help shows capsule orchestrator CLI with all expected flags (classify-only, status, force-model, json, tools, download)."
    },
    {
        "id": 20,
        "type": "status",
        "content": "Docker build closed. GPU-in-container runtime is the next dependency — nvidia-container-toolkit needs to be installed before docker run --gpus all will work."
    },
]

# ---------------------------------------------------------------------------
# Ground truth state at checkpoints (human-scored)
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    5: {
        "files_created": ["gateway.py", "config.local.json"],
        "files_modified": ["gateway.py"],
        "current_blocker": None,
        "errors_resolved": ["gateway import bug", "config paths"],
        "next_action": "build Docker image",
    },
    10: {
        "files_created": ["gateway.py", "config.local.json", "Dockerfile"],
        "files_modified": ["gateway.py", "Dockerfile"],
        "current_blocker": None,
        "errors_resolved": ["gateway import", "config paths", "DNS failure", "Python imports"],
        "next_action": "fix CUDA linker error",
    },
    15: {
        "files_created": ["gateway.py", "config.local.json", "Dockerfile"],
        "files_modified": ["gateway.py", "Dockerfile"],
        "current_blocker": "CUDA linker error (libcuda.so.1 not found)",
        "errors_resolved": ["gateway import", "config paths", "DNS", "Python imports"],
        "next_action": "apply --allow-shlib-undefined from official Dockerfile",
    },
    20: {
        "files_created": ["gateway.py", "config.local.json", "Dockerfile"],
        "files_modified": ["gateway.py", "Dockerfile"],
        "current_blocker": "nvidia-container-toolkit not installed",
        "errors_resolved": ["gateway import", "config paths", "DNS", "Python imports", "CUDA linker"],
        "next_action": "install nvidia-container-toolkit",
    },
}

# ---------------------------------------------------------------------------
# State schema and system prompt
# ---------------------------------------------------------------------------
STATE_SCHEMA = {
    "files_created": [],
    "files_modified": [],
    "current_blocker": None,
    "errors_resolved": [],
    "next_action": "",
}

SYSTEM_PROMPT = """You are a session state tracker. Your ONLY job is to maintain an accurate JSON state object that summarizes what has happened so far.

Rules:
1. Output ONLY valid JSON — no markdown, no explanation, no commentary.
2. Update the state based on each new event.
3. Keep lists deduplicated and concise.
4. current_blocker should be null if nothing is currently blocking, or a short string describing the active blocker.
5. errors_resolved lists problems that were fixed (short descriptions).
6. next_action is your best guess at what should happen next.

The state schema is:
{
    "files_created": ["list of files created this session"],
    "files_modified": ["list of files modified this session"],
    "current_blocker": null or "description of current blocker",
    "errors_resolved": ["list of resolved errors"],
    "next_action": "what should happen next"
}"""


def query_model(port, system_prompt, user_prompt, timeout=60):
    """Send a chat completion request to llama-server."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def build_user_prompt(state, event):
    """Build the prompt: current state + new event → produce updated state."""
    return f"""Current state:
```json
{json.dumps(state, indent=2)}
```

New event (#{event['id']}, type={event['type']}):
{event['content']}

Output the updated state JSON only."""


def parse_json_response(text):
    """Try to extract JSON from model response."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def score_state(predicted, truth, checkpoint):
    """Score predicted state against ground truth. Returns dict of per-field scores."""
    scores = {}

    # files_created: jaccard similarity
    pred_fc = set(predicted.get("files_created", []))
    true_fc = set(truth["files_created"])
    if true_fc:
        scores["files_created"] = len(pred_fc & true_fc) / len(pred_fc | true_fc) if (pred_fc | true_fc) else 1.0
    else:
        scores["files_created"] = 1.0 if not pred_fc else 0.0

    # files_modified: jaccard similarity
    pred_fm = set(predicted.get("files_modified", []))
    true_fm = set(truth["files_modified"])
    if true_fm:
        scores["files_modified"] = len(pred_fm & true_fm) / len(pred_fm | true_fm) if (pred_fm | true_fm) else 1.0
    else:
        scores["files_modified"] = 1.0 if not pred_fm else 0.0

    # current_blocker: binary (both null, or substring match)
    pred_cb = predicted.get("current_blocker")
    true_cb = truth["current_blocker"]
    if true_cb is None:
        scores["current_blocker"] = 1.0 if (pred_cb is None or pred_cb == "" or pred_cb == "None") else 0.0
    else:
        if pred_cb and isinstance(pred_cb, str):
            # Check if key terms overlap
            true_words = set(true_cb.lower().split())
            pred_words = set(pred_cb.lower().split())
            overlap = len(true_words & pred_words)
            scores["current_blocker"] = min(1.0, overlap / max(len(true_words), 1))
        else:
            scores["current_blocker"] = 0.0

    # errors_resolved: count-based (how many of the true errors are mentioned)
    pred_er = [str(e).lower() for e in predicted.get("errors_resolved", [])]
    true_er = truth["errors_resolved"]
    if true_er:
        matched = 0
        for te in true_er:
            te_words = set(te.lower().split())
            for pe in pred_er:
                pe_words = set(pe.lower().split())
                if len(te_words & pe_words) >= max(1, len(te_words) // 2):
                    matched += 1
                    break
        scores["errors_resolved"] = matched / len(true_er)
    else:
        scores["errors_resolved"] = 1.0 if not pred_er else 0.0

    # next_action: keyword overlap
    pred_na = str(predicted.get("next_action", "")).lower()
    true_na = truth["next_action"].lower()
    true_words = set(true_na.split())
    pred_words = set(pred_na.split())
    overlap = len(true_words & pred_words)
    scores["next_action"] = min(1.0, overlap / max(len(true_words), 1))

    # Overall
    scores["overall"] = sum(scores.values()) / len(scores)

    return scores


def run_test(port, model_name):
    """Run the full 20-event test and score at checkpoints."""
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    state = dict(STATE_SCHEMA)
    results = {
        "model": model_name,
        "checkpoints": {},
        "parse_failures": 0,
        "events_processed": 0,
        "per_event_latency_ms": [],
    }

    print(f"\n{'='*60}")
    print(f"  Memory Lane Test — {model_name}")
    print(f"{'='*60}\n")

    for event in EVENTS:
        prompt = build_user_prompt(state, event)

        t0 = time.time()
        response = query_model(port, SYSTEM_PROMPT, prompt)
        latency_ms = (time.time() - t0) * 1000
        results["per_event_latency_ms"].append(round(latency_ms, 1))

        parsed = parse_json_response(response)
        if parsed is None:
            results["parse_failures"] += 1
            print(f"  Event {event['id']:2d} [{event['type']:12s}] — PARSE FAIL ({latency_ms:.0f}ms)")
            print(f"    Raw: {response[:120]}...")
        else:
            state = parsed
            print(f"  Event {event['id']:2d} [{event['type']:12s}] — OK ({latency_ms:.0f}ms)")

        results["events_processed"] += 1

        # Score at checkpoints
        if event["id"] in GROUND_TRUTH:
            cp = event["id"]
            if parsed is not None:
                scores = score_state(state, GROUND_TRUTH[cp], cp)
                results["checkpoints"][cp] = {
                    "scores": scores,
                    "state_snapshot": state,
                }
                print(f"\n  --- Checkpoint {cp} ---")
                for field, score in scores.items():
                    bar = "#" * int(score * 20)
                    print(f"    {field:20s}: {score:.2f} [{bar:20s}]")
                print()
            else:
                results["checkpoints"][cp] = {
                    "scores": {"overall": 0.0},
                    "state_snapshot": None,
                    "note": "parse failure at checkpoint",
                }
                print(f"\n  --- Checkpoint {cp}: FAILED (no valid JSON) ---\n")

    # Summary
    checkpoint_scores = [
        cp["scores"]["overall"]
        for cp in results["checkpoints"].values()
        if "overall" in cp.get("scores", {})
    ]
    results["mean_overall_score"] = round(sum(checkpoint_scores) / len(checkpoint_scores), 4) if checkpoint_scores else 0
    results["median_latency_ms"] = round(sorted(results["per_event_latency_ms"])[len(results["per_event_latency_ms"])//2], 1)

    # Cost block
    results["cost"] = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS — {model_name}")
    print(f"{'='*60}")
    print(f"  Mean overall score:  {results['mean_overall_score']:.4f}")
    print(f"  Parse failures:     {results['parse_failures']}/{results['events_processed']}")
    print(f"  Median latency:     {results['median_latency_ms']}ms")
    print(f"  Wall time:          {results['cost']['wall_time_s']:.1f}s")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Memory Lane A/B Test")
    parser.add_argument("--model-name", required=True, help="Label for this model (e.g., zamba2-1.2b)")
    parser.add_argument("--port", type=int, default=8080, help="llama-server port")
    parser.add_argument("--output", default=None, help="Output JSON path (default: auto)")
    args = parser.parse_args()

    # Verify server is up
    try:
        r = requests.get(f"http://localhost:{args.port}/health", timeout=5)
        if r.status_code != 200:
            print(f"ERROR: llama-server at port {args.port} returned {r.status_code}")
            sys.exit(1)
    except requests.ConnectionError:
        print(f"ERROR: llama-server not running on port {args.port}")
        print(f"Start it with:")
        print(f"  llama-server -m <model.gguf> --port {args.port} -ngl 99")
        sys.exit(1)

    results = run_test(args.port, args.model_name)

    # Save receipt
    out_path = args.output or os.path.expanduser(
        f"~/receipts/memory_lane_test_{args.model_name.replace('/', '_')}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Receipt saved: {out_path}")


if __name__ == "__main__":
    main()
