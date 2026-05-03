#!/usr/bin/env python3
"""Live behavior smoke test — requires running backend.

Runs 5 prompts through the orchestrator and checks output quality.
This is NOT a unit test — it requires GPU and a model loaded.

Usage:
    PYTHONPATH=src:~/tools python3 tools/smoke_behavior_live.py
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.orchestrator import Orchestrator

CONFIG = os.path.expanduser("~/tools/capsule/config.json")


def check_output(output: str, must_contain: list = None,
                 must_not_contain: list = None,
                 max_tokens: int = None) -> tuple[bool, str]:
    """Check output against criteria. Returns (pass, reason)."""
    out_lower = output.lower()
    if must_contain:
        for term in must_contain:
            if term.lower() not in out_lower:
                return False, f"missing '{term}'"
    if must_not_contain:
        for term in must_not_contain:
            if term.lower() in out_lower:
                return False, f"contains forbidden '{term}'"
    return True, "ok"


def main():
    orch = Orchestrator(config_path=CONFIG)
    print(f"Default model: {orch.default_model}")
    print(f"Backend: {orch.roster.get(orch.default_model, {}).get('backend')}")
    print()

    tests = [
        {
            "prompt": "hey echo",
            "criteria": "greet, no hallucination",
            "must_not_contain": ["Alex", "Node.js", "npm", "Dear", "Best regards"],
            "max_tokens": 50,
        },
        {
            "prompt": "who are you?",
            "criteria": "identify as Echo",
            "must_contain": ["Echo"],
            "must_not_contain": ["Alex", "OpenAI", "ChatGPT"],
        },
        {
            "prompt": "what is 2 + 2?",
            "criteria": "answer 4",
            "must_contain": ["4"],
        },
        {
            "prompt": "write a python function that reverses a string",
            "criteria": "produce working code",
            "must_contain": ["def"],
        },
        {
            "prompt": "explain what entropy means in information theory",
            "criteria": "real explanation with uncertainty/information",
            "must_contain": ["information"],
        },
    ]

    passed = 0
    failed = 0
    results = []

    for t in tests:
        prompt = t["prompt"]
        print(f"TEST: {prompt}")
        print(f"  Criteria: {t['criteria']}")

        t0 = time.time()
        r = orch.process(prompt)
        elapsed = time.time() - t0

        if "error" in r:
            print(f"  FAIL: {r['error']}")
            failed += 1
            results.append({"prompt": prompt, "pass": False, "error": r["error"]})
            print()
            continue

        output = r["output"].strip()
        ok, reason = check_output(
            output,
            must_contain=t.get("must_contain"),
            must_not_contain=t.get("must_not_contain"),
        )

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        # Truncate for display
        display = output[:300] + "..." if len(output) > 300 else output
        print(f"  {status}: {reason}")
        print(f"  Output: {display}")
        print(f"  [{r['model']}] {r.get('tok_s', 0)} tok/s, "
              f"{r.get('eval_count', 0)} tokens, {elapsed:.1f}s")
        print()

        results.append({
            "prompt": prompt,
            "pass": ok,
            "reason": reason,
            "model": r["model"],
            "tok_s": r.get("tok_s", 0),
            "tokens": r.get("eval_count", 0),
            "wall_s": round(elapsed, 1),
        })

    print(f"{'=' * 50}")
    print(f"RESULT: {passed}/{passed + failed} passed")
    if failed:
        print(f"FAILURES: {failed}")
        sys.exit(1)
    else:
        print("All behavior smoke tests PASS.")

    # Write receipt
    receipt_path = os.path.expanduser(
        "~/receipts/behavior_smoke_latest.json")
    with open(receipt_path, "w") as f:
        json.dump({
            "test": "behavior_smoke_live",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "default_model": orch.default_model,
            "passed": passed,
            "failed": failed,
            "results": results,
        }, f, indent=2)
    print(f"Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
