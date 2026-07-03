#!/usr/bin/env python3
"""GBNF constrained-decode smoke test.

Runs a local LLM under GBNF grammar constraint, captures the JSON output,
validates it through program_from_json, lowers to Python+Rust, and runs
through exec_oracle. This is the first empirical proof that the json+gbnf
thesis holds — that an LLM can only emit valid, executable Poetica plans.

Usage:
    python3 tools/gbnf_smoke_test.py                    # all tasks
    python3 tools/gbnf_smoke_test.py --task add_two     # single task
    python3 tools/gbnf_smoke_test.py --dry-run           # show prompts only
    python3 tools/gbnf_smoke_test.py --model /path/to.gguf
    python3 tools/gbnf_smoke_test.py --no-exec           # parse-only, skip execution

Requires: llama-cli binary, a GGUF model file.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import resource
import platform

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poetica.plan_ir import program_from_json, lower_plan
from poetica.plan_ir_grammar import PLAN_GBNF

# -- Configuration -----------------------------------------------------------

LLAMA_SERVER = os.environ.get(
    "LLAMA_SERVER",
    os.path.expanduser("~/llama.cpp/build/bin/llama-server")
)
DEFAULT_MODEL = os.environ.get(
    "POETICA_MODEL",
    os.path.expanduser("~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf")
)
GRAMMAR_FILE = os.path.join(os.path.dirname(__file__), "..", "poetica_plan.gbnf")
SERVER_PORT = 8899  # avoid conflict with InvenTree (8080)
GPU_LAYERS = 99  # offload all to GPU (ignored if CPU-only build)


# -- Task definitions --------------------------------------------------------
# Each task: NL description, expected output (for exec_oracle), difficulty tier.
# These are ordered easiest → hardest to diagnose failure mode.

TASKS = {
    "add_two": {
        "prompt": (
            "Write a Poetica plan that seeds two variables a=5 and b=3, "
            "adds them using a weave op with a structured expr tree, "
            "and returns the result with bloom."
        ),
        "expected": "8",
        "tier": "trivial",
    },
    "negate": {
        "prompt": (
            "Write a Poetica plan that seeds x=-7, negates it using a weave "
            "with a structured neg expr, and returns the result with bloom."
        ),
        "expected": "7",
        "tier": "trivial",
    },
    "max_of_two": {
        "prompt": (
            "Write a Poetica plan that seeds a=10 and b=25, uses a weave op "
            "with a structured cond expr (if a > b then a else b) to find "
            "the maximum, and returns the result with bloom."
        ),
        "expected": "25",
        "tier": "easy",
    },
    "sum_1_to_5": {
        "prompt": (
            "Write a Poetica plan that seeds n=5, uses a cycle op to sum "
            "numbers 0 through n-1 (accumulator starts at 0, body_expr adds "
            "the loop variable to the accumulator using a structured add expr), "
            "and emits the result."
        ),
        "expected": "10",
        "tier": "medium",
    },
    "count_positive": {
        "prompt": (
            "Write a Poetica plan that seeds data=[-1, 2, -3, 4, 5] and "
            "count=0, uses for_each over data with a when condition "
            "(structured gt expr: x > 0) containing a weave that increments "
            "count (structured add expr: count + 1), and returns count "
            "with bloom."
        ),
        "expected": "3",
        "tier": "hard",
    },
}

# -- System prompt with plan-IR spec ----------------------------------------

SYSTEM_PROMPT = """You are a Poetica plan compiler. You emit ONLY valid Poetica plan JSON.

## Plan structure
A plan is a JSON object: {"name": "<string>", "ops": [<op>, ...]}

## Available ops
- seed: {"op": "seed", "name": "<id>", "value": <value>}
  Seeds a variable. value is a JSON value (number, string, bool, array).
- emit: {"op": "emit", "value": <value>}
  Prints a value to stdout.
- bloom: {"op": "bloom", "value": <value>}
  Returns a value (printed to stdout).
- weave: {"op": "weave", "inputs": ["<id>", ...], "expr": <expr>, "output": "<id>"}
  Computes an expression from inputs, binds result to output.
- when: {"op": "when", "condition": <expr>, "children": [<op>, ...]}
  Conditional block. Only executes children if condition is true.
- else: {"op": "else", "children": [<op>, ...]}
  Else block after a when.
- for_each: {"op": "for_each", "name": "<id>", "value": "<id>", "children": [<op>, ...]}
  Iterates over a list variable. "name" is a NEW iteration variable (bound fresh each
  iteration), "value" is the existing list variable to iterate over. They must be different.
  Example: name="item", value="items" gives `for item in items`.
- cycle: {"op": "cycle", "count": <value>, "accumulator": "<id>", "init": <value>, "iter_var": "<id>", "body_expr": <expr>}
  Bounded loop. Runs count times, updates accumulator with body_expr each iteration.
  iter_var names the loop counter variable (0-indexed scalar int). You choose the name.
  Use it in body_expr as {"kind": "var", "name": "<your iter_var>"}. NOT an array — do NOT use index on it.
- flow: {"op": "flow", "name": "<id>", "output": "<id>"}
  Copies/filters a variable.

## Expression trees (REQUIRED for expr, body_expr, condition)
Expressions are structured JSON trees, NOT strings.

Leaf nodes:
- {"kind": "lit", "value": <number|string|bool>}
- {"kind": "var", "name": "<id>"}

Binary ops:
- {"kind": "add|sub|mul|div|mod", "left": <expr>, "right": <expr>}
- {"kind": "gt|lt|gte|lte|eq|neq", "left": <expr>, "right": <expr>}

Unary ops:
- {"kind": "neg", "operand": <expr>}
- {"kind": "not", "operand": <expr>}

Conditional:
- {"kind": "cond", "test": <expr>, "true": <expr>, "false": <expr>}

Builtins:
- {"kind": "call", "fn": "len|abs|concat", "args": [<expr>, ...]}

Index:
- {"kind": "index", "target": "<id>", "idx": <expr>}

## Rules
- All variable names are identifiers: [a-zA-Z_][a-zA-Z0-9_]*
- Every variable in an expr/condition must be bound first (by seed, for_each, cycle, or weave)
- seed.value for lists: use JSON arrays like [1, 2, 3], not strings
- bloom/emit value should be an identifier referring to a bound variable
- index target must be a list variable, not a scalar
- Use structured expr trees everywhere, never raw strings

## Example
Task: add 5 and 3
{"name": "add_two", "ops": [{"op": "seed", "name": "a", "value": 5}, {"op": "seed", "name": "b", "value": 3}, {"op": "weave", "inputs": ["a", "b"], "expr": {"kind": "add", "left": {"kind": "var", "name": "a"}, "right": {"kind": "var", "name": "b"}}, "output": "result"}, {"op": "bloom", "value": "result"}]}

Emit ONLY the JSON plan. No explanation, no markdown, no code fences."""


def build_prompt(task_prompt: str) -> str:
    """Build the full prompt for llama-cli."""
    # Qwen2.5 ChatML format
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{task_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# -- LLM server management ---------------------------------------------------

_server_proc = None


def _start_server(model: str) -> bool:
    """Start llama-server if not already running. Returns True on success."""
    global _server_proc
    import urllib.request

    # Check if already running
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2)
        if resp.status == 200:
            print(f"  llama-server already running on :{SERVER_PORT}")
            return True
    except Exception:
        pass

    print(f"  Starting llama-server on :{SERVER_PORT} (CPU, may be slow)...")
    _server_proc = subprocess.Popen(
        [
            LLAMA_SERVER,
            "--model", model,
            "--port", str(SERVER_PORT),
            "--n-gpu-layers", str(GPU_LAYERS),
            "--ctx-size", "2048",
            "--log-disable",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to become ready (up to 120s for CPU model load)
    for i in range(120):
        time.sleep(1)
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2)
            if resp.status == 200:
                print(f"  Server ready ({i+1}s)")
                return True
        except Exception:
            pass
        if _server_proc.poll() is not None:
            print(f"  Server died (exit {_server_proc.returncode})")
            return False

    print(f"  Server failed to start within 120s")
    return False


def _stop_server():
    """Stop the llama-server if we started it."""
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        _server_proc.wait(timeout=10)
        _server_proc = None


# -- LLM invocation ----------------------------------------------------------

def run_llm(prompt: str, model: str, temperature: float = 0.0,
            max_tokens: int = 1024, timeout_s: int = 300) -> dict:
    """Send completion request to llama-server with GBNF grammar."""
    import urllib.request

    grammar_path = os.path.abspath(GRAMMAR_FILE)
    with open(grammar_path) as f:
        grammar_str = f.read()

    payload = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "grammar": grammar_str,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }).encode()

    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout_s)
        data = json.loads(resp.read())
        wall_time = time.time() - t_start
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "raw_output": "",
            "wall_time_s": round(time.time() - t_start, 2),
        }

    raw = data.get("content", "").strip()

    # Extract JSON object from response
    json_start = raw.find("{")
    json_end = raw.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        json_str = raw[json_start:json_end]
    else:
        json_str = raw

    return {
        "success": True,
        "raw_output": raw,
        "json_str": json_str,
        "wall_time_s": round(wall_time, 2),
        "tokens_predicted": data.get("tokens_predicted", 0),
        "tokens_per_second": round(
            data.get("tokens_predicted", 0) / max(wall_time, 0.01), 1),
        "stop_reason": data.get("stop_type", ""),
    }


# -- Validation pipeline -----------------------------------------------------

def validate_plan(json_str: str) -> dict:
    """Parse JSON, run through program_from_json, check validity."""
    result = {"stage": "validate"}

    # Stage 1: JSON parse
    try:
        data = json.loads(json_str)
        result["json_valid"] = True
    except json.JSONDecodeError as e:
        result["json_valid"] = False
        result["error"] = f"JSON parse error: {e}"
        return result

    # Stage 2: program_from_json
    plan = program_from_json(json_str)
    result["plan_valid"] = plan.valid
    result["plan_errors"] = plan.errors
    result["plan_name"] = plan.name
    result["op_count"] = len(plan.ops)
    result["ops_summary"] = [op.op.value for op in plan.ops]

    if not plan.valid:
        result["error"] = f"Plan validation failed: {plan.errors}"
        return result

    # Stage 3: Lower to Python
    try:
        py_code = lower_plan(plan, "python")
        result["python_code"] = py_code
        result["python_lower"] = True
    except Exception as e:
        result["python_lower"] = False
        result["error"] = f"Python lowering failed: {e}"
        return result

    # Stage 4: Lower to Rust
    try:
        rs_code = lower_plan(plan, "rust")
        result["rust_code"] = rs_code
        result["rust_lower"] = True
    except Exception as e:
        result["rust_lower"] = False
        result["error"] = f"Rust lowering failed: {e}"

    return result


def run_exec_oracle(json_str: str, expected: str) -> dict:
    """Run the plan through exec_oracle on Python (and Rust if available)."""
    from poetica.exec_oracle import OracleCase, run_single, available_targets

    result = {"stage": "execute"}
    data = json.loads(json_str)

    case = OracleCase(
        plan_json=json_str,
        expected_stdout=expected,
        description=data.get("name", "smoke"),
    )

    # Python
    py_verdict = run_single(case, "python")
    result["python_passed"] = py_verdict.passed
    result["python_actual"] = py_verdict.actual
    result["python_error"] = py_verdict.error

    # Rust (if available)
    targets = available_targets()
    if "rust" in targets:
        rs_verdict = run_single(case, "rust")
        result["rust_passed"] = rs_verdict.passed
        result["rust_actual"] = rs_verdict.actual
        result["rust_error"] = rs_verdict.error

        # Cross-target equality
        if py_verdict.passed and rs_verdict.passed:
            result["cross_target_equal"] = py_verdict.actual == rs_verdict.actual

    return result


# -- Main harness ------------------------------------------------------------

def run_smoke_test(task_name: str, task: dict, model: str,
                   dry_run: bool = False, no_exec: bool = False) -> dict:
    """Run one smoke test: prompt → LLM → validate → execute."""
    prompt = build_prompt(task["prompt"])

    receipt = {
        "task": task_name,
        "tier": task["tier"],
        "expected": task["expected"],
        "model": os.path.basename(model),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if dry_run:
        receipt["prompt"] = prompt
        receipt["status"] = "dry_run"
        return receipt

    # Step 1: LLM generation
    print(f"  [{task_name}] Generating under GBNF constraint...")
    llm_result = run_llm(prompt, model)
    receipt["llm"] = {
        "success": llm_result["success"],
        "wall_time_s": llm_result["wall_time_s"],
        "raw_length": len(llm_result.get("raw_output", "")),
    }

    if not llm_result["success"]:
        receipt["status"] = "llm_error"
        receipt["error"] = llm_result.get("error", "Unknown LLM error")
        # Include stderr for diagnosis
        if llm_result.get("error"):
            receipt["llm"]["stderr"] = llm_result["error"][:500]
        return receipt

    json_str = llm_result.get("json_str", "")
    receipt["llm"]["json_str"] = json_str

    # Step 2: Validate
    print(f"  [{task_name}] Validating plan...")
    val_result = validate_plan(json_str)
    receipt["validate"] = val_result

    if not val_result.get("plan_valid", False):
        receipt["status"] = "validation_error"
        return receipt

    if no_exec:
        receipt["status"] = "parse_only_pass"
        return receipt

    # Step 3: Execute
    print(f"  [{task_name}] Running exec_oracle...")
    exec_result = run_exec_oracle(json_str, task["expected"])
    receipt["execute"] = exec_result

    # Verdict
    py_pass = exec_result.get("python_passed", False)
    rs_pass = exec_result.get("rust_passed", None)  # None if not available
    cross_eq = exec_result.get("cross_target_equal", None)

    if py_pass and (rs_pass is None or rs_pass) and (cross_eq is None or cross_eq):
        receipt["status"] = "PASS"
    elif py_pass and rs_pass is False:
        receipt["status"] = "PARTIAL_python_only"
    else:
        receipt["status"] = "FAIL"

    return receipt


def main():
    parser = argparse.ArgumentParser(description="GBNF constrained-decode smoke test")
    parser.add_argument("--task", type=str, help="Run a single task by name")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to GGUF model file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts only, don't run LLM")
    parser.add_argument("--no-exec", action="store_true",
                        help="Parse-only, skip execution")
    parser.add_argument("--receipt-dir", type=str,
                        default=os.path.expanduser("~/receipts/gbnf_smoke_test"),
                        help="Directory for receipt JSON files")
    args = parser.parse_args()

    # Verify prerequisites
    if not args.dry_run:
        if not os.path.exists(LLAMA_SERVER):
            print(f"ERROR: llama-server not found at {LLAMA_SERVER}")
            print(f"  Set LLAMA_SERVER env var or check path")
            sys.exit(1)
        if not os.path.exists(args.model):
            print(f"ERROR: Model not found at {args.model}")
            sys.exit(1)
        if not os.path.exists(GRAMMAR_FILE):
            print(f"ERROR: Grammar file not found at {GRAMMAR_FILE}")
            print(f"  Run: python3 -c 'from poetica.plan_ir_grammar import PLAN_GBNF; print(PLAN_GBNF)' > poetica_plan.gbnf")
            sys.exit(1)
        if not _start_server(args.model):
            print("ERROR: Failed to start llama-server")
            sys.exit(1)

    # Select tasks
    if args.task:
        if args.task not in TASKS:
            print(f"ERROR: Unknown task '{args.task}'. Available: {list(TASKS.keys())}")
            sys.exit(1)
        tasks = {args.task: TASKS[args.task]}
    else:
        tasks = TASKS

    # Run
    t_start = time.time()
    cpu_start = time.process_time()
    results = []

    print(f"GBNF Smoke Test — {len(tasks)} task(s)")
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Grammar: {os.path.basename(GRAMMAR_FILE)}")
    print()

    for name, task in tasks.items():
        receipt = run_smoke_test(name, task, args.model,
                                dry_run=args.dry_run, no_exec=args.no_exec)
        results.append(receipt)

        status = receipt["status"]
        symbol = {"PASS": "+", "FAIL": "X", "dry_run": "~",
                  "parse_only_pass": "~"}.get(status, "?")
        print(f"  [{symbol}] {name}: {status}")
        if status == "FAIL":
            if "execute" in receipt:
                err = receipt["execute"].get("python_error", "")
                actual = receipt["execute"].get("python_actual", "")
                print(f"      Expected: {task['expected']}")
                print(f"      Actual:   {actual}")
                if err:
                    print(f"      Error:    {err}")
        print()

    # Summary
    wall_time = round(time.time() - t_start, 2)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    partial = sum(1 for r in results if "PARTIAL" in r.get("status", ""))
    errors = sum(1 for r in results if r["status"] in ("llm_error", "validation_error"))

    print(f"--- Results: {passed} PASS / {failed} FAIL / {partial} PARTIAL / {errors} ERROR ---")
    print(f"    Wall time: {wall_time}s")

    # Save receipt
    if not args.dry_run:
        os.makedirs(args.receipt_dir, exist_ok=True)
        receipt_path = os.path.join(
            args.receipt_dir,
            f"gbnf_smoke_{time.strftime('%Y%m%dT%H%M%S')}.json"
        )
        report = {
            "test": "gbnf_constrained_decode_smoke",
            "model": os.path.basename(args.model),
            "grammar": "poetica_plan.gbnf",
            "tasks": results,
            "summary": {
                "total": len(results),
                "passed": passed,
                "failed": failed,
                "partial": partial,
                "errors": errors,
            },
            "cost": {
                "wall_time_s": wall_time,
                "cpu_time_s": round(time.process_time() - cpu_start, 3),
                "peak_memory_mb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
                "timestamp_start": results[0]["timestamp"] if results else "",
                "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }
        with open(receipt_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"    Receipt: {receipt_path}")

    # Clean up server if we started it
    _stop_server()

    sys.exit(0 if failed == 0 and errors == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    finally:
        _stop_server()
