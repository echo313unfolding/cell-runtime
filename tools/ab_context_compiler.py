#!/usr/bin/env python3
"""A/B harness: legacy context path vs CPU-side context compiler.

Runs a fixed prompt set through the SAME Orchestrator, binary and model,
varying only CELL_CONTEXT_COMPILER (0 = legacy capsule + raw history +
full tool registry, 1 = compiler + compressed history + lean registry).

Every number in the receipt is measured, not estimated:
  - prompt tokens / predicted tokens / prompt_ms / predicted_ms come from
    llama-server's `timings` block (surfaced by orchestrator._openai_chat)
  - headroom = n_ctx - first_prompt_tokens - predicted_n
  - degeneration = tail uniqueness + top 4-gram coverage (see _degenerate)
  - retrieval hit = a gold path fragment appears in context_compiled.sources
    (compiler) or in a tool result / the output (legacy)

Scenarios go 3 turns deep on purpose: the history-tail bug in the compiler
(fixed in this patch) and legacy context overflow both only show at turn 3+.

Usage (on the research box, from ~/cell-runtime):

    CELL_ASK_PASS=0 python3 tools/ab_context_compiler.py \
        --config src/cell/config.json --start-server

    # or against an already-running llama-server on the config URL:
    CELL_ASK_PASS=0 python3 tools/ab_context_compiler.py --config src/cell/config.json

    # custom prompt set:
    python3 tools/ab_context_compiler.py --prompts my_scenarios.json

Receipt: ~/receipts/cell/echo_context_compiler_ab_v2_<UTC stamp>.json
"""
from __future__ import annotations

import argparse
import ast
import collections
import statistics
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

# ---------------------------------------------------------------------------
# Default scenarios. Each is 3 turns deep. `gold` lists path fragments, any
# one of which counts as a retrieval hit. Scenarios with no gold are scored
# on coherence/cost only.
# ---------------------------------------------------------------------------
DEFAULT_SCENARIOS = [
    {
        "id": "af6_bench",
        "gold": ["hxq_af6", "zamba2_t2000_bench", "phase0_receipt"],
        "turns": [
            "Find the HXQ AF6 benchmark receipt for the Quadro T2000. Identify the "
            "model, binary, and baseline formats measured.",
            "What were the measured tokens per second for prompt processing and for "
            "generation in that receipt?",
            "Compare that against the baseline format from the same receipt family. "
            "Which was faster and by roughly how much?",
        ],
    },
    {
        "id": "context_compiler_receipt",
        "gold": ["echo_context_compiler_integration"],
        "turns": [
            "What does the echo context compiler integration receipt say about "
            "turn 2 degeneration?",
            "How many gates passed and what token budget was used?",
            "Which source files did that receipt list as created or modified?",
        ],
    },
    {
        "id": "project_search",
        "gold": ["echo_local_assistant_project_search"],
        "turns": [
            "Summarize the echo local assistant project search receipt from 2026-10-28.",
            "How many tool calls did it record and which receipt did it find?",
            "What was the outcome of the multi-turn follow-up in that run?",
        ],
    },
    {
        "id": "hxq_contribution",
        "gold": ["WO_HXQ_CONTRIBUTION_MODEL_VALIDATION"],
        "turns": [
            "What did the HXQ contribution model validation receipt (WO 004) conclude?",
            "What gate or verdict did it record?",
            "List the artifacts it references.",
        ],
    },
    {
        # Reproduces the Oct 28 failure condition (commit 164522a): turn 1
        # "succeeded" by dumping the raw receipt into the answer, and turn 2
        # inherits that dump through history. Legacy passes it raw; the
        # compiler compresses it. Skipped if the receipt isn't present.
        "id": "af6_dump_repro",
        "gold": ["hxq_af6", "phase0_receipt"],
        "seed": [{"user": "Read the AF6 phase 0 receipt and show me its contents.",
                  "assistant_from_file": "~/receipts/hxq_af6_mmq_phase0/phase0_receipt.json"}],
        "turns": [
            "Based on that receipt, which model, binary and baseline formats were measured?",
            "What were the measured tokens per second, and how do they compare to the baseline?",
            "Summarize the verdict in two sentences.",
        ],
    },
    {
        "id": "coding",
        "gold": [],
        "turns": [
            "Write a Python function that parses llama-bench JSON output and returns "
            "a dict mapping test name to avg_ts.",
            "Add type hints and handle the case where the file holds a single dict "
            "instead of a list.",
            "Write a pytest for it with a two-entry fixture.",
        ],
    },
    {
        "id": "security_triage",
        "gold": [],
        "turns": [
            "Triage this: a cron job on the research box runs curl to an unknown IP "
            "every 5 minutes with base64-encoded arguments.",
            "What severity would you assign and what is the first containment step?",
            "Which logs should be preserved for forensics before anything is changed?",
        ],
    },
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _degenerate(text: str) -> dict:
    """Cheap degeneration detector.

    tail_unique: distinct characters in the last 120 chars. The failure that
                 motivated the compiler was "100000000..." (tail_unique ~2).
    top4_cover:  fraction of the output covered by its most frequent 4-gram.
                 Repeating loops push this toward 1.0; prose sits well under 0.2.
    """
    t = text or ""
    tail = t[-120:]
    tail_unique = len(set(tail))
    grams = collections.Counter(t[i:i + 4] for i in range(max(0, len(t) - 3)))
    top = grams.most_common(1)[0][1] if grams else 0
    top4_cover = (top * 4) / len(t) if t else 0.0
    flag = (len(t) >= 40 and tail_unique < 8) or top4_cover > 0.4
    return {"tail_unique": tail_unique,
            "top4_cover": round(top4_cover, 3),
            "degenerate": bool(flag)}


def _hit(gold: list[str], result: dict) -> bool | None:
    if not gold:
        return None
    hay = []
    hay.extend(result.get("context_compiled", {}).get("sources", []))
    hay.append(result.get("output", ""))
    # Legacy mode finds receipts through tool calls; look at the task record
    try:
        with open(result["receipt"]) as f:
            task = json.load(f)
        for tc in task.get("tool_calls", []):
            hay.append(json.dumps(tc.get("arguments", {})))
            hay.append(str(tc.get("result", "")))
    except (KeyError, OSError, json.JSONDecodeError):
        pass
    blob = "\n".join(hay).lower()
    return any(g.lower() in blob for g in gold)


# ---------------------------------------------------------------------------
# Coding output scorer
# ---------------------------------------------------------------------------
def _extract_python_block(text: str) -> str | None:
    """Extract the first fenced code block from model output.

    Accepts ```python, ```py, or bare ```.  Tolerates missing closing
    fence (model hit num_predict before emitting it).
    """
    m = re.search(r'```(?:python|py)?\s*\n(.*?)(?:```|$)', text, re.DOTALL)
    if not m:
        return None
    code = m.group(1).strip()
    return code if code else None


def _score_coding_t1(output: str) -> dict:
    """T1: extract function, ast.parse, exec, call on fixture, assert dict.
    Returns {"code_score": 0-3, "detail": str}."""
    code = _extract_python_block(output)
    if not code:
        return {"code_score": 0, "detail": "no ```python block found"}

    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"code_score": 1, "detail": f"syntax error: {e}"}

    # Run in subprocess with fixture
    fixture = json.dumps([
        {"model_filename": "zamba2.gguf", "test": "pp512", "avg_ts": 311.2},
        {"model_filename": "zamba2.gguf", "test": "tg128", "avg_ts": 37.4},
    ])
    test_script = (
        f"{code}\n\n"
        f"import json, sys\n"
        f"data = json.loads({fixture!r})\n"
        f"# Find the function name — first def at module level\n"
        f"import types\n"
        f"funcs = [v for v in dir() if callable(eval(v)) and not v.startswith('_') and isinstance(eval(v), types.FunctionType)]\n"
        f"assert funcs, 'no function found'\n"
        f"fn = eval(funcs[0])\n"
        f"# Try calling with the list\n"
        f"result = fn(data)\n"
        f"assert isinstance(result, dict), f'expected dict, got {{type(result).__name__}}'\n"
        f"assert 'pp512' in result or 'tg128' in result, f'expected test names in keys, got {{list(result.keys())}}'\n"
        f"print('PASS')\n"
    )
    return _run_code_subprocess(test_script, code, "T1")


def _score_coding_t2(output: str, prior_code: str) -> dict:
    """T2: extract code, check single-dict + type annotations.

    If no fenced block but prior_code already handles single-dict,
    accept the prior as a pass (model correctly said "it already does").
    Returns {"code_score": 0-3, "detail": str}.
    """
    code = _extract_python_block(output)

    fixture_list = json.dumps([
        {"model_filename": "zamba2.gguf", "test": "pp512", "avg_ts": 311.2},
        {"model_filename": "zamba2.gguf", "test": "tg128", "avg_ts": 37.4},
    ])
    fixture_single = json.dumps(
        {"model_filename": "zamba2.gguf", "test": "pp512", "avg_ts": 311.2}
    )

    # No new code block — check if the prior code already handles single-dict
    if not code and prior_code:
        try:
            ast.parse(prior_code)
        except SyntaxError:
            return {"code_score": 0, "detail": "no new block and prior has syntax error"}
        test_script = (
            f"{prior_code}\n\n"
            f"import json, sys, types\n"
            f"funcs = [v for v in dir() if callable(eval(v)) and not v.startswith('_') and isinstance(eval(v), types.FunctionType)]\n"
            f"assert funcs, 'no function found'\n"
            f"fn = eval(funcs[0])\n"
            f"r2 = fn(json.loads({fixture_single!r}))\n"
            f"assert isinstance(r2, dict), f'single case: expected dict, got {{type(r2).__name__}}'\n"
            f"print('PASS')\n"
        )
        result = _run_code_subprocess(test_script, prior_code, "T2-prior")
        if result["code_score"] == 3:
            return {"code_score": 2, "detail": "T2 no new block; prior already handles single-dict (accepted)"}
        return {"code_score": 0, "detail": "no new block and prior fails single-dict"}
    elif not code:
        return {"code_score": 0, "detail": "no ```python block found"}

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"code_score": 1, "detail": f"syntax error: {e}"}

    # Check for type annotations on the first function def
    has_annotations = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.returns or any(a.annotation for a in node.args.args):
                has_annotations = True
            break

    test_script = (
        f"{code}\n\n"
        f"import json, sys, types\n"
        f"funcs = [v for v in dir() if callable(eval(v)) and not v.startswith('_') and isinstance(eval(v), types.FunctionType)]\n"
        f"assert funcs, 'no function found'\n"
        f"fn = eval(funcs[0])\n"
        f"# List case\n"
        f"r1 = fn(json.loads({fixture_list!r}))\n"
        f"assert isinstance(r1, dict), f'list case: expected dict, got {{type(r1).__name__}}'\n"
        f"# Single-dict case\n"
        f"r2 = fn(json.loads({fixture_single!r}))\n"
        f"assert isinstance(r2, dict), f'single case: expected dict, got {{type(r2).__name__}}'\n"
        f"print('PASS')\n"
    )
    result = _run_code_subprocess(test_script, code, "T2")
    if result["code_score"] == 3 and not has_annotations:
        result["code_score"] = 2
        result["detail"] += "; missing type annotations"
    return result


def _score_coding_t3(output: str, prior_code: str) -> dict:
    """T3: extract pytest block, write both to temp dir, run pytest.
    Scores against prior_code (latest code block seen, from T1 or T2).
    Returns {"code_score": 0-3, "detail": str}."""
    test_code = _extract_python_block(output)
    if not test_code:
        return {"code_score": 0, "detail": "no ```python block found"}

    try:
        ast.parse(test_code)
    except SyntaxError as e:
        return {"code_score": 1, "detail": f"syntax error in test: {e}"}

    if not prior_code:
        return {"code_score": 1, "detail": "no prior function code to test against"}

    with tempfile.TemporaryDirectory(prefix="cell_coding_") as td:
        # Write the module under test
        mod_path = os.path.join(td, "parse_bench.py")
        with open(mod_path, "w") as f:
            f.write(prior_code)

        # Write the test file, patching imports to find the module
        test_path = os.path.join(td, "test_parse_bench.py")
        with open(test_path, "w") as f:
            f.write(f"import sys; sys.path.insert(0, {td!r})\n")
            f.write(test_code)

        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-q", "--tb=short", "--no-header"],
                capture_output=True, text=True, timeout=30,
                cwd=td,
            )
            if r.returncode == 0:
                return {"code_score": 3, "detail": f"pytest PASS: {r.stdout.strip()[-120:]}"}
            else:
                detail = (r.stdout + r.stderr).strip()[-200:]
                return {"code_score": 2, "detail": f"pytest FAIL: {detail}"}
        except subprocess.TimeoutExpired:
            return {"code_score": 1, "detail": "pytest timed out (30s)"}
        except Exception as e:
            return {"code_score": 1, "detail": f"pytest error: {e}"}


def _run_code_subprocess(script: str, code: str, label: str) -> dict:
    """Execute a test script in a subprocess with 30s timeout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        script_path = f.name
    try:
        r = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and "PASS" in r.stdout:
            return {"code_score": 3, "detail": f"{label} PASS"}
        else:
            detail = (r.stdout + r.stderr).strip()[-200:]
            return {"code_score": 2, "detail": f"{label} exec fail: {detail}"}
    except subprocess.TimeoutExpired:
        return {"code_score": 1, "detail": f"{label} timed out (30s)"}
    except Exception as e:
        return {"code_score": 1, "detail": f"{label} error: {e}"}
    finally:
        os.unlink(script_path)


def _score_coding_turn(scenario_id: str, turn: int, output: str,
                       prior_code: str | None) -> dict | None:
    """Score a coding scenario turn. Returns None for non-coding scenarios.

    prior_code is the latest code block seen in earlier turns (not just T1).
    """
    if scenario_id != "coding":
        return None
    if turn == 1:
        return _score_coding_t1(output)
    elif turn == 2:
        return _score_coding_t2(output, prior_code)
    elif turn == 3:
        return _score_coding_t3(output, prior_code)
    return None


# ---------------------------------------------------------------------------
# llama-server lifecycle
# ---------------------------------------------------------------------------
def _health(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def _find_binary() -> str | None:
    for c in ["~/llama-cpp-stage1/build/bin/llama-server",
              "~/llama-cpp-fresh/build/bin/llama-server",
              "/usr/local/bin/llama-server"]:
        p = os.path.expanduser(c)
        if os.path.exists(p):
            return p
    return None


def _start_server(config: dict, model: str,
                  extra_args: str = "") -> subprocess.Popen:
    binary = _find_binary()
    if not binary:
        sys.exit("no llama-server binary found")
    gguf = os.path.expanduser(config["roster"][model]["gguf"])
    url = config.get("llama_server_url", "http://localhost:8080")
    port = url.rsplit(":", 1)[-1]
    gen = config.get("generation", {})
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{os.path.dirname(binary)}:{env.get('LD_LIBRARY_PATH', '')}"
    log = Path(os.path.expanduser("~/receipts/cell/ab_llama_server.log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [binary, "--model", gguf, "--port", port,
           "--ctx-size", str(gen.get("num_ctx", 4096)),
           "--n-gpu-layers", "99", "--host", "127.0.0.1"]
    cache_k = gen.get("cache_type_k")
    cache_v = gen.get("cache_type_v")
    if cache_k:
        cmd.extend(["--cache-type-k", cache_k])
    if cache_v:
        cmd.extend(["--cache-type-v", cache_v])
    if extra_args:
        cmd.extend(extra_args.split())
    proc = subprocess.Popen(
        cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env,
        preexec_fn=os.setsid)
    for _ in range(120):
        if _health(url):
            return proc
        if proc.poll() is not None:
            sys.exit(f"llama-server exited early; see {log}")
        time.sleep(1)
    sys.exit(f"llama-server did not become healthy; see {log}")


# ---------------------------------------------------------------------------
# One mode, all scenarios
# ---------------------------------------------------------------------------
def run_mode(mode: str, config_path: str, scenarios: list[dict],
             use_tools: bool, depth: int) -> list[dict]:
    # ── Set env vars for this mode ──
    # compiler<N> = compiler with N-token evidence budget
    # compiler_notools = suppress tools when evidence present
    # compiler_syshint = hint in system prompt
    # All compiler* modes enable the compiler; anything else is legacy.
    for v in ("CELL_CONTEXT_BUDGET", "CELL_TOOLS_WHEN_EVIDENCE",
              "CELL_HINT_IN_SYSTEM", "CELL_CODING_EXEMPT"):
        os.environ.pop(v, None)

    if mode.startswith("compiler"):
        os.environ["CELL_CONTEXT_COMPILER"] = "1"
        if mode == "compiler600":
            os.environ["CELL_CONTEXT_BUDGET"] = "600"
        elif mode == "compiler2000":
            os.environ["CELL_CONTEXT_BUDGET"] = "2000"
        elif mode == "compiler3000":
            os.environ["CELL_CONTEXT_BUDGET"] = "3000"

        if mode in ("compiler_notools", "compiler_notools_strict"):
            os.environ["CELL_TOOLS_WHEN_EVIDENCE"] = "off"
        if mode == "compiler_notools":
            os.environ["CELL_CODING_EXEMPT"] = "1"  # control: exemption on
        if mode == "compiler_notools_strict":
            os.environ["CELL_CODING_EXEMPT"] = "0"  # experiment: exemption off
        if mode == "compiler_syshint":
            os.environ["CELL_HINT_IN_SYSTEM"] = "1"
    else:
        os.environ["CELL_CONTEXT_COMPILER"] = "0"
    from cell.orchestrator import Orchestrator  # import after env set is fine; flag read in __init__

    orch = Orchestrator(config_path)
    model = orch.default_model
    # Adopt the already-running server so sticky policy never tries to swap.
    if orch.pool._llama_server_alive():
        orch.pool._llama_current_model = model
    n_ctx = int(orch.gen_opts.get("num_ctx", 4096))

    # Helper: evict llama-server's single prompt-cache slot between
    # scenarios so T1 cache numbers aren't inflated by cross-scenario
    # prefix matches.  A 1-token completion with a unique prompt is
    # enough to overwrite the slot.
    def _evict_slot():
        url = orch.config.get("llama_server_url", "http://localhost:8080")
        nonce = f"cache_bust_{time.time_ns()}"
        try:
            orch.pool._http(f"{url}/v1/chat/completions", {
                "messages": [{"role": "user", "content": nonce}],
                "max_tokens": 1, "temperature": 0,
            }, timeout=10)
        except Exception:
            pass

    rows = []
    for sc in scenarios:
        _evict_slot()
        orch.reset_session() if hasattr(orch, "reset_session") else orch.memory.reset()
        history: list[dict] = []
        skip = False
        for seed in sc.get("seed", []):
            if "assistant_from_file" in seed:
                fp = os.path.expanduser(seed["assistant_from_file"])
                if not os.path.isfile(fp):
                    print(f"[{mode:8s}] {sc['id']:26s} skipped: {fp} missing", flush=True)
                    skip = True
                    break
                with open(fp) as f:
                    content = f.read()
                # Prime carry-forward so T1 can find the seed receipt.
                # Store the full excerpt so the byte prefix is stable.
                if hasattr(orch, "_prior_excerpts"):
                    from cell.context_compiler import _read_receipt_compact
                    excerpt = _read_receipt_compact(fp, 2000) or ""
                    orch._prior_excerpts = [{"path": fp,
                                             "name": os.path.basename(fp),
                                             "excerpt": excerpt}]
            else:
                content = seed.get("assistant", "")
            history.append({"role": "user", "content": seed["user"]})
            history.append({"role": "assistant", "content": content})
        if skip:
            continue
        latest_code = None  # Track latest code block for coding scorer
        for ti, q in enumerate(sc["turns"][:depth], start=1):
            t0 = time.time()
            err = None
            try:
                res = orch.process(q, force_model=model, use_tools=use_tools,
                                   history=history or None)
            except Exception as e:  # context overflow, HTTP error, etc.
                res = {"output": "", "error": repr(e)}
                err = repr(e)
            wall = round(time.time() - t0, 3)
            out = res.get("output", "") or ""
            deg = _degenerate(out)
            intent = res.get("intent", "")
            first_pt = res.get("first_prompt_tokens", 0)      # full context, 1st call
            first_ev = res.get("first_prompt_evaluated", 0)   # of which uncached
            max_pt = res.get("max_prompt_tokens", 0) or first_pt
            evalc = res.get("eval_count", 0)

            # Coding scorer — uses latest code block from any prior turn
            code_result = _score_coding_turn(sc["id"], ti, out, latest_code)
            if sc["id"] == "coding":
                new_code = _extract_python_block(out)
                if new_code:
                    latest_code = new_code

            row = {
                "mode": mode,
                "scenario": sc["id"],
                "turn": ti,
                "intent": intent,
                "error": err or res.get("error"),
                "timings_measured": res.get("timings_measured", False),
                "first_prompt_tokens": first_pt,
                "first_prompt_evaluated": first_ev,
                "first_prompt_cached": max(0, first_pt - first_ev),
                "max_prompt_tokens": max_pt,
                "prompt_tokens_total": res.get("prompt_tokens", 0),
                "prompt_ms": res.get("prompt_ms", 0.0),
                "eval_tokens": evalc,
                # conservative: largest context any call saw, minus all tokens
                # generated across the loop
                "headroom": (n_ctx - max_pt - evalc) if max_pt else None,
                "tok_s": res.get("tok_s", 0),
                "wall_s": wall,
                "tool_calls": res.get("tool_calls", 0),
                "retrieval_hit": _hit(sc["gold"], res) if not err else False,
                "output_chars": len(out),
                "output_head": out[:160],
                "evidence_tokens_est": res.get("context_compiled", {}).get("tokens_est", 0),
                "sources": res.get("context_compiled", {}).get("sources", []),
                "receipt": res.get("receipt"),
                "context_ceiling_hit": res.get("context_ceiling_hit", False),
                "stagnation": res.get("stagnation", False),
                "force_commit": res.get("force_commit"),
                "code_score": code_result["code_score"] if code_result else None,
                "code_detail": code_result["detail"] if code_result else None,
                **deg,
            }
            rows.append(row)
            flags = ""
            if row["context_ceiling_hit"]:
                flags += " CEIL"
            if row["stagnation"]:
                flags += " STAG"
            code_flag = ""
            if code_result:
                code_flag = f" code={code_result['code_score']}/3"
            print(f"[{mode:8s}] {sc['id']:26s} T{ti} [{intent:8s}] "
                  f"ctx={first_pt:5d} (uncached {first_ev:4d}) max={max_pt:5d} "
                  f"ev={evalc:4d} head={row['headroom']} "
                  f"deg={deg['degenerate']} hit={row['retrieval_hit']} "
                  f"wall={wall}s{code_flag}{flags}"
                  f"{'ERR ' + err if err else ''}", flush=True)
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": out})
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        scored = [r for r in rs if r["retrieval_hit"] is not None]
        measured = [r for r in rs if r["first_prompt_tokens"]]
        out[mode] = {
            "turns": len(rs),
            "errors": sum(1 for r in rs if r["error"]),
            "degenerate_turns": sum(1 for r in rs if r["degenerate"]),
            "retrieval_hits": f"{sum(1 for r in scored if r['retrieval_hit'])}/{len(scored)}",
            "mean_first_prompt_tokens": round(
                sum(r["first_prompt_tokens"] for r in measured) / len(measured), 1) if measured else None,
            "mean_max_prompt_tokens": round(
                sum(r["max_prompt_tokens"] for r in measured) / len(measured), 1) if measured else None,
            "max_prompt_tokens": max((r["max_prompt_tokens"] for r in measured), default=None),
            "mean_prompt_cached_frac": round(
                sum(r["first_prompt_cached"] for r in measured)
                / max(1, sum(r["first_prompt_tokens"] for r in measured)), 3) if measured else None,
            "cached_frac_by_turn": {
                f"T{t}": round(
                    sum(r["first_prompt_cached"] for r in measured if r["turn"] == t)
                    / max(1, sum(r["first_prompt_tokens"] for r in measured if r["turn"] == t)), 3)
                for t in sorted({r["turn"] for r in measured})
            } if measured else {},
            "mean_prompt_ms": round(sum(r["prompt_ms"] for r in rs) / len(rs), 1),
            "mean_prompt_ms_t2t3": round(
                sum(r["prompt_ms"] for r in rs if r["turn"] >= 2)
                / max(1, sum(1 for r in rs if r["turn"] >= 2)), 1),
            "mean_headroom": round(
                sum(r["headroom"] for r in measured) / len(measured), 1) if measured else None,
            "min_headroom": min((r["headroom"] for r in measured), default=None),
            "mean_wall_s": round(sum(r["wall_s"] for r in rs) / len(rs), 2),
            "median_wall_s": round(statistics.median(r["wall_s"] for r in rs), 2),
            "mean_wall_s_by_turn": {
                f"T{t}": round(
                    sum(r["wall_s"] for r in rs if r["turn"] == t)
                    / max(1, sum(1 for r in rs if r["turn"] == t)), 2)
                for t in sorted({r["turn"] for r in rs})
            },
            "mean_tool_calls": round(sum(r["tool_calls"] for r in rs) / len(rs), 2),
            "turn3_degenerate": sum(1 for r in rs if r["turn"] == 3 and r["degenerate"]),
            "context_ceiling_hits": sum(1 for r in rs if r.get("context_ceiling_hit")),
            "stagnation_hits": sum(1 for r in rs if r.get("stagnation")),
        }
        # Coding scorer summary
        code_rows = [r for r in rs if r.get("code_score") is not None]
        if code_rows:
            out[mode]["coding_pass"] = f"{sum(1 for r in code_rows if r['code_score'] == 3)}/{len(code_rows)}"
            out[mode]["code_scores"] = {
                f"T{r['turn']}": r["code_score"] for r in code_rows
            }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE.parent / "src" / "cell" / "config.json"))
    ap.add_argument("--modes", default="compiler_notools,compiler_notools_strict")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--tools", choices=["on", "off"], default="on")
    ap.add_argument("--prompts", help="JSON file: list of {id, gold, turns}")
    ap.add_argument("--start-server", action="store_true",
                    help="launch llama-server from config; killed on exit")
    ap.add_argument("--llama-args", default="",
                    help="extra args passed to llama-server (e.g. '--cache-reuse 256')")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each mode N times for noise measurement (default 1)")
    ap.add_argument("--out", help="receipt path (default ~/receipts/cell/echo_context_compiler_ab_v8_<ts>.json)")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    url = config.get("llama_server_url", "http://localhost:8080")
    scenarios = DEFAULT_SCENARIOS
    if args.prompts:
        with open(args.prompts) as f:
            scenarios = json.load(f)

    if os.environ.get("CELL_ASK_PASS", "1") != "0" and args.tools == "on":
        print("note: CELL_ASK_PASS is not 0 — privileged tools (shell) will prompt on stdin. "
              "Set CELL_ASK_PASS=0 for unattended runs.", file=sys.stderr)

    proc = None
    if args.start_server:
        proc = _start_server(config, config["default_model"],
                             extra_args=args.llama_args)
    elif not _health(url):
        sys.exit(f"llama-server not healthy at {url}; pass --start-server or start it yourself")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rows: list[dict] = []
    try:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        for rep in range(args.repeat):
            for mode in modes:
                run_label = f"{mode}" if args.repeat == 1 else f"{mode}_r{rep + 1}"
                run_rows = run_mode(mode, args.config, scenarios,
                                    use_tools=(args.tools == "on"), depth=args.depth)
                # Tag each row with the run label for noise analysis
                for r in run_rows:
                    r["run"] = rep + 1
                    r["mode_label"] = run_label
                rows.extend(run_rows)
    finally:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    summary = summarize(rows)
    receipt = {
        "receipt_id": f"echo_context_compiler_ab_v8_{stamp}",
        "title": "Context compiler A/B v8 — code carry-forward, intent num_predict, exemption test",
        "timestamp": stamp,
        "config": args.config,
        "llama_server_url": url,
        "model": config.get("default_model"),
        "n_ctx": config.get("generation", {}).get("num_ctx"),
        "num_predict": config.get("generation", {}).get("num_predict"),
        "use_tools": args.tools == "on",
        "depth": args.depth,
        "repeat": args.repeat,
        "scenarios": [{"id": s["id"], "gold": s["gold"], "n_turns": len(s["turns"])} for s in scenarios],
        "metric_definitions": {
            "first_prompt_tokens": "usage.prompt_tokens of the first call in the turn = FULL context the model saw (counts against n_ctx)",
            "first_prompt_evaluated": "timings.prompt_n of that call = tokens not served from llama-server prompt cache (speed, not occupancy)",
            "max_prompt_tokens": "largest usage.prompt_tokens of any call in the turn's tool loop",
            "headroom": "n_ctx - max_prompt_tokens - eval_tokens (conservative)",
            "degenerate": "tail_unique<8 over last 120 chars (len>=40) OR top 4-gram covers >40% of output",
            "retrieval_hit": "any gold fragment in context_compiled.sources, tool call args/results, or output",
            "context_ceiling_hit": "tool-loop guard fired: next tool result would exceed n_ctx - margin",
            "stagnation": "tool-loop guard fired: repeated (tool,args) or identical result from different call",
            "code_score": "coding scenario only: 0=no code block, 1=parse/timeout, 2=exec fail or missing annotations, 3=PASS",
            "coding_pass": "N/M coding turns that scored 3 (full pass)",
        },
        "summary": summary,
        "rows": rows,
    }
    out = Path(os.path.expanduser(args.out)) if args.out else \
        Path(os.path.expanduser(f"~/receipts/cell/{receipt['receipt_id']}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2)

    print("\n=== Summary ===")
    keys = ["turns", "errors", "degenerate_turns", "turn3_degenerate", "retrieval_hits",
            "context_ceiling_hits", "stagnation_hits",
            "mean_first_prompt_tokens", "mean_max_prompt_tokens", "max_prompt_tokens",
            "mean_prompt_cached_frac", "cached_frac_by_turn",
            "mean_headroom", "min_headroom",
            "mean_prompt_ms", "mean_prompt_ms_t2t3",
            "mean_wall_s", "median_wall_s", "mean_wall_s_by_turn", "mean_tool_calls",
            "coding_pass", "code_scores"]
    modes = list(summary)
    col_w = max(16, max(len(m) for m in modes) + 2) if modes else 16
    print(f"{'metric':30s}" + "".join(f"{m:>{col_w}s}" for m in modes))
    per_turn_keys = {"cached_frac_by_turn": "  cached_frac_",
                      "mean_wall_s_by_turn": "  wall_s_",
                      "code_scores": "  code_"}
    for k in keys:
        if k in per_turn_keys:
            all_turns = sorted({t for m in modes for t in summary[m].get(k, {})})
            prefix = per_turn_keys[k]
            for t in all_turns:
                label = f"{prefix}{t}"
                print(f"{label:30s}" + "".join(
                    f"{str(summary[m].get(k, {}).get(t, '')):>{col_w}s}" for m in modes))
        elif k in ("coding_pass",):
            # Optional key — skip if no mode has it
            if any(k in summary[m] for m in modes):
                print(f"{k:30s}" + "".join(
                    f"{str(summary[m].get(k, '-')):>{col_w}s}" for m in modes))
        else:
            print(f"{k:30s}" + "".join(f"{str(summary[m][k]):>{col_w}s}" for m in modes))
    print(f"\nreceipt: {out}")


if __name__ == "__main__":
    main()
