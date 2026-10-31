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
import collections
import json
import os
import signal
import subprocess
import sys
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


def _start_server(config: dict, model: str) -> subprocess.Popen:
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
    proc = subprocess.Popen(
        [binary, "--model", gguf, "--port", port,
         "--ctx-size", str(gen.get("num_ctx", 4096)),
         "--n-gpu-layers", "99", "--host", "127.0.0.1"],
        stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env,
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
    os.environ["CELL_CONTEXT_COMPILER"] = "1" if mode == "compiler" else "0"
    from cell.orchestrator import Orchestrator  # import after env set is fine; flag read in __init__

    orch = Orchestrator(config_path)
    model = orch.default_model
    # Adopt the already-running server so sticky policy never tries to swap.
    if orch.pool._llama_server_alive():
        orch.pool._llama_current_model = model
    n_ctx = int(orch.gen_opts.get("num_ctx", 4096))

    rows = []
    for sc in scenarios:
        orch.memory.reset()
        history: list[dict] = []
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
            first_pt = res.get("first_prompt_tokens", 0)
            evalc = res.get("eval_count", 0)
            row = {
                "mode": mode,
                "scenario": sc["id"],
                "turn": ti,
                "intent": res.get("intent"),
                "error": err or res.get("error"),
                "timings_measured": res.get("timings_measured", False),
                "first_prompt_tokens": first_pt,
                "prompt_tokens_total": res.get("prompt_tokens", 0),
                "eval_tokens": evalc,
                "headroom": (n_ctx - first_pt - evalc) if first_pt else None,
                "tok_s": res.get("tok_s", 0),
                "wall_s": wall,
                "tool_calls": res.get("tool_calls", 0),
                "retrieval_hit": _hit(sc["gold"], res) if not err else False,
                "output_chars": len(out),
                "output_head": out[:160],
                "evidence_tokens_est": res.get("context_compiled", {}).get("tokens_est", 0),
                "sources": res.get("context_compiled", {}).get("sources", []),
                "receipt": res.get("receipt"),
                **deg,
            }
            rows.append(row)
            print(f"[{mode:8s}] {sc['id']:26s} T{ti} "
                  f"pt={first_pt:5d} ev={evalc:4d} head={row['headroom']} "
                  f"deg={deg['degenerate']} hit={row['retrieval_hit']} "
                  f"wall={wall}s {'ERR ' + err if err else ''}", flush=True)
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
            "mean_headroom": round(
                sum(r["headroom"] for r in measured) / len(measured), 1) if measured else None,
            "min_headroom": min((r["headroom"] for r in measured), default=None),
            "mean_wall_s": round(sum(r["wall_s"] for r in rs) / len(rs), 2),
            "mean_tool_calls": round(sum(r["tool_calls"] for r in rs) / len(rs), 2),
            "turn3_degenerate": sum(1 for r in rs if r["turn"] == 3 and r["degenerate"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE.parent / "src" / "cell" / "config.json"))
    ap.add_argument("--modes", default="legacy,compiler")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--tools", choices=["on", "off"], default="on")
    ap.add_argument("--prompts", help="JSON file: list of {id, gold, turns}")
    ap.add_argument("--start-server", action="store_true",
                    help="launch llama-server from config; killed on exit")
    ap.add_argument("--out", help="receipt path (default ~/receipts/cell/echo_context_compiler_ab_v2_<ts>.json)")
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
        proc = _start_server(config, config["default_model"])
    elif not _health(url):
        sys.exit(f"llama-server not healthy at {url}; pass --start-server or start it yourself")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rows: list[dict] = []
    try:
        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            rows.extend(run_mode(mode, args.config, scenarios,
                                 use_tools=(args.tools == "on"), depth=args.depth))
    finally:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    summary = summarize(rows)
    receipt = {
        "receipt_id": f"echo_context_compiler_ab_v2_{stamp}",
        "title": "Context compiler A/B — legacy vs compiler on the same orchestrator/binary/model",
        "timestamp": stamp,
        "config": args.config,
        "llama_server_url": url,
        "model": config.get("default_model"),
        "n_ctx": config.get("generation", {}).get("num_ctx"),
        "num_predict": config.get("generation", {}).get("num_predict"),
        "use_tools": args.tools == "on",
        "depth": args.depth,
        "scenarios": [{"id": s["id"], "gold": s["gold"], "n_turns": len(s["turns"])} for s in scenarios],
        "metric_definitions": {
            "first_prompt_tokens": "llama-server timings.prompt_n of the first call in the turn (= context cost before any tool loop)",
            "headroom": "n_ctx - first_prompt_tokens - eval_tokens",
            "degenerate": "tail_unique<8 over last 120 chars (len>=40) OR top 4-gram covers >40% of output",
            "retrieval_hit": "any gold fragment in context_compiled.sources, tool call args/results, or output",
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
            "mean_first_prompt_tokens", "mean_headroom", "min_headroom", "mean_wall_s", "mean_tool_calls"]
    modes = list(summary)
    print(f"{'metric':26s}" + "".join(f"{m:>14s}" for m in modes))
    for k in keys:
        print(f"{k:26s}" + "".join(f"{str(summary[m][k]):>14s}" for m in modes))
    print(f"\nreceipt: {out}")


if __name__ == "__main__":
    main()
