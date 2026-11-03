"""Context compiler unit tests — no GPU, no llama-server.

Covers the four fixes layered on 9b67b64:
  1. history trimming keeps the most recent turns, not the oldest
  2. FTS5 terms are quoted so '.' and '-' don't break the MATCH expression
  3. array-format (llama-bench) receipts emit every test row, not just [0]
  4. receipt families (run17/run14/...) are deduped so a distinct receipt
     still gets read; dir scoring ignores the RECEIPT_DIR prefix
plus the orchestrator's CELL_CONTEXT_COMPILER kill switch.
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell import context_compiler as cc


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def receipts(tmp_path, monkeypatch):
    """A fake ~/receipts tree mirroring the shapes seen in the AF6 work."""
    root = tmp_path / "receipts"
    bench = root / "zamba2_t2000_bench"
    phase0 = root / "hxq_af6_mmq_phase0"
    other = root / "wp_hxq_d4_ppl"
    for d in (bench, phase0, other):
        d.mkdir(parents=True)

    llama_bench = [
        {"gpu_info": "Quadro T2000", "model_filename": "zamba2.gguf",
         "model_type": "AF6", "model_size": 1000, "n_batch": 512,
         "test": "pp512", "n_prompt": 512, "n_gen": 0, "avg_ts": 311.2, "stddev_ts": 2.1},
        {"gpu_info": "Quadro T2000", "model_filename": "zamba2.gguf",
         "model_type": "AF6", "model_size": 1000, "n_batch": 512,
         "test": "tg128", "n_prompt": 0, "n_gen": 128, "avg_ts": 37.4, "stddev_ts": 0.3},
    ]
    for i, run in enumerate((11, 14, 17)):
        p = bench / f"bench_interleaved_hxq_af6_run{run}_20261024T164048Z.json"
        p.write_text(json.dumps(llama_bench))
        os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))  # run17 newest

    (phase0 / "phase0_receipt.json").write_text(json.dumps({
        "title": "AF6 phase 0", "verdict": "PASS",
        "model": "SmolLM3-3B", "binary": "llama-cpp-stage1",
        "baselines": ["Q5_1", "cuBLAS"],
    }))
    (other / "WO_HXQ_CONTRIBUTION_MODEL_VALIDATION_004_RECEIPT.json").write_text(
        json.dumps({"title": "contribution model", "status": "done"}))

    monkeypatch.setattr(cc, "RECEIPT_DIR", root)
    monkeypatch.setattr(cc, "MEMORY_DIR", tmp_path / "no-memory")
    monkeypatch.setattr(cc, "FGIP_DB", str(tmp_path / "no-fgip.db"))
    return root


@pytest.fixture
def fgip_db(tmp_path, monkeypatch):
    db = tmp_path / "fgip.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE VIRTUAL TABLE claims_fts USING fts5(claim_text)")
    conn.executemany("INSERT INTO claims_fts(claim_text) VALUES (?)", [
        ("qwen2.5-coder 3B AF6 build reaches 37 t/s on the T2000",),
        ("SEC filing: revenue model baseline for FY2025",),
        ("hxq af6 block layout: 144-byte deterministic block",),
    ])
    conn.commit()
    conn.close()
    monkeypatch.setattr(cc, "FGIP_DB", str(db))
    return db


# ---------------------------------------------------------------------------
# 1. history tail
# ---------------------------------------------------------------------------
def test_history_trim_keeps_most_recent_turns(receipts):
    turns = "".join(f"Q: question {i}\nA: answer {i} " + "x" * 80 + "\n" for i in range(20))
    out = cc.compile_context("nothing matches zzz", "general",
                             history_summary=turns, token_budget=1200)
    ev = out["evidence"]
    assert "[Prior context]" in ev
    assert "question 19" in ev, "newest turn must survive"
    assert "question 0" not in ev, "oldest turn must be dropped"
    assert "earlier turns dropped" in ev
    # resynced to a whole turn: the first kept line is a Q line
    body = ev.split("[Prior context]\n", 1)[1]
    first_line = body.split("\n", 2)[1] if body.startswith("...") else body.split("\n", 1)[0]
    assert first_line.startswith("Q: ")


def test_short_history_untouched(receipts):
    h = "Q: a\nA: b\n"
    out = cc.compile_context("zzz", "general", history_summary=h)
    assert "Q: a\nA: b" in out["evidence"]
    assert "dropped" not in out["evidence"]


# ---------------------------------------------------------------------------
# 2. FTS5 quoting
# ---------------------------------------------------------------------------
def test_fts_quote():
    assert cc._fts_quote("qwen2.5-coder") == '"qwen2.5-coder"'
    assert cc._fts_quote('a"b') == '"a""b"'


def test_fgip_search_survives_punctuated_keywords(fgip_db):
    # pre-fix this raised an FTS5 syntax error that was swallowed → []
    hits = cc._search_fgip(["qwen2.5-coder", "af6"])
    texts = [h["text"] for h in hits]
    assert any("qwen2.5-coder" in t for t in texts)
    assert any("144-byte" in t for t in texts)


def test_fgip_noise_filter_drops_generic_terms(fgip_db):
    # "model" and "baseline" alone are filtered; nothing left → no query
    assert cc._search_fgip(["model", "baseline"]) == []


# ---------------------------------------------------------------------------
# 3. array receipts
# ---------------------------------------------------------------------------
def test_array_receipt_emits_every_test_row(receipts):
    p = next((receipts / "zamba2_t2000_bench").glob("*run17*"))
    txt = cc._read_receipt_compact(str(p), 2000)
    assert "pp512: 311.20 ± 2.10 t/s" in txt
    assert "tg128: 37.40 ± 0.30 t/s" in txt, "tg row was dropped before the fix"
    assert txt.count("gpu_info") == 1, "header fields emitted once"


# ---------------------------------------------------------------------------
# 4. family dedupe + dir scoring
# ---------------------------------------------------------------------------
def test_family_key_collapses_runs_and_timestamps():
    a = {"name": "bench_interleaved_hxq_af6_run17_20261024T164048Z.json", "path": "/r/b/x.json"}
    b = {"name": "bench_interleaved_hxq_af6_run14_20261024T164048Z.json", "path": "/r/b/y.json"}
    c = {"name": "phase0_receipt.json", "path": "/r/b/z.json"}
    assert cc._family_key(a) == cc._family_key(b)
    assert cc._family_key(a) != cc._family_key(c)


def test_receipt_dir_prefix_not_scored(receipts):
    # "receipt" must not match the "~/receipts" prefix and inflate every file
    hits = cc._search_receipts(["receipt"])
    names = {h["name"]: h["score"] for h in hits}
    assert names["phase0_receipt.json"] == 1
    assert names["WO_HXQ_CONTRIBUTION_MODEL_VALIDATION_004_RECEIPT.json"] == 1
    assert "bench_interleaved_hxq_af6_run17_20261024T164048Z.json" not in names


def test_compile_reads_phase0_alongside_bench_family(receipts):
    kw_question = "hxq af6 benchmark receipt quadro t2000"
    out = cc.compile_context(kw_question, "reasoning", token_budget=1200)
    src_names = [Path(s).name for s in out["sources"]]
    # Both tie at score 3 here (order falls to mtime); what matters is that
    # the distinct phase0 receipt is read and only the NEWEST bench run is.
    assert "phase0_receipt.json" in src_names
    assert any(n.startswith("bench_interleaved_hxq_af6_run17") for n in src_names)
    # only one member of the run family is read
    assert sum(n.startswith("bench_interleaved") for n in src_names) == 1
    assert "tg128: 37.40" in out["evidence"]
    assert "Q5_1" in out["evidence"]
    assert out["tokens_est"] <= 1200


# ---------------------------------------------------------------------------
# orchestrator kill switch
# ---------------------------------------------------------------------------
@pytest.fixture
def orch_config(tmp_path):
    cfg = {
        "_schema": "cell_config.v1",
        "llama_server_url": "http://127.0.0.1:1",   # nothing listens; never called
        "roster": {"m": {"backend": "llama-server", "gguf": "/nonexistent.gguf",
                         "intents": ["coding", "general", "reasoning"],
                         "system_prompt": "sys", "tok_s": 10}},
        "default_model": "m",
        "swap_policy": "sticky",
        "generation": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 32},
        "logging": {"swap_log": str(tmp_path / "swap.jsonl"),
                    "task_dir": str(tmp_path / "tasks")},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def _stub_generate(captured):
    def fake(self, model, user_input, use_tools=False, history=None,
             tool_registry=None, **kwargs):
        captured.update(dict(user_input=user_input, history=history,
                             registry_size=len(tool_registry._tools) if tool_registry else None))
        return {"content": "ok", "eval_count": 1, "eval_duration": 1,
                "prompt_tokens": 7, "first_prompt_tokens": 7,
                "timings_measured": True, "tool_calls": [],
                "context_ceiling_hit": False, "stagnation": False}
    return fake


@pytest.mark.parametrize("flag,expected_mode", [("0", "legacy"), ("1", "compiler")])
def test_context_mode_switch(orch_config, monkeypatch, receipts, flag, expected_mode):
    from cell.orchestrator import Orchestrator
    monkeypatch.setenv("CELL_CONTEXT_COMPILER", flag)
    captured = {}
    monkeypatch.setattr(Orchestrator, "_generate", _stub_generate(captured))
    o = Orchestrator(orch_config)
    o.pool._llama_current_model = "m"
    monkeypatch.setattr(o.pool, "_llama_server_alive", lambda: True)
    hist = [{"role": "user", "content": "earlier q"},
            {"role": "assistant", "content": "earlier answer with enough length"}]
    res = o.process("hxq af6 t2000 benchmark", force_model="m", use_tools=True, history=hist)

    assert res["context_mode"] == expected_mode
    assert res["first_prompt_tokens"] == 7 and res["timings_measured"] is True
    task = json.load(open(res["receipt"]))
    assert task["context_mode"] == expected_mode
    assert task["generation"]["prompt_tokens"] == 7

    if expected_mode == "compiler":
        assert captured["history"] is None, "raw history must not pass through"
        assert "--- Evidence" in captured["user_input"]
        assert "earlier q" in captured["user_input"], "compressed history in evidence"
        assert captured["registry_size"] is not None
        assert captured["registry_size"] <= 6, "lean registry"
    else:
        assert captured["history"] == hist, "raw history passes through"
        assert "--- Evidence" not in captured["user_input"]
        assert captured["registry_size"] is None, "full registry (default)"
        assert res["context_compiled"]["sources"] == []


def test_openai_chat_splits_usage_and_timings(orch_config, monkeypatch):
    from cell.orchestrator import Orchestrator
    monkeypatch.setenv("CELL_CONTEXT_COMPILER", "1")
    o = Orchestrator(orch_config)
    monkeypatch.setattr(o.pool, "_http", lambda url, payload, timeout=0: {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 999, "completion_tokens": 999},
        "timings": {"prompt_n": 410, "predicted_n": 57,
                    "prompt_ms": 120.5, "predicted_ms": 1540.0},
    })
    r = o._openai_chat("http://x", [{"role": "user", "content": "q"}])
    # usage.prompt_tokens is the full context; timings.prompt_n the uncached part
    assert r["prompt_tokens"] == 999 and r["prompt_evaluated"] == 410
    assert r["eval_count"] == 57 and r["predicted_ms"] == 1540.0

    g = o._generate("m", "q")
    assert g["timings_measured"] is True
    assert g["eval_duration"] == int(1540.0 * 1e6)
    assert g["prompt_eval_duration"] == int(120.5 * 1e6)
    assert g["first_prompt_tokens"] == 999
    assert g["first_prompt_evaluated"] == 410


# ---------------------------------------------------------------------------
# 5. follow-up carry-forward of prior sources
# ---------------------------------------------------------------------------
def test_follow_up_carries_prior_receipts_when_search_is_empty(receipts):
    first = cc.compile_context("hxq af6 benchmark quadro t2000", "reasoning")
    assert first["sources"]
    assert first["excerpts"]  # excerpts returned for next-turn carry
    prior_excerpts = first["excerpts"]

    follow = cc.compile_context("what were the tokens per second?", "reasoning",
                                history_summary="Q: hxq af6\nA: found it\n",
                                prior_excerpts=prior_excerpts)
    assert any(s["type"] == "carry_forward" for s in follow["searches"])
    prior_paths = [e["path"] for e in prior_excerpts]
    assert set(prior_paths) <= set(follow["sources"])
    assert "tg128: 37.40" in follow["evidence"]


def test_follow_up_reserves_one_slot_with_strong_fresh_hits(receipts):
    # Prior source is phase0 receipt.  Follow-up query uses keywords that
    # match the WO_HXQ_CONTRIBUTION file at score ≥2 but NOT phase0 (no
    # "hxq", "af6", "phase0" → phase0 scores 0 in fresh search).
    phase0_path = str(receipts / "hxq_af6_mmq_phase0" / "phase0_receipt.json")
    prior_excerpts = [{"path": phase0_path,
                       "name": "phase0_receipt.json",
                       "excerpt": "File: phase0_receipt.json\ntitle: AF6 phase 0"}]
    out = cc.compile_context("contribution model validation gate", "reasoning",
                             prior_excerpts=prior_excerpts)
    names = [Path(s).name for s in out["sources"]]
    # Fresh search found WO_HXQ at ≥2 (strong), but carry-forward still
    # reserves one slot for the prior receipt
    carry_searches = [s for s in out["searches"] if s["type"] == "carry_forward"]
    assert len(carry_searches) == 1
    assert carry_searches[0]["hits"] == 1  # only one slot reserved
    assert "phase0_receipt.json" in names
    # Fresh hit is still present
    assert any(n.startswith("WO_HXQ_CONTRIBUTION") for n in names)


def test_no_carry_forward_without_prior(receipts):
    out = cc.compile_context("what were the tokens per second?", "reasoning")
    assert not any(s["type"] == "carry_forward" for s in out["searches"])


# ---------------------------------------------------------------------------
# 6. prompt accounting: full context from usage, cache split from timings
# ---------------------------------------------------------------------------
def test_prompt_tokens_use_full_context_not_uncached_slice(orch_config, monkeypatch):
    from cell.orchestrator import Orchestrator
    monkeypatch.setenv("CELL_CONTEXT_COMPILER", "1")
    o = Orchestrator(orch_config)
    monkeypatch.setattr(o.pool, "_http", lambda url, payload, timeout=0: {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 1582, "completion_tokens": 50},
        # system prompt cached: llama-server only evaluated the question
        "timings": {"prompt_n": 31, "predicted_n": 50,
                    "prompt_ms": 40.0, "predicted_ms": 900.0},
    })
    r = o._openai_chat("http://x", [{"role": "user", "content": "q"}])
    assert r["prompt_tokens"] == 1582
    assert r["prompt_evaluated"] == 31
    assert r["prompt_cached"] == 1551
    g = o._generate("m", "q")
    assert g["first_prompt_tokens"] == 1582
    assert g["first_prompt_evaluated"] == 31
    assert g["max_prompt_tokens"] == 1582


def test_orchestrator_threads_prior_excerpts_across_turns(orch_config, monkeypatch, receipts):
    from cell.orchestrator import Orchestrator
    monkeypatch.setenv("CELL_CONTEXT_COMPILER", "1")
    captured = {}
    monkeypatch.setattr(Orchestrator, "_generate", _stub_generate(captured))
    o = Orchestrator(orch_config)
    o.pool._llama_current_model = "m"
    monkeypatch.setattr(o.pool, "_llama_server_alive", lambda: True)

    r1 = o.process("hxq af6 benchmark quadro t2000", force_model="m")
    assert r1["context_compiled"]["sources"]
    assert o._prior_excerpts
    assert all("excerpt" in e for e in o._prior_excerpts)
    hist = [{"role": "user", "content": "hxq af6 benchmark"},
            {"role": "assistant", "content": "found the AF6 receipt and read it"}]
    r2 = o.process("what were the tokens per second?", force_model="m", history=hist)
    types = [s["type"] for s in r2["context_compiled"]["searches"]]
    assert "carry_forward" in types
    assert "tg128" in captured["user_input"]

    o.reset_session()
    assert o._prior_excerpts == []


# ---------------------------------------------------------------------------
# 7. tool-loop guards: context ceiling & stagnation
# ---------------------------------------------------------------------------
def _make_tool_loop_orch(orch_config, monkeypatch, responses):
    """Set up an Orchestrator whose _openai_chat returns canned responses
    in sequence.  Each response is a dict with at least 'content'.
    Tokens are set in usage so ceiling math works.
    """
    from cell.orchestrator import Orchestrator
    monkeypatch.setenv("CELL_CONTEXT_COMPILER", "1")
    o = Orchestrator(orch_config)
    o.pool._llama_current_model = "m"
    monkeypatch.setattr(o.pool, "_llama_server_alive", lambda: True)

    call_idx = [0]
    def fake_http(url, payload, timeout=0):
        idx = call_idx[0]
        call_idx[0] += 1
        resp = responses[min(idx, len(responses) - 1)]
        pt = resp.get("prompt_tokens", 100)
        ev = resp.get("eval_tokens", 10)
        return {
            "choices": [{"message": {"content": resp["content"]}}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ev},
            "timings": {"prompt_n": pt, "predicted_n": ev,
                        "prompt_ms": 10.0, "predicted_ms": 50.0},
        }
    monkeypatch.setattr(o.pool, "_http", fake_http)

    # Stub _tokenize to use char/2.5 (no real server)
    monkeypatch.setattr(o, "_tokenize", lambda text: int(len(text) / 2.5))

    return o


def test_stagnation_repeated_call_triggers_force_commit(orch_config, monkeypatch):
    """When the model calls the same tool with the same args twice, _generate
    should force-commit instead of looping."""
    tool_call = '```tool_call\n{"name": "grep", "arguments": {"pattern": "af6", "path": "/tmp"}}\n```'
    responses = [
        # Turn 0: model emits a tool call
        {"content": tool_call, "prompt_tokens": 500, "eval_tokens": 20},
        # Turn 1: model emits the SAME tool call again (stagnation)
        {"content": tool_call, "prompt_tokens": 700, "eval_tokens": 20},
        # Turn 2: force-commit answer call
        {"content": "The AF6 benchmark showed good results.", "prompt_tokens": 750, "eval_tokens": 30},
    ]
    o = _make_tool_loop_orch(orch_config, monkeypatch, responses)
    # Need a tool registry that can execute "grep"
    from cell.tool_registry import ToolRegistry
    reg = ToolRegistry()
    reg.register("grep", lambda pattern="", path="": "some grep result", "search files")
    result = o._generate("m", "test query", use_tools=True, tool_registry=reg)
    assert result["stagnation"] is True
    assert result["force_commit"] == "stagnation_repeated_call"
    # The final content should be the force-commit answer, not the tool fence
    assert "AF6" in result["content"] or "context" in result["content"].lower()


def test_no_new_info_repeated_result_triggers_force_commit(orch_config, monkeypatch):
    """When two different tool calls return identical results, that's no-new-info."""
    call_a = '```tool_call\n{"name": "grep", "arguments": {"pattern": "af6", "path": "/a"}}\n```'
    call_b = '```tool_call\n{"name": "grep", "arguments": {"pattern": "af6", "path": "/b"}}\n```'
    responses = [
        {"content": call_a, "prompt_tokens": 500, "eval_tokens": 20},
        {"content": call_b, "prompt_tokens": 700, "eval_tokens": 20},
        # force-commit answer
        {"content": "Based on available data, the results are...", "prompt_tokens": 750, "eval_tokens": 30},
    ]
    o = _make_tool_loop_orch(orch_config, monkeypatch, responses)
    from cell.tool_registry import ToolRegistry
    reg = ToolRegistry()
    # Both calls return the exact same result → no_new_info (must be ≥40 chars)
    reg.register("grep", lambda pattern="", path="": "identical result text that is long enough to trigger the hash check", "search files")
    result = o._generate("m", "test query", use_tools=True, tool_registry=reg)
    assert result["stagnation"] is True
    assert "no_new_info" in result["force_commit"]


def test_context_ceiling_truncates_result(orch_config, monkeypatch):
    """When a tool result would push past n_ctx, it gets truncated."""
    tool_call = '```tool_call\n{"name": "read_file", "arguments": {"path": "/tmp/big.json"}}\n```'
    responses = [
        # First call reports high prompt usage, leaving little room
        {"content": tool_call, "prompt_tokens": 3500, "eval_tokens": 20},
        # After truncated result is appended, model answers
        {"content": "The file contained benchmark data.", "prompt_tokens": 3700, "eval_tokens": 30},
    ]
    o = _make_tool_loop_orch(orch_config, monkeypatch, responses)
    from cell.tool_registry import ToolRegistry
    reg = ToolRegistry()
    # Return a huge result that would overflow n_ctx=4096
    reg.register("read_file", lambda path="": "x" * 5000, "read a file")
    result = o._generate("m", "read big file", use_tools=True, tool_registry=reg)
    # Should NOT have errored out — the result was truncated to fit
    assert result["content"] == "The file contained benchmark data."
    # Receipt stores the FULL result; preview metadata records what model saw
    entry = result["tool_calls"][0]
    assert len(entry["result"]) == 5000  # full result preserved
    assert entry["result_bytes"] == 5000
    assert "model_preview_chars" in entry  # truncation recorded
    assert entry["model_preview_chars"] < 5000


def test_context_ceiling_force_commit_when_no_room(orch_config, monkeypatch):
    """When there's no room even for a truncated result, force-commit fires."""
    tool_call = '```tool_call\n{"name": "read_file", "arguments": {"path": "/tmp/big.json"}}\n```'
    responses = [
        # Almost full context — only ~100 tokens left
        {"content": tool_call, "prompt_tokens": 3900, "eval_tokens": 20},
        # force-commit answer
        {"content": "I cannot read more files.", "prompt_tokens": 3950, "eval_tokens": 15},
    ]
    o = _make_tool_loop_orch(orch_config, monkeypatch, responses)
    from cell.tool_registry import ToolRegistry
    reg = ToolRegistry()
    reg.register("read_file", lambda path="": "x" * 3000, "read a file")
    result = o._generate("m", "read big file", use_tools=True, tool_registry=reg)
    assert result["context_ceiling_hit"] is True
    assert "ceiling" in result.get("force_commit", "")


def test_multi_call_fence_ceiling_catches_accumulated_results(orch_config, monkeypatch):
    """Three tool calls in one assistant turn, each returning 1200 chars.
    Starting at pt=3300, the ceiling must fire before all three are appended.
    n_ctx=4096 - margin(128) = 3968 usable. pt=3300 + fence(~30) + 3*480tok
    would be ~4770 — the loop must truncate or force-commit before that."""
    three_calls = (
        '```tool_call\n{"name": "read_file", "arguments": {"path": "/a"}}\n```\n'
        '```tool_call\n{"name": "read_file", "arguments": {"path": "/b"}}\n```\n'
        '```tool_call\n{"name": "read_file", "arguments": {"path": "/c"}}\n```'
    )
    responses = [
        # Model emits 3 tool calls at once, context already at 3300
        {"content": three_calls, "prompt_tokens": 3300, "eval_tokens": 50},
        # Force-commit answer (if ceiling fires)
        {"content": "Here is a summary from the files.", "prompt_tokens": 3500, "eval_tokens": 30},
    ]
    o = _make_tool_loop_orch(orch_config, monkeypatch, responses)
    from cell.tool_registry import ToolRegistry
    reg = ToolRegistry()
    # Each call returns DIFFERENT content so no-new-info doesn't fire first
    call_counter = [0]
    def read_file_handler(path=""):
        call_counter[0] += 1
        return chr(64 + call_counter[0]) * 1200   # "AAA...", "BBB...", "CCC..."
    reg.register("read_file", read_file_handler, "read a file")
    result = o._generate("m", "read three files", use_tools=True, tool_registry=reg)
    # The loop must have hit the ceiling — 3300 + fence + 3×480 > 4096-128
    assert result["context_ceiling_hit"] is True
    assert "ceiling" in result.get("force_commit", "")


def test_short_result_no_new_info_not_triggered(orch_config, monkeypatch):
    """Short identical results (empty grep, empty dir) should NOT trigger
    no-new-info — call-hash stagnation covers true loops."""
    call_a = '```tool_call\n{"name": "grep", "arguments": {"pattern": "xyz", "path": "/a"}}\n```'
    call_b = '```tool_call\n{"name": "grep", "arguments": {"pattern": "xyz", "path": "/b"}}\n```'
    responses = [
        {"content": call_a, "prompt_tokens": 500, "eval_tokens": 20},
        {"content": call_b, "prompt_tokens": 700, "eval_tokens": 20},
        # Model answers normally (no force-commit)
        {"content": "No matches found in either directory.", "prompt_tokens": 800, "eval_tokens": 25},
    ]
    o = _make_tool_loop_orch(orch_config, monkeypatch, responses)
    from cell.tool_registry import ToolRegistry
    reg = ToolRegistry()
    # Short result — "no matches" (< 40 chars)
    reg.register("grep", lambda pattern="", path="": "no matches", "search files")
    result = o._generate("m", "search both dirs", use_tools=True, tool_registry=reg)
    # Should NOT have triggered stagnation — results are short
    assert result["stagnation"] is False
    assert result["content"] == "No matches found in either directory."


def test_error_results_excluded_from_no_new_info(orch_config, monkeypatch):
    """Identical error strings from different calls should NOT trigger no-new-info.
    The model is hitting a constraint, not getting no new information."""
    call_a = '```tool_call\n{"name": "grep", "arguments": {"pattern": "cron", "path": "/home/user"}}\n```'
    call_b = '```tool_call\n{"name": "grep", "arguments": {"pattern": "cron", "path": "/"}}\n```'
    responses = [
        {"content": call_a, "prompt_tokens": 500, "eval_tokens": 20},
        {"content": call_b, "prompt_tokens": 700, "eval_tokens": 20},
        {"content": "I could not find the cron job.", "prompt_tokens": 800, "eval_tokens": 25},
    ]
    o = _make_tool_loop_orch(orch_config, monkeypatch, responses)
    from cell.tool_registry import ToolRegistry
    reg = ToolRegistry()
    # Both return identical errors (≥40 chars) — should NOT trigger no_new_info
    reg.register("grep", lambda pattern="", path="": "Error: path must be under /home/voidstr3m33", "search files")
    result = o._generate("m", "find cron jobs", use_tools=True, tool_registry=reg)
    assert result["stagnation"] is False
    assert result["content"] == "I could not find the cron job."


def test_carry_forward_skips_when_prior_already_in_fresh_results(receipts):
    """When prior source is already found by fresh search, carry doesn't duplicate."""
    phase0_path = str(receipts / "hxq_af6_mmq_phase0" / "phase0_receipt.json")
    prior_excerpts = [{"path": phase0_path,
                       "name": "phase0_receipt.json",
                       "excerpt": "File: phase0_receipt.json\ntitle: AF6 phase 0"}]
    # "hxq" in query → phase0's dir scores 1 → phase0 already in fresh results
    out = cc.compile_context("hxq contribution model validation", "reasoning",
                             prior_excerpts=prior_excerpts)
    # No carry needed — prior is already found by fresh search
    assert not any(s["type"] == "carry_forward" for s in out["searches"])
    # But the prior source still appears (via fresh search, not carry)
    assert any("phase0" in s for s in out["sources"])


def test_carried_excerpt_is_byte_identical_across_turns(receipts):
    """The whole point of storing excerpts: the carried receipt text in T2's
    evidence block is byte-identical to T1's, so llama-server's prefix cache
    covers it."""
    t1 = cc.compile_context("hxq af6 benchmark quadro t2000", "reasoning")
    assert t1["excerpts"]
    # The evidence block for T1 starts with the receipt
    t1_prefix = t1["evidence"].split("\n\n")[0]  # first receipt block

    # T2: carry T1's excerpts forward (weak fresh search)
    t2 = cc.compile_context("what were the tokens per second?", "reasoning",
                            history_summary="Q: hxq af6\nA: found it\n",
                            prior_excerpts=t1["excerpts"])
    t2_prefix = t2["evidence"].split("\n\n")[0]  # first receipt block

    assert t1_prefix == t2_prefix, (
        f"Carried excerpt prefix differs!\nT1: {t1_prefix[:200]}\nT2: {t2_prefix[:200]}")


def test_evidence_hint_appended_when_compiler_has_sources(orch_config, monkeypatch, receipts):
    """When context compiler delivers evidence, the augmented input includes
    a hint telling the model not to re-search."""
    from cell.orchestrator import Orchestrator
    monkeypatch.setenv("CELL_CONTEXT_COMPILER", "1")
    captured = {}
    monkeypatch.setattr(Orchestrator, "_generate", _stub_generate(captured))
    o = Orchestrator(orch_config)
    o.pool._llama_current_model = "m"
    monkeypatch.setattr(o.pool, "_llama_server_alive", lambda: True)
    res = o.process("hxq af6 benchmark quadro t2000", force_model="m", use_tools=True)
    assert "Evidence above was retrieved for you" in captured["user_input"]
    assert "call tools only if" in captured["user_input"]
