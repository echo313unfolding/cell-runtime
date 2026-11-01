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
    def fake(self, model, user_input, use_tools=False, history=None, tool_registry=None):
        captured.update(dict(user_input=user_input, history=history,
                             registry_size=len(tool_registry._tools) if tool_registry else None))
        return {"content": "ok", "eval_count": 1, "eval_duration": 1,
                "prompt_tokens": 7, "first_prompt_tokens": 7,
                "timings_measured": True, "tool_calls": []}
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
    prior = [s for s in first["sources"] if s.endswith(".json")]

    follow = cc.compile_context("what were the tokens per second?", "reasoning",
                                history_summary="Q: hxq af6\nA: found it\n",
                                prior_sources=prior)
    assert any(s["type"] == "carry_forward" for s in follow["searches"])
    assert set(prior) <= set(follow["sources"])
    assert "tg128: 37.40" in follow["evidence"]


def test_follow_up_prefers_fresh_strong_hits(receipts):
    prior = [str(receipts / "hxq_af6_mmq_phase0" / "phase0_receipt.json")]
    out = cc.compile_context("hxq contribution model validation", "reasoning",
                             prior_sources=prior)
    names = [Path(s).name for s in out["sources"]]
    # fresh search scores >= 2 (hxq + contribution/model/validation) → no carry
    assert not any(s["type"] == "carry_forward" for s in out["searches"])
    assert names[0].startswith("WO_HXQ_CONTRIBUTION")


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


def test_orchestrator_threads_prior_sources_across_turns(orch_config, monkeypatch, receipts):
    from cell.orchestrator import Orchestrator
    monkeypatch.setenv("CELL_CONTEXT_COMPILER", "1")
    captured = {}
    monkeypatch.setattr(Orchestrator, "_generate", _stub_generate(captured))
    o = Orchestrator(orch_config)
    o.pool._llama_current_model = "m"
    monkeypatch.setattr(o.pool, "_llama_server_alive", lambda: True)

    r1 = o.process("hxq af6 benchmark quadro t2000", force_model="m")
    assert r1["context_compiled"]["sources"]
    assert o._prior_sources
    hist = [{"role": "user", "content": "hxq af6 benchmark"},
            {"role": "assistant", "content": "found the AF6 receipt and read it"}]
    r2 = o.process("what were the tokens per second?", force_model="m", history=hist)
    types = [s["type"] for s in r2["context_compiled"]["searches"]]
    assert "carry_forward" in types
    assert "tg128" in captured["user_input"]

    o.reset_session()
    assert o._prior_sources == []
