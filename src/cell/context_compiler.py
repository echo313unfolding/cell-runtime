"""Context compiler — CPU-side evidence retrieval and prompt assembly.

Sits between the intent classifier and the LLM. Searches receipts,
memory, and the FGIP graph directly on the CPU. Builds a compact
evidence packet that fits within a token budget.

The model receives the question plus relevant evidence — not the
search process.
"""
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Optional


# Rough approximation: 1 token ≈ 4 characters.
# NOTE: receipt JSON (paths, numbers, underscores) tokenizes closer to ~3
# chars/token on Qwen2.5. tokens_est is a lower bound; measure the real
# prompt_n from llama-server's `timings` when it matters.
CHARS_PER_TOKEN = 4

# How many receipt candidates the search returns, and how many distinct
# receipt families get read into the evidence block.
RECEIPT_CANDIDATES = 10
MAX_RECEIPTS = 3
# A fresh search whose best hit scores below this is treated as weak and the
# previous turn's receipts are read first. Score 1 = one keyword matched
# somewhere in the path, which "hxq" alone does for most of ~/receipts.
CARRY_FORWARD_BELOW = 2

# Search paths
RECEIPT_DIR = Path(os.path.expanduser("~/receipts"))
MEMORY_DIR = Path(os.path.expanduser(
    "~/.claude/projects/-home-voidstr3m33/memory"))
FGIP_DB = os.path.expanduser("~/fgip-engine/fgip.db")


def compile_context(question: str, intent: str,
                    history_summary: str = "",
                    token_budget: int = 1200,
                    prior_sources: Optional[list[str]] = None) -> dict:
    """Search for evidence relevant to the question and build a compact context.

    prior_sources: receipt paths read on the previous turn. Used only as a
        fallback when this turn's own search is empty or weak (top score
        < CARRY_FORWARD_BELOW), which is what a follow-up question looks
        like ("what were the tokens per second?" has no receipt keywords).

    Returns:
        {
            "evidence": str,       # formatted evidence block for the prompt
            "sources": list[str],  # file paths / sources used
            "tokens_est": int,     # estimated token count
            "searches": list[dict] # what was searched and found
        }
    """
    char_budget = token_budget * CHARS_PER_TOKEN
    searches = []
    evidence_parts = []
    sources = []

    # Extract search terms from the question
    keywords = _extract_keywords(question)

    # 1. Search receipts by filename
    receipt_hits = _search_receipts(keywords)
    searches.append({"type": "receipts", "keywords": keywords,
                     "hits": len(receipt_hits)})

    # 1b. Follow-up fallback: carry the previous turn's receipts forward
    #     when the fresh search found nothing worth reading.
    if prior_sources and (not receipt_hits
                          or receipt_hits[0]["score"] < CARRY_FORWARD_BELOW):
        seen = {h["path"] for h in receipt_hits}
        carried = []
        for p in prior_sources:
            if p in seen or not os.path.isfile(p):
                continue
            carried.append({"path": p, "name": os.path.basename(p),
                            "score": 0, "size": os.path.getsize(p),
                            "carried": True})
        if carried:
            receipt_hits = carried + receipt_hits
            searches.append({"type": "carry_forward", "hits": len(carried)})

    # 2. Search FGIP claims if DB exists
    fgip_hits = _search_fgip(keywords)
    if fgip_hits:
        searches.append({"type": "fgip", "hits": len(fgip_hits)})

    # 3. Search memory files
    memory_hits = _search_memory(keywords)
    if memory_hits:
        searches.append({"type": "memory", "hits": len(memory_hits)})

    # Budget allocation: receipts get 60%, FGIP 20%, memory 20%
    receipt_budget = int(char_budget * 0.6)
    fgip_budget = int(char_budget * 0.2)
    memory_budget = int(char_budget * 0.2)

    # 4. Read and extract from the top receipt matches.
    #    Files in the same family (bench_*_run17, run14, ...) are collapsed to
    #    their newest member first, so a run series can't crowd out a
    #    lower-scoring but distinct receipt such as phase0_receipt.json.
    if receipt_hits:
        picks = _dedupe_families(receipt_hits)[:MAX_RECEIPTS]
        remaining = receipt_budget
        for i, hit in enumerate(picks):
            share = remaining // (len(picks) - i)
            excerpt = _read_receipt_compact(hit["path"], share)
            if excerpt:
                evidence_parts.append(f"[Receipt: {hit['name']}]\n{excerpt}")
                sources.append(hit["path"])
                remaining -= len(excerpt)
            if remaining < 200:
                break

    # 5. Add FGIP claims
    if fgip_hits:
        fgip_text = "\n".join(
            f"- {h['text']}" for h in fgip_hits[:5]
        )[:fgip_budget]
        if fgip_text:
            evidence_parts.append(f"[Graph claims]\n{fgip_text}")
            sources.append("fgip.db")

    # 6. Add memory context
    if memory_hits:
        mem_text = "\n".join(
            f"- {h['file']}: {h['line']}" for h in memory_hits[:5]
        )[:memory_budget]
        if mem_text:
            evidence_parts.append(f"[Memory]\n{mem_text}")
            sources.append("memory")

    # 7. Assemble
    evidence = "\n\n".join(evidence_parts) if evidence_parts else ""

    # 8. Add history summary if present.
    #    history_summary is chronological (oldest turn first). When it exceeds
    #    the cap we must keep the TAIL — the most recent turns — and drop the
    #    oldest. Cutting at a line boundary so no half-turn leaks in.
    if history_summary:
        history_cap = int(char_budget * 0.15)
        kept = history_summary.rstrip("\n")
        if len(kept) > history_cap:
            kept = kept[-history_cap:]
            # Resync to the start of a whole turn ("Q: ..." line) if possible,
            # otherwise to the next line boundary.
            cut = kept.find("\nQ: ")
            if cut < 0:
                cut = kept.find("\n")
            if 0 <= cut < len(kept) - 1:
                kept = kept[cut + 1:]
            kept = "... (earlier turns dropped)\n" + kept
        evidence = f"[Prior context]\n{kept}\n\n{evidence}"

    # Trim to budget
    if len(evidence) > char_budget:
        evidence = evidence[:char_budget - 20] + "\n... (trimmed)"

    tokens_est = len(evidence) // CHARS_PER_TOKEN

    return {
        "evidence": evidence,
        "sources": sources,
        "tokens_est": tokens_est,
        "searches": searches,
    }


def summarize_turn(question: str, answer: str, max_chars: int = 300) -> str:
    """Produce a compact summary of a completed turn for history injection.

    Pure CPU — no LLM. Extracts the key facts from the answer.
    """
    # Take the first meaningful sentence(s) from the answer
    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    # Skip code fences and tool call blocks
    content_lines = []
    in_block = False
    for line in lines:
        if line.startswith("```"):
            in_block = not in_block
            continue
        if not in_block and len(line) > 10:
            content_lines.append(line)

    summary = " ".join(content_lines)[:max_chars]
    if len(summary) < len(" ".join(content_lines)):
        summary = summary.rsplit(" ", 1)[0] + "..."

    return f"Q: {question[:100]}\nA: {summary}" if summary else ""


# ── Internal search functions ──


def _extract_keywords(text: str) -> list[str]:
    """Extract searchable keywords from a question. No LLM needed."""
    # Remove common question words
    stop = {"the", "a", "an", "is", "was", "were", "are", "what", "which",
            "how", "does", "do", "did", "can", "could", "would", "should",
            "tell", "me", "about", "find", "our", "my", "your", "latest",
            "most", "recent", "and", "or", "in", "on", "for", "to", "of",
            "with", "from", "that", "this", "it", "its", "use", "using",
            "based", "identify", "report", "measured", "actual", "file",
            "been", "have", "has", "had", "we", "they", "them"}

    # Tokenize and filter
    words = re.findall(r'[a-zA-Z0-9_.\-/~]+', text.lower())
    # Strip trailing punctuation that leaks from sentence boundaries
    words = [w.rstrip('.,:;!?') for w in words]
    keywords = [w for w in words if w not in stop and len(w) > 1]

    # Deduplicate preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)

    return unique[:15]


def _search_receipts(keywords: list[str]) -> list[dict]:
    """Search receipt files by name and content. Returns sorted by relevance."""
    if not RECEIPT_DIR.exists():
        return []

    # Score each receipt file
    scored = []
    for f in RECEIPT_DIR.rglob("*.json"):
        name_lower = f.name.lower()
        # Count keyword matches in filename
        score = sum(1 for kw in keywords if kw in name_lower)
        # Directory components score equally — phase0_receipt.json under
        # hxq_af6_mmq_phase0/ should rank as high as a file with af6 in its name.
        # Only the path *below* RECEIPT_DIR counts, so a keyword like "receipt"
        # doesn't match "/home/.../receipts" and add +1 to every file.
        try:
            dir_lower = str(f.parent.relative_to(RECEIPT_DIR)).lower()
        except ValueError:
            dir_lower = str(f.parent).lower()
        if dir_lower == ".":
            dir_lower = ""
        score += sum(1 for kw in keywords if kw in dir_lower
                     and kw not in name_lower)
        if score > 0:
            scored.append({
                "path": str(f),
                "name": f.name,
                "score": score,
                "size": f.stat().st_size,
            })

    # Sort by score descending, then by modification time (newest first)
    scored.sort(key=lambda x: (-x["score"], -os.path.getmtime(x["path"])))
    return scored[:RECEIPT_CANDIDATES]


_FAMILY_RUN = re.compile(r"_?run\d+", re.IGNORECASE)
_FAMILY_TS = re.compile(r"_?\d{8}t\d{6}z?|_?\d{8}(?:[-_]\d{4,6})?", re.IGNORECASE)


def _family_key(hit: dict) -> str:
    """Collapse run numbers and timestamps so a series of bench runs counts
    as one family. bench_interleaved_hxq_af6_run17_20261024T164048Z.json and
    ..._run14_... share a key; phase0_receipt.json is its own family."""
    stem = Path(hit["name"]).stem.lower()
    stem = _FAMILY_RUN.sub("", stem)
    stem = _FAMILY_TS.sub("", stem)
    parent = str(Path(hit["path"]).parent)
    return f"{parent}/{stem}"


def _dedupe_families(hits: list[dict]) -> list[dict]:
    """Keep the first (highest score, newest) member of each family, in order."""
    seen = set()
    out = []
    for h in hits:
        key = _family_key(h)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _read_receipt_compact(path: str, char_budget: int) -> Optional[str]:
    """Read a receipt and extract the most relevant fields compactly."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Handle array-format receipts (e.g., llama-bench output).
    # llama-bench emits one entry per test (pp512, tg128, ...). The header
    # fields are shared, so emit them once from entry[0], then one line per
    # entry with its test name and throughput — the tg row is the number
    # people actually ask for and was previously dropped.
    if isinstance(data, list):
        if not data:
            return None
        entries = [e for e in data if isinstance(e, dict)]
        if not entries:
            return None
        first = entries[0]
        parts = [f"File: {os.path.basename(path)} (array, {len(data)} entries)"]
        header_keys = ["gpu_info", "model_filename", "model_type", "model_size",
                       "n_batch", "n_gpu_layers"]
        used = len(parts[0])
        for k in header_keys:
            if k in first:
                line = f"{k}: {first[k]}"
                if used + len(line) + 2 < char_budget:
                    parts.append(line)
                    used += len(line) + 2
        for e in entries:
            test = e.get("test") or f"pp{e.get('n_prompt', '?')}/tg{e.get('n_gen', '?')}"
            avg = e.get("avg_ts")
            std = e.get("stddev_ts")
            if avg is None:
                line = f"{test}: (no avg_ts)"
            elif std is not None:
                line = f"{test}: {avg:.2f} ± {std:.2f} t/s"
            else:
                line = f"{test}: {avg:.2f} t/s"
            if used + len(line) + 2 < char_budget:
                parts.append(line)
                used += len(line) + 2
            else:
                break
        return "\n".join(parts)

    # Priority fields — these carry the most information
    priority_keys = [
        "title", "gate", "receipt_id", "verdict", "status",
        "question", "overall", "summary", "interpretation",
        "environment", "measurements", "results", "benchmarks",
        "model", "binary", "artifacts",
        "conclusions", "correction_note",
    ]

    parts = [f"File: {os.path.basename(path)}"]

    # Add priority fields first
    used_budget = len(parts[0])
    for key in priority_keys:
        if key in data:
            val = data[key]
            formatted = _format_value(key, val)
            if used_budget + len(formatted) + 2 < char_budget:
                parts.append(formatted)
                used_budget += len(formatted) + 2

    # Then add remaining top-level keys (scalars only) if budget allows
    for key, val in data.items():
        if key in priority_keys or key == "cost":
            continue
        # Short lists of strings (baselines, formats, tags) are cheap and
        # often the exact thing being asked for; previously dropped.
        if (isinstance(val, list) and 0 < len(val) <= 10
                and all(isinstance(x, str) for x in val)):
            val = ", ".join(val)
        if isinstance(val, (str, int, float, bool)) and val:
            line = f"{key}: {val}"
            if len(line) > 200:
                line = line[:200] + "..."
            if used_budget + len(line) + 2 < char_budget:
                parts.append(line)
                used_budget += len(line) + 2

    # Always include cost if present (per WO-RECEIPT-COST-01)
    if "cost" in data:
        cost_line = f"cost: {json.dumps(data['cost'])}"
        if len(cost_line) > 300:
            cost_line = cost_line[:300] + "..."
        if used_budget + len(cost_line) < char_budget:
            parts.append(cost_line)

    return "\n".join(parts)


def _format_value(key: str, val) -> str:
    """Format a receipt field value compactly."""
    if isinstance(val, dict):
        # Flatten dict to key: value pairs
        lines = []
        for k, v in val.items():
            if isinstance(v, (dict, list)):
                v_str = json.dumps(v)
                if len(v_str) > 150:
                    v_str = v_str[:150] + "..."
            else:
                v_str = str(v)
            lines.append(f"  {k}: {v_str}")
        return f"{key}:\n" + "\n".join(lines)
    elif isinstance(val, list):
        if all(isinstance(x, str) for x in val):
            return f"{key}: {', '.join(val[:10])}"
        items = json.dumps(val)
        if len(items) > 300:
            items = items[:300] + "..."
        return f"{key}: {items}"
    else:
        s = str(val)
        if len(s) > 200:
            s = s[:200] + "..."
        return f"{key}: {s}"


def _fts_quote(term: str) -> str:
    """Quote a term for an FTS5 MATCH expression (doubling embedded quotes)."""
    return '"' + term.replace('"', '""') + '"'


def _search_fgip(keywords: list[str]) -> list[dict]:
    """Search FGIP claims via FTS5."""
    if not os.path.exists(FGIP_DB):
        return []

    # Filter out generic words that match the entire financial corpus.
    # Only send domain-specific terms to FTS5.
    fgip_noise = {"model", "benchmark", "baseline", "performance", "binary",
                  "measured", "format", "result", "report", "test", "data",
                  "system", "server", "build", "code", "status", "receipt",
                  "kernel", "output", "input", "config", "path", "file"}
    domain_kw = [kw for kw in keywords[:8]
                 if kw not in fgip_noise and re.match(r'^[a-zA-Z0-9_.\-]+$', kw)]
    # FTS5 barewords may not contain '.' or '-' ('-' is NOT, '.' is not a
    # token char), so "qwen2.5-coder OR af6" is a syntax error that the
    # except below would swallow silently. Quote every term; FTS5 then
    # tokenizes the quoted string as a phrase.
    fts_query = " OR ".join(_fts_quote(kw) for kw in domain_kw)
    if not fts_query:
        return []

    try:
        conn = sqlite3.connect(f"file:{FGIP_DB}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT rowid, * FROM claims_fts WHERE claims_fts MATCH ? LIMIT 5",
            (fts_query,))
        results = []
        for row in cursor:
            d = dict(row)
            text = d.get("claim_text", "") or d.get("text", "")
            results.append({"text": text[:200], "rowid": d.get("rowid")})
        conn.close()
        return results
    except Exception as e:
        # Fail open (graph context is optional) but don't fail silently:
        # a malformed MATCH expression looked exactly like "no hits" before.
        if os.environ.get("CELL_DEBUG"):
            print(f"[context_compiler] fgip search failed: {e!r} "
                  f"query={fts_query!r}", file=sys.stderr)
        return []


def _search_memory(keywords: list[str]) -> list[dict]:
    """Search Claude Code memory files by keyword."""
    if not MEMORY_DIR.exists():
        return []

    results = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        try:
            content = f.read_text()
        except OSError:
            continue
        for kw in keywords[:5]:
            if kw.lower() in content.lower():
                # Find the matching line
                for line in content.split("\n"):
                    if kw.lower() in line.lower() and line.strip():
                        results.append({
                            "file": f.stem,
                            "line": line.strip()[:120],
                            "keyword": kw,
                        })
                        break
                break  # one hit per file is enough

    return results[:5]
