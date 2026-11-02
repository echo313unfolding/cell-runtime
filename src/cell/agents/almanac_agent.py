"""Almanac agents — member data normalization and evidence queries.

These agents wrap the almanac-agent normalizer as Cell-dispatchable tools.
The normalizer does the heavy lifting (parsing, classification, provenance).
The LLM's role is to interpret the evidence table for the member.

Design principle: mechanisms carry the work, not the LLM.
  - almanac_normalize: runs deterministic parser, produces receipted evidence table
  - almanac_evidence: queries pre-computed evidence table for member-facing answers
"""
import hashlib
import json
import os
import platform
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

from cell.agents.base import AgentBase, AgentResult, Permission


# Default paths
TAKEOUT_WATCH_HISTORY = Path(os.path.expanduser(
    "~/fgip-engine/data/artifacts/youtube_takeout/Takeout/"
    "YouTube and YouTube Music/history/watch-history.html"))
EVIDENCE_DIR = Path(os.path.expanduser("~/receipts/almanac"))


def _ensure_normalizer():
    """Add almanac-agent/src to path so we can import the normalizer."""
    agent_src = Path(os.path.expanduser("~/almanac-agent/src"))
    if str(agent_src) not in sys.path:
        sys.path.insert(0, str(agent_src))


class AlmanacNormalizeAgent(AgentBase):
    """Run the YouTube Takeout normalizer and produce a member evidence table.

    This is Baseline A from the research protocol: conservative deterministic
    parser, no inferred attributes, all unknown fields retained. Every derived
    attribute traces to source bytes (SHA-256) and locator.
    """
    name = "almanac_normalize"
    description = (
        "Normalize a YouTube Takeout export into a member evidence table. "
        "Returns summary statistics, top channels by quality-weighted engagement, "
        "signal quality distribution, and a receipt with full provenance."
    )
    permission = Permission.WRITE  # Writes receipt files
    timeout_s = 60

    input_schema = {
        "type": "object",
        "properties": {
            "source_path": {
                "type": "string",
                "description": "Path to watch-history.html (default: standard Takeout location)",
            },
            "member_id": {
                "type": "string",
                "description": "Member identifier for hashing (default: 'member_0')",
            },
        },
        "required": [],
    }

    def execute(self, args: dict) -> AgentResult:
        source_path = args.get("source_path", str(TAKEOUT_WATCH_HISTORY))
        member_id = args.get("member_id", "member_0")

        source = Path(os.path.expanduser(source_path))
        if not source.exists():
            return AgentResult(error=f"Source not found: {source}")

        # Verify it's HTML (basic sanity)
        try:
            head = source.read_bytes()[:200]
            if b"<" not in head:
                return AgentResult(error=f"Source does not appear to be HTML: {source}")
        except Exception as e:
            return AgentResult(error=f"Cannot read source: {e}")

        # Import normalizer from almanac-agent
        _ensure_normalizer()
        try:
            from normalizer import normalize_youtube_takeout
        except ImportError as e:
            return AgentResult(
                error=f"Cannot import normalizer from ~/almanac-agent/src: {e}")

        # Run with cost measurement
        t_start = time.time()
        cpu_start = time.process_time()
        start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

        table = normalize_youtube_takeout(str(source), member_id=member_id)

        cost = {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        summary = table.summary()

        # Extract key evidence for the LLM
        affinities = [a for a in table.attributes
                      if a.attribute_name == "interest_channel_affinity"]
        affinities.sort(
            key=lambda a: -a.typed_value["quality_weighted_score"])
        top_channels = [
            {"channel": a.typed_value["channel"],
             "watches": a.typed_value["watch_count"],
             "weighted": round(a.typed_value["quality_weighted_score"], 1),
             "avg_quality": round(a.typed_value["avg_signal_quality"], 3)}
            for a in affinities[:15]
        ]

        sig_dist = None
        recency = None
        for a in table.attributes:
            if a.attribute_name == "signal_quality_distribution":
                sig_dist = a.typed_value
            elif a.attribute_name == "activity_recency":
                recency = a.typed_value

        # Write receipt
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema": "almanac.normalize.v1",
            "source_path": str(source),
            "source_sha256": table.source_file_sha256,
            "member_hash": table.member_hash,
            "transform_version": table.transform_version,
            "summary": summary,
            "top_channels_quality_weighted": top_channels,
            "signal_type_distribution": sig_dist,
            "activity_recency": recency,
            "cost": cost,
        }
        receipt_path = (
            EVIDENCE_DIR
            / f"normalize_{time.strftime('%Y%m%dT%H%M%SZ')}.json")
        receipt_path.write_text(json.dumps(receipt, indent=2, default=str))

        # Save aggregate evidence table (not per-video — those are huge)
        agg_attrs = [
            asdict(a) for a in table.attributes
            if a.attribute_name in (
                "interest_channel_affinity",
                "signal_quality_distribution",
                "activity_recency",
            )
        ]
        evidence_path = EVIDENCE_DIR / "latest_evidence_table.json"
        evidence_path.write_text(json.dumps({
            "member_hash": table.member_hash,
            "source_sha256": table.source_file_sha256,
            "total_records": table.total_source_records,
            "mapped_records": table.mapped_records,
            "transform_version": table.transform_version,
            "attributes": agg_attrs,
        }, indent=2, default=str))

        return AgentResult(
            output={
                "status": "normalized",
                "total_records": table.total_source_records,
                "mapped_records": table.mapped_records,
                "attributes_extracted": len(table.attributes),
                "provenance_coverage": summary["provenance_coverage"],
                "top_channels": top_channels[:5],
                "signal_distribution": sig_dist,
                "receipt_path": str(receipt_path),
            },
            receipt=receipt,
        )


def infer_almanac_query(user_text: str) -> str | None:
    """Deterministic query selection from user input for almanac_evidence.

    The 3B model sometimes picks the wrong query parameter (e.g.,
    signal_quality when the user asked about a specific channel). This
    function extracts the right query from the user's natural language
    so the mechanism doesn't depend on model tool-arg selection.

    Returns a query string like 'channel:lex fridman', 'top_channels',
    'signal_quality', 'recency', or None if ambiguous.
    """
    text = user_text.lower()

    # Channel-specific: "how much do I watch X", "my X viewing", etc.
    # Look for a quoted or named entity after watch/channel keywords.
    import re

    # Pattern: "watch <channel_name> on youtube" or "<channel_name> channel"
    # or "how much/many/often ... <channel_name>"
    channel_patterns = [
        # "watch Lex Fridman on youtube"
        r'\bwatch\b\s+(.+?)\s+(?:on\s+)?(?:youtube|yt)\b',
        # "watched NetworkChuck" / "I've watched NetworkChuck"
        r'\bwatched\s+([A-Z][\w\s]{1,30}?)(?:\s+on|\s*$|\s*\?)',
        # "how much do I watch Lex Fridman"
        r'\bhow\s+(?:much|many|often)\b.*\bwatch\s+([A-Z][\w\s]{1,30}?)(?:\s+on|\s*$|\s*\?)',
    ]
    # Filler words between "watched" and the channel name
    _FILLER = {'so', 'so much', 'a lot of', 'too much', 'lots of', 'tons of'}

    for pat in channel_patterns:
        m = re.search(pat, user_text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().rstrip('?.,!')
            # Strip filler phrases (e.g., "watched so much 3Blue1Brown")
            name_lower = name.lower()
            for filler in sorted(_FILLER, key=len, reverse=True):
                if name_lower.startswith(filler + ' '):
                    name = name[len(filler):].strip()
                    name_lower = name.lower()
                    break
            # Filter out generic words that aren't channel names
            first_word = name_lower.split()[0] if name.strip() else ""
            if first_word not in ('youtube', 'channel', 'channels', 'the',
                                  'my', 'for', 'on', 'news', 'cooking',
                                  'a', 'an', 'it', 'if', 'at', 'to'):
                return f"channel:{name_lower}"

    # Top channels
    if re.search(r'\b(top|most|favorite|best)\b.*\b(channels?|watch)', text):
        return "top_channels"
    if re.search(r'\bchannels?\b.*\b(top|most|favorite|best|watch)\b', text):
        return "top_channels"

    # Signal quality
    if re.search(r'\bsignal.?quality\b', text):
        return "signal_quality"

    # Recency
    if re.search(r'\b(recent|recen|when|year|timeline)\b.*\b(watch|view|viewing|history)\b', text):
        return "recency"

    return None


def _query_mentioned_in_text(query: str, text: str) -> bool:
    """Check if the query type is explicitly mentioned in the user's text.

    Used to protect valid subqueries in combined requests. If the user
    said "show top channels and signal quality", both top_channels and
    signal_quality are valid — the model should be allowed to call either
    without the correction system overriding it.
    """
    import re
    text_lower = text.lower()
    if query == "top_channels":
        return bool(re.search(
            r'\b(top|most|favorite|best)\b.*\b(channels?|watch)', text_lower))
    if query == "signal_quality":
        return bool(re.search(r'\bsignal.?quality\b', text_lower))
    if query == "recency":
        return bool(re.search(
            r'\b(recent|recen|when|year|timeline)\b.*\b(watch|view|viewing|history)\b',
            text_lower))
    if query.startswith("channel:"):
        channel_name = query[len("channel:"):].strip()
        return channel_name in text_lower
    return False


class AlmanacEvidenceAgent(AgentBase):
    """Query the latest almanac evidence table for member-facing answers.

    Reads the pre-computed evidence table (from almanac_normalize) and
    returns structured answers. The LLM uses this to explain the data
    to the member in natural language.

    Query correction: when _trusted_user_question is set by the
    orchestrator (not by the model), infer_almanac_query extracts the
    intended query from the raw question. This only overrides the
    model's query when inference is confident (returns non-None) AND
    the model's query doesn't already match. The model's query is
    preserved when inference is ambiguous.
    """
    name = "almanac_evidence"
    description = (
        "Query the member's evidence table. "
        "Returns channel affinities, signal quality, activity recency, "
        "or specific channel lookups. Use query: "
        "'top_channels', 'signal_quality', 'recency', 'channel:<name>'."
    )
    permission = Permission.READ
    timeout_s = 10

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to look up: 'top_channels', 'signal_quality', "
                    "'recency', 'channel:<name>'"
                ),
            },
        },
        "required": ["query"],
    }

    def execute(self, args: dict) -> AgentResult:
        requested_query = args["query"].strip().lower()
        query = requested_query
        correction = None

        # _trusted_user_question is set by the orchestrator from the raw
        # user input (before augmentation). It is never set by the model.
        # Any model-supplied "user_question" arg is ignored.
        trusted_question = args.get("_trusted_user_question", "")
        if trusted_question:
            inferred = infer_almanac_query(trusted_question)
            if inferred and inferred != query:
                # Before overriding: check if the model's query is a valid
                # subquery that the user explicitly mentioned. A combined
                # request like "show top channels and signal quality" should
                # let the model's signal_quality call through even though
                # inference picks top_channels as the primary.
                model_query_valid = _query_mentioned_in_text(
                    query, trusted_question)
                if model_query_valid:
                    # Model's query is valid for this user text — don't override.
                    pass
                else:
                    correction = {
                        "requested": requested_query,
                        "executed": inferred,
                        "reason": "deterministic override from raw user question",
                    }
                    query = inferred
        evidence_path = EVIDENCE_DIR / "latest_evidence_table.json"

        if not evidence_path.exists():
            return AgentResult(
                error="No evidence table found. Run almanac_normalize first.")

        data = json.loads(evidence_path.read_text())
        attrs = data.get("attributes", [])

        output = self._run_query(query, data, attrs)
        if correction:
            output["correction"] = correction
        return AgentResult(output=output, error=output.get("error"))

    @staticmethod
    def _run_query(query: str, data: dict, attrs: list) -> dict:
        """Execute a single query against the evidence table."""
        if query == "top_channels":
            channels = [a for a in attrs
                        if a["attribute_name"] == "interest_channel_affinity"]
            channels.sort(
                key=lambda a: -a["typed_value"]["quality_weighted_score"])
            return {
                "query": query,
                "total_records": data.get("total_records"),
                "results": [
                    {"channel": a["typed_value"]["channel"],
                     "watches": a["typed_value"]["watch_count"],
                     "weighted": a["typed_value"]["quality_weighted_score"],
                     "avg_quality": a["typed_value"]["avg_signal_quality"]}
                    for a in channels[:15]
                ],
            }

        if query == "signal_quality":
            for a in attrs:
                if a["attribute_name"] == "signal_quality_distribution":
                    return {
                        "query": query,
                        "distribution": a["typed_value"],
                    }
            return {"query": query, "error": "No signal quality distribution in evidence table."}

        if query == "recency":
            for a in attrs:
                if a["attribute_name"] == "activity_recency":
                    return {
                        "query": query,
                        "year_distribution": a["typed_value"],
                    }
            return {"query": query, "error": "No activity recency in evidence table."}

        if query.startswith("channel:"):
            channel_name = query[len("channel:"):].strip()
            channels = [a for a in attrs
                        if a["attribute_name"] == "interest_channel_affinity"]
            matches = [
                a for a in channels
                if channel_name in a["typed_value"]["channel"].lower()
            ]
            if not matches:
                return {
                    "query": query,
                    "results": [],
                    "note": "No matching channel found.",
                }
            return {
                "query": query,
                "results": [
                    {"channel": a["typed_value"]["channel"],
                     "watches": a["typed_value"]["watch_count"],
                     "weighted": a["typed_value"]["quality_weighted_score"],
                     "avg_quality": a["typed_value"]["avg_signal_quality"]}
                    for a in matches
                ],
            }

        return {
            "query": query,
            "error": (
                "Unknown query. Use: top_channels, signal_quality, "
                "recency, channel:<name>"
            ),
        }
