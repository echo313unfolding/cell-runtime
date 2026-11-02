"""Almanac evidence agent integration tests.

Tests the query correction pipeline end-to-end through
AlmanacEvidenceAgent.execute(), verifying:

1. Prior-context contamination: augmented prompt containing channel
   names from evidence blocks must NOT cause spurious channel lookups.
2. Combined requests: the model may issue two tool calls for a query
   like "show top channels and signal quality". Each call's query
   must be preserved when inference doesn't contradict it.
3. Model-supplied user_question: any model-supplied "user_question"
   or "_trusted_user_question" in tool_args must be ignored. Only
   the orchestrator's trusted injection matters.
4. Original defect: model calls signal_quality when user asked about
   a specific channel — the correction must fire and be recorded.

Usage:
    python3 -m pytest tests/test_almanac_evidence_agent.py -v
    python3 tests/test_almanac_evidence_agent.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.almanac_agent import (
    AlmanacEvidenceAgent, EVIDENCE_DIR, infer_almanac_query,
)

# Minimal evidence table for testing
MOCK_TABLE = {
    "member_hash": "test_member",
    "source_sha256": "abc123",
    "total_records": 100,
    "mapped_records": 100,
    "transform_version": "test",
    "attributes": [
        {
            "attribute_name": "interest_channel_affinity",
            "typed_value": {
                "channel": "Lex Fridman",
                "watch_count": 143,
                "quality_weighted_score": 120.5,
                "avg_signal_quality": 0.843,
            },
        },
        {
            "attribute_name": "interest_channel_affinity",
            "typed_value": {
                "channel": "3Blue1Brown",
                "watch_count": 89,
                "quality_weighted_score": 85.2,
                "avg_signal_quality": 0.957,
            },
        },
        {
            "attribute_name": "signal_quality_distribution",
            "typed_value": {
                "deliberate": 0.42,
                "music": 0.18,
                "quick_view": 0.25,
                "shorts": 0.15,
            },
        },
        {
            "attribute_name": "activity_recency",
            "typed_value": {"2024": 500, "2025": 300, "2026": 200},
        },
    ],
}


def _setup_evidence(tmp_dir):
    """Write mock evidence table to a temp directory."""
    evidence_path = Path(tmp_dir) / "latest_evidence_table.json"
    evidence_path.write_text(json.dumps(MOCK_TABLE, indent=2))
    return evidence_path


def _make_agent(tmp_dir):
    """Create an agent pointing at the temp evidence dir."""
    import cell.agents.almanac_agent as mod
    orig = mod.EVIDENCE_DIR
    mod.EVIDENCE_DIR = Path(tmp_dir)
    agent = AlmanacEvidenceAgent()
    return agent, orig


def _restore_evidence_dir(orig):
    import cell.agents.almanac_agent as mod
    mod.EVIDENCE_DIR = orig


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.details = []

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            self.details.append(f"  FAIL: {name}: {detail}")

    def report(self):
        for d in self.details:
            print(d)
        total = self.passed + self.failed
        print(f"\n  {self.passed}/{total} PASS, {self.failed}/{total} FAIL")
        return 0 if self.failed == 0 else 1


def run_tests():
    r = TestResults()

    with tempfile.TemporaryDirectory() as tmp:
        _setup_evidence(tmp)
        agent, orig = _make_agent(tmp)

        try:
            # ── Test 1: Prior-context contamination ──
            # Simulate an augmented prompt that contains evidence about
            # Lex Fridman from a prior turn (realistic: "You watch X on
            # youtube"), but the actual question is about signal quality.
            augmented = (
                "--- Evidence (retrieved from local files) ---\n"
                "You watch Lex Fridman on youtube: 143 times, quality: 0.843\n"
                "--- End evidence ---\n\n"
                "what is my signal quality distribution"
            )
            raw_question = "what is my signal quality distribution"

            # The augmented prompt DOES contaminate — the evidence block
            # text matches "watch Lex Fridman on youtube" and extracts
            # a spurious channel lookup.
            contaminated = infer_almanac_query(augmented)
            r.check("contamination_raw_detects_channel",
                    contaminated is not None and "lex fridman" in str(contaminated),
                    f"augmented prompt inference = {contaminated}")

            # But the raw question correctly infers signal_quality.
            clean = infer_almanac_query(raw_question)
            r.check("clean_raw_detects_quality",
                    clean == "signal_quality",
                    f"raw question inference = {clean}")

            # Agent called with _trusted_user_question = raw (not augmented).
            # Model chose signal_quality (correct this time).
            result = agent.execute({
                "query": "signal_quality",
                "_trusted_user_question": raw_question,
            })
            r.check("no_contamination_correction",
                    result.ok and result.output.get("correction") is None,
                    f"correction={result.output.get('correction')}")
            r.check("signal_quality_returned",
                    "distribution" in result.output,
                    f"output keys={list(result.output.keys())}")

            # ── Test 2: Combined request ──
            # User asks "show top channels and signal quality". Model
            # makes two tool calls. Each should preserve its query.
            raw_combined = "show top channels and signal quality"
            # infer_almanac_query picks one winner — top_channels.
            combined_infer = infer_almanac_query(raw_combined)
            r.check("combined_infers_top_channels",
                    combined_infer == "top_channels",
                    f"combined inference = {combined_infer}")

            # Call 1: model says top_channels. Inference agrees. No override.
            r1 = agent.execute({
                "query": "top_channels",
                "_trusted_user_question": raw_combined,
            })
            r.check("combined_call1_no_correction",
                    r1.ok and r1.output.get("correction") is None,
                    f"correction={r1.output.get('correction')}")
            r.check("combined_call1_has_results",
                    "results" in r1.output,
                    f"output keys={list(r1.output.keys())}")

            # Call 2: model says signal_quality. Inference says top_channels
            # (different). But "signal quality" is explicitly mentioned in
            # the user text, so the model's query is a valid subquery.
            # The correction must NOT fire — both queries are valid.
            r2 = agent.execute({
                "query": "signal_quality",
                "_trusted_user_question": raw_combined,
            })
            r.check("combined_call2_no_override",
                    r2.ok and r2.output.get("correction") is None,
                    f"correction={r2.output.get('correction')}")
            r.check("combined_call2_preserves_signal_quality",
                    r2.output.get("query") == "signal_quality",
                    f"query={r2.output.get('query')}")
            r.check("combined_call2_has_distribution",
                    "distribution" in r2.output,
                    f"output keys={list(r2.output.keys())}")

            # ── Test 3: Model-supplied user_question bypass ──
            # Model tries to set user_question or _trusted_user_question.
            # The orchestrator always overwrites _trusted_user_question.
            # Here we simulate what happens if only the model's field exists
            # (no orchestrator injection) — it should be ignored.

            # 3a: Model sets "user_question" (no _trusted_). Agent ignores it.
            r3a = agent.execute({
                "query": "signal_quality",
                "user_question": "how much do I watch Lex Fridman on youtube",
            })
            r.check("model_user_question_ignored",
                    r3a.ok and r3a.output.get("correction") is None,
                    f"correction={r3a.output.get('correction')}")

            # 3b: Orchestrator overwrites _trusted_ even if model set it.
            r3b = agent.execute({
                "query": "signal_quality",
                "_trusted_user_question": "how much do I watch Lex Fridman on youtube",
            })
            r.check("trusted_overrides_model_query",
                    r3b.ok and r3b.output.get("correction") is not None,
                    f"correction={r3b.output.get('correction')}")
            r.check("trusted_correction_to_channel",
                    r3b.output.get("query") == "channel:lex fridman",
                    f"query={r3b.output.get('query')}")

            # ── Test 4: Original defect — signal_quality instead of channel ──
            # Model called signal_quality but user asked about Lex Fridman.
            r4 = agent.execute({
                "query": "signal_quality",
                "_trusted_user_question": "how much do I watch Lex Fridman on youtube?",
            })
            r.check("defect_corrected",
                    r4.ok and r4.output.get("query") == "channel:lex fridman",
                    f"query={r4.output.get('query')}")
            r.check("defect_correction_recorded",
                    r4.output.get("correction") is not None,
                    f"correction={r4.output.get('correction')}")
            correction = r4.output.get("correction", {})
            r.check("defect_correction_fields",
                    correction.get("requested") == "signal_quality"
                    and correction.get("executed") == "channel:lex fridman"
                    and "reason" in correction,
                    f"correction={correction}")
            r.check("defect_returns_channel_data",
                    any("Lex Fridman" in str(x) for x in r4.output.get("results", [])),
                    f"results={r4.output.get('results')}")

        finally:
            _restore_evidence_dir(orig)

    return r


def test_all():
    """Pytest entry point."""
    r = run_tests()
    if r.failed:
        lines = r.details
        raise AssertionError(
            f"{r.failed}/{r.passed + r.failed} failed:\n" + "\n".join(lines))


def main():
    r = run_tests()
    return r.report()


if __name__ == "__main__":
    sys.exit(main())
