"""Almanac query inference tests.

Tests deterministic query selection for almanac_evidence. The 3B model
sometimes picks wrong query parameters (e.g., signal_quality when the
user asked about a channel). infer_almanac_query fixes this by extracting
the right query from the user's natural language.

The "DEFECT" cases below document known failures from live orchestrator
runs. They are regression tests — if the inference function stops
catching these, the model will mis-dispatch again.

Usage:
    python3 -m pytest tests/test_almanac_query.py -v
    python3 tests/test_almanac_query.py  # standalone
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.almanac_agent import infer_almanac_query

# (user_question, expected_query)
CASES = [
    # ── channel lookups ──
    ("how much do I watch Lex Fridman on youtube", "channel:lex fridman"),
    ("how many times have I watched NetworkChuck", "channel:networkchuck"),
    ("I've watched so much 3Blue1Brown", "channel:3blue1brown"),

    # ── DEFECT: model called signal_quality instead of channel:lex fridman ──
    # Live run 2026-11-02: user asked "how much do I watch Lex Fridman on
    # youtube?" and model called almanac_evidence(query="signal_quality").
    # The deterministic inference must override to channel:lex fridman.
    ("how much do I watch Lex Fridman on youtube?", "channel:lex fridman"),

    # ── top_channels ──
    ("what channels do I watch the most", "top_channels"),
    ("show my top channels by signal quality", "top_channels"),
    ("my most watched channels", "top_channels"),
    ("what are my favorite channels", "top_channels"),

    # ── signal_quality ──
    ("what is my signal quality distribution", "signal_quality"),
    ("signal quality for my youtube data", "signal_quality"),

    # ── recency ──
    ("when did I last watch youtube", "recency"),
    ("my recent viewing timeline", "recency"),

    # ── ambiguous / no match (should return None) ──
    ("my youtube watch history analysis", None),
    ("almanac evidence table", None),
    ("normalize my takeout export", None),

    # ── negative: should NOT infer a channel ──
    ("how often should I watch for memory leaks", None),
    ("watch the build output for errors", None),
    ("Have you watched for memory leaks in the daemon", None),
]


def test_all():
    failures = []
    for question, expected in CASES:
        got = infer_almanac_query(question)
        if got != expected:
            failures.append((question, expected, got))

    if failures:
        lines = [f"  FAIL: infer({q!r}) = {got!r}, expected {exp!r}"
                 for q, exp, got in failures]
        raise AssertionError(
            f"{len(failures)}/{len(CASES)} misclassified:\n" + "\n".join(lines))


def main():
    passed = 0
    failed = 0
    for question, expected in CASES:
        got = infer_almanac_query(question)
        ok = got == expected
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL  {got!r:30s} (expected {expected!r:30s}) <- {question}")

    total = passed + failed
    print(f"\n  {passed}/{total} PASS, {failed}/{total} FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
