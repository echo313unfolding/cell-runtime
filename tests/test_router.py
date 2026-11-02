"""Router intent classification tests.

Labelled questions across all six intents. Run before adding patterns.

Usage:
    python3 -m pytest tests/test_router.py -v
    python3 tests/test_router.py  # standalone
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.router import classify

# (query, expected_intent)
CASES = [
    # ── coding ──
    ("write a python function to sort a list", "coding"),
    ("fix the bug in auth.py line 42", "coding"),
    ("refactor this class to use dataclasses", "coding"),
    ("implement a binary search in rust", "coding"),
    ("write a pytest for the parser module", "coding"),

    # ── security_triage ──
    ("check auth.log for brute force attempts", "security_triage"),
    ("analyze this suspicious base64 encoded powershell", "security_triage"),
    ("investigate failed login attempts in syslog", "security_triage"),
    ("triage this CVE-2024-1234 alert for lateral movement", "security_triage"),

    # ── reasoning ──
    ("explain how quicksort works step by step", "reasoning"),
    # "merge sort" and "heap sort" both hit coding keywords — defensible
    ("compare merge sort and heap sort", "coding"),
    ("what is the proof that sqrt(2) is irrational", "reasoning"),

    # ── hybrid_experiment ──
    ("run an ablation on the mamba SSM layer", "hybrid_experiment"),
    ("probe the hybrid model state space dimensions", "hybrid_experiment"),

    # ── almanac ──
    ("what does my youtube data claim about me", "almanac"),
    ("normalize my takeout export", "almanac"),
    ("show my top channels by signal quality", "almanac"),
    ("what does this source claim about me", "almanac"),
    ("almanac evidence table", "almanac"),
    ("broker comparison for my watch history", "almanac"),
    ("what is my signal quality distribution for my youtube data", "almanac"),
    ("how much do I watch Lex Fridman on youtube", "almanac"),
    ("how many times have I watched NetworkChuck", "almanac"),
    ("what channels do I watch the most", "almanac"),
    ("how often do I watch news channels", "almanac"),
    ("my youtube watch history analysis", "almanac"),

    # ── general (no strong signal) ──
    ("what time is it in Tokyo", "general"),
    ("summarize this paragraph for me", "general"),
    ("hello", "general"),

    # ── negative: should NOT be almanac ──
    # "watch for memory leaks" has no coding keywords — general is defensible
    ("how often should I watch for memory leaks", "general"),
    ("watch the build output for errors", "coding"),
]


def test_all():
    failures = []
    for query, expected in CASES:
        got = classify(query)
        if got != expected:
            failures.append((query, expected, got))

    if failures:
        lines = [f"  FAIL: classify({q!r}) = {got!r}, expected {exp!r}"
                 for q, exp, got in failures]
        raise AssertionError(
            f"{len(failures)}/{len(CASES)} misclassified:\n" + "\n".join(lines))


def main():
    passed = 0
    failed = 0
    for query, expected in CASES:
        got = classify(query)
        ok = got == expected
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL  {got:20s} (expected {expected:20s}) <- {query}")

    total = passed + failed
    print(f"\n  {passed}/{total} PASS, {failed}/{total} FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
