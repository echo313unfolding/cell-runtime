#!/usr/bin/env python3
"""CPU router — classify intent and dispatch to the right local model.

Rules-based, zero VRAM. This is the traffic cop for the 3-model roster:
  - Qwen2.5-Coder (qwen2.5-coder:latest): coding tasks
  - Sentinel (qwen2.5-sentinel): security triage
  - SmolLM3 (smollm3): reasoning, general, default

Usage:
    # As library
    from tools.router import classify, dispatch
    intent = classify("reverse shell on port 4444")
    result = dispatch(intent, "reverse shell on port 4444")

    # CLI
    echo "check auth.log for brute force" | python3 tools/router.py
    python3 tools/router.py "write a python function to sort a list"
    python3 tools/router.py --classify "what is photosynthesis"
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from task_record import TaskRecord, INTENTS

# ============================================================================
# Intent classification — rules only, no model
# ============================================================================

SECURITY_PATTERNS = [
    r'\b(ssh)\b',
    r'\b(brute.?force|failed.*(auth|login|attempt)|login.?fail|failed.?login)\b',
    r'\b(reverse.?shell|bind.?shell)\b',
    r'(nc\s+-[el]|netcat\s+-[el]|\bncat\b)',
    r'\b(yara|malware|trojan|backdoor|webshell|rootkit|payload)\b',
    r'\b(cve-\d{4}|exploit|vulnerability|0day|zero.?day)\b',
    r'\b(suspicious|escalat|triage|alert|incident)\b',
    r'\b(crypto.?min|xmr|monero|coinhive)\b',
    r'\b(privilege.?escal|sudo.?root|setuid|suid)\b',
    r'\b(port.?scan|nmap|masscan|shodan)\b',
    r'\b(encoded.?powershell|base64.*exec|eval\(|base64.*payload|base64.*shell)\b',
    r'\b(firewall|iptables|ufw|selinux|apparmor)\b',
    r'\b(sentinel|security.?check|threat|ioc|indicator)\b',
    r'\b(auth\.log|syslog|/var/log)\b',
    r'\b(dns.?traffic|dns.?tunnel|dns.?exfil|outbound.?traffic|lateral.?movement)\b',
    r'\b(phishing|spear.?phish|credential.?harvest)\b',
    r'\b(investigate|analyze|inspect).*(logs?|alerts?|traffic|attempts?|events?)\b',
]

CODING_PATTERNS = [
    r'\b(write|implement|code|function|class|method|script)\b',
    r'\b(python|javascript|rust|go|java|c\+\+|typescript)\b',
    r'\b(bug|fix|debug|error|exception|traceback)\b',
    r'\b(refactor|optimize|test|unittest|pytest)\b',
    r'\b(git|commit|branch|merge|rebase|pr|pull.?request)\b',
    r'\b(api|endpoint|server|client|request|response)\b',
    r'(^|\s)(import |from [a-z_]\w+ import|def |class |return |if |for |while )',
    r'\b(dockerfile|docker|kubernetes|k8s|helm)\b',
    r'\b(pip|npm|cargo|make|cmake|gcc)\b',
]

REASONING_PATTERNS = [
    r'\b(explain|why|how\s+does|what\s+is|compare)\b',
    r'\b(reason|logic|proof|theorem|derive)\b',
    r'\b(step.?by.?step|think.?through|work.?out)\b',
    r'\b(math|calcul|equation|formula|algebra)\b',
]

HYBRID_PATTERNS = [
    r'\b(zamba|mamba|ssm|hybrid.?model|state.?space)\b',
    r'\b(architecture.?experiment|ablation|probe)\b',
]


def classify(text: str) -> str:
    """Classify input text into an intent category. Pure rules, zero cost."""
    text_lower = text.lower()

    # Score each category
    scores = {
        "security_triage": 0,
        "coding": 0,
        "reasoning": 0,
        "hybrid_experiment": 0,
    }

    for pattern in SECURITY_PATTERNS:
        if re.search(pattern, text_lower):
            scores["security_triage"] += 1

    for pattern in CODING_PATTERNS:
        if re.search(pattern, text_lower):
            scores["coding"] += 1

    for pattern in REASONING_PATTERNS:
        if re.search(pattern, text_lower):
            scores["reasoning"] += 1

    for pattern in HYBRID_PATTERNS:
        if re.search(pattern, text_lower):
            scores["hybrid_experiment"] += 1

    # Pick highest score
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general"

    # Security needs strong signal (>=2 matches) to avoid false routing
    if best == "security_triage" and scores[best] < 2:
        # Single security keyword — probably just mentioned in passing
        second = sorted(scores, key=scores.get, reverse=True)[1]
        if scores[second] > 0:
            return second
        return "general"

    return best


def route_model(intent: str) -> str:
    """Map intent to model name (must match config roster keys)."""
    return {
        "security_triage": "qwen2.5-sentinel",
        "coding": "qwen2.5-coder",
        "general": "smollm3",
        "reasoning": "smollm3",
        "hybrid_experiment": "smollm3",
    }.get(intent, "smollm3")


def dispatch(intent: str, input_text: str, save: bool = False) -> TaskRecord:
    """Create task record, dispatch to appropriate handler."""
    model = route_model(intent)
    task = TaskRecord.create(intent=intent, input_text=input_text, routed_model=model)
    task.route_reason = f"rules-based classification: {intent}"
    task.start()

    if intent == "security_triage":
        result = _dispatch_sentinel(input_text)
        task.set_result(
            model="qwen2.5-sentinel",
            verdict=result.get("verdict", "suspicious"),
            confidence=result.get("confidence", 0.0),
            reasoning=result.get("reasoning", ""),
            output_text=result.get("raw_output", ""),
        )
        if task.verdict == "escalate":
            task.escalate("Sentinel flagged for frontier analysis")
    else:
        # For non-security intents, just classify and record — actual generation
        # happens through whatever interface the user is already using (Claude Code,
        # Ollama, etc). The router's job is classification, not generation.
        task.set_result(
            model=model,
            reasoning=f"Routed to {model} for {intent}. Execute via user's preferred interface.",
        )

    if save:
        task.save()

    return task


def _dispatch_sentinel(input_text: str) -> dict:
    """Call sentinel-triage and parse result."""
    sentinel_path = Path(__file__).parent / "sentinel-triage"
    if not sentinel_path.exists():
        return {"verdict": "suspicious", "confidence": 0.0,
                "reasoning": "sentinel-triage not found"}

    try:
        result = subprocess.run(
            [sys.executable, str(sentinel_path), "--json", input_text[:2000]],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0:
            task_data = json.loads(result.stdout)
            return {
                "verdict": task_data.get("verdict", "suspicious"),
                "confidence": task_data.get("confidence", 0.0),
                "reasoning": task_data.get("reasoning", ""),
                "raw_output": task_data.get("output_text", ""),
            }
        else:
            return {"verdict": "suspicious", "confidence": 0.0,
                    "reasoning": f"sentinel-triage error: {result.stderr[:200]}"}
    except Exception as e:
        return {"verdict": "suspicious", "confidence": 0.0,
                "reasoning": f"dispatch error: {e}"}


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="router",
        description="CPU router — classify intent and dispatch to local model",
    )
    parser.add_argument("text", nargs="*", help="Input text")
    parser.add_argument("--classify", "-c", action="store_true",
                        help="Classify only, don't dispatch")
    parser.add_argument("--json", "-j", action="store_true",
                        help="JSON output")
    parser.add_argument("--save", "-s", action="store_true",
                        help="Save task record to ~/receipts/tasks/")
    parser.add_argument("--dispatch", "-d", action="store_true",
                        help="Actually dispatch to model (security only for now)")
    args = parser.parse_args()

    # Gather input
    if args.text:
        input_text = " ".join(args.text)
    elif not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()
    else:
        print("Usage: router <text>", file=sys.stderr)
        print("       echo 'text' | router", file=sys.stderr)
        sys.exit(1)

    intent = classify(input_text)
    model = route_model(intent)

    if args.classify and not args.dispatch:
        if args.json:
            print(json.dumps({"intent": intent, "model": model, "input": input_text[:100]}))
        else:
            print(f"  Intent: {intent}")
            print(f"  Model:  {model}")
        return

    if args.dispatch:
        task = dispatch(intent, input_text, save=args.save)
        if args.json:
            print(task.to_json())
        else:
            print(task.summary())
            if task.verdict:
                color = {"benign": "\033[32m", "suspicious": "\033[33m", "escalate": "\033[31m"}.get(task.verdict, "")
                reset = "\033[0m" if color else ""
                print(f"  {color}{task.verdict.upper()}{reset}: {task.reasoning[:200]}")
    else:
        # Default: classify + show routing decision
        print(f"  Intent: {intent}")
        print(f"  Route:  {model}")
        if intent == "security_triage":
            print(f"  Action: sentinel-triage --json '{input_text[:60]}...'")
        else:
            print(f"  Action: Use {model} via preferred interface")


if __name__ == "__main__":
    main()
