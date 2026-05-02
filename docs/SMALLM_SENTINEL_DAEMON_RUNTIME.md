# smaLLM + Sentinel Daemon Runtime Spec

**Work Order:** WO-SMALLM-DAEMON-01
**Status:** SPEC
**Date:** 2026-05-02

## Overview

A containerized local runtime where a small always-on LLM (smaLLM) acts as the
user-facing front-end and dispatcher, and Qwen Sentinel acts as a callable
backend specialist for security triage. The SSM memory spine tracks session
state. Privileged actions use ask-pass (user approval prompt).

## Architecture

```
User
  |
  v
+-----------------------+
| smaLLM (SmolLM3-3B)  |  Always-on front-end. Classifies, dispatches,
| "the receptionist"    |  answers general queries, calls tools.
+-----------------------+
  |          |         |
  v          v         v
+--------+ +--------+ +-----------+
| Tools  | | Memory | | Sentinel  |
| (local)| | (SSM)  | | (Qwen-3B) |
+--------+ +--------+ +-----------+
  |                        |
  v                        v
+--------+            +--------+
|ask-pass|            |Hybrid  |
|(user   |            |Stack   |
| prompt)|            |(gates) |
+--------+            +--------+
```

### Key Roles

| Component | Model | Role | VRAM | Always-on? |
|-----------|-------|------|------|------------|
| smaLLM | SmolLM3-3B (HXQ affine6) | Front-end dispatcher, general Q&A | ~2.3 GB | Yes |
| Sentinel | Qwen2.5-3B + LoRA repair v1 (Q8_0) | Security triage specialist | ~3.1 GB | No (loaded on demand) |
| Memory | Programmatic SSM state | Session continuity | 0 | Yes |

**GPU constraint:** T2000 has 4 GB. Only one model loaded at a time. smaLLM is
default-loaded. Sentinel swaps in when security_triage intent is detected, swaps
back to smaLLM after.

## What Changes vs Current cell-runtime

The existing cell-runtime (`cell-runtime/src/cell/`) already implements most of
this. This spec formalizes the architecture and adds the ask-pass gate.

### Already Built (keep as-is)
- `router.py` — intent classifier (regex-based)
- `model_pool.py` — multi-backend model management, swap logic
- `orchestrator.py` — end-to-end pipeline
- `tool_registry.py` — safe built-in tools (read_file, grep, shell, etc.)
- `memory_lane.py` — programmatic state capsule
- `specialists.py` — SentinelHybridAdapter wrapping frozen hybrid stack
- `gateway.py` — OpenAI-compatible HTTP API
- `mcp_server.py` — Claude Code integration (cell_status/classify/generate)
- `task_record.py` — handoff format with receipts

### New Components

1. **`ask_pass.py`** — Privileged action gate (user approval prompt)
2. **`runtime/policy.yaml`** — Declarative policy (which actions need approval)
3. **`runtime/model_registry.json`** — Model metadata and swap rules
4. **`runtime/docker-compose.yml`** — Multi-service containerization
5. **`runtime/mcp_tools/sentinel_tools.py`** — Sentinel-specific MCP tools
6. **`runtime/receipt_writer.py`** — Standardized receipt emission

### Modified Components

- `tool_registry.py` — Add ask-pass gate to `shell` and `delegate_to_host` tools
- `orchestrator.py` — Wire ask-pass into tool execution loop
- `config.native.json` — Add sentinel-specific model entry with LoRA path

## ask-pass Gate

### Design Principle

Like Claude Code: when the local model wants to do something privileged, it
prints a prompt and waits for the user to type `y` or `n`. No complex
allowlists, no gatekeeper model, no policy engine. Just ask.

### What Needs Approval

| Action | Why |
|--------|-----|
| `shell` tool (any command) | Could modify system state |
| `delegate_to_host` | Escalates to Claude Code / Tier 3 |
| Model swap to Sentinel | Loads security specialist (informational, auto-approve) |
| File write (future) | Modifies filesystem |

### What Does NOT Need Approval

| Action | Why |
|--------|-----|
| `read_file` | Read-only |
| `grep` | Read-only |
| `list_dir` | Read-only |
| `web_search` | Read-only |
| `memory_search` | Read-only |
| Model inference | Core function |
| Memory lane updates | Internal state |

### Implementation

```python
# ask_pass.py

import sys

# Actions that require user approval
PRIVILEGED_ACTIONS = {
    "shell",
    "delegate_to_host",
    "file_write",
}

# Actions that auto-approve (logged but no prompt)
AUTO_APPROVE = {
    "read_file",
    "grep",
    "list_dir",
    "web_search",
    "memory_search",
}


def ask_pass(action: str, detail: str, auto: bool = False) -> bool:
    """Prompt user for approval of a privileged action.

    Returns True if approved, False if denied.
    If auto=True or action is not privileged, returns True without prompting.
    """
    if auto or action in AUTO_APPROVE or action not in PRIVILEGED_ACTIONS:
        return True

    print(f"\n--- ask-pass ---")
    print(f"  Action: {action}")
    print(f"  Detail: {detail}")
    response = input("  Allow? [y/N] ").strip().lower()
    approved = response in ("y", "yes")

    if not approved:
        print("  Denied.")
    return approved
```

### Integration with Tool Registry

The orchestrator's tool execution loop checks ask-pass before executing:

```python
# In orchestrator._generate(), tool execution section:
for tc in calls:
    tool_name = tc.get("name", "")
    tool_args = tc.get("arguments", {})

    # ask-pass gate
    if not ask_pass(tool_name, json.dumps(tool_args, indent=2)):
        tool_result = "Action denied by user."
        messages.append({
            "role": "user",
            "content": f"Tool denied ({tool_name}): user declined the action.",
        })
        continue

    tool_result = self.tool_registry.execute(tool_name, tool_args)
```

## Model Registry

```json
{
  "_schema": "cell_model_registry.v1",
  "models": {
    "smollm3": {
      "role": "front-end",
      "always_on": true,
      "gguf": "/models/smollm3-3b-hxq-affine6.gguf",
      "vram_mb": 2300,
      "tok_s": 28.3,
      "intents": ["general", "reasoning", "coding"],
      "swap_priority": 0
    },
    "qwen2.5-sentinel": {
      "role": "specialist",
      "always_on": false,
      "gguf": "/models/sentinel-repair-v1-q8_0.gguf",
      "vram_mb": 3100,
      "tok_s": 14.66,
      "intents": ["security_triage"],
      "swap_priority": 1,
      "specialist_adapter": "sentinel_hybrid",
      "lora": null,
      "eval_receipt": "receipts/qwen_sentinel_same_200_eval_20260502T185400Z.json"
    }
  },
  "gpu_budget_mb": 4096,
  "swap_policy": "sticky",
  "default_model": "smollm3"
}
```

## Policy

```yaml
# policy.yaml — what needs approval, what auto-runs

ask_pass:
  # Tools that prompt the user before execution
  privileged:
    - shell
    - delegate_to_host
    - file_write

  # Tools that run without prompting
  auto_approve:
    - read_file
    - grep
    - list_dir
    - web_search
    - memory_search

  # Model swaps: log but don't block
  model_swap: auto_approve_with_log

# Escalation: when smaLLM can't handle it
escalation:
  # smaLLM detects it needs Sentinel
  security_triage: swap_to_sentinel
  # Sentinel detects it needs Tier 3
  tier3_needed: delegate_to_host
  # Either model detects it's stuck
  stuck: delegate_to_host

# Memory
memory:
  mode: programmatic  # or llm_enhanced
  capsule_injection: true
  max_turns: 100
```

## Container Layout

```yaml
# docker-compose.yml
version: "3.8"

services:
  cell-daemon:
    image: echo-cell:daemon
    build:
      context: ..
      dockerfile: cell-runtime/Dockerfile
    ports:
      - "8800:8800"   # OpenAI-compat API
    volumes:
      - /home/voidstr3m33/models:/models:ro
      - cell-receipts:/receipts
      - /home/voidstr3m33/tools/sentinel:/sentinel:ro
    environment:
      - CELL_MODELS_DIR=/models
      - CELL_RECEIPTS_DIR=/receipts
      - PYTHONUNBUFFERED=1
      - CELL_ASK_PASS=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    entrypoint: ["python3", "-m", "cell.gateway"]
    command: ["--port", "8800", "--config", "/app/cell-runtime/runtime/config.daemon.json"]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8800/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  llama-server:
    image: echo-llama-server:latest
    volumes:
      - /home/voidstr3m33/models:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: ["-m", "/models/smollm3-3b-hxq-affine6.gguf", "-ngl", "99", "-c", "4096", "--port", "8080"]
    ports:
      - "8080:8080"

volumes:
  cell-receipts:
```

**Note:** On T2000 (single GPU, 4GB), both services share the same GPU.
llama-server holds the model; cell-daemon talks to it via HTTP. Model swaps
are llama-server reloads (unload current GGUF, load new one).

## Sentinel as Callable Backend

smaLLM calls Sentinel through the existing specialist adapter path:

1. Router classifies intent as `security_triage`
2. Orchestrator checks for specialist → finds `SentinelHybridAdapter`
3. Adapter runs the frozen hybrid stack (SSM state → LLM → gate)
4. Result feeds back through smaLLM for user-facing output

The Sentinel model is NOT a separate service. It's a GGUF that llama-server
loads on demand (swap). The hybrid stack logic runs in the cell-daemon process.

## MCP Tools for Sentinel

```python
# sentinel_tools.py — additional MCP tools when Sentinel is active

sentinel_triage:
  description: "Run a security alert through the full Sentinel pipeline
    (SSM state → Qwen LLM → post-LLM gates). Returns structured verdict."
  input: { alert_text: string }
  output: { verdict, severity, is_benign, actions, gate_fired, receipt }

sentinel_status:
  description: "Show Sentinel state: SSM state vector, gate policy,
    recent verdicts, alert counts."
  input: {}
  output: { ssm_state, gate_policy, recent_verdicts, counts }
```

These are exposed through the existing MCP server alongside cell_status,
cell_classify, and cell_generate.

## Receipt Writer

Every action through the daemon produces a receipt:

```json
{
  "receipt_id": "cell_daemon_20260502T200000Z",
  "action": "security_triage",
  "model": "qwen2.5-sentinel",
  "swap_from": "smollm3",
  "swap_time_s": 2.1,
  "verdict": {"severity": "high", "is_benign": false},
  "gate_fired": true,
  "gate_rule": "G2",
  "ask_pass": {"shell": "denied", "read_file": "auto"},
  "cost": {
    "wall_time_s": 4.3,
    "cpu_time_s": 0.8,
    "peak_memory_mb": 3200,
    "hostname": "echo-labs",
    "timestamp_start": "2026-05-02T20:00:00",
    "timestamp_end": "2026-05-02T20:00:04"
  }
}
```

## Interaction Flow

### Happy Path: General Query
```
User: "what's the weather API endpoint?"
smaLLM: (classify=general, no swap needed, answer directly)
  → "The OpenWeatherMap API endpoint is..."
```

### Happy Path: Security Alert
```
User: "analyze this auth.log entry: Failed password for root from 203.0.113.5"
smaLLM: (classify=security_triage, swap to Sentinel)
  → [swap: smollm3 → qwen2.5-sentinel, 2.1s]
  → Sentinel hybrid stack runs:
    SSM: update state with new alert
    LLM: classify severity=high, is_benign=false
    Gate: G2 fires (compromised keywords)
  → smaLLM presents: "Severity: HIGH. Brute force attempt on root..."
  → [swap back: qwen2.5-sentinel → smollm3]
```

### Privileged Action
```
User: "check if port 4444 is open"
smaLLM: (classify=general, wants to call shell tool)
  → ask-pass:
    Action: shell
    Detail: {"command": "ss -tlnp | grep 4444"}
    Allow? [y/N] y
  → executes, returns result
```

### Escalation
```
User: "deep analysis of this APT campaign"
smaLLM → Sentinel: (classify=security_triage)
Sentinel: (this exceeds local model capability)
  → delegate_to_host escalation
  → ask-pass:
    Action: delegate_to_host
    Detail: {"goal": "APT campaign analysis", "reason": "requires frontier reasoning"}
    Allow? [y/N] y
  → Escalation surfaced to Claude Code / user
```

## Implementation Plan

### Phase 1: ask-pass (this PR)
- [ ] Create `cell-runtime/src/cell/ask_pass.py`
- [ ] Wire into orchestrator tool execution loop
- [ ] Add policy.yaml
- [ ] Tests: test_ask_pass.py

### Phase 2: Model Registry + Config
- [ ] Create runtime/model_registry.json
- [ ] Create config.daemon.json with Sentinel LoRA GGUF path
- [ ] Update docker-compose.yml for daemon mode

### Phase 3: Sentinel MCP Tools
- [ ] Add sentinel_triage and sentinel_status to MCP server
- [ ] Wire through specialist adapter

### Phase 4: Integration Test
- [ ] End-to-end: general query (no swap)
- [ ] End-to-end: security query (swap + hybrid stack)
- [ ] End-to-end: privileged action (ask-pass)
- [ ] End-to-end: escalation (delegate_to_host)

## Non-Goals

- No new model training
- No changes to the frozen hybrid stack
- No Ollama dependency (llama-server only)
- No complex gatekeeper model — ask-pass is a user prompt, not a model
- No multi-GPU — T2000 single GPU constraint is the design target
