# Local Agent Substrate

**Work Order:** WO-AGENT-SUBSTRATE-01
**Status:** SPEC
**Date:** 2026-05-02
**Supersedes:** WO-SMALLM-DAEMON-01 (which becomes Phase 1 of this)

## The Architecture Statement

A small front-end model routes user/system requests into a local agent
substrate. Qwen Sentinel is the production specialist backend. Handmade SSM
provides auditable sequential memory. RAG provides document/policy context.
The graph provides entity/evidence relationships. Agents expose bounded tools
through MCP/API. A gatekeeper controls privileged actions. Every step emits
receipts. Containers make models and agents swappable.

## Component Map

```
User / System Event
       |
       v
+------------------+
| smaLLM           |  SmolLM3-3B. Always-on router/face.
| (front-end)      |  Routes, answers general, dispatches.
+------------------+
  |     |     |     |     |     |
  v     v     v     v     v     v
+---+ +---+ +---+ +---+ +---+ +---+
|RAG| |GRF| |SSM| |AGT| |SEN| |GK |
+---+ +---+ +---+ +---+ +---+ +---+
  |     |     |     |     |     |
  v     v     v     v     v     v
+------------------------------------------------+
|                 Receipt Layer                   |
|  Every call → receipt with cost block           |
+------------------------------------------------+
```

| Code | Component | Answers | Store |
|------|-----------|---------|-------|
| RAG | Document retrieval | "What do docs/policies/logs say?" | fgip.db FTS5 + file search |
| GRF | Evidence graph | "How are entities connected?" | fgip.db (nodes, edges, claims) |
| SSM | Sequential memory | "What has this entity been doing over time?" | sentinel.db + handmade_ssm.py |
| AGT | Bounded agents | "What bounded job needs doing?" | Agent registry (YAML) |
| SEN | Qwen Sentinel | "Given context, what is the verdict/tool call?" | Hybrid stack (SSM→LLM→gates) |
| GK | Gatekeeper | "Is this action allowed?" | policy.yaml + ask-pass |

## Existing Infrastructure (Wired, Not Rebuilt)

| What | Where | Size | Status |
|------|-------|------|--------|
| FGIP evidence graph | `~/fgip-engine/fgip.db` | 127 MB | LIVE. 1905 nodes, 2378 edges, 32K claims. FTS5. |
| Sentinel SSM state | `~/tools/sentinel/sentinel.db` | 72 KB | LIVE. alerts, verdicts, investigations. |
| Handmade SSM | `~/tools/sentinel/memory/handmade_ssm.py` | 24 KB | LIVE. Runtime state accumulation. |
| Sentinel hybrid stack | `~/tools/sentinel/runtime/hybrid_stack.py` | — | FROZEN v0.1. SSM→LLM→gates. |
| Echo memory (archive) | `~/.echo_memory/knowledge_base.db` | 44 KB | ARCHIVE. Superseded by markdown. |
| POI ledger | `~/.poi_ledger.db` | — | LIVE. Work/cost accounting. |
| Claude memory | `~/.claude/.../memory/*.md` | — | LIVE. Canonical session memory. |

**Key decision:** We wire these existing stores as read-only backends behind
bounded agents. We do NOT rebuild graph/RAG/SSM from scratch. The substrate
is a routing and permission layer over proven stores.

## Agent Design Rules

### Agents Are Bounded Workers

```
CORRECT:
  rag_lookup(query="incident response policy", scope="policies")
  graph_neighbors(entity_id="IP:203.0.113.5", relation_filter="connects_to")
  ssm_get_state(entity_id="user:root")
  sentinel_triage(context_pack={alert, ssm_state, rag_context, graph_links})

WRONG:
  agent_go_do_whatever(instructions="figure it out")
```

### Every Agent Has

| Field | Required |
|-------|----------|
| `name` | Unique identifier |
| `input_schema` | JSON Schema for inputs |
| `output_schema` | JSON Schema for outputs |
| `permission` | `read` / `write` / `privileged` |
| `receipt` | Always emitted |
| `failure_mode` | What happens on error |
| `timeout_s` | Max execution time |

### Permission Levels

| Level | Can Do | Examples |
|-------|--------|---------|
| `read` | Query stores, no side effects | rag_lookup, graph_lookup, ssm_get_state, receipt_lookup |
| `write` | Modify state stores | ssm_update_event, graph_add_edge (gated) |
| `privileged` | System actions, external calls | shell, delegate_to_host, file_write |

- `read` agents auto-approve (no prompt)
- `write` agents log but auto-approve (state changes are internal)
- `privileged` agents require ask-pass (user prompt)

## The Full Loop

### Security Event

```
event comes in
  → smaLLM routes to security_triage
  → Level 0 extractors pull entities/signals
  → ssm_update_event(event) — update per-entity state
  → graph_lookup(entity, relation_filter) — check entity connections
  → rag_lookup(query, scope="policies") — retrieve relevant policy/docs
  → context_pack = {alert, ssm_state, rag_context, graph_links}
  → sentinel_triage(context_pack) — Qwen produces verdict/tool call
  → gatekeeper validates any proposed action
  → agent executes allowed tool
  → receipt_write(event) — decision, evidence, model, memory hash, policy version
```

### General Query

```
user asks question
  → smaLLM classifies intent
  → if general/reasoning: answer directly, emit receipt
  → if coding: answer directly (or swap to coder), emit receipt
  → if security: full security loop above
```

### Regulated Asset Event (Future)

```
transfer request
  → extract wallet/asset/jurisdiction/counterparty/amount/velocity
  → ssm_update_event(wallet_event)
  → graph_lookup(counterparty, asset, jurisdiction)
  → rag_lookup(query, scope="compliance")
  → sentinel backend emits allow / hold / review
  → gatekeeper blocks signing unless policy passes
  → receipt records decision + evidence + model + memory hash + policy version
```

## Context Pack

When Sentinel processes a security event, it receives a context pack — a
structured bundle of everything the agents gathered:

```json
{
  "alert": {
    "text": "Failed password for root from 203.0.113.5 port 22",
    "source": "auth.log",
    "timestamp": "2026-05-02T21:00:00Z"
  },
  "ssm_state": {
    "entity": "IP:203.0.113.5",
    "event_count": 47,
    "last_seen": "2026-05-02T20:58:00Z",
    "trend": "accelerating",
    "prior_verdicts": ["suspicious", "suspicious"]
  },
  "rag_context": [
    {"source": "incident_response_policy.md", "excerpt": "..."},
    {"source": "prior_receipt_20260501.json", "excerpt": "..."}
  ],
  "graph_links": [
    {"from": "IP:203.0.113.5", "relation": "scanned_by", "to": "nmap"},
    {"from": "IP:203.0.113.5", "relation": "geo", "to": "jurisdiction:CN"}
  ]
}
```

## File Layout

```
cell-runtime/
  src/cell/
    ask_pass.py              # Gatekeeper (Phase 1, DONE)
    router.py                # Intent classifier (EXISTS)
    orchestrator.py          # Pipeline + ask-pass gate (EXISTS, patched)
    model_pool.py            # Model swap management (EXISTS)
    tool_registry.py         # Built-in tools (EXISTS)
    memory_lane.py           # Programmatic state capsule (EXISTS)
    specialists.py           # SentinelHybridAdapter (EXISTS)
    gateway.py               # OpenAI-compat API (EXISTS)
    mcp_server.py            # MCP integration (EXISTS)
    task_record.py           # Receipt format (EXISTS)

    agents/                  # NEW — bounded agent implementations
      __init__.py
      base.py                # AgentBase class + registry
      rag_agent.py           # rag_lookup, rag_search
      graph_agent.py         # graph_lookup, graph_neighbors
      ssm_agent.py           # ssm_get_state, ssm_update_event
      sentinel_agent.py      # sentinel_triage (context pack → verdict)
      policy_agent.py        # gate_decide (permission check)
      receipt_agent.py       # receipt_write, receipt_lookup

  runtime/
    policy.yaml              # Permission policy (Phase 1, DONE — extend)
    model_registry.json      # Model metadata (Phase 1, DONE)
    agent_registry.yaml      # NEW — agent definitions + schemas
    tool_schemas.yaml        # NEW — MCP tool schemas
    graph_schema.sql         # NEW — evidence graph schema reference
    rag_config.yaml          # NEW — RAG configuration
    ssm_config.json          # NEW — SSM configuration
    docker-compose.yml       # Container layout (Phase 1, DONE)

  tests/
    test_ask_pass.py         # 10 tests (Phase 1, DONE)
    test_model_routing.py    # 6 tests (Phase 1, DONE)
    test_routing_contract.py # NEW — routing contract enforcement
    test_agent_permissions.py # NEW — agent permission validation
    test_graph_rag_ssm_context_pack.py # NEW — context pack assembly
    test_receipt_required.py # NEW — receipt enforcement

  docs/
    SMALLM_SENTINEL_DAEMON_RUNTIME.md  # Phase 1 spec (DONE)
    LOCAL_AGENT_SUBSTRATE.md            # This document
```

## What Is NOT Built Yet

This spec defines the architecture. Implementation is phased:

- **Phase 1 (DONE):** ask-pass, policy.yaml, model_registry, routing tests
- **Phase 2 (THIS):** Agent base class, registry, schemas, context pack, graph/RAG/SSM configs
- **Phase 3:** Wire agents into orchestrator tool loop
- **Phase 4:** Integration tests with live stores
- **Phase 5:** Regulated asset agent set (wallet_policy_check, etc.)

## Hard Rules

1. Small model routes only — never reasons about verdicts
2. Qwen Sentinel is production backend — no candidate promotion without eval receipt
3. Zamba2 is research/cold backend — not in prod routing
4. SSM updates are live (write-level agents, auto-approve with log)
5. RAG and graph are read-only unless approved writer agent is called
6. Agents are bounded and schema-validated — no open-ended "go do whatever"
7. Privileged actions require gatekeeper approval (ask-pass)
8. Every model call, memory update, graph write, agent call, and command emits a receipt
9. No direct sudo from any model
10. No candidate backend promotion without eval receipt
