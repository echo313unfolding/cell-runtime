# EchoScience Hybrid Model Plan

**Status:** PLANNING
**Date:** 2026-05-03
**Author:** voidstream + Claude Code

## Goal

Design and build a **science/coding-focused hybrid SSM-Transformer** to serve
as the Echo backend. Not a general chatbot — a local scientific/coding operator
with internal recurrent state and external auditable memory.

## Architecture Target

```
EchoScience-Hybrid
  Transformer blocks → deep reasoning, code, math, tool formatting
  SSM/Mamba blocks   → long sequence tracking, state continuity, streaming memory

  Training focus:
    code repair, math reasoning, ML/AI systems, physics, biology terminology,
    tool-call traces, receipt reasoning, constrained state tracking

  Runtime:
    GGUF / llama.cpp (requires Zamba2 PR or equivalent)
    HXQ candidate after Q5_K_M baseline passes
    cell-runtime orchestrator, SSM/RAG/graph memory, ask-pass gatekeeper
```

## Why Hybrid SSM-Transformer

Coding and science work has exactly the kind of structure SSMs should help with:

- Long files, dependency chains, symbol tables
- Multi-step state (bracket/function/class scope)
- Proof/derivation steps
- Logs over time, tool traces, repeated entities

A pure transformer can do this but pays high attention cost. An SSM-hybrid
carries sequential state more efficiently.

**Internal SSM** = model thinks over sequences (architecture benefit).
**External SSM** = runtime remembers and audits (cell-runtime benefit).
The strongest version has both.

## Why Not SmolLM3

SmolLM3-3B failed the simplest conversational test ("hey echo" → hallucinated
a fake Node.js tutorial). Root cause: wrong chat template in Ollama +
insufficient instruction tuning for identity/tool behavior. Even with the
template fixed, SmolLM3 is not reliable enough for front-end duty.

**Current production:** Qwen2.5-Coder-3B Q4_K_M via llama-server.
Passes all 5 behavior smoke tests. 14 tok/s on T2000.

## Stages

### Stage 1: Pick Hybrid Base

Candidates:
- **Zamba2-2.7B** — proven on this box, HXQ PASS, llama.cpp PR pending (#21412)
- **Mamba2-hybrid** — if available with permissive weights
- **Other SSM/Transformer hybrids** — evaluate as they appear

Selection criteria: permissive license, GGUF convertible, fits T2000 4GB.

### Stage 2: QLoRA on Echo Domains

Training data mixture:
- Code: Python repair pairs, test generation, stack trace explanation
- Science: physics/math derivations, unit consistency, numerical checks
- ML: tensor shape explanation, training log analysis, architecture comparison
- Biology/medical: concept classification, pathway explanation (no overclaiming)
- Tool-call traces: `tool_call` block emission, tool result synthesis
- Receipts: receipt JSON interpretation, cost comparison
- Constrained state: bracket depth, lexer state, multi-entity tracking

Data sources:
- Existing receipts and tool traces from cell-runtime
- Public code datasets (filtered for quality)
- arXiv abstracts (ML, physics, bio)
- Manual curation of Echo-specific operational patterns

### Stage 3: Evaluate Against Qwen-Coder Baseline

The eval covers the actual target domains, not general chat:

**Code:**
- Repair broken Python (syntax, logic, runtime)
- Write tests for given functions
- Explain stack traces
- Patch small files safely

**Math/Physics:**
- Symbolic derivation
- Unit consistency checks
- Numerical sanity (order of magnitude)

**ML:**
- Explain tensor shapes
- Debug training logs
- Compare architectures
- Read and interpret receipts

**Biology/Medical:**
- Classify concepts correctly
- Explain pathways without overclaiming
- Flag uncertainty

**SSM/MorphSAT:**
- Bracket depth tracking
- Lexer state maintenance
- Multi-entity state tracking
- Illegal transition detection

**Runtime:**
- Choose correct tool from available set
- Obey ask-pass permissions
- Emit receipt fields correctly
- Refuse unsafe shell commands
- Emit proper `tool_call` JSON blocks

### Stage 4: Quantization Ladder

```
F16 → Q8_0 → Q6_K → Q5_K_M → Q4_K_M
```

Each step: eval on full scorecard, compare to F16 baseline.

### Stage 5: HXQ Candidate

Only after Q5_K_M baseline passes:
- Tensor fidelity: cosine >= 0.998
- Behavioral eval: no regression beyond tolerance
- Promotion requires both gates

Fallback: HXQ → Q5_K_M → Qwen-Coder

### Stage 6: Deploy as Echo Backend

Replace Qwen-Coder as default only when hybrid model beats it on eval.

Not a chatbot. A **local science/coding/constraint operator** with:
- Internal SSM for long-range technical state
- External SSM/RAG/graph for auditable memory
- Receipt-backed eval proving every claim

## Current Production Stack (Interim)

```
Default brain:     qwen2.5-coder-3b (Q4_K_M, llama-server, 14 tok/s)
Security gate:     qwen2.5-sentinel (Q4_K_M, llama-server, 13.4 tok/s)
Memory:            external SSM + RAG (FGIP FTS5) + evidence graph
Router:            keyword classifier (zero GPU)
Tool loop:         20+ tools, multi-turn, ask-pass gated
Receipts:          per-turn JSON with cost block
```

SmolLM3 available via `/model smollm3` but not trusted as default until it
passes behavior smoke tests.

## Success Criteria

The hybrid model ships as Echo backend when:

1. Beats Qwen-Coder on code repair + tool selection tasks
2. Beats transformer-only baseline on long state tracking (SSM benefit)
3. Preserves `tool_call` JSON format (no regression)
4. Runs locally through GGUF/llama.cpp at >= 10 tok/s on T2000
5. Has receipt-backed eval for every claim
6. HXQ candidate only after Q5_K_M behavioral gate passes

## Non-Goals

- General chatbot behavior
- Multi-language fluency (English-first)
- Image/multimodal (text-only for now)
- Training from scratch (too expensive, too risky)
