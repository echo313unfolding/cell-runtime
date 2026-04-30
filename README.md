# Cell Runtime

Multi-lane local AI orchestrator. Runs multiple specialized models on a single GPU through intelligent routing, model swapping, and rolling session memory.

A **cell** is a self-contained runtime unit:
- **Membrane** — OpenAI-compatible API gateway
- **Organelles** — specialized model lanes (coder, security, reasoning)
- **State** — rolling memory lane for session continuity
- **Metabolism** — model pool with hot-swap on a single GPU

## Architecture

```
         ┌─────────────────────────────┐
         │     Gateway :8800           │
         │   (OpenAI-compatible API)   │
         └─────┬─────┬─────┬─────┬────┘
               │     │     │     │
          ┌────▼┐ ┌──▼──┐ ┌▼───┐ ┌▼──────┐
          │Coder│ │Secur│ │Reas│ │Memory │
          │Qwen │ │Qwen │ │Smol│ │ Lane  │
          │ 3B  │ │ 3B  │ │LM3 │ │(state)│
          └─────┘ └─────┘ └────┘ └───────┘
               │     │     │
               └─────┴─────┘
                     │
            ┌────────▼────────┐
            │  llama-server   │
            │  (one at a time)│
            └─────────────────┘
```

Only one model is loaded at a time. The orchestrator classifies intent, swaps models when needed, and maintains session state across swaps.

## Quick Start

```bash
# Start llama-server with any supported model
llama-server -m model.gguf --port 8080 -ngl 99

# Start the gateway
python3 -m cell.gateway --config configs/config.local.json

# Use it (auto-routes to the right lane)
curl http://localhost:8800/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"cell-auto","messages":[{"role":"user","content":"write a sort function"}]}'
```

## Docker

```bash
docker build -t echo-cell .
docker run --gpus all -v /path/to/models:/models echo-cell --status
docker run --gpus all -v /path/to/models:/models echo-cell "write a sort function"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat (auto-routes or force a lane) |
| GET | `/v1/models` | List available models/lanes |
| GET | `/v1/memory/capsule` | Current session memory state |
| POST | `/v1/memory/ingest` | Feed an event to memory |
| POST | `/v1/memory/reset` | Clear session memory |
| GET | `/health` | Liveness probe |
| GET | `/status` | Full runtime state |
| GET | `/swap-history` | Model swap event log |

## Model Names

| API Name | Lane | Description |
|----------|------|-------------|
| `cell-auto` | (routed) | Auto-routes based on intent classification |
| `cell-coder` | Coding | Forces the coding lane |
| `cell-sentinel` | Security | Forces the security triage lane |
| `cell-reasoning` | Reasoning | Forces the reasoning/explanation lane |

## Memory Lane

The memory lane maintains rolling session state across turns. Every generation result is automatically ingested. The accumulated state is injected into the next request's context.

Two modes:
- **Programmatic** (default): extracts files, errors, fixes, blockers from outputs. Zero latency overhead.
- **LLM-enhanced** (`llm_enhanced: true` in config): uses the loaded model to refine state. Adds ~1s per turn but produces better summaries.

```bash
# Check current memory state
curl http://localhost:8800/v1/memory/capsule

# Reset for a new session
curl -X POST http://localhost:8800/v1/memory/reset
```

## Specialist Adapters

Lanes can delegate to external specialist pipelines via the adapter interface. The orchestrator finds the right specialist by intent, delegates the full pipeline, and falls back to the generic memory lane if unavailable.

Current specialists:
- **Sentinel Hybrid Stack v0.1** — security triage via external SSM state + LLM + post-LLM gate (`tools/sentinel/`). Activated when `PYTHONPATH` includes the sentinel package.

For non-security intents (coding, reasoning, general), the generic memory lane handles state tracking.

### Model Compatibility

The runtime is model-pluggable. The external SSM memory and gate live outside the model, so they wrap any compatible llama-server backend. Model quality and instruction-following ability affect triage accuracy, not runtime compatibility.

| Model | Hard20 | Gate TP/FP | Tier |
|-------|--------|------------|------|
| Sentinel repair v1 (Qwen + LoRA) | 84% | 6/0 | COMPATIBLE_NATIVE |
| Qwen2.5-Coder-3B-Instruct | 69% | 3/0 | COMPATIBLE_NATIVE |
| SmolLM3-3B | 70% | 0/0 | COMPATIBLE_GRAMMAR |

**Tier definitions:**
- `COMPATIBLE_NATIVE` — works without grammar constraint
- `COMPATIBLE_GRAMMAR` — needs GBNF grammar for reliable structured output
- `LOAD_FAIL` — GGUF format not supported by current binary (not a model quality issue)

### Hardware Policy

| Hardware | Strategy |
|----------|----------|
| 4GB VRAM (T2000) | One model at a time, sticky sessions, rotate on lane change (~12s swap) |
| 24GB+ VRAM (3090/4090) | Multi-port hot servers possible, zero swap time |

## Ecosystem

| Component | Repo | Role |
|-----------|------|------|
| **Cell Runtime** | `echo313unfolding/cell-runtime` | Orchestration, routing, memory, API |
| **Helix Substrate** | `echo313unfolding/helix-substrate` | Codec, quantization, compression |
| **llama.cpp (HXQ fork)** | local | Inference backend with HXQ type support |

### Interface Contract (v1)

Cell Runtime consumes model artifacts produced by Helix Substrate:
- **Format**: GGUF (standard or HXQ_AFFINE_6)
- **Metadata**: model type, architecture, quantization method, bits-per-weight
- **Capability flags**: tool use, instruction following, context length

## Project Layout

```
cell/
  orchestrator.py    — classify → route → swap → generate → log
  gateway.py         — OpenAI-compatible HTTP API
  memory_lane.py     — rolling session state tracker
  model_pool.py      — llama-server model loader and swap manager
  router.py          — intent classification and lane routing
  tool_registry.py   — tool definitions for model tool use
  task_record.py     — structured task logging with cost blocks
  download_models.py — HuggingFace model downloader
  mcp_server.py      — MCP server for Claude Code integration
configs/
  config.native.json — container config (all llama-server)
  config.local.json  — host development config
tests/
  memory_lane_test.py    — A/B test: state tracking accuracy
  run_memory_lane_ab.sh  — automated comparison runner
bench/
  energy_bench.py             — energy/throughput measurement
  quant_comparison_bench.py   — quantization quality comparison
bin/
  ech0               — CLI entry point
Dockerfile           — GPU-enabled container image
docker-compose.yml   — compose with GPU reservation
```

## License

Apache 2.0
