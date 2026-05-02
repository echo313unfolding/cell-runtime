# Specialist Compute Pool — Cartridges + Shards

**Work Order:** WO-SPECIALIST-COMPUTE-POOL-01
**Status:** SPEC + IMPLEMENTATION
**Date:** 2026-05-02

## Concept

Specialist skill cartridges are **capability packages**, NOT monolithic models.
A cartridge may contain a LoRA adapter, small GGUF, grammar pack, RAG index,
prompt pack, policy file, or eval receipt. The runtime loads only the slice
needed for the task.

**This is NOT "load one giant coding model."**
**This IS "break capability into loadable, auditable, task-specific cartridges."**

## What a Cartridge Is

A cartridge can be:

| Type | Example |
|------|---------|
| Small specialist GGUF model | `parser_repair.gguf` (300 MB) |
| LoRA adapter | `adapter.gguf-lora` on Sentinel base |
| Prompt/tool grammar pack | `grammar.json` + system prompt |
| RAG index | `repo_index/` for codebase context |
| Repo-specific context bundle | File index + symbol table |
| Code repair ruleset | `examples.jsonl` + heuristics |
| Policy/eval manifest | `policy.yaml` + `eval_receipt.json` |
| Compressed HXQ/GGUF shard | `shard.hxq` (fraction of larger model) |

The unit is a **capability package**, not always a full model.

## Architecture

```
smaLLM front-end
  → routes request

Qwen Sentinel backend
  → decides risk / tool / escalation

Cartridge pool
  → loads only the needed skill cartridge
  → runs through base model with augmented prompt/adapter

RAG / graph / SSM
  → supplies context

Gatekeeper
  → blocks direct execution

Receipts
  → record cartridge used, version, hash, result
```

## Cartridge Layout

```
cartridges/
  code_parser_repair/
    manifest.json
    examples.jsonl
    eval_receipt.json

  rule_generation/
    manifest.json
    examples.jsonl
    eval_receipt.json

  patch_review/
    manifest.json
    eval_receipt.json

  exploit_analysis/
    manifest.json
    eval_receipt.json

  repo_context/
    manifest.json
    eval_receipt.json
```

## Manifest Schema

```json
{
  "cartridge_id": "code_parser_repair_v1",
  "type": "skill_cartridge",
  "activation_intents": ["parser_repair", "tool_fix", "json_repair"],
  "base_model": "qwen2.5-sentinel",
  "artifact_type": "prompt_pack",
  "scope": "code_repair",
  "max_context_tokens": 4096,
  "fallback": "qwen2.5-sentinel",
  "requires_ask_pass": false,
  "status": "active",
  "system_prompt": "...",
  "examples_path": "examples.jsonl",
  "eval_receipt": "eval_receipt.json",
  "sha256": "..."
}
```

## Runtime Behavior

```
Sentinel needs parser repair
  → router selects code_parser_repair cartridge
  → load prompt pack + examples
  → build augmented prompt
  → run through Sentinel base model
  → return repair proposal
  → gatekeeper requires approval before file write
  → receipt written
  → cartridge unloaded
```

No monolithic 14B/32B coder required.

## Safety Boundary

Cartridges **propose**. They do NOT execute.

- No cartridge can call `shell`
- No cartridge can call `file_write`
- No cartridge can call `delegate_to_host`
- All cartridge agents have `Permission.READ`
- File writes/shell/sudo require ask-pass through gatekeeper

## Cartridge Agents

| Agent | Intent(s) | Cartridge | Permission |
|-------|-----------|-----------|------------|
| `cartridge_dispatch` | any | routes to matching | READ |
| `code_repair` | code_repair, parser_repair | code_parser_repair | READ |
| `rule_generate` | yara_rule, sigma_rule | rule_generation | READ |
| `patch_review` | patch_review, diff_review | patch_review | READ |
| `exploit_analysis` | exploit_analysis, vuln_analysis | exploit_analysis | READ |
| `cartridge_list` | — | lists all cartridges | READ |

## Candidate Cartridge Policy

- Candidate cartridges (`status: "candidate"`) are NOT activated
- Promotion to `active` requires an eval receipt
- Eval receipt must include test results on real data
- No synthetic-data-only evaluations

## Fallback

If a cartridge is unavailable or fails:
1. Fall back to `fallback` model specified in manifest (default: Sentinel)
2. If fallback also fails, return error to caller
3. All fallback paths emit receipt

## Level 3: Monolithic Model Sharding (Shard Pool)

For tasks requiring deeper capability than any cartridge provides, the runtime
can load monolithic large models split across CPU/GPU/disk. This uses existing
llama.cpp/GGUF mechanisms, NOT custom tensor parallelism.

### Three Levels of Specialist Compute

| Level | Name | What | When |
|-------|------|------|------|
| 1 | Whole-model cold load | Load one model when needed | Simple, VRAM-heavy |
| 2 | Skill cartridges | Task-specific adapters/indexes/grammars | Modular, fast |
| 3 | Monolithic model sharding | Split large model across CPU/GPU/disk | Deep/heavy tasks |

### Shard Mechanisms

| Mechanism | How |
|-----------|-----|
| GGUF split files | `model-00001-of-00004.gguf` |
| CPU/GPU layer split | `--n-gpu-layers 20` (rest in CPU RAM) |
| Disk mmap | `--mmap` (weights memory-mapped from disk) |
| Quantized tiers | Hot layers higher precision, cold layers offloaded |

### Shard Manifest

```json
{
  "model_id": "qwen_coder_14b_split_cpu_gpu",
  "role": "escalation",
  "backend": "llama_cpp",
  "shard_paths": ["/models/coder/qwen-coder-14b-q5.gguf"],
  "offload_policy": {
    "gpu_layers": 10,
    "cpu_layers": "remaining",
    "mmap": true
  },
  "required_vram_mb": 2000,
  "required_ram_mb": 14000,
  "activation_intents": ["deep_code_analysis", "multi_file_refactor"],
  "fallback": "qwen_coder_7b_q5_local",
  "idle_unload_s": 180,
  "status": "candidate"
}
```

### Routing Hierarchy

```
request arrives
  → router classifies intent
  → if normal: smaLLM/Sentinel handles it
  → if specialist: check cartridge pool (Level 2)
  → if cartridge insufficient: check shard pool (Level 3)
  → load shard with offload policy
  → run task
  → write receipt
  → unload after idle
```

### Shard Pool Agents

| Agent | Permission | What |
|-------|-----------|------|
| `specialist_compute_route` | READ | Routes: cartridge first, shard second, fallback third |
| `shard_list` | READ | Lists all shards with status and resource requirements |
| `shard_resource_check` | READ | Checks if a shard fits available VRAM/RAM |

### Receipt for Shard Calls

```json
{
  "event": "sharded_model_call",
  "model_id": "qwen_coder_14b_split_cpu_gpu",
  "shard_count": 1,
  "offload_policy": {"gpu_layers": 10, "mmap": true},
  "caller": "sentinel_triage",
  "task": "patch_review",
  "wall_time_s": 41.2,
  "approved": true
}
```

## HXQ Asset Lifecycle

HXQ is the desired substrate for specialist assets, but Q5_K_M is the control group.

### Correct Order

```
1. Prove with normal GGUF quants (Q5_K_M / Q6_K / Q8_0)
2. Convert selected assets to HXQ
3. Re-run same evals (same prompts, same tasks, same scoring)
4. Promote HXQ only if it preserves behavior
```

### Codec Types

| Codec | Use | Where |
|-------|-----|-------|
| `q5_k_m` | Standard baseline / control group | Always available |
| `q6_k` | Higher fidelity baseline | When Q5 insufficient |
| `q8_0` | Maximum fidelity baseline | Pod / cloud |
| `hxq_affine_6` | GPU edge, tight memory | When model otherwise doesn't fit VRAM |
| `hxq_affine_g128` | CPU-friendly, safer packing | CPU fallback, boring reliability |

### Promotion Rules

**Baseline codecs (Q5/Q6/Q8):** Promote with behavioral eval receipt only.

**HXQ codecs:** Promote with BOTH:
1. Tensor fidelity receipt (cosine_min >= 0.998)
2. Behavioral eval receipt (same prompts, same tasks, same scoring as baseline)

### HXQ Asset Receipt Schema

```json
{
  "asset_id": "wizardcoder_15b_specialist_shard_0002",
  "asset_type": "llm_weight_shard",
  "codec": "hxq_affine_6",
  "group_size": 128,
  "sha256_original": "...",
  "sha256_compressed": "...",
  "cosine_min": 0.9994,
  "ppl_baseline": 8.22,
  "ppl_compressed": 8.30,
  "ppl_delta_pct": 0.97,
  "behavioral_eval_pass": true,
  "runtime_status": "candidate"
}
```

### Fallback Policy

```
HXQ shard (candidate) → baseline Q5 shard (active) → Sentinel
```

If HXQ load/eval fails:
1. Fall back to `fallback_shard` (standard GGUF quant)
2. If fallback unavailable, fall back to Sentinel
3. Quarantine the failed HXQ asset (`status: "quarantined"`)
4. Emit receipt recording the fallback

### Manifest Fields for HXQ

Shard and cartridge manifests include:
- `codec` — q5_k_m / q6_k / q8_0 / hxq_affine_6 / hxq_affine_g128
- `helix_codec_receipt` — path to tensor fidelity receipt
- `behavioral_eval_receipt` — path to behavioral eval receipt
- `fallback_shard` — ID of fallback shard (standard GGUF)
- `status` — candidate / active / disabled / quarantined

### The Rule

**No HXQ promotion without behavioral eval.**
Tensor cosine is necessary but not sufficient.

## Future Extensions

When hardware allows:
- LoRA adapters loaded onto Sentinel base (cartridge)
- Small specialist GGUFs (cartridge, 300-500 MB)
- HXQ shards for partial model loading (shard)
- RAG indices for domain-specific knowledge (cartridge)
- Multi-process shard workers (shard)
- Remote/cloud shard escalation (shard)
