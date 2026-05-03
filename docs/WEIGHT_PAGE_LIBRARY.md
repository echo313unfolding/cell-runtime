# Weight Page Library

**Status:** v0 SHIPPED (physical paging)
**Date:** 2026-05-03

## What It Is

OS-style virtual memory for model weights. Each tensor in a GGUF file
becomes an addressable page with metadata, verification, and residency
tracking.

```
GPU (4 GB)  = hot compute, active tensor pages
CPU RAM (64 GB) = warm weight store, mmap'd pages
NVMe/disk  = cold store, full GGUF file
runtime    = scheduler/pager
codec      = compact addressable representation (HXQ future)
```

## Architecture

```
GGUF file on disk
    ↓
build_weight_page_manifest.py → manifest.json
    ↓
WeightPageLibrary(manifest.json)
    ↓
.load_page(name) → mmap read → CPU RAM buffer
    ↓
.copy_to_gpu(name) → torch uint8 tensor on CUDA
    ↓
.evict_page(name) → free GPU/RAM
```

## Manifest Fields

Each page entry:

| Field | Type | Description |
|-------|------|-------------|
| tensor_name | str | Full GGUF tensor name (e.g. `blk.5.attn_q.weight`) |
| layer_id | int? | Layer number, None for embeddings/output |
| tensor_role | str | `attn_q`, `ffn_down`, `embed`, `output_head`, etc. |
| byte_offset | int | Byte offset in GGUF file |
| byte_length | int | Size in bytes |
| shape | list | Tensor dimensions |
| dtype_or_quant | str | `Q4_K`, `Q6_K`, `F32`, etc. |
| sha256 | str? | Hash for verification |
| residency | str | `disk` / `ram` / `gpu` |
| hotness_score | float | Access counter |
| last_used | str? | ISO timestamp |
| receipt_id | str? | Audit reference |

## Proven (v0)

- 435 tensors from Qwen-Coder-3B Q4_K_M parsed and manifested
- 36 layers, 12 tensor roles per layer
- SHA256 verification on load
- mmap-based page loading (zero-copy from disk to RAM)
- GPU copy via torch: 12.4 MB FFN block in 80ms
- Full load/evict cycle
- 13/13 unit tests pass
- Receipt: `receipts/weight_page_library_v0_*.json`

## Size Distribution (Qwen-Coder-3B Q4_K_M)

```
ffn_down:    535 MB  (26.7%)
ffn_gate:    435 MB  (21.7%)
ffn_up:      435 MB  (21.7%)
output_head: 243 MB  (12.1%)
embed:       167 MB  (8.3%)
attn_o:       81 MB  (4.0%)
attn_q:       81 MB  (4.0%)
attn_v:       12 MB  (0.6%)
attn_k:       10 MB  (0.5%)
norms/bias:   <1 MB
```

FFN dominates: 1.4 GB of 2.0 GB total. This means tensor-level offload
policy should prioritize keeping attention + norms on GPU and streaming
FFN blocks from RAM.

## Next Steps

### Phase B: Smarter Offload Policy

- Keep attention + norms + embeddings GPU-resident
- Stream FFN blocks from CPU RAM with prefetching
- Measure tok/s vs layer-level offload (llama.cpp default)

### Phase C: Bigger Model Offload

- Download Qwen-Coder-7B or 14B Q4_K_M
- Build manifest, profile layer sizes
- Run n_gpu_layers sweep with llama-server
- Measure tok/s, RAM, VRAM at each split point

### Phase D: Semantic Cartridge Discovery

- After physical paging works, run task ablations
- Measure which tensor pages matter for specific task types
- If a subset reliably helps a domain, it becomes a cartridge
- Prove by eval, not by assumption

## What This Is NOT

- Not semantic skill loading from dense weights (that requires proof)
- Not MoE expert routing (that requires MoE architecture)
- Not inference modification (paging layer only)
- Not a replacement for llama.cpp layer offload (complementary)

Physical paging first. Semantic cartridges by ablation later.
