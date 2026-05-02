"""Shard pool — monolithic large model sharding via llama.cpp split/mmap/offload.

A shard is NOT a skill cartridge. A shard is a piece of a monolithic large model
that does not fit in one GPU. The shard pool manages:
  - GGUF split files (model-00001-of-N.gguf)
  - CPU/GPU layer offloading (--n-gpu-layers)
  - Memory-mapped weights (--mmap/--mlock)
  - Load/unload lifecycle with receipts

This is Level 3 in the specialist compute hierarchy:
  Level 1: Whole-model cold load (simple, VRAM-heavy)
  Level 2: Skill cartridges (modular capability packages)
  Level 3: Monolithic model sharding (split across devices/storage)

The shard pool wraps existing llama.cpp/GGUF mechanisms. It does NOT
implement custom tensor parallelism.
"""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ShardManifest:
    """Manifest for a sharded monolithic model."""
    model_id: str
    role: str = "specialist"  # specialist | research | escalation
    backend: str = "llama_cpp"
    load_mode: str = "cold"  # cold | warm
    shard_paths: list[str] = field(default_factory=list)  # absolute paths
    shard_hashes: list[str] = field(default_factory=list)  # SHA256 per shard
    quant_type: str = ""  # q5_k_m, q8_0, hxq_affine6, etc.
    codec: str = ""  # q5_k_m | q6_k | q8_0 | hxq_affine_6 | hxq_affine_g128
    offload_policy: dict = field(default_factory=dict)  # gpu_layers, cpu_layers, mmap, mlock
    required_ram_mb: int = 0
    required_vram_mb: int = 0
    context_size: int = 4096
    activation_intents: list[str] = field(default_factory=list)
    fallback: str = "qwen2.5-sentinel"
    fallback_shard: str = ""  # ID of fallback shard (standard GGUF) if HXQ fails
    idle_unload_s: int = 300
    eval_receipt: str = ""
    helix_codec_receipt: str = ""  # tensor fidelity receipt (HXQ only)
    behavioral_eval_receipt: str = ""  # behavioral eval receipt
    status: str = "candidate"  # active | candidate | disabled | quarantined
    system_prompt: str = ""
    manifest_dir: str = ""

    @classmethod
    def from_file(cls, manifest_path: str) -> "ShardManifest":
        """Load manifest from JSON file."""
        with open(manifest_path) as f:
            data = json.load(f)
        manifest_dir = str(Path(manifest_path).parent)
        offload = data.get("offload_policy", {})
        # codec defaults to quant_type if not explicitly set
        quant_type = data.get("quant_type", "")
        codec = data.get("codec", quant_type)
        return cls(
            model_id=data.get("model_id", Path(manifest_dir).name),
            role=data.get("role", "specialist"),
            backend=data.get("backend", "llama_cpp"),
            load_mode=data.get("load_mode", "cold"),
            shard_paths=data.get("shard_paths", data.get("shards", [])),
            shard_hashes=data.get("shard_hashes", []),
            quant_type=quant_type,
            codec=codec,
            offload_policy=offload,
            required_ram_mb=data.get("required_ram_mb", 0),
            required_vram_mb=data.get("required_vram_mb", 0),
            context_size=data.get("context_size", 4096),
            activation_intents=data.get("activation_intents", []),
            fallback=data.get("fallback", "qwen2.5-sentinel"),
            fallback_shard=data.get("fallback_shard", ""),
            idle_unload_s=data.get("idle_unload_s", 300),
            eval_receipt=data.get("eval_receipt", ""),
            helix_codec_receipt=data.get("helix_codec_receipt", ""),
            behavioral_eval_receipt=data.get("behavioral_eval_receipt", ""),
            status=data.get("status", "candidate"),
            system_prompt=data.get("system_prompt", ""),
            manifest_dir=manifest_dir,
        )

    def is_active(self) -> bool:
        return self.status == "active"

    def has_eval(self) -> bool:
        if not self.eval_receipt:
            return False
        path = os.path.join(self.manifest_dir, self.eval_receipt)
        return os.path.exists(path)

    def gpu_layers(self) -> int:
        return self.offload_policy.get("gpu_layers", 99)

    def uses_mmap(self) -> bool:
        return self.offload_policy.get("mmap", True)

    def primary_gguf(self) -> Optional[str]:
        """Return the first shard path (used as --model arg for llama-server)."""
        if self.shard_paths:
            return self.shard_paths[0]
        return None

    def all_shards_exist(self) -> bool:
        """Check all shard files exist on disk."""
        for p in self.shard_paths:
            if not os.path.exists(p):
                return False
        return len(self.shard_paths) > 0

    def total_size_mb(self) -> float:
        """Sum of all shard file sizes in MB."""
        total = 0
        for p in self.shard_paths:
            if os.path.exists(p):
                total += os.path.getsize(p)
        return round(total / (1024 * 1024), 1)

    def build_llama_args(self, port: int = 8080) -> list[str]:
        """Build llama-server command line arguments from manifest."""
        gguf = self.primary_gguf()
        if not gguf:
            return []
        args = [
            "--model", gguf,
            "--port", str(port),
            "--ctx-size", str(self.context_size),
            "--n-gpu-layers", str(self.gpu_layers()),
            "--host", "0.0.0.0",
        ]
        if self.uses_mmap():
            args.append("--mmap")
        if self.offload_policy.get("mlock"):
            args.append("--mlock")
        return args


class ShardPool:
    """Manages monolithic model shards — manifests, loading, offload, lifecycle."""

    def __init__(self, shard_dir: str = None):
        self._shards: dict[str, ShardManifest] = {}
        self._intent_index: dict[str, str] = {}  # intent → model_id
        self._loaded_id: Optional[str] = None
        self._load_log: list[dict] = []

        if shard_dir:
            self.load_manifests(shard_dir)

    def load_manifests(self, shard_dir: str):
        """Scan directory for shard manifest files."""
        base = Path(shard_dir)
        if not base.is_dir():
            return
        for child in sorted(base.iterdir()):
            manifest_path = child / "shard_manifest.json"
            if child.is_dir() and manifest_path.exists():
                try:
                    m = ShardManifest.from_file(str(manifest_path))
                    self._shards[m.model_id] = m
                    for intent in m.activation_intents:
                        self._intent_index[intent] = m.model_id
                except (json.JSONDecodeError, KeyError, OSError):
                    continue

    def register(self, manifest: ShardManifest):
        """Register a shard manifest directly."""
        self._shards[manifest.model_id] = manifest
        for intent in manifest.activation_intents:
            self._intent_index[intent] = manifest.model_id

    def route(self, intent: str) -> Optional[ShardManifest]:
        """Find active shard for an intent."""
        mid = self._intent_index.get(intent)
        if not mid:
            return None
        shard = self._shards.get(mid)
        if shard and shard.is_active():
            return shard
        return None

    def get(self, model_id: str) -> Optional[ShardManifest]:
        return self._shards.get(model_id)

    def list_shards(self) -> list[dict]:
        """List all registered shards with metadata."""
        return [
            {
                "model_id": s.model_id,
                "role": s.role,
                "backend": s.backend,
                "codec": s.codec,
                "quant_type": s.quant_type,
                "shard_count": len(s.shard_paths),
                "required_vram_mb": s.required_vram_mb,
                "required_ram_mb": s.required_ram_mb,
                "gpu_layers": s.gpu_layers(),
                "mmap": s.uses_mmap(),
                "activation_intents": s.activation_intents,
                "fallback_shard": s.fallback_shard,
                "status": s.status,
                "has_eval": s.has_eval(),
                "shards_on_disk": s.all_shards_exist(),
            }
            for s in self._shards.values()
        ]

    def check_resources(self, model_id: str, available_vram_mb: int = 4096,
                        available_ram_mb: int = 16384) -> dict:
        """Check if a sharded model can be loaded with available resources."""
        shard = self._shards.get(model_id)
        if not shard:
            return {"fits": False, "error": f"Unknown shard: {model_id}"}

        result = {
            "model_id": model_id,
            "shards_exist": shard.all_shards_exist(),
            "total_size_mb": shard.total_size_mb(),
            "required_vram_mb": shard.required_vram_mb,
            "required_ram_mb": shard.required_ram_mb,
            "available_vram_mb": available_vram_mb,
            "available_ram_mb": available_ram_mb,
        }

        if not shard.all_shards_exist():
            result["fits"] = False
            result["error"] = "Not all shard files found on disk"
            return result

        vram_ok = shard.required_vram_mb <= available_vram_mb
        ram_ok = shard.required_ram_mb <= available_ram_mb
        result["fits"] = vram_ok and ram_ok
        if not vram_ok:
            result["error"] = f"Need {shard.required_vram_mb}MB VRAM, have {available_vram_mb}MB"
        elif not ram_ok:
            result["error"] = f"Need {shard.required_ram_mb}MB RAM, have {available_ram_mb}MB"

        return result

    def build_load_plan(self, model_id: str, port: int = 8080) -> dict:
        """Build the load plan (llama-server args + offload policy) for a shard.

        Does NOT actually start the server — the model_pool backend handles that.
        Returns the plan as a dict for the backend to execute.
        """
        shard = self._shards.get(model_id)
        if not shard:
            return {"error": f"Unknown shard: {model_id}"}
        if not shard.is_active():
            return {"error": f"Shard {model_id} is {shard.status} (not active)"}
        if not shard.all_shards_exist():
            return {"error": f"Shard files missing for {model_id}"}

        args = shard.build_llama_args(port)
        if not args:
            return {"error": f"No GGUF path for {model_id}"}

        return {
            "model_id": model_id,
            "backend": shard.backend,
            "llama_args": args,
            "primary_gguf": shard.primary_gguf(),
            "shard_count": len(shard.shard_paths),
            "offload_policy": shard.offload_policy,
            "context_size": shard.context_size,
            "idle_unload_s": shard.idle_unload_s,
            "system_prompt": shard.system_prompt,
        }

    def record_load(self, model_id: str, event_type: str = "load",
                    wall_time_s: float = 0, error: str = None):
        """Record a load/unload event for audit."""
        self._load_log.append({
            "model_id": model_id,
            "event": event_type,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_time_s": wall_time_s,
            "error": error,
        })

    def get_load_log(self) -> list[dict]:
        return list(self._load_log)

    @property
    def loaded(self) -> Optional[str]:
        return self._loaded_id

    def __len__(self) -> int:
        return len(self._shards)

    def intent_map(self) -> dict[str, str]:
        return dict(self._intent_index)
