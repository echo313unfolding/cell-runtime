"""Specialist compute agents — two-layer routing: cartridge first, shard second.

The specialist compute pool has two tiers:
  1. Cartridge pool — task-specific skill packages (fast, modular)
  2. Shard pool — monolithic large models split across CPU/GPU/disk (heavy, slow)

Routing: normal → smaLLM/Sentinel, specialized → cartridge, deep/heavy → shard.

No specialist executes privileged actions. All outputs are proposals.
"""
import os
from cell.agents.base import AgentBase, AgentResult, Permission
from cell.cartridge_pool import CartridgePool
from cell.shard_pool import ShardPool, ShardManifest


# Shared shard pool — initialized once
_shard_pool: ShardPool | None = None


def get_shard_pool(shard_dir: str = None) -> ShardPool:
    """Get or create the shared shard pool."""
    global _shard_pool
    if _shard_pool is None:
        if shard_dir is None:
            # Default: cell-runtime/shards/
            shard_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))),
                "shards",
            )
        _shard_pool = ShardPool(shard_dir)
    return _shard_pool


def reset_shard_pool():
    """Reset the shared shard pool (for testing)."""
    global _shard_pool
    _shard_pool = None


class SpecialistComputeRouteAgent(AgentBase):
    """Route a task through the specialist compute hierarchy.

    Checks cartridge pool first. If no cartridge matches or cartridge is
    insufficient, checks shard pool. Returns the routing plan (does NOT
    execute the model — the orchestrator does that).
    """
    name = "specialist_compute_route"
    description = (
        "Route a specialist task through the compute hierarchy: "
        "cartridge pool first (fast, modular), shard pool second (heavy, deep). "
        "Returns routing plan with selected backend. Does NOT execute."
    )
    permission = Permission.READ
    timeout_s = 30
    input_schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "Task intent (e.g., code_repair, deep_code_analysis, patch_review)",
            },
            "task": {
                "type": "string",
                "description": "Task description",
            },
            "prefer_shard": {
                "type": "string",
                "description": "If set, prefer this shard model_id over cartridge",
            },
        },
        "required": ["intent", "task"],
    }

    _orchestrator = None

    def execute(self, args: dict) -> AgentResult:
        intent = args["intent"]
        task = args["task"]
        prefer_shard = args.get("prefer_shard", "")

        # Import here to avoid circular dependency
        from cell.agents.cartridge_agent import get_cartridge_pool

        cart_pool = get_cartridge_pool()
        shard_pool = get_shard_pool()

        # If prefer_shard is set, go directly to shard
        if prefer_shard:
            shard = shard_pool.get(prefer_shard)
            if shard and shard.is_active():
                plan = shard_pool.build_load_plan(prefer_shard)
                return AgentResult(output={
                    "route": "shard",
                    "model_id": prefer_shard,
                    "plan": plan,
                    "reason": f"Preferred shard: {prefer_shard}",
                })

        # Level 2: Check cartridge pool
        cart = cart_pool.route(intent)
        if cart:
            return AgentResult(output={
                "route": "cartridge",
                "cartridge_id": cart.cartridge_id,
                "artifact_type": cart.artifact_type,
                "base_model": cart.base_model,
                "reason": f"Cartridge matched intent '{intent}'",
            })

        # Level 3: Check shard pool
        shard = shard_pool.route(intent)
        if shard:
            plan = shard_pool.build_load_plan(shard.model_id)
            return AgentResult(output={
                "route": "shard",
                "model_id": shard.model_id,
                "plan": plan,
                "reason": f"Shard matched intent '{intent}'",
            })

        # Fallback: no specialist available
        return AgentResult(output={
            "route": "fallback",
            "model_id": "qwen2.5-sentinel",
            "reason": f"No cartridge or shard for intent '{intent}'. Falling back to Sentinel.",
        })


class ShardListAgent(AgentBase):
    """List available monolithic model shards and their status."""
    name = "shard_list"
    description = "List all registered model shards with status, offload policy, and resource requirements."
    permission = Permission.READ
    timeout_s = 10
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: dict) -> AgentResult:
        pool = get_shard_pool()
        return AgentResult(output={
            "shards": pool.list_shards(),
            "intent_map": pool.intent_map(),
            "count": len(pool),
        })


class ShardResourceCheckAgent(AgentBase):
    """Check if a sharded model fits available resources."""
    name = "shard_resource_check"
    description = (
        "Check if a sharded model can be loaded given available VRAM and RAM. "
        "Returns fit/no-fit with details."
    )
    permission = Permission.READ
    timeout_s = 10
    input_schema = {
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "description": "Shard model ID to check"},
            "available_vram_mb": {"type": "integer", "description": "Available GPU VRAM in MB (default: 4096)"},
            "available_ram_mb": {"type": "integer", "description": "Available system RAM in MB (default: 16384)"},
        },
        "required": ["model_id"],
    }

    def execute(self, args: dict) -> AgentResult:
        pool = get_shard_pool()
        vram = args.get("available_vram_mb", 4096)
        ram = args.get("available_ram_mb", 16384)
        result = pool.check_resources(args["model_id"], vram, ram)
        return AgentResult(output=result)
