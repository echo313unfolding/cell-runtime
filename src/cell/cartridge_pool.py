"""Cartridge pool — manifest-driven specialist capability loading.

A cartridge is a capability package, NOT a monolithic model. It may contain:
  - LoRA adapter
  - Small specialist GGUF model
  - Grammar/schema pack
  - RAG index
  - Prompt pack
  - Policy file
  - Eval receipt
  - Compressed HXQ/GGUF artifact

The pool loads manifests, routes by activation intent, loads only the needed
artifacts, runs through the base model backend, and emits receipts.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CartridgeManifest:
    """Parsed cartridge manifest."""
    cartridge_id: str
    cartridge_dir: str  # absolute path to cartridge directory
    activation_intents: list[str] = field(default_factory=list)
    base_model: str = "qwen2.5-sentinel"
    artifact_type: str = "prompt_pack"  # lora_adapter | gguf_model | grammar_pack | rag_index | prompt_pack | policy | hxq_shard
    artifact_path: str = ""  # relative to cartridge_dir
    rag_index_path: str = ""
    eval_receipt: str = ""
    status: str = "active"  # active | disabled | candidate
    scope: str = ""
    max_context_tokens: int = 4096
    fallback: str = "qwen2.5-sentinel"
    requires_ask_pass: bool = False
    codec: str = ""  # q5_k_m | hxq_affine_6 | hxq_affine_g128 | "" (non-model)
    sha256: str = ""
    helix_codec_receipt: str = ""  # tensor fidelity receipt (HXQ only)
    behavioral_eval_receipt: str = ""  # behavioral eval receipt
    system_prompt: str = ""
    examples_path: str = ""
    grammar_path: str = ""
    policy_path: str = ""

    @classmethod
    def from_file(cls, manifest_path: str) -> "CartridgeManifest":
        """Load manifest from JSON file."""
        with open(manifest_path) as f:
            data = json.load(f)
        cartridge_dir = str(Path(manifest_path).parent)
        return cls(
            cartridge_id=data.get("cartridge_id", Path(cartridge_dir).name),
            cartridge_dir=cartridge_dir,
            activation_intents=data.get("activation_intents", []),
            base_model=data.get("base_model", "qwen2.5-sentinel"),
            artifact_type=data.get("artifact_type", "prompt_pack"),
            artifact_path=data.get("artifact_path", ""),
            rag_index_path=data.get("rag_index_path", ""),
            eval_receipt=data.get("eval_receipt", ""),
            status=data.get("status", "active"),
            scope=data.get("scope", ""),
            max_context_tokens=data.get("max_context_tokens", 4096),
            fallback=data.get("fallback", "qwen2.5-sentinel"),
            requires_ask_pass=data.get("requires_ask_pass", False),
            codec=data.get("codec", ""),
            sha256=data.get("sha256", ""),
            helix_codec_receipt=data.get("helix_codec_receipt", ""),
            behavioral_eval_receipt=data.get("behavioral_eval_receipt", ""),
            system_prompt=data.get("system_prompt", ""),
            examples_path=data.get("examples_path", ""),
            grammar_path=data.get("grammar_path", ""),
            policy_path=data.get("policy_path", ""),
        )

    def resolve_path(self, relative: str) -> str:
        """Resolve a relative path against the cartridge directory."""
        if not relative:
            return ""
        return os.path.join(self.cartridge_dir, relative)

    def is_active(self) -> bool:
        return self.status == "active"

    def has_eval(self) -> bool:
        if not self.eval_receipt:
            return False
        return os.path.exists(self.resolve_path(self.eval_receipt))


class CartridgePool:
    """Manages skill cartridges — loads manifests, routes by intent, dispatches."""

    def __init__(self, cartridge_dir: str = None):
        self._cartridges: dict[str, CartridgeManifest] = {}
        self._intent_index: dict[str, str] = {}  # intent → cartridge_id
        self._loaded_id: Optional[str] = None
        self._loaded_artifacts: dict = {}
        self._activation_log: list[dict] = []

        if cartridge_dir:
            self.load_manifests(cartridge_dir)

    def load_manifests(self, cartridge_dir: str):
        """Scan cartridge directory for manifest.json files."""
        base = Path(cartridge_dir)
        if not base.is_dir():
            return
        for child in sorted(base.iterdir()):
            manifest_path = child / "manifest.json"
            if child.is_dir() and manifest_path.exists():
                try:
                    m = CartridgeManifest.from_file(str(manifest_path))
                    self._cartridges[m.cartridge_id] = m
                    for intent in m.activation_intents:
                        self._intent_index[intent] = m.cartridge_id
                except (json.JSONDecodeError, KeyError, OSError):
                    continue

    def register(self, manifest: CartridgeManifest):
        """Register a cartridge manifest directly."""
        self._cartridges[manifest.cartridge_id] = manifest
        for intent in manifest.activation_intents:
            self._intent_index[intent] = manifest.cartridge_id

    def route(self, intent: str) -> Optional[CartridgeManifest]:
        """Find the active cartridge for an intent."""
        cid = self._intent_index.get(intent)
        if not cid:
            return None
        cart = self._cartridges.get(cid)
        if cart and cart.is_active():
            return cart
        return None

    def get(self, cartridge_id: str) -> Optional[CartridgeManifest]:
        """Get cartridge by ID."""
        return self._cartridges.get(cartridge_id)

    def list_cartridges(self) -> list[dict]:
        """List all registered cartridges with metadata."""
        return [
            {
                "cartridge_id": c.cartridge_id,
                "activation_intents": c.activation_intents,
                "artifact_type": c.artifact_type,
                "base_model": c.base_model,
                "status": c.status,
                "scope": c.scope,
                "has_eval": c.has_eval(),
                "requires_ask_pass": c.requires_ask_pass,
            }
            for c in self._cartridges.values()
        ]

    def load_cartridge(self, cartridge_id: str) -> dict:
        """Load artifacts for a cartridge. Returns loaded artifact info."""
        cart = self._cartridges.get(cartridge_id)
        if not cart:
            return {"error": f"Unknown cartridge: {cartridge_id}"}
        if not cart.is_active():
            return {"error": f"Cartridge {cartridge_id} is {cart.status}"}

        artifacts = {"cartridge_id": cartridge_id, "loaded": []}

        # Load system prompt
        if cart.system_prompt:
            artifacts["system_prompt"] = cart.system_prompt
            artifacts["loaded"].append("system_prompt")

        # Load examples if present
        examples_path = cart.resolve_path(cart.examples_path)
        if examples_path and os.path.exists(examples_path):
            try:
                with open(examples_path) as f:
                    artifacts["examples"] = f.read(20000)
                artifacts["loaded"].append("examples")
            except OSError:
                pass

        # Load grammar if present
        grammar_path = cart.resolve_path(cart.grammar_path)
        if grammar_path and os.path.exists(grammar_path):
            try:
                with open(grammar_path) as f:
                    artifacts["grammar"] = json.load(f)
                artifacts["loaded"].append("grammar")
            except (OSError, json.JSONDecodeError):
                pass

        # Load policy if present
        policy_path = cart.resolve_path(cart.policy_path)
        if policy_path and os.path.exists(policy_path):
            try:
                with open(policy_path) as f:
                    artifacts["policy"] = f.read(5000)
                artifacts["loaded"].append("policy")
            except OSError:
                pass

        # Note artifact type for the backend
        artifacts["artifact_type"] = cart.artifact_type
        artifacts["base_model"] = cart.base_model
        artifacts["max_context_tokens"] = cart.max_context_tokens

        # LoRA/GGUF paths (not loaded into memory here — backend handles that)
        if cart.artifact_type in ("lora_adapter", "gguf_model", "hxq_shard"):
            artifact_abs = cart.resolve_path(cart.artifact_path)
            if artifact_abs and os.path.exists(artifact_abs):
                artifacts["artifact_file"] = artifact_abs
                artifacts["loaded"].append(cart.artifact_type)
            else:
                artifacts["artifact_missing"] = cart.artifact_path

        self._loaded_id = cartridge_id
        self._loaded_artifacts = artifacts
        return artifacts

    def unload(self):
        """Unload current cartridge."""
        self._loaded_id = None
        self._loaded_artifacts = {}

    def build_prompt(self, cartridge_id: str, task_prompt: str) -> str:
        """Build an augmented prompt using cartridge artifacts.

        Assembles: system_prompt + examples + grammar constraints + task_prompt.
        """
        artifacts = self._loaded_artifacts
        if not artifacts or artifacts.get("cartridge_id") != cartridge_id:
            artifacts = self.load_cartridge(cartridge_id)
            if "error" in artifacts:
                return task_prompt

        parts = []

        if artifacts.get("system_prompt"):
            parts.append(artifacts["system_prompt"])

        if artifacts.get("examples"):
            parts.append(f"\n--- Examples ---\n{artifacts['examples'][:8000]}")

        if artifacts.get("grammar"):
            parts.append(f"\n--- Output Schema ---\n{json.dumps(artifacts['grammar'], indent=2)[:2000]}")

        if artifacts.get("policy"):
            parts.append(f"\n--- Policy ---\n{artifacts['policy'][:2000]}")

        parts.append(f"\n--- Task ---\n{task_prompt}")

        return "\n".join(parts)

    def dispatch(self, intent: str, task_prompt: str, orchestrator=None) -> dict:
        """Route intent to cartridge, build prompt, run through backend.

        Returns structured result dict.
        """
        t0 = time.time()

        # Route
        cart = self.route(intent)
        if not cart:
            return {"error": f"No active cartridge for intent: {intent}"}

        # Load
        load_result = self.load_cartridge(cart.cartridge_id)
        if "error" in load_result:
            return load_result

        # Build augmented prompt
        prompt = self.build_prompt(cart.cartridge_id, task_prompt)

        # Execute through orchestrator or return prompt for external execution
        result = {
            "cartridge_id": cart.cartridge_id,
            "intent": intent,
            "artifact_type": cart.artifact_type,
            "base_model": cart.base_model,
            "artifacts_loaded": load_result.get("loaded", []),
        }

        if orchestrator:
            gen_result = orchestrator.process(
                prompt,
                force_model=cart.base_model,
                use_tools=False,
            )
            if "error" in gen_result:
                result["error"] = f"Cartridge execution error: {gen_result['error']}"
            else:
                result["output"] = gen_result.get("output", "")
                result["model"] = gen_result.get("model", cart.base_model)
                result["wall_time_s"] = gen_result.get("wall_time_s", 0)
                result["swapped"] = gen_result.get("swapped", False)
        else:
            # No orchestrator — return the assembled prompt for external use
            result["assembled_prompt"] = prompt
            result["output"] = ""

        result["cartridge_wall_time_s"] = round(time.time() - t0, 3)

        # Log activation
        activation = {
            "cartridge_id": cart.cartridge_id,
            "intent": intent,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_time_s": result["cartridge_wall_time_s"],
            "error": result.get("error"),
        }
        self._activation_log.append(activation)

        # Unload
        self.unload()

        return result

    def get_activation_log(self) -> list[dict]:
        return list(self._activation_log)

    @property
    def loaded(self) -> Optional[str]:
        return self._loaded_id

    def __len__(self) -> int:
        return len(self._cartridges)

    def intent_map(self) -> dict[str, str]:
        """Return intent → cartridge_id mapping."""
        return dict(self._intent_index)
