"""Specialist adapter interface for cell-runtime.

Allows domain-specific processing pipelines (e.g., Sentinel security triage)
to integrate with the generic cell orchestrator without copying specialist
code into the runtime package.

Pattern:
    orchestrator → specialist adapter → external specialist pipeline → result

The orchestrator handles model swapping. The specialist handles domain logic
(SSM state, feature extraction, post-LLM gating, etc.).
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Interface
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpecialistResult:
    """Standardized result from a specialist pipeline."""
    intent: str
    verdict: dict                   # domain-specific structured output
    output: str                     # human-readable output (for API response)
    state_summary: Optional[str]    # specialist state injected into LLM context
    receipt: dict                   # full audit trail
    specialist: str                 # adapter name
    wall_s: float = 0.0
    gate_fired: bool = False
    gate_rules: list = field(default_factory=list)


class SpecialistAdapter:
    """Base class for specialist adapters."""

    name: str = "base"

    def can_handle(self, intent: str) -> bool:
        """Return True if this specialist handles the given intent."""
        return False

    def handle(self, text: str, port: int,
               context: Optional[dict] = None) -> SpecialistResult:
        """Process input through the specialist pipeline.

        Args:
            text: raw user input / alert text
            port: llama-server port (model already loaded by orchestrator)
            context: optional orchestrator context (config, model name, etc.)
        """
        raise NotImplementedError

    def reset(self):
        """Reset specialist state (e.g., for new session)."""
        pass

    def state_dict(self) -> dict:
        """Return specialist state for status endpoint."""
        return {"specialist": self.name, "available": True}


# ═══════════════════════════════════════════════════════════════════════════
# Sentinel Hybrid Stack adapter
# ═══════════════════════════════════════════════════════════════════════════

# Manifest for frozen v0.1 — used for receipt provenance
_SENTINEL_MANIFEST = Path.home() / "receipts" / "sentinel_hybrid_stack_v0_1_manifest_20260430T150118Z.json"


def _load_manifest_hash() -> str:
    """SHA256 of the frozen manifest for receipt provenance."""
    try:
        return hashlib.sha256(_SENTINEL_MANIFEST.read_bytes()).hexdigest()[:16]
    except FileNotFoundError:
        return "manifest_not_found"


class SentinelHybridAdapter(SpecialistAdapter):
    """Adapter for Sentinel Hybrid Stack v0.1.

    Delegates to the frozen hybrid stack at tools/sentinel/ without
    copying any security-specific code into cell-runtime.

    Requires: tools/sentinel on PYTHONPATH (typically via PYTHONPATH=~/tools).
    """

    name = "sentinel-hybrid-v0.1"

    def __init__(self):
        self._stack = None
        self._available = None  # lazy check
        self._manifest_hash = None
        self._event_counter = 0

    def _ensure_loaded(self, port: int) -> bool:
        """Lazy import and initialize the hybrid stack."""
        if self._available is False:
            return False

        if self._stack is not None:
            # Already loaded — update port if changed
            if self._stack.port != port:
                self._stack.port = port
            return True

        try:
            from sentinel.runtime.hybrid_stack import HybridStack
            self._stack = HybridStack(port=port, use_grammar=False)
            self._available = True
            self._manifest_hash = _load_manifest_hash()
            log.info("SentinelHybridAdapter: loaded hybrid stack v0.1")
            return True
        except ImportError as e:
            self._available = False
            log.warning("SentinelHybridAdapter: sentinel not available (%s). "
                        "Ensure PYTHONPATH includes ~/tools or install cell-sentinel.", e)
            return False

    def can_handle(self, intent: str) -> bool:
        return intent == "security_triage"

    def handle(self, text: str, port: int,
               context: Optional[dict] = None) -> SpecialistResult:
        """Run alert through the frozen Sentinel Hybrid Stack v0.1."""
        t0 = time.time()

        if not self._ensure_loaded(port):
            raise ImportError("Sentinel hybrid stack not available")

        self._event_counter += 1
        event_id = self._event_counter

        result = self._stack.process_alert(event_id, text)

        # Build human-readable output from verdict
        verdict = result.final_verdict or {}
        output_lines = []
        if verdict.get("is_benign") is True:
            output_lines.append("VERDICT: BENIGN")
        elif verdict.get("is_benign") is False:
            output_lines.append(f"VERDICT: {verdict.get('severity', 'UNKNOWN').upper()}")
        else:
            output_lines.append("VERDICT: PARSE FAILED")

        if verdict.get("summary"):
            output_lines.append(f"Summary: {verdict['summary']}")
        if verdict.get("actions"):
            output_lines.append(f"Actions: {', '.join(verdict['actions'])}")

        if result.gate_result.gate_fired:
            output_lines.append(f"Gate correction: {result.gate_result.gate_reason}")

        output = "\n".join(output_lines)

        wall_s = round(time.time() - t0, 3)

        receipt = {
            "specialist": self.name,
            "manifest_hash": self._manifest_hash,
            "event_id": event_id,
            "ssm_step": self._stack.step,
            "gate_fired": result.gate_result.gate_fired,
            "gate_rules": result.gate_result.gate_rules,
            "original_verdict": result.gate_result.original_verdict,
            "final_verdict": verdict,
            "wall_s": wall_s,
        }

        return SpecialistResult(
            intent="security_triage",
            verdict=verdict,
            output=output,
            state_summary=result.ssm_state_summary,
            receipt=receipt,
            specialist=self.name,
            wall_s=wall_s,
            gate_fired=result.gate_result.gate_fired,
            gate_rules=result.gate_result.gate_rules,
        )

    def reset(self):
        """Reset SSM state for a new session."""
        if self._stack:
            self._stack.reset()
            self._event_counter = 0

    def state_dict(self) -> dict:
        """Return specialist state for status/debug."""
        base = {
            "specialist": self.name,
            "available": self._available is not False,
            "manifest_hash": self._manifest_hash,
        }
        if self._stack:
            from sentinel.memory.handmade_ssm import export_receipt
            base["ssm_step"] = self._stack.step
            base["ssm_entities"] = len(self._stack.ssm_state.entities)
            base["ssm_global_risk"] = round(self._stack.ssm_state.global_risk, 4)
            base["ssm_global_stage"] = self._stack.ssm_state.global_stage
            base["event_count"] = self._event_counter
        return base


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

def create_specialist_registry() -> list[SpecialistAdapter]:
    """Create the default set of specialist adapters.

    Each adapter lazily checks availability on first use.
    """
    return [
        SentinelHybridAdapter(),
    ]


def find_specialist(registry: list[SpecialistAdapter],
                    intent: str) -> Optional[SpecialistAdapter]:
    """Find the first specialist that can handle the given intent."""
    for adapter in registry:
        if adapter.can_handle(intent):
            return adapter
    return None
