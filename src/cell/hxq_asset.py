"""HXQ asset validation for specialist compute pool.

Every shard and cartridge can exist in multiple codec formats:
  - Q5_K_M / Q6_K / Q8_0  — standard GGUF baseline (control group)
  - HXQ_AFFINE_6           — GPU edge, tight memory
  - HXQ_AFFINE_G128        — CPU-friendly, safer packing

The rule: No HXQ promotion without behavioral eval.

Lifecycle:
  1. Prove with normal GGUF quants first
  2. Convert selected assets to HXQ
  3. Re-run same evals (same prompts, same tasks, same scoring)
  4. Promote HXQ only if it preserves behavior

This module validates that HXQ assets have the required receipts
before they can be promoted to active status.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional


# Valid codec types
BASELINE_CODECS = {"q5_k_m", "q6_k", "q8_0", "fp16", "bf16"}
HXQ_CODECS = {"hxq_affine_6", "hxq_affine_g128"}
ALL_CODECS = BASELINE_CODECS | HXQ_CODECS | {"prompt_pack", "lora_adapter", "rag_index", ""}


@dataclass
class HXQAssetReceipt:
    """Validated receipt for an HXQ-compressed asset."""
    asset_id: str
    asset_type: str = ""  # llm_weight_shard | cartridge_tensor | adapter
    codec: str = ""  # hxq_affine_6 | hxq_affine_g128
    group_size: int = 128
    sha256_original: str = ""
    sha256_compressed: str = ""
    cosine_min: float = 0.0
    max_abs_diff: float = 0.0
    ppl_baseline: float = 0.0
    ppl_compressed: float = 0.0
    ppl_delta_pct: float = 0.0
    behavioral_eval_pass: bool = False
    behavioral_eval_receipt: str = ""
    runtime_status: str = "candidate"  # candidate | active | quarantined

    @classmethod
    def from_file(cls, path: str) -> "HXQAssetReceipt":
        """Load receipt from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            asset_id=data.get("asset_id", ""),
            asset_type=data.get("asset_type", ""),
            codec=data.get("codec", ""),
            group_size=data.get("group_size", 128),
            sha256_original=data.get("sha256_original", ""),
            sha256_compressed=data.get("sha256_compressed", ""),
            cosine_min=data.get("cosine_min", 0.0),
            max_abs_diff=data.get("max_abs_diff", 0.0),
            ppl_baseline=data.get("ppl_baseline", 0.0),
            ppl_compressed=data.get("ppl_compressed", 0.0),
            ppl_delta_pct=data.get("ppl_delta_pct", 0.0),
            behavioral_eval_pass=data.get("behavioral_eval_pass", False),
            behavioral_eval_receipt=data.get("behavioral_eval_receipt", ""),
            runtime_status=data.get("runtime_status", "candidate"),
        )

    def tensor_fidelity_pass(self, min_cosine: float = 0.998) -> bool:
        """Check if tensor fidelity meets threshold."""
        return self.cosine_min >= min_cosine

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "codec": self.codec,
            "group_size": self.group_size,
            "sha256_original": self.sha256_original,
            "sha256_compressed": self.sha256_compressed,
            "cosine_min": self.cosine_min,
            "max_abs_diff": self.max_abs_diff,
            "ppl_baseline": self.ppl_baseline,
            "ppl_compressed": self.ppl_compressed,
            "ppl_delta_pct": self.ppl_delta_pct,
            "behavioral_eval_pass": self.behavioral_eval_pass,
            "behavioral_eval_receipt": self.behavioral_eval_receipt,
            "runtime_status": self.runtime_status,
        }


def is_hxq_codec(codec: str) -> bool:
    """Check if codec is an HXQ type."""
    return codec.lower() in HXQ_CODECS


def is_baseline_codec(codec: str) -> bool:
    """Check if codec is a standard GGUF baseline."""
    return codec.lower() in BASELINE_CODECS


def validate_hxq_asset(manifest_dir: str, codec: str,
                       helix_receipt_path: str = "",
                       behavioral_receipt_path: str = "") -> dict:
    """Validate that an HXQ asset has the required receipts for promotion.

    Returns dict with 'valid' bool and list of 'issues'.
    """
    issues = []

    if not is_hxq_codec(codec):
        return {"valid": True, "issues": [], "note": f"Not HXQ ({codec}), no HXQ validation needed"}

    # HXQ assets require tensor fidelity receipt
    if not helix_receipt_path:
        issues.append("Missing helix_codec_receipt — tensor fidelity receipt required for HXQ assets")
    else:
        abs_path = os.path.join(manifest_dir, helix_receipt_path) if not os.path.isabs(helix_receipt_path) else helix_receipt_path
        if not os.path.exists(abs_path):
            issues.append(f"helix_codec_receipt not found on disk: {helix_receipt_path}")
        else:
            try:
                receipt = HXQAssetReceipt.from_file(abs_path)
                if not receipt.tensor_fidelity_pass():
                    issues.append(f"Tensor fidelity below threshold: cosine_min={receipt.cosine_min} (need >=0.998)")
                if not receipt.sha256_compressed:
                    issues.append("Missing sha256_compressed in receipt")
            except (json.JSONDecodeError, OSError) as e:
                issues.append(f"Cannot parse helix_codec_receipt: {e}")

    # HXQ assets require behavioral eval receipt for promotion
    if not behavioral_receipt_path:
        issues.append("Missing behavioral_eval_receipt — required for HXQ promotion")
    else:
        abs_path = os.path.join(manifest_dir, behavioral_receipt_path) if not os.path.isabs(behavioral_receipt_path) else behavioral_receipt_path
        if not os.path.exists(abs_path):
            issues.append(f"behavioral_eval_receipt not found on disk: {behavioral_receipt_path}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "codec": codec,
    }


def can_promote(codec: str, helix_receipt_path: str = "",
                behavioral_receipt_path: str = "",
                manifest_dir: str = "") -> dict:
    """Check if an asset can be promoted from candidate to active.

    Baseline codecs: only need behavioral eval.
    HXQ codecs: need tensor fidelity + behavioral eval.
    """
    if is_baseline_codec(codec):
        # Baselines only need behavioral eval
        if not behavioral_receipt_path:
            return {"promotable": False, "reason": "Missing behavioral eval receipt"}
        abs_path = os.path.join(manifest_dir, behavioral_receipt_path) if manifest_dir and not os.path.isabs(behavioral_receipt_path) else behavioral_receipt_path
        if abs_path and not os.path.exists(abs_path):
            return {"promotable": False, "reason": f"Behavioral eval receipt not found: {behavioral_receipt_path}"}
        return {"promotable": True, "reason": f"Baseline codec ({codec}) with behavioral eval"}

    if is_hxq_codec(codec):
        result = validate_hxq_asset(manifest_dir or "", codec, helix_receipt_path, behavioral_receipt_path)
        if result["valid"]:
            return {"promotable": True, "reason": f"HXQ codec ({codec}) with tensor + behavioral eval"}
        return {"promotable": False, "reason": f"HXQ validation failed: {'; '.join(result['issues'])}"}

    # Non-model codecs (prompt_pack, rag_index, etc.) — always promotable
    return {"promotable": True, "reason": f"Non-model codec ({codec}), no quant validation needed"}


def select_codec_for_target(target: str) -> str:
    """Recommend codec for a deployment target.

    target: 'gpu_edge' | 'cpu_fallback' | 'baseline' | 'pod'
    """
    recommendations = {
        "gpu_edge": "hxq_affine_6",
        "cpu_fallback": "hxq_affine_g128",
        "baseline": "q5_k_m",
        "pod": "q8_0",
        "production": "q5_k_m",
    }
    return recommendations.get(target, "q5_k_m")
