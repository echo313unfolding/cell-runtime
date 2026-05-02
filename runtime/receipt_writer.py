"""Receipt writer — standardized receipt emission for the cell daemon.

Every action through the daemon gets a receipt with cost block.
"""
import json
import os
import platform
import resource
import time
from pathlib import Path


def write_receipt(
    action: str,
    model: str,
    result: dict,
    receipt_dir: str = "~/receipts/cell/",
    extra: dict = None,
    t_start: float = None,
    cpu_start: float = None,
    start_iso: str = None,
) -> str:
    """Write a receipt JSON file. Returns the path."""
    now = time.strftime("%Y%m%dT%H%M%SZ")
    receipt_id = f"cell_{action}_{now}"

    cost = {
        "wall_time_s": round(time.time() - t_start, 3) if t_start else 0,
        "cpu_time_s": round(time.process_time() - cpu_start, 3) if cpu_start else 0,
        "peak_memory_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso or "",
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    receipt = {
        "receipt_id": receipt_id,
        "action": action,
        "model": model,
        "cost": cost,
    }
    receipt.update(result)
    if extra:
        receipt.update(extra)

    out_dir = Path(os.path.expanduser(receipt_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{receipt_id}.json"
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2, default=str)

    return str(path)
