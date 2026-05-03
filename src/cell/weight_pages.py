#!/usr/bin/env python3
"""Weight page library — addressable, verifiable tensor pages from GGUF models.

Provides OS-style virtual memory for model weights:
  - Manifest: directory of all tensor pages with metadata
  - Loader: mmap-based page loading into CPU RAM
  - GPU transfer: copy pages to CUDA device memory
  - Verification: SHA256 hash check on load
  - Eviction: free pages from RAM/GPU
  - Receipts: audit trail for every page operation

This is the physical paging layer. It does NOT claim semantic cartridge behavior.
Semantic cartridge discovery happens later through ablation/eval.
"""
import hashlib
import json
import mmap
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PageEntry:
    """A single tensor page in the weight library."""
    tensor_name: str
    layer_id: Optional[int]
    tensor_role: str
    byte_offset: int
    byte_length: int
    shape: list[int]
    dtype_or_quant: str
    sha256: Optional[str] = None
    residency: str = "disk"      # disk | ram | gpu
    hotness_score: float = 0.0
    last_used: Optional[str] = None
    receipt_id: Optional[str] = None

    # Runtime state (not serialized)
    _ram_buffer: Optional[bytes] = field(default=None, repr=False)
    _gpu_tensor: object = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "tensor_name": self.tensor_name,
            "layer_id": self.layer_id,
            "tensor_role": self.tensor_role,
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
            "shape": self.shape,
            "dtype_or_quant": self.dtype_or_quant,
            "sha256": self.sha256,
            "residency": self.residency,
            "hotness_score": self.hotness_score,
            "last_used": self.last_used,
            "receipt_id": self.receipt_id,
        }


class WeightPageLibrary:
    """Addressable weight-page library for a GGUF model."""

    def __init__(self, manifest_path: str):
        manifest_path = os.path.expanduser(manifest_path)
        with open(manifest_path) as f:
            data = json.load(f)

        self.model_id = data["model_id"]
        self.gguf_path = data["gguf_path"]
        self.total_tensors = data["total_tensors"]
        self.total_layers = data["total_layers"]

        self.pages: dict[str, PageEntry] = {}
        for p in data["pages"]:
            entry = PageEntry(
                tensor_name=p["tensor_name"],
                layer_id=p.get("layer_id"),
                tensor_role=p["tensor_role"],
                byte_offset=p["byte_offset"],
                byte_length=p["byte_length"],
                shape=p["shape"],
                dtype_or_quant=p["dtype_or_quant"],
                sha256=p.get("sha256"),
                residency="disk",
            )
            self.pages[entry.tensor_name] = entry

        self._mmap = None
        self._file = None
        self._ops_log: list[dict] = []

    def _ensure_mmap(self):
        """Open the GGUF file and create a memory map."""
        if self._mmap is None:
            self._file = open(self.gguf_path, 'rb')
            self._mmap = mmap.mmap(
                self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def load_page(self, tensor_name: str, verify: bool = True) -> bytes:
        """Load a tensor page into CPU RAM. Returns raw bytes.

        Args:
            tensor_name: name of the tensor to load
            verify: if True and page has sha256, verify hash on load
        """
        if tensor_name not in self.pages:
            raise KeyError(f"Unknown tensor: {tensor_name}")

        page = self.pages[tensor_name]

        # Already in RAM?
        if page._ram_buffer is not None and page.residency in ("ram", "gpu"):
            page.last_used = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            page.hotness_score += 1.0
            return page._ram_buffer

        self._ensure_mmap()
        t0 = time.time()

        # Read the page
        data = self._mmap[page.byte_offset:page.byte_offset + page.byte_length]

        # Verify if hash is known
        if verify and page.sha256:
            actual = hashlib.sha256(data).hexdigest()
            if actual != page.sha256:
                self._log_op("verify_fail", tensor_name, {
                    "expected": page.sha256[:16],
                    "actual": actual[:16],
                })
                raise ValueError(
                    f"SHA256 mismatch for {tensor_name}: "
                    f"expected {page.sha256[:16]}..., got {actual[:16]}...")

        page._ram_buffer = data
        page.residency = "ram"
        page.last_used = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        page.hotness_score += 1.0

        load_ms = round((time.time() - t0) * 1000, 2)
        self._log_op("load_ram", tensor_name, {
            "bytes": page.byte_length,
            "load_ms": load_ms,
            "verified": verify and page.sha256 is not None,
        })

        return data

    def copy_to_gpu(self, tensor_name: str, device: int = 0):
        """Copy a page from RAM to GPU. Returns a torch tensor (raw bytes view).

        Requires torch with CUDA. The tensor is stored as uint8 — the caller
        or inference engine handles interpretation based on dtype_or_quant.
        """
        page = self.pages[tensor_name]
        if page._ram_buffer is None:
            self.load_page(tensor_name)

        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
        except ImportError:
            raise RuntimeError("torch not installed")

        t0 = time.time()
        # Transfer raw bytes as uint8 tensor
        cpu_tensor = torch.frombuffer(
            bytearray(page._ram_buffer), dtype=torch.uint8)
        gpu_tensor = cpu_tensor.to(f"cuda:{device}")

        page._gpu_tensor = gpu_tensor
        page.residency = "gpu"
        page.last_used = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        copy_ms = round((time.time() - t0) * 1000, 2)
        self._log_op("copy_gpu", tensor_name, {
            "bytes": page.byte_length,
            "copy_ms": copy_ms,
            "device": device,
        })
        return gpu_tensor

    def evict_page(self, tensor_name: str, from_gpu: bool = True,
                   from_ram: bool = True):
        """Evict a page from GPU and/or RAM."""
        page = self.pages[tensor_name]
        if from_gpu and page._gpu_tensor is not None:
            del page._gpu_tensor
            page._gpu_tensor = None
        if from_ram:
            page._ram_buffer = None
            page.residency = "disk"
        elif from_gpu:
            page.residency = "ram"

        self._log_op("evict", tensor_name, {
            "from_gpu": from_gpu,
            "from_ram": from_ram,
        })

    def get_layer_pages(self, layer_id: int) -> list[PageEntry]:
        """Get all pages for a given layer."""
        return [p for p in self.pages.values() if p.layer_id == layer_id]

    def get_pages_by_role(self, role: str) -> list[PageEntry]:
        """Get all pages matching a tensor role."""
        return [p for p in self.pages.values() if p.tensor_role == role]

    def resident_pages(self) -> dict[str, list[str]]:
        """Return pages grouped by residency."""
        result = {"disk": [], "ram": [], "gpu": []}
        for name, page in self.pages.items():
            result[page.residency].append(name)
        return result

    def ram_bytes(self) -> int:
        """Total bytes currently in RAM."""
        return sum(p.byte_length for p in self.pages.values()
                   if p.residency in ("ram", "gpu"))

    def gpu_bytes(self) -> int:
        """Total bytes currently on GPU."""
        return sum(p.byte_length for p in self.pages.values()
                   if p.residency == "gpu")

    def summary(self) -> dict:
        """Summary statistics."""
        res = self.resident_pages()
        return {
            "model_id": self.model_id,
            "total_tensors": self.total_tensors,
            "total_layers": self.total_layers,
            "on_disk": len(res["disk"]),
            "in_ram": len(res["ram"]),
            "on_gpu": len(res["gpu"]),
            "ram_bytes": self.ram_bytes(),
            "gpu_bytes": self.gpu_bytes(),
        }

    def _log_op(self, op: str, tensor_name: str, details: dict):
        self._ops_log.append({
            "op": op,
            "tensor": tensor_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            **details,
        })

    def get_ops_log(self) -> list[dict]:
        return list(self._ops_log)

    def close(self):
        """Release mmap and file handles."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None
        # Evict everything
        for name in list(self.pages.keys()):
            self.evict_page(name)
