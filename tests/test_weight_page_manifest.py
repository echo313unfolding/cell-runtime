"""Tests for weight page manifest builder and library."""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

GGUF_PATH = os.path.expanduser(
    "~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf")
HAS_GGUF = os.path.exists(GGUF_PATH)


@pytest.mark.skipif(not HAS_GGUF, reason="GGUF model not found")
class TestManifestBuilder:
    def test_build_manifest(self):
        from build_weight_page_manifest import build_manifest
        m = build_manifest(GGUF_PATH, verify_n=0)
        assert m["total_tensors"] > 0
        assert m["total_layers"] > 0
        assert len(m["pages"]) == m["total_tensors"]

    def test_manifest_page_fields(self):
        from build_weight_page_manifest import build_manifest
        m = build_manifest(GGUF_PATH, verify_n=0)
        page = m["pages"][0]
        required = ["tensor_name", "layer_id", "tensor_role",
                     "byte_offset", "byte_length", "shape",
                     "dtype_or_quant", "residency"]
        for field in required:
            assert field in page, f"Missing field: {field}"

    def test_verify_hashes(self):
        from build_weight_page_manifest import build_manifest
        m = build_manifest(GGUF_PATH, verify_n=3)
        assert m["verified_count"] == 3
        verified = [p for p in m["pages"] if p["sha256"] is not None]
        assert len(verified) == 3
        for p in verified:
            assert len(p["sha256"]) == 64  # SHA256 hex

    def test_layer_role_parsing(self):
        from build_weight_page_manifest import parse_tensor_role
        assert parse_tensor_role("blk.5.attn_q.weight") == (5, "attn_q")
        assert parse_tensor_role("blk.0.ffn_down.weight") == (0, "ffn_down")
        assert parse_tensor_role("token_embd.weight") == (None, "embed")
        assert parse_tensor_role("output.weight") == (None, "output_head")

    def test_all_offsets_within_file(self):
        from build_weight_page_manifest import build_manifest
        m = build_manifest(GGUF_PATH, verify_n=0)
        file_size = m["file_size_bytes"]
        for p in m["pages"]:
            assert p["byte_offset"] >= 0
            assert p["byte_offset"] + p["byte_length"] <= file_size, \
                f"{p['tensor_name']} exceeds file bounds"


@pytest.mark.skipif(not HAS_GGUF, reason="GGUF model not found")
class TestWeightPageLibrary:
    @pytest.fixture
    def manifest_path(self, tmp_path):
        from build_weight_page_manifest import build_manifest
        m = build_manifest(GGUF_PATH, verify_n=5)
        path = str(tmp_path / "manifest.json")
        with open(path, 'w') as f:
            json.dump(m, f)
        return path

    def test_load_library(self, manifest_path):
        from cell.weight_pages import WeightPageLibrary
        lib = WeightPageLibrary(manifest_path)
        assert lib.total_tensors > 0
        assert len(lib.pages) == lib.total_tensors
        lib.close()

    def test_load_page_to_ram(self, manifest_path):
        from cell.weight_pages import WeightPageLibrary
        lib = WeightPageLibrary(manifest_path)
        name = list(lib.pages.keys())[0]
        page = lib.pages[name]
        data = lib.load_page(name, verify=False)
        assert len(data) == page.byte_length
        assert page.residency == "ram"
        lib.close()

    def test_load_verified_page(self, manifest_path):
        from cell.weight_pages import WeightPageLibrary
        lib = WeightPageLibrary(manifest_path)
        # Find a page with a hash
        verified_name = None
        for name, page in lib.pages.items():
            if page.sha256 is not None:
                verified_name = name
                break
        if verified_name is None:
            pytest.skip("No verified pages in manifest")
        data = lib.load_page(verified_name, verify=True)
        assert len(data) > 0
        lib.close()

    def test_evict_page(self, manifest_path):
        from cell.weight_pages import WeightPageLibrary
        lib = WeightPageLibrary(manifest_path)
        name = list(lib.pages.keys())[0]
        lib.load_page(name, verify=False)
        assert lib.pages[name].residency == "ram"
        lib.evict_page(name)
        assert lib.pages[name].residency == "disk"
        assert lib.ram_bytes() == 0
        lib.close()

    def test_get_layer_pages(self, manifest_path):
        from cell.weight_pages import WeightPageLibrary
        lib = WeightPageLibrary(manifest_path)
        layer_0 = lib.get_layer_pages(0)
        assert len(layer_0) > 0
        for p in layer_0:
            assert p.layer_id == 0
        lib.close()

    def test_get_pages_by_role(self, manifest_path):
        from cell.weight_pages import WeightPageLibrary
        lib = WeightPageLibrary(manifest_path)
        attn_q = lib.get_pages_by_role("attn_q")
        assert len(attn_q) > 0
        for p in attn_q:
            assert p.tensor_role == "attn_q"
        lib.close()

    def test_summary(self, manifest_path):
        from cell.weight_pages import WeightPageLibrary
        lib = WeightPageLibrary(manifest_path)
        s = lib.summary()
        assert s["total_tensors"] > 0
        assert s["on_disk"] == s["total_tensors"]
        assert s["in_ram"] == 0
        assert s["on_gpu"] == 0
        lib.close()

    def test_ops_log(self, manifest_path):
        from cell.weight_pages import WeightPageLibrary
        lib = WeightPageLibrary(manifest_path)
        name = list(lib.pages.keys())[0]
        lib.load_page(name, verify=False)
        lib.evict_page(name)
        ops = lib.get_ops_log()
        assert len(ops) >= 2
        assert ops[0]["op"] == "load_ram"
        assert ops[1]["op"] == "evict"
        lib.close()
