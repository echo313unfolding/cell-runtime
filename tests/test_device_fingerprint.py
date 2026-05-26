"""Tests for device_fingerprint.py — hardware ID collection + KDF."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cell.device_fingerprint import (
    collect_hardware_ids,
    derive_device_key,
    derive_passphrase_key,
    generate_salt,
    hardware_fingerprint_hash,
)


class TestHardwareIds:
    def test_collect_returns_dict(self):
        ids = collect_hardware_ids()
        assert isinstance(ids, dict)
        assert "machine_id" in ids
        assert "cpu_model" in ids

    def test_collect_never_crashes(self):
        # Must not raise even if DMI paths are unreadable
        ids = collect_hardware_ids()
        assert len(ids) >= 2  # at least machine_id and cpu_model

    def test_fingerprint_hash_deterministic(self):
        ids = {"a": "1", "b": "2", "c": "3"}
        h1 = hardware_fingerprint_hash(ids)
        h2 = hardware_fingerprint_hash(ids)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_fingerprint_hash_order_independent(self):
        ids1 = {"b": "2", "a": "1"}
        ids2 = {"a": "1", "b": "2"}
        assert hardware_fingerprint_hash(ids1) == hardware_fingerprint_hash(ids2)

    def test_different_ids_different_hash(self):
        ids1 = {"a": "1", "b": "2"}
        ids2 = {"a": "1", "b": "3"}
        assert hardware_fingerprint_hash(ids1) != hardware_fingerprint_hash(ids2)


class TestSalt:
    def test_salt_length(self):
        salt = generate_salt()
        assert len(salt) == 32

    def test_salt_unique(self):
        s1 = generate_salt()
        s2 = generate_salt()
        assert s1 != s2


class TestPassphraseKey:
    def test_deterministic_with_same_inputs(self):
        salt = b'\x00' * 32
        k1 = derive_passphrase_key("test-passphrase", salt)
        k2 = derive_passphrase_key("test-passphrase", salt)
        assert k1 == k2

    def test_length_32(self):
        salt = generate_salt()
        key = derive_passphrase_key("hello", salt)
        assert len(key) == 32

    def test_different_passphrase_different_key(self):
        salt = b'\xab' * 32
        k1 = derive_passphrase_key("alpha", salt)
        k2 = derive_passphrase_key("bravo", salt)
        assert k1 != k2

    def test_different_salt_different_key(self):
        k1 = derive_passphrase_key("same", b'\x01' * 32)
        k2 = derive_passphrase_key("same", b'\x02' * 32)
        assert k1 != k2


class TestDeviceKey:
    def test_deterministic(self):
        salt = b'\x00' * 32
        hw = {"machine_id": "test-id", "cpu_model": "test-cpu"}
        k1 = derive_device_key("pass", salt, hw)
        k2 = derive_device_key("pass", salt, hw)
        assert k1 == k2

    def test_length_32(self):
        salt = generate_salt()
        hw = {"machine_id": "x"}
        key = derive_device_key("pass", salt, hw)
        assert len(key) == 32

    def test_different_passphrase_different_key(self):
        salt = b'\x00' * 32
        hw = {"machine_id": "same"}
        k1 = derive_device_key("alpha", salt, hw)
        k2 = derive_device_key("bravo", salt, hw)
        assert k1 != k2

    def test_different_hardware_different_key(self):
        salt = b'\x00' * 32
        hw1 = {"machine_id": "device-A"}
        hw2 = {"machine_id": "device-B"}
        k1 = derive_device_key("same-pass", salt, hw1)
        k2 = derive_device_key("same-pass", salt, hw2)
        assert k1 != k2

    def test_different_salt_different_key(self):
        hw = {"machine_id": "same"}
        k1 = derive_device_key("pass", b'\x01' * 32, hw)
        k2 = derive_device_key("pass", b'\x02' * 32, hw)
        assert k1 != k2
