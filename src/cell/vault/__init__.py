"""Cell vault — structure-bound encrypted evidence storage.

Vendors essential crypto primitives from almanac_core:
  - structure_key: canonical context + SHA-256 binding
  - crypto_v2: AES-256-GCM with scrypt KDF + HKDF
  - bridge: minimal store/load/verify for cell-runtime

The vault_secret must never be committed, exported, or stored in the vault.
"""
