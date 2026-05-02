# Regulated Asset Adapter

**Phase 6 — Synthetic regulated-asset event evaluation on the security substrate.**
**Status:** DEMO + TESTED
**Date:** 2026-05-02

## What This Is

An adapter layer that takes synthetic regulated-asset transfer events and
evaluates them through the same SSM/RAG/graph/Sentinel/gate/receipt substrate
proven in Phase 5.

## What This Is Not

- Not a chain
- Not custody
- Not KYC (consumes mock attestations only)
- Not settlement
- Not a wallet

## Flow

```
synthetic transfer event
  → parse: wallet, asset, amount, jurisdiction, counterparty, velocity
  → deterministic risk policy (reason codes, risk score 0-100)
  → SSM update (entity history)
  → graph lookup (entity relationships)
  → RAG policy lookup (compliance documents)
  → Sentinel risk assessment (natural language)
  → gate: allow / hold / review / reject
  → receipt
```

## Decision Outputs

| Decision | Meaning |
|----------|---------|
| `allow` | Risk score < 25. No policy triggers. |
| `review` | Risk score 25-49. Manual review recommended. |
| `hold` | Risk score 50-79. Automatic hold pending investigation. |
| `reject` | Risk score >= 80. Blocked by policy. |

## Reason Codes

| Code | Trigger |
|------|---------|
| `RC-CLEAN` | No policy triggers |
| `RC-SANCTION-ORIGIN` | Origin jurisdiction is sanctioned |
| `RC-SANCTION-COUNTERPARTY` | Counterparty jurisdiction is sanctioned |
| `RC-KYC-NONE` | No KYC attestation |
| `RC-KYC-INSUFFICIENT` | Basic KYC for high-value transfer |
| `RC-SANCTIONS-FAIL` | Sanctions screening failed |
| `RC-PEP-FAIL` | PEP screening failed |
| `RC-HIGH-VALUE` | Amount > $10,000 USD |
| `RC-CUMULATIVE-HIGH` | 24h cumulative > $50,000 USD |
| `RC-HIGH-VELOCITY` | > 20 transactions in 24h |
| `RC-CROSS-BORDER` | Cross-border transfer |
| `RC-CROSS-BORDER-KYC-GAP` | Cross-border without enhanced KYC |
| `RC-ORACLE-HIGH-RISK` | Oracle risk score > 70 |
| `RC-ORACLE-MEDIUM-RISK` | Oracle risk score > 40 |

## Schemas

```
schemas/
  regulated_asset_event.schema.json     — Transfer event
  kyc_attestation.schema.json           — Mock KYC attestation
  oracle_signal.schema.json             — Mock oracle signal
  asset_decision_receipt.schema.json    — Decision receipt
```

## Architecture

The deterministic policy runs BEFORE the model. The model adds explanation
and may flag additional concerns, but does not override the policy decision.

```
Policy (deterministic, fast, auditable)
  → produces: decision, risk_level, reason_codes

Sentinel (model, slower, richer)
  → produces: natural-language risk assessment
  → does NOT override policy decision

Gate (deterministic)
  → controls execution of the decision
```

## Testing

```bash
# Smoke test (4 scenarios, with live Sentinel)
python3 tools/smoke_regulated_asset_event_path.py

# Without Sentinel backend
python3 tools/smoke_regulated_asset_event_path.py --no-sentinel

# Pytest (21 tests: policy + context + live chain)
python3 -m pytest tests/test_phase6_regulated_asset_adapter.py -v
```

## Receipt

```json
{
  "receipt_id": "...",
  "event_id": "RA-001",
  "wallet_id_hash": "abc123...",
  "asset_type": "stablecoin",
  "jurisdiction": "US",
  "amount_usd": 500.0,
  "policy_version": "regulated_asset_v0.1",
  "context_pack": {
    "ssm_summary": {},
    "rag_hits": 3,
    "graph_neighbors": 0,
    "kyc_level": "enhanced",
    "oracle_risk_score": null
  },
  "model_id": "sentinel-repair-v1-q8_0.gguf",
  "decision": "allow",
  "risk_level": "low",
  "reason_codes": ["RC-CLEAN"],
  "gate_result": {"action": "asset_allow", "allowed": true},
  "cost": {"wall_time_s": 3.2, "...": "..."}
}
```
