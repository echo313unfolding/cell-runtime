# Regulated Asset Adapter — Failure Mode Matrix

**Purpose:** Document every failure mode in the regulated asset adapter so that
failures are bounded, deterministic, receipted, and testable.

**Rule:** Policy decides hard boundaries. Model provides assessment. Gate enforces.
Receipt proves. Model cannot override hard policy.

## Matrix

| Component | Failure Mode | Expected Behavior | Fail-Closed? | Test | Status |
|-----------|-------------|-------------------|--------------|------|--------|
| **TransferEvent** | Missing event_id | KeyError raised | YES | `test_missing_event_id` | TESTED |
| **TransferEvent** | Missing wallet_from | KeyError raised | YES | `test_missing_wallet_from` | TESTED |
| **TransferEvent** | Missing wallet_to | KeyError raised | YES | `test_missing_wallet_to` | TESTED |
| **TransferEvent** | Negative amount | Accepted, no threshold triggers | NO (gap) | `test_negative_amount` | TESTED |
| **TransferEvent** | Zero amount | Clean policy result | N/A | `test_zero_amount` | TESTED |
| **TransferEvent** | Extreme amount (1e15) | Score bounded 0-100 | YES | `test_extreme_amount` | TESTED |
| **TransferEvent** | Amount as string/injection | TypeError or safe string comparison | YES | `test_amount_as_string_type` | TESTED |
| **TransferEvent** | Path traversal in jurisdiction | Inert string, no file access | YES | `test_jurisdiction_path_traversal` | TESTED |
| **TransferEvent** | SQL injection in jurisdiction | Inert string (set membership check) | YES | `test_jurisdiction_sql_injection` | TESTED |
| **TransferEvent** | Unknown asset_type | Accepted, policy doesn't filter by type | N/A | `test_unknown_asset_type` | TESTED |
| **TransferEvent** | wallet_from == wallet_to | Accepted, same hash | NO (gap) | `test_wallet_from_equals_wallet_to` | TESTED |
| **TransferEvent** | Empty wallet IDs | Valid hash of empty string | N/A | `test_empty_wallet_ids` | TESTED |
| **TransferEvent** | Unicode wallet IDs | Hashed without error | N/A | `test_unicode_wallet_id` | TESTED |
| **TransferEvent** | Extra malicious fields | Silently ignored | YES | `test_extra_fields_ignored` | TESTED |
| **Policy** | Sanctions check | Hard reject (score +80) | YES | `test_all_sanctioned_jurisdictions` | TESTED |
| **Policy** | Both jurisdictions sanctioned | Score capped at 100 | YES | `test_both_jurisdictions_sanctioned` | TESTED |
| **Policy** | KYC verified + sanctioned jurisdiction | Sanctions override KYC | YES | `test_kyc_verified_but_sanctioned_jurisdiction` | TESTED |
| **Policy** | Oracle low risk + velocity high | Both rules fire independently | YES | `test_oracle_low_risk_but_velocity_high` | TESTED |
| **Policy** | PEP + sanctions both fail | Both codes and scores added | YES | `test_pep_and_sanctions_both_fail` | TESTED |
| **Policy** | Deterministic with same inputs | Identical result every time | YES | `test_policy_deterministic_same_inputs` | TESTED |
| **Policy/Model** | Hard reject vs model allow | Policy decision wins | YES | `test_hard_reject_cannot_be_overridden` | TESTED |
| **Policy/Model** | Model cannot lower score | Policy is pure function | YES | `test_model_cannot_lower_risk_score` | TESTED |
| **Context** | Malicious RAG text | Not included in prompt (count only) | YES | `test_injection_in_context_pack_rag` | TESTED |
| **Context** | Malicious graph text | Extra fields ignored by template | YES | `test_injection_in_context_pack_graph` | TESTED |
| **Context** | "Ignore policy" injection | Policy already decided | YES | `test_injection_ignore_policy` | TESTED |
| **Context** | "Do not write receipt" injection | Gate doesn't parse detail | YES | `test_injection_do_not_write_receipt` | TESTED |
| **Context** | Oracle signal_type injection | Non-matching type ignored | YES | `test_oracle_type_injection` | TESTED |
| **Receipt** | policy_version always present | Present in every result | YES | `test_policy_version_always_present` | TESTED |
| **Receipt** | reason_codes never empty | RC-CLEAN if no triggers | YES | `test_reason_codes_never_empty` | TESTED |
| **Receipt** | risk_score bounded 0-100 | min(score, 100) enforced | YES | `test_risk_score_always_bounded` | TESTED |
| **Receipt** | Wallet hash deterministic | SHA256 match | YES | `test_wallet_hash_deterministic` | TESTED |
| **Receipt** | Prompt leaks raw wallet | Raw wallet NOT in prompt | YES | `test_prompt_does_not_leak_raw_wallet` | TESTED |
| **Receipt** | Gate receipt has agent field | Always present | YES | `test_gate_receipt_always_has_agent_field` | TESTED |
| **Sequence** | Structuring below threshold | Individual events clean | N/A | `test_structuring_individual_events_clean` | TESTED |
| **Sequence** | Structuring cumulative | RC-CUMULATIVE-HIGH fires | YES | `test_structuring_cumulative_triggers` | TESTED |
| **Sequence** | Structuring velocity | RC-HIGH-VELOCITY fires | YES | `test_structuring_velocity_triggers` | TESTED |
| **Sequence** | Structuring with no KYC | Escalates faster (+30) | YES | `test_structuring_no_kyc_accelerates` | TESTED |
| **Sequence** | Velocity burst | Hold/reject with RC-HIGH-VELOCITY | YES | `test_velocity_burst_triggers_hold` | TESTED |
| **Sequence** | Velocity boundary (exact) | Threshold is > not >= | YES | `test_velocity_just_at_threshold` | TESTED |
| **Sequence** | KYC downgrade | Risk increases monotonically | YES | `test_kyc_downgrade_increases_risk` | TESTED |
| **Sequence** | Jurisdiction hop to sanctioned | Hard reject | YES | `test_jurisdiction_hop_into_sanctioned` | TESTED |

## Gaps Found by Break Tests — Patch Status

### 1. Negative amount validation — PATCHED (v0.2)

**Was:** Negative amounts accepted, no threshold triggers.
**Fix:** `RC-INVALID-AMOUNT` (+50 risk) when `amount_usd < 0`.
**Test:** `test_negative_amount`

### 2. Self-transfer detection — PATCHED (v0.2)

**Was:** `wallet_from == wallet_to` accepted without reason code.
**Fix:** `RC-SELF-TRANSFER` (+25 risk) when `is_self_transfer()`.
**Test:** `test_wallet_from_equals_wallet_to`

### 3. Counterparty diversity tracking — PATCHED (v0.2)

**Was:** No detection of fan-in/fan-out patterns.
**Fix:** New field `unique_counterparties_24h`. `RC-COUNTERPARTY-DIVERSITY` (+20 risk) when above threshold (10).
**Test:** `test_counterparty_diversity_triggers`, `test_counterparty_diversity_below_threshold`

### 4. KYC expiry/staleness — PATCHED (v0.2)

**Was:** No `issued_at` or expiry tracking.
**Fix:** `issued_at` and `max_age_days` fields on `KYCAttestation`. `is_stale()` method. `RC-KYC-STALE` (+15 risk).
**Test:** `test_stale_kyc_triggers_reason_code`, `test_fresh_kyc_no_stale_code`, `test_kyc_no_issued_at_not_stale`

### 5. Event deduplication — REMAINING GAP (by design)

**Gap:** The adapter is stateless — same `event_id` can be evaluated
multiple times with identical results.

**Impact:** Low. Deduplication is the caller's responsibility.
The adapter is designed to be a pure function.

### 6. KYC jurisdiction mismatch — PATCHED (v0.2)

**Was:** KYC from one jurisdiction, event from another, no flag.
**Fix:** `RC-KYC-JURISDICTION-MISMATCH` (+10 risk) when `kyc.jurisdiction != event.jurisdiction`.
**Test:** `test_kyc_jurisdiction_mismatch`, `test_kyc_jurisdiction_match_no_code`

### 7. Structuring sequence detection — PATCHED (v0.2)

**Was:** Cumulative-only structuring insufficient with enhanced KYC.
**Fix:** New field `subthreshold_repeat_count_24h`. `RC-STRUCTURING-SEQUENCE` (+30 risk) when repeat count > 5 AND cumulative > $50K.
**Test:** `test_structuring_sequence_triggers`, `test_structuring_sequence_needs_both`

## Design Rules (Enforced by Tests)

| Rule | Test Evidence |
|------|---------------|
| Policy decides hard boundaries | `test_hard_reject_cannot_be_overridden` |
| Model provides assessment, not decision | `test_model_cannot_lower_risk_score` |
| Gate enforces | `test_gate_receipt_always_has_agent_field` |
| Receipt proves | `test_policy_version_always_present`, `test_reason_codes_never_empty` |
| No raw wallet in prompt | `test_prompt_does_not_leak_raw_wallet` |
| Score always bounded | `test_risk_score_always_bounded` |
| Deterministic policy | `test_policy_deterministic_same_inputs` |
| Injection-safe context | `test_injection_*` (6 tests) |
| Sanctions cannot be overridden | `test_kyc_verified_but_sanctioned_jurisdiction` |
