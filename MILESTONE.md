# Local Auditable Agent Substrate v0

**Date:** 2026-05-02
**Commit:** 7efae7c
**Tests:** 261/261 PASS

## What exists

| Component | Status |
|-----------|--------|
| Bounded agents (RAG, graph, SSM, Sentinel, gate, receipt) | Tested |
| ask-pass privilege gate (shell, file_write, delegate) | Tested |
| Skill cartridge pool (manifest-driven routing) | Tested |
| Shard pool (HXQ lifecycle: candidate → eval → active) | Tested |
| Specialist compute routing (cartridge → shard → Sentinel fallback) | Tested |
| Regulated asset adapter (12 reason codes, 4 decision levels) | Tested |
| Failure mode matrix (30+ modes documented) | Committed |
| Security red-team tests (42 tests) | PASS |
| Graceful degradation tests (21 tests) | PASS |

## Security boundary

- All agents: Permission.READ or Permission.WRITE. No PRIVILEGED agents.
- Privileged actions (shell, file_write, delegate_to_host) require ask-pass.
- HXQ promotion requires tensor fidelity receipt AND behavioral eval receipt.
- Candidate/quarantined/disabled assets cannot route.
- SQL injection safe (parameterized queries).
- File reads confined to home directory.

## Known risks

1. **Shell denylist is fragile.** ask-pass is the primary gate; denylist is defense-in-depth. Recommended: replace with allowlist.
2. **RAG prompt injection.** Retrieved text enters model prompts. Mitigated by proposal-only agents (Permission.READ). Recommended: bracket RAG results with context markers.

## Fail-closed / fail-open policy

| Category | Policy |
|----------|--------|
| Privileged execution | Fail closed |
| HXQ promotion | Fail closed |
| Asset decisions | Fail closed |
| Receipt writing (privileged actions) | Fail closed |
| RAG/graph/SSM context | Fail open (graceful degradation) |
| Cartridge/shard routing | Fail open (fallback chain) |

## Receipts

- `~/receipts/phase4_integration_test_20260502T161500Z.json`
- `~/receipts/phase5_live_sentinel_context_path_20260502T163957Z.json`
- `~/receipts/phase6_regulated_asset_adapter_demo_20260502T165500Z.json`
- `~/receipts/failure_mode_matrix_and_security_redteam_v0_20260502T171800Z.json`
