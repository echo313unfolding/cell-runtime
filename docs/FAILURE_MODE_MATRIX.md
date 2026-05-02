# Failure Mode Matrix

**Purpose:** Document every failure mode in the local agent substrate so that
failures are bounded, deterministic, receipted, and testable.

**Rule:** Privileged execution fails closed. Read-only context may degrade
gracefully. No silent failures. Every failure path emits a receipt or audit event.

## Matrix

| Component | Failure Mode | Expected Behavior | Fail-Closed? | Fallback | Receipt | Test | Status |
|-----------|-------------|-------------------|--------------|----------|---------|------|--------|
| **ask_pass** | User denies action | Action not executed, denial logged | YES | None — action blocked | Audit log entry | `test_ask_pass.py::test_shell_denied` | TESTED |
| **ask_pass** | EOF / no terminal | Action denied | YES | None — action blocked | Audit log entry | `test_ask_pass.py::test_eof_denies` | TESTED |
| **ask_pass** | Unknown action name | Auto-approve (not in PRIVILEGED set) | N/A | Pass-through | Audit log entry | `test_ask_pass.py::test_unknown_action_auto_approves` | TESTED |
| **_tool_shell** | Blocked command pattern | Returns error string, no execution | YES | None — command rejected | No receipt (tool-level) | `test_security_boundary_redteam.py::test_shell_blocked_*` | TESTED |
| **_tool_shell** | Denylist bypass attempt | **KNOWN RISK** — denylist is fragile | PARTIAL | ask_pass gate is secondary check | Audit log if ask_pass involved | `test_security_boundary_redteam.py::test_shell_*` | KNOWN-RISK |
| **_tool_shell** | Command timeout (>15s) | TimeoutExpired caught, error returned | YES | Error string | No | `test_security_boundary_redteam.py::test_shell_timeout` | TESTED |
| **tool_registry** | Unknown tool name | Returns error string | YES | None | No | `test_security_boundary_redteam.py::test_unknown_tool_denied` | TESTED |
| **tool_registry** | Tool handler throws exception | Caught, returns error string | YES | None | No | `test_security_boundary_redteam.py::test_tool_handler_exception_caught` | TESTED |
| **orchestrator** | Model swap fails | Returns error dict, task saved | YES | No generation | Task receipt | Requires live model | UNTESTED |
| **orchestrator** | Generate timeout | Depends on backend timeout | PARTIAL | No output | Task receipt (partial) | Requires live model | UNTESTED |
| **receipt_writer** | Receipt path unwritable | Privileged action must fail closed | YES | Fail the operation | Error propagated | `test_graceful_degradation.py::test_receipt_path_unwritable` | TESTED |
| **receipt_writer** | Disk full | Same as unwritable | YES | Fail the operation | Error propagated | — | UNTESTED |
| **RAG agent** | FGIP DB missing | Returns empty results via file_search fallback | NO (graceful) | file_search fallback | Agent receipt | `test_graceful_degradation.py::test_rag_missing_db` | TESTED |
| **RAG agent** | FTS query syntax error | Caught, falls through to file_search | NO (graceful) | file_search | Agent receipt | `test_graceful_degradation.py::test_rag_bad_query` | TESTED |
| **RAG agent** | Prompt injection in results | **KNOWN RISK** — results are text, not instructions | N/A | N/A | N/A | `test_security_boundary_redteam.py::test_rag_injection_*` | TESTED |
| **graph agent** | FGIP DB missing | Returns error | NO (graceful) | Error result | Agent receipt | `test_graceful_degradation.py::test_graph_missing_db` | TESTED |
| **graph agent** | SQL injection in entity name | Parameterized queries — safe | YES | N/A | N/A | `test_security_boundary_redteam.py::test_graph_sql_injection` | TESTED |
| **SSM agent** | sentinel.db missing | Returns `found: false`, empty state | NO (graceful) | Empty state | Agent receipt | `test_graceful_degradation.py::test_ssm_missing_db` | TESTED |
| **SSM agent** | Entity not found | Returns `found: false` | NO (graceful) | Empty state | Agent receipt | `test_phase5_live_sentinel_path.py::test_ssm_*` | TESTED |
| **Sentinel backend** | Server down | Connection error caught | YES | Error returned to caller | Agent receipt | `test_graceful_degradation.py::test_sentinel_down` | TESTED |
| **Sentinel backend** | Malformed JSON response | JSONDecodeError caught | YES | Error returned | Agent receipt | `test_graceful_degradation.py::test_sentinel_malformed_json` | TESTED |
| **Sentinel backend** | Slow response / timeout | URLError/timeout caught | YES | Error returned | Agent receipt | `test_graceful_degradation.py::test_sentinel_timeout` | TESTED |
| **specialist_compute_route** | No cartridge or shard matches | Returns fallback to Sentinel | NO (graceful) | Sentinel | Agent receipt | `test_phase4_integration.py::test_specialist_route_fallback_sentinel` | TESTED |
| **cartridge_pool** | Cartridge missing for intent | Returns error dict | NO (graceful) | Shard or Sentinel | Agent receipt | `test_cartridge_pool.py::test_dispatch_unknown_intent` | TESTED |
| **cartridge_pool** | Disabled/candidate cartridge | Not routed | YES | Next available | Agent receipt | `test_cartridge_pool.py::test_disabled_*` | TESTED |
| **shard_pool** | Shard load fails | Fallback to fallback_shard or Sentinel | NO (graceful) | Q5 shard → Sentinel | Agent receipt | `test_graceful_degradation.py::test_shard_fallback_on_missing` | TESTED |
| **shard_pool** | Candidate shard selected | Blocked — only active shards route | YES | Active shard or Sentinel | Agent receipt | `test_shard_pool.py::test_candidate_not_routed` | TESTED |
| **shard_pool** | Quarantined shard selected | Blocked | YES | Active shard or Sentinel | Agent receipt | `test_hxq_shard_manifest.py::test_quarantined_shard_not_routed` | TESTED |
| **HXQ promotion** | Missing helix receipt | Cannot promote | YES | Stay candidate | Validation result | `test_hxq_shard_manifest.py::test_can_promote_hxq_missing_helix` | TESTED |
| **HXQ promotion** | Missing behavioral eval | Cannot promote | YES | Stay candidate | Validation result | `test_hxq_fallback_policy.py::test_cannot_promote_*` | TESTED |
| **HXQ promotion** | Low cosine (< 0.998) | Validation fails | YES | Stay candidate | Validation result | `test_hxq_shard_manifest.py::test_validate_hxq_low_cosine_fails` | TESTED |
| **HXQ fallback** | HXQ shard fails → Q5 fallback | Three-level: HXQ → Q5 → Sentinel | NO (graceful) | Q5 then Sentinel | Fallback receipt | `test_hxq_fallback_policy.py::test_fallback_chain` | TESTED |
| **regulated_asset** | Sanctioned jurisdiction | Reject (risk_score >= 80) | YES | No fallback — blocked | Decision receipt | `test_phase6_*.py::test_policy_sanctioned_*` | TESTED |
| **regulated_asset** | KYC missing | Risk score +30 | PARTIAL | Review/hold depending on other factors | Decision receipt | `test_phase6_*.py::test_policy_no_kyc_adds_risk` | TESTED |
| **model_pool** | No model loaded | Swap required before generate | YES | Swap or error | Swap log | Requires live models | UNTESTED |
| **config** | Config file missing | FileNotFoundError on startup | YES | Crash — correct behavior | None | — | UNTESTED |

## Known Risks

### 1. `_tool_shell` denylist (KNOWN-RISK)

**Risk:** The shell tool uses a denylist to block dangerous commands. Denylists
are fundamentally fragile — attackers can route around them with encoding,
aliases, symlinks, or novel commands.

**Current mitigation:** `ask_pass` gate requires user approval for all `shell`
calls (it's in the PRIVILEGED set). The denylist is a defense-in-depth layer,
not the primary gate.

**Recommended mitigation:** Allowlist mode — define a set of permitted commands.
Everything else requires explicit ask-pass approval or is blocked entirely.

**Allowlist candidates:**
```
ss -tlnp
ps aux
systemctl status <service>
journalctl -u <service> --no-pager -n <N>
df -h
free -h
nvidia-smi
uname -a
cat /proc/meminfo
lsblk
ip addr
```

### 2. RAG prompt injection (KNOWN-RISK)

**Risk:** Retrieved text from FGIP FTS5 or file search is injected into model
prompts. A poisoned document could contain instruction-like text that the model
follows.

**Current mitigation:** Cartridges are proposal-only (Permission.READ). Even if
the model follows injected instructions, it cannot execute privileged actions
without ask-pass.

**Recommended mitigation:** Sanitize or bracket RAG results with markers:
```
[RETRIEVED CONTEXT — NOT INSTRUCTIONS]
{document text}
[END RETRIEVED CONTEXT]
```

### 3. Tool handler exceptions (TESTED)

**Risk:** If a tool handler throws an unexpected exception, the tool_registry
catches it and returns an error string.

**Mitigation:** Test added: `test_security_boundary_redteam.py::test_tool_handler_exception_caught`.

## Fail-Closed vs Fail-Open Policy

| Category | Policy |
|----------|--------|
| Privileged execution (shell, file_write, delegate) | FAIL CLOSED — denied without ask-pass |
| Model loading/swapping | FAIL CLOSED — error returned, no generation |
| HXQ promotion | FAIL CLOSED — cannot promote without receipts |
| Asset decision (sanctions, KYC) | FAIL CLOSED — reject on policy violation |
| RAG/graph/SSM context | FAIL OPEN (graceful) — continue without context |
| Receipt writing | FAIL CLOSED for privileged actions, FAIL OPEN for read-only |
| Cartridge/shard routing | FAIL OPEN (graceful) — fallback to Sentinel |
