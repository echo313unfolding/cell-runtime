# Poetica Codegen-IR Roadmap

Plan-IR layer: validates operations, lowers to a real language.
Distinct from the language-surface layer (parser, poem types, gate, curriculum).

The language-surface layer (`compiler.py`, `parser.py`, emitters, `gate.py`)
is shipped and working. This roadmap covers the **codegen plan-IR sub-project**:
the toolchain that makes Poetica a target an LLM or agent can actually code in.

Key files: `plan_ir.py`, `plan_ir_grammar.py`, `exec_oracle.py` (this repo).

---

## The Six Words

### Built + unit-tested (Python path verified)

- **sorted-carry** — propagate `sorted()` predicate through `flow` (filter)
  and `emit` (passthrough) so binary-search plans validate instead of being
  falsely rejected. Carrying a fact about the data through the IR, not a
  copy operation. **3 unit tests: add, propagate, don't-invent.**

- **weave** — variadic verb: derive a scalar/bool from an input expression.
  Covers scalar arithmetic (`max_of_two`) and boolean predicates (`is_even`) —
  the cases the MBPP bake-off was abstaining on. **Unit-tested on Python
  lowering path. Not yet proven against actual MBPP dataset (needs
  exec_oracle + bakeoff harness).**

- **cycle** — bounded loop with accumulator (fold over `range(count)`).
  General iteration beyond list-driven `for each`. **Unit-tested on Python
  lowering path (sum, factorial).**

### Built, Python-path only — key claims unverified

- **json+gbnf** — `program_from_json()` + GBNF grammar written. The grammar
  passes structural self-checks. **NOT tested against a real LLM under
  constrained decoding.** The claim "an LLM can only emit valid IR" is
  unproven until a real model (llama.cpp + this grammar) is tested. This is
  the highest-value unproven claim.

- **multi-target** — `lower_plan(plan, target)` dispatches to all 6 emitters.
  **Emits strings for all 6 languages. Python output is verified correct
  by exec_oracle (10 programs executed and checked).** Non-Python output
  (Rust, JS, Go, Bash, SQL) compiles to temp files but is NOT yet verified
  to produce correct results. The exec_oracle harness exists for all 5
  available targets — the missing piece is target-specific bloom patching
  and cross-target test cases.

  Available toolchains on box: rustc 1.95, node v22, sqlite3 3.37, bash.
  Missing: go (not installed). 5 of 6 backends have runners.

### Deferred

- **co-edit** — bidirectional plan editing: human edits the agent's plan
  before it runs / agent proposes, human approves. Collaboration vs
  delegation. **Needs UX design before implementation.**

---

## exec_oracle — BUILT

Cross-target execution harness. `exec_oracle.py` runs emitted programs
against expected I/O with subprocess execution, timeout, and output
comparison.

**Python path: VERIFIED.** 10 programs executed — seed, emit, weave
(scalar + bool), cycle (sum + factorial), for_each, nested when+for_each
(filter_and_count), conditional, string ops. All produce correct output.
Bloom (return value) patching captures return values as stdout.

**Bug found and fixed:** exec_oracle exposed a nested-indent bug in
`_plan_op_to_surface()` — children of nested blocks (when inside for_each)
had flattened indentation, producing invalid Python. Fixed by changing
absolute indent assignment (`=`) to additive (`+=`). This bug was invisible
to string-emission tests — only caught by actual execution.

**Other targets:** Runners exist for Rust (compile+run), Node.js, Bash,
and SQLite. Bloom patching not yet implemented for non-Python targets.

**Failure detection:** wrong output, runtime errors, invalid plans, and
timeouts are all correctly caught and reported.

## Next: MBPP bakeoff (Python)

With exec_oracle built, the next step is running real MBPP tasks through
the pipeline: MBPP task description → plan JSON → exec_oracle → score.
This requires writing plan JSONs for MBPP tasks (either by hand or by
running an LLM with GBNF constrained decoding).

Build order: MBPP bakeoff (Python) → cross-target oracle (Rust/JS/SQL
bloom patching) → GBNF constrained-decode test (needs running model).

---

## Name Collision Warning

The language-surface layer (`POETICA_SYNTAX_ARCHAEOLOGY.md`) uses `carry`,
`weave`, and `flow` as copy/assign synonyms from the Ur-Poem era. The IR
layer redefines these words with different meanings:

| Word | Surface layer meaning | IR layer meaning |
|------|----------------------|------------------|
| carry | copy/assign synonym | propagate a predicate (sorted-carry) |
| weave | multi-stream synonym | variadic scalar/bool derivation |
| flow | pipe/assign | filter (preserves predicates) |

When reading docs, check which layer you're in.

---

## Design Lineage

The plan-IR borrows from JAX's tracing/staging model and other compiler
systems to create a unique intermediate representation: operations are
traced into a plan graph, validated against the gate, then lowered to
target-specific code. This is distinct from the surface compiler's
direct pattern-match-and-emit approach.

---

## Status

| Item | Status | Verified | Location |
|------|--------|----------|----------|
| Language surface (parser, emitters, gate) | SHIPPED | 337 tests | this repo |
| sorted-carry | BUILT | 3 unit tests | `plan_ir.py` |
| weave | BUILT | Python path only | `plan_ir.py` |
| cycle | BUILT | Python path only | `plan_ir.py` |
| json+gbnf | BUILT | structural check only — no real LLM test | `plan_ir.py`, `plan_ir_grammar.py` |
| multi-target lowering | BUILT | string emission only — no compile/execute | `plan_ir.py` |
| exec_oracle | BUILT | 10 Python programs executed correctly, 30 tests | `exec_oracle.py` |
| nested indent fix | FIXED | caught by exec_oracle, invisible to string tests | `plan_ir.py` |
| MBPP bakeoff | TODO | needs plan JSONs for MBPP tasks | |
| cross-target verification | TODO | runners exist, bloom patching needed | `exec_oracle.py` |
| co-edit | DEFERRED | | |
