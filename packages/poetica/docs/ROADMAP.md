# Poetica Codegen-IR Roadmap

Plan-IR layer: validates operations, lowers to a real language.
Distinct from the language-surface layer (parser, poem types, gate, curriculum).

The language-surface layer (`compiler.py`, `parser.py`, emitters, `gate.py`)
is shipped and working. This roadmap covers the **codegen plan-IR sub-project**:
the toolchain that makes Poetica a target an LLM or agent can actually code in.

Key files: `plan_ir.py`, `plan_ir_grammar.py` (this repo).

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
  **Emits strings for all 6 languages. Only the Python output is verified to
  be correct.** Non-Python output (Rust, JS, Go, Bash, SQL) has NOT been
  compiled or executed. Without `exec_oracle`, "multi-target" means
  "multi-string-emission," not "multi-correct." This is the classic
  multi-target trap.

  Available toolchains on box: rustc 1.95, node v22, sqlite3 3.37.
  Missing: go (not installed). A cross-target oracle could verify 4 of 6
  backends today.

### Deferred

- **co-edit** — bidirectional plan editing: human edits the agent's plan
  before it runs / agent proposes, human approves. Collaboration vs
  delegation. **Needs UX design before implementation.**

---

## Next: exec_oracle

The single receipt that backs the most claims at once. Run each emitted
program against expected I/O. Even a Python-only version gives the real
MBPP bake-off number. Harness for all future multi-target proofs.

Build order: exec_oracle → MBPP bakeoff (Python) → cross-target oracle
(Rust/JS/SQL) → GBNF constrained-decode test (needs running model).

Reasoning: no point proving an LLM can emit valid IR until you've proven
the IR it emits actually runs correctly across targets.

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
| exec_oracle | TODO | | |
| MBPP bakeoff | TODO | | |
| co-edit | DEFERRED | | |
