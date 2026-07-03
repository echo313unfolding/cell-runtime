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

- **json+gbnf** — `program_from_json()` + GBNF grammar with **structured
  expression trees**. Expression rules are generated from `ExprKind` enum
  (single source of truth). The grammar fully constrains expression structure:
  no free-form Python strings. Bounded operator set (`add`/`sub`/`mul`/`div`/
  `mod`, comparisons, `neg`/`not`, `cond`, `index`) + bounded builtins
  (`len`/`abs`/`concat`). `add` is numeric-only; `concat` is a distinct
  builtin. **NOT tested against a real LLM under constrained decoding.**
  The claim "an LLM can only emit valid IR" is unproven until a real model
  (llama.cpp + this grammar) is tested.

- **multi-target** — `lower_plan(plan, target)` dispatches to all 6 emitters.
  **Python: 10 programs verified. Rust: 8 programs verified (Path B).**
  Rust backend produces correct, executable code for seed/emit, weave
  (ternary + scalar), cycle (sum + factorial), for_each, conditionals
  (when + weave reassignment), and nested when+for_each (filter_and_count).
  JS/Bash/SQL emit strings, not yet verified to execute correctly.

  **Path B (IR-native lowering):** Weave and cycle pass through as semantic
  ops with structured metadata (ternary decomposition, binding tracking).
  Each emitter renders natively instead of receiving Python-shaped surface
  strings. Base emitter defaults preserve Python behavior. Rust emitter
  uses `if cond { a } else { b }`, `0..n`, `let mut`, `println!` for bloom.

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

**Rust path: VERIFIED.** 8 programs compiled and executed — seed+emit,
weave ternary (max_of_two), weave scalar (conditional_abs), cycle sum,
cycle factorial, for_each, filter_and_count (nested when+weave reassignment).
All produce correct output via IR-native lowering (Path B).

**Other targets:** Runners exist for Node.js, Bash, and SQLite. Not yet
verified with exec_oracle (need per-emitter semantic op handlers).

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
| weave | BUILT | Python + Rust (Path B) | `plan_ir.py` |
| cycle | BUILT | Python + Rust (Path B) | `plan_ir.py` |
| ExprKind / structured Expr trees | BUILT | 10 validation + 18 rendering tests, bounded operator set | `plan_ir.py` |
| json+gbnf | **VALIDATED (n=5)** | 5/5 hand-picked tasks PASS end-to-end (LLM→GBNF→validate→execute→correct output→cross-target equality). Grammar constrains all output to valid JSON. Pass rate pending MBPP. Qwen2.5-Coder-3B Q4_K_M, CPU. | `plan_ir.py`, `plan_ir_grammar.py`, `tools/gbnf_smoke_test.py` |
| multi-target lowering | BUILT | Python 10/10, **Rust 8/8 (Path B)**, JS/Bash/SQL string only | `plan_ir.py` |
| IR-native lowering (Path B) | BUILT | semantic weave/cycle ops, ternary decomposition, binding tracking | `plan_ir.py`, emitters |
| cross-target equality check | BUILT | 8 plans verified Python == Rust output | `test_exec_oracle.py` |
| structured Expr cross-target | BUILT | 8 plans using dict Expr trees, verified Python == Rust | `test_exec_oracle.py` |
| exec_oracle | BUILT | 10 Python + 8 Rust + 8 cross-target + 12 structured Expr, **482 total tests** | `exec_oracle.py` |
| nested indent fix | FIXED | caught by exec_oracle, invisible to string tests | `plan_ir.py` |
| `when` condition → Expr | BUILT | 6 validation/render + 4 cross-target execution tests | `plan_ir.py`, emitters |
| Expr binding validator | BUILT | 8 tests: unbound var/index in expr trees, index-on-scalar rejection | `plan_ir.py` |
| cycle `iter_var` field | BUILT | 2 tests: explicit name accepted, mismatch rejected. Fixes unpredictable `_i_{op_id}` naming. | `plan_ir.py`, `plan_ir_grammar.py` |
| MBPP bakeoff | TODO | needs plan JSONs for MBPP tasks | |
| co-edit | DEFERRED | | |
