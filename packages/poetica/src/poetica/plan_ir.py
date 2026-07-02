"""Poetica Plan-IR — validated operation graph for LLM-targetable codegen.

This is the codegen plan-IR layer, DISTINCT from the language-surface layer
(parser.py, compiler.py, emitters/, gate.py). The surface layer does
pattern-match-and-emit. This layer does:

1. Accept a JSON plan (from an LLM under constrained decoding, or from a human)
2. Build a validated plan graph with typed inputs/outputs
3. Track predicates through the graph (sorted-carry)
4. Lower to any backend via parameterized emitter dispatch

Design lineage: borrows from JAX's trace-then-lower model. Operations are
declared into a plan, validated against the gate, then lowered to target code.
Unlike JAX (which targets hardware via XLA), this targets programming languages
via the existing emitter set.

Name collision warning: surface-layer uses 'carry', 'weave', 'flow' as
copy/assign synonyms. This IR layer redefines them:
  carry  = propagate a predicate (sorted-carry)
  weave  = variadic scalar/bool derivation
  flow   = filter (preserves predicates)
See docs/ROADMAP.md for the full collision table.
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# -- Plan IR types -----------------------------------------------------------

class OpKind(str, Enum):
    """The verbs of the plan IR."""
    # Data
    SEED = "seed"          # bind name = literal
    EMIT = "emit"          # output a value (print / return)
    FLOW = "flow"          # filter: pass items matching predicate
    BLOOM = "bloom"        # return / final output

    # Iteration
    FOR_EACH = "for_each"  # iterate over a collection
    CYCLE = "cycle"        # bounded fold: accumulator over range(n)

    # Derivation
    WEAVE = "weave"        # derive a scalar/bool from inputs

    # Control
    WHEN = "when"          # conditional branch
    ELSE = "else"          # else branch

    # External
    USE = "use"            # call an external tool
    PACK = "pack"          # serialize data


class DType(str, Enum):
    """Value types in the plan IR."""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    LIST = "list"
    ANY = "any"


@dataclass
class Predicate:
    """A fact about a value that can be carried through operations.

    sorted-carry: if a list is sorted, filter (flow) and passthrough (emit)
    preserve that fact. This prevents the validator from falsely rejecting
    binary-search plans that depend on sorted input.
    """
    name: str         # e.g. "sorted", "positive", "non_empty"
    target: str       # variable name this predicate applies to

    def __eq__(self, other):
        return isinstance(other, Predicate) and self.name == other.name and self.target == other.target

    def __hash__(self):
        return hash((self.name, self.target))


@dataclass
class PlanOp:
    """A single operation in the plan graph."""
    op: OpKind
    id: int = 0

    # Data bindings
    name: str = ""             # variable name (seed, flow target)
    value: str = ""            # literal value or expression
    inputs: List[str] = field(default_factory=list)   # input variable names
    output: str = ""           # output variable name

    # Weave-specific
    expr: str = ""             # expression for weave (e.g. "a + b", "x > 0")
    result_type: DType = DType.ANY

    # Cycle-specific
    count: str = ""            # iteration count (literal or variable)
    accumulator: str = ""      # accumulator variable name
    init: str = ""             # initial accumulator value
    body_expr: str = ""        # fold expression (references acc and iteration var)

    # Control
    condition: str = ""        # when condition

    # External
    tool: str = ""             # use tool name
    params: Dict[str, str] = field(default_factory=dict)
    format: str = ""           # pack format

    # Block structure
    indent: int = 0
    children: List['PlanOp'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"op": self.op.value, "id": self.id}
        for attr in ("name", "value", "output", "expr", "result_type",
                      "count", "accumulator", "init", "body_expr",
                      "condition", "tool", "format"):
            v = getattr(self, attr)
            if v and v != DType.ANY:
                d[attr] = v.value if isinstance(v, Enum) else v
        if self.inputs:
            d["inputs"] = self.inputs
        if self.params:
            d["params"] = self.params
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class Plan:
    """A complete plan: a sequence of validated operations."""
    version: str = "poetica-plan-ir-v1"
    name: str = ""
    ops: List[PlanOp] = field(default_factory=list)
    predicates: Set[Predicate] = field(default_factory=set)
    bindings: Dict[str, DType] = field(default_factory=dict)
    source_hash: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "source_hash": self.source_hash,
            "valid": self.valid,
            "ops": [op.to_dict() for op in self.ops],
            "predicates": [{"name": p.name, "target": p.target} for p in self.predicates],
            "bindings": {k: v.value for k, v in self.bindings.items()},
            "errors": self.errors,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# -- JSON plan ingestion -----------------------------------------------------

def _parse_op(raw: Dict[str, Any], op_id: int) -> Tuple[PlanOp, List[str]]:
    """Parse a single op dict from JSON into a PlanOp.

    Returns (PlanOp, list_of_errors).
    """
    errors = []
    op_str = raw.get("op", "")

    try:
        kind = OpKind(op_str)
    except ValueError:
        errors.append(f"op[{op_id}]: unknown op '{op_str}'")
        kind = OpKind.SEED  # placeholder

    plan_op = PlanOp(op=kind, id=op_id)

    # Common fields
    plan_op.name = raw.get("name", "")
    plan_op.value = str(raw.get("value", ""))
    plan_op.inputs = raw.get("inputs", [])
    plan_op.output = raw.get("output", "")
    plan_op.indent = raw.get("indent", 0)

    # Weave
    plan_op.expr = raw.get("expr", "")
    rt = raw.get("result_type", "any")
    try:
        plan_op.result_type = DType(rt)
    except ValueError:
        plan_op.result_type = DType.ANY

    # Cycle
    plan_op.count = str(raw.get("count", ""))
    plan_op.accumulator = raw.get("accumulator", "")
    plan_op.init = str(raw.get("init", ""))
    plan_op.body_expr = raw.get("body_expr", "")

    # Control
    plan_op.condition = raw.get("condition", "")

    # External
    plan_op.tool = raw.get("tool", "")
    plan_op.params = raw.get("params", {})
    plan_op.format = raw.get("format", "")

    # Children (for blocks like when/for_each/cycle)
    raw_children = raw.get("children", [])
    for i, child_raw in enumerate(raw_children):
        child_op, child_errors = _parse_op(child_raw, op_id * 100 + i + 1)
        plan_op.children.append(child_op)
        errors.extend(child_errors)

    # Validate required fields per op kind
    if kind == OpKind.SEED and not plan_op.name:
        errors.append(f"op[{op_id}]: seed requires 'name'")
    if kind == OpKind.WEAVE and not plan_op.expr:
        errors.append(f"op[{op_id}]: weave requires 'expr'")
    if kind == OpKind.WEAVE and not plan_op.output:
        errors.append(f"op[{op_id}]: weave requires 'output'")
    if kind == OpKind.CYCLE and not plan_op.count:
        errors.append(f"op[{op_id}]: cycle requires 'count'")
    if kind == OpKind.CYCLE and not plan_op.accumulator:
        errors.append(f"op[{op_id}]: cycle requires 'accumulator'")

    return plan_op, errors


def program_from_json(json_str: str) -> Plan:
    """Parse a JSON plan into a validated Plan.

    This is the LLM entry point. An LLM emits JSON under GBNF-constrained
    decoding; this function ingests that JSON, validates it, and produces
    a Plan ready for multi-target lowering.

    Args:
        json_str: JSON string with {"name": ..., "ops": [...]}

    Returns:
        Plan with .valid == True if all ops parsed and validated.
    """
    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError as e:
        plan = Plan()
        plan.errors.append(f"JSON parse error: {e}")
        return plan

    plan = Plan()
    plan.name = raw.get("name", "")
    plan.source_hash = hashlib.sha256(json_str.encode()).hexdigest()[:16]

    raw_ops = raw.get("ops", [])
    if not isinstance(raw_ops, list):
        plan.errors.append("'ops' must be a list")
        return plan

    for i, raw_op in enumerate(raw_ops):
        if not isinstance(raw_op, dict):
            plan.errors.append(f"op[{i}]: must be an object")
            continue
        op, errors = _parse_op(raw_op, i)
        plan.ops.append(op)
        plan.errors.extend(errors)

    # Run validation passes
    _validate_bindings(plan)
    _propagate_predicates(plan)

    return plan


# -- Validation passes -------------------------------------------------------

def _validate_bindings(plan: Plan) -> None:
    """Check that all referenced variables are bound before use."""
    bound: Set[str] = set()

    def _check_op(op: PlanOp):
        # Seed introduces a binding
        if op.op == OpKind.SEED:
            if op.name:
                bound.add(op.name)
                plan.bindings[op.name] = _infer_type(op.value)

        # Weave introduces an output binding
        elif op.op == OpKind.WEAVE:
            for inp in op.inputs:
                if inp not in bound:
                    plan.errors.append(
                        f"op[{op.id}]: weave input '{inp}' not bound"
                    )
            if op.output:
                bound.add(op.output)
                plan.bindings[op.output] = op.result_type

        # Cycle introduces accumulator
        elif op.op == OpKind.CYCLE:
            if op.accumulator:
                bound.add(op.accumulator)
                plan.bindings[op.accumulator] = DType.ANY

        # For_each: check collection exists, bind iteration var
        elif op.op == OpKind.FOR_EACH:
            if op.value and op.value not in bound:
                plan.errors.append(
                    f"op[{op.id}]: for_each collection '{op.value}' not bound"
                )
            if op.name:
                bound.add(op.name)

        # Flow: check source exists, bind dest
        elif op.op == OpKind.FLOW:
            if op.name and op.name not in bound:
                plan.errors.append(
                    f"op[{op.id}]: flow source '{op.name}' not bound"
                )
            if op.output:
                bound.add(op.output)

        # Emit: check value exists
        elif op.op == OpKind.EMIT:
            val = op.value or op.name
            if val and val not in bound and not _is_literal(val):
                plan.errors.append(
                    f"op[{op.id}]: emit value '{val}' not bound"
                )

        # Recurse into children
        for child in op.children:
            _check_op(child)

    for op in plan.ops:
        _check_op(op)


def _propagate_predicates(plan: Plan) -> None:
    """Sorted-carry: propagate predicates through flow and emit ops.

    If a variable has predicate 'sorted', flow (filter) and emit (passthrough)
    preserve it. This prevents false rejection of plans that depend on
    sorted input for binary search.
    """
    for op in plan.ops:
        if op.op == OpKind.SEED:
            # Check if value indicates a sorted predicate
            # (e.g., from prior plan context or explicit annotation)
            pass

        elif op.op == OpKind.FLOW:
            # Flow = filter: preserves predicates on the source
            source_preds = {p for p in plan.predicates if p.target == op.name}
            for p in source_preds:
                if op.output:
                    plan.predicates.add(Predicate(name=p.name, target=op.output))

        elif op.op == OpKind.EMIT:
            # Emit = passthrough: preserves predicates
            val = op.value or op.name
            source_preds = {p for p in plan.predicates if p.target == val}
            # Predicates survive through emit (they're still true)


def add_predicate(plan: Plan, name: str, target: str) -> None:
    """Explicitly add a predicate to a plan variable.

    Used when the caller knows a fact about the data (e.g., input is sorted).
    """
    plan.predicates.add(Predicate(name=name, target=target))


def has_predicate(plan: Plan, name: str, target: str) -> bool:
    """Check if a predicate holds for a variable."""
    return Predicate(name=name, target=target) in plan.predicates


# -- Type inference -----------------------------------------------------------

def _infer_type(value: str) -> DType:
    """Infer the DType of a literal value string."""
    if not value:
        return DType.ANY
    if value.startswith('"') or value.startswith("'"):
        return DType.STRING
    if value.lower() in ("true", "false"):
        return DType.BOOL
    try:
        int(value)
        return DType.INT
    except ValueError:
        pass
    try:
        float(value)
        return DType.FLOAT
    except ValueError:
        pass
    if value.startswith("["):
        return DType.LIST
    return DType.ANY


def _is_literal(value: str) -> bool:
    """Check if a value is a literal (not a variable reference)."""
    if not value:
        return False
    if value.startswith('"') or value.startswith("'"):
        return True
    if value.lower() in ("true", "false", "none", "null"):
        return True
    try:
        float(value)
        return True
    except ValueError:
        pass
    return False


# -- Multi-target lowering ---------------------------------------------------

def lower_plan(plan: Plan, target: str = "python") -> str:
    """Lower a validated plan to target language code.

    This is the multi-target lowering: one plan emits Python AND Rust
    (etc.) by parameterizing the emitter dispatch.

    Args:
        plan: A validated Plan (from program_from_json).
        target: Target language / poem type name.

    Returns:
        Generated source code string.

    Raises:
        ValueError: If plan is invalid.
    """
    if not plan.valid:
        raise ValueError(f"Cannot lower invalid plan: {plan.errors}")

    from poetica.emitters import get_emitter

    # Convert plan ops back to surface IR format for the existing emitters
    surface_ir = _plan_to_surface_ir(plan)

    emitter = get_emitter(target)
    return emitter.emit(surface_ir)


def _plan_to_surface_ir(plan: Plan) -> Dict[str, Any]:
    """Convert a Plan to the surface-layer IR format that emitters expect."""
    ops = []
    for plan_op in plan.ops:
        surface_ops = _plan_op_to_surface(plan_op)
        ops.extend(surface_ops)

    return {
        "version": "poetica-ir-v1",
        "name": plan.name,
        "source_hash": plan.source_hash,
        "ops": ops,
    }


def _plan_op_to_surface(plan_op: PlanOp) -> List[Dict[str, Any]]:
    """Convert a single PlanOp to one or more surface IR ops."""
    result = []

    if plan_op.op == OpKind.SEED:
        result.append({
            "op": "seed", "name": plan_op.name,
            "value": plan_op.value, "indent": plan_op.indent,
        })

    elif plan_op.op == OpKind.EMIT:
        op: Dict[str, Any] = {"op": "emit", "indent": plan_op.indent}
        op["value"] = plan_op.value or plan_op.name
        if plan_op.name and plan_op.value:
            op["label"] = plan_op.name
        result.append(op)

    elif plan_op.op == OpKind.BLOOM:
        result.append({
            "op": "bloom", "value": plan_op.value or plan_op.name,
            "indent": plan_op.indent,
        })

    elif plan_op.op == OpKind.FLOW:
        result.append({
            "op": "flow", "source": plan_op.name,
            "dest": plan_op.output, "indent": plan_op.indent,
        })

    elif plan_op.op == OpKind.FOR_EACH:
        result.append({
            "op": "for", "var": plan_op.name,
            "collection": plan_op.value, "body": "",
            "indent": plan_op.indent,
        })
        for child in plan_op.children:
            child_ops = _plan_op_to_surface(child)
            for cop in child_ops:
                cop["indent"] = plan_op.indent + 1
            result.extend(child_ops)

    elif plan_op.op == OpKind.WHEN:
        result.append({
            "op": "when", "condition": plan_op.condition,
            "indent": plan_op.indent,
        })
        for child in plan_op.children:
            child_ops = _plan_op_to_surface(child)
            for cop in child_ops:
                cop["indent"] = plan_op.indent + 1
            result.extend(child_ops)

    elif plan_op.op == OpKind.ELSE:
        result.append({
            "op": "else", "indent": plan_op.indent,
        })
        for child in plan_op.children:
            child_ops = _plan_op_to_surface(child)
            for cop in child_ops:
                cop["indent"] = plan_op.indent + 1
            result.extend(child_ops)

    elif plan_op.op == OpKind.WEAVE:
        # Weave lowers to a seed with an expression value
        result.append({
            "op": "seed", "name": plan_op.output,
            "value": plan_op.expr, "indent": plan_op.indent,
        })

    elif plan_op.op == OpKind.CYCLE:
        # Cycle lowers to: seed accumulator, for range, update accumulator
        result.append({
            "op": "seed", "name": plan_op.accumulator,
            "value": plan_op.init, "indent": plan_op.indent,
        })
        # Use a for-each over range
        iter_var = f"_i_{plan_op.id}"
        result.append({
            "op": "for", "var": iter_var,
            "collection": f"range({plan_op.count})", "body": "",
            "indent": plan_op.indent,
        })
        # Body: update accumulator
        result.append({
            "op": "seed", "name": plan_op.accumulator,
            "value": plan_op.body_expr, "indent": plan_op.indent + 1,
        })
        # Include any explicit children
        for child in plan_op.children:
            child_ops = _plan_op_to_surface(child)
            for cop in child_ops:
                cop["indent"] = plan_op.indent + 1
            result.extend(child_ops)

    elif plan_op.op == OpKind.USE:
        result.append({
            "op": "use", "tool": plan_op.tool,
            "params": plan_op.params, "indent": plan_op.indent,
        })

    elif plan_op.op == OpKind.PACK:
        result.append({
            "op": "pack", "data": plan_op.name,
            "format": plan_op.format, "indent": plan_op.indent,
        })

    return result
