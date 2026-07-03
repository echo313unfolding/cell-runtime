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
import re
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


class ExprKind(str, Enum):
    """Bounded expression vocabulary for plan-IR.

    This enum is the SINGLE SOURCE OF TRUTH for the expression language.
    The GBNF grammar and the plan validator both read from this enum.
    No expression form exists outside this set.

    Design decisions (pinned):
    - add is NUMERIC ONLY. String joining is call("concat", ...).
    - div truncates toward zero (C/Rust semantics).
    - mod returns remainder with dividend sign (C/Rust semantics).
    - No operator overloading. Types determine which ops are legal.
    """
    # Leaves
    LIT = "lit"            # {"kind": "lit", "value": 42} or {"kind": "lit", "value": "hello"}
    VAR = "var"            # {"kind": "var", "name": "x"}

    # Arithmetic (numeric only — no string concat)
    ADD = "add"            # {"kind": "add", "left": expr, "right": expr}
    SUB = "sub"
    MUL = "mul"
    DIV = "div"            # integer division, truncates toward zero
    MOD = "mod"            # remainder, follows dividend sign

    # Comparison (returns bool)
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    EQ = "eq"
    NEQ = "neq"

    # Unary
    NEG = "neg"            # {"kind": "neg", "operand": expr}
    NOT = "not_"           # {"kind": "not", "operand": expr} — trailing _ avoids Python keyword

    # Conditional
    COND = "cond"          # {"kind": "cond", "test": expr, "true": expr, "false": expr}

    # Access
    INDEX = "index"        # {"kind": "index", "target": "items", "idx": expr}

    # Builtins (bounded names — extend this set, not the operators)
    CALL = "call"          # {"kind": "call", "fn": name, "args": [expr...]}


# Bounded builtin function names. The GBNF grammar constrains "fn" to this set.
EXPR_BUILTINS = frozenset({
    "len",       # length of list/string
    "abs",       # absolute value
    "concat",    # string concatenation (NOT overloaded add)
})


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
    expr: Any = ""             # expression: str (legacy) or dict (structured Expr tree)
    result_type: DType = DType.ANY

    # Cycle-specific
    count: str = ""            # iteration count (literal or variable)
    accumulator: str = ""      # accumulator variable name
    init: str = ""             # initial accumulator value
    body_expr: Any = ""        # fold expression: str (legacy) or dict (structured Expr tree)
    iter_var: str = ""         # loop variable name (model-specified, e.g. "_i_1")

    # Control
    condition: Any = ""        # when condition: str (legacy) or dict (structured Expr tree)

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

    # Weave — expr can be string (legacy) or dict (structured Expr tree)
    raw_expr = raw.get("expr", "")
    if isinstance(raw_expr, dict):
        expr_errors = validate_expr(raw_expr)
        if expr_errors:
            errors.extend(f"op[{op_id}].expr: {e}" for e in expr_errors)
        plan_op.expr = raw_expr
    else:
        plan_op.expr = raw_expr
    rt = raw.get("result_type", "any")
    try:
        plan_op.result_type = DType(rt)
    except ValueError:
        plan_op.result_type = DType.ANY

    # Cycle — body_expr can be string (legacy) or dict (structured Expr tree)
    plan_op.count = str(raw.get("count", ""))
    plan_op.accumulator = raw.get("accumulator", "")
    plan_op.init = str(raw.get("init", ""))
    plan_op.iter_var = raw.get("iter_var", "")
    raw_body = raw.get("body_expr", "")
    if isinstance(raw_body, dict):
        body_errors = validate_expr(raw_body)
        if body_errors:
            errors.extend(f"op[{op_id}].body_expr: {e}" for e in body_errors)
        plan_op.body_expr = raw_body
    else:
        plan_op.body_expr = raw_body

    # Control — condition can be string (legacy) or dict (structured Expr tree)
    raw_cond = raw.get("condition", "")
    if isinstance(raw_cond, dict):
        cond_errors = validate_expr(raw_cond)
        if cond_errors:
            errors.extend(f"op[{op_id}].condition: {e}" for e in cond_errors)
        plan_op.condition = raw_cond
    else:
        plan_op.condition = raw_cond

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

def _check_expr_vars(expr: Any, bound: Set[str], bindings: Dict[str, 'DType'],
                     prefix: str, errors: List[str]) -> None:
    """Walk a structured Expr tree and check all variable/index references.

    - var: name must be in bound
    - index: target must be in bound AND must be list-typed
    - Recurse into all sub-expressions.
    """
    if not isinstance(expr, dict):
        return  # legacy string expr — can't check statically

    kind = expr.get("kind", "")

    if kind == "var":
        name = expr.get("name", "")
        if name and name not in bound:
            errors.append(f"{prefix}: variable '{name}' not bound")

    elif kind == "index":
        target = expr.get("target", "")
        if target and target not in bound:
            errors.append(f"{prefix}: index target '{target}' not bound")
        elif target and bindings.get(target, DType.ANY) not in (DType.LIST, DType.ANY):
            errors.append(
                f"{prefix}: index target '{target}' is {bindings[target].value}, not list"
            )
        idx = expr.get("idx")
        if idx:
            _check_expr_vars(idx, bound, bindings, prefix, errors)

    elif kind in _BINARY_OPS:
        _check_expr_vars(expr.get("left", {}), bound, bindings, prefix, errors)
        _check_expr_vars(expr.get("right", {}), bound, bindings, prefix, errors)

    elif kind in _UNARY_OPS:
        _check_expr_vars(expr.get("operand", {}), bound, bindings, prefix, errors)

    elif kind == "cond":
        _check_expr_vars(expr.get("test", {}), bound, bindings, prefix, errors)
        _check_expr_vars(expr.get("true", {}), bound, bindings, prefix, errors)
        _check_expr_vars(expr.get("false", {}), bound, bindings, prefix, errors)

    elif kind == "call":
        for arg in expr.get("args", []):
            _check_expr_vars(arg, bound, bindings, prefix, errors)


def _validate_bindings(plan: Plan) -> None:
    """Check that all referenced variables are bound before use.

    Checks flat op fields (inputs, value) AND descends into structured
    Expr trees (expr, body_expr, condition) to catch unbound var/index refs.
    """
    bound: Set[str] = set()

    def _check_op(op: PlanOp):
        # Seed introduces a binding
        if op.op == OpKind.SEED:
            if op.name:
                bound.add(op.name)
                plan.bindings[op.name] = _infer_type(str(op.value))

        # Weave introduces an output binding
        elif op.op == OpKind.WEAVE:
            for inp in op.inputs:
                if inp not in bound:
                    plan.errors.append(
                        f"op[{op.id}]: weave input '{inp}' not bound"
                    )
            # Check expr tree for unbound vars
            _check_expr_vars(op.expr, bound, plan.bindings,
                             f"op[{op.id}].expr", plan.errors)
            if op.output:
                bound.add(op.output)
                plan.bindings[op.output] = op.result_type

        # Cycle introduces accumulator + loop var
        elif op.op == OpKind.CYCLE:
            if op.accumulator:
                bound.add(op.accumulator)
                plan.bindings[op.accumulator] = DType.ANY
            # Bind the iteration variable (scalar int)
            iter_var = op.iter_var or f"_i_{op.id}"
            bound.add(iter_var)
            plan.bindings[iter_var] = DType.INT
            # Check body_expr for unbound vars
            _check_expr_vars(op.body_expr, bound, plan.bindings,
                             f"op[{op.id}].body_expr", plan.errors)

        # For_each: check collection exists, bind iteration var
        elif op.op == OpKind.FOR_EACH:
            if op.value and op.value not in bound:
                plan.errors.append(
                    f"op[{op.id}]: for_each collection '{op.value}' not bound"
                )
            if op.name:
                bound.add(op.name)

        # When: check condition expr for unbound vars
        elif op.op == OpKind.WHEN:
            _check_expr_vars(op.condition, bound, plan.bindings,
                             f"op[{op.id}].condition", plan.errors)

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
            if val and val not in bound and not _is_literal(str(val)):
                plan.errors.append(
                    f"op[{op.id}]: emit value '{val}' not bound"
                )

        # Bloom: check value exists
        elif op.op == OpKind.BLOOM:
            val = op.value or op.name
            if val and val not in bound and not _is_literal(str(val)):
                plan.errors.append(
                    f"op[{op.id}]: bloom value '{val}' not bound"
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


def _parse_ternary(expr: str):
    """Parse Python ternary 'true_val if condition else false_val' into components.

    Returns dict with true_val/condition/false_val keys, or None if not a ternary.
    Used by emitters that need native conditional syntax (e.g. Rust's if/else expression).
    """
    m = re.match(r'^(.+?)\s+if\s+(.+?)\s+else\s+(.+)$', expr.strip())
    if m:
        return {
            "true_val": m.group(1).strip(),
            "condition": m.group(2).strip(),
            "false_val": m.group(3).strip(),
        }
    return None


# -- Structured expression trees ---------------------------------------------

# Operator precedence for parenthesization (lower = binds tighter)
_EXPR_PRECEDENCE = {
    "lit": 0, "var": 0, "index": 0, "call": 0,
    "neg": 1, "not": 1,
    "mul": 2, "div": 2, "mod": 2,
    "add": 3, "sub": 3,
    "gt": 4, "lt": 4, "gte": 4, "lte": 4,
    "eq": 5, "neq": 5,
    "cond": 6,
}

_BINARY_OPS = {"add", "sub", "mul", "div", "mod", "gt", "lt", "gte", "lte", "eq", "neq"}
_UNARY_OPS = {"neg", "not"}


def validate_expr(expr: Any) -> List[str]:
    """Validate a structured expression tree. Returns list of errors (empty = valid)."""
    if not isinstance(expr, dict):
        return [f"Expression must be a dict, got {type(expr).__name__}"]

    kind = expr.get("kind", "")
    errors = []

    # Check kind is in ExprKind
    valid_kinds = {e.value for e in ExprKind}
    # NOT uses "not" in JSON but "not_" in enum (Python keyword avoidance)
    json_kinds = {("not" if k == "not_" else k) for k in valid_kinds}
    if kind not in json_kinds:
        return [f"Unknown expression kind '{kind}'"]

    if kind == "lit":
        if "value" not in expr:
            errors.append("lit requires 'value'")
    elif kind == "var":
        if "name" not in expr:
            errors.append("var requires 'name'")
    elif kind in _BINARY_OPS:
        if "left" not in expr:
            errors.append(f"{kind} requires 'left'")
        else:
            errors.extend(validate_expr(expr["left"]))
        if "right" not in expr:
            errors.append(f"{kind} requires 'right'")
        else:
            errors.extend(validate_expr(expr["right"]))
    elif kind in _UNARY_OPS:
        if "operand" not in expr:
            errors.append(f"{kind} requires 'operand'")
        else:
            errors.extend(validate_expr(expr["operand"]))
    elif kind == "cond":
        for field in ("test", "true", "false"):
            if field not in expr:
                errors.append(f"cond requires '{field}'")
            else:
                errors.extend(validate_expr(expr[field]))
    elif kind == "index":
        if "target" not in expr:
            errors.append("index requires 'target'")
        if "idx" not in expr:
            errors.append("index requires 'idx'")
        else:
            errors.extend(validate_expr(expr["idx"]))
    elif kind == "call":
        fn = expr.get("fn", "")
        if fn not in EXPR_BUILTINS:
            errors.append(f"Unknown builtin '{fn}', valid: {sorted(EXPR_BUILTINS)}")
        args = expr.get("args", [])
        for i, arg in enumerate(args):
            errors.extend(validate_expr(arg))

    return errors


def render_expr_python(expr: Dict[str, Any], parent_prec: int = 99) -> str:
    """Render a structured expression tree to Python source."""
    kind = expr["kind"]
    prec = _EXPR_PRECEDENCE.get(kind, 0)

    if kind == "lit":
        v = expr["value"]
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, str):
            return repr(v)
        return str(v)
    elif kind == "var":
        return expr["name"]
    elif kind in _BINARY_OPS:
        op_map = {
            "add": "+", "sub": "-", "mul": "*",
            "div": "//", "mod": "%",
            "gt": ">", "lt": "<", "gte": ">=", "lte": "<=",
            "eq": "==", "neq": "!=",
        }
        left = render_expr_python(expr["left"], prec)
        right = render_expr_python(expr["right"], prec)
        result = f"{left} {op_map[kind]} {right}"
        if prec > parent_prec:
            result = f"({result})"
        return result
    elif kind == "neg":
        operand = render_expr_python(expr["operand"], prec)
        return f"-{operand}"
    elif kind == "not":
        operand = render_expr_python(expr["operand"], prec)
        return f"not {operand}"
    elif kind == "cond":
        test = render_expr_python(expr["test"], 99)
        true_val = render_expr_python(expr["true"], 99)
        false_val = render_expr_python(expr["false"], 99)
        result = f"{true_val} if {test} else {false_val}"
        if prec > parent_prec:
            result = f"({result})"
        return result
    elif kind == "index":
        idx = render_expr_python(expr["idx"], 99)
        return f"{expr['target']}[{idx}]"
    elif kind == "call":
        fn = expr["fn"]
        args = ", ".join(render_expr_python(a, 99) for a in expr.get("args", []))
        if fn == "concat":
            # Python concat: join with +
            parts = [render_expr_python(a, 3) for a in expr["args"]]
            return " + ".join(parts)
        return f"{fn}({args})"
    else:
        return f"/* unknown expr: {kind} */"


def render_expr_rust(expr: Dict[str, Any], parent_prec: int = 99) -> str:
    """Render a structured expression tree to Rust source."""
    kind = expr["kind"]
    prec = _EXPR_PRECEDENCE.get(kind, 0)

    if kind == "lit":
        v = expr["value"]
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return repr(v).replace("'", '"')
        return str(v)
    elif kind == "var":
        return expr["name"]
    elif kind in _BINARY_OPS:
        op_map = {
            "add": "+", "sub": "-", "mul": "*",
            "div": "/", "mod": "%",
            "gt": ">", "lt": "<", "gte": ">=", "lte": "<=",
            "eq": "==", "neq": "!=",
        }
        left = render_expr_rust(expr["left"], prec)
        right = render_expr_rust(expr["right"], prec)
        result = f"{left} {op_map[kind]} {right}"
        if prec > parent_prec:
            result = f"({result})"
        return result
    elif kind == "neg":
        operand = render_expr_rust(expr["operand"], prec)
        return f"-{operand}"
    elif kind == "not":
        operand = render_expr_rust(expr["operand"], prec)
        return f"!{operand}"
    elif kind == "cond":
        test = render_expr_rust(expr["test"], 99)
        true_val = render_expr_rust(expr["true"], 99)
        false_val = render_expr_rust(expr["false"], 99)
        return f"if {test} {{ {true_val} }} else {{ {false_val} }}"
    elif kind == "index":
        idx = render_expr_rust(expr["idx"], 99)
        return f"{expr['target']}[{idx}]"
    elif kind == "call":
        fn = expr["fn"]
        args = expr.get("args", [])
        if fn == "concat":
            # Rust concat: format!("{}{}", a, b)
            fmt = "{}".join("" for _ in range(len(args) + 1))
            rendered = ", ".join(render_expr_rust(a, 99) for a in args)
            return f'format!("{fmt}", {rendered})'
        elif fn == "len":
            return f"{render_expr_rust(args[0], 99)}.len()"
        elif fn == "abs":
            return f"{render_expr_rust(args[0], 99)}.abs()"
        return f"{fn}({', '.join(render_expr_rust(a, 99) for a in args)})"
    else:
        return f"/* unknown expr: {kind} */"


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
    declared: Set[str] = set()
    for plan_op in plan.ops:
        surface_ops = _plan_op_to_surface(plan_op, declared)
        ops.extend(surface_ops)

    return {
        "version": "poetica-ir-v1",
        "name": plan.name,
        "source_hash": plan.source_hash,
        "ops": ops,
    }


def _plan_op_to_surface(plan_op: PlanOp, declared: Set[str]) -> List[Dict[str, Any]]:
    """Convert a single PlanOp to one or more surface IR ops.

    Passes semantic weave/cycle ops through for per-emitter rendering
    instead of converting to Python-specific seed/for ops.

    Args:
        declared: Mutable set tracking which variables have been declared.
            Used to distinguish new bindings from reassignments (matters
            for targets like Rust that need 'let' vs bare assignment).
    """
    result = []

    if plan_op.op == OpKind.SEED:
        result.append({
            "op": "seed", "name": plan_op.name,
            "value": plan_op.value, "indent": plan_op.indent,
        })
        declared.add(plan_op.name)

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
        declared.add(plan_op.name)
        for child in plan_op.children:
            child_ops = _plan_op_to_surface(child, declared)
            for cop in child_ops:
                cop["indent"] += plan_op.indent + 1
            result.extend(child_ops)

    elif plan_op.op == OpKind.WHEN:
        result.append({
            "op": "when", "condition": plan_op.condition,
            "indent": plan_op.indent,
        })
        for child in plan_op.children:
            child_ops = _plan_op_to_surface(child, declared)
            for cop in child_ops:
                cop["indent"] += plan_op.indent + 1
            result.extend(child_ops)

    elif plan_op.op == OpKind.ELSE:
        result.append({
            "op": "else", "indent": plan_op.indent,
        })
        for child in plan_op.children:
            child_ops = _plan_op_to_surface(child, declared)
            for cop in child_ops:
                cop["indent"] += plan_op.indent + 1
            result.extend(child_ops)

    elif plan_op.op == OpKind.WEAVE:
        # Pass weave through as semantic op — each emitter renders natively
        # Only parse ternary for legacy string expressions; structured dicts
        # handle conditionals via the "cond" ExprKind.
        ternary = _parse_ternary(plan_op.expr) if isinstance(plan_op.expr, str) else None
        is_reassignment = plan_op.output in declared
        result.append({
            "op": "weave", "output": plan_op.output,
            "inputs": plan_op.inputs, "expr": plan_op.expr,
            "ternary": ternary,
            "is_reassignment": is_reassignment,
            "indent": plan_op.indent,
        })
        declared.add(plan_op.output)

    elif plan_op.op == OpKind.CYCLE:
        # Pass cycle through as semantic ops — each emitter renders natively
        iter_var = plan_op.iter_var or f"_i_{plan_op.id}"
        result.append({
            "op": "cycle_init", "accumulator": plan_op.accumulator,
            "init": plan_op.init, "indent": plan_op.indent,
        })
        declared.add(plan_op.accumulator)
        result.append({
            "op": "cycle_for", "iter_var": iter_var,
            "count": plan_op.count, "indent": plan_op.indent,
        })
        result.append({
            "op": "cycle_update", "accumulator": plan_op.accumulator,
            "body_expr": plan_op.body_expr, "indent": plan_op.indent + 1,
        })
        declared.add(iter_var)
        for child in plan_op.children:
            child_ops = _plan_op_to_surface(child, declared)
            for cop in child_ops:
                cop["indent"] += plan_op.indent + 1
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
