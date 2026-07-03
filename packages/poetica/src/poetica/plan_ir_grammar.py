"""GBNF grammar for Poetica Plan-IR constrained decoding.

An LLM emitting a Poetica plan can be constrained by this grammar so that
every token sequence is a valid plan JSON. This is the piece that makes
Poetica an LLM-targetable compiler.

Expression rules are GENERATED from the ExprKind enum and EXPR_BUILTINS set —
the enum is the single source of truth (amendment #3). Adding a new ExprKind
or builtin automatically updates the grammar.

Usage with llama.cpp:
    llama-server --grammar-file poetica_plan.gbnf ...

Usage from Python:
    from poetica.plan_ir_grammar import PLAN_GBNF, validate_against_grammar
    # Pass PLAN_GBNF to your constrained-decoding engine
"""

from poetica.plan_ir import ExprKind, EXPR_BUILTINS, _BINARY_OPS, _UNARY_OPS


def _generate_expr_grammar() -> str:
    """Generate GBNF expression rules from ExprKind enum.

    This is the enforcement of amendment #3: the grammar is derived from
    the enum, not hand-written parallel. Adding a new ExprKind automatically
    adds a grammar rule.
    """
    # Build the alternatives for the top-level expr rule
    alternatives = []
    rules = []

    for kind in ExprKind:
        json_kind = kind.value.rstrip("_")  # not_ → not

        if json_kind == "lit":
            alternatives.append("lit-expr")
            rules.append(
                'lit-expr ::= "{" ws '
                '"\\\"kind\\\"" ws ":" ws "\\\"lit\\\"" "," ws '
                '"\\\"value\\\"" ws ":" ws value ws '
                '"}"'
            )
        elif json_kind == "var":
            alternatives.append("var-expr")
            rules.append(
                'var-expr ::= "{" ws '
                '"\\\"kind\\\"" ws ":" ws "\\\"var\\\"" "," ws '
                '"\\\"name\\\"" ws ":" ws identifier ws '
                '"}"'
            )
        elif json_kind in _BINARY_OPS:
            rule_name = f"{json_kind}-expr"
            alternatives.append(rule_name)
            rules.append(
                f'{rule_name} ::= "{{" ws '
                f'"\\\"kind\\\"" ws ":" ws "\\\"{ json_kind}\\\"" "," ws '
                f'"\\\"left\\\"" ws ":" ws expr "," ws '
                f'"\\\"right\\\"" ws ":" ws expr ws '
                f'"}}"'
            )
        elif json_kind in _UNARY_OPS:
            rule_name = f"{json_kind}-expr"
            alternatives.append(rule_name)
            rules.append(
                f'{rule_name} ::= "{{" ws '
                f'"\\\"kind\\\"" ws ":" ws "\\\"{ json_kind}\\\"" "," ws '
                f'"\\\"operand\\\"" ws ":" ws expr ws '
                f'"}}"'
            )
        elif json_kind == "cond":
            alternatives.append("cond-expr")
            rules.append(
                'cond-expr ::= "{" ws '
                '"\\\"kind\\\"" ws ":" ws "\\\"cond\\\"" "," ws '
                '"\\\"test\\\"" ws ":" ws expr "," ws '
                '"\\\"true\\\"" ws ":" ws expr "," ws '
                '"\\\"false\\\"" ws ":" ws expr ws '
                '"}"'
            )
        elif json_kind == "index":
            alternatives.append("index-expr")
            rules.append(
                'index-expr ::= "{" ws '
                '"\\\"kind\\\"" ws ":" ws "\\\"index\\\"" "," ws '
                '"\\\"target\\\"" ws ":" ws identifier "," ws '
                '"\\\"idx\\\"" ws ":" ws expr ws '
                '"}"'
            )
        elif json_kind == "call":
            alternatives.append("call-expr")
            rules.append(
                'call-expr ::= "{" ws '
                '"\\\"kind\\\"" ws ":" ws "\\\"call\\\"" "," ws '
                '"\\\"fn\\\"" ws ":" ws builtin-name "," ws '
                '"\\\"args\\\"" ws ":" ws "[" ws expr-list ws "]" ws '
                '"}"'
            )

    # Generate builtin-name rule from EXPR_BUILTINS
    builtin_alts = " | ".join(f'"\\\"{ b}\\\""' for b in sorted(EXPR_BUILTINS))
    rules.append(f"builtin-name ::= {builtin_alts}")
    rules.append('expr-list ::= expr | expr "," ws expr-list |')

    # Top-level expr rule
    expr_rule = f"expr ::= {' | '.join(alternatives)}"

    return "\n".join([expr_rule] + rules)


# Base grammar (ops, terminals) — hand-written.
# Expression rules are generated and appended.
# IMPORTANT: all rules must be single-line — llama.cpp GBNF parser (b837+)
# treats newlines as rule terminators. Multi-line rules cause parse errors.
_BASE_GBNF = r'''root ::= plan
plan ::= "{" ws "\"name\"" ws ":" ws string "," ws "\"ops\"" ws ":" ws "[" ws ops ws "]" ws "}"
ops ::= op | op "," ws ops
op ::= seed-op | emit-op | bloom-op | flow-op | for-each-op | cycle-op | weave-op | when-op | else-op | use-op | pack-op
seed-op ::= "{" ws "\"op\"" ws ":" ws "\"seed\"" "," ws "\"name\"" ws ":" ws identifier "," ws "\"value\"" ws ":" ws value ws "}"
emit-op ::= "{" ws "\"op\"" ws ":" ws "\"emit\"" "," ws "\"value\"" ws ":" ws value ws "}"
bloom-op ::= "{" ws "\"op\"" ws ":" ws "\"bloom\"" "," ws "\"value\"" ws ":" ws value ws "}"
flow-op ::= "{" ws "\"op\"" ws ":" ws "\"flow\"" "," ws "\"name\"" ws ":" ws identifier "," ws "\"output\"" ws ":" ws identifier ws "}"
for-each-op ::= "{" ws "\"op\"" ws ":" ws "\"for_each\"" "," ws "\"name\"" ws ":" ws identifier "," ws "\"value\"" ws ":" ws identifier "," ws "\"children\"" ws ":" ws "[" ws ops ws "]" ws "}"
cycle-op ::= "{" ws "\"op\"" ws ":" ws "\"cycle\"" "," ws "\"count\"" ws ":" ws value "," ws "\"accumulator\"" ws ":" ws identifier "," ws "\"init\"" ws ":" ws value "," ws "\"iter_var\"" ws ":" ws identifier "," ws "\"body_expr\"" ws ":" ws expr ws cycle-children-opt "}"
cycle-children-opt ::= "" | "," ws "\"children\"" ws ":" ws "[" ws ops ws "]"
weave-op ::= "{" ws "\"op\"" ws ":" ws "\"weave\"" "," ws "\"inputs\"" ws ":" ws "[" ws identifiers ws "]" "," ws "\"expr\"" ws ":" ws expr "," ws "\"output\"" ws ":" ws identifier ws weave-type-opt "}"
weave-type-opt ::= "" | "," ws "\"result_type\"" ws ":" ws dtype
when-op ::= "{" ws "\"op\"" ws ":" ws "\"when\"" "," ws "\"condition\"" ws ":" ws expr "," ws "\"children\"" ws ":" ws "[" ws ops ws "]" ws "}"
else-op ::= "{" ws "\"op\"" ws ":" ws "\"else\"" "," ws "\"children\"" ws ":" ws "[" ws ops ws "]" ws "}"
use-op ::= "{" ws "\"op\"" ws ":" ws "\"use\"" "," ws "\"tool\"" ws ":" ws string ws use-params-opt "}"
use-params-opt ::= "" | "," ws "\"params\"" ws ":" ws "{" ws kv-pairs ws "}"
pack-op ::= "{" ws "\"op\"" ws ":" ws "\"pack\"" "," ws "\"name\"" ws ":" ws identifier "," ws "\"format\"" ws ":" ws pack-format ws "}"
identifier ::= "\"" [a-zA-Z_] [a-zA-Z0-9_]* "\""
identifiers ::= identifier | identifier "," ws identifiers
string ::= "\"" [^"\\]* "\""
value ::= string | number | "true" | "false" | "null" | "[" ws values ws "]"
values ::= value | value "," ws values |
number ::= "-"? [0-9]+ ("." [0-9]+)?
dtype ::= "\"int\"" | "\"float\"" | "\"bool\"" | "\"string\"" | "\"list\"" | "\"any\""
pack-format ::= "\"json\"" | "\"csv\"" | "\"text\""
kv-pairs ::= kv-pair | kv-pair "," ws kv-pairs
kv-pair ::= string ws ":" ws value
ws ::= [ \t\n]*
'''


# Compose final grammar: base + generated expression rules
PLAN_GBNF = _BASE_GBNF.strip() + "\n\n# Expression rules (generated from ExprKind enum)\n" + _generate_expr_grammar()


def get_grammar() -> str:
    """Return the GBNF grammar string."""
    return PLAN_GBNF


def validate_against_grammar(json_str: str) -> bool:
    """Quick structural validation that a JSON string could be a valid plan.

    This does NOT do full GBNF parsing (that's the LLM engine's job).
    It checks the structural requirements that program_from_json() needs.

    Returns True if the JSON is structurally valid for plan ingestion.
    """
    import json as _json
    try:
        data = _json.loads(json_str)
    except _json.JSONDecodeError:
        return False

    if not isinstance(data, dict):
        return False
    if "ops" not in data:
        return False
    if not isinstance(data["ops"], list):
        return False

    valid_ops = {"seed", "emit", "bloom", "flow", "for_each", "cycle",
                 "weave", "when", "else", "use", "pack"}

    for op in data["ops"]:
        if not isinstance(op, dict):
            return False
        if "op" not in op:
            return False
        if op["op"] not in valid_ops:
            return False

    return True
