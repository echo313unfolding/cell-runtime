"""GBNF grammar for Poetica Plan-IR constrained decoding.

An LLM emitting a Poetica plan can be constrained by this grammar so that
every token sequence is a valid plan JSON. This is the piece that makes
Poetica an LLM-targetable compiler.

Usage with llama.cpp:
    llama-server --grammar-file poetica_plan.gbnf ...

Usage from Python:
    from poetica.plan_ir_grammar import PLAN_GBNF, validate_against_grammar
    # Pass PLAN_GBNF to your constrained-decoding engine
"""


# GBNF grammar for Poetica Plan-IR JSON
# Constrains LLM output to valid plan JSON that program_from_json() can parse.
PLAN_GBNF = r'''
root ::= plan

plan ::= "{" ws
  "\"name\"" ws ":" ws string "," ws
  "\"ops\"" ws ":" ws "[" ws ops ws "]" ws
"}"

ops ::= op | op "," ws ops

op ::= seed-op | emit-op | bloom-op | flow-op | for-each-op | cycle-op | weave-op | when-op | else-op | use-op | pack-op

seed-op ::= "{" ws
  "\"op\"" ws ":" ws "\"seed\"" "," ws
  "\"name\"" ws ":" ws identifier "," ws
  "\"value\"" ws ":" ws value ws
"}"

emit-op ::= "{" ws
  "\"op\"" ws ":" ws "\"emit\"" "," ws
  "\"value\"" ws ":" ws value ws
"}"

bloom-op ::= "{" ws
  "\"op\"" ws ":" ws "\"bloom\"" "," ws
  "\"value\"" ws ":" ws value ws
"}"

flow-op ::= "{" ws
  "\"op\"" ws ":" ws "\"flow\"" "," ws
  "\"name\"" ws ":" ws identifier "," ws
  "\"output\"" ws ":" ws identifier ws
"}"

for-each-op ::= "{" ws
  "\"op\"" ws ":" ws "\"for_each\"" "," ws
  "\"name\"" ws ":" ws identifier "," ws
  "\"value\"" ws ":" ws identifier "," ws
  "\"children\"" ws ":" ws "[" ws ops ws "]" ws
"}"

cycle-op ::= "{" ws
  "\"op\"" ws ":" ws "\"cycle\"" "," ws
  "\"count\"" ws ":" ws value "," ws
  "\"accumulator\"" ws ":" ws identifier "," ws
  "\"init\"" ws ":" ws value "," ws
  "\"body_expr\"" ws ":" ws string ws
  cycle-children-opt
"}"

cycle-children-opt ::= "" | "," ws "\"children\"" ws ":" ws "[" ws ops ws "]"

weave-op ::= "{" ws
  "\"op\"" ws ":" ws "\"weave\"" "," ws
  "\"inputs\"" ws ":" ws "[" ws identifiers ws "]" "," ws
  "\"expr\"" ws ":" ws string "," ws
  "\"output\"" ws ":" ws identifier ws
  weave-type-opt
"}"

weave-type-opt ::= "" | "," ws "\"result_type\"" ws ":" ws dtype

when-op ::= "{" ws
  "\"op\"" ws ":" ws "\"when\"" "," ws
  "\"condition\"" ws ":" ws string "," ws
  "\"children\"" ws ":" ws "[" ws ops ws "]" ws
"}"

else-op ::= "{" ws
  "\"op\"" ws ":" ws "\"else\"" "," ws
  "\"children\"" ws ":" ws "[" ws ops ws "]" ws
"}"

use-op ::= "{" ws
  "\"op\"" ws ":" ws "\"use\"" "," ws
  "\"tool\"" ws ":" ws string ws
  use-params-opt
"}"

use-params-opt ::= "" | "," ws "\"params\"" ws ":" ws "{" ws kv-pairs ws "}"

pack-op ::= "{" ws
  "\"op\"" ws ":" ws "\"pack\"" "," ws
  "\"name\"" ws ":" ws identifier "," ws
  "\"format\"" ws ":" ws pack-format ws
"}"

# Terminals

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


def get_grammar() -> str:
    """Return the GBNF grammar string."""
    return PLAN_GBNF.strip()


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
