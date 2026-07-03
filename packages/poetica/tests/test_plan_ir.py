"""Tests for the Poetica Plan-IR layer."""

import json
import pytest

from poetica.plan_ir import (
    Plan, PlanOp, OpKind, DType, Predicate, ExprKind, EXPR_BUILTINS,
    program_from_json, lower_plan, add_predicate, has_predicate,
    validate_expr, render_expr_python, render_expr_rust,
)
from poetica.plan_ir_grammar import get_grammar, validate_against_grammar


# -- program_from_json --------------------------------------------------------

class TestProgramFromJson:
    """Test JSON plan ingestion."""

    def test_minimal_plan(self):
        plan_json = json.dumps({
            "name": "hello",
            "ops": [
                {"op": "seed", "name": "msg", "value": '"hello"'},
                {"op": "emit", "value": "msg"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid
        assert plan.name == "hello"
        assert len(plan.ops) == 2
        assert plan.ops[0].op == OpKind.SEED
        assert plan.ops[1].op == OpKind.EMIT

    def test_source_hash(self):
        plan_json = '{"name": "test", "ops": []}'
        plan = program_from_json(plan_json)
        assert plan.source_hash
        assert len(plan.source_hash) == 16

    def test_invalid_json(self):
        plan = program_from_json("not json at all")
        assert not plan.valid
        assert any("JSON parse" in e for e in plan.errors)

    def test_ops_not_list(self):
        plan = program_from_json('{"name": "bad", "ops": "nope"}')
        assert not plan.valid
        assert any("must be a list" in e for e in plan.errors)

    def test_unknown_op(self):
        plan_json = json.dumps({
            "name": "bad",
            "ops": [{"op": "explode"}]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("unknown op" in e for e in plan.errors)

    def test_seed_requires_name(self):
        plan_json = json.dumps({
            "name": "bad",
            "ops": [{"op": "seed", "value": "42"}]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("seed requires 'name'" in e for e in plan.errors)

    def test_weave_requires_expr_and_output(self):
        plan_json = json.dumps({
            "name": "bad",
            "ops": [
                {"op": "seed", "name": "a", "value": "1"},
                {"op": "weave", "inputs": ["a"]},
            ]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("weave requires 'expr'" in e for e in plan.errors)

    def test_cycle_requires_count_and_accumulator(self):
        plan_json = json.dumps({
            "name": "bad",
            "ops": [{"op": "cycle", "init": "0", "body_expr": "acc + 1"}]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("cycle requires 'count'" in e for e in plan.errors)
        assert any("cycle requires 'accumulator'" in e for e in plan.errors)

    def test_emit_unbound_variable(self):
        plan_json = json.dumps({
            "name": "bad",
            "ops": [{"op": "emit", "value": "nonexistent"}]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("not bound" in e for e in plan.errors)

    def test_emit_literal_no_error(self):
        plan_json = json.dumps({
            "name": "ok",
            "ops": [{"op": "emit", "value": '"hello"'}]
        })
        plan = program_from_json(plan_json)
        assert plan.valid

    def test_weave_unbound_input(self):
        plan_json = json.dumps({
            "name": "bad",
            "ops": [
                {"op": "weave", "inputs": ["x"], "expr": "x + 1", "output": "y"},
            ]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("weave input 'x' not bound" in e for e in plan.errors)


# -- Weave verb ---------------------------------------------------------------

class TestWeave:
    """Test the weave verb (variadic scalar/bool derivation)."""

    def test_weave_scalar(self):
        """Weave derives a scalar from inputs."""
        plan_json = json.dumps({
            "name": "max_of_two",
            "ops": [
                {"op": "seed", "name": "a", "value": "5"},
                {"op": "seed", "name": "b", "value": "3"},
                {"op": "weave", "inputs": ["a", "b"],
                 "expr": "a if a > b else b", "output": "result",
                 "result_type": "int"},
                {"op": "emit", "value": "result"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid
        assert plan.ops[2].op == OpKind.WEAVE
        assert plan.ops[2].result_type == DType.INT
        assert "result" in plan.bindings

    def test_weave_bool_predicate(self):
        """Weave derives a boolean (is_even)."""
        plan_json = json.dumps({
            "name": "is_even",
            "ops": [
                {"op": "seed", "name": "n", "value": "4"},
                {"op": "weave", "inputs": ["n"],
                 "expr": "n % 2 == 0", "output": "even",
                 "result_type": "bool"},
                {"op": "emit", "value": "even"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid
        assert plan.ops[1].result_type == DType.BOOL


# -- Cycle verb ---------------------------------------------------------------

class TestCycle:
    """Test the cycle verb (bounded loop with accumulator)."""

    def test_cycle_sum(self):
        """Cycle: sum 0..n."""
        plan_json = json.dumps({
            "name": "sum_n",
            "ops": [
                {"op": "seed", "name": "n", "value": "10"},
                {"op": "cycle", "count": "n", "accumulator": "total",
                 "init": "0", "body_expr": "total + _i_1"},
                {"op": "emit", "value": "total"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid
        assert plan.ops[1].op == OpKind.CYCLE
        assert "total" in plan.bindings

    def test_cycle_factorial(self):
        """Cycle: factorial via fold."""
        plan_json = json.dumps({
            "name": "factorial",
            "ops": [
                {"op": "seed", "name": "n", "value": "5"},
                {"op": "cycle", "count": "n", "accumulator": "result",
                 "init": "1", "body_expr": "result * (_i_1 + 1)"},
                {"op": "emit", "value": "result"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid


# -- Sorted-carry predicate propagation ---------------------------------------

class TestSortedCarry:
    """Test predicate propagation through flow and emit."""

    def test_add_and_check_predicate(self):
        plan_json = json.dumps({
            "name": "sorted_test",
            "ops": [
                {"op": "seed", "name": "data", "value": "[1, 2, 3]"},
            ]
        })
        plan = program_from_json(plan_json)
        add_predicate(plan, "sorted", "data")
        assert has_predicate(plan, "sorted", "data")

    def test_flow_carries_predicate(self):
        """Flow (filter) preserves sorted predicate on the output."""
        plan_json = json.dumps({
            "name": "filter_sorted",
            "ops": [
                {"op": "seed", "name": "data", "value": "[1, 2, 3, 4, 5]"},
                {"op": "flow", "name": "data", "output": "filtered"},
            ]
        })
        plan = program_from_json(plan_json)
        # Manually add predicate to source, then re-propagate
        plan.predicates.add(Predicate(name="sorted", target="data"))
        # Re-run propagation
        from poetica.plan_ir import _propagate_predicates
        _propagate_predicates(plan)
        assert has_predicate(plan, "sorted", "filtered")

    def test_predicate_not_invented(self):
        """Don't create predicates that weren't declared."""
        plan_json = json.dumps({
            "name": "no_predicate",
            "ops": [
                {"op": "seed", "name": "data", "value": "[3, 1, 2]"},
                {"op": "flow", "name": "data", "output": "filtered"},
            ]
        })
        plan = program_from_json(plan_json)
        assert not has_predicate(plan, "sorted", "data")
        assert not has_predicate(plan, "sorted", "filtered")


# -- Multi-target lowering ---------------------------------------------------

class TestMultiTargetLowering:
    """Test that one plan lowers to multiple languages."""

    def _make_plan(self):
        return json.dumps({
            "name": "greeter",
            "ops": [
                {"op": "seed", "name": "msg", "value": '"hello world"'},
                {"op": "emit", "value": "msg"},
            ]
        })

    def test_lower_to_python(self):
        plan = program_from_json(self._make_plan())
        code = lower_plan(plan, "python")
        assert "def greeter" in code
        assert "hello world" in code
        assert "print" in code

    def test_lower_to_rust(self):
        plan = program_from_json(self._make_plan())
        code = lower_plan(plan, "rust")
        assert "fn greeter" in code
        assert "println!" in code

    def test_lower_to_javascript(self):
        plan = program_from_json(self._make_plan())
        code = lower_plan(plan, "javascript")
        assert "function greeter" in code
        assert "console.log" in code

    def test_lower_to_go(self):
        plan = program_from_json(self._make_plan())
        code = lower_plan(plan, "go")
        assert "func greeter" in code
        assert "fmt.Println" in code

    def test_lower_to_bash(self):
        plan = program_from_json(self._make_plan())
        code = lower_plan(plan, "bash")
        assert "greeter()" in code
        assert "echo" in code

    def test_lower_to_sql(self):
        plan = program_from_json(self._make_plan())
        code = lower_plan(plan, "sql")
        assert "SELECT" in code

    def test_lower_invalid_plan_raises(self):
        plan = program_from_json('{"name": "bad", "ops": [{"op": "explode"}]}')
        with pytest.raises(ValueError, match="Cannot lower invalid plan"):
            lower_plan(plan, "python")

    def test_lower_by_poem_type(self):
        """Poem type names work as target selectors."""
        plan = program_from_json(self._make_plan())
        code_sonnet = lower_plan(plan, "sonnet")
        code_python = lower_plan(plan, "python")
        assert code_sonnet == code_python

        code_haiku = lower_plan(plan, "haiku")
        assert "fn greeter" in code_haiku


# -- Weave lowering -----------------------------------------------------------

class TestWeaveLowering:
    """Test that weave lowers correctly to target code."""

    def test_weave_lowers_to_assignment(self):
        plan_json = json.dumps({
            "name": "derive",
            "ops": [
                {"op": "seed", "name": "a", "value": "5"},
                {"op": "seed", "name": "b", "value": "3"},
                {"op": "weave", "inputs": ["a", "b"],
                 "expr": "a + b", "output": "total"},
                {"op": "emit", "value": "total"},
            ]
        })
        plan = program_from_json(plan_json)
        code = lower_plan(plan, "python")
        assert "total = a + b" in code
        assert "print(total)" in code

    def test_weave_lowers_to_rust(self):
        plan_json = json.dumps({
            "name": "derive",
            "ops": [
                {"op": "seed", "name": "a", "value": "5"},
                {"op": "seed", "name": "b", "value": "3"},
                {"op": "weave", "inputs": ["a", "b"],
                 "expr": "a + b", "output": "total"},
                {"op": "emit", "value": "total"},
            ]
        })
        plan = program_from_json(plan_json)
        code = lower_plan(plan, "rust")
        assert "let mut total = a + b;" in code


# -- Cycle lowering -----------------------------------------------------------

class TestCycleLowering:
    """Test that cycle lowers correctly."""

    def test_cycle_lowers_to_for_loop(self):
        plan_json = json.dumps({
            "name": "counter",
            "ops": [
                {"op": "seed", "name": "n", "value": "5"},
                {"op": "cycle", "count": "n", "accumulator": "total",
                 "init": "0", "body_expr": "total + 1"},
                {"op": "emit", "value": "total"},
            ]
        })
        plan = program_from_json(plan_json)
        code = lower_plan(plan, "python")
        assert "total = 0" in code
        assert "range(n)" in code
        assert "total = total + 1" in code


# -- Nested structures --------------------------------------------------------

class TestNestedStructures:
    """Test plans with children (when/for_each with nested ops)."""

    def test_when_with_children(self):
        plan_json = json.dumps({
            "name": "conditional",
            "ops": [
                {"op": "seed", "name": "x", "value": "5"},
                {"op": "when", "condition": "x > 0", "children": [
                    {"op": "emit", "value": '"positive"'},
                ]},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid
        assert plan.ops[1].op == OpKind.WHEN
        assert len(plan.ops[1].children) == 1

    def test_for_each_with_children(self):
        plan_json = json.dumps({
            "name": "iterate",
            "ops": [
                {"op": "seed", "name": "items", "value": "[1, 2, 3]"},
                {"op": "for_each", "name": "item", "value": "items",
                 "children": [
                     {"op": "emit", "value": "item"},
                 ]},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid
        code = lower_plan(plan, "python")
        assert "for item in items:" in code
        assert "print(item)" in code


# -- Plan serialization -------------------------------------------------------

class TestPlanSerialization:
    """Test Plan.to_dict() and to_json()."""

    def test_to_dict_roundtrip(self):
        plan_json = json.dumps({
            "name": "test",
            "ops": [
                {"op": "seed", "name": "x", "value": "42"},
                {"op": "emit", "value": "x"},
            ]
        })
        plan = program_from_json(plan_json)
        d = plan.to_dict()
        assert d["valid"] is True
        assert d["name"] == "test"
        assert len(d["ops"]) == 2

    def test_to_json(self):
        plan_json = json.dumps({
            "name": "test",
            "ops": [{"op": "seed", "name": "x", "value": "42"}]
        })
        plan = program_from_json(plan_json)
        j = plan.to_json()
        parsed = json.loads(j)
        assert parsed["valid"] is True


# -- GBNF grammar -------------------------------------------------------------

class TestGBNFGrammar:
    """Test the GBNF grammar and structural validator."""

    def test_grammar_not_empty(self):
        g = get_grammar()
        assert len(g) > 100
        assert "root" in g
        assert "seed-op" in g
        assert "weave-op" in g
        assert "cycle-op" in g

    def test_validate_valid_plan(self):
        plan_json = json.dumps({
            "name": "hello",
            "ops": [
                {"op": "seed", "name": "x", "value": "42"},
                {"op": "emit", "value": "x"},
            ]
        })
        assert validate_against_grammar(plan_json) is True

    def test_validate_invalid_json(self):
        assert validate_against_grammar("nope") is False

    def test_validate_missing_ops(self):
        assert validate_against_grammar('{"name": "bad"}') is False

    def test_validate_unknown_op(self):
        plan_json = json.dumps({
            "name": "bad",
            "ops": [{"op": "destroy"}]
        })
        assert validate_against_grammar(plan_json) is False

    def test_validate_all_op_types(self):
        """Every valid op type passes structural validation."""
        for op_type in ["seed", "emit", "bloom", "flow", "for_each",
                        "cycle", "weave", "when", "else", "use", "pack"]:
            plan_json = json.dumps({
                "name": "test",
                "ops": [{"op": op_type}]
            })
            assert validate_against_grammar(plan_json) is True, f"Failed for {op_type}"


# -- Complex plan (MBPP-style) -----------------------------------------------

class TestComplexPlans:
    """Test realistic plans that an LLM might emit for MBPP-style tasks."""

    def test_max_of_two(self):
        """MBPP: find the maximum of two numbers."""
        plan_json = json.dumps({
            "name": "max_of_two",
            "ops": [
                {"op": "seed", "name": "a", "value": "5"},
                {"op": "seed", "name": "b", "value": "3"},
                {"op": "weave", "inputs": ["a", "b"],
                 "expr": "a if a > b else b",
                 "output": "result", "result_type": "int"},
                {"op": "bloom", "value": "result"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid

        # Multi-target: same plan, two languages
        py = lower_plan(plan, "python")
        rs = lower_plan(plan, "rust")
        assert "result = a if a > b else b" in py
        assert "let mut result = if a > b { a } else { b };" in rs

    def test_sum_list(self):
        """MBPP: sum all elements of a list."""
        plan_json = json.dumps({
            "name": "sum_list",
            "ops": [
                {"op": "seed", "name": "items", "value": "[1, 2, 3, 4, 5]"},
                {"op": "cycle", "count": "len(items)",
                 "accumulator": "total", "init": "0",
                 "body_expr": "total + items[_i_1]"},
                {"op": "bloom", "value": "total"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid
        py = lower_plan(plan, "python")
        assert "total = 0" in py
        assert "total = total + items[_i_1]" in py

    def test_filter_and_count(self):
        """MBPP: filter a list and count results."""
        plan_json = json.dumps({
            "name": "count_positive",
            "ops": [
                {"op": "seed", "name": "data", "value": "[-1, 2, -3, 4, 5]"},
                {"op": "seed", "name": "count", "value": "0"},
                {"op": "for_each", "name": "x", "value": "data",
                 "children": [
                     {"op": "when", "condition": "x > 0", "children": [
                         {"op": "weave", "inputs": ["count"],
                          "expr": "count + 1", "output": "count"},
                     ]},
                 ]},
                {"op": "bloom", "value": "count"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid
        py = lower_plan(plan, "python")
        assert "for x in data:" in py


# -- Structured Expr trees ---------------------------------------------------

class TestExprValidation:
    """Test expression tree validation."""

    def test_valid_add(self):
        expr = {"kind": "add",
                "left": {"kind": "var", "name": "a"},
                "right": {"kind": "lit", "value": 1}}
        assert validate_expr(expr) == []

    def test_valid_cond(self):
        expr = {"kind": "cond",
                "test": {"kind": "gt",
                         "left": {"kind": "var", "name": "a"},
                         "right": {"kind": "var", "name": "b"}},
                "true": {"kind": "var", "name": "a"},
                "false": {"kind": "var", "name": "b"}}
        assert validate_expr(expr) == []

    def test_unknown_kind(self):
        errors = validate_expr({"kind": "explode"})
        assert any("Unknown" in e for e in errors)

    def test_lit_requires_value(self):
        errors = validate_expr({"kind": "lit"})
        assert any("value" in e for e in errors)

    def test_binary_requires_operands(self):
        errors = validate_expr({"kind": "add"})
        assert any("left" in e for e in errors)
        assert any("right" in e for e in errors)

    def test_unknown_builtin(self):
        errors = validate_expr({"kind": "call", "fn": "destroy", "args": []})
        assert any("Unknown builtin" in e for e in errors)

    def test_valid_builtins(self):
        for fn in EXPR_BUILTINS:
            expr = {"kind": "call", "fn": fn, "args": [{"kind": "lit", "value": 1}]}
            assert validate_expr(expr) == [], f"Failed for {fn}"

    def test_not_a_dict(self):
        errors = validate_expr("x + 1")
        assert any("must be a dict" in e for e in errors)

    def test_plan_validates_structured_expr(self):
        """Structured expr in JSON plan passes validation."""
        plan_json = json.dumps({
            "name": "test_struct",
            "ops": [
                {"op": "seed", "name": "a", "value": "5"},
                {"op": "weave", "inputs": ["a"],
                 "expr": {"kind": "add",
                          "left": {"kind": "var", "name": "a"},
                          "right": {"kind": "lit", "value": 1}},
                 "output": "result"},
                {"op": "emit", "value": "result"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid

    def test_plan_catches_invalid_expr(self):
        """Invalid structured expr in JSON plan produces errors."""
        plan_json = json.dumps({
            "name": "bad_expr",
            "ops": [
                {"op": "seed", "name": "a", "value": "5"},
                {"op": "weave", "inputs": ["a"],
                 "expr": {"kind": "explode"},
                 "output": "result"},
            ]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("Unknown" in e for e in plan.errors)


class TestExprRendering:
    """Test expression rendering to Python and Rust."""

    def test_lit_int(self):
        assert render_expr_python({"kind": "lit", "value": 42}) == "42"
        assert render_expr_rust({"kind": "lit", "value": 42}) == "42"

    def test_lit_bool_python(self):
        assert render_expr_python({"kind": "lit", "value": True}) == "True"
        assert render_expr_python({"kind": "lit", "value": False}) == "False"

    def test_lit_bool_rust(self):
        assert render_expr_rust({"kind": "lit", "value": True}) == "true"
        assert render_expr_rust({"kind": "lit", "value": False}) == "false"

    def test_lit_string_python(self):
        assert render_expr_python({"kind": "lit", "value": "hello"}) == "'hello'"

    def test_lit_string_rust(self):
        assert render_expr_rust({"kind": "lit", "value": "hello"}) == '"hello"'

    def test_var(self):
        assert render_expr_python({"kind": "var", "name": "x"}) == "x"
        assert render_expr_rust({"kind": "var", "name": "x"}) == "x"

    def test_add(self):
        expr = {"kind": "add",
                "left": {"kind": "var", "name": "a"},
                "right": {"kind": "var", "name": "b"}}
        assert render_expr_python(expr) == "a + b"
        assert render_expr_rust(expr) == "a + b"

    def test_div_python_is_floor(self):
        expr = {"kind": "div",
                "left": {"kind": "var", "name": "a"},
                "right": {"kind": "var", "name": "b"}}
        assert render_expr_python(expr) == "a // b"

    def test_div_rust_is_truncating(self):
        expr = {"kind": "div",
                "left": {"kind": "var", "name": "a"},
                "right": {"kind": "var", "name": "b"}}
        assert render_expr_rust(expr) == "a / b"

    def test_neg(self):
        expr = {"kind": "neg", "operand": {"kind": "var", "name": "x"}}
        assert render_expr_python(expr) == "-x"
        assert render_expr_rust(expr) == "-x"

    def test_not_python(self):
        expr = {"kind": "not", "operand": {"kind": "var", "name": "flag"}}
        assert render_expr_python(expr) == "not flag"

    def test_not_rust(self):
        expr = {"kind": "not", "operand": {"kind": "var", "name": "flag"}}
        assert render_expr_rust(expr) == "!flag"

    def test_cond_python(self):
        expr = {"kind": "cond",
                "test": {"kind": "gt",
                         "left": {"kind": "var", "name": "a"},
                         "right": {"kind": "var", "name": "b"}},
                "true": {"kind": "var", "name": "a"},
                "false": {"kind": "var", "name": "b"}}
        assert render_expr_python(expr) == "a if a > b else b"

    def test_cond_rust(self):
        expr = {"kind": "cond",
                "test": {"kind": "gt",
                         "left": {"kind": "var", "name": "a"},
                         "right": {"kind": "var", "name": "b"}},
                "true": {"kind": "var", "name": "a"},
                "false": {"kind": "var", "name": "b"}}
        assert render_expr_rust(expr) == "if a > b { a } else { b }"

    def test_index(self):
        expr = {"kind": "index", "target": "items",
                "idx": {"kind": "var", "name": "i"}}
        assert render_expr_python(expr) == "items[i]"
        assert render_expr_rust(expr) == "items[i]"

    def test_concat_python(self):
        expr = {"kind": "call", "fn": "concat",
                "args": [{"kind": "lit", "value": "hello "},
                         {"kind": "var", "name": "name"}]}
        assert render_expr_python(expr) == "'hello ' + name"

    def test_concat_rust(self):
        expr = {"kind": "call", "fn": "concat",
                "args": [{"kind": "lit", "value": "hello "},
                         {"kind": "var", "name": "name"}]}
        result = render_expr_rust(expr)
        assert "format!" in result
        assert "name" in result

    def test_nested_precedence(self):
        """result * (_i_1 + 1) — mul wraps add in parens."""
        expr = {"kind": "mul",
                "left": {"kind": "var", "name": "result"},
                "right": {"kind": "add",
                          "left": {"kind": "var", "name": "_i_1"},
                          "right": {"kind": "lit", "value": 1}}}
        py = render_expr_python(expr)
        rs = render_expr_rust(expr)
        assert py == "result * (_i_1 + 1)"
        assert rs == "result * (_i_1 + 1)"

    def test_exprkind_enum_completeness(self):
        """Every ExprKind has a rendering path (no unknown fallback)."""
        for kind in ExprKind:
            json_kind = kind.value.rstrip("_")  # not_ → not
            assert json_kind in ("lit", "var", "add", "sub", "mul", "div", "mod",
                                 "gt", "lt", "gte", "lte", "eq", "neq",
                                 "neg", "not", "cond", "index", "call")


# -- Structured when conditions -----------------------------------------------

class TestStructuredCondition:
    """Test when-op with structured Expr tree conditions."""

    def test_when_structured_condition_valid(self):
        """Plan with structured when condition parses and validates."""
        plan_json = json.dumps({
            "name": "cond_test",
            "ops": [
                {"op": "seed", "name": "x", "value": "5"},
                {"op": "when",
                 "condition": {"kind": "gt",
                               "left": {"kind": "var", "name": "x"},
                               "right": {"kind": "lit", "value": 0}},
                 "children": [
                     {"op": "emit", "value": '"positive"'},
                 ]},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid, plan.errors

    def test_when_structured_condition_renders_python(self):
        """Structured when condition renders to Python if-statement."""
        plan_json = json.dumps({
            "name": "cond_py",
            "ops": [
                {"op": "seed", "name": "x", "value": "5"},
                {"op": "when",
                 "condition": {"kind": "lt",
                               "left": {"kind": "var", "name": "x"},
                               "right": {"kind": "lit", "value": 0}},
                 "children": [
                     {"op": "emit", "value": '"negative"'},
                 ]},
            ]
        })
        plan = program_from_json(plan_json)
        py = lower_plan(plan, "python")
        assert "if x < 0:" in py

    def test_when_structured_condition_renders_rust(self):
        """Structured when condition renders to Rust if-statement."""
        plan_json = json.dumps({
            "name": "cond_rs",
            "ops": [
                {"op": "seed", "name": "x", "value": "5"},
                {"op": "when",
                 "condition": {"kind": "gt",
                               "left": {"kind": "var", "name": "x"},
                               "right": {"kind": "lit", "value": 3}},
                 "children": [
                     {"op": "emit", "value": '"big"'},
                 ]},
            ]
        })
        plan = program_from_json(plan_json)
        rs = lower_plan(plan, "rust")
        assert "if x > 3 {" in rs

    def test_when_invalid_condition_caught(self):
        """Invalid structured condition is caught during parse."""
        plan_json = json.dumps({
            "name": "bad_cond",
            "ops": [
                {"op": "seed", "name": "x", "value": "5"},
                {"op": "when",
                 "condition": {"kind": "bogus"},
                 "children": [
                     {"op": "emit", "value": '"x"'},
                 ]},
            ]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("condition" in e for e in plan.errors)

    def test_when_complex_condition(self):
        """Compound condition: x > 0 AND x < 10 isn't in ExprKind,
        but eq/neq/gt/lt compose via nested cond or multiple when blocks.
        Here: test a nested comparison renders correctly."""
        # Use eq with a mod — (x % 2) == 0
        cond = {"kind": "eq",
                "left": {"kind": "mod",
                         "left": {"kind": "var", "name": "x"},
                         "right": {"kind": "lit", "value": 2}},
                "right": {"kind": "lit", "value": 0}}
        plan_json = json.dumps({
            "name": "even_check",
            "ops": [
                {"op": "seed", "name": "x", "value": "4"},
                {"op": "when", "condition": cond,
                 "children": [
                     {"op": "emit", "value": '"even"'},
                 ]},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid
        py = lower_plan(plan, "python")
        assert "if x % 2 == 0:" in py
        rs = lower_plan(plan, "rust")
        assert "if x % 2 == 0 {" in rs

    def test_when_string_condition_still_works(self):
        """Legacy string conditions still work (backward compat)."""
        plan_json = json.dumps({
            "name": "legacy",
            "ops": [
                {"op": "seed", "name": "x", "value": "5"},
                {"op": "when", "condition": "x > 0",
                 "children": [
                     {"op": "emit", "value": '"positive"'},
                 ]},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid
        py = lower_plan(plan, "python")
        assert "if x > 0:" in py


# -- Expr binding validation (gate tightening) --------------------------------

class TestExprBindingValidation:
    """Validator must reject plans with unbound vars inside Expr trees.

    These tests reproduce the exact failure modes from the GBNF smoke test
    (2026-07-02): plans that are grammar-valid but reference unbound variables
    or misuse index on scalars. The gate should reject them at validation,
    not let them through to fail at runtime.
    """

    def test_unbound_var_in_condition(self):
        """Reject when condition referencing unbound variable 'x'.

        Reproduces count_positive GBNF smoke test failure:
        for_each.name = "data" but condition uses var "x".
        """
        plan_json = json.dumps({
            "name": "bad_ref",
            "ops": [
                {"op": "seed", "name": "data", "value": "[-1, 2, 3]"},
                {"op": "seed", "name": "count", "value": "0"},
                {"op": "for_each", "name": "data", "value": "data",
                 "children": [
                     {"op": "when",
                      "condition": {"kind": "gt",
                                    "left": {"kind": "var", "name": "x"},
                                    "right": {"kind": "lit", "value": 0}},
                      "children": [
                          {"op": "weave", "inputs": ["count"],
                           "expr": {"kind": "add",
                                    "left": {"kind": "var", "name": "count"},
                                    "right": {"kind": "lit", "value": 1}},
                           "output": "count"},
                      ]},
                 ]},
            ]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid, f"Should reject unbound 'x', got: {plan.errors}"
        assert any("'x' not bound" in e for e in plan.errors)

    def test_unbound_var_in_weave_expr(self):
        """Reject weave expr referencing unbound variable."""
        plan_json = json.dumps({
            "name": "bad_weave",
            "ops": [
                {"op": "seed", "name": "a", "value": "5"},
                {"op": "weave", "inputs": ["a"],
                 "expr": {"kind": "add",
                          "left": {"kind": "var", "name": "a"},
                          "right": {"kind": "var", "name": "b"}},
                 "output": "result"},
            ]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("'b' not bound" in e for e in plan.errors)

    def test_unbound_var_in_cycle_body(self):
        """Reject cycle body_expr referencing unbound variable."""
        plan_json = json.dumps({
            "name": "bad_cycle",
            "ops": [
                {"op": "seed", "name": "n", "value": "5"},
                {"op": "cycle", "count": "5", "accumulator": "total",
                 "init": "0",
                 "body_expr": {"kind": "add",
                               "left": {"kind": "var", "name": "total"},
                               "right": {"kind": "var", "name": "missing"}}},
            ]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("'missing' not bound" in e for e in plan.errors)

    def test_index_on_scalar_rejected(self):
        """Reject index on a scalar (int-typed) variable.

        Reproduces sum_1_to_5 GBNF smoke test failure:
        model used index on _i_1 (the scalar loop counter).
        """
        plan_json = json.dumps({
            "name": "bad_index",
            "ops": [
                {"op": "seed", "name": "n", "value": "5"},
                {"op": "cycle", "count": "5", "accumulator": "total",
                 "init": "0",
                 "body_expr": {"kind": "add",
                               "left": {"kind": "var", "name": "total"},
                               "right": {"kind": "index",
                                         "target": "_i_0",
                                         "idx": {"kind": "lit", "value": 0}}}},
            ]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("not list" in e or "not bound" in e for e in plan.errors)

    def test_valid_index_on_list_passes(self):
        """Index on a list-typed variable should still pass."""
        plan_json = json.dumps({
            "name": "good_index",
            "ops": [
                {"op": "seed", "name": "items", "value": "[10, 20, 30]"},
                {"op": "weave", "inputs": ["items"],
                 "expr": {"kind": "index", "target": "items",
                          "idx": {"kind": "lit", "value": 1}},
                 "output": "result"},
                {"op": "bloom", "value": "result"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid, f"Should accept index on list, got: {plan.errors}"

    def test_cycle_iter_var_is_bound(self):
        """Cycle body can reference its own iter var (_i_N)."""
        plan_json = json.dumps({
            "name": "sum_ok",
            "ops": [
                {"op": "seed", "name": "n", "value": "5"},
                {"op": "cycle", "count": "5", "accumulator": "total",
                 "init": "0",
                 "body_expr": {"kind": "add",
                               "left": {"kind": "var", "name": "total"},
                               "right": {"kind": "var", "name": "_i_1"}}},
                {"op": "emit", "value": "total"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid, f"Should accept _i_1, got: {plan.errors}"

    def test_cycle_explicit_iter_var(self):
        """Cycle with explicit iter_var field uses that name."""
        plan_json = json.dumps({
            "name": "sum_explicit",
            "ops": [
                {"op": "seed", "name": "n", "value": "5"},
                {"op": "cycle", "count": "5", "accumulator": "total",
                 "init": "0", "iter_var": "i",
                 "body_expr": {"kind": "add",
                               "left": {"kind": "var", "name": "total"},
                               "right": {"kind": "var", "name": "i"}}},
                {"op": "emit", "value": "total"},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid, f"Should accept iter_var='i', got: {plan.errors}"
        py = lower_plan(plan, "python")
        assert "for i in range" in py

    def test_cycle_explicit_iter_var_wrong_name_rejected(self):
        """Cycle with iter_var='i' rejects body_expr referencing '_i_1'."""
        plan_json = json.dumps({
            "name": "sum_mismatch",
            "ops": [
                {"op": "seed", "name": "n", "value": "5"},
                {"op": "cycle", "count": "5", "accumulator": "total",
                 "init": "0", "iter_var": "i",
                 "body_expr": {"kind": "add",
                               "left": {"kind": "var", "name": "total"},
                               "right": {"kind": "var", "name": "_i_1"}}},
                {"op": "emit", "value": "total"},
            ]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("_i_1" in e and "not bound" in e for e in plan.errors)

    def test_valid_condition_with_bound_vars(self):
        """Structured condition with all vars bound should pass."""
        plan_json = json.dumps({
            "name": "valid_cond",
            "ops": [
                {"op": "seed", "name": "x", "value": "5"},
                {"op": "when",
                 "condition": {"kind": "gt",
                               "left": {"kind": "var", "name": "x"},
                               "right": {"kind": "lit", "value": 0}},
                 "children": [
                     {"op": "emit", "value": '"yes"'},
                 ]},
            ]
        })
        plan = program_from_json(plan_json)
        assert plan.valid, f"Should accept bound 'x', got: {plan.errors}"

    def test_nested_unbound_in_cond_expr(self):
        """Reject unbound var nested deep in a cond expression."""
        plan_json = json.dumps({
            "name": "deep_unbound",
            "ops": [
                {"op": "seed", "name": "a", "value": "5"},
                {"op": "weave", "inputs": ["a"],
                 "expr": {"kind": "cond",
                          "test": {"kind": "gt",
                                   "left": {"kind": "var", "name": "a"},
                                   "right": {"kind": "lit", "value": 0}},
                          "true": {"kind": "var", "name": "a"},
                          "false": {"kind": "var", "name": "z"}},
                 "output": "result"},
            ]
        })
        plan = program_from_json(plan_json)
        assert not plan.valid
        assert any("'z' not bound" in e for e in plan.errors)
