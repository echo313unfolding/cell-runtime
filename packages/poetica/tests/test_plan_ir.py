"""Tests for the Poetica Plan-IR layer."""

import json
import pytest

from poetica.plan_ir import (
    Plan, PlanOp, OpKind, DType, Predicate,
    program_from_json, lower_plan, add_predicate, has_predicate,
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
        assert "let total = a + b;" in code


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
        assert "let result = a if a > b else b;" in rs

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
