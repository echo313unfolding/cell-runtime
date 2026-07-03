"""Tests for the exec_oracle — cross-target execution verification.

These tests prove that plan-IR lowers to CORRECT, EXECUTABLE code.
Not just "emits a string" — actually runs and produces the right output.

Python path is verified. Other targets are tested if their toolchain
is available on the box.
"""

import json
import pytest

from poetica.exec_oracle import (
    OracleCase, OracleVerdict, OracleReport,
    run_single, run_oracle, detect_toolchains, available_targets,
    _normalize_output, _patch_python_bloom,
)


# -- Helpers ----------------------------------------------------------------

def make_case(name, ops, expected, description=""):
    return OracleCase(
        plan_json=json.dumps({"name": name, "ops": ops}),
        expected_stdout=expected,
        description=description or name,
    )


# -- Python oracle: the verified path ---------------------------------------

class TestPythonOracle:
    """Prove that plan-IR -> Python -> execute -> correct output."""

    def test_hello_world(self):
        """Seed + emit: print a string."""
        case = make_case("hello_world", [
            {"op": "seed", "name": "msg", "value": '"hello world"'},
            {"op": "emit", "value": "msg"},
        ], "hello world")
        v = run_single(case, "python")
        assert v.passed, f"FAIL: {v.error}\nActual: {v.actual!r}\nCode:\n{v.generated_code}"

    def test_numeric_emit(self):
        """Seed numeric + emit: print a number."""
        case = make_case("num", [
            {"op": "seed", "name": "x", "value": "42"},
            {"op": "emit", "value": "x"},
        ], "42")
        v = run_single(case, "python")
        assert v.passed, f"FAIL: {v.error}\nActual: {v.actual!r}\nCode:\n{v.generated_code}"

    def test_max_of_two_bloom(self):
        """Weave + bloom: derive scalar, return it."""
        case = make_case("max_of_two", [
            {"op": "seed", "name": "a", "value": "5"},
            {"op": "seed", "name": "b", "value": "3"},
            {"op": "weave", "inputs": ["a", "b"],
             "expr": "a if a > b else b", "output": "result",
             "result_type": "int"},
            {"op": "bloom", "value": "result"},
        ], "5")
        v = run_single(case, "python")
        assert v.passed, f"FAIL: {v.error}\nActual: {v.actual!r}\nCode:\n{v.generated_code}"

    def test_is_even(self):
        """Weave bool predicate: n % 2 == 0."""
        case = make_case("is_even", [
            {"op": "seed", "name": "n", "value": "4"},
            {"op": "weave", "inputs": ["n"],
             "expr": "n % 2 == 0", "output": "even",
             "result_type": "bool"},
            {"op": "emit", "value": "even"},
        ], "True")
        v = run_single(case, "python")
        assert v.passed, f"FAIL: {v.error}\nActual: {v.actual!r}\nCode:\n{v.generated_code}"

    def test_sum_range(self):
        """Cycle: sum 0+1+2+3+4 = 10."""
        case = make_case("sum_range", [
            {"op": "seed", "name": "n", "value": "5"},
            {"op": "cycle", "count": "n", "accumulator": "total",
             "init": "0", "body_expr": "total + _i_1"},
            {"op": "emit", "value": "total"},
        ], "10")
        v = run_single(case, "python")
        assert v.passed, f"FAIL: {v.error}\nActual: {v.actual!r}\nCode:\n{v.generated_code}"

    def test_factorial_bloom(self):
        """Cycle: 5! = 120 via bloom."""
        case = make_case("factorial", [
            {"op": "seed", "name": "n", "value": "5"},
            {"op": "cycle", "count": "n", "accumulator": "result",
             "init": "1", "body_expr": "result * (_i_1 + 1)"},
            {"op": "bloom", "value": "result"},
        ], "120")
        v = run_single(case, "python")
        assert v.passed, f"FAIL: {v.error}\nActual: {v.actual!r}\nCode:\n{v.generated_code}"

    def test_for_each_emit(self):
        """For_each: iterate and print each item."""
        case = make_case("print_items", [
            {"op": "seed", "name": "items", "value": "[10, 20, 30]"},
            {"op": "for_each", "name": "item", "value": "items",
             "children": [
                 {"op": "emit", "value": "item"},
             ]},
        ], "10\n20\n30")
        v = run_single(case, "python")
        assert v.passed, f"FAIL: {v.error}\nActual: {v.actual!r}\nCode:\n{v.generated_code}"

    def test_filter_and_count_bloom(self):
        """For_each + when + weave: count positives in [-1,2,-3,4,5] = 3."""
        case = make_case("count_positive", [
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
        ], "3")
        v = run_single(case, "python")
        assert v.passed, f"FAIL: {v.error}\nActual: {v.actual!r}\nCode:\n{v.generated_code}"

    def test_conditional_abs(self):
        """When: absolute value of -5 = 5."""
        case = make_case("abs_value", [
            {"op": "seed", "name": "x", "value": "-5"},
            {"op": "when", "condition": "x < 0", "children": [
                {"op": "weave", "inputs": ["x"],
                 "expr": "-x", "output": "x"},
            ]},
            {"op": "bloom", "value": "x"},
        ], "5")
        v = run_single(case, "python")
        assert v.passed, f"FAIL: {v.error}\nActual: {v.actual!r}\nCode:\n{v.generated_code}"

    def test_string_operations(self):
        """Weave on strings: concatenation."""
        case = make_case("greet", [
            {"op": "seed", "name": "name", "value": '"world"'},
            {"op": "weave", "inputs": ["name"],
             "expr": '"hello " + name', "output": "greeting"},
            {"op": "emit", "value": "greeting"},
        ], "hello world")
        v = run_single(case, "python")
        assert v.passed, f"FAIL: {v.error}\nActual: {v.actual!r}\nCode:\n{v.generated_code}"


# -- Oracle failure detection -----------------------------------------------

class TestOracleFailures:
    """Prove the oracle detects failures correctly."""

    def test_wrong_output(self):
        case = make_case("wrong", [
            {"op": "seed", "name": "msg", "value": '"hello"'},
            {"op": "emit", "value": "msg"},
        ], "goodbye")
        v = run_single(case, "python")
        assert not v.passed
        assert v.error == "Output mismatch"
        assert v.actual == "hello"
        assert v.expected == "goodbye"

    def test_runtime_error(self):
        case = make_case("crash", [
            {"op": "seed", "name": "x", "value": "1/0"},
            {"op": "emit", "value": "x"},
        ], "")
        v = run_single(case, "python")
        assert not v.passed
        assert v.exit_code != 0

    def test_invalid_plan(self):
        case = OracleCase(
            plan_json='{"name": "bad", "ops": [{"op": "explode"}]}',
            expected_stdout="",
            description="invalid op",
        )
        v = run_single(case, "python")
        assert not v.passed
        assert "Invalid plan" in v.error

    def test_timeout(self):
        case = make_case("slow", [
            {"op": "seed", "name": "n", "value": "999999999"},
            {"op": "cycle", "count": "n", "accumulator": "total",
             "init": "0", "body_expr": "total + 1"},
            {"op": "emit", "value": "total"},
        ], "")
        v = run_single(case, "python", timeout=1.0)
        assert not v.passed
        assert "Timeout" in v.error


# -- Output normalization ---------------------------------------------------

class TestNormalization:

    def test_strips_trailing_newlines(self):
        assert _normalize_output("hello\n") == "hello"
        assert _normalize_output("hello\n\n\n") == "hello"

    def test_strips_trailing_whitespace(self):
        assert _normalize_output("hello   \n") == "hello"

    def test_preserves_multiline(self):
        assert _normalize_output("a\nb\nc\n") == "a\nb\nc"

    def test_empty_string(self):
        assert _normalize_output("") == ""
        assert _normalize_output("\n") == ""


# -- Bloom patching ---------------------------------------------------------

class TestBloomPatch:

    def test_patches_return(self):
        code = (
            '# Generated by Poetica (target: python)\n'
            '\n'
            'def foo():\n'
            '    x = 5\n'
            '    return x\n'
            '\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    foo()\n'
        )
        patched = _patch_python_bloom(code, "foo")
        assert "_r = foo()" in patched
        assert "print(_r)" in patched
        assert "    foo()" not in patched

    def test_no_patch_without_return(self):
        code = (
            '# Generated by Poetica (target: python)\n'
            '\n'
            'def bar():\n'
            '    print("hello")\n'
            '\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    bar()\n'
        )
        patched = _patch_python_bloom(code, "bar")
        assert patched == code

    def test_patches_with_sanitized_name(self):
        code = (
            '# Generated by Poetica (target: python)\n'
            '\n'
            'def my_func():\n'
            '    return 42\n'
            '\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    my_func()\n'
        )
        patched = _patch_python_bloom(code, "my-func")
        assert "_r = my_func()" in patched


# -- Toolchain detection ---------------------------------------------------

class TestToolchain:

    def test_python_always_available(self):
        tools = detect_toolchains()
        assert tools["python"] is not None

    def test_available_targets_includes_python(self):
        targets = available_targets()
        assert "python" in targets

    def test_bash_always_available(self):
        tools = detect_toolchains()
        assert tools["bash"] is not None


# -- Report -----------------------------------------------------------------

class TestOracleReport:

    def test_report_tracks_pass_rate(self):
        cases = [
            make_case("ok", [
                {"op": "seed", "name": "x", "value": "42"},
                {"op": "emit", "value": "x"},
            ], "42"),
            make_case("wrong", [
                {"op": "seed", "name": "x", "value": "42"},
                {"op": "emit", "value": "x"},
            ], "99"),
        ]
        report = run_oracle(cases, ["python"])
        assert report.total_cases == 2
        assert report.total_passed == 1
        assert report.pass_rate == 0.5

    def test_report_summary_structure(self):
        cases = [
            make_case("ok", [
                {"op": "seed", "name": "x", "value": "42"},
                {"op": "emit", "value": "x"},
            ], "42"),
        ]
        report = run_oracle(cases, ["python"])
        s = report.summary()
        assert s["total"] == 1
        assert s["passed"] == 1
        assert "python" in s["targets"]
        assert "toolchain" in s

    def test_empty_report(self):
        report = run_oracle([], ["python"])
        assert report.total_cases == 0
        assert report.pass_rate == 0.0


# -- Multi-target (conditional on available toolchains) ----------------------

class TestMultiTarget:
    """Run same plan on multiple targets. Skip if toolchain unavailable."""

    @pytest.fixture
    def simple_emit_plan(self):
        """A plan that emits a string — should work on any target."""
        return json.dumps({
            "name": "greet",
            "ops": [
                {"op": "seed", "name": "msg", "value": '"hello"'},
                {"op": "emit", "value": "msg"},
            ]
        })

    @pytest.mark.skipif(
        "rust" not in available_targets(),
        reason="rustc not available"
    )
    def test_rust_emit(self, simple_emit_plan):
        """Rust target: same plan, same output."""
        case = OracleCase(simple_emit_plan, "hello", "rust emit")
        v = run_single(case, "rust")
        # Note: Rust emitter output may differ from Python (quotes, types).
        # This test documents current behavior, not asserting PASS.
        assert v.generated_code  # Code was generated
        # v.passed may be False if Rust output differs — that's the point
        # of exec_oracle: exposing multi-target correctness gaps.

    @pytest.mark.skipif(
        "javascript" not in available_targets(),
        reason="node not available"
    )
    def test_javascript_emit(self, simple_emit_plan):
        case = OracleCase(simple_emit_plan, "hello", "js emit")
        v = run_single(case, "javascript")
        assert v.generated_code

    @pytest.mark.skipif(
        "bash" not in available_targets(),
        reason="bash not available"
    )
    def test_bash_emit(self, simple_emit_plan):
        case = OracleCase(simple_emit_plan, "hello", "bash emit")
        v = run_single(case, "bash")
        assert v.generated_code


# -- Rust oracle: the first non-Python verified path -------------------------

@pytest.mark.skipif(
    "rust" not in available_targets(),
    reason="rustc not available"
)
class TestRustOracle:
    """Prove that plan-IR -> Rust -> compile -> execute -> correct output.

    Path B: semantic ops lowered natively per target, not via Python surface.
    """

    def test_hello_world(self):
        case = make_case("hello_world", [
            {"op": "seed", "name": "msg", "value": '"hello world"'},
            {"op": "emit", "value": "msg"},
        ], "hello world")
        v = run_single(case, "rust")
        assert v.passed, f"FAIL: {v.error}\nActual: {repr(v.actual)}\nCode:\n{v.generated_code}"

    def test_numeric_emit(self):
        case = make_case("num", [
            {"op": "seed", "name": "x", "value": "42"},
            {"op": "emit", "value": "x"},
        ], "42")
        v = run_single(case, "rust")
        assert v.passed, f"FAIL: {v.error}\nActual: {repr(v.actual)}\nCode:\n{v.generated_code}"

    def test_max_of_two_bloom(self):
        """Weave ternary + bloom: Rust native if/else expression."""
        case = make_case("max_of_two", [
            {"op": "seed", "name": "a", "value": "5"},
            {"op": "seed", "name": "b", "value": "3"},
            {"op": "weave", "inputs": ["a", "b"],
             "expr": "a if a > b else b", "output": "result",
             "result_type": "int"},
            {"op": "bloom", "value": "result"},
        ], "5")
        v = run_single(case, "rust")
        assert v.passed, f"FAIL: {v.error}\nActual: {repr(v.actual)}\nCode:\n{v.generated_code}"

    def test_sum_range(self):
        """Cycle: sum 0+1+2+3+4 = 10 via Rust 0..n range."""
        case = make_case("sum_range", [
            {"op": "seed", "name": "n", "value": "5"},
            {"op": "cycle", "count": "n", "accumulator": "total",
             "init": "0", "body_expr": "total + _i_1"},
            {"op": "emit", "value": "total"},
        ], "10")
        v = run_single(case, "rust")
        assert v.passed, f"FAIL: {v.error}\nActual: {repr(v.actual)}\nCode:\n{v.generated_code}"

    def test_factorial_bloom(self):
        """Cycle: 5! = 120 via bloom."""
        case = make_case("factorial", [
            {"op": "seed", "name": "n", "value": "5"},
            {"op": "cycle", "count": "n", "accumulator": "result",
             "init": "1", "body_expr": "result * (_i_1 + 1)"},
            {"op": "bloom", "value": "result"},
        ], "120")
        v = run_single(case, "rust")
        assert v.passed, f"FAIL: {v.error}\nActual: {repr(v.actual)}\nCode:\n{v.generated_code}"

    def test_for_each_emit(self):
        """For_each: iterate and print each item."""
        case = make_case("print_items", [
            {"op": "seed", "name": "items", "value": "[10, 20, 30]"},
            {"op": "for_each", "name": "item", "value": "items",
             "children": [
                 {"op": "emit", "value": "item"},
             ]},
        ], "10\n20\n30")
        v = run_single(case, "rust")
        assert v.passed, f"FAIL: {v.error}\nActual: {repr(v.actual)}\nCode:\n{v.generated_code}"

    def test_conditional_abs(self):
        """When + weave reassignment + bloom: absolute value."""
        case = make_case("abs_value", [
            {"op": "seed", "name": "x", "value": "-5"},
            {"op": "when", "condition": "x < 0", "children": [
                {"op": "weave", "inputs": ["x"],
                 "expr": "-x", "output": "x"},
            ]},
            {"op": "bloom", "value": "x"},
        ], "5")
        v = run_single(case, "rust")
        assert v.passed, f"FAIL: {v.error}\nActual: {repr(v.actual)}\nCode:\n{v.generated_code}"

    def test_filter_and_count_bloom(self):
        """For_each + when + weave reassignment: count positives = 3."""
        case = make_case("count_positive", [
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
        ], "3")
        v = run_single(case, "rust")
        assert v.passed, f"FAIL: {v.error}\nActual: {repr(v.actual)}\nCode:\n{v.generated_code}"


# -- Cross-target equality: same plan → same output -------------------------

@pytest.mark.skipif(
    "rust" not in available_targets(),
    reason="rustc not available"
)
class TestCrossTargetEquality:
    """Assert same plan produces identical output on Python AND Rust.

    This is the correctness instrument for the expression-layer refactor.
    Built BEFORE changing expressions to structured Expr trees so we have
    a green baseline to refactor under. If the same plan produces different
    output on different targets, the expression semantics are under-specified.

    Cases NOT included here (known expression-layer gaps):
    - is_even: Python prints True, Rust prints true (bool formatting)
    - string_operations: Python "hello " + name, Rust can't add &str
    These gaps are what the Expr refactor will fix.
    """

    def _assert_equal(self, name, ops, expected):
        """Run same case on Python and Rust, assert both pass with same output."""
        case = make_case(name, ops, expected)
        py = run_single(case, "python")
        rs = run_single(case, "rust")
        assert py.passed, f"Python FAIL: {py.error}\nCode:\n{py.generated_code}"
        assert rs.passed, f"Rust FAIL: {rs.error}\nCode:\n{rs.generated_code}"
        assert py.actual == rs.actual, (
            f"Cross-target output divergence on '{name}':\n"
            f"  Python: {repr(py.actual)}\n"
            f"  Rust:   {repr(rs.actual)}\n"
            f"Same plan must produce same output on every target."
        )

    def test_hello_world(self):
        self._assert_equal("hello_world", [
            {"op": "seed", "name": "msg", "value": '"hello world"'},
            {"op": "emit", "value": "msg"},
        ], "hello world")

    def test_numeric_emit(self):
        self._assert_equal("num", [
            {"op": "seed", "name": "x", "value": "42"},
            {"op": "emit", "value": "x"},
        ], "42")

    def test_max_of_two_bloom(self):
        self._assert_equal("max_of_two", [
            {"op": "seed", "name": "a", "value": "5"},
            {"op": "seed", "name": "b", "value": "3"},
            {"op": "weave", "inputs": ["a", "b"],
             "expr": "a if a > b else b", "output": "result",
             "result_type": "int"},
            {"op": "bloom", "value": "result"},
        ], "5")

    def test_sum_range(self):
        self._assert_equal("sum_range", [
            {"op": "seed", "name": "n", "value": "5"},
            {"op": "cycle", "count": "n", "accumulator": "total",
             "init": "0", "body_expr": "total + _i_1"},
            {"op": "emit", "value": "total"},
        ], "10")

    def test_factorial_bloom(self):
        self._assert_equal("factorial", [
            {"op": "seed", "name": "n", "value": "5"},
            {"op": "cycle", "count": "n", "accumulator": "result",
             "init": "1", "body_expr": "result * (_i_1 + 1)"},
            {"op": "bloom", "value": "result"},
        ], "120")

    def test_for_each_emit(self):
        self._assert_equal("print_items", [
            {"op": "seed", "name": "items", "value": "[10, 20, 30]"},
            {"op": "for_each", "name": "item", "value": "items",
             "children": [
                 {"op": "emit", "value": "item"},
             ]},
        ], "10\n20\n30")

    def test_conditional_abs(self):
        self._assert_equal("abs_value", [
            {"op": "seed", "name": "x", "value": "-5"},
            {"op": "when", "condition": "x < 0", "children": [
                {"op": "weave", "inputs": ["x"],
                 "expr": "-x", "output": "x"},
            ]},
            {"op": "bloom", "value": "x"},
        ], "5")

    def test_filter_and_count(self):
        self._assert_equal("count_positive", [
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
        ], "3")


# -- Structured Expr tree: the new path (no Python strings in IR) -----------

@pytest.mark.skipif(
    "rust" not in available_targets(),
    reason="rustc not available"
)
class TestStructuredExpr:
    """Prove that structured Expr trees produce correct, cross-target-equal output.

    These tests use the new dict-based expression format instead of Python
    strings. Same plan → same output on Python AND Rust, verified by execution.
    """

    def _assert_both(self, name, ops, expected):
        """Run on both targets, assert pass and cross-target equality."""
        case = make_case(name, ops, expected)
        py = run_single(case, "python")
        rs = run_single(case, "rust")
        assert py.passed, f"Python FAIL: {py.error}\nCode:\n{py.generated_code}"
        assert rs.passed, f"Rust FAIL: {rs.error}\nCode:\n{rs.generated_code}"
        assert py.actual == rs.actual, (
            f"Cross-target divergence on '{name}':\n"
            f"  Python: {repr(py.actual)}\n"
            f"  Rust:   {repr(rs.actual)}"
        )

    def test_add(self):
        """Structured add: 5 + 3 = 8."""
        self._assert_both("add_expr", [
            {"op": "seed", "name": "a", "value": "5"},
            {"op": "seed", "name": "b", "value": "3"},
            {"op": "weave", "inputs": ["a", "b"],
             "expr": {"kind": "add",
                      "left": {"kind": "var", "name": "a"},
                      "right": {"kind": "var", "name": "b"}},
             "output": "result"},
            {"op": "bloom", "value": "result"},
        ], "8")

    def test_cond(self):
        """Structured conditional: max(5, 3) = 5."""
        self._assert_both("max_expr", [
            {"op": "seed", "name": "a", "value": "5"},
            {"op": "seed", "name": "b", "value": "3"},
            {"op": "weave", "inputs": ["a", "b"],
             "expr": {"kind": "cond",
                      "test": {"kind": "gt",
                               "left": {"kind": "var", "name": "a"},
                               "right": {"kind": "var", "name": "b"}},
                      "true": {"kind": "var", "name": "a"},
                      "false": {"kind": "var", "name": "b"}},
             "output": "result"},
            {"op": "bloom", "value": "result"},
        ], "5")

    def test_neg(self):
        """Structured negation: -(-5) = 5."""
        self._assert_both("neg_expr", [
            {"op": "seed", "name": "x", "value": "-5"},
            {"op": "when", "condition": "x < 0", "children": [
                {"op": "weave", "inputs": ["x"],
                 "expr": {"kind": "neg",
                          "operand": {"kind": "var", "name": "x"}},
                 "output": "x"},
            ]},
            {"op": "bloom", "value": "x"},
        ], "5")

    def test_mul_add_nested(self):
        """Structured nested: result * (_i_1 + 1) for factorial."""
        self._assert_both("factorial_expr", [
            {"op": "seed", "name": "n", "value": "5"},
            {"op": "cycle", "count": "n", "accumulator": "result",
             "init": "1",
             "body_expr": {"kind": "mul",
                           "left": {"kind": "var", "name": "result"},
                           "right": {"kind": "add",
                                     "left": {"kind": "var", "name": "_i_1"},
                                     "right": {"kind": "lit", "value": 1}}}},
            {"op": "bloom", "value": "result"},
        ], "120")

    def test_cycle_sum(self):
        """Structured cycle body: total + _i_1 for sum."""
        self._assert_both("sum_expr", [
            {"op": "seed", "name": "n", "value": "5"},
            {"op": "cycle", "count": "n", "accumulator": "total",
             "init": "0",
             "body_expr": {"kind": "add",
                           "left": {"kind": "var", "name": "total"},
                           "right": {"kind": "var", "name": "_i_1"}}},
            {"op": "emit", "value": "total"},
        ], "10")

    def test_mod_eq(self):
        """Structured mod + eq: 4 % 2 == 0 → True/true, print as int via cond."""
        # Use cond to convert to int so output is target-independent
        self._assert_both("even_expr", [
            {"op": "seed", "name": "n", "value": "4"},
            {"op": "weave", "inputs": ["n"],
             "expr": {"kind": "cond",
                      "test": {"kind": "eq",
                               "left": {"kind": "mod",
                                        "left": {"kind": "var", "name": "n"},
                                        "right": {"kind": "lit", "value": 2}},
                               "right": {"kind": "lit", "value": 0}},
                      "true": {"kind": "lit", "value": 1},
                      "false": {"kind": "lit", "value": 0}},
             "output": "even"},
            {"op": "bloom", "value": "even"},
        ], "1")

    def test_index(self):
        """Structured index: items[1] = 20."""
        self._assert_both("index_expr", [
            {"op": "seed", "name": "items", "value": "[10, 20, 30]"},
            {"op": "weave", "inputs": ["items"],
             "expr": {"kind": "index", "target": "items",
                      "idx": {"kind": "lit", "value": 1}},
             "output": "result"},
            {"op": "bloom", "value": "result"},
        ], "20")

    def test_filter_count_structured(self):
        """Full structured: count positives in [-1, 2, -3, 4, 5] = 3."""
        self._assert_both("count_structured", [
            {"op": "seed", "name": "data", "value": "[-1, 2, -3, 4, 5]"},
            {"op": "seed", "name": "count", "value": "0"},
            {"op": "for_each", "name": "x", "value": "data",
             "children": [
                 {"op": "when", "condition": "x > 0", "children": [
                     {"op": "weave", "inputs": ["count"],
                      "expr": {"kind": "add",
                               "left": {"kind": "var", "name": "count"},
                               "right": {"kind": "lit", "value": 1}},
                      "output": "count"},
                 ]},
             ]},
            {"op": "bloom", "value": "count"},
        ], "3")

    # -- Structured when conditions (GBNF hole closure) ----------------------

    def test_structured_when_gt(self):
        """Structured when condition: x > 0 → emit positive."""
        self._assert_both("when_gt", [
            {"op": "seed", "name": "x", "value": "5"},
            {"op": "when",
             "condition": {"kind": "gt",
                           "left": {"kind": "var", "name": "x"},
                           "right": {"kind": "lit", "value": 0}},
             "children": [
                 {"op": "emit", "value": '"yes"'},
             ]},
        ], "yes")

    def test_structured_when_lt_neg(self):
        """Structured when + neg: abs(-5) = 5."""
        self._assert_both("when_lt_abs", [
            {"op": "seed", "name": "x", "value": "-5"},
            {"op": "when",
             "condition": {"kind": "lt",
                           "left": {"kind": "var", "name": "x"},
                           "right": {"kind": "lit", "value": 0}},
             "children": [
                 {"op": "weave", "inputs": ["x"],
                  "expr": {"kind": "neg",
                           "operand": {"kind": "var", "name": "x"}},
                  "output": "x"},
             ]},
            {"op": "bloom", "value": "x"},
        ], "5")

    def test_structured_when_eq_mod(self):
        """Structured compound condition: x % 2 == 0 → even."""
        self._assert_both("when_eq_mod", [
            {"op": "seed", "name": "x", "value": "4"},
            {"op": "when",
             "condition": {"kind": "eq",
                           "left": {"kind": "mod",
                                    "left": {"kind": "var", "name": "x"},
                                    "right": {"kind": "lit", "value": 2}},
                           "right": {"kind": "lit", "value": 0}},
             "children": [
                 {"op": "emit", "value": '"even"'},
             ]},
        ], "even")

    def test_structured_filter_count_full(self):
        """Fully structured: both when condition AND body expr are dicts."""
        self._assert_both("filter_count_full", [
            {"op": "seed", "name": "data", "value": "[-1, 2, -3, 4, 5]"},
            {"op": "seed", "name": "count", "value": "0"},
            {"op": "for_each", "name": "x", "value": "data",
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
            {"op": "bloom", "value": "count"},
        ], "3")
