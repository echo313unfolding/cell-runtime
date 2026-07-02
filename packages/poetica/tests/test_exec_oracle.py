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
