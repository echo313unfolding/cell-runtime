"""Tests for the poetica standalone package."""

import pytest
from poetica import compile_poem, __version__
from poetica.parser import PoeticaParser, Element
from poetica.compiler import PoeticaCompiler
from poetica.gate import Gate, GateError, GateLevel
from poetica.receipt import Receipt
from poetica.emitters import get_emitter, list_targets


# --- Parser ---

class TestParser:
    def setup_method(self):
        self.parser = PoeticaParser()

    def test_seed(self):
        elements = self.parser.parse("seed x with 42")
        assert len(elements) == 1
        assert elements[0].kind == "seed"
        assert elements[0].label == "x"
        assert elements[0].target == "42"

    def test_grow(self):
        elements = self.parser.parse("grow plant with soil")
        assert elements[0].kind == "grow"
        assert elements[0].label == "plant"
        assert elements[0].target == "soil"

    def test_emit(self):
        elements = self.parser.parse("emit hello")
        assert elements[0].kind == "emit"
        assert elements[0].target == "hello"

    def test_emit_labeled(self):
        elements = self.parser.parse('emit "status" readings')
        assert elements[0].kind == "emit"
        assert elements[0].label == "status"
        assert elements[0].target == "readings"

    def test_pack(self):
        elements = self.parser.parse("pack data as json")
        assert elements[0].kind == "pack"
        assert elements[0].label == "data"
        assert elements[0].target == "json"

    def test_lift(self):
        elements = self.parser.parse("lift artifact to registry")
        assert elements[0].kind == "lift"
        assert elements[0].label == "artifact"
        assert elements[0].target == "registry"

    def test_use_with_params(self):
        elements = self.parser.parse("use healthcheck(url: api, timeout: 30)")
        assert elements[0].kind == "use"
        assert elements[0].label == "healthcheck"
        assert elements[0].params == {"url": "api", "timeout": "30"}

    def test_use_no_params(self):
        elements = self.parser.parse("use cleanup")
        assert elements[0].kind == "use"
        assert elements[0].label == "cleanup"

    def test_when(self):
        elements = self.parser.parse("when ready:")
        assert elements[0].kind == "when"
        assert elements[0].label == "ready"

    def test_when_in(self):
        elements = self.parser.parse("when item in collection:")
        assert elements[0].kind == "when_in"
        assert elements[0].label == "item"
        assert elements[0].target == "collection"

    def test_if_echoes(self):
        elements = self.parser.parse("if status echoes active")
        assert elements[0].kind == "if"
        assert elements[0].label == "status"
        assert elements[0].target == "active"

    def test_flow(self):
        elements = self.parser.parse("flow input to output")
        assert elements[0].kind == "flow"
        assert elements[0].label == "input"
        assert elements[0].target == "output"

    def test_bloom(self):
        elements = self.parser.parse("bloom result")
        assert elements[0].kind == "bloom"
        assert elements[0].target == "result"

    def test_remember(self):
        elements = self.parser.parse("remember key: value")
        assert elements[0].kind == "remember"
        assert elements[0].label == "key"
        assert elements[0].target == "value"

    def test_learn(self):
        elements = self.parser.parse('learn pattern "anomaly"')
        assert elements[0].kind == "learn"
        assert elements[0].label == "anomaly"

    def test_for_each(self):
        elements = self.parser.parse("for each item in items: emit item")
        assert elements[0].kind == "for"
        assert elements[0].label == "item"
        assert elements[0].target == "items"
        assert elements[0].params == {"body": "emit item"}

    def test_name(self):
        elements = self.parser.parse("name my_program")
        assert elements[0].kind == "name"
        assert elements[0].label == "my_program"

    def test_comments_skipped(self):
        elements = self.parser.parse("# this is a comment\nseed x with 1")
        assert len(elements) == 1
        assert elements[0].kind == "seed"

    def test_blank_lines_skipped(self):
        elements = self.parser.parse("\n\nseed x with 1\n\n")
        assert len(elements) == 1

    def test_unknown_text(self):
        elements = self.parser.parse("something random")
        assert elements[0].kind == "text"

    def test_multi_line(self):
        source = """name test
seed a with 1
seed b with 2
emit a
bloom b"""
        elements = self.parser.parse(source)
        assert len(elements) == 5
        assert [e.kind for e in elements] == ["name", "seed", "seed", "emit", "bloom"]


# --- Compiler ---

class TestCompiler:
    def setup_method(self):
        self.parser = PoeticaParser()
        self.compiler = PoeticaCompiler()

    def test_basic_ir(self):
        elements = self.parser.parse("name test\nseed x with 42\nemit x")
        ir = self.compiler.compile(elements, "name test\nseed x with 42\nemit x")
        assert ir["version"] == "poetica-ir-v1"
        assert ir["name"] == "test"
        assert len(ir["ops"]) == 2
        assert ir["ops"][0]["op"] == "seed"
        assert ir["ops"][1]["op"] == "emit"
        assert ir["source_hash"] != ""

    def test_all_ops(self):
        source = """name full
seed x with 1
grow y with x
emit y
pack y as json
lift y to dest
use tool(k: v)
when ready:
flow a to b
bloom done"""
        elements = self.parser.parse(source)
        ir = self.compiler.compile(elements, source)
        ops = [op["op"] for op in ir["ops"]]
        assert ops == ["seed", "grow", "emit", "pack", "lift", "use", "when", "flow", "bloom"]


# --- Gate ---

class TestGate:
    def test_l1_allows_pure(self):
        gate = Gate(level=1)
        ir = {"ops": [
            {"op": "seed", "name": "x", "value": "1"},
            {"op": "emit", "value": "x"},
            {"op": "bloom", "value": "done"},
        ]}
        decisions = gate.check(ir)
        assert all(d.verdict == "ALLOW" for d in decisions)

    def test_l1_rejects_logic(self):
        gate = Gate(level=1)
        ir = {"ops": [{"op": "when", "condition": "ready"}]}
        with pytest.raises(GateError) as exc_info:
            gate.check(ir)
        assert exc_info.value.decision.reason == "LEVEL-EXCEEDED"

    def test_l2_allows_logic(self):
        gate = Gate(level=2)
        ir = {"ops": [
            {"op": "seed", "name": "x", "value": "1"},
            {"op": "when", "condition": "ready"},
            {"op": "if", "left": "a", "right": "b"},
            {"op": "for", "var": "i", "collection": "items"},
        ]}
        decisions = gate.check(ir)
        assert all(d.verdict == "ALLOW" for d in decisions)

    def test_l2_rejects_transform(self):
        gate = Gate(level=2)
        ir = {"ops": [{"op": "grow", "name": "y", "source": "x"}]}
        with pytest.raises(GateError):
            gate.check(ir)

    def test_l3_allows_transform(self):
        gate = Gate(level=3)
        ir = {"ops": [
            {"op": "grow", "name": "y", "source": "x"},
            {"op": "pack", "data": "y", "format": "json"},
            {"op": "learn", "pattern": "anomaly"},
        ]}
        decisions = gate.check(ir)
        assert all(d.verdict == "ALLOW" for d in decisions)

    def test_l3_rejects_external(self):
        gate = Gate(level=3)
        ir = {"ops": [{"op": "lift", "name": "x", "dest": "remote"}]}
        with pytest.raises(GateError) as exc_info:
            gate.check(ir)
        # External check fires before level check
        assert exc_info.value.decision.reason == "EXTERNAL-DENIED"

    def test_l4_requires_allow_external(self):
        gate = Gate(level=4, allow_external=False)
        ir = {"ops": [{"op": "use", "tool": "curl", "params": {}}]}
        with pytest.raises(GateError) as exc_info:
            gate.check(ir)
        assert exc_info.value.decision.reason == "EXTERNAL-DENIED"

    def test_l4_with_external(self):
        gate = Gate(level=4, allow_external=True)
        ir = {"ops": [
            {"op": "lift", "name": "x", "dest": "remote"},
            {"op": "use", "tool": "curl", "params": {}},
        ]}
        decisions = gate.check(ir)
        assert all(d.verdict == "ALLOW" for d in decisions)

    def test_unknown_op_rejected(self):
        gate = Gate(level=3)
        ir = {"ops": [{"op": "explode", "target": "everything"}]}
        with pytest.raises(GateError) as exc_info:
            gate.check(ir)
        assert exc_info.value.decision.reason == "UNKNOWN-OP"

    def test_check_all_no_raise(self):
        gate = Gate(level=1)
        ir = {"ops": [
            {"op": "seed", "name": "x", "value": "1"},
            {"op": "grow", "name": "y", "source": "x"},
        ]}
        decisions = gate.check_all(ir)
        assert decisions[0].verdict == "ALLOW"
        assert decisions[1].verdict == "REJECT"

    def test_invalid_level(self):
        with pytest.raises(ValueError):
            Gate(level=0)
        with pytest.raises(ValueError):
            Gate(level=6)

    def test_policy_hash_stable(self):
        g1 = Gate(level=2)
        g2 = Gate(level=2)
        assert g1.policy_hash == g2.policy_hash

    def test_l5_allows_unknown(self):
        gate = Gate(level=5, allow_external=True)
        ir = {"ops": [{"op": "custom_thing"}]}
        decisions = gate.check(ir)
        assert decisions[0].verdict == "ALLOW"


# --- Emitters ---

class TestEmitters:
    SIMPLE_SOURCE = 'name hello\nseed msg with "greeting"\nemit msg'

    def _compile(self, source=None):
        parser = PoeticaParser()
        compiler = PoeticaCompiler()
        elements = parser.parse(source or self.SIMPLE_SOURCE)
        return compiler.compile(elements, source or self.SIMPLE_SOURCE)

    def test_python_emitter(self):
        emitter = get_emitter("python")
        code = emitter.emit(self._compile())
        assert "def hello():" in code
        assert 'msg = "greeting"' in code
        assert "print(msg)" in code

    def test_sonnet_is_python(self):
        e1 = get_emitter("sonnet")
        e2 = get_emitter("python")
        ir = self._compile()
        assert e1.emit(ir) == e2.emit(ir)

    def test_javascript_emitter(self):
        emitter = get_emitter("javascript")
        code = emitter.emit(self._compile())
        assert "function hello()" in code
        assert "const msg" in code
        assert "console.log" in code

    def test_ballad_is_javascript(self):
        e1 = get_emitter("ballad")
        e2 = get_emitter("javascript")
        ir = self._compile()
        assert e1.emit(ir) == e2.emit(ir)

    def test_rust_emitter(self):
        emitter = get_emitter("rust")
        code = emitter.emit(self._compile())
        assert "fn hello()" in code
        assert "let msg" in code
        assert "println!" in code

    def test_haiku_is_rust(self):
        e1 = get_emitter("haiku")
        e2 = get_emitter("rust")
        ir = self._compile()
        assert e1.emit(ir) == e2.emit(ir)

    def test_go_emitter(self):
        emitter = get_emitter("go")
        code = emitter.emit(self._compile())
        assert "func hello()" in code
        assert 'msg := "greeting"' in code
        assert "fmt.Println" in code

    def test_ode_is_go(self):
        e1 = get_emitter("ode")
        e2 = get_emitter("go")
        ir = self._compile()
        assert e1.emit(ir) == e2.emit(ir)

    def test_bash_emitter(self):
        emitter = get_emitter("bash")
        code = emitter.emit(self._compile())
        assert "#!/usr/bin/env bash" in code
        assert "hello()" in code
        assert "echo" in code

    def test_prose_is_bash(self):
        e1 = get_emitter("prose")
        e2 = get_emitter("bash")
        ir = self._compile()
        assert e1.emit(ir) == e2.emit(ir)

    def test_sql_emitter(self):
        emitter = get_emitter("sql")
        code = emitter.emit(self._compile())
        assert "SET @msg" in code
        assert "SELECT" in code

    def test_verse_is_sql(self):
        e1 = get_emitter("verse")
        e2 = get_emitter("sql")
        ir = self._compile()
        assert e1.emit(ir) == e2.emit(ir)

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError):
            get_emitter("cobol")

    def test_list_targets(self):
        targets = list_targets()
        assert targets["sonnet"] == "python"
        assert targets["haiku"] == "rust"
        assert targets["ballad"] == "javascript"
        assert targets["ode"] == "go"
        assert targets["prose"] == "bash"
        assert targets["verse"] == "sql"


# --- compile_poem top-level ---

class TestCompilePoem:
    def test_basic(self):
        code = compile_poem("name test\nseed x with 42\nemit x")
        assert "x = " in code
        assert "print" in code

    def test_all_targets(self):
        source = "name test\nseed x with 42\nemit x"
        for target in ["python", "javascript", "rust", "go", "bash", "sql"]:
            code = compile_poem(source, target=target)
            assert len(code) > 0

    def test_all_poem_types(self):
        source = "name test\nseed x with 42\nemit x"
        for poem_type in ["sonnet", "haiku", "ballad", "ode", "prose", "verse"]:
            code = compile_poem(source, target=poem_type)
            assert len(code) > 0

    def test_gate_blocks(self):
        source = "lift data to remote"
        with pytest.raises(GateError):
            compile_poem(source, level=1)

    def test_level_2(self):
        source = "seed x with 1\nwhen x > 0:"
        code = compile_poem(source, level=2)
        assert len(code) > 0

    def test_version(self):
        assert __version__ == "0.1.0"


# --- Receipt ---

class TestReceipt:
    def test_receipt_creation(self):
        r = Receipt(
            source_hash="abc123",
            target="python",
            gate_level=1,
            gate_policy="pol123",
            decisions=[{"op": "seed", "verdict": "ALLOW", "reason": "OK", "level": 1}],
            output_hash="def456",
        )
        d = r.to_dict()
        assert d["schema"] == "poetica.receipt.v1"
        assert d["all_allowed"] is True
        assert d["source_hash"] == "abc123"

    def test_receipt_json(self):
        r = Receipt(
            source_hash="abc", target="rust", gate_level=2, gate_policy="p",
            decisions=[], output_hash="xyz",
        )
        j = r.to_json()
        assert '"poetica.receipt.v1"' in j

    def test_hash_output(self):
        h = Receipt.hash_output("hello")
        assert len(h) == 64  # SHA256 hex


# --- End-to-end: example files ---

class TestExamples:
    def _read_example(self, name):
        import pathlib
        path = pathlib.Path(__file__).parent.parent / "examples" / name
        return path.read_text()

    def test_hello_l1(self):
        source = self._read_example("hello.poem")
        for target in ["python", "rust", "javascript", "go", "bash", "sql"]:
            code = compile_poem(source, target=target, level=1)
            assert len(code) > 10

    def test_garden_l3(self):
        source = self._read_example("garden.poem")
        code = compile_poem(source, target="python", level=3)
        assert "grow" in code.lower() or "build" in code.lower()

    def test_garden_blocked_at_l1(self):
        source = self._read_example("garden.poem")
        with pytest.raises(GateError):
            compile_poem(source, target="python", level=1)

    def test_pipeline_l4(self):
        source = self._read_example("pipeline.poem")
        code = compile_poem(source, target="python", level=4)
        assert len(code) > 10

    def test_pipeline_blocked_at_l3(self):
        source = self._read_example("pipeline.poem")
        with pytest.raises(GateError):
            compile_poem(source, target="python", level=3)

    def test_deploy_all_targets(self):
        source = self._read_example("deploy.poem")
        for target in ["sonnet", "haiku", "ballad", "ode", "prose", "verse"]:
            code = compile_poem(source, target=target, level=4)
            assert len(code) > 10
