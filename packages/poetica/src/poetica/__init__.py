"""Poetica — write what you mean, compile to what you need.

A human-readable DSL that compiles to real code in any target language.
Poem types select the backend. The gate controls what's allowed.

    from poetica import compile_poem, Gate

    code = compile_poem('''
    name greeter
    seed message with "hello world"
    emit message
    ''', target="python")

    print(code)
    # def greeter():
    #     message = "hello world"
    #     print(message)
"""

from poetica.parser import PoeticaParser, Element
from poetica.compiler import PoeticaCompiler
from poetica.gate import Gate, GateDecision, GateLevel, GateError
from poetica.receipt import Receipt
from poetica.emitters import get_emitter, list_targets

__version__ = "0.1.0"


def compile_poem(source: str, target: str = "python", level: int = 1) -> str:
    """Compile a poem to target language code.

    Args:
        source: Poetica source text.
        target: Target language ("python", "javascript", "rust", "go", "bash", "sql").
        level: Capability level 1-5. Higher levels unlock more operations.

    Returns:
        Generated source code as a string.

    Raises:
        GateError: If any operation exceeds the capability level.
    """
    parser = PoeticaParser()
    elements = parser.parse(source)

    compiler = PoeticaCompiler()
    ir = compiler.compile(elements, source)

    gate = Gate(level=level, allow_external=(level >= 4))
    gate.check(ir)

    emitter = get_emitter(target)
    return emitter.emit(ir)
