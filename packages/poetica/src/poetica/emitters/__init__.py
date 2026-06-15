"""Poetica emitters — poem types that generate real code.

Each emitter translates Poetica IR into a target language.
The poem type IS the compiler selector.

| Poem Type  | Target     | Character                            |
|------------|------------|--------------------------------------|
| sonnet     | Python     | Flowing, expressive, readable        |
| haiku      | Rust       | Minimal, precise, safe               |
| ballad     | JavaScript | Event-driven, flowing, async         |
| ode        | Go         | Structured, concurrent, explicit     |
| prose      | Bash       | Imperative, step-by-step, direct     |
| verse      | SQL        | Declarative, set-based, query-like   |
"""

from poetica.emitters.python_emitter import PythonEmitter
from poetica.emitters.javascript_emitter import JavaScriptEmitter
from poetica.emitters.rust_emitter import RustEmitter
from poetica.emitters.go_emitter import GoEmitter
from poetica.emitters.bash_emitter import BashEmitter
from poetica.emitters.sql_emitter import SQLEmitter

_EMITTERS = {
    "python": PythonEmitter,
    "sonnet": PythonEmitter,
    "javascript": JavaScriptEmitter,
    "js": JavaScriptEmitter,
    "ballad": JavaScriptEmitter,
    "rust": RustEmitter,
    "haiku": RustEmitter,
    "go": GoEmitter,
    "ode": GoEmitter,
    "bash": BashEmitter,
    "prose": BashEmitter,
    "sql": SQLEmitter,
    "verse": SQLEmitter,
}


def get_emitter(target: str):
    """Get an emitter by target name or poem type."""
    key = target.lower().strip()
    if key not in _EMITTERS:
        available = sorted(set(cls.__name__ for cls in _EMITTERS.values()))
        raise ValueError(
            f"Unknown target '{target}'. "
            f"Available: {', '.join(_EMITTERS.keys())}"
        )
    return _EMITTERS[key]()


def list_targets() -> dict:
    """Return mapping of poem type -> language."""
    return {
        "sonnet": "python",
        "haiku": "rust",
        "ballad": "javascript",
        "ode": "go",
        "prose": "bash",
        "verse": "sql",
    }
