"""Poetica compiler — compile parsed elements into intermediate representation."""

import hashlib
import json
from typing import Any, Dict, List

from poetica.parser import Element


class PoeticaCompiler:
    def compile(self, elements: List[Element], source: str = "") -> Dict[str, Any]:
        name = ""
        ops = []

        for el in elements:
            if el.kind == 'name':
                name = el.label
                continue
            if el.kind == 'text':
                continue
            op = self._element_to_op(el)
            if op:
                ops.append(op)

        source_hash = hashlib.sha256(source.encode()).hexdigest() if source else ""
        return {
            "version": "poetica-ir-v1",
            "name": name,
            "source_hash": source_hash,
            "ops": ops,
        }

    def _element_to_op(self, el: Element) -> Dict[str, Any]:
        if el.kind == 'seed':
            return {"op": "seed", "name": el.label, "value": el.target, "indent": el.indent}
        if el.kind == 'grow':
            return {"op": "grow", "name": el.label, "source": el.target, "indent": el.indent}
        if el.kind == 'emit':
            op = {"op": "emit", "value": el.target, "indent": el.indent}
            if el.label:
                op["label"] = el.label
            return op
        if el.kind == 'pack':
            return {"op": "pack", "data": el.label, "format": el.target, "indent": el.indent}
        if el.kind == 'lift':
            return {"op": "lift", "name": el.label, "dest": el.target, "indent": el.indent}
        if el.kind == 'use':
            return {"op": "use", "tool": el.label, "params": el.params, "indent": el.indent}
        if el.kind == 'when_in':
            return {"op": "when_in", "subject": el.label, "container": el.target, "indent": el.indent}
        if el.kind == 'when':
            return {"op": "when", "condition": el.label, "indent": el.indent}
        if el.kind == 'if':
            return {"op": "if", "left": el.label, "right": el.target, "indent": el.indent}
        if el.kind == 'else_when':
            return {"op": "else_when", "condition": el.label, "indent": el.indent}
        if el.kind == 'else':
            return {"op": "else", "indent": el.indent}
        if el.kind == 'flow':
            return {"op": "flow", "source": el.label, "dest": el.target, "indent": el.indent}
        if el.kind == 'bloom':
            return {"op": "bloom", "value": el.target, "indent": el.indent}
        if el.kind == 'remember':
            return {"op": "remember", "key": el.label, "value": el.target, "indent": el.indent}
        if el.kind == 'learn':
            return {"op": "learn", "pattern": el.label, "indent": el.indent}
        if el.kind == 'for':
            body = el.params.get('body', '')
            return {"op": "for", "var": el.label, "collection": el.target,
                    "body": body, "indent": el.indent}
        return None
