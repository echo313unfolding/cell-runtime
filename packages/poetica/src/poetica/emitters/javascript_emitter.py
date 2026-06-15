"""Ballad — JavaScript emitter. Event-driven, flowing, async."""

from typing import Any, Dict, List
from poetica.emitters.base import BaseEmitter


class JavaScriptEmitter(BaseEmitter):
    INDENT = "  "
    COMMENT = "//"
    LANG = "javascript"

    def _fn_open(self, name: str) -> List[str]:
        safe = name.replace(':', '_').replace('-', '_').replace('.', '_')
        return [f"", f"function {safe}() {{"]

    def _fn_close(self, name: str) -> List[str]:
        safe = name.replace(':', '_').replace('-', '_').replace('.', '_')
        return [f"}}", f"", f"{safe}();"]

    def _op_seed(self, op: Dict[str, Any]) -> str:
        return f"const {op['name']} = {self._quote(op['value'])};"

    def _op_grow(self, op: Dict[str, Any]) -> str:
        return f"const {op['name']} = build({self._quote(op['source'])});"

    def _op_emit(self, op: Dict[str, Any]) -> str:
        if op.get('label'):
            return f'console.log(`[{op["label"]}] ${{{self._quote(op["value"])}}}`);\n'
        return f"console.log({self._quote(op['value'])});"

    def _op_pack(self, op: Dict[str, Any]) -> str:
        return f"const {op['format']}Packed = compress({op['data']}, {self._quote(op['format'])});"

    def _op_lift(self, op: Dict[str, Any]) -> str:
        return f"await deploy({op['name']}, {self._quote(op['dest'])});"

    def _op_use(self, op: Dict[str, Any]) -> str:
        params = ", ".join(f"{k}: {self._quote(str(v))}" for k, v in op.get('params', {}).items())
        if params:
            return f"const result = {op['tool']}({{ {params} }});"
        return f"const result = {op['tool']}();"

    def _op_when(self, op: Dict[str, Any]) -> str:
        return f"if ({op['condition']}) {{"

    def _op_when_in(self, op: Dict[str, Any]) -> str:
        return f"if ({op['container']}.includes({op['subject']})) {{"

    def _op_if(self, op: Dict[str, Any]) -> str:
        return f"if ({op['left']} === {op['right']}) {{"

    def _op_flow(self, op: Dict[str, Any]) -> str:
        return f"let {op['dest']} = {op['source']};"

    def _op_bloom(self, op: Dict[str, Any]) -> str:
        return f"return {self._quote(op['value'])};"

    def _op_remember(self, op: Dict[str, Any]) -> str:
        return f"state[{self._quote(op['key'])}] = {self._quote(op['value'])};"

    def _op_learn(self, op: Dict[str, Any]) -> str:
        return f'model.fit({{ pattern: {self._quote(op["pattern"])} }});'

    def _op_for(self, op: Dict[str, Any]) -> str:
        lines = [f"for (const {op['var']} of {op['collection']}) {{"]
        if op.get('body'):
            lines.append(f"  {op['body']};")
        lines.append("}")
        return lines
