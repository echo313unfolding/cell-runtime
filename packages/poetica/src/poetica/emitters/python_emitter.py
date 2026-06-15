"""Sonnet — Python emitter. Flowing, expressive, readable."""

from typing import Any, Dict, List
from poetica.emitters.base import BaseEmitter


class PythonEmitter(BaseEmitter):
    INDENT = "    "
    COMMENT = "#"
    LANG = "python"

    def _fn_open(self, name: str) -> List[str]:
        safe = name.replace(':', '_').replace('-', '_').replace('.', '_')
        return [f"", f"def {safe}():"]

    def _fn_close(self, name: str) -> List[str]:
        safe = name.replace(':', '_').replace('-', '_').replace('.', '_')
        return [f"", f"", f'if __name__ == "__main__":', f"    {safe}()"]

    def _op_seed(self, op: Dict[str, Any]) -> str:
        return f"{op['name']} = {self._quote(op['value'])}"

    def _op_grow(self, op: Dict[str, Any]) -> str:
        return f"{op['name']} = build({self._quote(op['source'])})"

    def _op_emit(self, op: Dict[str, Any]) -> str:
        if op.get('label'):
            return f'print(f"[{op["label"]}] {{{self._quote(op["value"])}}}")'
        return f"print({self._quote(op['value'])})"

    def _op_pack(self, op: Dict[str, Any]) -> str:
        return f"{op['format']}_packed = compress({op['data']}, format={self._quote(op['format'])})"

    def _op_lift(self, op: Dict[str, Any]) -> str:
        return f"deploy({op['name']}, dest={self._quote(op['dest'])})"

    def _op_use(self, op: Dict[str, Any]) -> str:
        params = ", ".join(f"{k}={self._quote(str(v))}" for k, v in op.get('params', {}).items())
        if params:
            return f"result = {op['tool']}({params})"
        return f"result = {op['tool']}()"

    def _op_when(self, op: Dict[str, Any]) -> str:
        return f"if {op['condition']}:"

    def _op_when_in(self, op: Dict[str, Any]) -> str:
        return f"if {op['subject']} in {op['container']}:"

    def _op_if(self, op: Dict[str, Any]) -> str:
        return f"if {op['left']} == {op['right']}:"

    def _op_flow(self, op: Dict[str, Any]) -> str:
        return f"{op['dest']} = {op['source']}"

    def _op_bloom(self, op: Dict[str, Any]) -> str:
        return f"return {self._quote(op['value'])}"

    def _op_remember(self, op: Dict[str, Any]) -> str:
        return f"state[{self._quote(op['key'])}] = {self._quote(op['value'])}"

    def _op_learn(self, op: Dict[str, Any]) -> str:
        return f'model.fit(pattern={self._quote(op["pattern"])})'

    def _op_for(self, op: Dict[str, Any]) -> str:
        lines = [f"for {op['var']} in {op['collection']}:"]
        if op.get('body'):
            lines.append(f"    {op['body']}")
        return lines
