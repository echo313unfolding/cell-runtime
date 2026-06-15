"""Haiku — Rust emitter. Minimal, precise, safe."""

from typing import Any, Dict, List
from poetica.emitters.base import BaseEmitter


class RustEmitter(BaseEmitter):
    INDENT = "    "
    COMMENT = "//"
    LANG = "rust"

    def _fn_open(self, name: str) -> List[str]:
        safe = name.replace(':', '_').replace('-', '_').replace('.', '_')
        return [f"", f"fn {safe}() {{"]

    def _fn_close(self, name: str) -> List[str]:
        return [f"}}"]

    def _footer(self, ir) -> List[str]:
        name = ir.get("name", "program").replace(':', '_').replace('-', '_').replace('.', '_')
        return [f"", f"fn main() {{", f"    {name}();", f"}}"]

    def _op_seed(self, op: Dict[str, Any]) -> str:
        return f"let {op['name']} = {self._quote(op['value'])};"

    def _op_grow(self, op: Dict[str, Any]) -> str:
        return f"let {op['name']} = build({self._quote(op['source'])});"

    def _op_emit(self, op: Dict[str, Any]) -> str:
        if op.get('label'):
            return f'println!("[{op["label"]}] {{}}", {self._quote(op["value"])});'
        return f"println!(\"{{}}\", {self._quote(op['value'])});"

    def _op_pack(self, op: Dict[str, Any]) -> str:
        return f"let {op['format']}_packed = compress(&{op['data']}, Format::{op['format'].title()});"

    def _op_lift(self, op: Dict[str, Any]) -> str:
        return f"deploy(&{op['name']}, {self._quote(op['dest'])});"

    def _op_use(self, op: Dict[str, Any]) -> str:
        params = ", ".join(f"{k}: {self._quote(str(v))}" for k, v in op.get('params', {}).items())
        if params:
            return f"let result = {op['tool']}({params});"
        return f"let result = {op['tool']}();"

    def _op_when(self, op: Dict[str, Any]) -> str:
        return f"if {op['condition']} {{"

    def _op_when_in(self, op: Dict[str, Any]) -> str:
        return f"if {op['container']}.contains(&{op['subject']}) {{"

    def _op_if(self, op: Dict[str, Any]) -> str:
        return f"if {op['left']} == {op['right']} {{"

    def _op_flow(self, op: Dict[str, Any]) -> str:
        return f"let {op['dest']} = {op['source']};"

    def _op_bloom(self, op: Dict[str, Any]) -> str:
        return f"return {self._quote(op['value'])};"

    def _op_remember(self, op: Dict[str, Any]) -> str:
        return f'state.insert({self._quote(op["key"])}, {self._quote(op["value"])});'

    def _op_learn(self, op: Dict[str, Any]) -> str:
        return f'model.fit({self._quote(op["pattern"])});'

    def _op_for(self, op: Dict[str, Any]) -> str:
        lines = [f"for {op['var']} in {op['collection']}.iter() {{"]
        if op.get('body'):
            lines.append(f"    {op['body']};")
        lines.append("}")
        return lines
