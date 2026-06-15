"""Haiku — Rust emitter. Minimal, precise, safe."""

from typing import Any, Dict, List, Optional
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

    def _fn_preamble(self, ir) -> List[str]:
        lines = []
        ops = {op["op"] for op in ir.get("ops", [])}
        if "remember" in ops:
            lines.append("let mut _state = std::collections::HashMap::new();")
        return lines

    def _footer(self, ir) -> List[str]:
        name = ir.get("name", "program").replace(':', '_').replace('-', '_').replace('.', '_')
        return [f"", f"fn main() {{", f"    {name}();", f"}}"]

    def _block_close(self, block_type: str) -> Optional[str]:
        return "}"

    def _op_seed(self, op: Dict[str, Any]) -> str:
        return f"let {op['name']} = {self._quote(op['value'])};"

    def _op_grow(self, op: Dict[str, Any]) -> str:
        return f"{op['name']}.push({self._quote(op['source'])});"

    def _op_emit(self, op: Dict[str, Any]) -> str:
        if op.get('label'):
            return f'println!("[{op["label"]}] {{}}", {op["value"]});'
        return f"println!(\"{{}}\", {self._quote(op['value'])});"

    def _op_pack(self, op: Dict[str, Any]) -> str:
        fmt = op['format']
        data = op['data']
        if fmt == 'json':
            return f"let {data}_json = serde_json::to_string(&{data}).unwrap();"
        return f"let {data}_{fmt} = format!(\"{{}}\", {data});"

    def _op_lift(self, op: Dict[str, Any]) -> str:
        return f"std::fs::write({self._quote(op['dest'])}, {op['name']}.to_string()).unwrap();"

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

    def _op_else_when(self, op: Dict[str, Any]) -> str:
        return f"}} else if {op['condition']} {{"

    def _op_else(self, op: Dict[str, Any]) -> str:
        return "} else {"

    def _op_flow(self, op: Dict[str, Any]) -> str:
        return f"let {op['dest']} = {op['source']};"

    def _op_bloom(self, op: Dict[str, Any]) -> str:
        return f"return {self._quote(op['value'])};"

    def _op_remember(self, op: Dict[str, Any]) -> str:
        return f'_state.insert("{op["key"]}", {self._quote(op["value"])});'

    def _op_learn(self, op: Dict[str, Any]) -> str:
        return f'// learn pattern: {op["pattern"]}'

    def _op_for(self, op: Dict[str, Any]) -> str:
        return f"for {op['var']} in {op['collection']}.iter() {{"
