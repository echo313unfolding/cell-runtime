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
        return f"let mut {op['name']} = {self._quote(op['value'])};"

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
        return f"if {self._render_expr(op['condition'])} {{"

    def _op_when_in(self, op: Dict[str, Any]) -> str:
        return f"if {op['container']}.contains(&{op['subject']}) {{"

    def _op_if(self, op: Dict[str, Any]) -> str:
        return f"if {op['left']} == {op['right']} {{"

    def _op_else_when(self, op: Dict[str, Any]) -> str:
        return f"}} else if {self._render_expr(op['condition'])} {{"

    def _op_else(self, op: Dict[str, Any]) -> str:
        return "} else {"

    def _op_flow(self, op: Dict[str, Any]) -> str:
        return f"let {op['dest']} = {op['source']};"

    def _op_bloom(self, op: Dict[str, Any]) -> str:
        return f'println!("{{}}", {self._quote(op["value"])});'

    def _op_remember(self, op: Dict[str, Any]) -> str:
        return f'_state.insert("{op["key"]}", {self._quote(op["value"])});'

    def _op_learn(self, op: Dict[str, Any]) -> str:
        return f'// learn pattern: {op["pattern"]}'

    def _op_for(self, op: Dict[str, Any]) -> str:
        return f"for {op['var']} in {op['collection']} {{"

    # -- Semantic ops (IR-native rendering) ----

    def _render_expr(self, expr) -> str:
        """Render expression using Rust syntax. Handles string (legacy) and dict (structured)."""
        if isinstance(expr, dict):
            from poetica.plan_ir import render_expr_rust
            return render_expr_rust(expr)
        # Legacy: string expr with ternary fallback
        ternary = None
        from poetica.plan_ir import _parse_ternary
        ternary = _parse_ternary(str(expr))
        if ternary:
            return f"if {ternary['condition']} {{ {ternary['true_val']} }} else {{ {ternary['false_val']} }}"
        return str(expr)

    def _op_weave(self, op: Dict[str, Any]) -> str:
        is_reassignment = op.get("is_reassignment", False)
        rendered = self._render_expr(op["expr"])
        if is_reassignment:
            return f"{op['output']} = {rendered};"
        return f"let mut {op['output']} = {rendered};"

    def _op_cycle_init(self, op: Dict[str, Any]) -> str:
        return f"let mut {op['accumulator']} = {self._quote(op['init'])};"

    def _op_cycle_for(self, op: Dict[str, Any]) -> str:
        return f"for {op['iter_var']} in 0..{op['count']} {{"

    def _op_cycle_update(self, op: Dict[str, Any]) -> str:
        rendered = self._render_expr(op["body_expr"])
        return f"{op['accumulator']} = {rendered};"
