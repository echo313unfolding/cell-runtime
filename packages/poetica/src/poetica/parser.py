"""Poetica parser — tokenize human-readable source into structured elements."""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Element:
    """A single meaningful element in a Poetica program."""
    kind: str
    content: str
    indent: int = 0
    label: str = ""
    target: str = ""
    params: dict = field(default_factory=dict)


class PoeticaParser:
    PATTERNS = [
        ('name',       r'name\s+(.+)'),
        ('seed',       r'seed\s+([^\s]+)\s+with\s+(.+)'),
        ('grow',       r'grow\s+([^\s]+)\s+with\s+(.+)'),
        ('emit_label', r'emit\s+"([^"]+)"\s+(.+)'),
        ('emit',       r'emit\s+(.+)'),
        ('pack',       r'pack\s+(.+)\s+as\s+(\S+)'),
        ('lift',       r'lift\s+([^\s]+)\s+to\s+(.+)'),
        ('use',        r'use\s+([^\s(]+)(?:\s*\(([^)]*)\))?'),
        ('when_in',    r'when\s+(.+?)\s+in\s+(.+?)\s*:'),
        ('when',       r'when\s+([^:]+?)\s*:'),
        ('if',         r'if\s+(.+?)\s+echoes?\s+(.+)'),
        ('else_when',  r'else\s+when\s+([^:]+?)\s*:'),
        ('else',       r'else\s*:'),
        ('flow',       r'flow\s+(.+?)\s+to\s+(.+)'),
        ('bloom',      r'bloom\s+(.+)'),
        ('remember',   r'remember\s+(\w+)\s*:\s*(.+)'),
        ('learn',      r'learn\s+pattern\s+"([^"]+)"'),
        ('for',        r'for\s+each\s+(\w+)\s+in\s+(.+?)\s*:\s*(.*)'),
    ]

    def parse(self, source: str) -> List[Element]:
        elements = []
        for line in source.split('\n'):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            indent = self._indent_level(line)
            element = self._match_line(stripped, indent)
            elements.append(element)
        return elements

    def _match_line(self, line: str, indent: int) -> Element:
        for kind, pattern in self.PATTERNS:
            m = re.match(pattern, line)
            if m:
                return self._build_element(kind, m, line, indent)
        return Element(kind='text', content=line, indent=indent)

    def _build_element(self, kind: str, m: re.Match, line: str, indent: int) -> Element:
        if kind == 'name':
            return Element(kind='name', content=line, indent=indent, label=m.group(1).strip())
        if kind == 'seed':
            return Element(kind='seed', content=line, indent=indent,
                           label=m.group(1).strip(), target=m.group(2).strip())
        if kind == 'grow':
            return Element(kind='grow', content=line, indent=indent,
                           label=m.group(1).strip(), target=m.group(2).strip())
        if kind == 'emit_label':
            return Element(kind='emit', content=line, indent=indent,
                           label=m.group(1).strip(), target=m.group(2).strip())
        if kind == 'emit':
            return Element(kind='emit', content=line, indent=indent,
                           target=m.group(1).strip())
        if kind == 'pack':
            return Element(kind='pack', content=line, indent=indent,
                           label=m.group(1).strip(), target=m.group(2).strip())
        if kind == 'lift':
            return Element(kind='lift', content=line, indent=indent,
                           label=m.group(1).strip(), target=m.group(2).strip())
        if kind == 'use':
            params = {}
            if m.group(2):
                for pair in m.group(2).split(','):
                    pair = pair.strip()
                    if ':' in pair:
                        k, v = pair.split(':', 1)
                        params[k.strip()] = v.strip()
                    elif pair:
                        params[pair] = True
            return Element(kind='use', content=line, indent=indent,
                           label=m.group(1).strip(), params=params)
        if kind == 'when_in':
            return Element(kind='when_in', content=line, indent=indent,
                           label=m.group(1).strip(), target=m.group(2).strip())
        if kind == 'when':
            return Element(kind='when', content=line, indent=indent,
                           label=m.group(1).strip())
        if kind == 'if':
            return Element(kind='if', content=line, indent=indent,
                           label=m.group(1).strip(), target=m.group(2).strip())
        if kind == 'else_when':
            return Element(kind='else_when', content=line, indent=indent,
                           label=m.group(1).strip())
        if kind == 'else':
            return Element(kind='else', content=line, indent=indent)
        if kind == 'flow':
            return Element(kind='flow', content=line, indent=indent,
                           label=m.group(1).strip(), target=m.group(2).strip())
        if kind == 'bloom':
            return Element(kind='bloom', content=line, indent=indent,
                           target=m.group(1).strip())
        if kind == 'remember':
            return Element(kind='remember', content=line, indent=indent,
                           label=m.group(1).strip(), target=m.group(2).strip())
        if kind == 'learn':
            return Element(kind='learn', content=line, indent=indent,
                           label=m.group(1).strip())
        if kind == 'for':
            return Element(kind='for', content=line, indent=indent,
                           label=m.group(1).strip(), target=m.group(2).strip(),
                           params={'body': m.group(3).strip()} if m.group(3) else {})
        return Element(kind='text', content=line, indent=indent)

    def _indent_level(self, line: str) -> int:
        stripped = line.lstrip()
        if not stripped:
            return 0
        leading = len(line) - len(stripped)
        return leading // 4
