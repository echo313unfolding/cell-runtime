"""Poetica CLI — compile poems from the command line.

Usage:
    poetica compile program.poem --type sonnet
    poetica compile program.poem --target python --level 2
    poetica targets
    echo "seed x with 42\nemit x" | poetica compile - --type haiku
"""

import argparse
import json
import sys

from poetica import compile_poem, __version__
from poetica.parser import PoeticaParser
from poetica.compiler import PoeticaCompiler
from poetica.gate import Gate, GateError
from poetica.receipt import Receipt
from poetica.emitters import get_emitter, list_targets


def cmd_compile(args):
    """Compile a .poem file to target language code."""
    if args.file == '-':
        source = sys.stdin.read()
    else:
        with open(args.file, 'r') as f:
            source = f.read()

    target = args.type or args.target or "python"
    level = args.level

    try:
        code = compile_poem(source, target=target, level=level)
    except GateError as e:
        print(f"Gate REJECT: {e}", file=sys.stderr)
        sys.exit(1)

    if args.receipt:
        parser = PoeticaParser()
        elements = parser.parse(source)
        compiler = PoeticaCompiler()
        ir = compiler.compile(elements, source)
        gate = Gate(level=level, allow_external=(level >= 4))
        decisions = [d.to_dict() for d in gate.check_all(ir)]
        receipt = Receipt(
            source_hash=ir["source_hash"],
            target=target,
            gate_level=level,
            gate_policy=gate.policy_hash,
            decisions=decisions,
            output_hash=Receipt.hash_output(code),
        )
        print(receipt.to_json(), file=sys.stderr)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(code)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(code)


def cmd_targets(args):
    """List available poem types and their target languages."""
    targets = list_targets()
    for poem_type, lang in targets.items():
        print(f"  {poem_type:10s} -> {lang}")


def cmd_check(args):
    """Dry-run: parse and gate-check without generating code."""
    if args.file == '-':
        source = sys.stdin.read()
    else:
        with open(args.file, 'r') as f:
            source = f.read()

    parser = PoeticaParser()
    elements = parser.parse(source)
    compiler = PoeticaCompiler()
    ir = compiler.compile(elements, source)

    gate = Gate(level=args.level, allow_external=(args.level >= 4))
    decisions = gate.check_all(ir)

    rejected = [d for d in decisions if d.verdict == "REJECT"]
    if rejected:
        for d in rejected:
            print(f"  REJECT  {d.op:12s}  {d.reason} (needs L{d.level})", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"OK: {len(decisions)} ops, all allowed at L{args.level}")


def main():
    p = argparse.ArgumentParser(
        prog="poetica",
        description="Write what you mean. Compile to what you need.",
    )
    p.add_argument("--version", action="version", version=f"poetica {__version__}")
    sub = p.add_subparsers(dest="command")

    # compile
    comp = sub.add_parser("compile", help="Compile a .poem file to code")
    comp.add_argument("file", help="Path to .poem file (or - for stdin)")
    comp.add_argument("--type", "-t", help="Poem type (sonnet, haiku, ballad, ode, prose, verse)")
    comp.add_argument("--target", help="Target language (python, rust, javascript, go, bash, sql)")
    comp.add_argument("--level", "-l", type=int, default=1, help="Capability level 1-5 (default: 1)")
    comp.add_argument("--output", "-o", help="Output file (default: stdout)")
    comp.add_argument("--receipt", action="store_true", help="Print receipt to stderr")
    comp.set_defaults(func=cmd_compile)

    # targets
    tgt = sub.add_parser("targets", help="List available poem types")
    tgt.set_defaults(func=cmd_targets)

    # check
    chk = sub.add_parser("check", help="Gate-check a poem without compiling")
    chk.add_argument("file", help="Path to .poem file (or - for stdin)")
    chk.add_argument("--level", "-l", type=int, default=1, help="Capability level 1-5 (default: 1)")
    chk.set_defaults(func=cmd_check)

    args = p.parse_args()
    if not args.command:
        p.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
