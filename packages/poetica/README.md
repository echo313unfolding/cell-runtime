# Poetica

Write what you mean. Compile to what you need.

Poetica is a human-readable DSL that compiles to real code in six languages. The **poem type** selects the backend. The **gate** controls what's allowed.

```
seed message with "hello world"
emit message
```

That's a program. Compile it:

```bash
poetica compile hello.poem --type sonnet    # → Python
poetica compile hello.poem --type haiku     # → Rust
poetica compile hello.poem --type ballad    # → JavaScript
poetica compile hello.poem --type ode       # → Go
poetica compile hello.poem --type prose     # → Bash
poetica compile hello.poem --type verse     # → SQL
```

## Why

Programming is gatekept by syntax. SQL proved you can make databases accessible to non-programmers with a constrained language. Poetica does the same for general computation.

**For non-coders:** Write programs in words you already understand. The gate prevents you from doing anything dangerous at lower levels. You can't `lift` data to a remote server at Level 1 — you have to earn it.

**For coders:** Write polyglot programs once. Think in operations, not syntax. `seed`, `grow`, `emit`, `pack`, `lift`, `bloom` — these map to the same intent in every language. The poem type is just a compiler flag.

## Install

```bash
pip install poetica
```

Or from source:

```bash
cd packages/poetica
pip install -e .
```

## Poem Types

| Poem Type | Target     | Character                        |
|-----------|------------|----------------------------------|
| sonnet    | Python     | Flowing, expressive, readable    |
| haiku     | Rust       | Minimal, precise, safe           |
| ballad    | JavaScript | Event-driven, flowing, async     |
| ode       | Go         | Structured, concurrent, explicit |
| prose     | Bash       | Imperative, step-by-step, direct |
| verse     | SQL        | Declarative, set-based           |

## Grammar

```
name <identifier>              — name the program
seed <thing> with <value>      — create / initialize
grow <thing> with <source>     — build / construct
emit <message>                 — output / print
emit "<label>" <message>       — labeled output
pack <data> as <format>        — serialize / bundle
lift <thing> to <dest>         — deploy / upload
use <tool>(key: val, ...)      — call external tool
when <condition>:              — conditional
when <x> in <y>:              — membership test
if <a> echoes <b>             — equality check
flow <source> to <dest>        — pipe / assign
bloom <value>                  — return
remember <key>: <value>        — persist
learn pattern "<name>"         — fit / recognize
for each <x> in <y>: <body>   — iterate
```

## Capability Levels (The Gate)

Every compilation goes through the gate. The gate checks each operation against the current level. If an operation exceeds the level, compilation fails. No bypass.

| Level | Name           | Operations Unlocked              |
|-------|----------------|----------------------------------|
| L1    | Pure           | seed, emit, flow, bloom, remember |
| L2    | + Logic        | if, when, for                     |
| L3    | + Transform    | pack, grow, learn                 |
| L4    | + External     | lift, use                         |
| L5    | Unrestricted   | everything                        |

**L1 is safe.** A Level 1 program can create values and print them. It cannot build, transform, deploy, or call external tools. Give this to anyone.

```bash
# This works at L1:
poetica compile hello.poem --level 1

# This fails at L1 (grow needs L3):
poetica compile garden.poem --level 1
# Gate REJECT: op='grow': LEVEL-EXCEEDED
```

## Python API

```python
from poetica import compile_poem

# Simple
code = compile_poem("seed x with 42\nemit x", target="python")
print(code)

# With gate level
code = compile_poem(source, target="rust", level=3)

# All poem types
from poetica.emitters import list_targets
print(list_targets())
# {'sonnet': 'python', 'haiku': 'rust', 'ballad': 'javascript', ...}
```

## CLI

```bash
# Compile to stdout
poetica compile program.poem --type sonnet

# Compile to file
poetica compile program.poem --type haiku -o program.rs

# Set capability level
poetica compile program.poem -t ballad --level 3

# Gate-check without compiling
poetica check program.poem --level 2

# List targets
poetica targets

# Pipe from stdin
echo "seed x with 42\nemit x" | poetica compile - --type prose
```

## Receipt

Every compilation can produce a receipt — an audit trail of what was checked and what was generated.

```bash
poetica compile hello.poem --type sonnet --receipt 2>receipt.json
```

```json
{
  "schema": "poetica.receipt.v1",
  "source_hash": "a1b2c3...",
  "target": "sonnet",
  "gate_level": 1,
  "all_allowed": true,
  "decisions": [
    {"op": "seed", "verdict": "ALLOW", "reason": "OK", "level": 1},
    {"op": "emit", "verdict": "ALLOW", "reason": "OK", "level": 1}
  ],
  "output_hash": "d4e5f6..."
}
```

## Examples

See `examples/` for complete programs:

- `hello.poem` — Level 1, pure output
- `garden.poem` — Level 3, growth and transformation
- `fizzbuzz.poem` — Level 2, logic and iteration
- `pipeline.poem` — Level 4, external data pipeline
- `deploy.poem` — Level 4, full deployment workflow

## Same program, six languages

```
name greeter
seed message with "hello world"
emit message
bloom "done"
```

**sonnet (Python):**
```python
def greeter():
    message = "hello world"
    print(message)
    return "done"
```

**haiku (Rust):**
```rust
fn greeter() {
    let message = "hello world";
    println!("{}", message);
    return "done";
}
```

**ballad (JavaScript):**
```javascript
function greeter() {
  const message = "hello world";
  console.log(message);
  return "done";
}
```

**ode (Go):**
```go
func greeter() {
	message := "hello world"
	fmt.Println(message)
	return "done"
}
```

**prose (Bash):**
```bash
greeter() {
    message="hello world"
    echo "message"
    echo "done"
}
```

**verse (SQL):**
```sql
SET @message = 'hello world';
SELECT "message" AS output;
SELECT 'done' AS result;
```

## License

Apache-2.0
