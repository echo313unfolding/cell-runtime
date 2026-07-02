"""Poetica exec_oracle — cross-target execution harness for plan-IR verification.

Run emitted programs against expected I/O to prove multi-target correctness.
Python-only version is the minimum for MBPP bakeoff numbers.
Cross-target (Rust/JS/Bash/SQL) extends the proof to all backends.

Build order (from ROADMAP):
  exec_oracle -> MBPP bakeoff (Python) -> cross-target oracle -> GBNF test

Available toolchains on box: Python, rustc 1.95, node v22, sqlite3 3.37, bash.
Missing: Go (not installed). 5 of 6 backends verifiable.

Usage:
    from poetica.exec_oracle import OracleCase, run_oracle, run_single

    case = OracleCase(
        plan_json='{"name": "hello", "ops": [{"op": "seed", "name": "x", "value": "42"}, {"op": "emit", "value": "x"}]}',
        expected_stdout="42",
    )
    verdict = run_single(case, "python")
    assert verdict.passed
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from poetica.plan_ir import program_from_json, lower_plan


# -- Data types --------------------------------------------------------------

@dataclass
class OracleCase:
    """A single test case: complete plan + expected stdout."""
    plan_json: str
    expected_stdout: str
    description: str = ""


@dataclass
class OracleVerdict:
    """Result of executing one case on one target."""
    target: str
    case_index: int
    description: str
    passed: bool
    actual: str
    expected: str
    exit_code: int
    stderr: str
    wall_time_s: float
    generated_code: str
    error: str = ""


@dataclass
class OracleReport:
    """Aggregate result across all cases and targets."""
    targets_tested: List[str]
    verdicts: List[OracleVerdict] = field(default_factory=list)
    toolchain: Dict[str, Optional[str]] = field(default_factory=dict)

    @property
    def total_cases(self) -> int:
        return len(self.verdicts)

    @property
    def total_passed(self) -> int:
        return sum(1 for v in self.verdicts if v.passed)

    @property
    def pass_rate(self) -> float:
        return self.total_passed / max(self.total_cases, 1)

    def summary(self) -> Dict[str, Any]:
        by_target: Dict[str, Any] = {}
        for v in self.verdicts:
            if v.target not in by_target:
                by_target[v.target] = {"passed": 0, "failed": 0, "cases": []}
            if v.passed:
                by_target[v.target]["passed"] += 1
            else:
                by_target[v.target]["failed"] += 1
            by_target[v.target]["cases"].append({
                "description": v.description,
                "passed": v.passed,
                "error": v.error,
            })
        return {
            "total": self.total_cases,
            "passed": self.total_passed,
            "pass_rate": round(self.pass_rate, 4),
            "targets": by_target,
            "toolchain": self.toolchain,
        }


# -- Toolchain detection ----------------------------------------------------

def detect_toolchains() -> Dict[str, Optional[str]]:
    """Detect available compilers/interpreters for each target."""
    tools: Dict[str, Optional[str]] = {}
    for name, cmd in [
        ("python", "python3"),
        ("rust", "rustc"),
        ("javascript", "node"),
        ("bash", "bash"),
        ("sql", "sqlite3"),
        ("go", "go"),
    ]:
        path = shutil.which(cmd)
        if path:
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                tools[name] = result.stdout.strip().split("\n")[0]
            except Exception:
                tools[name] = path
        else:
            tools[name] = None
    return tools


def available_targets() -> List[str]:
    """Return list of targets with available toolchains."""
    tools = detect_toolchains()
    return [name for name, version in tools.items() if version is not None]


# -- Helpers -----------------------------------------------------------------

def _normalize_output(s: str) -> str:
    """Strip trailing whitespace per line and trailing newlines."""
    lines = s.rstrip("\n").split("\n")
    return "\n".join(line.rstrip() for line in lines)


def _safe_fn_name(name: str) -> str:
    """Sanitize plan name to match emitter naming convention."""
    return name.replace(":", "_").replace("-", "_").replace(".", "_")


# -- Bloom patching ----------------------------------------------------------

def _patch_python_bloom(code: str, plan_name: str) -> str:
    """Patch Python __main__ block to print bloom (return) values.

    The emitter generates `fn_name()` in __main__ which discards return values.
    This patches it to capture and print bloom output.
    """
    safe = _safe_fn_name(plan_name)

    # Check if function body has a return statement
    has_return = False
    in_fn = False
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith(f"def {safe}"):
            in_fn = True
        elif in_fn and stripped.startswith("return "):
            has_return = True
            break
        elif in_fn and stripped.startswith("if __name__"):
            break

    if not has_return:
        return code

    old_block = f'if __name__ == "__main__":\n    {safe}()'
    new_block = (
        f'if __name__ == "__main__":\n'
        f'    _r = {safe}()\n'
        f'    if _r is not None:\n'
        f'        print(_r)'
    )
    return code.replace(old_block, new_block, 1)


# -- Target runners ---------------------------------------------------------

def _run_python(code: str, timeout: float) -> Tuple[str, str, int, float]:
    """Execute Python code. Returns (stdout, stderr, exit_code, wall_time)."""
    fd, path = tempfile.mkstemp(suffix=".py", dir="/tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)

        t0 = time.monotonic()
        result = subprocess.run(
            ["python3", path],
            capture_output=True, text=True, timeout=timeout,
        )
        wall = time.monotonic() - t0
        return result.stdout, result.stderr, result.returncode, wall
    except subprocess.TimeoutExpired:
        return "", "timeout", -1, timeout
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _run_rust(code: str, timeout: float) -> Tuple[str, str, int, float]:
    """Compile and execute Rust code."""
    fd, src = tempfile.mkstemp(suffix=".rs", dir="/tmp")
    binary = src.replace(".rs", "")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)

        comp = subprocess.run(
            ["rustc", "-o", binary, src],
            capture_output=True, text=True, timeout=30,
        )
        if comp.returncode != 0:
            return "", comp.stderr, comp.returncode, 0.0

        t0 = time.monotonic()
        result = subprocess.run(
            [binary], capture_output=True, text=True, timeout=timeout,
        )
        wall = time.monotonic() - t0
        return result.stdout, result.stderr, result.returncode, wall
    except subprocess.TimeoutExpired:
        return "", "timeout", -1, timeout
    finally:
        for p in [src, binary]:
            if os.path.exists(p):
                os.unlink(p)


def _run_node(code: str, timeout: float) -> Tuple[str, str, int, float]:
    """Execute JavaScript code with Node.js."""
    fd, path = tempfile.mkstemp(suffix=".js", dir="/tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)

        t0 = time.monotonic()
        result = subprocess.run(
            ["node", path],
            capture_output=True, text=True, timeout=timeout,
        )
        wall = time.monotonic() - t0
        return result.stdout, result.stderr, result.returncode, wall
    except subprocess.TimeoutExpired:
        return "", "timeout", -1, timeout
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _run_bash(code: str, timeout: float) -> Tuple[str, str, int, float]:
    """Execute Bash code."""
    fd, path = tempfile.mkstemp(suffix=".sh", dir="/tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)

        t0 = time.monotonic()
        result = subprocess.run(
            ["bash", path],
            capture_output=True, text=True, timeout=timeout,
        )
        wall = time.monotonic() - t0
        return result.stdout, result.stderr, result.returncode, wall
    except subprocess.TimeoutExpired:
        return "", "timeout", -1, timeout
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _run_sql(code: str, timeout: float) -> Tuple[str, str, int, float]:
    """Execute SQL code with sqlite3 in-memory."""
    try:
        t0 = time.monotonic()
        result = subprocess.run(
            ["sqlite3", ":memory:"],
            input=code, capture_output=True, text=True, timeout=timeout,
        )
        wall = time.monotonic() - t0
        return result.stdout, result.stderr, result.returncode, wall
    except subprocess.TimeoutExpired:
        return "", "timeout", -1, timeout


_RUNNERS = {
    "python": _run_python,
    "sonnet": _run_python,
    "rust": _run_rust,
    "haiku": _run_rust,
    "javascript": _run_node,
    "js": _run_node,
    "ballad": _run_node,
    "bash": _run_bash,
    "prose": _run_bash,
    "sql": _run_sql,
    "verse": _run_sql,
}

# Map poem types / aliases to base toolchain name for detection
_TARGET_BASE = {
    "python": "python", "sonnet": "python",
    "rust": "rust", "haiku": "rust",
    "javascript": "javascript", "js": "javascript", "ballad": "javascript",
    "bash": "bash", "prose": "bash",
    "sql": "sql", "verse": "sql",
    "go": "go", "ode": "go",
}


# -- Core oracle -------------------------------------------------------------

def run_single(
    case: OracleCase,
    target: str = "python",
    timeout: float = 10.0,
    case_index: int = 0,
) -> OracleVerdict:
    """Run one case on one target. Returns a verdict."""
    # Parse and validate plan
    plan = program_from_json(case.plan_json)
    if not plan.valid:
        return OracleVerdict(
            target=target, case_index=case_index,
            description=case.description,
            passed=False, actual="", expected=case.expected_stdout,
            exit_code=-1, stderr="", wall_time_s=0.0,
            generated_code="",
            error=f"Invalid plan: {plan.errors}",
        )

    # Lower to target
    try:
        code = lower_plan(plan, target)
    except ValueError as e:
        return OracleVerdict(
            target=target, case_index=case_index,
            description=case.description,
            passed=False, actual="", expected=case.expected_stdout,
            exit_code=-1, stderr="", wall_time_s=0.0,
            generated_code="",
            error=f"Lowering error: {e}",
        )

    # Patch bloom for Python targets (return → print)
    if target.lower() in ("python", "sonnet"):
        code = _patch_python_bloom(code, plan.name)

    # Get runner
    runner = _RUNNERS.get(target.lower())
    if runner is None:
        return OracleVerdict(
            target=target, case_index=case_index,
            description=case.description,
            passed=False, actual="", expected=case.expected_stdout,
            exit_code=-1, stderr="", wall_time_s=0.0,
            generated_code=code,
            error=f"No runner for target '{target}'",
        )

    # Execute
    stdout, stderr, exit_code, wall_time = runner(code, timeout)

    # Compare output
    actual = _normalize_output(stdout)
    expected = _normalize_output(case.expected_stdout)
    passed = (exit_code == 0) and (actual == expected)

    error = ""
    if stderr == "timeout":
        error = f"Timeout after {timeout}s"
    elif exit_code != 0:
        error = f"Exit code {exit_code}"
    elif actual != expected:
        error = "Output mismatch"

    return OracleVerdict(
        target=target, case_index=case_index,
        description=case.description,
        passed=passed, actual=actual, expected=expected,
        exit_code=exit_code, stderr=stderr,
        wall_time_s=round(wall_time, 4),
        generated_code=code,
        error=error,
    )


def run_oracle(
    cases: List[OracleCase],
    targets: Optional[List[str]] = None,
    timeout: float = 10.0,
) -> OracleReport:
    """Run all cases on all targets. Returns aggregate report."""
    toolchain = detect_toolchains()

    if targets is None:
        targets = ["python"]

    # Filter to available targets
    actual_targets = []
    for t in targets:
        base = _TARGET_BASE.get(t.lower(), t.lower())
        if toolchain.get(base) is not None:
            actual_targets.append(t)

    report = OracleReport(
        targets_tested=actual_targets,
        toolchain=toolchain,
    )

    for i, case in enumerate(cases):
        for target in actual_targets:
            verdict = run_single(case, target, timeout, i)
            report.verdicts.append(verdict)

    return report
