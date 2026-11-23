"""Safety layer for LLM-generated feature code (used by the gate's stage 1/2 and by feature_recompute).
Static AST whitelist + restricted builtins + per-call time limit. Code may only touch its own parameters
(the data tables), pandas (pd) and numpy (np): no imports, no dunder/private attributes, no file/network/
process/eval APIs, no globals/getattr, no I/O methods (read_*/to_*), no while-loops.

Timeout: SIGALRM is used only on a POSIX main thread where it is actually armable. Otherwise execution
is isolated in `python -m src.sandbox_worker` and the parent kills the process when the wall clock expires.
The worker never calls run() (no recursive spawn). AST validation runs in both the parent and the worker.
"""
from __future__ import annotations
import ast, builtins, os, pickle, shutil, signal, subprocess, sys, tempfile, threading
from pathlib import Path
import numpy as np
import pandas as pd

PARAMS = ("app_train", "bureau", "bureau_bal", "prev_app", "install", "cc_bal", "pos_cash")
SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "len", "min", "max", "abs", "sum", "round", "float", "int", "str", "bool", "list", "dict", "tuple",
    "set", "range", "zip", "enumerate", "sorted", "any", "all", "map", "filter", "reversed", "isinstance",
    "ValueError", "ZeroDivisionError", "KeyError", "Exception", "True", "False", "None") if hasattr(builtins, k)}
BANNED_NAMES = {"open", "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr", "globals",
                "locals", "vars", "input", "breakpoint", "dir", "type", "object", "os", "sys", "subprocess",
                "builtins", "help", "exit", "quit", "print", "memoryview", "bytes", "bytearray", "super",
                "classmethod", "staticmethod", "property", "importlib", "socket", "shutil", "pathlib", "Path",
                "requests", "urllib", "pickle", "joblib", "io"}
BANNED_ATTR_PREFIX = ("_", "read_", "to_", "from_")
BANNED_ATTRS = {"system", "popen", "load", "loads", "save", "savez", "savez_compressed", "tofile", "fromfile",
                "memmap", "eval", "query", "pipe", "open", "lib", "ctypes", "f_globals", "gi_frame",
                "testing", "util", "set_option", "options", "io", "api", "plotting", "style", "DataFrame.to_csv"}
ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.Expr,
    ast.Name, ast.Load, ast.Store, ast.Del, ast.Attribute, ast.Subscript, ast.Slice, ast.Constant, ast.List,
    ast.Tuple, ast.Dict, ast.Set, ast.Call, ast.keyword, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.If, ast.For, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.comprehension,
    ast.Lambda, ast.Starred, ast.Pass, ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context,
    ast.JoinedStr, ast.FormattedValue)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

class UnsafeCode(Exception):
    pass

class Timeout(Exception):
    """Feature code exceeded the wall-clock time limit."""

_Timeout = Timeout  # backward-compatible alias used by older tests/callers

ALLOWED_IMPORTS = {("numpy", "np"), ("pandas", "pd")}

class _StripSafeImports(ast.NodeTransformer):
    """`import numpy as np` / `import pandas as pd` are no-ops (np/pd are pre-bound in the sandbox namespace);
    they are removed from the tree. Any other import stays and is rejected below."""
    def visit_Import(self, node):
        if all((a.name, a.asname) in ALLOWED_IMPORTS for a in node.names):
            return ast.Expr(value=ast.Constant(value=None))
        return node

def check(code: str) -> tuple[ast.Module, set[str]]:
    """Raise UnsafeCode if not whitelisted. Returns (tree, used_param_names)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise  # stage 2 (code_error), like the original exec() failure
    tree = ast.fix_missing_locations(_StripSafeImports().visit(tree))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and n.value.value is None)]
    if not tree.body or not all(isinstance(n, ast.FunctionDef) for n in tree.body):
        raise UnsafeCode("top level must contain only function definitions")
    for fn in tree.body:
        if fn.decorator_list:
            raise UnsafeCode("decorators not allowed")
    used = set()
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise UnsafeCode(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or node.id.startswith("__"):
                raise UnsafeCode(f"disallowed name: {node.id}")
            if node.id in PARAMS:
                used.add(node.id)
        if isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith(BANNED_ATTR_PREFIX) or a in BANNED_ATTRS:
                raise UnsafeCode(f"disallowed attribute: .{a}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise UnsafeCode("dunder string constant")
    return tree, used

def string_constants(tree: ast.Module) -> set[str]:
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}

def can_use_sigalrm() -> bool:
    """True only when SIGALRM can actually be armed in this thread (POSIX main thread)."""
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "alarm") or not callable(getattr(signal, "signal", None)):
        return False
    if threading.current_thread() is not threading.main_thread():
        return False
    try:
        old = signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    except (ValueError, OSError, AttributeError):
        return False
    return True

def execute(code: str, fname: str, tables: dict):
    """In-process execution after AST check. No timeout. Used by run() and the worker.

    Never call this from untrusted request paths without a surrounding timeout
    (SIGALRM in run(), or the parent of src.sandbox_worker).
    """
    tree, _ = check(code)
    ns = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np}
    exec(compile(tree, f"<llm:{fname}>", "exec"), ns)
    fn = f"compute_{fname}"
    if fn not in ns:
        fns = [k for k, v in ns.items() if callable(v) and not k.startswith("_") and k not in ("pd", "np")]
        if not fns:
            raise NameError(f"{fn} not found")
        fn = fns[-1]
    args = [tables.get(p) if tables is not None else None for p in PARAMS]
    return ns[fn](*args)

def _alarm(signum, frame):
    raise Timeout("feature code exceeded time limit")

def _run_sigalrm(code: str, fname: str, tables: dict, timeout_s: int):
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(max(1, timeout_s)))
    try:
        return execute(code, fname, tables)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

def _run_subprocess(code: str, fname: str, tables: dict, timeout_s: int):
    """Isolated worker: `python -m src.sandbox_worker`. Parent enforces the wall-clock timeout."""
    work = Path(tempfile.mkdtemp(prefix="bfsi_sbx_"))
    in_path = work / "in.pkl"
    out_path = work / "out.pkl"
    try:
        with open(in_path, "wb") as fh:
            pickle.dump({"code": code, "fname": fname, "tables": tables}, fh, protocol=4)
        env = os.environ.copy()
        root = str(PACKAGE_ROOT)
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["BFSI_SANDBOX_WORKER"] = "1"
        cmd = [sys.executable, "-m", "src.sandbox_worker",
               "--input", str(in_path), "--output", str(out_path)]
        popen_kw = dict(
            cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if sys.platform == "win32":
            popen_kw["creationflags"] = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        proc = subprocess.Popen(cmd, **popen_kw)
        try:
            _out, _err = proc.communicate(timeout=max(1.0, float(timeout_s) + 1.0))
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
            raise Timeout("feature code exceeded time limit")
        if not out_path.is_file():
            detail = (_err or b"").decode("utf-8", "replace")[:400]
            raise RuntimeError(
                f"sandbox worker exited {proc.returncode} without a result"
                + (f": {detail}" if detail else "")
            )
        with open(out_path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("ok"):
            return payload["result"]
        kind = payload.get("error_type") or ""
        msg = payload.get("error") or "sandbox worker failed"
        if kind in ("Timeout", "TimeoutError"):
            raise Timeout(msg)
        if kind == "UnsafeCode":
            raise UnsafeCode(msg)
        if kind == "SyntaxError":
            raise SyntaxError(msg)
        if kind == "NameError":
            raise NameError(msg)
        raise RuntimeError(f"{kind}: {msg}" if kind else msg)
    finally:
        shutil.rmtree(work, ignore_errors=True)

def run(code: str, fname: str, tables: dict, timeout_s: int = 300):
    """Execute whitelisted code with restricted builtins and call compute_<fname> (or last function).

    Signature is unchanged. timeout_s is a wall-clock limit: SIGALRM on a POSIX main
    thread, otherwise an isolated `python -m src.sandbox_worker` process that is killed
    when the limit expires.
    """
    check(code)  # fail fast in the caller (worker also re-checks)
    if timeout_s and int(timeout_s) > 0 and can_use_sigalrm():
        return _run_sigalrm(code, fname, tables, int(timeout_s))
    if timeout_s and int(timeout_s) > 0:
        return _run_subprocess(code, fname, tables, int(timeout_s))
    return execute(code, fname, tables)
