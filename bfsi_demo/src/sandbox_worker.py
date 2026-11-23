"""Isolated worker for src.sandbox.run when SIGALRM is not armable (Windows, Streamlit threads).

Invoked only as:  python -m src.sandbox_worker --input IN.pkl --output OUT.pkl

This process is a fresh interpreter main thread. It must call sandbox.execute() (in-process,
AST-checked, restricted builtins) and must never call sandbox.run() — that would recurse
into another worker on Windows. The parent is responsible for killing this process on timeout.
No Streamlit import. No network. stdin/stdout are unused (payloads are pickle files).
"""
from __future__ import annotations
import argparse, pickle, sys, traceback
from pathlib import Path


def _work(in_path: Path, out_path: Path) -> int:
    # Import inside main so `python -m` bootstrap stays small and cannot re-enter Streamlit.
    from src.sandbox import UnsafeCode, execute

    try:
        with open(in_path, "rb") as fh:
            job = pickle.load(fh)
        result = execute(job["code"], job["fname"], job.get("tables") or {})
        payload = {"ok": True, "result": result}
    except Exception as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc) or traceback.format_exc(limit=2),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    return 0 if payload["ok"] else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="src.sandbox_worker")
    parser.add_argument("--input", required=True, help="Path to the input pickle")
    parser.add_argument("--output", required=True, help="Path to the output pickle")
    args = parser.parse_args(argv)
    return _work(Path(args.input), Path(args.output))


if __name__ == "__main__":
    sys.exit(main())
