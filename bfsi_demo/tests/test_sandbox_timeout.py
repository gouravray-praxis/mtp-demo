"""Portable sandbox timeout: POSIX SIGALRM on the main thread, otherwise isolated
`python -m src.sandbox_worker`. Native Windows is not executed here; missing
SIGALRM/alarm is emulated by deleting those attributes.
"""
from __future__ import annotations
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import sandbox


SAFE = (
    "def compute_ok(app_train, bureau, bureau_bal, prev_app, install, cc_bal, pos_cash):\n"
    "    return app_train['v'] + 1\n"
)

# Tight for-loop (allowed AST). Not a while-loop and not a huge numpy alloc.
SLOW = (
    "def compute_slow(app_train, bureau, bureau_bal, prev_app, install, cc_bal, pos_cash):\n"
    "    s = 0\n"
    "    for i in range(10 ** 9):\n"
    "        s = s + 1\n"
    "    return s\n"
)

BANNED = (
    "def compute_x(app_train, bureau, bureau_bal, prev_app, install, cc_bal, pos_cash):\n"
    "    return open('/etc/passwd').read()\n"
)


def _tables():
    return {"app_train": pd.DataFrame({"v": [1, 2, 3]})}


def test_run_returns_series_on_main_thread():
    out = sandbox.run(SAFE, "ok", _tables(), timeout_s=10)
    assert list(out) == [2, 3, 4]


def test_run_from_non_main_thread_still_returns():
    """Streamlit/AppTest run off the main thread; SIGALRM is unavailable there."""
    box = {}

    def _target():
        try:
            box["out"] = sandbox.run(SAFE, "ok", _tables(), timeout_s=15)
        except Exception as exc:  # noqa: BLE001
            box["err"] = exc

    t = threading.Thread(target=_target)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "sandbox.run hung in a worker thread"
    assert "err" not in box, box.get("err")
    assert list(box["out"]) == [2, 3, 4]


def test_missing_sigalrm_uses_worker(monkeypatch):
    """Emulate Windows: signal module has neither SIGALRM nor alarm."""
    monkeypatch.delattr(sandbox.signal, "SIGALRM", raising=False)
    monkeypatch.delattr(sandbox.signal, "alarm", raising=False)
    assert sandbox.can_use_sigalrm() is False
    out = sandbox.run(SAFE, "ok", _tables(), timeout_s=15)
    assert list(out) == [2, 3, 4]


def test_timeout_kills_slow_loop_via_worker(monkeypatch):
    """Designed timeout: a long for-loop is rejected; worker process is killed."""
    monkeypatch.setattr(sandbox, "can_use_sigalrm", lambda: False)
    with pytest.raises(sandbox.Timeout, match="time limit"):
        sandbox.run(SLOW, "slow", _tables(), timeout_s=1)


def test_timeout_from_non_main_thread():
    box = {}

    def _target():
        try:
            sandbox.run(SLOW, "slow", _tables(), timeout_s=1)
            box["ok"] = True
        except sandbox.Timeout as exc:
            box["timeout"] = exc
        except Exception as exc:  # noqa: BLE001
            box["err"] = exc

    t = threading.Thread(target=_target)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "timeout path hung in a worker thread"
    assert "timeout" in box, box
    assert "ok" not in box and "err" not in box


def test_import_ban_still_enforced_before_and_in_worker(monkeypatch):
    with pytest.raises(sandbox.UnsafeCode):
        sandbox.check(BANNED)
    with pytest.raises(sandbox.UnsafeCode):
        sandbox.run(BANNED, "x", _tables(), timeout_s=10)
    monkeypatch.setattr(sandbox, "can_use_sigalrm", lambda: False)
    with pytest.raises(sandbox.UnsafeCode):
        sandbox.run(BANNED, "x", _tables(), timeout_s=10)


def test_run_api_signature_unchanged():
    assert sandbox.run.__defaults__ == (300,)
    out = sandbox.run(SAFE, "ok", _tables())  # default timeout_s
    assert list(out) == [2, 3, 4]
