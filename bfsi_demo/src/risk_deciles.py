"""Fixed OOF risk-decile cutoffs for the Streamlit demo.

Cutoffs are the nine finite quantiles q10..q90 of each pipeline's out-of-fold
score on the full official scored table. They are loaded from
features/risk_decile_cutoffs.json and must never be refit on a customer, an
edit, a demo cohort, or an incoming batch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PIPELINES = ("A", "B", "C")
SOURCE_COLUMNS = {"A": "oof_A", "B": "oof_B", "C": "oof_C"}


def compute_cutoffs(values, quantiles=QUANTILES) -> list[float]:
    """Nine finite q10..q90 cutoffs from an OOF score vector (NaNs dropped)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("no finite scores from which to compute cutoffs")
    qs = np.asarray(list(quantiles), dtype=float)
    return [float(v) for v in np.quantile(x, qs, method="linear")]


def validate_cutoffs(cutoffs) -> list[float]:
    c = [float(v) for v in cutoffs]
    if len(c) != 9 or not all(np.isfinite(v) for v in c):
        raise ValueError("cutoffs must be 9 finite values (q10..q90)")
    if any(c[i] > c[i + 1] for i in range(8)):
        raise ValueError("cutoffs must be nondecreasing")
    return c


def assign_decile(score, cutoffs) -> int | None:
    """Return D1..D10 (D1 = highest risk) or None if the score is not finite.

    cutoffs are [q10, q20, ..., q90] in ascending order.
    decile = 10 - np.searchsorted(cutoffs, score, side="right"):
      score < q10          -> 10 (lowest risk)
      q10 <= score < q20   -> 9
      ...
      score >= q90         -> 1 (highest risk, ties at q90 included)
    A tie at a cutoff enters the higher-risk (lower-index) decile.
    Non-finite scores are not assigned (never silently D10).
    """
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    c = np.asarray(validate_cutoffs(cutoffs), dtype=float)
    idx = int(np.searchsorted(c, value, side="right"))  # 0..9
    return 10 - idx


def format_decile(decile: int | None) -> str:
    if decile is None:
        return "n/a"
    return f"D{int(decile)}"


def load_cutoffs(root: Path) -> dict[str, Any]:
    """Load the packaged fixed OOF cutoffs. Raises if the file is missing or invalid."""
    path = Path(root) / "features" / "risk_decile_cutoffs.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("n_rows") or 0) <= 0:
        raise ValueError("risk_decile_cutoffs.json is missing n_rows")
    out = {"payload": payload, "cutoffs": {}}
    for p in PIPELINES:
        out["cutoffs"][p] = validate_cutoffs(payload["cutoffs"][p])
    return out


def cutoffs_from_scores(scores, columns=SOURCE_COLUMNS) -> dict[str, list[float]]:
    """Build cutoffs from a scored table's OOF columns (for synthetic tests)."""
    return {p: compute_cutoffs(scores[col]) for p, col in columns.items() if col in scores.columns}
