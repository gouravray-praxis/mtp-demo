"""Helpers shared by A and B/C feature code (category-safe, numerically identical to object dtype)."""
from __future__ import annotations
import numpy as np
import pandas as pd

STATUS_MAP = {'C': 0, 'X': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}

def map_status_num(s: pd.Series) -> pd.Series:
    """Equivalent of s.map(STATUS_MAP).fillna(0) for object or category dtype (no object blow-up)."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        lut = np.array([STATUS_MAP.get(str(c), np.nan) for c in s.cat.categories] + [np.nan], dtype="float64")
        codes = s.cat.codes.to_numpy()
        vals = lut[np.where(codes < 0, len(lut) - 1, codes)]
        return pd.Series(vals, index=s.index).fillna(0)
    return s.map(STATUS_MAP).fillna(0)

def eq_mask(s: pd.Series, value: str) -> pd.Series:
    """(s == value) as plain bool numpy-backed Series for object or category."""
    return pd.Series(np.asarray(s == value, dtype=bool), index=s.index)
