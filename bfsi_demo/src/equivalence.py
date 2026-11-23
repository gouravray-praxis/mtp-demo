"""Assert that reading auxiliary-table strings as 'category' gives bit-identical human features vs object
(the original notebooks' dtype). Uses the first N rows of each real table (or synthetic data)."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from . import data, features_a as FA, features_bc as FB

def _cmp(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    a = a.sort_values('SK_ID_CURR').reset_index(drop=True)
    b = b.sort_values('SK_ID_CURR').reset_index(drop=True)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False, f"shape/cols differ {a.shape} vs {b.shape}"
    for c in a.columns:
        x = a[c].to_numpy(dtype='float64'); y = b[c].to_numpy(dtype='float64')
        if not np.array_equal(x, y, equal_nan=True):
            return False, f"col {c} differs (max abs {np.nanmax(np.abs(x-y))})"
    return True, "identical"

def run(nrows=400_000, data_dir=None, log=print):
    t = time.time()
    kw = {} if data_dir is None else {'data_dir': data_dir}
    res = []
    for mod, tag in ((FA, 'A'), (FB, 'Aprime')):
        for key, (tables, fn) in mod.AUX_BUILDERS.items():
            obj = [data.load(tb, as_category=False, nrows=nrows, **kw) for tb in tables]
            cat = [data.load(tb, as_category=True, nrows=nrows, **kw) for tb in tables]
            ok, msg = _cmp(fn(*obj), fn(*cat))
            res.append({'name': f'equiv_category_vs_object_{tag}_{key}', 'passed': ok, 'detail': f'{msg}; nrows={nrows}'})
            log(f"  {tag}/{key}: {ok} {msg}")
            del obj, cat
    log(f"  equivalence done in {time.time()-t:.1f}s")
    return res
