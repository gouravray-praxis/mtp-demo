"""Re-execute every accepted LLM feature on the real tables and verify customer-index alignment against the
exported matrices: unique SK_ID_CURR index, coverage, and matrix column == series.reindex(SK_ID_CURR)."""
from __future__ import annotations
import gc, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from . import config as C, data, gate as G, features_bc as FB, sandbox

def run(log=print):
    specs = json.loads((C.FEAT_DIR / "llm_feature_specs.json").read_text())
    acc = [s for s in specs if s.get("in_B") or s.get("in_C")]
    app_proc, _, y, _ = FB.build_shared_base(data.load("app_train", as_category=False))
    ids = app_proc[C.ID_COL].to_numpy("int64")
    loader = lambda n: data.load(n, as_category=True)
    acc.sort(key=lambda s: sorted(G.tables_needed(s["python_code"]) - {"app_train"}))
    cache, out, cur = {}, [], None
    mats = {P: pq.read_table(C.FEAT_DIR / f"features_{P}.parquet", columns=[C.ID_COL] + [s["feature_name"] for s in acc if s.get(f"in_{P}")]).to_pandas()
            for P in ("B", "C")}
    for P in mats:
        assert np.array_equal(mats[P][C.ID_COL].to_numpy(), ids)
    for s in acc:
        used = G.tables_needed(s["python_code"]); key = tuple(sorted(used - {"app_train"}))
        if key != cur:
            for k in list(cache):
                if k not in key: del cache[k]
            gc.collect(); cur = key
        t = G.prepare_tables(used, s["python_code"], app_proc, loader, cache)
        r = sandbox.run(s["python_code"], s["feature_name"], t); del t
        idx_unique = bool(r.index.is_unique); in_ids = float(pd.Index(r.index).isin(ids).mean())
        aligned = r.reindex(ids).to_numpy("float64")
        eq = {P: bool(np.array_equal(mats[P][s["feature_name"]].to_numpy("float64"), aligned, equal_nan=True)) for P in mats if s.get(f"in_{P}")}
        rec = {"feature_name": s["feature_name"], "index_unique": idx_unique, "index_in_customer_ids": round(in_ids, 6),
               "coverage_of_customers": round(float(pd.Index(ids).isin(r.index).mean()), 6), "matrix_equal": eq,
               "passed": idx_unique and in_ids == 1.0 and all(eq.values())}
        out.append(rec); log(f"  audit {s['feature_name']}: {rec}")
    cache.clear(); del app_proc, mats; gc.collect()
    res = {"all_passed": all(r["passed"] for r in out), "n_features": len(out), "features": out}
    (C.OUT_DIR / "llm_alignment_audit.json").write_text(json.dumps(res, indent=2))
    return res
