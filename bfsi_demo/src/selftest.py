"""Package integrity + recompute identity tests on the REAL exported artifacts -> outputs/selftest.json."""
from __future__ import annotations
import json, re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from . import config as C
from . import feature_recompute as FR

SECRET_RE = re.compile(rb"(gsk_[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]{16,}|\"key\"\s*:\s*\"[0-9a-f]{32}\")")
SKIP_DIRS = {"raw", "cache", ".venv", "__pycache__"}

def secret_scan(root: Path):
    hits = []
    for p in root.rglob("*"):
        if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_DIRS) and p.stat().st_size < 200_000_000:
            with open(p, "rb") as fh:
                if SECRET_RE.search(fh.read()):
                    hits.append(str(p.relative_to(root)))
    return hits

def run(root=C.ROOT, n_ids=40, log=print):
    root = Path(root); checks = []
    def chk(name, ok, detail=""):
        checks.append({"name": name, "passed": bool(ok), "detail": str(detail)[:300]}); log(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")
    sc = pd.read_parquet(root / "outputs" / "scored_table.parquet")
    chk("scored_ids_unique", sc[C.ID_COL].is_unique, len(sc))
    raw = pd.read_parquet(root / "features" / "raw_inputs.parquet")
    chk("raw_inputs_ids_match_scored", np.array_equal(raw[C.ID_COL].to_numpy(), sc[C.ID_COL].to_numpy()))
    b = FR.load_bundle(root)
    # selected customers: first/last, random, plus null / sentinel / categorical variations
    rng = np.random.default_rng(42)
    ids = list(sc[C.ID_COL].iloc[[0, len(sc) - 1]]) + list(rng.choice(sc[C.ID_COL].to_numpy(), n_ids, replace=False))
    special = {"DAYS_EMPLOYED_365243": raw["DAYS_EMPLOYED"] == 365243, "EXT_SOURCE_1_null": raw["EXT_SOURCE_1"].isna(),
               "AMT_GOODS_PRICE_null": raw["AMT_GOODS_PRICE"].isna(), "AMT_ANNUITY_null": raw["AMT_ANNUITY"].isna(),
               "OWN_CAR_AGE_null": raw["OWN_CAR_AGE"].isna(), "CNT_FAM_MEMBERS_null": raw["CNT_FAM_MEMBERS"].isna(),
               "all_EXT_null": raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().all(axis=1)}
    spec_ids = {}
    for k, m in special.items():
        sel = raw.loc[m, C.ID_COL].head(3).tolist(); spec_ids[k] = sel; ids += sel
    try:  # categorical variety: one customer per CODE_GENDER / NAME_CONTRACT_TYPE level (if stored in a matrix as OHE)
        fA = pq.read_table(root / "features" / "features_B.parquet").to_pandas()
        for col in [c for c in fA.columns if c.startswith(("CODE_GENDER_", "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE_"))][:6]:
            v = fA.loc[fA[col] == 1, C.ID_COL].head(1).tolist(); spec_ids[f"cat_{col}"] = v; ids += v
        del fA
    except Exception as e:
        log("  categorical sample skipped", e)
    ids = list(dict.fromkeys(int(i) for i in ids))
    der_by = {"A": FR.derive_app(raw, "A"), "BC": FR.derive_app(raw, "C")}
    fe_cols = [c for c in FR.FE_NAMES]
    chk("fe_formulas_A_equal_Aprime_all_rows", all(np.array_equal(der_by["A"][c].to_numpy(), der_by["BC"][c].to_numpy(), equal_nan=True) for c in fe_cols))
    for P in ("A", "B", "C"):
        f = root / "features" / f"features_{P}.parquet"
        names = json.loads((root / "features" / f"feature_names_{P}.json").read_text())
        sch = json.loads((root / "features" / f"schema_{P}.json").read_text())
        model = joblib.load(root / "models" / f"model_{P}.pkl")
        pf = pq.ParquetFile(f); cols = pf.schema_arrow.names
        chk(f"{P}_columns_order_eq_model", cols == [C.ID_COL] + names and names == list(model.booster_.feature_name()), len(names))
        chk(f"{P}_no_target_in_features", C.TARGET_COL not in cols)
        chk(f"{P}_float64", all(str(pf.schema_arrow.field(n).type) == "double" for n in names))
        chk(f"{P}_rows_307511", pf.metadata.num_rows == len(sc) == 307511, pf.metadata.num_rows)
        chk(f"{P}_schema_matches_names", [x["name"] for x in sch["features"]] == names and sch["n_features"] == len(names))
        idcol = pq.read_table(f, columns=[C.ID_COL]).column(0).to_numpy()
        chk(f"{P}_ids_unique_and_order_eq_scored", len(set(idcol)) == len(idcol) and np.array_equal(idcol, sc[C.ID_COL].to_numpy()))
        chk(f"{P}_no_null_oof_pred", sc[[f"oof_{P}", f"pred_{P}"]].notna().all().all())
        # full-parquet batched prediction parity vs scored table
        maxd, off = 0.0, 0
        for batch in pf.iter_batches(batch_size=50_000, columns=names):
            Xb = batch.to_pandas(); n = len(Xb)
            maxd = max(maxd, float(np.max(np.abs(model.predict_proba(Xb)[:, 1] - sc[f"pred_{P}"].to_numpy()[off:off + n])))); off += n
        chk(f"{P}_full_parquet_pred_parity_batched", off == len(sc) and maxd <= 1e-12, f"rows={off} maxdiff={maxd:.3g}")
        # vectorised identity for ALL 307,511 rows on every recomputable app-level feature
        der_all = der_by['A' if P == 'A' else 'BC']
        rec = [n for n in names if n in der_all.columns]
        stored = pq.read_table(f, columns=rec).to_pandas()
        bad = [n for n in rec if not np.array_equal(stored[n].to_numpy("float64"), der_all[n].to_numpy("float64"), equal_nan=True)]
        llm_rl = [s for s in b.llm_specs if s.get("row_level") and s["feature_name"] in names]
        for s in llm_rl:
            v = FR.llm_series(s["python_code"], s["feature_name"], der_all).reindex(raw[C.ID_COL]).to_numpy("float64")
            if not np.array_equal(pq.read_table(f, columns=[s["feature_name"]]).column(0).to_numpy(), v, equal_nan=True):
                bad.append(s["feature_name"])
        chk(f"{P}_recompute_identity_all_rows_vectorised", not bad, f"{len(rec)} app + {len(llm_rl)} row-level LLM features; bad={bad[:5]}")
        del stored
        # per-customer recompute through the public API (what the app calls)
        mism, pdiff = [], 0.0
        scx = sc.set_index(C.ID_COL)
        for i in ids:
            st = FR.stored_row(b, P, i); rc = FR.recompute_row(b, pipeline=P, sk_id=int(i), edits={})
            if not np.array_equal(st.to_numpy(), rc.to_numpy(), equal_nan=True):
                mism.append((int(i), [n for n in names if not np.array_equal(st[n].to_numpy(), rc[n].to_numpy(), equal_nan=True)][:5]))
            pdiff = max(pdiff, abs(model.predict_proba(rc)[:, 1][0] - float(scx.at[i, f"pred_{P}"])))
        chk(f"{P}_recompute_identity_selected_customers", not mism, f"{len(ids)} ids incl. {list(spec_ids)}; mismatches={mism[:3]}")
        chk(f"{P}_no_edit_pred_matches_scored", pdiff <= 1e-9, f"maxdiff={pdiff:.3g}")
        # edits propagate (not stale) and never touch held features
        held = {x["name"] for x in sch["features"] if not x["editable_via_recompute"]}
        i = int(raw.loc[raw["AMT_GOODS_PRICE"].notna() & raw["AMT_ANNUITY"].notna(), C.ID_COL].iloc[5]); r = FR.raw_row(b, i)
        for field, factor, expect in [("AMT_CREDIT", 1.5, ["FE_CREDIT_TO_INCOME", "FE_CREDIT_TO_GOODS", "FE_ANNUITY_TO_CREDIT"]),
                                      ("AMT_GOODS_PRICE", 0.8, ["FE_CREDIT_TO_GOODS", "FE_GOODS_PRICE_GAP"]),
                                      ("AMT_INCOME_TOTAL", 2.0, ["FE_ANNUITY_TO_INCOME", "FE_CREDIT_TO_INCOME", "FE_INCOME_PER_PERSON"]),
                                      ("EXT_SOURCE_2", 0.5, ["FE_EXT_SOURCE_MEAN", "EXT_SOURCE_2"]),
                                      ("DAYS_EMPLOYED", None, ["DAYS_EMPLOYED_ANOM", "EMPLOYMENT_YEARS", "FE_EMPLOYMENT_TO_AGE"])]:
            val = 365243.0 if factor is None else float(r[field].iloc[0]) * factor
            ch = FR.changed_features(b, P, i, {field: val})
            exp = [e for e in expect if e in names]
            ok = set(exp) <= set(ch) and not (set(ch) & held) and (len(ch) > 0 or not any(
                field in x["depends_on"] for x in sch["features"] if x["editable_via_recompute"]))
            chk(f"{P}_edit_{field}_propagates", ok, f"changed={ch[:10]} expected_present={exp}")
        p0 = model.predict_proba(FR.recompute_row(b, P, i, {}))[:, 1][0]
        p1 = model.predict_proba(FR.recompute_row(b, P, i, {"AMT_CREDIT": float(r["AMT_CREDIT"].iloc[0]) * 3}))[:, 1][0]
        chk(f"{P}_edit_changes_prediction", np.isfinite(p1) and p1 != p0, f"{p0:.6f}->{p1:.6f}")
    hits = secret_scan(root)
    chk("secret_scan_package", not hits, hits[:5])
    out = {"all_passed": all(c["passed"] for c in checks), "n_checks": len(checks), "checks": checks, "n_ids_tested": len(ids)}
    (root / "outputs" / "selftest.json").write_text(json.dumps(out, indent=2))
    return out
