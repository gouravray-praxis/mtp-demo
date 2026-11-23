"""Generate SYNTHETIC demo customers for the Streamlit app (no real Home Credit rows).

Why: the Kaggle Home Credit Default Risk competition rules do not allow redistributing the data,
so the public repository ships synthetic customers instead of the real 307,511 applicants.

Two stages
----------
1. fit   (needs the REAL package files; run once on the author's machine, never needed by users)
     python scripts/make_synthetic_data.py fit --real-root /path/to/real/BFSI_MTP_Demo
   Writes features/synthetic_marginals.json: per-column AGGREGATE statistics only
   (missing-value rate, 101 smoothed quantiles between the 0.5th and 99.5th percentile, or a
   frequency table for low-cardinality / categorical columns; levels seen fewer than 50 times are
   dropped). No row, no SK_ID_CURR and no TARGET label is stored.

2. generate (default; works on a fresh clone, deterministic for a given seed)
     python scripts/make_synthetic_data.py generate --n 200 --seed 42
   * Application inputs (features/application_edit_inputs.parquet, features/raw_inputs.parquet):
     every column is sampled INDEPENDENTLY from its marginal, so no real row can be copied.
     A few light consistency rules keep customers plausible (goods price and annuity follow the
     credit amount through sampled ratios, family size >= children + 1, employment shorter than age).
   * Feature matrices features_{A,B,C}.parquet:
       - application-level features (raw fields, DAYS->years mechanics, the 20 human FE_ ratios,
         binary / one-hot encodings, row-level LLM features) are computed with the package's OWN
         pipeline code: src.feature_recompute.recompute_row (the same code path the app uses
         for what-if edits);
       - features aggregated from Kaggle's auxiliary tables (bureau, previous applications,
         installments, credit card, POS) cannot be rebuilt without those tables, so they are
         sampled directly from their marginals (same method, NaN rates respected).
   * outputs/scored_table.parquet/.csv: pred_{A,B,C} = the UNCHANGED trained models scored on the
     synthetic rows. oof_{A,B,C} are set equal to pred_{A,B,C} (there is no cross-validation on
     synthetic data). TARGET is a synthetic label drawn from the model-C probability (corrected
     for scale_pos_weight); fold is empty.
   * features/demo_customer_ids.json lists the synthetic IDs (900001 ...), clearly outside the real
     SK_ID_CURR range (100002 .. 456255).
   * outputs/synthetic_data_check.json records the no-edit recompute identity check for every row.

The trained models, feature lists, schemas and risk-decile cutoffs are NOT changed. Decile cutoffs
remain the official OOF cutoffs of the real run, so a synthetic customer's decile says where its
score would fall in the real population.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import feature_recompute as FR  # noqa: E402

PIPELINES = ("A", "B", "C")
ID_COL = "SK_ID_CURR"
ID_START = 900001
MARGINALS = ROOT / "features" / "synthetic_marginals.json"
QLEVELS = np.linspace(0.005, 0.995, 101)
MIN_LEVEL_COUNT = 50          # rarer categorical / discrete levels are dropped
DISCRETE_MAX_UNIQUE = 25
SENTINEL = 365243.0
DEMO_LIST_SIZE = 100          # IDs shown in the app's dropdown; the rest are reachable via Load ID
SCALE_POS_WEIGHT = 11.387150050352467  # src/config.py BEST_PARAMS


# ----------------------------------------------------------------------------- helpers
def _sig(x: float, digits: int = 7) -> float:
    return float(f"{float(x):.{digits}g}")


def _numeric_marginal(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    n = len(s)
    nan_rate = float(s.isna().mean())
    v = s.dropna()
    out: dict = {"kind": "numeric", "nan_rate": round(nan_rate, 6)}
    if v.empty:
        out["kind"] = "all_nan"
        return out
    vc = v.value_counts()
    if len(vc) <= DISCRETE_MAX_UNIQUE:
        vc = vc[vc >= MIN_LEVEL_COUNT]
        if vc.empty:
            vc = v.value_counts().head(1)
        out["kind"] = "discrete"
        out["values"] = [_sig(x) for x in vc.index.tolist()]
        out["probs"] = [round(float(c) / float(vc.sum()), 6) for c in vc.tolist()]
        return out
    out["integer"] = bool(np.all(np.isclose(v.to_numpy(), np.round(v.to_numpy()))))
    out["quantiles"] = [_sig(x) for x in np.quantile(v.to_numpy(), QLEVELS)]
    return out


def _string_marginal(s: pd.Series) -> dict:
    nan_rate = float(s.isna().mean())
    vc = s.dropna().astype(str).value_counts()
    vc = vc[vc >= MIN_LEVEL_COUNT]
    return {"kind": "category", "nan_rate": round(nan_rate, 6),
            "values": vc.index.tolist(), "probs": [round(float(c) / float(vc.sum()), 6) for c in vc.tolist()]}


def _sample(m: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    kind = m["kind"]
    if kind == "all_nan":
        return np.full(n, np.nan)
    if kind in ("discrete", "category"):
        p = np.asarray(m["probs"], dtype=float); p = p / p.sum()
        idx = rng.choice(len(p), size=n, p=p)
        vals = np.asarray(m["values"], dtype=object if kind == "category" else float)[idx]
    else:
        q = np.asarray(m["quantiles"], dtype=float)
        u = rng.uniform(QLEVELS[0], QLEVELS[-1], n)
        vals = np.interp(u, QLEVELS, q)
        if m.get("integer"):
            vals = np.round(vals)
    miss = rng.random(n) < float(m.get("nan_rate", 0.0))
    if kind == "category":
        vals = vals.astype(object); vals[miss] = None
    else:
        vals = vals.astype(float); vals[miss] = np.nan
    return vals


def _settable_columns(pipeline: str, names: list[str], edit_cols: dict, encoding: dict, specs: list[dict]) -> set:
    """Model columns that src.feature_recompute.recompute_row rebuilds from application inputs."""
    out = {c for c in FR.FE_INPUTS + FR.MECHANICAL + FR.FE_NAMES if c in names}
    skipped = set(FR.FE_INPUTS + FR.MECHANICAL + FR.FE_NAMES)
    out |= {c for c, dt in edit_cols.items() if c in names and c not in skipped and dt != "string"
            and c not in FR.FORBIDDEN_EDITS}
    out |= {c for c in (encoding.get("binary_maps") or {}) if c in names}
    for _col, mapping in ((encoding.get("ohe_columns") or {}).get(pipeline) or {}).items():
        out |= {d for d in mapping.values() if d in names}
    out |= {s["feature_name"] for s in specs if s.get("row_level") and s["feature_name"] in names}
    return out


def _load_meta(root: Path):
    names = {p: json.loads((root / "features" / f"feature_names_{p}.json").read_text()) for p in PIPELINES}
    encoding = json.loads((root / "features" / "edit_encoding.json").read_text())
    specs = json.loads((root / "features" / "llm_feature_specs.json").read_text())
    return names, encoding, specs


# ----------------------------------------------------------------------------- fit (author only)
def fit(real_root: Path) -> None:
    import pyarrow.parquet as pq
    real_root = Path(real_root)
    names, encoding, specs = _load_meta(ROOT)
    edit_path = real_root / "features" / "application_edit_inputs.parquet"
    raw_path = real_root / "features" / "raw_inputs.parquet"
    edit_schema = pq.read_schema(edit_path)
    edit_cols = {f.name: ("string" if str(f.type) in ("string", "large_string") else str(f.type)) for f in edit_schema}
    edit = pd.read_parquet(edit_path)
    cols: dict[str, dict] = {}
    for c in edit.columns:
        if c == ID_COL:
            continue
        s = edit[c]
        if c == "DAYS_EMPLOYED":
            m = _numeric_marginal(s.where(s != SENTINEL))
            m["sentinel_rate"] = round(float((s == SENTINEL).mean()), 6)
            m["nan_rate"] = round(float(s.isna().mean()), 6)
            cols[c] = m
        elif edit_cols[c] == "string":
            cols[c] = _string_marginal(s)
        else:
            cols[c] = _numeric_marginal(s)
    ratios = {
        "GOODS_TO_CREDIT": _numeric_marginal(edit["AMT_GOODS_PRICE"] / edit["AMT_CREDIT"]),
        "ANNUITY_TO_CREDIT": _numeric_marginal(edit["AMT_ANNUITY"] / edit["AMT_CREDIT"]),
    }
    del edit
    held_cols: dict[str, dict] = {}
    held_by_pipeline: dict[str, list[str]] = {}
    for p in PIPELINES:
        settable = _settable_columns(p, names[p], edit_cols, encoding, specs)
        held = [c for c in names[p] if c not in settable]
        held_by_pipeline[p] = held
        need = [c for c in held if c not in held_cols]
        if need:
            df = pq.read_table(real_root / "features" / f"features_{p}.parquet", columns=need).to_pandas()
            for c in need:
                held_cols[c] = _numeric_marginal(df[c])
            del df
    payload = {
        "description": "Aggregate per-column marginals of the real Home Credit package, used only to sample "
                       "SYNTHETIC customers. Contains no rows, no SK_ID_CURR values and no TARGET labels.",
        "quantile_levels": [round(float(x), 6) for x in QLEVELS],
        "min_level_count": MIN_LEVEL_COUNT,
        "edit_columns": edit_cols,
        "raw_input_columns": pq.read_schema(raw_path).names,
        "application": cols,
        "ratios": ratios,
        "held_features": held_cols,
        "held_by_pipeline": held_by_pipeline,
    }
    MARGINALS.write_text(json.dumps(payload, indent=1))
    print(f"wrote {MARGINALS.relative_to(ROOT)} ({MARGINALS.stat().st_size/1e3:.0f} kB); "
          f"{len(cols)} application columns, {len(held_cols)} held feature columns")


# ----------------------------------------------------------------------------- generate
def _application_table(M: dict, n: int, rng: np.random.Generator) -> pd.DataFrame:
    app = M["application"]
    ids = np.arange(ID_START, ID_START + n, dtype="int64")
    data: dict[str, object] = {ID_COL: ids}
    for c, dt in M["edit_columns"].items():
        if c == ID_COL:
            continue
        data[c] = _sample(app[c], n, rng)
    df = pd.DataFrame(data)
    # light consistency rules (values stay inside the sampled marginal ranges)
    credit = np.round(df["AMT_CREDIT"].to_numpy() * 2) / 2
    df["AMT_CREDIT"] = credit
    g = _sample(M["ratios"]["GOODS_TO_CREDIT"], n, rng)
    df["AMT_GOODS_PRICE"] = np.where(np.isnan(g), np.nan, np.round(credit * g / 500.0) * 500.0)
    a = _sample(M["ratios"]["ANNUITY_TO_CREDIT"], n, rng)
    df["AMT_ANNUITY"] = np.where(np.isnan(a), np.nan, np.round(credit * a * 2) / 2)
    df["AMT_INCOME_TOTAL"] = np.round(df["AMT_INCOME_TOTAL"].to_numpy() / 4500.0) * 4500.0
    kids = df["CNT_CHILDREN"].fillna(0).to_numpy()
    fam = df["CNT_FAM_MEMBERS"].to_numpy()
    df["CNT_FAM_MEMBERS"] = np.where(np.isnan(fam), fam, np.maximum(fam, kids + 1))
    # employment: sentinel (not employed) with its real rate, otherwise shorter than age - 18y
    emp = df["DAYS_EMPLOYED"].to_numpy().astype(float)
    birth = df["DAYS_BIRTH"].to_numpy().astype(float)
    sentinel = rng.random(n) < float(app["DAYS_EMPLOYED"].get("sentinel_rate", 0.0))
    emp = np.where(np.isnan(emp), emp, np.maximum(emp, birth + 18 * 365))
    emp = np.where(sentinel, SENTINEL, emp)
    df["DAYS_EMPLOYED"] = emp
    if "FLAG_OWN_CAR" in df and "OWN_CAR_AGE" in df:
        df.loc[df["FLAG_OWN_CAR"] == "N", "OWN_CAR_AGE"] = np.nan
    for c, dt in M["edit_columns"].items():
        if c == ID_COL:
            continue
        if dt == "string":
            df[c] = df[c].astype("string")
        else:
            df[c] = df[c].astype("float64")
    return df


def generate(n: int, seed: int) -> None:
    import joblib
    import pyarrow as pa
    import pyarrow.parquet as pq

    M = json.loads(MARGINALS.read_text())
    names, _encoding, _specs = _load_meta(ROOT)
    rng = np.random.default_rng(seed)
    feat = ROOT / "features"; out = ROOT / "outputs"

    edit = _application_table(M, n, rng)
    edit.to_parquet(feat / "application_edit_inputs.parquet", index=False)
    raw_cols = M["raw_input_columns"]
    raw = edit[raw_cols].copy()
    for c in raw_cols[1:]:
        raw[c] = raw[c].astype("float64")
    raw.to_parquet(feat / "raw_inputs.parquet", index=False)

    # held (auxiliary-table) features: one draw per feature name, shared by every pipeline using it
    held_vals = {c: _sample(m, n, rng) for c, m in M["held_features"].items()}

    # 1) placeholder matrices: held columns filled, application columns NaN (rebuilt next)
    for p in PIPELINES:
        cols = {ID_COL: edit[ID_COL].to_numpy()}
        for c in names[p]:
            cols[c] = held_vals[c] if c in held_vals else np.full(n, np.nan)
        X = pd.DataFrame(cols)
        X.to_parquet(feat / f"features_{p}.parquet", index=False)
    # 2) application-level columns from the package's own recompute pipeline (no edits)
    bundle = FR.load_bundle(ROOT)
    ids = edit[ID_COL].tolist()
    mats = {}
    for p in PIPELINES:
        rows = [FR.recompute_row(bundle, p, int(i), {}) for i in ids]
        X = pd.concat(rows, ignore_index=True).astype("float64")
        X = pd.concat([pd.DataFrame({ID_COL: np.asarray(ids, dtype="int64")}), X], axis=1)
        mats[p] = X
    for p, X in mats.items():
        pq.write_table(pa.Table.from_pandas(X, preserve_index=False), feat / f"features_{p}.parquet")
    # 3) identity check: no-edit recompute must equal the stored row for every customer
    bundle = FR.load_bundle(ROOT)
    bad = {p: [] for p in PIPELINES}
    for p in PIPELINES:
        for i in ids:
            st = FR.stored_row(bundle, p, int(i)).to_numpy()
            rc = FR.recompute_row(bundle, p, int(i), {}).to_numpy()
            if not np.array_equal(st, rc, equal_nan=True):
                bad[p].append(int(i))
    if any(bad.values()):
        raise SystemExit(f"recompute identity failed: { {p: len(v) for p, v in bad.items()} }")

    # 4) scores from the UNCHANGED models
    scores = pd.DataFrame({ID_COL: np.asarray(ids, dtype="int64")})
    for p in PIPELINES:
        model = joblib.load(ROOT / "models" / f"model_{p}.pkl")
        X = mats[p][names[p]]
        scores[f"pred_{p}"] = model.predict_proba(X)[:, 1]
    for p in PIPELINES:
        scores[f"oof_{p}"] = scores[f"pred_{p}"]
    pc = scores["pred_C"].to_numpy()
    calibrated = pc / (pc + (1.0 - pc) * SCALE_POS_WEIGHT)
    scores["TARGET"] = (rng.random(n) < calibrated).astype("int8")
    scores["fold"] = np.nan
    scores = scores[["SK_ID_CURR", "TARGET", "fold", "oof_A", "oof_B", "oof_C", "pred_A", "pred_B", "pred_C"]]
    scores.to_parquet(out / "scored_table.parquet", index=False)
    scores.to_csv(out / "scored_table.csv", index=False)

    demo = {
        "description": "SYNTHETIC demo customers generated by scripts/make_synthetic_data.py. "
                       "These are not real Home Credit applicants. Decile cutoffs are NOT computed from this list.",
        "synthetic": True, "seed": seed, "n_customers": n, "id_range": [ids[0], ids[-1]],
        "note": "The selector lists the first 100 IDs; every ID in id_range can be opened with Load ID.",
        "n_ids": min(n, DEMO_LIST_SIZE), "ids": ids[:DEMO_LIST_SIZE],
    }
    (feat / "demo_customer_ids.json").write_text(json.dumps(demo, indent=1))
    check = {
        "synthetic": True, "seed": seed, "n_customers": n, "id_range": [ids[0], ids[-1]],
        "no_edit_recompute_identity": {p: {"rows_checked": n, "mismatches": len(bad[p])} for p in PIPELINES},
        "application_level_columns_from_pipeline": {p: len(names[p]) - len(M["held_by_pipeline"][p]) for p in PIPELINES},
        "auxiliary_columns_sampled_from_marginals": {p: len(M["held_by_pipeline"][p]) for p in PIPELINES},
        "oof_equals_pred": True,
        "synthetic_target_rate": round(float(scores["TARGET"].mean()), 4),
        "mean_pred": {p: round(float(scores[f"pred_{p}"].mean()), 4) for p in PIPELINES},
    }
    (out / "synthetic_data_check.json").write_text(json.dumps(check, indent=1))
    print(json.dumps(check, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fit", help="author only: build features/synthetic_marginals.json from the real package")
    f.add_argument("--real-root", required=True, type=Path)
    g = sub.add_parser("generate", help="generate synthetic customers (default)")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.real_root)
    else:
        generate(getattr(a, "n", 200), getattr(a, "seed", 42))


if __name__ == "__main__":
    main()
