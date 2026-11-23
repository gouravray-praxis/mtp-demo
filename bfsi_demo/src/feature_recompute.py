"""Editable what-if recomputation for the demo app (contract: ARTIFACT_CONTRACT.md §5).

Dependency DAG (application level only):
  raw inputs (features/raw_inputs.parquet, raw units)
    -> mechanical: DAYS_EMPLOYED sentinel 365243->NaN, DAYS_EMPLOYED_ANOM, AGE_YEARS, EMPLOYMENT_YEARS,
       REGISTRATION_YEARS, ID_PUBLISH_YEARS, LAST_PHONE_CHANGE_YEARS     (same expressions as preprocess)
    -> human application ratios FE_*      (features_a.fe_application_ratios == features_bc version)
    -> row-level LLM features (row_level==true in llm_feature_specs.json, run via the AST sandbox)
Everything aggregated from bureau / previous / installments / credit card / POS tables is HELD at the
stored value. Only features present in the pipeline's model input are returned, in model order."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from . import features_a as FA
from . import features_bc as FB
from . import sandbox

EDITABLE = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH",
            "DAYS_EMPLOYED", "DAYS_REGISTRATION", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "OWN_CAR_AGE",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
LABELS = {"AMT_INCOME_TOTAL": ("Annual income", "currency"), "AMT_CREDIT": ("Credit amount", "currency"),
          "AMT_ANNUITY": ("Loan annuity", "currency"), "AMT_GOODS_PRICE": ("Goods price", "currency"),
          "DAYS_BIRTH": ("Age (days, negative)", "days"), "DAYS_EMPLOYED": ("Employment (days, negative; 365243=not employed)", "days"),
          "DAYS_REGISTRATION": ("Registration change (days, negative)", "days"),
          "CNT_CHILDREN": ("Children", "count"), "CNT_FAM_MEMBERS": ("Family members", "count"),
          "OWN_CAR_AGE": ("Car age (years)", "years"), "EXT_SOURCE_1": ("External score 1", "score"),
          "EXT_SOURCE_2": ("External score 2", "score"), "EXT_SOURCE_3": ("External score 3", "score")}
FE_INPUTS = ["DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE",
             "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_FAM_MEMBERS",
             "CNT_CHILDREN", "OWN_CAR_AGE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
             "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
             "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
             "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"] + \
            [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
MECHANICAL = ["DAYS_EMPLOYED_ANOM", "AGE_YEARS", "EMPLOYMENT_YEARS", "REGISTRATION_YEARS",
              "ID_PUBLISH_YEARS", "LAST_PHONE_CHANGE_YEARS"]
FE_NAMES = ["FE_ANNUITY_TO_INCOME", "FE_CREDIT_TO_INCOME", "FE_CREDIT_TO_GOODS", "FE_ANNUITY_TO_CREDIT",
            "FE_INCOME_PER_PERSON", "FE_GOODS_PRICE_GAP", "FE_LOAN_TENOR_MONTHS", "FE_EMPLOYMENT_TO_AGE",
            "FE_WORKING_SINCE_AGE", "FE_DOCS_SUBMITTED", "FE_SOCIAL_CIRCLE_DEFAULT", "FE_ENQUIRY_TOTAL",
            "FE_ENQUIRY_RECENT_RATIO", "FE_EXT_SOURCE_MEAN", "FE_EXT_SOURCE_MIN", "FE_EXT_SOURCE_MAX",
            "FE_EXT_SOURCE_STD", "FE_EXT_SOURCE_PRODUCT", "FE_EXT_MEAN_ANNUITY", "FE_EXT_MEAN_CREDIT"]
FE_DEPS = {"FE_ANNUITY_TO_INCOME": ["AMT_ANNUITY", "AMT_INCOME_TOTAL"], "FE_CREDIT_TO_INCOME": ["AMT_CREDIT", "AMT_INCOME_TOTAL"],
           "FE_CREDIT_TO_GOODS": ["AMT_CREDIT", "AMT_GOODS_PRICE"], "FE_ANNUITY_TO_CREDIT": ["AMT_ANNUITY", "AMT_CREDIT"],
           "FE_INCOME_PER_PERSON": ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"], "FE_GOODS_PRICE_GAP": ["AMT_CREDIT", "AMT_GOODS_PRICE"],
           "FE_LOAN_TENOR_MONTHS": ["AMT_CREDIT", "AMT_ANNUITY"], "FE_EMPLOYMENT_TO_AGE": ["DAYS_EMPLOYED", "DAYS_BIRTH"],
           "FE_WORKING_SINCE_AGE": ["DAYS_EMPLOYED", "DAYS_BIRTH"], "FE_DOCS_SUBMITTED": [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)],
           "FE_SOCIAL_CIRCLE_DEFAULT": ["DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"],
           "FE_ENQUIRY_TOTAL": [c for c in FE_INPUTS if c.startswith("AMT_REQ")],
           "FE_ENQUIRY_RECENT_RATIO": [c for c in FE_INPUTS if c.startswith("AMT_REQ")],
           **{k: ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] for k in
              ["FE_EXT_SOURCE_MEAN", "FE_EXT_SOURCE_MIN", "FE_EXT_SOURCE_MAX", "FE_EXT_SOURCE_STD", "FE_EXT_SOURCE_PRODUCT"]},
           "FE_EXT_MEAN_ANNUITY": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_ANNUITY"],
           "FE_EXT_MEAN_CREDIT": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT"]}
MECH_DEPS = {"DAYS_EMPLOYED_ANOM": ["DAYS_EMPLOYED"], "AGE_YEARS": ["DAYS_BIRTH"], "EMPLOYMENT_YEARS": ["DAYS_EMPLOYED"],
             "REGISTRATION_YEARS": ["DAYS_REGISTRATION"], "ID_PUBLISH_YEARS": ["DAYS_ID_PUBLISH"],
             "LAST_PHONE_CHANGE_YEARS": ["DAYS_LAST_PHONE_CHANGE"]}

def mechanical(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric part of preprocess_application (identical expressions, same order)."""
    df = df.copy()
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
    df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25
    df['REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365.25
    df['ID_PUBLISH_YEARS'] = -df['DAYS_ID_PUBLISH'] / 365.25
    df['LAST_PHONE_CHANGE_YEARS'] = -df['DAYS_LAST_PHONE_CHANGE'] / 365.25
    return df

def derive_app(raw: pd.DataFrame, pipeline: str = "A") -> pd.DataFrame:
    """A uses features_a.fe_application_ratios (original Pipeline A); B/C use features_bc (A' notebook) —
    the formulas are identical (asserted in selftest), the call is explicit per pipeline."""
    fe = FA.fe_application_ratios if pipeline == "A" else FB.fe_application_ratios
    return fe(mechanical(raw))

def raw_input_columns(llm_specs: list[dict]) -> list[str]:
    cols = list(FE_INPUTS)
    for s in llm_specs:
        if s.get("row_level"):
            for c in s.get("app_columns", []):
                if c not in cols:
                    cols.append(c)
    return cols

def build_raw_inputs(app_train_raw: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = app_train_raw[["SK_ID_CURR"] + cols].copy()
    out["SK_ID_CURR"] = out["SK_ID_CURR"].astype("int64")
    for c in cols:
        out[c] = out[c].astype("float64")
    return out

def editable_meta(raw_inputs: pd.DataFrame) -> list[dict]:
    meta = []
    for c in EDITABLE:
        s = raw_inputs[c]
        if c == "DAYS_EMPLOYED":
            s = s[s != 365243]
        lo, hi = (float(np.nanpercentile(s, 0.5)), float(np.nanpercentile(s, 99.5))) if s.notna().any() else (0.0, 1.0)
        step = 0.01 if c.startswith("EXT_SOURCE") else (1.0 if c.startswith(("CNT", "DAYS", "OWN")) else 1000.0)
        meta.append({"name": c, "label": LABELS[c][0], "unit": LABELS[c][1], "min": lo, "max": hi, "step": step})
    return meta

def llm_row_level(code: str, allowed_cols: set, app_columns_all: set) -> tuple[bool, list[str]]:
    """Candidate row-level iff the code only uses app_train and only references application columns that
    can be rebuilt from raw_inputs (allowed_cols). Final status is confirmed empirically in run_all."""
    try:
        tree, used = sandbox.check(code)
    except Exception:
        return False, []
    if used - {"app_train"}:
        return False, []
    consts = sandbox.string_constants(tree)
    cols = sorted(c for c in consts if c in app_columns_all and c != "SK_ID_CURR")
    if not set(cols) <= allowed_cols:
        return False, cols
    return True, cols

def llm_series(code: str, fname: str, app: pd.DataFrame) -> pd.Series:
    """Run LLM code in the sandbox. The frame's (unnamed) index is set to SK_ID_CURR, so code that returns a
    row-aligned Series and code that does .set_index('SK_ID_CURR') both yield an SK_ID_CURR-indexed result."""
    app = app.copy(); app.index = pd.Index(app["SK_ID_CURR"].to_numpy("int64"))
    r = sandbox.run(code, fname, {"app_train": app}, timeout_s=30)
    if not isinstance(r, pd.Series) or not r.index.is_unique:
        raise ValueError(f"{fname}: result is not a uniquely indexed Series")
    return r

def llm_value(code: str, fname: str, app: pd.DataFrame, sk_id: int) -> float:
    v = llm_series(code, fname, app).reindex([sk_id]).values[0]
    return float(v) if v is not None else np.nan

# ---------------------------------------------------------------------------------------- runtime
FORBIDDEN_EDITS = frozenset({"SK_ID_CURR", "TARGET"})

@dataclass
class Bundle:
    root: Path
    feature_names: dict = field(default_factory=dict)
    llm_specs: list = field(default_factory=list)
    editable: list = field(default_factory=list)
    raw_cols: list = field(default_factory=list)
    selftest_ok: bool | None = None
    encoding: dict = field(default_factory=dict)
    edit_table: str = "raw_inputs"
    _ds: dict = field(default_factory=dict)

def load_bundle(root) -> Bundle:
    root = Path(root)
    b = Bundle(root=root)
    for p in ("A", "B", "C"):
        f = root / "features" / f"feature_names_{p}.json"
        if f.exists():
            b.feature_names[p] = json.loads(f.read_text())
    f = root / "features" / "llm_feature_specs.json"
    b.llm_specs = json.loads(f.read_text()) if f.exists() else []
    f = root / "features" / "editable_inputs.json"
    b.editable = json.loads(f.read_text()) if f.exists() else []
    b.raw_cols = raw_input_columns(b.llm_specs)
    enc = root / "features" / "edit_encoding.json"
    b.encoding = json.loads(enc.read_text()) if enc.exists() else {}
    if (root / "features" / "application_edit_inputs.parquet").is_file():
        b.edit_table = "application_edit_inputs"
    f = root / "outputs" / "selftest.json"
    if f.exists():
        b.selftest_ok = bool(json.loads(f.read_text()).get("all_passed"))
    return b

def _row(b: Bundle, name: str, sk_id: int) -> pd.DataFrame:
    if name not in b._ds:
        b._ds[name] = ds.dataset(str(b.root / "features" / f"{name}.parquet"), format="parquet")
    t = b._ds[name].to_table(filter=pc.field("SK_ID_CURR") == int(sk_id))
    if t.num_rows != 1:
        raise KeyError(f"SK_ID_CURR {sk_id} not found in {name}")
    return t.to_pandas()

def stored_row(b: Bundle, pipeline: str, sk_id: int) -> pd.DataFrame:
    return _row(b, f"features_{pipeline}", sk_id)[b.feature_names[pipeline]].astype("float64")

def raw_row(b: Bundle, sk_id: int) -> pd.DataFrame:
    return _row(b, "raw_inputs", sk_id)

def source_row(b: Bundle, sk_id: int) -> pd.DataFrame:
    """Application inputs used for what-if edits (expanded table when present)."""
    return _row(b, getattr(b, "edit_table", "raw_inputs") or "raw_inputs", sk_id)

def allowed_edit_names(b: Bundle) -> set[str]:
    names: set[str] = set(EDITABLE)
    for item in b.editable or []:
        if isinstance(item, dict) and item.get("name"):
            names.add(str(item["name"]))
        elif isinstance(item, str):
            names.add(item)
    for f in (b.encoding or {}).get("fields") or []:
        if isinstance(f, dict) and f.get("name"):
            names.add(str(f["name"]))
    names -= FORBIDDEN_EDITS
    return names

def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False

def _assign_edit(frame: pd.DataFrame, column: str, value) -> None:
    if _is_missing(value):
        frame[column] = np.nan
        return
    # Keep categoricals as strings; coerce numeric-looking values when the column is numeric-like.
    if frame[column].dtype == object or str(frame[column].dtype).startswith("string"):
        frame[column] = str(value)
        return
    frame[column] = float(value)

def apply_application_encodings(row: pd.Series, pipeline: str, names: list[str], encoding: dict) -> dict:
    """Map edited application values to model columns (binary label-encodings and OHE dummies).

    Dummy columns not present in this pipeline's feature list are left untouched.
    A level with no dummy (drop_first / later selection) yields all existing dummies = 0.
    """
    updates: dict[str, float] = {}
    for col, mapping in (encoding.get("binary_maps") or {}).items():
        if col not in names:
            continue
        value = row[col] if col in row.index else np.nan
        if _is_missing(value):
            updates[col] = np.nan
        else:
            updates[col] = float(mapping[str(value)])
    for col, mapping in ((encoding.get("ohe_columns") or {}).get(pipeline) or {}).items():
        value = row[col] if col in row.index else np.nan
        level = "nan" if _is_missing(value) else str(value)
        for lev, dummy in mapping.items():
            if dummy in names:
                updates[dummy] = 1.0 if lev == level else 0.0
    return updates

def recompute_row(b: Bundle, pipeline: str, sk_id: int, edits: dict | None = None) -> pd.DataFrame:
    edits = edits or {}
    forbidden = sorted(set(edits) & FORBIDDEN_EDITS)
    if forbidden:
        raise ValueError(f"non-editable inputs: {forbidden}")
    allowed = allowed_edit_names(b)
    bad = set(edits) - allowed
    if bad:
        raise ValueError(f"non-editable inputs: {sorted(bad)}")
    names = b.feature_names[pipeline]
    X = stored_row(b, pipeline, sk_id).reset_index(drop=True)
    raw = source_row(b, sk_id).reset_index(drop=True)
    for k, v in edits.items():
        if k not in raw.columns:
            raise ValueError(f"non-editable inputs: {k}")
        _assign_edit(raw, k, v)
    der = derive_app(raw, pipeline)
    # mechanical + application ratios + original numeric FE inputs
    for c in [c for c in FE_INPUTS + MECHANICAL + FE_NAMES if c in names and c in der.columns]:
        val = der[c].iloc[0]
        X[c] = float(val) if not _is_missing(val) else np.nan
    # Other raw numeric application fields that are themselves model inputs (housing, flags, ratings, …)
    skipped = set(FE_INPUTS + MECHANICAL + FE_NAMES)
    for c in raw.columns:
        if c in names and c not in skipped and c not in FORBIDDEN_EDITS:
            if str(raw[c].dtype).startswith("string") or raw[c].dtype == object:
                continue  # encoded below
            val = raw[c].iloc[0]
            X[c] = float(val) if not _is_missing(val) else np.nan
    # Binary encodings + one-hot columns for this pipeline (history aggregates stay at stored values)
    if b.encoding:
        for col, val in apply_application_encodings(raw.iloc[0], pipeline, names, b.encoding).items():
            X[col] = val
    # row-level LLM features (pipelines B, C)
    for s in b.llm_specs:
        if s.get("row_level") and s["feature_name"] in names:
            app = der.copy()
            X[s["feature_name"]] = llm_value(s["python_code"], s["feature_name"], app, int(sk_id))
    return X[names].astype("float64")

def changed_features(b: Bundle, pipeline: str, sk_id: int, edits: dict) -> list[str]:
    a = stored_row(b, pipeline, sk_id).iloc[0]; c = recompute_row(b, pipeline, sk_id, edits).iloc[0]
    return [n for n in a.index if not (np.isnan(a[n]) and np.isnan(c[n])) and a[n] != c[n]]
