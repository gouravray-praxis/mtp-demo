"""BFSI Home Credit — Pipelines A / B / C live scoring demo (Final MTP).

Run from the package root with the project venv:

    /workspace/bfsi-mtp/.venv/bin/streamlit run app/streamlit_demo.py

Aligned to ARTIFACT_CONTRACT.md v1.0. The app:

* reads only package artifacts under models/, features/, outputs/ and imports
  src.feature_recompute (it never reads data/raw/);
* loads customer IDs and exported scores once, and reads ONE feature row per
  pipeline with a pyarrow dataset ID filter (never the full ~1 GB matrices);
* joblib-loads the three package models only; unchanged rows display held-out
  (OOF) scores by default, and Recalculate predictions runs live what-if scores
  (exported pred_* remains a no-edit health check, never the default display);
* enables application-field edits only when the shared recompute module is
  present and validated (fail closed otherwise).

Core helpers are plain Python so tests can run without starting the UI.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PIPELINES = ("A", "B", "C")
TITLES = {
    "A": "Pipeline A — Human features",
    "B": "Pipeline B — LLM features (mechanical base only)",
    "C": "Pipeline C — Hybrid (human + LLM)",
}
ID_COL = "SK_ID_CURR"
SCORE_COLS = [
    "SK_ID_CURR", "TARGET", "fold",
    "oof_A", "oof_B", "oof_C",
    "pred_A", "pred_B", "pred_C",
]
PRED_TOL = 1e-6
PREFERRED_DEMO_IDS = [900001, 900002, 900003, 900004, 900005]  # synthetic demo IDs (public repo)
MODE_HELD_OUT = "Held-out (OOF)"
MODE_LIVE = "Live model"
MODE_WHATIF = "Live what-if"
SCORE_MODES = (MODE_HELD_OUT, MODE_LIVE)
RECALCULATE_LABEL = "Recalculate predictions"
DIRTY_HINT = "Inputs changed—click Recalculate predictions"
DAYS_EMPLOYED_SENTINEL = 365243.0
DAYS_PER_YEAR = 365.25
PREFERRED_EDITS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS",
    "EXT_SOURCE_2", "EXT_SOURCE_3",
]
MAX_EDIT_CONTROLS = 9
NULLABLE = {
    "AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_FAM_MEMBERS", "OWN_CAR_AGE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
}
HELD_ORIGINS = frozenset({"human_aux", "llm_aux"})
ORIGIN_LABELS = {
    "raw": "raw application field",
    "ohe": "one-hot (mechanical)",
    "mechanical": "mechanical preprocessing",
    "human_app": "human FE (application)",
    "human_aux": "historical tables — held fixed",
    "llm_app": "LLM FE (application)",
    "llm_aux": "historical tables — held fixed",
}


# --------------------------------------------------------------------------- paths
def default_root() -> Path:
    env = os.environ.get("BFSI_DEMO_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parent.parent


def required_paths(root: Path) -> dict[str, Path]:
    root = Path(root)
    paths = {"scored_table": root / "outputs" / "scored_table.parquet"}
    for p in PIPELINES:
        paths[f"model_{p}"] = root / "models" / f"model_{p}.pkl"
        paths[f"features_{p}"] = root / "features" / f"features_{p}.parquet"
        paths[f"feature_names_{p}"] = root / "features" / f"feature_names_{p}.json"
        paths[f"schema_{p}"] = root / "features" / f"schema_{p}.json"
    return paths


def optional_paths(root: Path) -> dict[str, Path]:
    root = Path(root)
    return {
        "raw_inputs": root / "features" / "raw_inputs.parquet",
        "editable_inputs": root / "features" / "editable_inputs.json",
        "feature_recompute": root / "src" / "feature_recompute.py",
        "selftest": root / "outputs" / "selftest.json",
        "final_results": root / "outputs" / "final_results.json",
        "risk_decile_cutoffs": root / "features" / "risk_decile_cutoffs.json",
        "demo_customer_ids": root / "features" / "demo_customer_ids.json",
        "application_edit_inputs": root / "features" / "application_edit_inputs.parquet",
        "edit_encoding": root / "features" / "edit_encoding.json",
    }


def edit_inputs_path(root: Path) -> Path:
    """Prefer the expanded application-edit table; fall back to packaged raw_inputs."""
    expanded = optional_paths(root)["application_edit_inputs"]
    return expanded if expanded.is_file() else optional_paths(root)["raw_inputs"]


def missing_assets(root: Path) -> list[str]:
    return [f"{k}: {v}" for k, v in required_paths(root).items() if not v.is_file()]


def pipeline_asset_paths(root: Path, pipeline: str) -> dict[str, Path]:
    root = Path(root)
    return {
        f"model_{pipeline}": root / "models" / f"model_{pipeline}.pkl",
        f"features_{pipeline}": root / "features" / f"features_{pipeline}.parquet",
        f"feature_names_{pipeline}": root / "features" / f"feature_names_{pipeline}.json",
        f"schema_{pipeline}": root / "features" / f"schema_{pipeline}.json",
    }


def complete_pipelines(root: Path) -> tuple[str, ...]:
    """Pipelines whose contract model + matrix + names + schema are all present."""
    return tuple(
        p for p in PIPELINES
        if all(path.is_file() for path in pipeline_asset_paths(root, p).values())
    )


# --------------------------------------------------------------------------- loaders
def load_feature_names(root: Path, pipeline: str) -> list[str]:
    payload = json.loads(
        (Path(root) / "features" / f"feature_names_{pipeline}.json").read_text(encoding="utf-8")
    )
    if not isinstance(payload, list) or not payload or not all(isinstance(n, str) and n for n in payload):
        raise ValueError(f"feature_names_{pipeline}.json must be a non-empty JSON array of strings")
    if len(set(payload)) != len(payload):
        raise ValueError(f"feature_names_{pipeline}.json has duplicate names")
    return payload


def load_schema(root: Path, pipeline: str, names: list[str]) -> dict:
    schema = json.loads(
        (Path(root) / "features" / f"schema_{pipeline}.json").read_text(encoding="utf-8")
    )
    feats = schema.get("features")
    if not isinstance(feats, list):
        raise ValueError(f"schema_{pipeline}.json is missing a features list")
    if [f.get("name") for f in feats] != names:
        raise ValueError(
            f"schema_{pipeline}.json feature names/order do not match feature_names_{pipeline}.json"
        )
    if int(schema.get("n_features", len(names))) != len(names):
        raise ValueError(f"schema_{pipeline}.json n_features does not equal {len(names)}")
    return schema


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq
    return list(pq.read_schema(path).names)


def check_parquet_columns(path: Path, names: list[str]) -> None:
    cols = parquet_columns(path)
    expected = [ID_COL] + list(names)
    if cols != expected:
        raise ValueError(
            f"{Path(path).name}: columns must be [{ID_COL}] + feature_names in the same order "
            f"(found {len(cols)} columns, expected {len(expected)})"
        )


def load_model(root: Path, pipeline: str, names: list[str]):
    """joblib-load a LightGBM classifier from this package only."""
    import joblib

    root = Path(root).resolve()
    path = (root / "models" / f"model_{pipeline}.pkl").resolve()
    if root not in path.parents:
        raise ValueError("model path escapes the package root")
    if not path.is_file():
        raise FileNotFoundError(path)
    model = joblib.load(path)
    model_names = list(getattr(model, "feature_name_", []) or [])
    if model_names != names:
        raise ValueError(
            f"model_{pipeline}.pkl feature_name_ ({len(model_names)}) does not match "
            f"feature_names_{pipeline}.json ({len(names)}) in count/order"
        )
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"model_{pipeline}.pkl has no predict_proba")
    return model


def load_scores(root: Path) -> pd.DataFrame:
    """Load the official contract scored table (all SCORE_COLS required)."""
    import pyarrow.parquet as pq

    path = Path(root) / "outputs" / "scored_table.parquet"
    available = list(pq.read_schema(path).names)
    missing = [c for c in SCORE_COLS if c not in available]
    if missing:
        raise ValueError(f"scored_table.parquet missing columns: {missing}")
    df = pq.read_table(path, columns=SCORE_COLS).to_pandas()
    if df[ID_COL].duplicated().any():
        raise ValueError("scored_table.parquet has duplicate SK_ID_CURR")
    return df.set_index(ID_COL, drop=False)


def read_row(path: Path, sk_id: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Single-customer read via pyarrow dataset filter (row-group pruning; no full load)."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    table = ds.dataset(str(path), format="parquet").to_table(
        columns=columns,
        filter=pc.field(ID_COL) == int(sk_id),
    )
    return table.to_pandas()


def as_model_input(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ValueError(f"row is missing {len(missing)} model features, e.g. {missing[:3]}")
    return df[list(names)].astype("float64").reset_index(drop=True)


def feature_row(root: Path, pipeline: str, sk_id: int, names: list[str]) -> pd.DataFrame:
    df = read_row(Path(root) / "features" / f"features_{pipeline}.parquet", int(sk_id))
    if len(df) != 1:
        raise KeyError(f"SK_ID_CURR {sk_id} found {len(df)} times in features_{pipeline}.parquet")
    return as_model_input(df, names)


def predict_proba_row(model: Any, X: pd.DataFrame) -> float:
    return float(model.predict_proba(X)[:, 1][0])


# --------------------------------------------------------------------------- days semantics (display only; not feature recompute)
def years_to_days(years: float) -> float:
    return -float(round(float(years) * DAYS_PER_YEAR))


def days_to_years(days: float) -> float:
    return -float(days) / DAYS_PER_YEAR


def age_years_bounds(spec: dict, current_years: float | None = None) -> tuple[float, float]:
    """Map raw DAYS_BIRTH percentile bounds (negative days) to age-in-years bounds.

    Always expand to include the customer's current age so a tail value (outside
    the 0.5/99.5 percentile metadata) is not silently clamped into a new DAYS_BIRTH.
    """
    lo, hi = spec.get("min"), spec.get("max")
    amin, amax = 18.0, 80.0
    if hi is not None:
        amin = max(1.0, math.floor(days_to_years(float(hi)) * 10) / 10)
    if lo is not None:
        amax = min(100.0, math.ceil(days_to_years(float(lo)) * 10) / 10)
    if current_years is not None and not pd.isna(current_years):
        y = float(current_years)
        amin = min(amin, math.floor(y * 10) / 10)
        amax = max(amax, math.ceil(y * 10) / 10)
    if amin > amax:
        return 18.0, 80.0
    return amin, amax


# --------------------------------------------------------------------------- recompute adapter (fail closed)
@dataclass
class RecomputeStatus:
    enabled: bool
    reason: str
    bundle: Any = None
    module: Any = None
    editable: list = field(default_factory=list)


def load_editable_inputs(root: Path) -> list[dict]:
    """Return every packaged editable application field (no 9-field cap). TARGET and ID are excluded."""
    items = json.loads((Path(root) / "features" / "editable_inputs.json").read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("editable_inputs.json must be a list")
    out = []
    seen = set()
    for item in items:
        if not isinstance(item, dict) or "name" not in item:
            continue
        name = str(item["name"])
        if name in {"SK_ID_CURR", "TARGET"} or name in seen:
            continue
        seen.add(name)
        out.append(item)
    return out


def _import_feature_recompute(root: Path):
    """Import src.feature_recompute from this package root, not a previously imported copy."""
    root = Path(root).resolve()
    path = (root / "src" / "feature_recompute.py").resolve()
    existing = sys.modules.get("src.feature_recompute")
    have = Path(getattr(existing, "__file__", "")).resolve() if existing is not None else None
    if existing is not None and have != path:
        sys.modules.pop("src.feature_recompute", None)
        pkg = sys.modules.get("src")
        pkg_paths = [str(Path(p).resolve()) for p in list(getattr(pkg, "__path__", []) or [])] if pkg else []
        if pkg is not None and str(root / "src") not in pkg_paths:
            sys.modules.pop("src", None)
    # This root must be first: leftover tmp roots on sys.path would otherwise win.
    root_s = str(root)
    if sys.path[:1] != [root_s]:
        sys.path = [p for p in sys.path if p != root_s]
        sys.path.insert(0, root_s)
    return importlib.import_module("src.feature_recompute")


def load_recompute(root: Path, names: dict[str, list[str]], probe_id: int | None) -> RecomputeStatus:
    """Enable edits only when the shared module is present and validated."""
    root = Path(root).resolve()
    opt = optional_paths(root)
    for key in ("feature_recompute", "raw_inputs", "editable_inputs"):
        if not opt[key].is_file():
            return RecomputeStatus(False, f"Editing disabled: {opt[key].relative_to(root)} is not available.")
    try:
        editable = load_editable_inputs(root)
        if not editable:
            return RecomputeStatus(False, "Editing disabled: editable_inputs.json lists no fields.")
        mod = _import_feature_recompute(root)
        for fn in ("load_bundle", "recompute_row"):
            if not callable(getattr(mod, fn, None)):
                return RecomputeStatus(False, f"Editing disabled: src.feature_recompute.{fn} is missing.")
        if not opt["selftest"].is_file():
            return RecomputeStatus(False, "Editing disabled: outputs/selftest.json is not available.")
        bundle = mod.load_bundle(root)
        if getattr(bundle, "selftest_ok", None) is not True:
            return RecomputeStatus(
                False,
                "Editing disabled: shared recompute self-test did not pass (bundle.selftest_ok is not True).",
            )
        if probe_id is not None:
            for pipeline in names:
                X = mod.recompute_row(bundle, pipeline=pipeline, sk_id=int(probe_id), edits={})
                if list(X.columns) != names[pipeline] or len(X) != 1:
                    return RecomputeStatus(
                        False,
                        f"Editing disabled: recompute output for pipeline {pipeline} does not match feature_names order.",
                    )
                stored = feature_row(root, pipeline, int(probe_id), names[pipeline])
                if not np.array_equal(
                    np.asarray(X.astype("float64")), np.asarray(stored), equal_nan=True
                ):
                    return RecomputeStatus(
                        False,
                        f"Editing disabled: no-edit recompute for pipeline {pipeline} differs from the stored row.",
                    )
        return RecomputeStatus(True, "Shared recompute module loaded and validated.", bundle, mod, editable)
    except Exception as exc:  # noqa: BLE001 — fail closed with a visible reason
        return RecomputeStatus(
            False,
            f"Editing disabled: recompute module failed to load ({type(exc).__name__}: {exc}).",
        )


def recompute_features(
    status: RecomputeStatus, pipeline: str, sk_id: int, edits: dict, names: list[str]
) -> pd.DataFrame:
    X = status.module.recompute_row(
        status.bundle, pipeline=pipeline, sk_id=int(sk_id), edits=dict(edits)
    )
    if list(X.columns) != names or len(X) != 1:
        raise ValueError(f"recompute output for {pipeline} does not match feature_names")
    return X.astype("float64").reset_index(drop=True)


def _is_missing_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_edits(proposed: dict, current: dict) -> dict:
    """Keep only fields whose value actually differs from the stored inputs (NaN-aware, numeric or text)."""
    out: dict = {}
    for key, value in proposed.items():
        cur = current.get(key, np.nan)
        if _is_missing_value(value) and _is_missing_value(cur):
            continue
        same = False
        if not _is_missing_value(value) and not _is_missing_value(cur):
            try:
                same = float(value) == float(cur)
            except (TypeError, ValueError):
                same = str(value) == str(cur)
        if not same:
            if _is_missing_value(value):
                out[key] = float("nan")
            else:
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    out[key] = value
    return out



# --------------------------------------------------------------------------- demo cohort, OOF deciles, score display
def _ensure_root_on_path(root: Path) -> None:
    root_s = str(Path(root).resolve())
    if sys.path[:1] != [root_s]:
        sys.path = [p for p in sys.path if p != root_s]
        sys.path.insert(0, root_s)


def _import_risk_deciles(root: Path):
    """Load src/risk_deciles.py from this package root by file path (avoids a leftover src package)."""
    path = (Path(root).resolve() / "src" / "risk_deciles.py").resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    key = "bfsi_risk_deciles"
    existing = sys.modules.get(key)
    have = Path(getattr(existing, "__file__", "") or "").resolve() if existing is not None else None
    if existing is not None and have == path:
        return existing
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load risk_deciles from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def load_demo_ids(root: Path, scores: pd.DataFrame) -> list[int]:
    """Selector IDs only. Never used to compute decile cutoffs."""
    valid = set(int(v) for v in scores[ID_COL].tolist())
    path = Path(root) / "features" / "demo_customer_ids.json"
    loaded: list[int] = []
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded = [int(i) for i in (payload.get("ids") or []) if int(i) in valid]
    if not loaded:
        loaded = [int(v) for v in scores[ID_COL].tolist()]
    out: list[int] = []
    seen: set[int] = set()
    for sk in list(PREFERRED_DEMO_IDS) + loaded:
        if sk in valid and sk not in seen:
            out.append(sk)
            seen.add(sk)
    return out


EDIT_WIDGET_PREFIX = "ed_"
SELECT_KEY = "sk_id_select"
LOOKUP_KEY = "sk_id_lookup"


def customer_options(
    demo_ids: list[int],
    current_id: int,
    valid_ids,
    keep_id=None,
) -> list[int]:
    """Demo cohort, plus current/keep IDs when they are valid off-list lookups.

    keep_id is the current dropdown value. It must remain in the options list so
    Streamlit does not reset the selectbox (and fire on_change) when the user
    types an invalid or other off-list ID.
    """
    options = [int(i) for i in demo_ids]
    seen = set(options)
    extras: list[int] = []
    for raw in (current_id, keep_id):
        if raw is None:
            continue
        try:
            vid = int(raw)
        except (TypeError, ValueError):
            continue
        if vid not in seen and vid in valid_ids:
            extras.append(vid)
            seen.add(vid)
    return extras + options


def clear_customer_edit_state(session) -> None:
    """Drop staged edits and live what-if results when the customer changes."""
    session.pop("applied_bundle", None)
    for key in [k for k in list(session.keys()) if str(k).startswith(EDIT_WIDGET_PREFIX)]:
        session.pop(key, None)


def apply_customer_id(
    session,
    new_id: int,
    *,
    options: list[int] | None = None,
    source: str = "",
    sync_widgets: bool = True,
) -> bool:
    """Switch the selected customer. Returns True if the ID changed.

    source is 'select', 'lookup', or 'button'. The other widget is synced via a
    one-shot flag so a stale lookup cannot overwrite a new dropdown value (and
    vice versa) on the following rerun. When called after widgets are already
    instantiated, pass sync_widgets=False and only set the flags; the next run
    applies the widget values before the widgets are created.
    """
    new_id = int(new_id)
    old = int(session["sk_id"]) if "sk_id" in session else None
    session["sk_id"] = new_id
    if source != "lookup":
        session["_sync_lookup"] = True
    if source != "select":
        session["_sync_select"] = True
    if sync_widgets:
        if options is not None and new_id in options:
            session[SELECT_KEY] = new_id
            session.pop("_sync_select", None)
        if source != "lookup":
            session[LOOKUP_KEY] = new_id
            session.pop("_sync_lookup", None)
    changed = old is None or old != new_id
    if changed:
        clear_customer_edit_state(session)
    return changed


def load_reference_cutoffs(root: Path, scores: pd.DataFrame) -> dict[str, list[float]]:
    """Fixed OOF q10..q90 cutoffs. Packaged JSON if present, else once from this table's oof_*."""
    rd = _import_risk_deciles(root)
    path = Path(root) / "features" / "risk_decile_cutoffs.json"
    if path.is_file():
        return rd.load_cutoffs(root)["cutoffs"]
    return rd.cutoffs_from_scores(scores)


def fmt_probability(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def assign_risk_decile(score, cutoffs, root: Path | None = None) -> int | None:
    rd = _import_risk_deciles(Path(root) if root is not None else default_root())
    return rd.assign_decile(score, cutoffs)


def format_risk_decile(decile: int | None, root: Path | None = None) -> str:
    rd = _import_risk_deciles(Path(root) if root is not None else default_root())
    return rd.format_decile(decile)


def score_edits_equal(a: dict, b: dict) -> bool:
    keys = set(a) | set(b)
    if not keys:
        return True
    for key in keys:
        if key not in a or key not in b:
            return False
        va, vb = a[key], b[key]
        if pd.isna(va) and pd.isna(vb):
            continue
        try:
            if float(va) != float(vb):
                return False
        except (TypeError, ValueError):
            if va != vb:
                return False
    return True


def displayed_prediction(
    *,
    applied_edits: dict | None,
    oof: float | None,
    whatif: float | None,
    mode: str | None = None,  # unused; kept so older tests can still call it
    live: float | None = None,
) -> tuple[float | None, str]:
    """Original customer: held-out (OOF). After Recalculate: live what-if. No source toggle."""
    if applied_edits:
        return whatif, MODE_WHATIF
    return oof, MODE_HELD_OUT


# --------------------------------------------------------------------------- display tables
def origin_table(
    pipeline: str,
    schema: dict,
    X: pd.DataFrame,
    stored: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for index, feat in enumerate(schema["features"]):
        name = feat["name"]
        origin = feat.get("origin", "")
        value = X.iloc[0][name]
        row = {
            "#": index + 1,
            "pipeline": pipeline,
            "feature": name,
            "value": value,
            "origin": ORIGIN_LABELS.get(origin, origin or "unknown"),
            "depends_on": ", ".join(feat.get("depends_on") or []),
            "status": (
                "held fixed"
                if origin in HELD_ORIGINS
                else ("recomputed on edit" if feat.get("editable_via_recompute") else "stored")
            ),
        }
        if stored is not None:
            base = stored.iloc[0][name]
            row["stored_value"] = base
            row["changed"] = bool(not ((pd.isna(base) and pd.isna(value)) or base == value))
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_score(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.6f}"


def scores_match(live: dict[str, float], meta: pd.Series, tol: float = PRED_TOL) -> list[str]:
    out = []
    for p in live:
        col = f"pred_{p}"
        if col not in meta.index or pd.isna(meta[col]):
            continue
        if abs(live[p] - float(meta[col])) > tol:
            out.append(p)
    return out


def filter_feature_table(table: pd.DataFrame, query: str, only_changed: bool) -> pd.DataFrame:
    out = table
    if query:
        needle = query.lower()
        mask = (
            out["feature"].astype(str).str.lower().str.contains(needle, regex=False)
            | out["origin"].astype(str).str.lower().str.contains(needle, regex=False)
            | out["depends_on"].astype(str).str.lower().str.contains(needle, regex=False)
        )
        out = out[mask]
    if only_changed and "changed" in out.columns:
        out = out[out["changed"]]
    return out


# --------------------------------------------------------------------------- validation used by UI and tests
@dataclass
class PreparedApp:
    ok: bool
    missing: list[str]
    error: str | None
    names: dict | None = None
    schemas: dict | None = None
    models: dict | None = None
    scores: pd.DataFrame | None = None
    available: tuple[str, ...] = ()
    score_source: str = ""


def prepare_app(root: Path) -> PreparedApp:
    """Require the full contract set (A, B, C models/matrices/names/schemas + scored_table).

    Missing files produce a clear error and no scores — never a cache fallback or invented pred_*.
    """
    missing = missing_assets(root)
    if missing:
        return PreparedApp(
            False,
            missing,
            "Required artifacts are missing; no scores are shown.",
        )
    try:
        names = {p: load_feature_names(root, p) for p in PIPELINES}
        schemas = {p: load_schema(root, p, names[p]) for p in PIPELINES}
        for p in PIPELINES:
            check_parquet_columns(Path(root) / "features" / f"features_{p}.parquet", names[p])
        models = {p: load_model(root, p, names[p]) for p in PIPELINES}
        scores = load_scores(root)
    except Exception as exc:  # noqa: BLE001
        return PreparedApp(False, missing, f"{type(exc).__name__}: {exc}")
    return PreparedApp(
        True, [], None, names, schemas, models, scores, PIPELINES, "outputs/scored_table.parquet"
    )


# --------------------------------------------------------------------------- Streamlit UI
def main() -> None:  # pragma: no cover - widgets exercised by AppTest
    import streamlit as st

    st.set_page_config(page_title="BFSI Home Credit — A/B/C default probability", layout="wide")
    st.title("Home Credit Default Risk — Pipelines A / B / C")
    st.caption(
        "Final MTP engineering demonstration — Gourav Ray (M25AI2085), "
        "Executive M.Tech AI, IIT Jodhpur."
    )
    st.caption("Uncalibrated model estimates; higher means greater default risk.")

    root = default_root()
    rd = _import_risk_deciles(root)
    assign_decile = rd.assign_decile
    format_decile = rd.format_decile

    @st.cache_resource(show_spinner="Loading models and feature metadata …")
    def _cached_prepare(root_s: str) -> PreparedApp:
        return prepare_app(Path(root_s))

    @st.cache_resource(show_spinner="Validating shared recompute module …")
    def _cached_recompute(root_s: str, probe: int, names: dict) -> RecomputeStatus:
        return load_recompute(Path(root_s), names, probe)

    prepared = _cached_prepare(str(root))
    if not prepared.ok:
        st.error(prepared.error or "Required artifacts are missing, so no scores are shown.")
        if prepared.missing:
            st.write("Missing files:")
            st.code("\n".join(prepared.missing))
        st.stop()

    names, schemas, models, scores = prepared.names, prepared.schemas, prepared.models, prepared.scores
    available = prepared.available
    cutoffs = load_reference_cutoffs(root, scores)
    demo_ids = load_demo_ids(root, scores)
    status = _cached_recompute(str(root), int(demo_ids[0] if demo_ids else scores[ID_COL].iloc[0]), names)

    with st.sidebar:
        st.header("Customer")
        if "sk_id" not in st.session_state:
            st.session_state.sk_id = int(demo_ids[0] if demo_ids else scores[ID_COL].iloc[0])
        cur = int(st.session_state.sk_id)
        options = customer_options(
            demo_ids, cur, scores.index, keep_id=st.session_state.get(SELECT_KEY)
        )
        # Initialise widget keys once. Never copy the previous ID over a newly
        # chosen widget value — that is what made both selectors appear broken.
        if SELECT_KEY not in st.session_state and cur in options:
            st.session_state[SELECT_KEY] = cur
        if LOOKUP_KEY not in st.session_state:
            st.session_state[LOOKUP_KEY] = int(cur)
        if st.session_state.pop("_sync_select", False) and cur in options:
            st.session_state[SELECT_KEY] = cur
        if st.session_state.pop("_sync_lookup", False):
            st.session_state[LOOKUP_KEY] = int(cur)

        def _on_select_customer() -> None:
            apply_customer_id(st.session_state, int(st.session_state[SELECT_KEY]), options=options, source="select")

        def _on_lookup_customer() -> None:
            apply_customer_id(st.session_state, int(st.session_state[LOOKUP_KEY]), options=options, source="lookup")

        prev_select = int(st.session_state.get("_prev_select", st.session_state[SELECT_KEY] if SELECT_KEY in st.session_state else cur))
        picked = st.selectbox(
            "Select a customer",
            options=options,
            format_func=lambda x: str(int(x)),
            key=SELECT_KEY,
            on_change=_on_select_customer,
        )
        st.session_state["_prev_select"] = int(picked)
        # Honor a real dropdown change only. Do not treat "dropdown still shows
        # the last valid ID" as a change — that is expected after an off-list
        # or invalid manual lookup, and overwriting sk_id would hide the error.
        if int(picked) != int(prev_select):
            apply_customer_id(
                st.session_state, int(picked), options=options, source="select", sync_widgets=False
            )
            st.rerun()
        c1, c2 = st.columns(2)
        if c1.button("Random customer"):
            apply_customer_id(
                st.session_state,
                int(np.random.default_rng().choice(options if options else scores[ID_COL].to_numpy())),
                options=options,
                source="button",
                sync_widgets=False,
            )
            st.rerun()
        if c2.button("Random defaulter"):
            pool = (
                scores.loc[scores[ID_COL].isin(options) & (scores["TARGET"] == 1), ID_COL].to_numpy()
                if "TARGET" in scores.columns else []
            )
            if len(pool) == 0:
                st.warning("No defaulter is available in this customer list.")
            else:
                apply_customer_id(
                    st.session_state,
                    int(np.random.default_rng().choice(pool)),
                    options=options,
                    source="button",
                    sync_widgets=False,
                )
                st.rerun()
        lookup = st.number_input(
            "Or enter another ID",
            step=1,
            format="%d",
            key=LOOKUP_KEY,
            on_change=_on_lookup_customer,
            help="Type a customer ID and press Enter, or click Load ID.",
        )
        load = st.button("Load ID")
        if int(lookup) != int(st.session_state.sk_id) or (
            load and int(lookup) != int(st.session_state.sk_id)
        ):
            apply_customer_id(
                st.session_state, int(lookup), options=options, source="lookup", sync_widgets=False
            )
            st.rerun()
        if load and int(lookup) == int(st.session_state.sk_id):
            # Value already applied (e.g. on_change); nothing more to do this run.
            pass
        sk_id = int(st.session_state.sk_id)
        if sk_id not in scores.index:
            st.error(f"SK_ID_CURR {sk_id} is not in the score table.")
            st.stop()
        meta = scores.loc[sk_id]
        with st.expander("Actual outcome (optional)", expanded=False):
            if "TARGET" in scores.columns and pd.notna(meta.TARGET):
                st.write(f"TARGET (actual default label): {int(meta.TARGET)}")
            else:
                st.write("TARGET is not available.")
            if "fold" in scores.columns and pd.notna(meta.fold):
                st.write(f"Validation fold: {int(meta.fold)}")
            st.caption("This label is context only and is never passed into the models.")

    sk_id = int(st.session_state.sk_id)
    meta = scores.loc[sk_id]
    stored_rows = {p: feature_row(root, p, sk_id, names[p]) for p in available}
    # Cheap health check: live final-model scores on the stored row must match pred_*.
    live_stored = {p: predict_proba_row(models[p], stored_rows[p]) for p in available}
    mismatched = scores_match(live_stored, meta)
    if mismatched:
        st.error(
            f"Live score differs from exported pred_* for {mismatched} "
            f"(tolerance {PRED_TOL}). Check that the models and feature matrices match."
        )

    applied = st.session_state.get("applied_bundle")
    if not isinstance(applied, dict) or int(applied.get("sk_id", -1)) != sk_id:
        applied = None
        st.session_state.pop("applied_bundle", None)

    left, right = st.columns([1.35, 1.0], gap="large")
    proposed: dict = {}
    submitted = False
    with right:
        st.subheader("Edit application inputs")
        if not status.enabled:
            st.info(status.reason)
        else:
            raw = read_row(edit_inputs_path(root), sk_id)
            if len(raw) != 1:
                st.info("Editing is unavailable for this customer: application inputs were not found.")
            else:
                current = raw.iloc[0].to_dict()
                st.caption(
                    f"{len(status.editable)} application fields. "
                    "Derived ratios and encodings update automatically. "
                    "Historical-table features stay fixed. "
                    "Click Recalculate predictions to score."
                )
                with st.form("application_edits", clear_on_submit=False):
                    groups: dict[str, list] = {}
                    for spec in status.editable:
                        groups.setdefault(spec.get("group") or "Application inputs", []).append(spec)
                    for group_name, specs in groups.items():
                        with st.expander(f"{group_name} ({len(specs)})", expanded=True):
                            columns = st.columns(2)
                            for i, spec in enumerate(specs):
                                with columns[i % 2]:
                                    proposed[spec["name"]] = _edit_control(
                                        st, spec, current.get(spec["name"], np.nan), sk_id
                                    )
                    submitted = st.form_submit_button(RECALCULATE_LABEL, type="primary")
                if st.button("Reset edits"):
                    for key in [k for k in st.session_state if str(k).startswith(f"ed_{sk_id}_")]:
                        del st.session_state[key]
                    st.session_state.pop("applied_bundle", None)
                    st.rerun()
                draft_edits = clean_edits(proposed, current) if proposed else {}
                if submitted:
                    if draft_edits:
                        try:
                            rows = {
                                p: recompute_features(status, p, sk_id, draft_edits, names[p])
                                for p in available
                            }
                            whatif = {p: predict_proba_row(models[p], rows[p]) for p in available}
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Recompute failed; previous scores were kept: {exc}")
                        else:
                            st.session_state.applied_bundle = {
                                "sk_id": sk_id,
                                "edits": dict(draft_edits),
                                "scores": whatif,
                                "rows": rows,
                            }
                            applied = st.session_state.applied_bundle
                    else:
                        st.session_state.pop("applied_bundle", None)
                        applied = None
                dirty = bool(applied) and not score_edits_equal(draft_edits, applied.get("edits") or {})
                if (not applied) and draft_edits and not submitted:
                    dirty = True
                if dirty:
                    st.warning(DIRTY_HINT)
                    last = applied.get("edits") if applied else {}
                    st.caption(
                        "Showing the last calculated inputs "
                        + (", ".join(f"{k}={v}" for k, v in (last or {}).items()) or "(original values)")
                        + ". Draft changes have not been scored."
                    )
                elif applied:
                    st.caption(
                        "Calculated with: " + ", ".join(f"{k}={v}" for k, v in applied["edits"].items())
                    )

    applied_edits = (applied or {}).get("edits") or None
    display_rows = (applied or {}).get("rows") or stored_rows

    with left:
        st.subheader("Predicted default probability")
        st.caption("Original rows use held-out scores; edits use the saved final models.")
        oofs = {
            p: (float(meta[f"oof_{p}"]) if f"oof_{p}" in meta.index and pd.notna(meta[f"oof_{p}"]) else None)
            for p in PIPELINES
        }
        whatifs = (applied or {}).get("scores") or {}
        cols = st.columns(3)
        for col, p in zip(cols, PIPELINES):
            with col, st.container(border=True):
                st.markdown(f"**{TITLES[p]}**")
                if p not in available:
                    st.caption("This pipeline is not available.")
                    continue
                value, state = displayed_prediction(
                    applied_edits=applied_edits,
                    oof=oofs[p],
                    whatif=whatifs.get(p),
                )
                st.metric("Predicted default probability", fmt_probability(value))
                decile = assign_decile(value, cutoffs[p]) if p in cutoffs else None
                st.caption(f"Risk decile {format_decile(decile)} · {state}")
        st.caption("D1 highest risk · D10 lowest risk")
        with st.expander("Reference decile cutoffs (OOF)", expanded=False):
            st.caption(
                "Fixed from held-out scores on the official score table. "
                "The same cutoffs are used for held-out and what-if scores and are not refit on edits."
            )
            for p in available:
                if p not in cutoffs:
                    continue
                edges = ", ".join(f"q{10*(i+1)}={c:.4f}" for i, c in enumerate(cutoffs[p]))
                st.write(f"Pipeline {p}: {edges}")

    st.subheader("Model inputs")
    query = st.text_input("Search features (name, origin, or dependency)", "")
    only_changed = st.checkbox(
        "Show only features that changed after the last calculation",
        value=False,
        disabled=not bool(applied_edits),
    )
    tables = {
        p: origin_table(p, schemas[p], display_rows[p], stored_rows[p] if applied_edits else None)
        for p in available
    }
    tabs = st.tabs([f"Pipeline {p}" for p in PIPELINES])
    for tab, p in zip(tabs, PIPELINES):
        with tab:
            if p not in available:
                st.caption("Feature values are not available for this pipeline.")
                continue
            view = filter_feature_table(tables[p], query, only_changed)
            st.dataframe(
                view,
                hide_index=True,
                width="stretch",
                height=480,
                column_config={"value": st.column_config.NumberColumn(format="%.6g")},
            )
    combined = pd.concat(tables.values(), ignore_index=True) if tables else pd.DataFrame()
    combined.insert(0, ID_COL, sk_id)
    st.download_button(
        "Download all feature values (A, B and C) as CSV",
        combined.to_csv(index=False).encode("utf-8"),
        file_name=f"features_{sk_id}{'_edited' if applied_edits else ''}.csv",
        mime="text/csv",
    )


def _edit_control(st, spec: dict, value, sk_id: int):  # pragma: no cover - Streamlit widgets
    name = spec["name"]
    label = spec.get("label", name)
    key = f"ed_{sk_id}_{name}"
    lo, hi, step = spec.get("min"), spec.get("max"), spec.get("step")
    kind = spec.get("kind") or "numeric"
    nullable = bool(spec.get("nullable")) or name in NULLABLE or _is_missing_value(value)

    if kind in {"categorical", "binary"}:
        options = [str(x) for x in (spec.get("options") or [])]
        current = "(missing)" if _is_missing_value(value) else str(value)
        choices = ["(missing)"] + [o for o in options if o != "(missing)"]
        if current not in choices:
            choices.append(current)
        picked = st.selectbox(label, choices, index=choices.index(current), key=key)
        return np.nan if picked == "(missing)" else picked

    if kind == "flag":
        if nullable and st.checkbox(f"{label}: missing", value=_is_missing_value(value), key=key + "_nan"):
            return np.nan
        checked = False if _is_missing_value(value) else bool(int(float(value)))
        return 1.0 if st.checkbox(label, value=checked, key=key) else 0.0

    if name == "DAYS_EMPLOYED":
        is_sentinel = (not _is_missing_value(value)) and float(value) == DAYS_EMPLOYED_SENTINEL
        if st.checkbox("Not employed / pensioner (DAYS_EMPLOYED = 365243)", value=is_sentinel, key=key + "_sent"):
            return DAYS_EMPLOYED_SENTINEL
        years = 0.0 if _is_missing_value(value) or is_sentinel else days_to_years(float(value))
        emp_max = max(50.0, float(math.ceil(years * 10) / 10))
        entered = st.number_input(
            f"{label} — years employed",
            value=float(round(years, 1)),
            min_value=0.0,
            max_value=emp_max,
            step=0.5,
            key=key,
        )
        if (not _is_missing_value(value)) and (not is_sentinel) and abs(entered - round(years, 1)) < 1e-9:
            return float(value)
        return years_to_days(entered)

    if name.startswith("DAYS_"):
        if nullable and st.checkbox(f"{label}: missing", value=_is_missing_value(value), key=key + "_nan"):
            return np.nan
        years = 40.0 if _is_missing_value(value) else days_to_years(float(value))
        amin, amax = age_years_bounds(spec, None if _is_missing_value(value) else years)
        if name != "DAYS_BIRTH":
            amin, amax = 0.0, max(80.0, float(math.ceil(years * 10) / 10))
        entered = st.number_input(
            f"{label} (years)",
            value=float(round(years, 1)),
            min_value=float(amin),
            max_value=float(amax),
            step=0.5,
            key=key,
        )
        if (not _is_missing_value(value)) and abs(entered - round(years, 1)) < 1e-9:
            return float(value)
        return years_to_days(entered)

    if nullable and st.checkbox(f"{label}: missing", value=_is_missing_value(value), key=key + "_nan"):
        return np.nan

    start = float(value) if not _is_missing_value(value) else float(lo if lo is not None else 0.0)
    kwargs: dict = {"value": start, "key": key}
    if lo is not None:
        kwargs["min_value"] = float(min(lo, start))
    if hi is not None:
        kwargs["max_value"] = float(max(hi, start))
    if step is not None:
        kwargs["step"] = float(step)
    return float(st.number_input(label, **kwargs))
if __name__ == "__main__":
    main()
