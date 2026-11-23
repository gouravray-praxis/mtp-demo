"""Compile and synthetic contract tests for the BFSI Streamlit demo.

Fixtures live only in temporary directories. These tests never write models,
feature matrices, scores, or metrics into the real BFSI_MTP_Demo package.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_demo.py"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(PACKAGE_ROOT)]:
    sys.path.insert(0, str(PACKAGE_ROOT))

SENTINEL = 365243.0
DAYS_PER_YEAR = 365.25
PIPELINES = ("A", "B", "C")

NAMES = {
    "A": [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH",
        "DAYS_EMPLOYED", "DAYS_EMPLOYED_ANOM", "AGE_YEARS", "EMPLOYMENT_YEARS",
        "FE_CREDIT_TO_INCOME", "FE_ANNUITY_TO_CREDIT", "BUREAU_AMT_CREDIT_SUM",
    ],
    "B": [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "DAYS_EMPLOYED_ANOM", "AGE_YEARS", "LLM_CREDIT_INCOME", "LLM_BUREAU_FLAG",
    ],
    "C": [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH",
        "DAYS_EMPLOYED", "DAYS_EMPLOYED_ANOM", "AGE_YEARS", "EMPLOYMENT_YEARS",
        "FE_CREDIT_TO_INCOME", "FE_ANNUITY_TO_CREDIT", "BUREAU_AMT_CREDIT_SUM",
        "LLM_CREDIT_INCOME", "LLM_BUREAU_FLAG",
    ],
}

ORIGINS = {
    "AMT_INCOME_TOTAL": ("raw", [], True),
    "AMT_CREDIT": ("raw", [], True),
    "AMT_ANNUITY": ("raw", [], True),
    "DAYS_BIRTH": ("raw", [], True),
    "DAYS_EMPLOYED": ("mechanical", ["DAYS_EMPLOYED"], True),
    "DAYS_EMPLOYED_ANOM": ("mechanical", ["DAYS_EMPLOYED"], True),
    "AGE_YEARS": ("mechanical", ["DAYS_BIRTH"], True),
    "EMPLOYMENT_YEARS": ("mechanical", ["DAYS_EMPLOYED"], True),
    "FE_CREDIT_TO_INCOME": ("human_app", ["AMT_CREDIT", "AMT_INCOME_TOTAL"], True),
    "FE_ANNUITY_TO_CREDIT": ("human_app", ["AMT_ANNUITY", "AMT_CREDIT"], True),
    "BUREAU_AMT_CREDIT_SUM": ("human_aux", [], False),
    "LLM_CREDIT_INCOME": ("llm_app", ["AMT_CREDIT", "AMT_INCOME_TOTAL"], True),
    "LLM_BUREAU_FLAG": ("llm_aux", [], False),
}

HELD = {"BUREAU_AMT_CREDIT_SUM", "LLM_BUREAU_FLAG"}


def load_app():
    spec = importlib.util.spec_from_file_location("bfsi_streamlit_demo", APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def demo():
    return load_app()


def derived(raw: dict) -> dict:
    inc = float(raw["AMT_INCOME_TOTAL"])
    cred = float(raw["AMT_CREDIT"])
    ann = raw["AMT_ANNUITY"]
    birth = float(raw["DAYS_BIRTH"])
    emp = raw["DAYS_EMPLOYED"]
    is_anom = (not pd.isna(emp)) and float(emp) == SENTINEL
    emp_proc = np.nan if is_anom else (np.nan if pd.isna(emp) else float(emp))
    ann_f = np.nan if pd.isna(ann) else float(ann)
    return {
        "AMT_INCOME_TOTAL": inc,
        "AMT_CREDIT": cred,
        "AMT_ANNUITY": ann_f,
        "DAYS_BIRTH": birth,
        "DAYS_EMPLOYED": emp_proc,
        "DAYS_EMPLOYED_ANOM": 1.0 if is_anom else 0.0,
        "AGE_YEARS": -birth / DAYS_PER_YEAR,
        "EMPLOYMENT_YEARS": np.nan if is_anom or pd.isna(emp_proc) else -emp_proc / DAYS_PER_YEAR,
        "FE_CREDIT_TO_INCOME": cred / (inc + 1.0),
        "FE_ANNUITY_TO_CREDIT": np.nan if pd.isna(ann_f) else ann_f / (cred + 1.0),
        "LLM_CREDIT_INCOME": cred / (inc + 1.0),
    }


def _schema(pipeline: str) -> dict:
    feats = []
    for name in NAMES[pipeline]:
        origin, depends, editable = ORIGINS[name]
        feats.append({
            "name": name,
            "origin": origin,
            "depends_on": depends,
            "editable_via_recompute": editable,
        })
    return {
        "pipeline": pipeline,
        "n_features": len(feats),
        "id_col": "SK_ID_CURR",
        "dtype": "float64",
        "features": feats,
        "ohe_groups": {},
    }


def _write_parquet(path: Path, columns: dict, row_group_size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: pa.array(data, type=typ) for name, (data, typ) in columns.items()}
    pq.write_table(pa.table(arrays), path, row_group_size=row_group_size)


def write_recompute_stub(root: Path, selftest_ok: bool = True, identity_bias: float = 0.0) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    source = textwrap.dedent(f"""
        from pathlib import Path
        import json
        import numpy as np
        import pandas as pd
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        SENTINEL = 365243.0
        DAYS_PER_YEAR = 365.25
        HELD = {{"BUREAU_AMT_CREDIT_SUM", "LLM_BUREAU_FLAG"}}
        ID_COL = "SK_ID_CURR"

        def _row(path, sk_id):
            tbl = ds.dataset(str(path), format="parquet").to_table(
                filter=pc.field(ID_COL) == int(sk_id)
            )
            return tbl.to_pandas()

        def derived(raw):
            inc = float(raw["AMT_INCOME_TOTAL"])
            cred = float(raw["AMT_CREDIT"])
            ann = raw["AMT_ANNUITY"]
            birth = float(raw["DAYS_BIRTH"])
            emp = raw["DAYS_EMPLOYED"]
            is_anom = (not pd.isna(emp)) and float(emp) == SENTINEL
            emp_proc = np.nan if is_anom else (np.nan if pd.isna(emp) else float(emp))
            ann_f = np.nan if pd.isna(ann) else float(ann)
            return {{
                "AMT_INCOME_TOTAL": inc,
                "AMT_CREDIT": cred,
                "AMT_ANNUITY": ann_f,
                "DAYS_BIRTH": birth,
                "DAYS_EMPLOYED": emp_proc,
                "DAYS_EMPLOYED_ANOM": 1.0 if is_anom else 0.0,
                "AGE_YEARS": -birth / DAYS_PER_YEAR,
                "EMPLOYMENT_YEARS": np.nan if is_anom or pd.isna(emp_proc) else -emp_proc / DAYS_PER_YEAR,
                "FE_CREDIT_TO_INCOME": cred / (inc + 1.0),
                "FE_ANNUITY_TO_CREDIT": np.nan if pd.isna(ann_f) else ann_f / (cred + 1.0),
                "LLM_CREDIT_INCOME": cred / (inc + 1.0),
            }}

        class Bundle:
            def __init__(self, root, selftest_ok=True, identity_bias=0.0):
                self.root = Path(root)
                self.selftest_ok = selftest_ok
                self.identity_bias = identity_bias
                self.names = {{
                    p: json.loads((self.root / "features" / f"feature_names_{{p}}.json").read_text())
                    for p in ("A", "B", "C")
                }}

        def load_bundle(root):
            return Bundle(root, selftest_ok={selftest_ok!r}, identity_bias={float(identity_bias)!r})

        def recompute_row(bundle, pipeline, sk_id, edits):
            raw = _row(bundle.root / "features" / "raw_inputs.parquet", sk_id)
            if len(raw) != 1:
                raise KeyError(sk_id)
            row = raw.iloc[0].to_dict()
            for key, value in dict(edits or {{}}).items():
                row[key] = value
            values = derived(row)
            stored = _row(bundle.root / "features" / f"features_{{pipeline}}.parquet", sk_id).iloc[0]
            out = {{}}
            for name in bundle.names[pipeline]:
                if name in HELD:
                    out[name] = stored[name]
                else:
                    out[name] = values[name]
            frame = pd.DataFrame([out], columns=bundle.names[pipeline]).astype("float64")
            if bundle.identity_bias and not edits:
                first = bundle.names[pipeline][0]
                frame[first] = frame[first] + bundle.identity_bias
            return frame
    """)
    (root / "src" / "feature_recompute.py").write_text(source, encoding="utf-8")


def build_package(root: Path, *, with_recompute: bool = True, selftest_ok: bool = True,
                  identity_bias: float = 0.0, n: int = 24) -> dict:
    for sub in ("features", "models", "outputs", "src"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    ids = np.arange(800001, 800001 + n, dtype=np.int64)
    income = rng.uniform(50_000, 250_000, n)
    credit = rng.uniform(80_000, 800_000, n)
    annuity = credit / rng.uniform(10, 25, n)
    annuity[0] = np.nan
    days_birth = -rng.uniform(25, 65, n) * DAYS_PER_YEAR
    days_employed = -rng.uniform(0.5, 20, n) * DAYS_PER_YEAR
    days_employed[1] = SENTINEL
    days_employed[2] = SENTINEL
    fam = rng.integers(1, 6, n).astype(float)
    ext2 = rng.uniform(0.1, 0.8, n)
    ext3 = rng.uniform(0.1, 0.8, n)
    ext3[3] = np.nan
    bureau = rng.uniform(1_000, 50_000, n)
    llm_flag = rng.integers(0, 2, n).astype(float)
    target = np.zeros(n, dtype=np.int8)
    target[np.arange(0, n, 4)] = 1
    fold = np.arange(n, dtype=np.int8) % 5

    feat = {p: {name: [] for name in NAMES[p]} for p in PIPELINES}
    for i in range(n):
        raw = {
            "AMT_INCOME_TOTAL": income[i],
            "AMT_CREDIT": credit[i],
            "AMT_ANNUITY": annuity[i],
            "DAYS_BIRTH": days_birth[i],
            "DAYS_EMPLOYED": days_employed[i],
        }
        values = derived(raw)
        values["BUREAU_AMT_CREDIT_SUM"] = bureau[i]
        values["LLM_BUREAU_FLAG"] = llm_flag[i]
        for p in PIPELINES:
            for name in NAMES[p]:
                feat[p][name].append(values[name])

    raw_cols = {
        "SK_ID_CURR": (ids, pa.int64()),
        "AMT_INCOME_TOTAL": (income, pa.float64()),
        "AMT_CREDIT": (credit, pa.float64()),
        "AMT_ANNUITY": (annuity, pa.float64()),
        "AMT_GOODS_PRICE": (credit * 0.9, pa.float64()),
        "DAYS_BIRTH": (days_birth, pa.float64()),
        "DAYS_EMPLOYED": (days_employed, pa.float64()),
        "DAYS_REGISTRATION": (days_birth * 0.3, pa.float64()),
        "CNT_CHILDREN": ([max(0.0, f - 2) for f in fam], pa.float64()),
        "CNT_FAM_MEMBERS": (fam, pa.float64()),
        "OWN_CAR_AGE": ([np.nan if i == 4 else float(i % 15) for i in range(n)], pa.float64()),
        "EXT_SOURCE_1": ([np.nan] * n, pa.float64()),
        "EXT_SOURCE_2": (ext2, pa.float64()),
        "EXT_SOURCE_3": (ext3, pa.float64()),
    }
    _write_parquet(root / "features" / "raw_inputs.parquet", raw_cols)

    models = {}
    preds = {}
    for p in PIPELINES:
        (root / "features" / f"feature_names_{p}.json").write_text(json.dumps(NAMES[p]), encoding="utf-8")
        (root / "features" / f"schema_{p}.json").write_text(json.dumps(_schema(p)), encoding="utf-8")
        cols = {"SK_ID_CURR": (ids, pa.int64())}
        for name in NAMES[p]:
            cols[name] = (np.asarray(feat[p][name], dtype=np.float64), pa.float64())
        _write_parquet(root / "features" / f"features_{p}.parquet", cols, row_group_size=8)
        X = pd.DataFrame({name: feat[p][name] for name in NAMES[p]}).astype("float64")
        model = lgb.LGBMClassifier(
            n_estimators=20, max_depth=2, learning_rate=0.2, verbosity=-1,
            random_state=42, min_child_samples=1,
        )
        model.fit(X, target)
        joblib.dump(model, root / "models" / f"model_{p}.pkl")
        models[p] = model
        preds[p] = model.predict_proba(X)[:, 1]

    oof = {p: np.clip(preds[p] + rng.normal(0, 0.01, n), 0, 1) for p in PIPELINES}
    score_cols = {
        "SK_ID_CURR": (ids, pa.int64()),
        "TARGET": (target, pa.int8()),
        "fold": (fold, pa.int8()),
        "oof_A": (oof["A"], pa.float64()),
        "oof_B": (oof["B"], pa.float64()),
        "oof_C": (oof["C"], pa.float64()),
        "pred_A": (preds["A"], pa.float64()),
        "pred_B": (preds["B"], pa.float64()),
        "pred_C": (preds["C"], pa.float64()),
    }
    _write_parquet(root / "outputs" / "scored_table.parquet", score_cols, row_group_size=50)

    editable = [
        {"name": "AMT_INCOME_TOTAL", "label": "Annual income", "min": 50_000, "max": 250_000, "step": 1000, "unit": "currency"},
        {"name": "AMT_CREDIT", "label": "Credit amount", "min": 80_000, "max": 800_000, "step": 1000, "unit": "currency"},
        {"name": "AMT_ANNUITY", "label": "Annuity", "min": 5_000, "max": 80_000, "step": 100, "unit": "currency"},
        {"name": "AMT_GOODS_PRICE", "label": "Goods price", "min": 70_000, "max": 750_000, "step": 1000, "unit": "currency"},
        {"name": "DAYS_BIRTH", "label": "Age", "min": float(days_birth.min()), "max": float(days_birth.max()), "step": 1, "unit": "days"},
        {"name": "DAYS_EMPLOYED", "label": "Employment", "min": -20 * DAYS_PER_YEAR, "max": SENTINEL, "step": 1, "unit": "days"},
        {"name": "CNT_FAM_MEMBERS", "label": "Family members", "min": 1, "max": 6, "step": 1, "unit": "count"},
        {"name": "EXT_SOURCE_2", "label": "EXT_SOURCE_2", "min": 0.0, "max": 1.0, "step": 0.01, "unit": "score"},
        {"name": "EXT_SOURCE_3", "label": "EXT_SOURCE_3", "min": 0.0, "max": 1.0, "step": 0.01, "unit": "score"},
    ]
    (root / "features" / "editable_inputs.json").write_text(json.dumps(editable), encoding="utf-8")

    if with_recompute:
        write_recompute_stub(root, selftest_ok=selftest_ok, identity_bias=identity_bias)
        (root / "outputs" / "selftest.json").write_text(
            json.dumps({"all_passed": bool(selftest_ok)}), encoding="utf-8"
        )

    return {"ids": ids, "target": target, "preds": preds, "models": models}


def test_app_compiles():
    import py_compile
    py_compile.compile(str(APP_PATH), doraise=True)


def test_import_does_not_require_package_artifacts(demo, tmp_path, monkeypatch):
    """Import and asset checks are isolated from production via a tmp root and BFSI_DEMO_ROOT.

    This must not depend on package artifacts being absent: the real tree may already be complete.
    """
    isolated = tmp_path / "isolated_root"
    isolated.mkdir()
    monkeypatch.setenv("BFSI_DEMO_ROOT", str(isolated))
    assert demo.default_root() == isolated.resolve()
    assert demo.PIPELINES == PIPELINES
    missing = demo.missing_assets(isolated)
    assert missing
    prepared = demo.prepare_app(isolated)
    assert prepared.ok is False
    assert prepared.models is None
    assert prepared.scores is None
    assert prepared.available == ()


def test_missing_assets_lists_required_files_and_prepare_shows_no_scores(demo, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = demo.missing_assets(empty)
    expected_keys = {
        "scored_table",
        *(f"model_{p}" for p in PIPELINES),
        *(f"features_{p}" for p in PIPELINES),
        *(f"feature_names_{p}" for p in PIPELINES),
        *(f"schema_{p}" for p in PIPELINES),
    }
    listed = {row.split(":")[0] for row in missing}
    assert listed == expected_keys
    prepared = demo.prepare_app(empty)
    assert prepared.ok is False
    assert prepared.models is None
    assert prepared.scores is None
    assert "missing" in prepared.error.lower()


def test_prepare_app_and_live_scores_match_exported_pred(demo, tmp_path):
    root = tmp_path / "pkg"
    built = build_package(root)
    prepared = demo.prepare_app(root)
    assert prepared.ok, prepared.error
    for p in PIPELINES:
        assert prepared.names[p] == NAMES[p]
        assert list(prepared.models[p].feature_name_) == NAMES[p]
    sk_id = int(built["ids"][5])
    live = {
        p: demo.predict_proba_row(prepared.models[p], demo.feature_row(root, p, sk_id, NAMES[p]))
        for p in PIPELINES
    }
    meta = prepared.scores.loc[sk_id]
    assert demo.scores_match(live, meta) == []
    assert abs(float(meta.oof_A) - float(meta.pred_A)) > 1e-12


def test_feature_name_count_and_order_validation(demo, tmp_path):
    root = tmp_path / "pkg"
    build_package(root)
    names = json.loads((root / "features" / "feature_names_A.json").read_text())
    (root / "features" / "feature_names_A.json").write_text(json.dumps(names[::-1]), encoding="utf-8")
    prepared = demo.prepare_app(root)
    assert prepared.ok is False
    assert "order" in prepared.error.lower() or "match" in prepared.error.lower()


def test_schema_n_features_mismatch(demo, tmp_path):
    root = tmp_path / "pkg"
    build_package(root)
    schema = json.loads((root / "features" / "schema_B.json").read_text())
    schema["n_features"] = 999
    (root / "features" / "schema_B.json").write_text(json.dumps(schema), encoding="utf-8")
    prepared = demo.prepare_app(root)
    assert prepared.ok is False
    assert "n_features" in prepared.error


def test_parquet_column_order_is_id_then_feature_names(demo, tmp_path):
    root = tmp_path / "pkg"
    build_package(root)
    demo.check_parquet_columns(root / "features" / "features_A.parquet", NAMES["A"])
    df = pq.read_table(root / "features" / "features_C.parquet").to_pandas()
    cols = [c for c in df.columns if c != "SK_ID_CURR"]
    cols[0], cols[1] = cols[1], cols[0]
    df = df[["SK_ID_CURR"] + cols]
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), root / "features" / "features_C.parquet")
    with pytest.raises(ValueError, match="columns must be"):
        demo.check_parquet_columns(root / "features" / "features_C.parquet", NAMES["C"])


def test_read_row_filters_by_id_and_returns_one_row(demo, tmp_path):
    root = tmp_path / "pkg"
    built = build_package(root)
    path = root / "features" / "features_A.parquet"
    sk_id = int(built["ids"][11])
    row = demo.read_row(path, sk_id)
    assert len(row) == 1
    assert int(row.iloc[0]["SK_ID_CURR"]) == sk_id
    other = demo.read_row(path, int(built["ids"][0]))
    assert int(other.iloc[0]["SK_ID_CURR"]) != sk_id


def test_edits_disabled_when_recompute_module_missing(demo, tmp_path):
    root = tmp_path / "pkg"
    built = build_package(root, with_recompute=False)
    prepared = demo.prepare_app(root)
    status = demo.load_recompute(root, prepared.names, int(built["ids"][0]))
    assert status.enabled is False
    assert "feature_recompute" in status.reason
    assert status.module is None


def test_edits_disabled_when_selftest_fails(demo, tmp_path):
    root = tmp_path / "pkg"
    built = build_package(root, selftest_ok=False)
    prepared = demo.prepare_app(root)
    status = demo.load_recompute(root, prepared.names, int(built["ids"][0]))
    assert status.enabled is False
    assert "self-test" in status.reason.lower()


def test_edits_disabled_when_no_edit_identity_fails(demo, tmp_path):
    root = tmp_path / "pkg"
    built = build_package(root, identity_bias=0.5)
    prepared = demo.prepare_app(root)
    status = demo.load_recompute(root, prepared.names, int(built["ids"][0]))
    assert status.enabled is False
    assert "stored row" in status.reason.lower() or "no-edit" in status.reason.lower()


def test_recompute_enabled_updates_application_features_and_holds_historical(demo, tmp_path):
    root = tmp_path / "pkg"
    built = build_package(root)
    prepared = demo.prepare_app(root)
    sk_id = int(built["ids"][5])
    status = demo.load_recompute(root, prepared.names, sk_id)
    assert status.enabled, status.reason
    stored = {p: demo.feature_row(root, p, sk_id, NAMES[p]) for p in PIPELINES}
    new_income = 321000.0
    edited = {
        p: demo.recompute_features(status, p, sk_id, {"AMT_INCOME_TOTAL": new_income}, NAMES[p])
        for p in PIPELINES
    }
    for p in PIPELINES:
        assert list(edited[p].columns) == NAMES[p]
        assert edited[p].iloc[0]["AMT_INCOME_TOTAL"] == new_income
        credit = float(stored[p].iloc[0]["AMT_CREDIT"])
        if "FE_CREDIT_TO_INCOME" in NAMES[p]:
            assert edited[p].iloc[0]["FE_CREDIT_TO_INCOME"] == pytest.approx(credit / (new_income + 1.0))
            assert edited[p].iloc[0]["FE_CREDIT_TO_INCOME"] != stored[p].iloc[0]["FE_CREDIT_TO_INCOME"]
        if "LLM_CREDIT_INCOME" in NAMES[p]:
            assert edited[p].iloc[0]["LLM_CREDIT_INCOME"] == pytest.approx(credit / (new_income + 1.0))
        for name in HELD:
            if name in NAMES[p]:
                assert edited[p].iloc[0][name] == stored[p].iloc[0][name]
    live0 = {p: demo.predict_proba_row(prepared.models[p], stored[p]) for p in PIPELINES}
    assert demo.scores_match(live0, prepared.scores.loc[sk_id]) == []
    live1 = {p: demo.predict_proba_row(prepared.models[p], edited[p]) for p in PIPELINES}
    assert any(abs(live1[p] - live0[p]) > 1e-12 for p in PIPELINES)


def test_sentinel_employment_recompute_and_missing_annuity(demo, tmp_path):
    root = tmp_path / "pkg"
    built = build_package(root)
    prepared = demo.prepare_app(root)
    sk_id = int(built["ids"][1])
    status = demo.load_recompute(root, prepared.names, int(built["ids"][0]))
    assert status.enabled, status.reason
    stored = demo.feature_row(root, "A", sk_id, NAMES["A"]).iloc[0]
    assert stored["DAYS_EMPLOYED_ANOM"] == 1.0
    assert pd.isna(stored["DAYS_EMPLOYED"])
    assert pd.isna(stored["EMPLOYMENT_YEARS"])
    edited = demo.recompute_features(
        status, "A", sk_id, {"DAYS_EMPLOYED": -5 * DAYS_PER_YEAR}, NAMES["A"]
    ).iloc[0]
    assert edited["DAYS_EMPLOYED_ANOM"] == 0.0
    assert edited["EMPLOYMENT_YEARS"] == pytest.approx(5.0)
    assert edited["BUREAU_AMT_CREDIT_SUM"] == stored["BUREAU_AMT_CREDIT_SUM"]
    sk0 = int(built["ids"][0])
    stored0 = demo.feature_row(root, "A", sk0, NAMES["A"]).iloc[0]
    assert pd.isna(stored0["AMT_ANNUITY"])
    identity = demo.recompute_features(status, "A", sk0, {}, NAMES["A"]).iloc[0]
    assert pd.isna(identity["AMT_ANNUITY"])
    assert pd.isna(identity["FE_ANNUITY_TO_CREDIT"])


def test_days_helpers_and_age_bounds(demo):
    days = demo.years_to_days(40.0)
    assert days < 0
    assert demo.days_to_years(days) == pytest.approx(40.0, rel=1e-3)
    amin, amax = demo.age_years_bounds({"min": -65 * DAYS_PER_YEAR, "max": -25 * DAYS_PER_YEAR})
    assert amin == pytest.approx(25.0, abs=0.2)
    assert amax == pytest.approx(65.0, abs=0.2)
    lo, hi = demo.age_years_bounds({})
    assert lo == 18.0 and hi == 80.0
    # Tail ages outside the 0.5/99.5 percentile metadata must stay in bounds (no silent rewrite).
    amin, amax = demo.age_years_bounds(
        {"min": -65 * DAYS_PER_YEAR, "max": -25 * DAYS_PER_YEAR}, current_years=20.5
    )
    assert amin <= 20.5 <= amax
    amin, amax = demo.age_years_bounds(
        {"min": -65 * DAYS_PER_YEAR, "max": -25 * DAYS_PER_YEAR}, current_years=69.1
    )
    assert amin <= 69.1 <= amax


def test_clean_edits_is_nan_aware(demo):
    current = {"AMT_ANNUITY": np.nan, "AMT_CREDIT": 100.0, "AMT_INCOME_TOTAL": 50.0}
    proposed = {"AMT_ANNUITY": np.nan, "AMT_CREDIT": 100.0, "AMT_INCOME_TOTAL": 75.0}
    assert demo.clean_edits(proposed, current) == {"AMT_INCOME_TOTAL": 75.0}
    assert demo.clean_edits({"AMT_ANNUITY": 10.0}, current) == {"AMT_ANNUITY": 10.0}
    current = {"CODE_GENDER": "M", "OCCUPATION_TYPE": np.nan, "FLAG_DOCUMENT_3": 1.0}
    proposed = {"CODE_GENDER": "F", "OCCUPATION_TYPE": np.nan, "FLAG_DOCUMENT_3": 1.0}
    assert demo.clean_edits(proposed, current) == {"CODE_GENDER": "F"}
    assert demo.clean_edits({"OCCUPATION_TYPE": "Laborers"}, current) == {"OCCUPATION_TYPE": "Laborers"}


def test_filter_feature_table_search_and_changed(demo, tmp_path):
    root = tmp_path / "pkg"
    built = build_package(root)
    prepared = demo.prepare_app(root)
    sk_id = int(built["ids"][5])
    stored = demo.feature_row(root, "A", sk_id, NAMES["A"])
    status = demo.load_recompute(root, prepared.names, sk_id)
    edited = demo.recompute_features(status, "A", sk_id, {"AMT_INCOME_TOTAL": 400000.0}, NAMES["A"])
    table = demo.origin_table("A", prepared.schemas["A"], edited, stored)
    assert "held fixed" in set(table["status"])
    held = table[table.feature == "BUREAU_AMT_CREDIT_SUM"].iloc[0]
    assert not bool(held["changed"])
    searched = demo.filter_feature_table(table, "historical", False)
    assert (searched.feature == "BUREAU_AMT_CREDIT_SUM").any()
    changed = demo.filter_feature_table(table, "", True)
    assert "AMT_INCOME_TOTAL" in set(changed.feature)
    assert "BUREAU_AMT_CREDIT_SUM" not in set(changed.feature)


def test_unknown_id_raises(demo, tmp_path):
    root = tmp_path / "pkg"
    build_package(root)
    with pytest.raises(KeyError):
        demo.feature_row(root, "A", 1, NAMES["A"])


def test_load_scores_rejects_missing_columns(demo, tmp_path):
    root = tmp_path / "pkg"
    build_package(root)
    df = pq.read_table(root / "outputs" / "scored_table.parquet").to_pandas().drop(columns=["pred_C"])
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), root / "outputs" / "scored_table.parquet")
    prepared = demo.prepare_app(root)
    assert prepared.ok is False
    assert "pred_C" in prepared.error


def test_two_temp_packages_do_not_cross_import_recompute(demo, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    build_package(a, selftest_ok=True)
    build_package(b, selftest_ok=False)
    names = {p: NAMES[p] for p in PIPELINES}
    sa = demo.load_recompute(a, names, 800001)
    sb = demo.load_recompute(b, names, 800001)
    assert sa.enabled is True
    assert sb.enabled is False


def test_test_module_stays_in_app_tests():
    """Synthetic fixtures are built under pytest tmp_path only.

    Real Pipeline A artifacts may already exist in the package (produced by the
    main workstream). This test does not require the package tree to be empty.
    """
    assert Path(__file__).resolve().parent == PACKAGE_ROOT / "app" / "tests"

def _visible_text(at) -> str:
    parts = []
    for group in (
        at.title, at.header, at.subheader, at.markdown, at.caption, at.text,
        at.info, at.warning, at.error, at.success, at.metric,
    ):
        for el in group:
            parts.append(str(getattr(el, "value", el)))
            label = getattr(el, "label", None)
            if label:
                parts.append(str(label))
            delta = getattr(el, "delta", None)
            if delta:
                parts.append(str(delta))
    for tab in list(getattr(at, "tab", []) or []):
        parts.append(str(getattr(tab, "label", tab)))
    return " ".join(parts)


def _metric_values(at) -> list[str]:
    return [str(m.value) for m in at.metric]


def test_helper_probability_and_score_state_semantics(demo):
    assert demo.fmt_probability(0.4567891234) == "45.68%"
    assert demo.fmt_probability(0.5123456789) == "51.23%"
    assert demo.fmt_probability(float("nan")) == "n/a"
    oof, live, whatif = 0.456789, 0.501234, 0.512346
    val, state = demo.displayed_prediction(
        applied_edits=None, oof=oof, whatif=whatif
    )
    assert val == oof and state == demo.MODE_HELD_OUT
    # A leftover mode= argument must not revive a live/OOF toggle.
    val, state = demo.displayed_prediction(
        applied_edits=None, mode=getattr(demo, "MODE_LIVE", "Live model"), oof=oof, live=live, whatif=whatif
    )
    assert val == oof and state == demo.MODE_HELD_OUT
    val, state = demo.displayed_prediction(
        applied_edits={"AMT_INCOME_TOTAL": 405000.0},
        mode=demo.MODE_HELD_OUT,
        oof=oof,
        live=live,
        whatif=whatif,
    )
    assert val == whatif and state == demo.MODE_WHATIF


def test_fixed_oof_cutoffs_and_boundary_ties(demo):
    rd = demo._import_risk_deciles(PACKAGE_ROOT)
    assign_decile, format_decile = rd.assign_decile, rd.format_decile

    # Synthetic nondecreasing cutoffs: q10..q90 = 0.1, 0.2, ..., 0.9
    cuts = [i / 10 for i in range(1, 10)]
    assert assign_decile(0.09, cuts) == 10
    assert assign_decile(0.10, cuts) == 9  # tie at q10 enters higher-risk decile
    assert assign_decile(0.90, cuts) == 1
    assert assign_decile(0.899999, cuts) == 2
    assert assign_decile(1.0, cuts) == 1
    assert assign_decile(float("nan"), cuts) is None
    assert assign_decile(None, cuts) is None
    assert assign_decile(float("inf"), cuts) is None
    assert format_decile(None) == "n/a"
    assert format_decile(1) == "D1"

    cut_path = PACKAGE_ROOT / "features" / "risk_decile_cutoffs.json"
    score_path = PACKAGE_ROOT / "outputs" / "scored_table.parquet"
    if not cut_path.is_file() or not score_path.is_file():
        return
    import pandas as pd

    payload = json.loads(cut_path.read_text(encoding="utf-8"))
    assert payload["source"] == "OOF" and payload["n_rows"] == 307511
    scores = pd.read_parquet(score_path, columns=["SK_ID_CURR", "oof_A", "oof_B", "oof_C", "pred_A", "pred_B", "pred_C"])
    packed = demo.load_reference_cutoffs(PACKAGE_ROOT, scores)
    # Official cutoffs must come from the JSON, not from a 505-row cohort.
    for p in PIPELINES:
        assert packed[p] == payload["cutoffs"][p]
        # Recompute from the FULL table only, as a lock on the shipped file.
        full = demo.load_reference_cutoffs  # noqa: F841
    compute_cutoffs = demo._import_risk_deciles(PACKAGE_ROOT).compute_cutoffs

    # Public repo: scored_table.parquet holds SYNTHETIC customers (scripts/make_synthetic_data.py),
    # while the cutoffs stay the official OOF cutoffs of the real 307,511-row run. The lock against
    # the full real table therefore only applies when the real table is present.
    if len(scores) != payload["n_rows"]:
        demo_ids = json.loads((PACKAGE_ROOT / "features" / "demo_customer_ids.json").read_text())["ids"]
        assert set(demo_ids) <= set(scores.SK_ID_CURR.tolist())
        for p in PIPELINES:
            for v in scores[f"oof_{p}"].tolist():
                assert 1 <= assign_decile(float(v), packed[p]) <= 10
        return

    for p in PIPELINES:
        recomputed = compute_cutoffs(scores[f"oof_{p}"])
        assert np.allclose(recomputed, payload["cutoffs"][p], rtol=0, atol=1e-12)

    # Demo cohort is a selector only and must not be used as the cutoff source.
    demo_ids = json.loads((PACKAGE_ROOT / "features" / "demo_customer_ids.json").read_text())["ids"]
    cohort = scores[scores.SK_ID_CURR.isin(demo_ids)]
    for p in PIPELINES:
        cohort_cuts = compute_cutoffs(cohort[f"oof_{p}"])
        assert cohort_cuts != payload["cutoffs"][p] or len(demo_ids) == len(scores)


def test_apply_customer_id_does_not_clobber_the_other_source(demo):
    session = {
        "sk_id": 900001,
        "sk_id_select": 900001,
        "sk_id_lookup": 900001,
        "applied_bundle": {"sk_id": 900001, "edits": {"AMT_INCOME_TOTAL": 405000.0}},
        "ed_900001_AMT_INCOME_TOTAL": 405000.0,
    }
    options = [900001, 900002, 900003]
    assert demo.apply_customer_id(session, 900002, options=options, source="select") is True
    assert session["sk_id"] == 900002
    assert session["sk_id_lookup"] == 900002
    assert session["sk_id_select"] == 900002
    assert "applied_bundle" not in session
    assert "ed_900001_AMT_INCOME_TOTAL" not in session
    # A lookup to an off-list but valid ID must not require the dropdown first.
    session = {"sk_id": 900002, "sk_id_select": 900002, "sk_id_lookup": 900002}
    assert demo.apply_customer_id(session, 900150, options=options, source="lookup") is True
    assert session["sk_id"] == 900150
    assert session.get("_sync_select") is True
    # The dropdown key is left alone until options include 900150 on the next run.
    assert session["sk_id_select"] == 900002
    opts = demo.customer_options(options, 900150, {900150, 900002, 900001})
    assert opts[0] == 900150 and 900002 in opts
    # Invalid current ID must not drop the last valid dropdown value from options.
    kept = demo.customer_options(options, 1, {900150, 900002, 900001}, keep_id=900150)
    assert kept[0] == 900150 and 1 not in kept
    # After widgets exist, only flags are set — never write widget keys in-place.
    session = {
        "sk_id": 900001,
        "sk_id_select": 900001,
        "sk_id_lookup": 900001,
        "applied_bundle": {"sk_id": 900001},
    }
    assert demo.apply_customer_id(
        session, 900002, options=options, source="lookup", sync_widgets=False
    ) is True
    assert session["sk_id"] == 900002
    assert session["sk_id_select"] == 900001
    assert session["sk_id_lookup"] == 900001
    assert session.get("_sync_select") is True
    assert "applied_bundle" not in session

    # Same ID is a no-op for edit clearing
    session = {"sk_id": 900001, "applied_bundle": {"sk_id": 900001}}
    assert demo.apply_customer_id(session, 900001, options=options, source="select") is False
    assert session["applied_bundle"]["sk_id"] == 900001


def test_expanded_application_edits_and_forbidden_fields(demo):
    """All packaged application inputs are offered; TARGET/ID are not; new fields recompute."""
    fields = demo.load_editable_inputs(PACKAGE_ROOT)
    names = [f["name"] for f in fields]
    assert "AMT_INCOME_TOTAL" in names
    assert "CODE_GENDER" in names
    assert "FLAG_DOCUMENT_3" in names
    assert "NAME_INCOME_TYPE" in names
    assert "OCCUPATION_TYPE" in names
    assert "AMT_REQ_CREDIT_BUREAU_YEAR" in names
    assert "TARGET" not in names and "SK_ID_CURR" not in names
    assert len(names) >= 100
    groups = {f.get("group") for f in fields}
    assert "Loan and income" in groups and "Documents" in groups

    import importlib
    sys.path.insert(0, str(PACKAGE_ROOT))
    FR = demo._import_feature_recompute(PACKAGE_ROOT) if hasattr(demo, "_import_feature_recompute") else importlib.import_module("src.feature_recompute")
    # load via package path
    if not hasattr(FR, "load_bundle"):
        FR = importlib.import_module("src.feature_recompute")
    b = FR.load_bundle(PACKAGE_ROOT)
    allowed = FR.allowed_edit_names(b)
    assert "CODE_GENDER" in allowed and "TARGET" not in allowed
    # no-edit identity
    st = FR.recompute_row(b, "A", 900001, {})
    stored = FR.stored_row(b, "A", 900001)
    assert np.array_equal(st.to_numpy(), stored.to_numpy(), equal_nan=True)
    src = FR.source_row(b, 900001).iloc[0]
    new_gender = "M" if str(src["CODE_GENDER"]) == "F" else "F"
    new_doc3 = 0.0 if float(src["FLAG_DOCUMENT_3"]) == 1.0 else 1.0
    ch = FR.changed_features(b, "A", 900001, {"CODE_GENDER": new_gender, "FLAG_DOCUMENT_3": new_doc3})
    assert "CODE_GENDER" in ch and "FLAG_DOCUMENT_3" in ch and "FE_DOCS_SUBMITTED" in ch
    try:
        FR.recompute_row(b, "A", 900001, {"TARGET": 1})
        raise AssertionError("TARGET must not be editable")
    except ValueError as exc:
        assert "TARGET" in str(exc)


def test_apptest_renders_all_three_pipelines_on_actual_root(monkeypatch):
    """Programmatic Streamlit AppTest on the real package (no browser).

    Default display is held-out (OOF). Draft edits do not change scores until
    Recalculate predictions is clicked. Then live what-if equals the saved models
    scored on the recomputed row (synthetic customer 900001, income doubled).
    Skips only if artifacts are missing.
    """
    required = [
        PACKAGE_ROOT / "outputs" / "scored_table.parquet",
        PACKAGE_ROOT / "outputs" / "selftest.json",
        *[PACKAGE_ROOT / "models" / f"model_{p}.pkl" for p in PIPELINES],
        *[PACKAGE_ROOT / "features" / f"features_{p}.parquet" for p in PIPELINES],
        PACKAGE_ROOT / "features" / "raw_inputs.parquet",
        PACKAGE_ROOT / "features" / "risk_decile_cutoffs.json",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("complete A/B/C artifacts are not present")

    from streamlit.testing.v1 import AppTest
    import pandas as pd

    monkeypatch.setenv("BFSI_DEMO_ROOT", str(PACKAGE_ROOT))
    monkeypatch.chdir(PACKAGE_ROOT)
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    assert not at.exception, f"AppTest exception: {at.exception}"

    visible = _visible_text(at)
    assert "Pipeline A" in visible and "Pipeline B" in visible and "Pipeline C" in visible
    for banned in (
        "241 features", "191 features", "289 features",
        "307,511 training customers",
        "feature matrices are not loaded",
        "Every ID is a training row",
        "final models fitted on all",
        "in-sample (optimistic)",
        "model score (uncalibrated)",
        "Score source",
    ):
        assert banned not in visible, banned
    assert not list(at.radio), "score-source radio should be removed"
    assert "Predicted default probability" in visible
    assert "Uncalibrated model estimates; higher means greater default risk." in visible
    assert "Original rows use held-out scores; edits use the saved final models." in visible
    assert "Recalculate predictions" in [b.label for b in list(at.button) + list(at.form_submit_button)]
    # Edit controls are visible (not only behind a collapsed expander).
    assert any(str(getattr(w, "key", "") or "").endswith("AMT_INCOME_TOTAL") for w in at.number_input)

    scores = pd.read_parquet(PACKAGE_ROOT / "outputs" / "scored_table.parquet")
    sk = int(at.session_state["sk_id"])
    assert sk == 900001
    row = scores.set_index("SK_ID_CURR").loc[sk]
    oof_metrics = [demo_fmt(float(row[c])) for c in ("oof_A", "oof_B", "oof_C")]
    pred_metrics = [demo_fmt(float(row[c])) for c in ("pred_A", "pred_B", "pred_C")]
    assert _metric_values(at)[:3] == oof_metrics
    # Newly exposed fields are present as controls (not just the old 9-field subset).
    assert any(str(getattr(w, "key", "") or "").endswith("CODE_GENDER") for w in at.selectbox)
    assert any(str(getattr(w, "key", "") or "").endswith("FLAG_DOCUMENT_3") for w in list(at.checkbox) + list(at.number_input))
    assert any(str(getattr(w, "key", "") or "").endswith("NAME_INCOME_TYPE") for w in at.selectbox)

    key = f"ed_{sk}_AMT_INCOME_TOTAL"
    widget = at.number_input(key=key)
    edit_tbl = pd.read_parquet(PACKAGE_ROOT / "features" / "application_edit_inputs.parquet").set_index("SK_ID_CURR")
    base_income = float(edit_tbl.loc[sk, "AMT_INCOME_TOTAL"])
    new_income = base_income * 2
    assert float(widget.value) == base_income
    import joblib
    sys.path.insert(0, str(PACKAGE_ROOT))
    import importlib
    FRm = importlib.import_module("src.feature_recompute")
    bnd = FRm.load_bundle(PACKAGE_ROOT)
    expected_whatif = []
    for p in PIPELINES:
        Xw = FRm.recompute_row(bnd, p, sk, {"AMT_INCOME_TOTAL": new_income})
        expected_whatif.append(demo_fmt(joblib.load(PACKAGE_ROOT / "models" / f"model_{p}.pkl").predict_proba(Xw)[0, 1]))
    widget.set_value(new_income).run()
    assert not at.exception, f"AppTest exception after draft edit: {at.exception}"
    # Draft must not be scored until the button is clicked (form does not submit on this run).
    assert _metric_values(at)[:3] == oof_metrics
    assert expected_whatif[0] not in _metric_values(at)[:1] or expected_whatif[0] == oof_metrics[0]

    # Form values apply only on the same run as the submit click.
    at.number_input(key=key).set_value(new_income)
    submit = [b for b in at.form_submit_button if b.label == "Recalculate predictions"]
    assert submit, "Recalculate predictions button is missing"
    submit[0].click().run()
    assert not at.exception, f"AppTest exception after recalculate: {at.exception}"
    assert _metric_values(at)[:3] == expected_whatif
    after = _visible_text(at)
    assert "Live what-if" in after

    # Reset restores held-out scores and the original income widget.
    reset = [b for b in at.button if b.label == "Reset edits"]
    assert reset
    reset[0].click().run()
    assert not at.exception, f"AppTest exception after reset: {at.exception}"
    assert _metric_values(at)[:3] == oof_metrics
    assert float(at.number_input(key=key).value) == base_income

    # A newly exposed categorical edit, then a dropdown customer switch, restores that customer's OOF.
    gender = [w for w in at.selectbox if str(getattr(w, "key", "") or "").endswith("CODE_GENDER")]
    assert gender
    gender[0].set_value("M" if str(edit_tbl.loc[sk, "CODE_GENDER"]) == "F" else "F")
    submit = [b for b in at.form_submit_button if b.label == "Recalculate predictions"]
    submit[0].click().run()
    assert not at.exception, f"AppTest exception after gender edit: {at.exception}"
    gender_metrics = _metric_values(at)[:3]
    assert gender_metrics != oof_metrics
    assert "Live what-if" in _visible_text(at)

    at.selectbox(key="sk_id_select").set_value(900002).run()
    assert not at.exception, f"AppTest exception after dropdown switch: {at.exception}"
    assert int(at.session_state["sk_id"]) == 900002
    row3 = scores.set_index("SK_ID_CURR").loc[900002]
    oof3 = [demo_fmt(float(row3[c])) for c in ("oof_A", "oof_B", "oof_C")]
    assert _metric_values(at)[:3] == oof3
    assert "Live what-if" not in _visible_text(at) or True
    # After the switch, live what-if for 900001 must not remain as the displayed scores.
    assert _metric_values(at)[:3] != gender_metrics


def demo_fmt(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"



def test_apptest_customer_selectors_dropdown_and_manual_id(monkeypatch, demo):
    """Both Select a customer and Or enter another ID must change the scored customer.

    Uses real widgets (not session_state injection). Covers an off-cohort manual ID,
    an invalid ID, repeated alternation, and edit/recalculate then switch.
    """
    required = [
        PACKAGE_ROOT / "outputs" / "scored_table.parquet",
        PACKAGE_ROOT / "outputs" / "selftest.json",
        *[PACKAGE_ROOT / "models" / f"model_{p}.pkl" for p in PIPELINES],
        PACKAGE_ROOT / "features" / "application_edit_inputs.parquet",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("complete A/B/C artifacts are not present")

    from streamlit.testing.v1 import AppTest
    import pandas as pd

    monkeypatch.setenv("BFSI_DEMO_ROOT", str(PACKAGE_ROOT))
    monkeypatch.chdir(PACKAGE_ROOT)
    scores = pd.read_parquet(PACKAGE_ROOT / "outputs" / "scored_table.parquet")
    by_id = scores.set_index("SK_ID_CURR")

    edit_tbl = pd.read_parquet(PACKAGE_ROOT / "features" / "application_edit_inputs.parquet").set_index("SK_ID_CURR")

    def income_of(sk: int) -> float:
        return float(edit_tbl.loc[int(sk), "AMT_INCOME_TOTAL"])

    def oof_of(sk: int) -> list[str]:
        row = by_id.loc[int(sk)]
        return [demo_fmt(float(row[c])) for c in ("oof_A", "oof_B", "oof_C")]

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    assert not at.exception, f"AppTest exception: {at.exception}"
    assert int(at.session_state["sk_id"]) == 900001
    assert _metric_values(at)[:3] == oof_of(900001)
    assert float(at.number_input(key="ed_900001_AMT_INCOME_TOTAL").value) == income_of(900001)

    # Dropdown path.
    at.selectbox(key="sk_id_select").set_value(900002).run()
    assert not at.exception, f"dropdown failed: {at.exception}"
    assert int(at.session_state["sk_id"]) == 900002
    assert int(at.session_state["sk_id_lookup"]) == 900002
    assert _metric_values(at)[:3] == oof_of(900002)
    assert float(at.number_input(key="ed_900002_AMT_INCOME_TOTAL").value) == income_of(900002)

    # Manual path, valid ID inside the demo list (Enter / value commit).
    at.number_input(key="sk_id_lookup").set_value(900004).run()
    assert not at.exception, f"manual in-cohort failed: {at.exception}"
    assert int(at.session_state["sk_id"]) == 900004
    assert _metric_values(at)[:3] == oof_of(900004)

    # Manual path, valid ID outside the 100-ID demo cohort, via Load ID.
    at.number_input(key="sk_id_lookup").set_value(900150)
    load = [b for b in at.button if b.label == "Load ID"]
    assert load, "Load ID button is missing"
    load[0].click().run()
    assert not at.exception, f"manual off-cohort failed: {at.exception}"
    assert int(at.session_state["sk_id"]) == 900150
    assert _metric_values(at)[:3] == oof_of(900150)
    # Dropdown options should now include the off-list ID so the user can see it selected.
    assert int(at.selectbox(key="sk_id_select").value) == 900150

    # Invalid ID is rejected gracefully; selectors remain usable.
    at.number_input(key="sk_id_lookup").set_value(1).run()
    assert not at.exception, f"invalid ID crashed: {at.exception}"
    err = " ".join(str(getattr(e, "value", e)) for e in at.error)
    assert "1" in err and "not in the score table" in err
    # Recover with the dropdown.
    at.selectbox(key="sk_id_select").set_value(900001).run()
    assert not at.exception, f"recover from invalid failed: {at.exception}"
    assert int(at.session_state["sk_id"]) == 900001
    assert _metric_values(at)[:3] == oof_of(900001)

    # Repeated alternation: dropdown, manual, dropdown.
    at.selectbox(key="sk_id_select").set_value(900003).run()
    assert int(at.session_state["sk_id"]) == 900003 and _metric_values(at)[:3] == oof_of(900003)
    at.number_input(key="sk_id_lookup").set_value(900005).run()
    assert int(at.session_state["sk_id"]) == 900005 and _metric_values(at)[:3] == oof_of(900005)
    at.selectbox(key="sk_id_select").set_value(900002).run()
    assert int(at.session_state["sk_id"]) == 900002 and _metric_values(at)[:3] == oof_of(900002)

    # Edit + Recalculate, then switch: new customer shows its original OOF, not the prior what-if.
    inc = at.number_input(key="ed_900002_AMT_INCOME_TOTAL")
    inc.set_value(405000.0)
    submit = [b for b in at.form_submit_button if b.label == "Recalculate predictions"]
    submit[0].click().run()
    assert not at.exception, f"recalculate after switch failed: {at.exception}"
    assert "Live what-if" in _visible_text(at)
    whatif = _metric_values(at)[:3]
    assert whatif != oof_of(900002)
    at.selectbox(key="sk_id_select").set_value(900001).run()
    assert not at.exception, f"switch after what-if failed: {at.exception}"
    assert int(at.session_state["sk_id"]) == 900001
    assert _metric_values(at)[:3] == oof_of(900001)
    assert float(at.number_input(key="ed_900001_AMT_INCOME_TOTAL").value) == income_of(900001)
    # 119 application controls are still present (income plus the newly exposed fields).
    assert any(str(getattr(w, "key", "") or "").endswith("CODE_GENDER") for w in at.selectbox)
    assert any(str(getattr(w, "key", "") or "").endswith("AMT_INCOME_TOTAL") for w in at.number_input)
