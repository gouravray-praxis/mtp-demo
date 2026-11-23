"""SYNTHETIC smoke test (not real data): train -> export float64 parquet -> recompute identity -> predictions match,
for three toy models sharing one raw_inputs file. Writes only to a pytest tmp dir."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src import feature_recompute as FR, export, sandbox
import lightgbm as lgb

def _raw(n=600, seed=0):
    r = np.random.default_rng(seed)
    d = {c: r.normal(0, 1, n) for c in FR.FE_INPUTS}
    d.update(AMT_INCOME_TOTAL=r.uniform(5e4, 5e5, n), AMT_CREDIT=r.uniform(5e4, 1e6, n), AMT_ANNUITY=r.uniform(5e3, 5e4, n),
             AMT_GOODS_PRICE=r.uniform(4e4, 9e5, n), DAYS_BIRTH=-r.uniform(8000, 25000, n),
             DAYS_EMPLOYED=np.where(r.random(n) < .15, 365243, -r.uniform(0, 8000, n)), CNT_FAM_MEMBERS=r.integers(1, 6, n).astype(float),
             EXT_SOURCE_1=np.where(r.random(n) < .4, np.nan, r.random(n)), EXT_SOURCE_2=r.random(n), EXT_SOURCE_3=r.random(n))
    for i in range(2, 22): d[f"FLAG_DOCUMENT_{i}"] = r.integers(0, 2, n).astype(float)
    df = pd.DataFrame(d); df.insert(0, "SK_ID_CURR", np.arange(800000, 800000 + n, dtype="int64")); return df

def test_synthetic_roundtrip(tmp_path):
    for sub in ("features", "models", "outputs"): (tmp_path / sub).mkdir()
    raw = FR.build_raw_inputs(_raw(), FR.FE_INPUTS); raw.to_parquet(tmp_path / "features/raw_inputs.parquet", index=False)
    code = "def compute_syn_ratio(app_train, bureau, bureau_bal, prev_app, install, cc_bal, pos_cash):\n    df = app_train[['SK_ID_CURR','AMT_CREDIT','AMT_INCOME_TOTAL']].copy()\n    df['v'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)\n    return df.set_index('SK_ID_CURR')['v']\n"
    specs = [{"feature_name": "syn_ratio", "source_group": "application", "row_level": True, "app_columns": ["AMT_CREDIT", "AMT_INCOME_TOTAL"], "python_code": code}]
    (tmp_path / "features/llm_feature_specs.json").write_text(json.dumps(specs))
    der = FR.derive_app(raw); der["syn_ratio"] = FR.llm_series(code, "syn_ratio", der).reindex(raw.SK_ID_CURR).values
    y = pd.Series((der["FE_CREDIT_TO_INCOME"].rank(pct=True) + np.random.default_rng(1).random(len(der)) > 1.2).astype(int))
    sets = {"A": ["AMT_CREDIT", "AGE_YEARS", "FE_CREDIT_TO_INCOME", "FE_EXT_SOURCE_MEAN"], "B": ["AMT_CREDIT", "syn_ratio", "EMPLOYMENT_YEARS"],
            "C": ["FE_CREDIT_TO_INCOME", "syn_ratio", "FE_EXT_SOURCE_STD", "DAYS_EMPLOYED"]}
    models = {}
    for P, cols in sets.items():
        X = der[cols]; m = lgb.LGBMClassifier(n_estimators=30, verbose=-1).fit(X, y); models[P] = m
        export.save_matrix(X, raw.SK_ID_CURR, P, feat_dir=tmp_path / "features")
    b = FR.load_bundle(tmp_path)
    for P, m in models.items():
        for i in (800000, 800123, 800599):
            st, rc = FR.stored_row(b, P, i), FR.recompute_row(b, P, i, {})
            assert np.array_equal(st.to_numpy(), rc.to_numpy(), equal_nan=True)
            assert m.predict_proba(rc)[0, 1] == m.predict_proba(st)[0, 1]
        ch = FR.changed_features(b, P, 800123, {"AMT_CREDIT": 123456.0})
        assert ch, P
    # sandbox rejects unsafe code
    import pytest
    with pytest.raises(sandbox.UnsafeCode):
        sandbox.check("def compute_x(app_train, bureau, bureau_bal, prev_app, install, cc_bal, pos_cash):\n    return open('/etc/passwd').read()\n")
