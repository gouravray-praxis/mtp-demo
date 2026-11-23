# ARTIFACT CONTRACT — BFSI MTP Demo (Home Credit)  v1.0  (2026-11-11 IST)

Root: `/workspace/bfsi-mtp/BFSI_MTP_Demo/` (all paths below are relative to it).
Owner of everything except `app/streamlit_demo.py` and `app/README_demo.md` (app worker) = pipeline worker.
App code must import only from `src/` + read the artifacts below. Never read `data/raw/` (not shipped).

## 1. Models (pipeline ∈ {A,B,C})
| path | format | notes |
|---|---|---|
| `models/model_{P}.pkl` | joblib dump of `lightgbm.LGBMClassifier` (lightgbm==4.5.0) | final fit on ALL 307,511 training rows, `n_estimators=375`, no early stopping, original Pipeline-A best params |
| `models/model_{P}.txt` | LightGBM native booster text | same model; `lgb.Booster(model_file=...)` |
Predict: `model.predict_proba(X[feature_names_P].astype("float64"))[:, 1]`  → raw default score.
**Scores are NOT calibrated** (training used `scale_pos_weight≈11.387`; raw scores overstate PD; mean ≈ 0.3–0.4). Label in UI: "model score (uncalibrated)".

## 2. Feature matrices
`features/features_{P}.parquet` — one row per training customer (307,511 rows), pyarrow, row groups ≈ 50,000.
- column 0: `SK_ID_CURR` (int64, unique, same order in A/B/C = application_train.csv order)
- columns 1..n: exactly `feature_names_{P}.json` order, all **float64** (bools/dummies stored 0.0/1.0, missing = NaN)
- NO `TARGET` column inside feature parquets.
Efficient single-row read: `pyarrow.dataset.dataset(path).to_table(filter=pc.field("SK_ID_CURR")==id).to_pandas()`.

`features/feature_names_{P}.json` — JSON array of strings (model input order; equals `model.feature_name_`).

`features/schema_{P}.json` —
```json
{"pipeline":"A","n_features":241,"id_col":"SK_ID_CURR","dtype":"float64",
 "features":[{"name":"FE_CREDIT_TO_INCOME","origin":"human_app|human_aux|mechanical|raw|ohe|llm_app|llm_aux",
              "depends_on":["AMT_CREDIT","AMT_INCOME_TOTAL"],"editable_via_recompute":true}],
 "ohe_groups":{"NAME_EDUCATION_TYPE":{"baseline_category":"Academic degree",
               "columns":{"Higher education":"NAME_EDUCATION_TYPE_Higher_education", "...":"..."},
               "nan_column":"NAME_EDUCATION_TYPE_nan"}}}
```
(`ohe_groups` columns that were removed by feature selection are listed with value `null`.)

`features/raw_inputs.parquet` — per customer original application inputs used by the editable recompute
(float64 unless noted): `SK_ID_CURR`(int64), `AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
DAYS_BIRTH, DAYS_EMPLOYED (raw, 365243 sentinel kept), DAYS_REGISTRATION, CNT_CHILDREN, CNT_FAM_MEMBERS,
OWN_CAR_AGE, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3`.

`features/editable_inputs.json` — UI metadata list:
`[{"name":"AMT_INCOME_TOTAL","label":"Annual income","min":..,"max":..,"step":..,"unit":"currency"} ...]`
(min/max = training 0.5/99.5 percentiles).

`features/llm_feature_specs.json` — accepted LLM features (B & C):
`[{"feature_name":..,"source_group":"application|bureau|installments|previous_app|credit_card|pos_cash",
   "source_tables":[..],"row_level":bool,"python_code":"def compute_x(app_train, bureau, ...): ...",
   "gate":"passed","rationale":..}]`

## 3. Scores
`outputs/scored_table.parquet` (+ identical `outputs/scored_table.csv`), 307,511 rows, joined on `SK_ID_CURR`:
| col | dtype | meaning |
|---|---|---|
| SK_ID_CURR | int64 | key |
| TARGET | int8 | label (external; never a feature) |
| fold | int8 | 0..4, StratifiedKFold(5, shuffle, seed 42) validation fold |
| oof_A, oof_B, oof_C | float64 | out-of-fold CV scores → used for ALL reported metrics |
| pred_A, pred_B, pred_C | float64 | final all-data model scores (in-sample; demo only, optimistic) |

## 4. Results
`outputs/final_results.json` keys: `execution_status` ("complete"|"partial"|"blocked"), `run` (utc/ist timestamps,
host, cpu, ram, timings), `data` (row counts, sha256 of CSVs), `llm` (requested/selected model id, available
qwen ids from authenticated list, calls[6] with tokens/retries/cached), `gate` (proposed/passed/failure taxonomy),
`pipelines.{A,A_prime,B,C}` (n_features, oof_auc, ks, gini, brier, lift, cv_mean, cv_std, fold_aucs),
`checks` (list of {name, passed, detail}), `historical_reference` (June-2026 Llama-4-Scout / original-A numbers —
NOT new results). `A_prime` = human block re-implemented in the B/C notebook, used inside C (distinct from A).

## 5. Editable recompute (src/feature_recompute.py)
```python
from src.feature_recompute import load_bundle, recompute_row
bundle = load_bundle(root)                # loads specs/schemas once
X = recompute_row(bundle, pipeline="A", sk_id=100002, edits={"AMT_INCOME_TOTAL": 250000.0})
# -> pandas.DataFrame, 1 row, columns == feature_names_A (float64), ready for model.predict_proba
```
Contract:
- `edits` keys ⊆ names in `editable_inputs.json` (raw application units; DAYS_* negative days).
- Starts from the customer's stored feature row; recomputes, in dependency order:
  raw → mechanical (AGE_YEARS, EMPLOYMENT_YEARS, DAYS_EMPLOYED_ANOM, …YEARS) → human application ratios (FE_*) →
  row-level LLM features (`row_level==true`, executed through the same AST-sandbox used by the gate).
- Aggregates from bureau/previous/installments/credit-card/POS tables are **held** at stored values.
- Features dropped by selection are computed internally but not returned.
- `recompute_row(..., edits={})` must equal the stored parquet row exactly (assert in `src/selftest.py`; the
  app should call `bundle.selftest_ok` / show a warning if False).
- `changed_features(bundle, pipeline, sk_id, edits)` → list of feature names whose value changed.
