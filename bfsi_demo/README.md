# BFSI MTP — Engineering Demonstration (Home Credit Default Risk)
Gourav Ray (M25AI2085), Executive M.Tech AI, IIT Jodhpur.

Three credit-default pipelines on the Kaggle **Home Credit Default Risk** training set (307,511 applicants):

| Pipeline | Features | Source of features |
|---|---|---|
| **A** (human) | 241 | Exact port of the original Pipeline A notebook (`src/features_a.py`) |
| **B** (LLM-only) | mechanical application base + gate-passed LLM features | 6 live Qwen calls on Groq (`src/llm.py`) |
| **C** (hybrid) | A′ human block + gate-passed LLM features | `src/features_bc.py` + LLM |

All three use the same LightGBM settings (the original Optuna best params, `src/config.py`), StratifiedKFold(5, shuffle, seed 42),
early stopping 100 on the validation fold, then a final all-data fit with 375 trees.
**Scores are raw, uncalibrated LightGBM probabilities** (`scale_pos_weight` ≈ 11.39 raises them versus the 8.07% base rate). Use them for ranking, not as calibrated PDs.

Verified metrics of this run: `outputs/final_results.json` → `pipelines`. Historical June-2026 numbers are kept separately under `historical_reference` and were **not** produced by this run.

## Verified results of this run (5-fold OOF, 307,511 customers; raw uncalibrated scores)
| Pipeline | Features | LLM features | OOF AUC | KS | CV std | Fold AUCs |
|---|---|---|---|---|---|---|
| A | 241 | – | 0.78362 | 0.42897 | 0.00349 | 0.78014 / 0.78828 / 0.78220 / 0.78730 / 0.78025 |
| B | 191 | 25 | 0.77360 | 0.41214 | 0.00391 | 0.76906 / 0.77807 / 0.77278 / 0.77827 / 0.76996 |
| C | 289 | 25 | 0.78580 | 0.43190 | 0.00328 | 0.78297 / 0.78993 / 0.78416 / 0.78962 / 0.78241 |

* A reproduces the June-2026 Pipeline A run exactly (same 241 features, same fold AUCs).
* LLM: `qwen/qwen3.8-27b`, 6/6 calls OK, 32 proposed. Gate: 26 passed, 5 `constant`, 1 `security_violation`. 25 of the 26 survive selection (`instal_zero_payment_rate` was dropped by the near-zero-variance filter).
  The 5 credit-card proposals were genuinely broken: they aggregate by `SK_ID_PREV` and then rename it to `SK_ID_CURR`, so after the left join and `fillna(0)` they are constant 0 (see `outputs/validation_log.csv`).
  The security rejection is the original `os.` substring rule matching `recent_pos.groupby` (a known false positive, kept). No rejected feature was replaced by hand.
* Checks: `outputs/selftest.json` (55 checks, all passed) and `outputs/llm_alignment_audit.json` (every accepted LLM feature re-executed on the real tables: unique SK_ID_CURR index, matrix equal).

## Quickest demo
```bash
pip install -r requirements.txt          # Python 3.11
streamlit run app/streamlit_demo.py      # reads only models/, features/, outputs/ (no raw data or keys needed)
```
Pick an `SK_ID_CURR` to see its stored inputs and the three side-by-side probabilities, or edit application fields (see "Editable inputs" below).
App documentation: `app/README_demo.md`.

## Exact rebuild
```bash
pip install -r requirements.txt
# credentials: KAGGLE_API_TOKEN (Kaggle API) and GROQ_API_KEY, as env vars or files in $BFSI_SECRETS_DIR (never committed)
python run_all.py all          # = A, BC, finalize   (resumable: python run_all.py A | BC | finalize)
```
* The data is fetched with the official Kaggle API (`src/data.py`, competition `home-credit-default-risk`; accept its rules once on kaggle.com) into `data/raw/` (not packaged).
* LLM responses are cached by prompt hash in `outputs/llm_logs/`. A rebuild with the packaged cache reuses the exact same 6 responses and makes no Groq calls.
  Delete `outputs/llm_logs/call*_*.json` to regenerate them live.
* Notebooks (`notebooks/Pipeline_A_BFSI_Baseline_v2.ipynb`, `notebooks/Pipeline_B_C_BFSI_v2.ipynb`) are saved **with outputs** from a fresh-kernel nbconvert run.
  Each runs a fresh 5-fold CV *trained in the kernel* on the cached feature parquets and reproduces the packaged OOF scores exactly (max diff 0).
  It then *loads* the final `model_*.pkl` files for live predictions. Features, LLM responses and gate results are loaded, not regenerated (see `outputs/notebook_execution_manifest.json`).
  The rebuild itself is `run_all.py`.
* Tests: `pytest -q tests app/tests`. `tests/test_synthetic_roundtrip.py` is a **synthetic** smoke test, not real results.

## Package layout
```
models/    model_{A,B,C}.pkl (joblib LGBMClassifier) + model_{A,B,C}.txt (LightGBM booster text)
features/  features_{A,B,C}.parquet  every customer: SK_ID_CURR (int64) + model inputs in model order (float64); no TARGET
           feature_names_{A,B,C}.json, schema_{A,B,C}.json (origin, depends_on, editable_via_recompute, ohe_groups)
           raw_inputs.parquet, editable_inputs.json, llm_feature_specs.json (code, gate result, row_level, used in B/C)
outputs/   scored_table.parquet/.csv  SK_ID_CURR, TARGET, fold, oof_A/B/C (out-of-fold CV), pred_A/B/C (final model, in-sample)
           final_results.json, selftest.json, validation_log.csv (8 gate stages × every candidate), feature_importance_{A,B,C}.csv,
           run_log.txt, llm_logs/ (prompts, raw responses, per-call model/usage/attempt/retry records)
src/       all pipeline code;  run_all.py  orchestrator;  app/  Streamlit demo (separate owner)
ARTIFACT_CONTRACT.md  file formats shared by the pipeline and the app
```

## Pipeline details
### A — original human pipeline (unchanged procedure)
preprocessing → 20 `FE_` application ratios → bureau (44), installments (16), previous (19), credit card (14), POS (10) aggregates →
missing > 80% drop → ORGANIZATION_TYPE drop (> 50 levels) → one-hot (`dummy_na=True, drop_first=True`) → variance < 1e-6 drop →
Spearman |ρ| > 0.97 collinearity drop (50k sample, seed 42) → 241 features. The regenerated list equals the June-2026 registry in content and order.
The registry stores names *before* `clean_feature_names` (e.g. `NAME_EDUCATION_TYPE_Higher education`). The model uses the cleaned names
(`NAME_EDUCATION_TYPE_Higher_education`), and the comparison is made after applying the same cleaning to the registry.

### B / C — LLM design (corrected)
* **Base** = mechanical preprocessing only (sentinel 365243 → NaN + flag, day → year conversions, XNA/XAP → NaN, binary label-encoding,
  one-hot with high-cardinality drop). No human ratios or aggregates are in B (asserted in code).
* **6 focused calls**: application (6), bureau + bureau_balance (6), installments (5), previous application (5), credit card (5), POS (5).
  Prompts are verbatim from the original notebook. Each call is told to avoid the matching Pipeline-A human features. Model: chosen from the
  **authenticated** Groq model list (Qwen preferred, `qwen/qwen3.8-27b`). Settings: JSON mode, temperature 0, hidden reasoning, `max_completion_tokens=4096`.
  4096 instead of the original 8192 because Groq's free-tier tokens-per-minute limit counts the full completion budget. Real completions were about 1.2–1.5k tokens and none was truncated.
  429 rate-limit retries honour the server's retry-after hint, and every attempt is recorded.
* **8-stage gate** (`src/gate.py`): 1 security (the original forbidden-substring list + an AST whitelist sandbox: no file, network, OS,
  dunder or IO access; restricted builtins; only `import numpy as np` / `import pandas as pd` tolerated as no-ops), 2 code execution (timeout),
  3 format (pd.Series), 4 high null > 95%, 5 constant, 6 infinite / > 1e12, 7 target-leakage |Spearman| > 0.95, 8 exact duplicate.
  **No collinearity filter.** Rejected features are dropped. There are no hand-written replacements.
* **C** keeps the A′ human block from the B/C notebook (base + the same 20 `FE_` ratios + A′ aggregates). How it differs from original A:
  missing > 85% and near-zero-variance > 99% selection (numeric only) instead of > 80% / variance / Spearman 0.97, a slightly different aggregate set,
  and no collinearity pruning. A′ itself is not cross-validated in this run (historical only).
* Selection for B and C: missing > 85%, near-zero-variance > 99% (bool OHE columns exempt), as in the original notebook.

### Deviations from the June notebooks (all deliberate, all logged)
1. Llama-4-Scout was decommissioned on Groq, so B/C use Qwen; new LLM features, new B/C scores. Old Llama numbers appear only under `historical_reference`.
2. Stage 1 hardened with an AST sandbox. LLM code runs on shallow table copies, so features cannot mutate shared tables for later features.
3. Gate leakage and duplicate checks align on SK_ID_CURR. The notebook reindexed by row position, which compared misaligned rows.
4. Memory: auxiliary-table strings are loaded as pandas `category`. Equivalence with `object` dtype was verified for every aggregate builder (`data/cache/equivalence.json`, copied into `final_results.json → checks`).
5. `max_completion_tokens` 8192 → 4096 (rate limit, see above).
6. One candidate was rejected by the **original** forbidden-substring rule `os.` matching the variable name `recent_pos.`. This is a known false positive of the original rule, kept for faithfulness.

## Editable inputs and recompute (`src/feature_recompute.py`)
Editable: AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, CNT_CHILDREN,
CNT_FAM_MEMBERS, OWN_CAR_AGE, EXT_SOURCE_1/2/3 (ranges in `features/editable_inputs.json`).
An edit recomputes the mechanical columns, all 20 `FE_` ratios and the row-level LLM application features with the **same functions** used for training.
Example: AMT_GOODS_PRICE changes FE_CREDIT_TO_GOODS / FE_GOODS_PRICE_GAP even in A, where the raw AMT_GOODS_PRICE column itself is not a model input.
Bureau, previous, installments, card and POS aggregates are **held** at stored values. `outputs/selftest.json` proves that unedited recomputation equals the stored
matrices bit-for-bit (vectorised for all 307,511 rows and per-customer through the API), and that edits propagate only to editable features.

## Provenance
`final_results.json` holds the SHA-256 of every input CSV, model file and feature matrix, the Groq model actually used, the authenticated model list,
per-call token usage and retry attempts, gate outcomes, stage timings and peak memory, plus the UTC and IST finish time.

## Security
No credentials are stored in this package (a secret scan runs in the selftest and at packaging). Keys are read only from env vars or `$BFSI_SECRETS_DIR`.
