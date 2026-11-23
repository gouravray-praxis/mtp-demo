# Hindi Word-Order Results — evaluation-set dashboard (v2, live model scoring)

Paste a sentence from the 2,365-sentence evaluation set. The app shows the reference sentence and every
variant ordering. For each variant it loads the saved models `model_A/B/C.pkl` and scores the
pre-extracted feature row with `predict_proba[:, 1]` = probability that the reference order is preferred.
For a chosen variant it lists each model's features with their values, ranked by the absolute coefficient
of the loaded model. Sentences outside the evaluation set are not scored (short message + 5 examples).
No feature extraction and no LLM calls happen in the app.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
The .pkl files were saved with **scikit-learn 1.6.1** (Python 3.13); unpickling needs that exact version
(requirements.txt pins it; use Python 3.9–3.13). Keep all files below in one folder — app.py alone is not enough.

## Files
```
app.py                        the whole app
requirements.txt              streamlit, pandas, pyarrow, scikit-learn==1.6.1, numpy
model_A.pkl model_B.pkl model_C.pkl   unchanged, from MTP_Language_demo_models.zip
selected_features_A/B/C.json  unchanged (exact model input order)
demo_lookup.parquet           unchanged (sentence/variant texts used for matching)
feature_values.parquet        raw feature values of the 7,812 evaluation pairs (reference-minus-variant)
```

## What the numbers are
- **Feature values**: original reference-minus-variant values (not imputed or scaled) from the outer-validation
  fold copy of each pair: A = `feature_chunks/valid_pairs_fold_{0,1,2}.parquet` of run
  `pipeline_a_20261023T120347Z_logit_rfecv_2ep`; B = `repeat_01/llm_difference_features.parquet`
  (partition = valid); C = the same human + LLM values. Joined one-to-one; all 7,812 pairs covered, no missing
  values (a missing value would be passed to the model as NaN, median-imputed by the model's own pipeline, and shown as "—").
- **Probabilities** come from the FINAL models, which were refit on the training rows of all folds. Every
  evaluation sentence was therefore seen during that refit (via another fold's feature copy), so these
  scores are in-sample. Held-out performance is the original out-of-fold result (`prob_A/B/C` in
  `demo_lookup.parquet`, not displayed). Live vs OOF over all 7,812 pairs:
  A Pearson 0.935, mean |diff| 0.049, same preference 93.0%;
  B 0.984 / 0.027 / 95.1%; C 0.923 / 0.055 / 93.3%.
- **Weights / ranking**: absolute logistic coefficients (standardized inputs) read from the loaded models.

## Sources
`MTP_Language_demo_models.zip` sha256 902d90f1…af02e (identical to the uploaded copy);
A run zip sha256 d1f2c7ec…52f8 (same bytes as `pipeline_a_logit_rfecv_2ep (1).zip`; identity with the
"-3" upload assumed from name/size, not hash-proven);
`Language_BC_full_outputs (1).zip` sha256 eaade147…bd33, repeat_01.
