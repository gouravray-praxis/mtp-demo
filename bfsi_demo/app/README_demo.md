# Streamlit demo — Home Credit Pipelines A / B / C

Live scoring app for the Final MTP engineering demonstration. Pick a customer,
edit application fields, and click **Recalculate predictions** to score a what-if.

This file describes only the app. Training artifacts are defined in
`../ARTIFACT_CONTRACT.md`.

## Run

From the package root (`BFSI_MTP_Demo/`):

```bash
streamlit run app/streamlit_demo.py
```

On Windows (PowerShell), from the extracted package folder, without activating the venv:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_demo.py
```

Then open http://localhost:8501. The saved-model demo never reads `data/raw/` and
needs no network access, Kaggle credentials, or Groq API keys.

## What you will see

1. **Customer** — a searchable demo list plus an optional ID lookup (type an ID and press Enter, or click **Load ID**). **TARGET**
   is in an optional expander and is never a model input.
2. **Predicted default probability** — a percent (two decimals) and a risk
   decile. An unchanged customer shows the **Held-out (OOF)** score. After you
   click **Recalculate predictions**, the cards switch to **Live what-if**
   (saved final models). **Reset edits** returns to the original held-out
   scores. There is no separate live/OOF source toggle.
3. **Edit application inputs** — every packaged raw application field that can
   safely be recomputed (numeric, flag, and categorical), grouped and visible.
   Derived ratios, missing indicators, and encodings update from those inputs.
   Historical-table aggregates stay **held fixed**. Changing a field is a draft
   until you click **Recalculate predictions**.
4. **Model inputs** — one tab per pipeline, with every feature value and a
   user-friendly origin label. Download A+B+C values as CSV.

A short line under the title states: *Uncalibrated model estimates; higher
means greater default risk.*

## What can be edited

The editor lists all application-table inputs that affect at least one
pipeline, including income and loan amounts, personal and family fields,
employment, region flags, external scores, bureau enquiries, documents, and
property attributes, plus categoricals such as contract type, gender, income
type, education, family status, occupation, and housing type. `TARGET`,
`SK_ID_CURR`, and high-cardinality fields that the models do not use
(`ORGANIZATION_TYPE`) are not editable. Bureau, previous-application,
instalment, credit-card, and POS aggregates are held at their stored values.

Deciles use the fixed OOF cutoffs in `features/risk_decile_cutoffs.json`
(D1 highest risk, D10 lowest). They are never refit on a customer or an edit.
