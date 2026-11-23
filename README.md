# MTP Demo: BFSI Credit Risk and Hindi Word-Order Modeling (LLM-assisted features)

This repository contains two Streamlit demos prepared for the MTP presentation.

## About this demo (important)

Our MTP proposal was an empirical evaluation of LLM-assisted feature generation, comparing baseline features (A), LLM-generated features (B) and the combined set (C) on BFSI credit-default prediction and Hindi word-order preference. The deliverables were the experiments, their evaluation and the analysis reported in MTP2. Building an interactive application was not part of that scope: the research question is answered by the evaluation metrics, not by a user interface, and the project timeline was planned around experimentation and analysis.

These Streamlit apps were added only to explain the work visually during the presentation. They are **not** the official runs that the MTP2 report results are based on, and were re-run separately for two reasons:

1. **Time limit.** The full experiments take far longer than the 7-minute presentation slot, so the demos load pre-computed results and a small set of example inputs.
2. **GitHub size limit.** The full datasets and intermediate files run to several GB, and GitHub limits single files to 100 MB. We have therefore pushed only the files the demos need. The largest file (`features_C.parquet`, about 120 MB) is stored with Git LFS, and the raw Home Credit data is not included.

Please refer to the MTP2 report for the official numbers.

## 1. BFSI demo (`bfsi_demo/`)

This demo predicts the default probability for Home Credit customers using three LightGBM models (A, B, C) and shows risk deciles. The raw data is available from the Kaggle Home Credit Default Risk competition.

## 2. Language demo (`language_demo/`)

This demo compares an original Hindi sentence from the UD Hindi Dependency Treebank (HDTB) with reordered variants, using three pairwise logistic regression models: A (baseline features), B (LLM features) and C (combined).

## How to run

Before cloning, run `git lfs install` so the large BFSI file downloads correctly.

```
git lfs install
git clone https://github.com/gouravray-praxis/mtp-demo.git
cd mtp-demo

# BFSI demo
cd bfsi_demo
pip install -r requirements.txt
streamlit run app/streamlit_demo.py

# Language demo (in a new terminal, from the repo root)
cd language_demo
pip install -r requirements.txt
streamlit run app.py
```
