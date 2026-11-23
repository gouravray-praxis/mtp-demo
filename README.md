# MTP Demo: BFSI Credit Risk and Hindi Word-Order Modeling (LLM-assisted features)

This repository contains two Streamlit demos prepared for the MTP presentation.

## About this demo (important)

Our MTP proposal was an empirical evaluation of LLM-assisted feature generation, comparing baseline features (A), LLM-generated features (B) and the combined set (C) on BFSI credit-default prediction and Hindi word-order preference. The deliverables were the experiments, their evaluation and the analysis reported in MTP2. Building an interactive application was not part of that scope: the research question is answered by the evaluation metrics, not by a user interface, and the project timeline was planned around experimentation and analysis.

This repository contains only the two demo apps (their code, the saved models and the small input files they need). The official MTP experiment/replication code and data are not here (the `bfsi_demo` folder only keeps the build scripts that came with its demo package, see below). The full MTP replication pipeline, with its own README and step-by-step instructions, is a separate submission made directly to the professor.

These Streamlit apps were added only to explain the work visually during the presentation. They are **not** the official runs that the MTP2 report results are based on, and were re-run separately for two reasons:

1. **Time limit.** The full experiments take far longer than the 7-minute presentation slot, so the demos load pre-computed results and a small set of example inputs.
2. **GitHub size limit.** The full datasets and intermediate files run to several GB, and GitHub limits single files to 100 MB. We have therefore pushed only the small files the demos need.

**The BFSI demo in this repository runs on synthetic customers, not on real Home Credit data.** The Kaggle Home Credit Default Risk competition rules do not allow the data to be redistributed, so no real applicant rows, customer IDs or default labels are included. Instead, `bfsi_demo/scripts/make_synthetic_data.py` generates 200 synthetic customers (IDs 900001 to 900200) from aggregate column statistics, and the unchanged trained models score them. Because these customers are synthetic, their scores will differ from the numbers shown in the presentation and in MTP2. The presentation itself was run locally on the real data.

To work with the real data, download it from the Kaggle competition page, https://www.kaggle.com/competitions/home-credit-default-risk (you need a Kaggle account and must accept the competition rules). The `bfsi_demo` folder still includes the build script from the demo package (`run_all.py`, described in `bfsi_demo/README.md` under "Exact rebuild"). It needs a Kaggle API token and a Groq API key, takes a long time and a lot of memory, and retrains the models, and it does not rebuild every file the app uses, so it is not a one-step way to get the presentation version back. The official experiments are in the separate replication submission.

The Language demo uses the UD Hindi Dependency Treebank (HDTB), which is licensed CC BY-NC-SA [UD Hindi HDTB page](https://universaldependencies.org/treebanks/hi_hdtb/index.html)., and runs on the same data as in the presentation.

Please refer to the MTP2 report for the official numbers.

## 1. BFSI demo (`bfsi_demo/`)

This demo predicts the default probability for Home Credit-style customers using three LightGBM models (A, B, C) and shows risk deciles. In this repository it runs on synthetic customers (see above). The dropdown lists IDs 900001 to 900100, and any ID from 900001 to 900200 can be opened with "Load ID". Risk deciles use the fixed cutoffs from the real run, so a decile shows where a synthetic customer's score would fall among the real applicants.

## 2. Language demo (`language_demo/`)

This demo compares an original Hindi sentence from the UD Hindi Dependency Treebank (HDTB) with reordered variants, using three pairwise logistic regression models: A (baseline features), B (LLM features) and C (combined).

## How to run

```
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

The synthetic BFSI files are already included. To regenerate them (same result for the same seed), run `python scripts/make_synthetic_data.py generate --n 200 --seed 42` from `bfsi_demo/`.
