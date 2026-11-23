"""Shared configuration. Secrets are read lazily from env vars or the private secrets dir
(outside the package) and are never printed, logged or written into artifacts."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(os.environ.get("BFSI_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("BFSI_DATA_DIR", ROOT / "data" / "raw"))
FEAT_DIR, MODEL_DIR, OUT_DIR = ROOT / "features", ROOT / "models", ROOT / "outputs"
LLM_DIR = OUT_DIR / "llm_logs"
CACHE_DIR = ROOT / "data" / "cache"          # intermediate, not packaged
SECRETS_DIR = Path(os.environ.get("BFSI_SECRETS_DIR", "/workspace/bfsi-mtp/.secrets"))

RANDOM_SEED, N_FOLDS = 42, 5
TARGET_COL, ID_COL = "TARGET", "SK_ID_CURR"

# Original Pipeline A best params (pipeline_a_best_params.json, Optuna trial 49, 2026-06-03)
BEST_PARAMS = {
    "n_estimators": 375, "learning_rate": 0.06520428345169353, "num_leaves": 73,
    "max_depth": 4, "min_child_samples": 39, "subsample": 0.9009025593333467,
    "colsample_bytree": 0.5646949393948328, "reg_alpha": 0.09479741965476791,
    "reg_lambda": 0.8690655078595865, "min_split_gain": 0.04971641436663787,
    "objective": "binary", "metric": "auc", "verbosity": -1, "boosting_type": "gbdt",
    "random_state": 42, "n_jobs": -1, "scale_pos_weight": 11.387150050352467,
}
N_THREADS = int(os.environ.get("BFSI_THREADS", os.cpu_count() or 4))

HISTORICAL_REFERENCE = {
    "note": "Historical numbers from the June-2026 notebooks (Colab). NOT produced by this run.",
    "original_A_2026_06_03": {"n_features": 241, "oof_auc": 0.78362, "ks": 0.42897, "gini": 0.56723,
        "brier": 0.18035, "lift": 3.677, "cv_std": 0.00349,
        "fold_aucs": [0.78014, 0.78828, 0.78220, 0.78730, 0.78025]},
    "bc_notebook_llama4_scout_2026_06_16": {
        "llm": "meta-llama/llama-4-scout-17b-16e-instruct (decommissioned on Groq 2026-07-17)",
        "A_prime": {"n_features": 264, "oof_auc": 0.78402, "fold_aucs": [0.78036, 0.78887, 0.78313, 0.78715, 0.78071]},
        "B": {"n_features": 197, "n_llm": 31, "oof_auc": 0.77665, "fold_aucs": [0.7722, 0.78139, 0.77702, 0.77999, 0.77273]},
        "C": {"n_features": 295, "oof_auc": 0.78428, "fold_aucs": [0.78111, 0.78833, 0.78326, 0.78809, 0.7807]},
        "gate": {"proposed": 32, "passed": 31, "code_error": 1}},
}

def get_secret(name: str) -> str | None:
    """Return secret value from env or SECRETS_DIR/<name>; never log it."""
    v = os.environ.get(name)
    if v:
        return v.strip()
    p = SECRETS_DIR / name
    if p.is_file():
        return p.read_text().strip() or None
    return None

def secret_available(name: str) -> bool:
    return get_secret(name) is not None

def ensure_dirs():
    for d in (DATA_DIR, FEAT_DIR, MODEL_DIR, OUT_DIR, LLM_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
