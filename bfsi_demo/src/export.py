"""Artifact writers (see ARTIFACT_CONTRACT.md)."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from . import config as C

def save_model(model, pipeline, model_dir=C.MODEL_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"model_{pipeline}.pkl")
    model.booster_.save_model(str(model_dir / f"model_{pipeline}.txt"))

def save_matrix(X, ids, pipeline, feat_dir=C.FEAT_DIR):
    """SK_ID_CURR int64 + features float64 in model order. Returns float64 frame."""
    feat_dir.mkdir(parents=True, exist_ok=True)
    Xf = X.astype('float64')
    df = pd.concat([pd.Series(ids.to_numpy(dtype='int64'), name=C.ID_COL), Xf.reset_index(drop=True)], axis=1)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), feat_dir / f"features_{pipeline}.parquet",
                   row_group_size=50_000, compression="zstd")
    (feat_dir / f"feature_names_{pipeline}.json").write_text(json.dumps(list(X.columns), indent=1))
    return Xf

def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=conv))
