"""5-fold CV (original run_cv logic: StratifiedKFold(5, shuffle, 42), early_stopping(100) on the validation
fold, n_estimators=375 cap) + final all-data fit (n_estimators=375, no early stopping)."""
from __future__ import annotations
import gc, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, average_precision_score
from . import config as C

def compute_ks(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(np.abs(tpr - fpr)))

def decile_lift(y_true, y_proba, n=10):
    df = pd.DataFrame({'y': y_true, 'p': y_proba})
    df['d'] = n - pd.qcut(df['p'], n, labels=False, duplicates='drop')
    t = df.groupby('d').agg(Count=('y', 'count'), Pos=('y', 'sum')).reset_index()
    return float((t['Pos'] / t['Count']).iloc[0] / y_true.mean())

def params(n_threads=C.N_THREADS):
    p = dict(C.BEST_PARAMS)
    p['n_jobs'] = n_threads
    return p

def fold_ids(y, n_folds=C.N_FOLDS, seed=C.RANDOM_SEED):
    f = np.full(len(y), -1, dtype=np.int8)
    for k, (_, val) in enumerate(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        f[val] = k
    return f

def run_cv(X, y, label, log=print, n_folds=C.N_FOLDS, seed=C.RANDOM_SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X)); fold_aucs, trees, fi = [], [], pd.DataFrame(index=X.columns)
    t0 = time.time()
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(**params())
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[val], y.iloc[val])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        p = m.predict_proba(X.iloc[val])[:, 1]
        oof[val] = p
        fa = roc_auc_score(y.iloc[val], p); fold_aucs.append(fa)
        bi = getattr(m, 'best_iteration_', None); trees.append(int(bi) if bi else None)
        fi[f'f{fold+1}'] = m.booster_.feature_importance(importance_type='gain')
        log(f'  {label} fold {fold+1} AUC={fa:.5f} trees={bi} t={time.time()-t0:.0f}s')
        del m; gc.collect()
    auc = roc_auc_score(y, oof)
    metrics = {'oof_auc': round(float(auc), 5), 'ks': round(compute_ks(y.values, oof), 5),
               'gini': round(float(2 * auc - 1), 5), 'brier': round(float(brier_score_loss(y, oof)), 5),
               'avg_precision': round(float(average_precision_score(y, oof)), 5),
               'lift_top_decile': round(decile_lift(y.values, oof), 4),
               'cv_mean': round(float(np.mean(fold_aucs)), 5), 'cv_std': round(float(np.std(fold_aucs)), 5),
               'fold_aucs': [round(float(a), 5) for a in fold_aucs], 'best_iterations': trees,
               'n_features': int(X.shape[1]), 'cv_time_s': round(time.time() - t0, 1)}
    fi['mean_gain'] = fi[[f'f{i}' for i in range(1, n_folds + 1)]].mean(axis=1)
    log(f'  {label} OOF AUC={auc:.5f} KS={metrics["ks"]} cv_std={metrics["cv_std"]}')
    return oof, metrics, fi

def fit_final(X, y, log=print, label=''):
    t0 = time.time()
    m = lgb.LGBMClassifier(**params())
    m.fit(X, y)
    log(f'  {label} final fit on {len(X):,} rows, {m.booster_.num_trees()} trees, {time.time()-t0:.0f}s')
    return m
