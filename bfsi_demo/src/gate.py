"""8-stage validation gate — logic of B/C notebook cell 27, with stage 1 hardened by the AST sandbox.
Stages: 1 security | 2 code (exec/runtime) | 3 format (pd.Series) | 4 high_null (>95%) | 5 constant |
6 infinite (>1e12 after inf->NaN) | 7 leakage: |Spearman(feature, TARGET)| > 0.95 | 8 exact duplicate.
(The 'no Spearman filter' design refers to feature *collinearity* selection, which is NOT applied.)
Memory: each feature receives only the tables its code references (AST), as shallow copies; string
columns it names are converted to object dtype so semantics equal the original object-dtype run."""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
from . import sandbox

FORBIDDEN = ['import os', 'import sys', 'subprocess', 'eval(', 'exec(', '__import__', 'open(',
             'requests', 'urllib', 'TARGET', 'os.', 'sys.']
FAILURE_TYPES = {
    'passed': 'All stages passed', 'security_violation': 'Forbidden pattern / unsafe construct found',
    'code_error': 'Syntax or runtime error (wrong col name / bad logic)',
    'wrong_format': 'Not a pd.Series indexed by SK_ID_CURR', 'high_null': '>95% null - likely hallucinated column',
    'constant': 'Zero variance', 'infinite_values': 'inf or >1e12 values - unguarded division',
    'leakage': 'Spearman(feature,TARGET) > 0.95', 'exact_duplicate': 'Identical values to already-validated feature',
}
TABLE_PARAM = {"bureau": "bureau", "bureau_bal": "bureau_bal", "prev_app": "prev_app", "install": "install",
               "cc_bal": "cc_bal", "pos_cash": "pos_cash"}

def tables_needed(code: str) -> set[str]:
    try:
        _, used = sandbox.check(code)
        return used
    except Exception:
        return set()

def prepare_tables(used: set[str], code: str, app_train: pd.DataFrame, loader, cache: dict) -> dict:
    """Shallow copies of referenced tables; referenced category columns -> object."""
    try:
        consts = sandbox.string_constants(sandbox.check(code)[0])
    except Exception:
        consts = set()
    out = {}
    for p in used:
        if p == "app_train":
            df = app_train
        else:
            if p not in cache:
                cache[p] = loader(p)
            df = cache[p]
        df = df.copy(deep=False)
        for c in df.columns:
            if c in consts and isinstance(df[c].dtype, pd.CategoricalDtype):
                df[c] = df[c].astype(object)
        out[p] = df
    return out

def validate_one(feat_dict, y_series, tables, already_validated=None):
    fname = feat_dict.get('feature_name', 'unnamed')
    code = feat_dict.get('python_code', '')
    group = feat_dict.get('source_group', 'unknown')
    log = {'feature_name': fname, 'source_group': group, 'failure_type': None, 'failure_detail': None,
           'rationale': feat_dict.get('rationale', '')}
    # stage 1: security (original substring list + AST whitelist)
    for pat in FORBIDDEN:
        if pat in code:
            log.update({'failure_type': 'security_violation', 'failure_detail': f'Found: {pat}'})
            return 'failed', log, None
    try:
        sandbox.check(code)
    except sandbox.UnsafeCode as e:
        log.update({'failure_type': 'security_violation', 'failure_detail': f'AST: {e}'})
        return 'failed', log, None
    except SyntaxError as e:
        log.update({'failure_type': 'code_error', 'failure_detail': str(e)[:200]})
        return 'failed', log, None
    # stage 2: execution
    try:
        result = sandbox.run(code, fname, tables)
    except Exception as e:
        log.update({'failure_type': 'code_error', 'failure_detail': str(e)[:200]})
        return 'failed', log, None
    # stage 3
    if not isinstance(result, pd.Series):
        log.update({'failure_type': 'wrong_format', 'failure_detail': f'Got {type(result).__name__}'})
        return 'failed', log, None
    result = result.rename(fname)
    idx_ok = result.index.is_unique and len(result) > 0 and result.index.isin(y_series.index).mean() >= 0.99
    if not idx_ok:
        log.update({'failure_type': 'wrong_format', 'failure_detail': 'index is not unique SK_ID_CURR values'})
        return 'failed', log, None
    # stage 4
    null_rate = result.isna().mean()
    if null_rate > 0.95:
        log.update({'failure_type': 'high_null', 'failure_detail': f'{null_rate*100:.1f}% null'})
        return 'failed', log, None
    # stage 5
    if result.dropna().nunique() <= 1:
        log.update({'failure_type': 'constant', 'failure_detail': 'zero unique values'})
        return 'failed', log, None
    # stage 6
    num = pd.to_numeric(result, errors='coerce')
    if np.isinf(num.dropna()).any() or (num.dropna().abs() > 1e12).any():
        result = result.replace([np.inf, -np.inf], np.nan)
        num = pd.to_numeric(result, errors='coerce')
        if (num.dropna().abs() > 1e12).any():
            log.update({'failure_type': 'infinite_values', 'failure_detail': 'extreme values remain'})
            return 'failed', log, None
    # stage 7: target leakage
    try:
        corr = abs(result.reindex(y_series.index).corr(y_series, method='spearman'))
        if corr > 0.95:
            log.update({'failure_type': 'leakage', 'failure_detail': f'corr={corr:.4f}'})
            return 'failed', log, None
    except Exception:
        pass
    # stage 8
    if already_validated:
        aligned = result.reindex(y_series.index).fillna(-9999)
        for en, es in already_validated.items():
            if aligned.equals(es.reindex(y_series.index).fillna(-9999)):
                log.update({'failure_type': 'exact_duplicate', 'failure_detail': f'dup of {en}'})
                return 'failed', log, None
    log.update({'failure_type': 'passed', 'null_rate': round(float(null_rate), 4),
                'n_unique': int(result.dropna().nunique())})
    return 'passed', log, result

def run_gate(all_features, y_by_id, app_train, loader, log=print):
    """Validate features grouped by required tables (so the big tables are resident one at a time)."""
    order = sorted(range(len(all_features)),
                   key=lambda i: (sorted(tables_needed(all_features[i].get('python_code', '')) - {"app_train"}), i))
    results, cache, current_key = {}, {}, None
    for i in order:
        feat = all_features[i]
        used = tables_needed(feat.get('python_code', ''))
        key = tuple(sorted(used - {"app_train"}))
        if key != current_key:
            for k in list(cache):
                if k not in key:
                    del cache[k]
            gc.collect(); current_key = key
        tables = prepare_tables(used, feat.get('python_code', ''), app_train, loader, cache)
        # stage 8 (duplicates) is replayed below in ORIGINAL proposal order
        results[i] = validate_one(feat, y_by_id, tables, already_validated=None)
        del tables
    cache.clear(); gc.collect()
    # replay stage 8 (exact duplicate) in original proposal order, like the notebook loop
    VAL_LOG, PASSED, FAILED = [], {}, []
    for i, feat in enumerate(all_features):
        status, lg, series = results[i]
        if status == 'passed' and PASSED:
            aligned = series.reindex(y_by_id.index).fillna(-9999)
            for en, es in PASSED.items():
                if aligned.equals(es.reindex(y_by_id.index).fillna(-9999)):
                    status = 'failed'
                    lg = {**lg, 'failure_type': 'exact_duplicate', 'failure_detail': f'dup of {en}'}
                    lg.pop('null_rate', None); lg.pop('n_unique', None)
                    break
        VAL_LOG.append(lg)
        if status == 'passed':
            PASSED[lg['feature_name']] = series
        else:
            FAILED.append(lg)
        log(f"  [{i+1:2d}/{len(all_features)}] [{lg['source_group']:<13}] {lg['feature_name']}: {lg['failure_type']}"
            + (f" ({str(lg['failure_detail'])[:60]})" if lg['failure_type'] != 'passed' else ''))
    return VAL_LOG, PASSED, FAILED
