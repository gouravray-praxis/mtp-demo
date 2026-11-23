"""Pipeline B/C — exact port of 'Pipeline_B_BFSI_FINAL_CORRECTED (3).ipynb' (2026-06-16).
Cells: 10 sanitize | 12 TRUE shared base (mechanical only) | 32 human FE (A' block, used ONLY by C) |
33 select_features (missing>85% + NZV 1%, NO Spearman collinearity filter) + B/C matrix assembly.
Memory change only: aux tables processed one at a time; string columns may be 'category'."""
from __future__ import annotations
import gc, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .common import map_status_num
from . import config as C

TARGET_COL, ID_COL = C.TARGET_COL, C.ID_COL
AUX_PREFIXES = ('BUREAU_', 'INSTAL_', 'PREV_', 'CC_', 'POS_')

# ---- cell 10
def sanitize_column_names(df, verbose=False):
    df = df.copy()
    new_cols = []
    for c in df.columns.tolist():
        safe = re.sub(r'[^0-9a-zA-Z_]+', '_', str(c))
        safe = re.sub(r'_+', '_', safe).strip('_')
        new_cols.append(safe if safe else 'col_' + str(hash(c))[:8])
    seen, final_cols = {}, []
    for c in new_cols:
        if c in seen:
            seen[c] += 1; final_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0; final_cols.append(c)
    df.columns = final_cols
    return df

# ---- cell 12
def preprocess_application(df):
    df = df.copy()
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    for col in df.select_dtypes('object').columns:
        df[col] = df[col].replace({'XNA': np.nan, 'XAP': np.nan})
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
    df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25
    df['REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365.25
    df['ID_PUBLISH_YEARS'] = -df['DAYS_ID_PUBLISH'] / 365.25
    df['LAST_PHONE_CHANGE_YEARS'] = -df['DAYS_LAST_PHONE_CHANGE'] / 365.25
    binary_cats = [c for c in df.select_dtypes('object').columns if df[c].nunique() <= 2]
    for col in binary_cats:
        le = LabelEncoder()
        mask = df[col].notna()
        df.loc[mask, col] = le.fit_transform(df.loc[mask, col])
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def build_shared_base(app_train_raw):
    """Returns (app_proc_train [OHE, unsanitized, incl. TARGET — exactly what the notebook passes to
    LLM code and to the human FE], BASE_TRAIN [sanitized, incl. SK_ID_CURR, no TARGET], y, meta)."""
    app_proc_train = preprocess_application(app_train_raw)
    y = app_proc_train[TARGET_COL].copy()
    cat_cols = [c for c in app_proc_train.select_dtypes('object').columns.tolist() if c not in [TARGET_COL, ID_COL]]
    high_card = [c for c in cat_cols if app_proc_train[c].nunique() > 50]
    app_proc_train = app_proc_train.drop(columns=high_card)
    cat_cols = [c for c in cat_cols if c not in high_card]
    cat_levels = {c: sorted(app_proc_train[c].dropna().unique().tolist()) for c in cat_cols}
    app_proc_train = pd.get_dummies(app_proc_train, columns=cat_cols, dummy_na=True, drop_first=True)
    BASE_TRAIN = sanitize_column_names(app_proc_train.drop(columns=[TARGET_COL]))
    fe_in_base = [c for c in BASE_TRAIN.columns if c.startswith('FE_')]
    assert len(fe_in_base) == 0, f'ERROR: FE_ in base: {fe_in_base}'
    return app_proc_train, BASE_TRAIN, y, {'high_card': high_card, 'ohe_levels': cat_levels,
                                           'base_shape': list(BASE_TRAIN.shape)}

# ---- cell 32 (human FE A' — used only for Pipeline C)
def fe_application_ratios(df):
    df = df.copy()
    df['FE_ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['FE_CREDIT_TO_INCOME'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['FE_CREDIT_TO_GOODS'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
    df['FE_ANNUITY_TO_CREDIT'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
    df['FE_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    df['FE_GOODS_PRICE_GAP'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    df['FE_LOAN_TENOR_MONTHS'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
    df['FE_EMPLOYMENT_TO_AGE'] = df['EMPLOYMENT_YEARS'] / (df['AGE_YEARS'] + 1)
    df['FE_WORKING_SINCE_AGE'] = df['AGE_YEARS'] - df['EMPLOYMENT_YEARS']
    doc_cols = [c for c in df.columns if 'FLAG_DOCUMENT' in c]
    df['FE_DOCS_SUBMITTED'] = df[doc_cols].sum(axis=1)
    df['FE_SOCIAL_CIRCLE_DEFAULT'] = (df['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(0) +
                                      df['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(0))
    enq = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
           'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
    df['FE_ENQUIRY_TOTAL'] = df[enq].fillna(0).sum(axis=1)
    df['FE_ENQUIRY_RECENT_RATIO'] = df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0) / (df['FE_ENQUIRY_TOTAL'] + 1)
    df['FE_EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['FE_EXT_SOURCE_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    df['FE_EXT_SOURCE_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    df['FE_EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['FE_EXT_SOURCE_PRODUCT'] = (df['EXT_SOURCE_1'].fillna(df['FE_EXT_SOURCE_MEAN']) *
                                   df['EXT_SOURCE_2'].fillna(df['FE_EXT_SOURCE_MEAN']) *
                                   df['EXT_SOURCE_3'].fillna(df['FE_EXT_SOURCE_MEAN']))
    df['FE_EXT_MEAN_ANNUITY'] = df['FE_EXT_SOURCE_MEAN'] * df['AMT_ANNUITY']
    df['FE_EXT_MEAN_CREDIT'] = df['FE_EXT_SOURCE_MEAN'] * df['AMT_CREDIT']
    return df

def fe_bureau(bureau_df, bureau_bal_df):
    bb = bureau_bal_df[['SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS']].copy()
    bb['STATUS_NUM'] = map_status_num(bb['STATUS'])
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(
        BB_STATUS_MEAN=('STATUS_NUM', 'mean'), BB_STATUS_MAX=('STATUS_NUM', 'max'),
        BB_DPD_COUNT=('STATUS_NUM', lambda x: (x > 0).sum()),
        BB_MONTHS_COUNT=('MONTHS_BALANCE', 'count')).reset_index()
    del bb; gc.collect()
    bdf = bureau_df.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    agg = bdf.groupby('SK_ID_CURR').agg(
        LOAN_COUNT=('SK_ID_BUREAU', 'count'),
        ACTIVE_COUNT=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
        AMT_CREDIT_MEAN=('AMT_CREDIT_SUM', 'mean'), AMT_CREDIT_MAX=('AMT_CREDIT_SUM', 'max'),
        AMT_DEBT_MEAN=('AMT_CREDIT_SUM_DEBT', 'mean'), OVERDUE_MEAN=('AMT_CREDIT_SUM_OVERDUE', 'mean'),
        DAYS_CREDIT_MEAN=('DAYS_CREDIT', 'mean'), DAYS_CREDIT_MIN=('DAYS_CREDIT', 'min'),
        CNT_PROLONG_SUM=('CNT_CREDIT_PROLONG', 'sum'), BB_STATUS_MEAN=('BB_STATUS_MEAN', 'mean'),
        BB_DPD_SUM=('BB_DPD_COUNT', 'sum'),
        ACTIVE_RATIO=('CREDIT_ACTIVE', lambda x: (x == 'Active').mean()),
        DAYS_UPDATE_MIN=('DAYS_CREDIT_UPDATE', 'min'), CREDIT_TYPE_N=('CREDIT_TYPE', 'nunique'),
        BB_STATUS_MAX=('BB_STATUS_MAX', 'max'), ENDDATE_MEAN=('DAYS_CREDIT_ENDDATE', 'mean'),
        BB_MONTHS_MEAN=('BB_MONTHS_COUNT', 'mean'), AMT_CREDIT_LIMIT_MEAN=('AMT_CREDIT_SUM_LIMIT', 'mean'),
        DAYS_UPDATE_MEAN=('DAYS_CREDIT_UPDATE', 'mean'),
        CLOSED_COUNT=('CREDIT_ACTIVE', lambda x: (x == 'Closed').sum()),
    ).reset_index()
    agg.columns = ['SK_ID_CURR'] + [f'BUREAU_{c}' for c in agg.columns[1:]]
    return agg

def fe_installments(install_df):
    df = install_df[['SK_ID_PREV', 'SK_ID_CURR', 'AMT_PAYMENT', 'AMT_INSTALMENT', 'DAYS_ENTRY_PAYMENT',
                     'DAYS_INSTALMENT', 'NUM_INSTALMENT_NUMBER']].copy()
    df['PAY_RATIO'] = df['AMT_PAYMENT'] / (df['AMT_INSTALMENT'] + 1)
    df['PAY_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    df['DPD'] = (df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']).clip(lower=0)
    df['DBD'] = (df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']).clip(lower=0)
    agg = df.groupby('SK_ID_CURR').agg(
        PAY_MEAN=('AMT_PAYMENT', 'mean'), PAY_SUM=('AMT_PAYMENT', 'sum'),
        INST_MEAN=('AMT_INSTALMENT', 'mean'), PAY_RATIO_MEAN=('PAY_RATIO', 'mean'),
        PAY_RATIO_MIN=('PAY_RATIO', 'min'), PAY_DIFF_MEAN=('PAY_DIFF', 'mean'),
        DPD_MEAN=('DPD', 'mean'), DPD_MAX=('DPD', 'max'), DPD_SUM=('DPD', 'sum'),
        DBD_MEAN=('DBD', 'mean'), COUNT=('SK_ID_CURR', 'count'),
        LATE_COUNT=('DPD', lambda x: (x > 0).sum()), LATE_RATIO=('DPD', lambda x: (x > 0).mean()),
        ENTRY_MEAN=('DAYS_ENTRY_PAYMENT', 'mean'), NUM_INST_N=('NUM_INSTALMENT_NUMBER', 'nunique'),
        PREV_N=('SK_ID_PREV', 'nunique'),
    ).reset_index()
    agg.columns = ['SK_ID_CURR'] + [f'INSTAL_{c}' for c in agg.columns[1:]]
    return agg

def fe_prev_app(prev_df):
    df = prev_df.copy()
    df['CREDIT_PERC'] = df['AMT_APPLICATION'] / (df['AMT_CREDIT'] + 1)
    df['DOWN_TO_CREDIT'] = df['AMT_DOWN_PAYMENT'] / (df['AMT_CREDIT'] + 1)
    agg = df.groupby('SK_ID_CURR').agg(
        COUNT=('SK_ID_PREV', 'count'),
        APPROVED=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
        REFUSED=('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
        CANCELED=('NAME_CONTRACT_STATUS', lambda x: (x == 'Canceled').sum()),
        APPROVED_RATIO=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').mean()),
        AMT_CREDIT_MEAN=('AMT_CREDIT', 'mean'), AMT_CREDIT_MAX=('AMT_CREDIT', 'max'),
        AMT_APP_MEAN=('AMT_APPLICATION', 'mean'), DOWN_MEAN=('AMT_DOWN_PAYMENT', 'mean'),
        CREDIT_PERC_MEAN=('CREDIT_PERC', 'mean'), CREDIT_PERC_MIN=('CREDIT_PERC', 'min'),
        DAYS_DEC_MEAN=('DAYS_DECISION', 'mean'), DAYS_DEC_MIN=('DAYS_DECISION', 'min'),
        RATE_MEAN=('RATE_DOWN_PAYMENT', 'mean'), LAST_DUE_MEAN=('DAYS_LAST_DUE', 'mean'),
        CNT_PAY_MEAN=('CNT_PAYMENT', 'mean'), CONTRACT_N=('NAME_CONTRACT_TYPE', 'nunique'),
        DOWN_TO_CREDIT_MEAN=('DOWN_TO_CREDIT', 'mean'), HOUR_MEAN=('HOUR_APPR_PROCESS_START', 'mean'),
    ).reset_index()
    agg.columns = ['SK_ID_CURR'] + [f'PREV_{c}' for c in agg.columns[1:]]
    return agg

def fe_cc_balance(cc_df):
    df = cc_df.copy()
    df['UTIL'] = df['AMT_BALANCE'] / (df['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    df['ATM_RATIO'] = df['AMT_DRAWINGS_ATM_CURRENT'] / (df['AMT_DRAWINGS_CURRENT'] + 1)
    agg = df.groupby('SK_ID_CURR').agg(
        BAL_MEAN=('AMT_BALANCE', 'mean'), BAL_MAX=('AMT_BALANCE', 'max'),
        LIMIT_MEAN=('AMT_CREDIT_LIMIT_ACTUAL', 'mean'), UTIL_MEAN=('UTIL', 'mean'), UTIL_MAX=('UTIL', 'max'),
        ATM_RATIO_MEAN=('ATM_RATIO', 'mean'), PAY_TOTAL_MEAN=('AMT_PAYMENT_TOTAL_CURRENT', 'mean'),
        DRAW_CNT_MEAN=('CNT_DRAWINGS_CURRENT', 'mean'), INST_CUM_MEAN=('CNT_INSTALMENT_MATURE_CUM', 'mean'),
        DPD_MEAN=('SK_DPD', 'mean'), DPD_DEF_MEAN=('SK_DPD_DEF', 'mean'), DPD_MAX=('SK_DPD', 'max'),
        MONTHS_CNT=('MONTHS_BALANCE', 'count'), PREV_N=('SK_ID_PREV', 'nunique'),
    ).reset_index()
    agg.columns = ['SK_ID_CURR'] + [f'CC_{c}' for c in agg.columns[1:]]
    return agg

def fe_pos_cash(pos_df):
    agg = pos_df.groupby('SK_ID_CURR').agg(
        DPD_MEAN=('SK_DPD', 'mean'), DPD_MAX=('SK_DPD', 'max'), DPD_DEF_MEAN=('SK_DPD_DEF', 'mean'),
        INST_MEAN=('CNT_INSTALMENT', 'mean'), INST_FUTURE_MEAN=('CNT_INSTALMENT_FUTURE', 'mean'),
        COMPLETED=('NAME_CONTRACT_STATUS', lambda x: (x == 'Completed').sum()),
        ACTIVE=('NAME_CONTRACT_STATUS', lambda x: (x == 'Active').sum()),
        COMPLETED_RATIO=('NAME_CONTRACT_STATUS', lambda x: (x == 'Completed').mean()),
        MONTHS_CNT=('MONTHS_BALANCE', 'count'), PREV_N=('SK_ID_PREV', 'nunique'),
    ).reset_index()
    agg.columns = ['SK_ID_CURR'] + [f'POS_{c}' for c in agg.columns[1:]]
    return agg

AUX_BUILDERS = {  # cell 33 merge order: bur, ins, prv, ccd, pos
    'bureau': (('bureau', 'bureau_bal'), fe_bureau), 'install': (('install',), fe_installments),
    'prev': (('prev_app',), fe_prev_app), 'cc': (('cc_bal',), fe_cc_balance), 'pos': (('pos_cash',), fe_pos_cash),
}

def build_human_aux(loader):
    blocks = {}
    for key, (tables, fn) in AUX_BUILDERS.items():
        dfs = [loader(t) for t in tables]
        blocks[key] = fn(*dfs)
        del dfs; gc.collect()
    return blocks

# ---- cell 33
def select_features(df, target_col=None, id_col=ID_COL, missing_thresh=0.85, nzv_thresh=0.01, log=print, tag=''):
    drop = [id_col]
    if target_col and target_col in df.columns:
        drop.append(target_col)
    feat = df.drop(columns=[c for c in drop if c in df.columns])
    miss_rate = feat.isnull().mean()
    miss_drop = miss_rate[miss_rate > missing_thresh].index.tolist()
    feat = feat.drop(columns=miss_drop)
    nzv = []
    for col in feat.select_dtypes(include=[np.number]).columns:
        try:
            if feat[col].value_counts(normalize=True).iloc[0] > (1 - nzv_thresh):
                nzv.append(col)
        except Exception:
            pass
    feat = feat.drop(columns=nzv)
    log(f'  {tag} feature sel: start={df.shape[1]-len(drop)} miss_drop={len(miss_drop)} nzv_drop={len(nzv)} final={feat.shape[1]}')
    return feat, feat.columns.tolist(), {'miss_drop': miss_drop, 'nzv_drop': nzv}

def build_human_df(app_proc_train, loader):
    app_with_fe = fe_application_ratios(app_proc_train)
    blocks = build_human_aux(loader)
    df_human = app_with_fe
    for key in ('bureau', 'install', 'prev', 'cc', 'pos'):
        df_human = df_human.merge(blocks[key], on=ID_COL, how='left')
    del blocks; gc.collect()
    return sanitize_column_names(df_human)

def build_matrix_B(BASE_TRAIN, passed_features, log=print):
    df_B = BASE_TRAIN.copy()
    for fname, series in passed_features.items():
        df_B[fname] = series.reindex(df_B[ID_COL]).values
    fe_in_B = [c for c in df_B.columns if c.startswith('FE_')]
    hum_in_B = [c for c in df_B.columns if c.startswith(AUX_PREFIXES)]
    assert len(fe_in_B) == 0, f'ERROR: FE_ in Pipeline B: {fe_in_B}'
    assert len(hum_in_B) == 0, f'ERROR: Human aux in Pipeline B: {hum_in_B}'
    ids = df_B[ID_COL].copy()
    X_B, feat_B, sel = select_features(df_B, target_col=None, log=log, tag='B')
    return X_B, feat_B, ids, sel

def build_matrix_C(df_human, passed_features, log=print):
    df_C = df_human
    for fname, series in passed_features.items():
        if fname not in df_C.columns:
            df_C[fname] = series.reindex(df_C[ID_COL]).values
    ids = df_C[ID_COL].copy()
    X_C, feat_C, sel = select_features(df_C, target_col=TARGET_COL, log=log, tag='C')
    return X_C, feat_C, ids, sel
