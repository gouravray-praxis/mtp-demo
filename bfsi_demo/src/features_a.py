"""Pipeline A — exact port of Pipeline_A_BFSI_Baseline.ipynb (2026-06-03).
Cells: 15 preprocess | 17 application ratios | 18 bureau | 19 installments | 21 previous app |
22 credit card + POS | 23 merge | 25 missing/OHE/variance | 26 Spearman collinearity | 29-30 X,y + name cleaning.
Only change: auxiliary tables are processed one at a time (memory) and string columns may be 'category'
(equivalence asserted in src/equivalence.py). Logic/thresholds/ordering are unchanged."""
from __future__ import annotations
import gc, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .common import map_status_num
from . import config as C

TARGET_COL, ID_COL = C.TARGET_COL, C.ID_COL
MISSING_THRESHOLD, CORR_THRESHOLD = 0.80, 0.97

# ---- cell 15
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

# ---- cell 17
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
    enq_cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
    df['FE_ENQUIRY_TOTAL'] = df[enq_cols].fillna(0).sum(axis=1)
    df['FE_ENQUIRY_RECENT_RATIO'] = (df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0) /
                                     (df['FE_ENQUIRY_TOTAL'] + 1))
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

# ---- cell 18
def fe_bureau(bureau_df, bureau_bal_df):
    bureau_bal_df = bureau_bal_df[['SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS']].copy()
    bureau_bal_df['STATUS_NUM'] = map_status_num(bureau_bal_df['STATUS'])
    bb_agg = bureau_bal_df.groupby('SK_ID_BUREAU').agg(
        BB_STATUS_MEAN=('STATUS_NUM', 'mean'), BB_STATUS_MAX=('STATUS_NUM', 'max'),
        BB_DPD_COUNT=('STATUS_NUM', lambda x: (x > 0).sum()),
        BB_MONTHS_COUNT=('MONTHS_BALANCE', 'count')).reset_index()
    del bureau_bal_df; gc.collect()
    bureau_df = bureau_df.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    num_agg = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'std'], 'CREDIT_DAY_OVERDUE': ['max', 'mean', 'sum'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'], 'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'], 'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'], 'AMT_CREDIT_SUM_LIMIT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'], 'DAYS_CREDIT_UPDATE': ['mean', 'min'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'], 'BB_STATUS_MEAN': ['mean', 'max'],
        'BB_STATUS_MAX': ['max'], 'BB_DPD_COUNT': ['sum', 'mean'], 'BB_MONTHS_COUNT': ['sum', 'mean'],
    }
    bureau_df['IS_ACTIVE'] = (bureau_df['CREDIT_ACTIVE'] == 'Active').astype(int)
    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg(
        **{f'BUREAU_{col}_{stat.upper()}': (col, stat)
           for col, stats_list in num_agg.items() for stat in stats_list}).reset_index()
    bureau_counts = bureau_df.groupby('SK_ID_CURR').agg(
        BUREAU_TOTAL_LOANS=('SK_ID_BUREAU', 'count'), BUREAU_ACTIVE_LOANS=('IS_ACTIVE', 'sum'),
        BUREAU_CREDIT_TYPES=('CREDIT_TYPE', 'nunique'), BUREAU_IS_ACTIVE_MEAN=('IS_ACTIVE', 'mean'),
    ).reset_index()
    bureau_agg = bureau_agg.merge(bureau_counts, on='SK_ID_CURR', how='left')
    bureau_agg['BUREAU_UTILIZATION'] = (bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] /
                                        (bureau_agg['BUREAU_AMT_CREDIT_SUM_LIMIT_SUM'] + 1))
    bureau_agg['BUREAU_OVERDUE_RATE'] = (bureau_agg['BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM'] /
                                         (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1))
    bureau_agg['BUREAU_ACTIVE_RATIO'] = (bureau_agg['BUREAU_ACTIVE_LOANS'] /
                                         (bureau_agg['BUREAU_TOTAL_LOANS'] + 1))
    return bureau_agg

# ---- cell 19
def fe_installments(install_df):
    df = install_df[['SK_ID_PREV', 'SK_ID_CURR', 'AMT_PAYMENT', 'AMT_INSTALMENT',
                     'DAYS_ENTRY_PAYMENT', 'DAYS_INSTALMENT']].copy()
    df['PAYMENT_RATIO'] = df['AMT_PAYMENT'] / (df['AMT_INSTALMENT'] + 1)
    df['PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    df['DAYS_LATE'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['IS_LATE'] = (df['DAYS_LATE'] > 0).astype(int)
    df['IS_UNDERPAID'] = (df['PAYMENT_DIFF'] > 0).astype(int)
    agg = df.groupby('SK_ID_CURR').agg(
        INSTAL_COUNT=('SK_ID_PREV', 'count'), INSTAL_PAYMENT_RATIO_MEAN=('PAYMENT_RATIO', 'mean'),
        INSTAL_PAYMENT_RATIO_MIN=('PAYMENT_RATIO', 'min'), INSTAL_PAYMENT_DIFF_MEAN=('PAYMENT_DIFF', 'mean'),
        INSTAL_PAYMENT_DIFF_MAX=('PAYMENT_DIFF', 'max'), INSTAL_PAYMENT_DIFF_SUM=('PAYMENT_DIFF', 'sum'),
        INSTAL_DAYS_LATE_MEAN=('DAYS_LATE', 'mean'), INSTAL_DAYS_LATE_MAX=('DAYS_LATE', 'max'),
        INSTAL_DAYS_LATE_STD=('DAYS_LATE', 'std'), INSTAL_LATE_COUNT=('IS_LATE', 'sum'),
        INSTAL_LATE_RATE=('IS_LATE', 'mean'), INSTAL_UNDERPAID_RATE=('IS_UNDERPAID', 'mean'),
        INSTAL_AMT_PAID_SUM=('AMT_PAYMENT', 'sum'), INSTAL_AMT_INSTALMENT_SUM=('AMT_INSTALMENT', 'sum'),
        INSTAL_UNIQUE_LOANS=('SK_ID_PREV', 'nunique'),
    ).reset_index()
    agg['INSTAL_TOTAL_COVERAGE'] = agg['INSTAL_AMT_PAID_SUM'] / (agg['INSTAL_AMT_INSTALMENT_SUM'] + 1)
    return agg

# ---- cell 21
def fe_previous_application(prev_df):
    df = prev_df.copy()
    df['IS_APPROVED'] = (df['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    df['IS_REFUSED'] = (df['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    df['IS_CANCELLED'] = (df['NAME_CONTRACT_STATUS'] == 'Canceled').astype(int)
    df['DAYS_DECISION'] = df['DAYS_DECISION'].abs()
    df['AMT_APPLIED_TO_CREDIT'] = df['AMT_APPLICATION'] / (df['AMT_CREDIT'] + 1)
    df['AMT_DOWN_RATE'] = df['AMT_DOWN_PAYMENT'] / (df['AMT_APPLICATION'] + 1)
    agg_dict = {
        'PREV_COUNT': ('SK_ID_PREV', 'count'), 'PREV_APPROVED_COUNT': ('IS_APPROVED', 'sum'),
        'PREV_REFUSED_COUNT': ('IS_REFUSED', 'sum'), 'PREV_CANCELLED_COUNT': ('IS_CANCELLED', 'sum'),
        'PREV_APPROVAL_RATE': ('IS_APPROVED', 'mean'), 'PREV_DAYS_DEC_MIN': ('DAYS_DECISION', 'min'),
        'PREV_DAYS_DEC_MEAN': ('DAYS_DECISION', 'mean'), 'PREV_AMT_CREDIT_MEAN': ('AMT_CREDIT', 'mean'),
        'PREV_AMT_CREDIT_MAX': ('AMT_CREDIT', 'max'), 'PREV_AMT_ANNUITY_MEAN': ('AMT_ANNUITY', 'mean'),
        'PREV_AMT_ANNUITY_MAX': ('AMT_ANNUITY', 'max'), 'PREV_RATE_DOWN_MEAN': ('RATE_DOWN_PAYMENT', 'mean'),
        'PREV_AMT_APPLIED_CREDIT': ('AMT_APPLIED_TO_CREDIT', 'mean'), 'PREV_DOWN_RATE': ('AMT_DOWN_RATE', 'mean'),
        'PREV_UNIQUE_TYPES': ('NAME_CONTRACT_TYPE', 'nunique'),
    }
    if 'RATE_INTEREST_PRIMARY' in df.columns:
        agg_dict['PREV_RATE_INTEREST_MEAN'] = ('RATE_INTEREST_PRIMARY', 'mean')
        agg_dict['PREV_INTEREST_MAX'] = ('RATE_INTEREST_PRIMARY', 'max')
    elif 'RATE_INTEREST_RECD' in df.columns:
        agg_dict['PREV_RATE_INTEREST_MEAN'] = ('RATE_INTEREST_RECD', 'mean')
        agg_dict['PREV_INTEREST_MAX'] = ('RATE_INTEREST_RECD', 'max')
    if 'RATE_INTEREST_PRIVILEGED' in df.columns:
        agg_dict['PREV_RATE_PRIVILEGED_MEAN'] = ('RATE_INTEREST_PRIVILEGED', 'mean')
    agg = df.groupby('SK_ID_CURR').agg(**agg_dict).reset_index()
    agg['PREV_REFUSAL_RATE'] = agg['PREV_REFUSED_COUNT'] / (agg['PREV_COUNT'] + 1)
    return agg

# ---- cell 22
def fe_credit_card(cc_df):
    df = cc_df.copy()
    df['CC_UTILIZATION'] = df['AMT_BALANCE'] / (df['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    df['CC_PAYMENT_RATIO'] = df['AMT_PAYMENT_CURRENT'] / (df['AMT_INST_MIN_REGULARITY'] + 1)
    df['CC_DRAWING_RATIO'] = df['AMT_DRAWINGS_CURRENT'] / (df['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    df['IS_OVERDRAWN'] = (df['AMT_BALANCE'] > df['AMT_CREDIT_LIMIT_ACTUAL']).astype(int)
    return df.groupby('SK_ID_CURR').agg(
        CC_COUNT=('SK_ID_PREV', 'count'), CC_BALANCE_MEAN=('AMT_BALANCE', 'mean'),
        CC_BALANCE_MAX=('AMT_BALANCE', 'max'), CC_UTILIZATION_MEAN=('CC_UTILIZATION', 'mean'),
        CC_UTILIZATION_MAX=('CC_UTILIZATION', 'max'), CC_PAYMENT_RATIO_MEAN=('CC_PAYMENT_RATIO', 'mean'),
        CC_PAYMENT_RATIO_MIN=('CC_PAYMENT_RATIO', 'min'), CC_DRAWING_RATIO_MEAN=('CC_DRAWING_RATIO', 'mean'),
        CC_OVERDRAWN_RATE=('IS_OVERDRAWN', 'mean'), CC_OVERDRAWN_COUNT=('IS_OVERDRAWN', 'sum'),
        CC_DPD_DEF_MEAN=('SK_DPD_DEF', 'mean'), CC_DPD_DEF_MAX=('SK_DPD_DEF', 'max'),
        CC_DPD_MEAN=('SK_DPD', 'mean'), CC_TOTAL_DRAWINGS=('CNT_DRAWINGS_CURRENT', 'sum'),
    ).reset_index()

def fe_pos_cash(pos_df):
    df = pos_df.copy()
    df['IS_DPD'] = (df['SK_DPD'] > 0).astype(int)
    return df.groupby('SK_ID_CURR').agg(
        POS_COUNT=('SK_ID_PREV', 'count'), POS_MONTHS_BALANCE_MEAN=('MONTHS_BALANCE', 'mean'),
        POS_CNT_INSTALMENT_MEAN=('CNT_INSTALMENT', 'mean'), POS_DPD_MEAN=('SK_DPD', 'mean'),
        POS_DPD_MAX=('SK_DPD', 'max'), POS_DPD_DEF_MEAN=('SK_DPD_DEF', 'mean'),
        POS_DPD_DEF_MAX=('SK_DPD_DEF', 'max'), POS_DPD_RATE=('IS_DPD', 'mean'),
        POS_DPD_COUNT=('IS_DPD', 'sum'),
        POS_ACTIVE_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Active').sum()),
    ).reset_index()

AUX_BUILDERS = {  # order = cell 23 merge order
    'bureau': (('bureau', 'bureau_bal'), fe_bureau),
    'install': (('install',), fe_installments),
    'prev': (('prev_app',), fe_previous_application),
    'cc': (('cc_bal',), fe_credit_card),
    'pos': (('pos_cash',), fe_pos_cash),
}

def build_aux_blocks(loader):
    """loader(name)->DataFrame. Loads each table only while its block is built."""
    blocks = {}
    for key, (tables, fn) in AUX_BUILDERS.items():
        dfs = [loader(t) for t in tables]
        blocks[key] = fn(*dfs)
        del dfs; gc.collect()
    return blocks

# ---- cell 23
def merge_features(app_df, blocks):
    df = app_df.copy()
    for key in ('bureau', 'install', 'prev', 'cc', 'pos'):
        df = df.merge(blocks[key], on='SK_ID_CURR', how='left')
    return df

# ---- cell 30
def clean_feature_names(df):
    new_cols = {}
    for col in df.columns:
        new_col = re.sub(r'[^A-Za-z0-9_]', '_', col)
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        new_cols[col] = new_col
    return df.rename(columns=new_cols)

def select_and_encode(train_df, seed=C.RANDOM_SEED, log=print):
    """cells 25-26 (train only; test set handling dropped — not a deliverable)."""
    miss_rate = train_df.isnull().mean()
    drop_high_miss = [c for c in miss_rate[miss_rate > MISSING_THRESHOLD].index.tolist()
                      if c not in [TARGET_COL, ID_COL]]
    train_df = train_df.drop(columns=drop_high_miss)
    cat_cols = [c for c in train_df.select_dtypes('object').columns.tolist() if c != TARGET_COL]
    high_card = [c for c in cat_cols if train_df[c].nunique() > 50]
    train_df = train_df.drop(columns=high_card)
    cat_cols = [c for c in cat_cols if c not in high_card]
    cat_levels = {c: sorted(train_df[c].dropna().unique().tolist()) for c in cat_cols}
    train_df = pd.get_dummies(train_df, columns=cat_cols, dummy_na=True, drop_first=True)
    feat_cols = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]
    variances = train_df[feat_cols].var()
    drop_zero_var = variances[variances < 1e-6].index.tolist()
    train_df = train_df.drop(columns=drop_zero_var)
    feat_cols = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]
    n_after_var = len(feat_cols)
    sample_size = min(50000, len(train_df))
    corr_sample = train_df[feat_cols].sample(n=sample_size, random_state=seed)
    corr_matrix = corr_sample.corr(method='spearman').abs()
    del corr_sample
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_corr = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
    del corr_matrix, upper; gc.collect()
    train_df = train_df.drop(columns=drop_corr)
    feat_cols = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]
    log(f"  A selection: miss_drop={len(drop_high_miss)} high_card={high_card} "
        f"zero_var={len(drop_zero_var)} after_var={n_after_var} corr_drop={len(drop_corr)} final={len(feat_cols)}")
    registry = {'pipeline': 'A', 'total_features': len(feat_cols), 'feature_names': feat_cols,
                'dropped_high_missing': drop_high_miss, 'dropped_high_cardinality': high_card,
                'dropped_zero_variance': drop_zero_var, 'dropped_collinear': drop_corr,
                'ohe_levels': cat_levels}
    return train_df, feat_cols, registry

def build_matrix_A(app_train_raw, loader, log=print):
    app = fe_application_ratios(preprocess_application(app_train_raw))
    blocks = build_aux_blocks(loader)
    train_df = merge_features(app, blocks)
    del app, blocks; gc.collect()
    log(f"  A merged shape: {train_df.shape}")
    train_df, feat_cols, registry = select_and_encode(train_df, log=log)
    X = train_df[feat_cols].copy()
    y = train_df[TARGET_COL].copy()
    ids = train_df[ID_COL].copy()
    del train_df; gc.collect()
    bool_cols = X.select_dtypes('bool').columns
    X[bool_cols] = X[bool_cols].astype(int)
    X = clean_feature_names(X)
    registry['feature_names_clean'] = X.columns.tolist()
    return X, y, ids, registry
