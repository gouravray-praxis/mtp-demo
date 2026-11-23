"""Groq LLM client for Pipeline B/C (6 focused calls). Prompts are verbatim from the B/C notebook cells 16-22.
Model chosen at run time from the *authenticated* model list. Key never printed/logged/saved."""
from __future__ import annotations
import hashlib, json, re, time
from pathlib import Path
from json_repair import repair_json
from . import config as C

PREFERRED = ["qwen/qwen3.8-27b"]
FALLBACK = "openai/gpt-oss-120b"
LLM_TEMP, LLM_MAX_TOKENS = 0, 4096  # free-tier TPM counts prompt+max_completion; real completions ~1.2-1.5k (original 8192)

SYSTEM_PROMPT = """You are a senior BFSI data scientist specializing in LightGBM credit risk feature engineering.
You must respond ONLY with valid JSON - no markdown, no preamble, no text outside JSON."""

CODE_PATTERN = """
MANDATORY CODE PATTERN - follow EXACTLY, no deviations:

# Pattern 1: For auxiliary table features (groupby + merge to SK_ID_CURR)
def compute_FEATURENAME(app_train, bureau, bureau_bal, prev_app, install, cc_bal, pos_cash):
    agg = bureau.groupby('SK_ID_CURR').agg(val=('AMT_CREDIT_SUM','mean')).reset_index()
    result = app_train[['SK_ID_CURR']].merge(agg, on='SK_ID_CURR', how='left')
    return result.set_index('SK_ID_CURR')['val'].rename('feature_name')

# Pattern 2: For bureau_balance (join bureau first to get SK_ID_CURR)
def compute_FEATURENAME(app_train, bureau, bureau_bal, prev_app, install, cc_bal, pos_cash):
    bb  = bureau_bal.merge(bureau[['SK_ID_BUREAU','SK_ID_CURR']], on='SK_ID_BUREAU', how='left')
    agg = bb.groupby('SK_ID_CURR').agg(val=('STATUS','count')).reset_index()
    result = app_train[['SK_ID_CURR']].merge(agg, on='SK_ID_CURR', how='left')
    return result.set_index('SK_ID_CURR')['val'].rename('feature_name')

# Pattern 3: For application_train (direct row-level computation)
def compute_FEATURENAME(app_train, bureau, bureau_bal, prev_app, install, cc_bal, pos_cash):
    result = app_train[['SK_ID_CURR']].copy()
    result['feature_name'] = app_train['AMT_CREDIT'] / (app_train['AMT_INCOME_TOTAL'] + 1)
    return result.set_index('SK_ID_CURR')['feature_name']
"""

def get_schema(df, table_name, join_key, col_desc_df=None, sample_rows=3):
    """cell 14 verbatim; category dtype reported as 'object' so the prompt matches the original."""
    lines = [f'TABLE: {table_name}', f'  join_key: {join_key} | rows: {df.shape[0]:,} | cols: {df.shape[1]}',
             '  columns (name | dtype | samples | description):']
    for col in df.columns:
        if col == join_key:
            continue
        dtype = str(df[col].dtype)
        if dtype == 'category':
            dtype = 'object'
        sample = str(df[col].dropna().head(sample_rows).tolist())[:60]
        desc = ''
        if col_desc_df is not None:
            m = col_desc_df[col_desc_df['Row'].astype(str).str.strip() == col]
            if len(m) > 0:
                desc = str(m.iloc[0].get('Description', '')).strip()[:90]
        lines.append(f'    {col} ({dtype}) | {sample} | {desc}' if desc else f'    {col} ({dtype}) | {sample}')
    return '\n'.join(lines)

def build_table_prompt(group_name, schemas_dict, n_features, avoid_human_features=None, avoid_existing_llm=None):
    schema_text = '\n\n'.join(schemas_dict.values())
    avoid_text = ''
    if avoid_human_features:
        avoid_text = 'DO NOT recreate these human-designed features:\n'
        avoid_text += '\n'.join([f'  - {f}' for f in avoid_human_features[:25]])
    existing_text = ''
    if avoid_existing_llm:
        existing_text = 'LLM features already generated (no exact duplicates):\n'
        existing_text += '\n'.join([f'  - {f}' for f in avoid_existing_llm])
    return f"""TASK: Generate {n_features} customer-level features for BFSI credit default prediction.
TARGET: Binary - 1=default (8.1% positive rate), 0=no default
FOCUS: {group_name}

=== TABLE SCHEMAS ===
{schema_text}

{CODE_PATTERN}

{avoid_text}

{existing_text}

REQUIREMENTS:
1. Generate EXACTLY {n_features} features, all from the specified tables
2. Every feature aggregated to SK_ID_CURR level
3. Use ONLY column names as shown in schemas - no hallucination
4. Handle NaN: use .fillna(0) or how='left' merge
5. Guard all divisions: divide by (denominator + 1) or use np.where
6. Always return pd.Series indexed by SK_ID_CURR (follow patterns above)
7. feature_name: snake_case only, max 50 chars

Return ONLY this JSON - no markdown, no preamble:
{{"features": [
  {{"feature_name": "snake_case_name",
    "rationale": "BFSI credit risk rationale in one sentence",
    "source_tables": ["table_name"],
    "python_code": "def compute_FEATURENAME(app_train, bureau, bureau_bal, prev_app, install, cc_bal, pos_cash): ..."
  }}
]}}"""

# cells 17-22: (label, group description, schema keys, n_features, avoid prefix, source_group)
CALLS = [
    ("call1_app", "application_train (single-table row-level features)", ["application_train"], 6, "FE_", "application"),
    ("call2_bureau", "bureau + bureau_balance (external credit history)", ["bureau", "bureau_balance"], 6, "BUREAU_", "bureau"),
    ("call3_instal", "installments_payments (payment behavior history)", ["installments_payments"], 5, "INSTAL_", "installments"),
    ("call4_prev", "previous_application (prior loan application history)", ["previous_application"], 5, "PREV_", "previous_app"),
    ("call5_cc", "credit_card_balance (revolving credit utilization)", ["credit_card_balance"], 5, "CC_", "credit_card"),
    ("call6_pos", "POS_CASH_balance (point-of-sale and cash loan history)", ["POS_CASH_balance"], 5, "POS_", "pos_cash"),
]

class LLMClient:
    def __init__(self, log_dir: Path = C.LLM_DIR, log=print):
        from groq import Groq
        key = C.get_secret("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not available")
        self.client = Groq(api_key=key, max_retries=0)
        self.log_dir, self.log = Path(log_dir), log
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.available = sorted(m.id for m in self.client.models.list().data)
        self.model = next((m for m in PREFERRED if m in self.available), None) \
            or next((m for m in self.available if m.startswith("qwen/")), None) \
            or (FALLBACK if FALLBACK in self.available else None)
        if not self.model:
            raise RuntimeError("No Qwen or fallback model in authenticated Groq model list")
        self.calls = []

    def _kwargs(self):
        kw = dict(model=self.model, temperature=LLM_TEMP, max_completion_tokens=LLM_MAX_TOKENS,
                  response_format={"type": "json_object"})
        if self.model.startswith("qwen/"):
            kw["reasoning_format"] = "hidden"
        elif self.model.startswith("openai/gpt-oss"):
            kw["include_reasoning"] = False
        return kw

    def call(self, system_prompt, user_prompt, label, max_attempts=8):
        h = hashlib.sha256((self.model + "\n" + system_prompt + "\n" + user_prompt).encode()).hexdigest()[:16]
        (self.log_dir / f"prompt_{label}.txt").write_text(user_prompt)
        cache = self.log_dir / f"{label}_{h}.json"
        if cache.exists():
            rec = json.loads(cache.read_text())
            rec["cached"] = True
            self.calls.append({k: rec[k] for k in ("label", "model", "prompt_sha16", "finish_reason", "usage", "attempts", "cached", "elapsed_s")})
            self.log(f"  [{label}] cache hit ({rec['model']})")
            return rec["content"], rec["finish_reason"]
        attempts, last_err = [], None
        for a in range(1, max_attempts + 1):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}], **self._kwargs())
                fin = resp.choices[0].finish_reason
                content = resp.choices[0].message.content or ""
                attempts.append({"attempt": a, "ok": True, "finish_reason": fin, "s": round(time.time() - t0, 2)})
                if fin == "length" and a < max_attempts:
                    last_err = "truncated"; time.sleep(2 * a); continue
                u = resp.usage
                rec = {"label": label, "model": resp.model, "prompt_sha16": h, "finish_reason": fin,
                       "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                                 "total_tokens": u.total_tokens},
                       "attempts": attempts, "cached": False, "elapsed_s": round(time.time() - t0, 2),
                       "content": content}
                cache.write_text(json.dumps(rec, indent=1))
                (self.log_dir / f"llm_raw_{label}.txt").write_text(content)
                self.calls.append({k: rec[k] for k in rec if k != "content"})
                self.log(f"  [{label}] {fin} | model={resp.model} | tokens={u.total_tokens:,} | attempts={a}")
                return content, fin
            except Exception as e:  # API/network errors: exponential backoff
                last_err = f"{type(e).__name__}: {str(e)[:160]}"
                attempts.append({"attempt": a, "ok": False, "error": last_err, "s": round(time.time() - t0, 2)})
                m = re.search(r"try again in ([0-9.]+)(ms|s)", str(e))
                wait = (float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) + 1.5) if m else min(30, 3 * 2 ** (a - 1))
                if "Request too large" in str(e) or "rate_limit" in str(e).lower() or "Rate limit" in str(e):
                    wait = max(wait, 20)
                attempts[-1]["wait_s"] = round(min(wait, 65), 1)
                time.sleep(min(wait, 65))
        raise RuntimeError(f"LLM call {label} failed after {max_attempts} attempts: {last_err}")

def parse_llm(raw, label='', log=print):
    """cell 24 verbatim."""
    cleaned = re.sub(r'^```json\s*', '', raw.strip())
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()
    s = cleaned.find('{'); e = cleaned.rfind('}') + 1
    if s == -1:
        log(f'  WARNING [{label}]: no JSON found'); return []
    try:
        return json.loads(cleaned[s:e]).get('features', [])
    except json.JSONDecodeError as ex:
        log(f'  WARNING [{label}]: JSON error ({ex}) - attempting repair...')
        try:
            result = json.loads(repair_json(cleaned[s:e])).get('features', [])
            log(f'  REPAIRED [{label}]: recovered {len(result)} features'); return result
        except Exception as ex2:
            log(f'  REPAIR FAILED [{label}]: {ex2}'); return []
