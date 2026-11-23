"""Raw data acquisition (Kaggle CLI, same flow as the original notebooks) and memory-aware loading.

Memory policy (box has ~4 GB free): tables are loaded one at a time; *auxiliary-table* string columns
are read as pandas 'category' (numeric columns untouched, float64/int64). application_train/test keep
object dtype exactly as in the notebooks. Equivalence of category vs object for every aggregation used
is asserted by src/equivalence.py before the real run.
"""
from __future__ import annotations
import hashlib, os, subprocess, zipfile
from pathlib import Path
import pandas as pd
from . import config as C

FILES = {
    "app_train": "application_train.csv", "app_test": "application_test.csv",
    "bureau": "bureau.csv", "bureau_bal": "bureau_balance.csv",
    "prev_app": "previous_application.csv", "install": "installments_payments.csv",
    "cc_bal": "credit_card_balance.csv", "pos_cash": "POS_CASH_balance.csv",
    "col_desc": "HomeCredit_columns_description.csv",
}
AUX = ("bureau", "bureau_bal", "prev_app", "install", "cc_bal", "pos_cash")

def data_present(data_dir: Path = C.DATA_DIR) -> bool:
    return all((data_dir / f).exists() for k, f in FILES.items() if k != "col_desc")

def download(data_dir: Path = C.DATA_DIR) -> None:
    """kaggle competitions download -c home-credit-default-risk (token via env only, never printed)."""
    if data_present(data_dir):
        return
    tok = C.get_secret("KAGGLE_API_TOKEN")
    if not tok:
        raise RuntimeError("KAGGLE_API_TOKEN not available (env or secrets dir)")
    data_dir.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("GROQ_API_KEY",)}
    env["KAGGLE_API_TOKEN"] = tok
    kaggle = str(Path(__file__).resolve().parents[2] / ".venv/bin/kaggle")
    if not Path(kaggle).exists():
        kaggle = "kaggle"
    r = subprocess.run([kaggle, "competitions", "download", "-c", "home-credit-default-risk",
                        "-p", str(data_dir)], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout)[-500:].replace(tok, "<REDACTED>")
        raise RuntimeError(f"kaggle download failed: {msg}")
    z = data_dir / "home-credit-default-risk.zip"
    with zipfile.ZipFile(z) as zf:
        for m in zf.infolist():
            p = (data_dir / m.filename).resolve()
            if not str(p).startswith(str(data_dir.resolve())):
                raise RuntimeError("unsafe path in zip")
        zf.extractall(data_dir)
    z.unlink()

def sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def _string_cols(path: Path) -> list[str]:
    head = pd.read_csv(path, nrows=20000)
    return head.select_dtypes("object").columns.tolist()

def load(name: str, data_dir: Path = C.DATA_DIR, as_category: bool | None = None, nrows=None) -> pd.DataFrame:
    path = data_dir / FILES[name]
    if name == "col_desc":
        return pd.read_csv(path, encoding="latin-1") if path.exists() else None
    if as_category is None:
        as_category = name in AUX
    if not as_category:
        return pd.read_csv(path, nrows=nrows)
    dtypes = {c: "category" for c in _string_cols(path)}
    return pd.read_csv(path, dtype=dtypes, nrows=nrows)
