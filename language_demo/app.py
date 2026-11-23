"""Hindi word-order evaluation-set dashboard (Pipelines A, B, C).

Loads the saved models model_A/B/C.pkl and scores the pre-extracted feature rows of a
matched evaluation-set sentence with predict_proba. No feature extraction, no LLM calls.
"""

import json
import math
import unicodedata
from pathlib import Path

import pickle

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODELS = [
    ("A", "Model A - Hand-crafted features"),
    ("B", "Model B - LLM features"),
    ("C", "Model C - Hand-crafted + LLM"),
]
OUTSIDE_MSG = "This sentence is outside the evaluation set."
TOP_LABEL = "Top features (by model weight)"
JOIN = ["sentid", "fold", "variant_rank"]


def find_file(name, base=HERE):
    for d in (Path(base), Path.cwd()):
        if (d / name).is_file():
            return d / name
    raise FileNotFoundError(name)


def norm(text):
    return unicodedata.normalize("NFC", str(text)).strip()


def fmt_value(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(x) or math.isinf(x):
        return "—"
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


@st.cache_resource(show_spinner=False)
def load_models(base):
    models, order, ranked = {}, {}, {}
    for m, _ in MODELS:
        with open(
            find_file(f"selected_features_{m}.json", base), encoding="utf-8"
        ) as fh:
            cols = list(json.load(fh)["feature_order"])
        with open(find_file(f"model_{m}.pkl", base), "rb") as fh:
            mdl = pickle.load(fh)
        if list(mdl.feature_names_in_) != cols:
            raise ValueError(f"feature order mismatch for model {m}")
        coef = mdl.named_steps["clf"].coef_.ravel()
        pos = {f: i for i, f in enumerate(cols)}
        models[m], order[m] = mdl, cols
        ranked[m] = sorted(
            zip(cols, coef.tolist()), key=lambda t: (-abs(t[1]), pos[t[0]])
        )
    return models, order, ranked


@st.cache_data(show_spinner=False)
def load_data(base):
    lk = pd.read_parquet(find_file("demo_lookup.parquet", base))
    ref = lk[lk["label"] == 1].copy()
    ref["key"] = ref["reference_text"].map(norm)
    fv = pd.read_parquet(find_file("feature_values.parquet", base))
    ex = (
        ref.drop_duplicates("sentid")
        .assign(n=lambda d: d["key"].str.len())
        .sort_values(["n", "sentid"])
    )
    examples = ex["key"].head(5).tolist()
    return ref, fv, examples


def score(models, order, feats):
    """predict_proba[:, 1] = P(reference order preferred) for each label-1 (reference-first) row."""
    out = {}
    for m, _ in MODELS:
        X = feats.reindex(columns=order[m]).apply(pd.to_numeric, errors="coerce")
        X = X.replace([float("inf"), float("-inf")], float("nan"))
        out[m] = models[m].predict_proba(X)[:, 1]
    return out


def find_sentence(ref, text):
    hits = ref[ref["key"] == norm(text)]
    if hits.empty:
        return None
    first = sorted(hits["sentid"].unique())[0]  # deterministic if a text ever repeats
    return (
        hits[hits["sentid"] == first].sort_values("variant_rank").reset_index(drop=True)
    )


def prefers(p):
    return "Prefers: Reference" if p > 0.5 else "Prefers: Variant"


def main():
    st.set_page_config(page_title="Hindi Word-Order Results", layout="wide")
    st.markdown(
        "<style>.stApp, [data-testid='stHeader']{background:#ffffff;color:#111111;}</style>",
        unsafe_allow_html=True,
    )  # static style only
    st.title("Hindi Word-Order Results")
    st.caption("Results for sentences in the evaluation set.")

    try:
        ref, fv, examples = load_data(str(HERE))
        models, order, ranked = load_models(str(HERE))
    except Exception:
        st.error(
            "Could not load the dashboard data files. Keep all package files next to app.py."
        )
        st.stop()

    text = st.text_input(
        "Sentence", placeholder="Paste a Hindi sentence", label_visibility="collapsed"
    )
    if st.button("Predict", type="primary"):
        st.session_state["query"] = norm(text)
        st.session_state["variant_choice"] = "Variant 1"

    query = st.session_state.get("query")
    if query is None:
        return
    if not query:
        st.info("Please paste a Hindi sentence.")
        return

    rows = find_sentence(ref, query)
    if rows is None:
        st.warning(OUTSIDE_MSG)
        for e in examples:
            st.code(e, language=None)
        return

    feats = rows[JOIN].merge(fv, on=JOIN, how="left", validate="one_to_one")
    probs_all = score(models, order, feats)
    for m, _ in MODELS:
        rows[f"live_{m}"] = probs_all[m]

    st.subheader("Reference sentence")
    st.text(str(rows.loc[0, "reference_text"]))

    labels = [f"Variant {i + 1}" for i in range(len(rows))]
    st.subheader("Variants")
    for i, r in rows.iterrows():
        with st.container(border=True):
            st.markdown(f"**{labels[i]}**")
            st.text(str(r["variant_text"]))
            cols = st.columns(3)
            for col, (m, title) in zip(cols, MODELS):
                p = float(r[f"live_{m}"])
                with col:
                    st.markdown(f"**{title}**")
                    st.metric("Reference preferred", f"{p * 100:.1f}%")
                    st.progress(min(max(p, 0.0), 1.0))
                    st.write(prefers(p))

    if st.session_state.get("variant_choice") not in labels:
        st.session_state["variant_choice"] = labels[0]
    choice = st.radio(
        "Variant for features", labels, horizontal=True, key="variant_choice"
    )
    sel = rows.iloc[labels.index(choice)]

    probs = {m: float(sel[f"live_{m}"]) for m, _ in MODELS}
    conf = {m: max(p, 1 - p) for m, p in probs.items()}
    best = max(conf, key=lambda m: (conf[m], -"ABC".index(m)))
    agree = (
        "A and C agree"
        if (probs["A"] > 0.5) == (probs["C"] > 0.5)
        else "A and C disagree"
    )
    st.write(
        f"{choice}: Model {best} is most confident ({conf[best] * 100:.1f}%); {agree}."
    )

    frow = feats.iloc[[labels.index(choice)]]
    st.subheader(f"Features — {choice}")
    fcols = st.columns(3)
    for col, (m, title) in zip(fcols, MODELS):
        items = ranked[m]
        vals = [
            fmt_value(frow.iloc[0][f]) if len(frow) == 1 and f in frow.columns else "—"
            for f, _ in items
        ]
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.markdown(f"**{TOP_LABEL}**")
                st.dataframe(
                    pd.DataFrame(
                        {"Feature": [f for f, _ in items[:3]], "Value": vals[:3]}
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.write(f"All {len(items)} features")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Feature": [f for f, _ in items],
                            "Value": vals,
                            "Weight": [f"{w:+.3f}" for _, w in items],
                        }
                    ),
                    hide_index=True,
                    height=300,
                    width="stretch",
                )


main()
