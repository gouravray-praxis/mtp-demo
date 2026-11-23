"""End-to-end real run (resumable).  Usage:  python run_all.py A | BC | finalize | all
Stages write intermediates to data/cache (not packaged) and artifacts to features/ models/ outputs/."""
from __future__ import annotations
import gc, json, os, sys, time, platform, resource
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BFSI_THREADS", "8")
from src import config as C, data, train, export
from src import features_a as FA, features_bc as FB, feature_recompute as FR
C.ensure_dirs()
CACHE = C.CACHE_DIR; CACHE.mkdir(parents=True, exist_ok=True)
LOGF = open(C.OUT_DIR / "run_log.txt", "a")

def log(*a):
    s = " ".join(str(x) for x in a)
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True); LOGF.write(line + "\n"); LOGF.flush()

def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)

def loader(name):
    return data.load(name, as_category=True)

MECH = set(FR.MECHANICAL)
AUX_PREF = ('BUREAU_', 'INSTAL_', 'PREV_', 'CC_', 'POS_')

def schema_for(pipeline, names, ohe_levels, clean_map, llm_specs_by_name):
    """origin in raw|ohe|mechanical|human_app|human_aux|llm_app|llm_aux; arrays keep model order."""
    groups = {}
    for col in ohe_levels:
        cname = clean_map(col)
        members = [n for n in names if n.startswith(cname + "_")]
        if members:
            groups[col] = members
    ohe_members = {m: g for g, ms in groups.items() for m in ms}
    feats = []
    for n in names:
        if n in llm_specs_by_name:
            s = llm_specs_by_name[n]
            origin = "llm_app" if s["source_group"] == "application" else "llm_aux"
            deps = s.get("app_columns", []) if s.get("row_level") else []
            editable = bool(s.get("row_level")) and any(d in FR.EDITABLE or d in MECH for d in deps)
            feats.append({"name": n, "origin": origin, "depends_on": deps, "editable_via_recompute": editable,
                          "row_level": bool(s.get("row_level")), "source_group": s["source_group"]})
            continue
        if n in ohe_members:
            origin, deps = "ohe", [ohe_members[n]]
        elif n in MECH:
            origin, deps = "mechanical", FR.MECH_DEPS[n]
        elif n == "DAYS_EMPLOYED":
            origin, deps = "mechanical", ["DAYS_EMPLOYED"]  # sentinel 365243 -> NaN
        elif n.startswith("FE_"):
            origin, deps = "human_app", FR.FE_DEPS.get(n, [])
        elif n.startswith(AUX_PREF):
            origin, deps = "human_aux", []
        else:
            origin, deps = "raw", [n]
        editable = origin in ("raw", "mechanical", "human_app") and any(
            d in FR.EDITABLE or (d == "DAYS_EMPLOYED") for d in deps) and (origin != "raw" or n in FR.EDITABLE)
        feats.append({"name": n, "origin": origin, "depends_on": list(deps), "editable_via_recompute": bool(editable)})
    return {"pipeline": pipeline, "n_features": len(names), "features": feats,
            "ohe_groups": groups, "editable_inputs": FR.EDITABLE,
            "note": "Aggregated auxiliary features (human_aux, llm_aux, non-row-level LLM) are held at stored values."}

def train_export(X, y, ids, P, log=log):
    log(f"== {P}: CV on {X.shape} (rss_max={rss_gb()}GB)")
    oof, metrics, fi = train.run_cv(X, y, P, log=log)
    model = train.fit_final(X, y, log=log, label=P)
    pred_mem = model.predict_proba(X)[:, 1]
    export.save_model(model, P)
    Xf = export.save_matrix(X, ids, P)
    del Xf; gc.collect()
    # read-back check: parquet float64 -> identical predictions
    back = pq.read_table(C.FEAT_DIR / f"features_{P}.parquet").to_pandas()
    assert back[C.ID_COL].is_unique and len(back) == len(X) and C.TARGET_COL not in back.columns
    assert list(back.columns[1:]) == list(model.booster_.feature_name())
    pred_back = model.predict_proba(back.drop(columns=[C.ID_COL]))[:, 1]
    maxdiff = float(np.max(np.abs(pred_back - pred_mem)))
    log(f"  {P} parquet read-back pred maxdiff={maxdiff:.3g}")
    del back; gc.collect()
    sc = pd.DataFrame({C.ID_COL: ids.to_numpy("int64"), f"oof_{P}": oof, f"pred_{P}": pred_back})
    sc.to_parquet(CACHE / f"scores_{P}.parquet", index=False)
    fi.sort_values("mean_gain", ascending=False).to_csv(C.OUT_DIR / f"feature_importance_{P}.csv")
    metrics["parquet_readback_pred_maxdiff"] = maxdiff
    metrics["final_fit_trees"] = int(model.booster_.num_trees())
    return metrics

# ------------------------------------------------------------------------------------------- A
def stage_A():
    t0 = time.time()
    app = data.load("app_train", as_category=False)
    X, y, ids, reg = FA.build_matrix_A(app, loader, log=log)
    del app; gc.collect()
    ref = json.loads((ROOT / "src/reference/pipeline_a_feature_registry_2026-06-03.json").read_text())["feature_names"]
    ref = list(FA.clean_feature_names(pd.DataFrame(columns=ref)).columns)  # registry stores pre-cleaning names
    same = list(X.columns) == list(ref); same_set = set(X.columns) == set(ref)
    log(f"  A features={X.shape[1]} equal_to_reference_order={same} set={same_set}")
    if not same_set:
        log("  diff +", sorted(set(X.columns) - set(ref))[:20], " -", sorted(set(ref) - set(X.columns))[:20])
    m = train_export(X, y, ids, "A")
    pd.DataFrame({C.ID_COL: ids.to_numpy("int64"), "TARGET": y.to_numpy("int8"),
                  "fold": train.fold_ids(y.values)}).to_parquet(CACHE / "target_fold.parquet", index=False)
    sch = schema_for("A", list(X.columns), reg.get("ohe_levels", {}), lambda c: FA.clean_feature_names(pd.DataFrame(columns=[c])).columns[0], {})
    export.write_json(sch, C.FEAT_DIR / "schema_A.json")
    st = {"metrics": m, "features_equal_reference_241_order": same, "features_equal_reference_set": same_set,
          "registry": {k: v for k, v in reg.items() if k not in ("feature_names_clean",)}, "time_s": round(time.time() - t0, 1),
          "rss_max_gb": rss_gb()}
    export.write_json(st, CACHE / "stage_A.json")
    log(f"== A done {time.time()-t0:.0f}s")

# ------------------------------------------------------------------------------------------ BC
def stage_BC():
    from src import llm as L, gate as G
    t0 = time.time()
    app_raw = data.load("app_train", as_category=False)
    raw_numeric = [c for c in app_raw.columns if pd.api.types.is_numeric_dtype(app_raw[c]) and c not in ("TARGET",)]
    app_proc, BASE, y, meta = FB.build_shared_base(app_raw)
    log(f"  base shape {BASE.shape}")
    y_by_id = pd.Series(y.values, index=app_proc[C.ID_COL].values, name="TARGET")
    # ---------- schemas (cell 14) — tables loaded one at a time
    cd_path = C.DATA_DIR / "HomeCredit_columns_description.csv"
    COL_DESC = pd.read_csv(cd_path, encoding="latin-1") if cd_path.exists() else None
    schemas = {"application_train": L.get_schema(app_raw.drop(columns=["TARGET"], errors="ignore"), "application_train", C.ID_COL, COL_DESC)}
    for name, key, jk in [("bureau", "bureau", "SK_ID_CURR"), ("bureau_balance", "bureau_bal", "SK_ID_BUREAU"),
                          ("previous_application", "prev_app", "SK_ID_CURR"), ("installments_payments", "install", "SK_ID_CURR"),
                          ("credit_card_balance", "cc_bal", "SK_ID_CURR"), ("POS_CASH_balance", "pos_cash", "SK_ID_CURR")]:
        df = loader(key); schemas[name] = L.get_schema(df, name, jk, COL_DESC); del df; gc.collect()
    ref = json.loads((ROOT / "src/reference/pipeline_a_feature_registry_2026-06-03.json").read_text())["feature_names"]
    # ---------- 6 live calls
    client = L.LLMClient(log=log)
    log(f"  Groq authenticated models={len(client.available)}; selected={client.model}")
    proposals = []
    for label, focus, tables, n, avoid_prefix, group in L.CALLS:
        prompt = L.build_table_prompt(focus, {t: schemas[t] for t in tables}, n_features=n,
                                      avoid_human_features=[f for f in ref if f.startswith(avoid_prefix)])
        raw, fin = client.call(L.SYSTEM_PROMPT, prompt, label)
        feats = L.parse_llm(raw, label, log=log)
        for f in feats:
            f["source_group"] = group; f["call"] = label
        log(f"  {label}: parsed {len(feats)} (requested {n}) finish={fin}")
        proposals += feats
    export.write_json({"model": client.model, "available": client.available, "calls": client.calls,
                       "n_proposed": len(proposals)}, CACHE / "llm_calls.json")
    export.write_json(proposals, C.LLM_DIR / "proposals.json")
    # ---------- 8-stage gate
    VAL_LOG, PASSED, FAILED = G.run_gate(proposals, y_by_id, app_proc, loader, log=log)
    pd.DataFrame(VAL_LOG).to_csv(C.OUT_DIR / "validation_log.csv", index=False)
    log(f"  gate: proposed={len(proposals)} passed={len(PASSED)} failed={len(FAILED)}")
    gc.collect()
    # ---------- row-level LLM candidates (for editable recompute)
    allowed = set(raw_numeric) - {"DAYS_EMPLOYED"} | set(FR.MECHANICAL) | {"DAYS_EMPLOYED"}
    specs = []
    for f in proposals:
        lg = next((v for v in VAL_LOG if v["feature_name"] == f.get("feature_name")), {})
        rl, cols = FR.llm_row_level(f.get("python_code", ""), allowed, set(app_proc.columns))
        specs.append({"feature_name": f.get("feature_name"), "source_group": f["source_group"], "call": f["call"],
                      "source_tables": f.get("source_tables", []), "row_level": rl and f.get("feature_name") in PASSED,
                      "app_columns": cols, "python_code": f.get("python_code", ""),
                      "gate": lg.get("failure_type", "unknown"), "gate_detail": lg.get("failure_detail"),
                      "rationale": f.get("rationale", "")})
    raw_cols = FR.raw_input_columns(specs)
    raw_inputs = FR.build_raw_inputs(app_raw, raw_cols)
    der = FR.derive_app(raw_inputs)
    for s in specs:
        if not s["row_level"]:
            continue
        try:
            full = FR.llm_series(s["python_code"], s["feature_name"], der).reindex(y_by_id.index)
            stored = PASSED[s["feature_name"]].reindex(y_by_id.index)
            ok_full = np.array_equal(full.to_numpy("float64"), stored.to_numpy("float64"), equal_nan=True)
            samp = raw_inputs.sample(25, random_state=7)
            ok_one = all(np.array_equal([FR.llm_value(s["python_code"], s["feature_name"], FR.derive_app(raw_inputs.loc[[i]]), int(raw_inputs.at[i, C.ID_COL]))],
                                        [float(stored.at[int(raw_inputs.at[i, C.ID_COL])])], equal_nan=True) for i in samp.index)
            s["row_level"] = bool(ok_full and ok_one)
            s["row_level_check"] = {"full_frame_equal": bool(ok_full), "single_row_equal_25": bool(ok_one)}
        except Exception as e:
            s["row_level"] = False; s["row_level_check"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        log(f"  row-level {s['feature_name']}: {s['row_level']} {s.get('row_level_check')}")
    raw_cols = FR.raw_input_columns(specs)
    raw_inputs = FR.build_raw_inputs(app_raw, raw_cols)
    raw_inputs.to_parquet(C.FEAT_DIR / "raw_inputs.parquet", index=False)
    export.write_json(FR.editable_meta(raw_inputs), C.FEAT_DIR / "editable_inputs.json")
    del der, app_raw; gc.collect()
    by_name = {s["feature_name"]: s for s in specs if s["feature_name"] in PASSED}
    # ---------- C human df first (needs app_proc), then B
    df_human = FB.build_human_df(app_proc, loader)
    del app_proc; gc.collect()
    log(f"  A' human df {df_human.shape}")
    X_B, feat_B, ids_B, sel_B = FB.build_matrix_B(BASE, PASSED, log=log)
    del BASE; gc.collect()
    llm_in_B = [c for c in feat_B if c in PASSED]
    mB = train_export(X_B, y, ids_B, "B"); mB["n_llm_features"] = len(llm_in_B); mB["llm_features"] = llm_in_B
    sB = schema_for("B", feat_B, meta["ohe_levels"], lambda c: FB.sanitize_column_names(pd.DataFrame(columns=[c])).columns[0], by_name)
    export.write_json(sB, C.FEAT_DIR / "schema_B.json")
    ids_B_arr = ids_B.to_numpy("int64")
    del X_B; gc.collect()
    X_C, feat_C, ids_C, sel_C = FB.build_matrix_C(df_human, PASSED, log=log)
    del df_human; gc.collect()
    assert np.array_equal(ids_C.to_numpy("int64"), ids_B_arr)
    llm_in_C = [c for c in feat_C if c in PASSED]
    mC = train_export(X_C, y, ids_C, "C"); mC["n_llm_features"] = len(llm_in_C); mC["llm_features"] = llm_in_C
    mC["n_human_features"] = len(feat_C) - len(llm_in_C)
    sC = schema_for("C", feat_C, meta["ohe_levels"], lambda c: FB.sanitize_column_names(pd.DataFrame(columns=[c])).columns[0], by_name)
    export.write_json(sC, C.FEAT_DIR / "schema_C.json")
    for s in specs:
        s["in_B"] = s["feature_name"] in llm_in_B; s["in_C"] = s["feature_name"] in llm_in_C
    export.write_json(specs, C.FEAT_DIR / "llm_feature_specs.json")
    gate_sum = pd.DataFrame(VAL_LOG)["failure_type"].value_counts().to_dict()
    export.write_json({"B": mB, "C": mC, "sel_B": sel_B, "sel_C": sel_C, "base_shape": meta["base_shape"],
                       "gate": {"proposed": len(proposals), "passed": len(PASSED), "failed": len(FAILED), "by_type": gate_sum},
                       "row_level_llm": [s["feature_name"] for s in specs if s["row_level"]],
                       "time_s": round(time.time() - t0, 1), "rss_max_gb": rss_gb()}, CACHE / "stage_BC.json")
    log(f"== BC done {time.time()-t0:.0f}s")

def stage_schemas():
    """(Re)write schema_{A,B,C}.json from the exported feature lists with the current origin rules."""
    stA = json.loads((CACHE / "stage_A.json").read_text())
    names = {P: json.loads((C.FEAT_DIR / f"feature_names_{P}.json").read_text()) for P in "ABC"}
    specs = json.loads((C.FEAT_DIR / "llm_feature_specs.json").read_text())
    _, _, _, meta = FB.build_shared_base(data.load("app_train", as_category=False))
    export.write_json(schema_for("A", names["A"], stA["registry"]["ohe_levels"], lambda c: FA.clean_feature_names(pd.DataFrame(columns=[c])).columns[0], {}), C.FEAT_DIR / "schema_A.json")
    for P in "BC":
        by = {s["feature_name"]: s for s in specs if s.get(f"in_{P}")}
        export.write_json(schema_for(P, names[P], meta["ohe_levels"], lambda c: FB.sanitize_column_names(pd.DataFrame(columns=[c])).columns[0], by), C.FEAT_DIR / f"schema_{P}.json")
    log("  schemas rewritten")

if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("A", "all"): stage_A()
    if st in ("BC", "all"): stage_BC()
    if st in ("schemas",):
        stage_schemas()
    if st in ("audit", "all"):
        from src import llm_audit; llm_audit.run(log)
    if st in ("finalize", "all"):
        from src import finalize; finalize.main(log)
