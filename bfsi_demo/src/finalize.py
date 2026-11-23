"""Assemble scored table, run selftest, write outputs/final_results.json (verified values only)."""
from __future__ import annotations
import json, platform, time
import numpy as np
import pandas as pd
from . import config as C, data, export, selftest

def main(log=print):
    cache = C.CACHE_DIR
    tf = pd.read_parquet(cache / "target_fold.parquet")
    sc = tf.copy()
    for P in ("A", "B", "C"):
        s = pd.read_parquet(cache / f"scores_{P}.parquet")
        assert np.array_equal(s[C.ID_COL].to_numpy(), sc[C.ID_COL].to_numpy()), P
        sc[f"oof_{P}"] = s[f"oof_{P}"].astype("float64")
    for P in ("A", "B", "C"):
        sc[f"pred_{P}"] = pd.read_parquet(cache / f"scores_{P}.parquet")[f"pred_{P}"].astype("float64")
    sc["TARGET"] = sc["TARGET"].astype("int8"); sc["fold"] = sc["fold"].astype("int8")
    sc.to_parquet(C.OUT_DIR / "scored_table.parquet", index=False)
    sc.to_csv(C.OUT_DIR / "scored_table.csv", index=False, float_format="%.10g")
    log(f"  scored table {sc.shape}")
    st = selftest.run(log=log)
    A = json.loads((cache / "stage_A.json").read_text()); BC = json.loads((cache / "stage_BC.json").read_text())
    llm = json.loads((cache / "llm_calls.json").read_text())
    eqv = json.loads((cache / "equivalence.json").read_text())
    eqv = eqv if isinstance(eqv, list) else eqv.get("checks", eqv)
    from sklearn.metrics import roc_auc_score
    checks = list(st["checks"]) + (eqv if isinstance(eqv, list) else [])
    for P in ("A", "B", "C"):
        auc = roc_auc_score(sc["TARGET"], sc[f"oof_{P}"])
        m = A["metrics"] if P == "A" else BC[P]
        checks.append({"name": f"{P}_oof_auc_recomputed_from_scored_table", "passed": abs(auc - m["oof_auc"]) < 1e-5, "detail": f"{auc:.6f}"})
    au_path = C.OUT_DIR / "llm_alignment_audit.json"
    au = json.loads(au_path.read_text()) if au_path.exists() else {"all_passed": False, "n_features": 0}
    checks.append({"name": "llm_features_index_alignment_audit", "passed": au["all_passed"], "detail": f"{au['n_features']} accepted LLM features re-executed on real tables"})
    checks.append({"name": "A_features_equal_reference_registry_set", "passed": A["features_equal_reference_set"], "detail": f"order_equal={A['features_equal_reference_241_order']}"})
    checks.append({"name": "six_live_llm_calls_successful", "passed": len(llm["calls"]) == 6 and all(c.get("finish_reason") == "stop" for c in llm["calls"]), "detail": llm["model"]})
    # per-stage gate outcome for every candidate (derived from the first failing stage; later stages not run)
    STAGES = ["s1_security", "s2_code_exec", "s3_format", "s4_high_null", "s5_constant", "s6_infinite", "s7_leakage_spearman", "s8_exact_duplicate"]
    FAIL_AT = {"security_violation": 0, "code_error": 1, "wrong_format": 2, "high_null": 3, "constant": 4,
               "infinite_values": 5, "leakage": 6, "exact_duplicate": 7, "passed": 8}
    vl = pd.read_csv(C.OUT_DIR / "validation_log.csv")
    for k, st_name in enumerate(STAGES):
        vl[st_name] = [("pass" if k < FAIL_AT.get(ft, 0) else ("FAIL" if k == FAIL_AT.get(ft, 0) else "not_run")) for ft in vl["failure_type"]]
    vl.to_csv(C.OUT_DIR / "validation_log.csv", index=False)
    checks.append({"name": "gate_log_all_candidates_8_stages", "passed": len(vl) == llm["n_proposed"] and vl["failure_type"].notna().all(),
                   "detail": vl["failure_type"].value_counts().to_dict()})
    model_sha = {P: {"pkl": data.sha256(C.MODEL_DIR / f"model_{P}.pkl"), "txt": data.sha256(C.MODEL_DIR / f"model_{P}.txt")} for P in ("A", "B", "C")}
    feat_sha = {P: data.sha256(C.FEAT_DIR / f"features_{P}.parquet") for P in ("A", "B", "C")}
    csv_sha = {n: data.sha256(C.DATA_DIR / f) for n, f in data.FILES.items() if (C.DATA_DIR / f).exists()}
    res = {
        "execution_status": "complete" if all(c["passed"] for c in checks) else "complete_with_failed_checks",
        "artifacts_sha256": {"models": model_sha, "features": feat_sha},
        "run": {"finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S") + " Asia/Kolkata (UTC+05:30)",
                "stage_times_s": {"A": A.get("time_s"), "BC": BC.get("time_s")},
                "peak_rss_gb": {"A": A.get("rss_max_gb"), "BC": BC.get("rss_max_gb")}, "python": platform.python_version(),
                "seed": C.RANDOM_SEED, "n_folds": C.N_FOLDS, "params": C.BEST_PARAMS,
                "scores": "raw uncalibrated LightGBM probabilities (scale_pos_weight-weighted); oof_* = out-of-fold CV, pred_* = final all-data model (in-sample)"},
        "data": {"source": "Kaggle competition home-credit-default-risk (official API)", "rows_train": int(len(sc)),
                 "positive_rate": round(float(sc['TARGET'].mean()), 5), "sha256": csv_sha},
        "llm": {"model": llm["model"], "authenticated_models": llm["available"], "calls": llm["calls"], "n_proposed": llm["n_proposed"]},
        "gate": BC["gate"],
        "pipelines": {"A": A["metrics"], "B": BC["B"], "C": BC["C"]},
        "row_level_llm_features": BC["row_level_llm"],
        "checks": checks,
        "historical_reference": {**C.HISTORICAL_REFERENCE,
                                 "_note": "Earlier Colab run (Llama-4-Scout era); reference only, NOT produced by this run. A' was CV'd only historically."},
    }
    export.write_json(res, C.OUT_DIR / "final_results.json")
    log(f"  final_results.json status={res['execution_status']}")
    return res
