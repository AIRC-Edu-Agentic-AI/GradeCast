"""
GradeCast — Student Risk & Grade Trajectory Predictor
======================================================
Run:    python 01_train_and_predict.py
Output: results.json  (read by dashboard.html)

DATASET
  OULAD files required in same folder:
    studentInfo.csv, studentVle.csv, studentAssessment.csv,
    assessments.csv, courses.csv, vle.csv, studentRegistration.csv

DESIGN
  Split:    80% train / 10% validation / 10% test — stratified by risk label
  Tuning:   Hyperparameters selected on validation set per 5-week band
  Eval:     All 39 weeks, test + validation + per-outcome-group breakdown
  Features: 11 features including 3 new registration-derived signals
  Two paths:
    OBSERVED  (solid)  — real accumulated data at each week t → predict AT t
    PREDICTED (dashed) — features frozen at snapshot week, only week index advances
    NOTE: date_unregistration is NEVER used as a feature (would be data leakage)
"""

import pandas as pd
import numpy as np
import json
import warnings
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────
DATA_DIR      = "."
EARLY_MODULES = ["AAA", "BBB", "CCC"]
PRESENTATION  = "2013J"
N_TEST        = 20
DAYS_PER_WEEK = 7
WEEKS         = list(range(1, 40))
RANDOM_STATE  = 42
OUTPUT_FILE   = "results.json"

print("=" * 62)
print("  GradeCast — Risk & Grade Trajectory Predictor  v2.0")
print("=" * 62)

# ─── 1. LOAD ──────────────────────────────────────────────────────
print("\n[1/7] Loading data...")
info   = pd.read_csv(f"{DATA_DIR}/studentInfo.csv")
svle   = pd.read_csv(f"{DATA_DIR}/studentVle.csv")
sa     = pd.read_csv(f"{DATA_DIR}/studentAssessment.csv")
assess = pd.read_csv(f"{DATA_DIR}/assessments.csv")
sreg   = pd.read_csv(f"{DATA_DIR}/studentRegistration.csv")  # NEW

info_filt = info[
    info["code_module"].isin(EARLY_MODULES) &
    (info["code_presentation"] == PRESENTATION)
].copy()
info_filt["at_risk"] = info_filt["final_result"].isin(["Withdrawn", "Fail"]).astype(int)

# Merge registration data
sreg_filt = sreg[
    sreg["code_module"].isin(EARLY_MODULES) &
    (sreg["code_presentation"] == PRESENTATION)
]
info_filt = info_filt.merge(
    sreg_filt[["id_student", "date_registration", "date_unregistration"]],
    on="id_student", how="left"
)
# Registration timing (negative = registered before course start)
info_filt["reg_days_before"] = info_filt["date_registration"].fillna(0)
# Unregistration week — stored for reference ONLY, never used as a feature
info_filt["unreg_week"] = (info_filt["date_unregistration"] / DAYS_PER_WEEK).round(1)

# Grade label
assess_filt = assess[
    assess["code_module"].isin(EARLY_MODULES) &
    (assess["code_presentation"] == PRESENTATION)
]
tma    = assess_filt[assess_filt["assessment_type"].isin(["TMA", "CMA"])]
sa_tma = sa.merge(tma[["id_assessment", "weight", "date"]], on="id_assessment", how="inner")
sa_tma["week"] = (sa_tma["date"] // DAYS_PER_WEEK + 1).clip(1, 39)

def compute_grade(sid):
    rows = sa_tma[sa_tma["id_student"] == sid].dropna(subset=["score"])
    if len(rows) == 0:
        return np.nan
    return float(np.average(rows["score"], weights=rows["weight"].fillna(1)))

info_filt["final_grade"] = info_filt["id_student"].map(compute_grade)
info_filt = info_filt.dropna(subset=["final_grade"])
print(f"    Students: {len(info_filt)} | at-risk: {info_filt['at_risk'].mean():.1%}")

# ─── 2. 80 / 10 / 10 STRATIFIED SPLIT ────────────────────────────
print(f"\n[2/7] Stratified 80/10/10 split ({N_TEST} test students)...")
X, y = info_filt["id_student"].values, info_filt["at_risk"].values
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
for tr_i, rem_i in sss1.split(X, y):
    train_ids  = X[tr_i]
    remain_ids = X[rem_i]
    y_remain   = y[rem_i]
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=N_TEST, random_state=RANDOM_STATE)
for val_i, te_i in sss2.split(remain_ids, y_remain):
    val_ids  = remain_ids[val_i]
    test_ids = remain_ids[te_i]

train_info = info_filt[info_filt["id_student"].isin(train_ids)]
val_info   = info_filt[info_filt["id_student"].isin(val_ids)]
test_info  = info_filt[info_filt["id_student"].isin(test_ids)]
print(f"    Train: {len(train_ids):>5} | at-risk: {train_info['at_risk'].mean():.1%}")
print(f"    Val:   {len(val_ids):>5} | at-risk: {val_info['at_risk'].mean():.1%}")
print(f"    Test:  {len(test_ids):>5} | at-risk: {test_info['at_risk'].mean():.1%}")
print(f"    Test outcomes: {test_info['final_result'].value_counts().to_dict()}")

# ─── 3. FEATURE BUILDER ───────────────────────────────────────────
print("\n[3/7] Preparing features...")
svle_filt = svle[
    svle["code_module"].isin(EARLY_MODULES) &
    (svle["code_presentation"] == PRESENTATION)
].copy()
svle_filt["week"] = (svle_filt["date"] // DAYS_PER_WEEK + 1).clip(1, 39)
wclicks = svle_filt.groupby(["id_student", "week"])["sum_click"].sum().reset_index()
wc_idx  = wclicks.set_index("id_student")
sa_idx  = sa_tma.set_index("id_student")
reg_lookup = info_filt.set_index("id_student")["reg_days_before"].to_dict()

FEAT_COLS = [
    "week",
    "total_clicks",           # cumulative VLE clicks up to week t
    "weekly_avg_clicks",      # average weekly clicks up to week t
    "recent_clicks",          # clicks this week only
    "click_trend",            # current week - previous week clicks
    "avg_score",              # weighted avg TMA/CMA score up to week t
    "assessments_done",       # number of assessments submitted up to week t
    "days_active",            # number of weeks with any VLE activity
    "reg_days_before",        # NEW: days before course start registered (pre-course commitment)
    "weeks_since_any_click",  # NEW: weeks of inactivity (dropout warning signal)
    "click_acceleration",     # NEW: rate of change in engagement (2nd derivative)
]

def get_features(sid, t):
    """
    Feature vector using ONLY data available up to week t.
    date_unregistration is deliberately excluded — using it would
    be data leakage since we only know it after the student drops.
    """
    try:
        c       = wc_idx.loc[sid] if sid in wc_idx.index else pd.DataFrame()
        c       = c[c["week"] <= t] if len(c) > 0 else c
        recent  = float(c[c["week"] == t]["sum_click"].sum())   if len(c) > 0 else 0
        prev    = float(c[c["week"] == t-1]["sum_click"].sum()) if t > 1 and len(c) > 0 else 0
        prev2   = float(c[c["week"] == t-2]["sum_click"].sum()) if t > 2 and len(c) > 0 else 0
        tot     = float(c["sum_click"].sum())  if len(c) > 0 else 0
        avg     = float(c["sum_click"].mean()) if len(c) > 0 else 0
        days    = int(c["week"].nunique())     if len(c) > 0 else 0
        aw      = c["week"].values if len(c) > 0 else np.array([])
        last_active     = int(aw.max()) if len(aw) > 0 else 0
        weeks_inactive  = t - last_active
        accel           = (recent - prev) - (prev - prev2)
    except Exception:
        recent = prev = prev2 = tot = avg = days = weeks_inactive = accel = 0
    try:
        s      = sa_idx.loc[sid] if sid in sa_idx.index else pd.DataFrame()
        s      = s[s["week"] <= t] if len(s) > 0 else s
        ascore = float(np.average(s["score"], weights=s["weight"].fillna(1))) if len(s) > 0 else -1
        ndone  = len(s)
    except Exception:
        ascore = -1; ndone = 0

    reg_days = float(reg_lookup.get(sid, 0) or 0)
    return [t, tot, avg, recent, recent-prev, ascore, ndone, days,
            reg_days, weeks_inactive, accel]

# ─── 4. TRAIN MATRIX ──────────────────────────────────────────────
print("\n[4/7] Building train matrix...")
sample_weeks = list(range(1, 40, 2))
BANDS = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35),(36,39)]
rows_r, rows_g = [], []
for sid in train_ids:
    ri = info_filt[info_filt["id_student"] == sid]
    if len(ri) == 0: continue
    lr = int(ri["at_risk"].values[0])
    lg = float(ri["final_grade"].values[0])
    for t in sample_weeks:
        f = get_features(sid, t)
        rows_r.append(f + [lr])
        rows_g.append(f + [lg])
df_r = pd.DataFrame(rows_r, columns=FEAT_COLS + ["label"])
df_g = pd.DataFrame(rows_g, columns=FEAT_COLS + ["label"])
print(f"    Rows: {len(df_r)}")

# ─── 5. HYPERPARAMETER TUNING ─────────────────────────────────────
print("\n[5/7] Tuning on validation set...")
np.random.seed(RANDOM_STATE)
val_sample = np.random.choice(val_ids, size=min(40, len(val_ids)), replace=False)
vrows_r, vrows_g = [], []
for sid in val_sample:
    ri = info_filt[info_filt["id_student"] == sid]
    if len(ri) == 0: continue
    lr = int(ri["at_risk"].values[0]); lg = float(ri["final_grade"].values[0])
    for t in sample_weeks:
        f = get_features(sid, t)
        vrows_r.append(f + [lr]); vrows_g.append(f + [lg])
vdf_r = pd.DataFrame(vrows_r, columns=FEAT_COLS + ["label"])
vdf_g = pd.DataFrame(vrows_g, columns=FEAT_COLS + ["label"])

PARAM_GRID = [
    {"n_estimators": 60,  "max_depth": 2},
    {"n_estimators": 80,  "max_depth": 3},
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 4},
]
best_params_r, best_params_g = {}, {}
for lo, hi in BANDS:
    tr_r = df_r[df_r["week"].between(lo,hi)]; tr_r = tr_r if len(tr_r)>=20 else df_r
    tr_g = df_g[df_g["week"].between(lo,hi)]; tr_g = tr_g if len(tr_g)>=20 else df_g
    vl_r = vdf_r[vdf_r["week"].between(lo,hi)]; vl_r = vl_r if len(vl_r)>=10 else vdf_r
    vl_g = vdf_g[vdf_g["week"].between(lo,hi)]; vl_g = vl_g if len(vl_g)>=10 else vdf_g
    best_auc, best_pr = -1, PARAM_GRID[0]
    best_mae, best_pg = 9999, PARAM_GRID[0]
    for p in PARAM_GRID:
        mr = GradientBoostingClassifier(**p, random_state=RANDOM_STATE)
        mr.fit(tr_r[FEAT_COLS].fillna(0), tr_r["label"])
        vp = mr.predict_proba(vl_r[FEAT_COLS].fillna(0))[:, 1]
        try: auc = roc_auc_score(vl_r["label"], vp)
        except: auc = 0
        if auc > best_auc: best_auc = auc; best_pr = p
        mg = GradientBoostingRegressor(**p, random_state=RANDOM_STATE)
        mg.fit(tr_g[FEAT_COLS].fillna(0), tr_g["label"])
        mae = mean_absolute_error(vl_g["label"], mg.predict(vl_g[FEAT_COLS].fillna(0)))
        if mae < best_mae: best_mae = mae; best_pg = p
    best_params_r[(lo,hi)] = best_pr; best_params_g[(lo,hi)] = best_pg
    print(f"    W{lo:02d}-W{hi:02d}: risk={best_pr} (AUC={best_auc:.3f}) | grade={best_pg} (MAE={best_mae:.2f})")

# ─── 6. TRAIN FINAL MODELS ────────────────────────────────────────
print("\n[6/7] Training final models...")
risk_models, grade_models = {}, {}
for lo, hi in BANDS:
    tr_r = df_r[df_r["week"].between(lo,hi)]; tr_r = tr_r if len(tr_r)>=20 else df_r
    tr_g = df_g[df_g["week"].between(lo,hi)]; tr_g = tr_g if len(tr_g)>=20 else df_g
    mr = GradientBoostingClassifier(**best_params_r[(lo,hi)], random_state=RANDOM_STATE)
    mr.fit(tr_r[FEAT_COLS].fillna(0), tr_r["label"])
    mg = GradientBoostingRegressor(**best_params_g[(lo,hi)], random_state=RANDOM_STATE)
    mg.fit(tr_g[FEAT_COLS].fillna(0), tr_g["label"])
    risk_models[(lo,hi)] = mr; grade_models[(lo,hi)] = mg
    print(f"    W{lo:02d}-W{hi:02d} ✓")

def get_model(T, models):
    for (lo,hi), m in models.items():
        if lo <= T <= hi: return m
    return list(models.values())[-1]

# ─── 7. PREDICTIONS + EVALUATION ─────────────────────────────────
print("\n[7/7] Generating trajectories and evaluating...")

def make_trajectories(student_ids, info_df):
    results = {}
    for sid in student_ids:
        ri = info_df[info_df["id_student"] == sid]
        if len(ri) == 0: continue
        true_risk   = int(ri["at_risk"].values[0])
        true_result = ri["final_result"].values[0]
        true_grade  = float(ri["final_grade"].values[0])
        unreg_week  = float(ri["unreg_week"].values[0]) if pd.notna(ri["unreg_week"].values[0]) else None
        reg_days    = float(ri["reg_days_before"].values[0]) if pd.notna(ri["reg_days_before"].values[0]) else 0
        real_feats  = {t: get_features(sid, t) for t in WEEKS}

        # PATH A: OBSERVED — real data accumulated at each week
        risk_obs, grade_obs = {}, {}
        for t in WEEKS:
            X = pd.DataFrame([real_feats[t]], columns=FEAT_COLS).fillna(0)
            risk_obs[t]  = round(float(get_model(t, risk_models).predict_proba(X)[0][1]), 4)
            grade_obs[t] = round(float(get_model(t, grade_models).predict(X)[0]), 2)

        # PATH B: PREDICTED — features frozen at snapshot, only week index advances
        def frozen_traj(snap):
            frozen = real_feats[snap].copy()
            rp, gp = {}, {}
            for T in range(snap, 40):
                f = frozen.copy(); f[0] = T
                X = pd.DataFrame([f], columns=FEAT_COLS).fillna(0)
                rp[T] = round(float(get_model(T, risk_models).predict_proba(X)[0][1]), 4)
                gp[T] = round(float(get_model(T, grade_models).predict(X)[0]), 2)
            return rp, gp

        rw1,  gw1  = frozen_traj(1)
        rw10, gw10 = frozen_traj(10)
        rw20, gw20 = frozen_traj(20)

        results[sid] = {
            "true_risk":       true_risk,
            "true_result":     true_result,
            "true_grade":      round(true_grade, 1),
            "unreg_week":      unreg_week,
            "reg_days_before": reg_days,
            "risk_obs":        risk_obs,   "grade_obs":  grade_obs,
            "risk_w1":         rw1,        "grade_w1":   gw1,
            "risk_w10":        rw10,       "grade_w10":  gw10,
            "risk_w20":        rw20,       "grade_w20":  gw20,
        }
    return results

test_results = make_trajectories(test_ids, test_info)

# Validation evaluation
val_sids      = [s for s in val_sample if len(info_filt[info_filt["id_student"]==s])>0]
val_true_r    = [int(info_filt[info_filt["id_student"]==s]["at_risk"].values[0])    for s in val_sids]
val_true_g    = [float(info_filt[info_filt["id_student"]==s]["final_grade"].values[0]) for s in val_sids]
val_eval_rows = []
for T in WEEKS:
    probs, gpreds = [], []
    for sid in val_sids:
        f = get_features(sid, T)
        X = pd.DataFrame([f], columns=FEAT_COLS).fillna(0)
        probs.append(float(get_model(T, risk_models).predict_proba(X)[0][1]))
        gpreds.append(float(get_model(T, grade_models).predict(X)[0]))
    preds = [1 if p >= 0.5 else 0 for p in probs]
    racc  = accuracy_score(val_true_r, preds)
    try:   rauc = round(roc_auc_score(val_true_r, probs), 4)
    except: rauc = None
    val_eval_rows.append({"week": T, "risk_acc": round(racc,4), "risk_auc": rauc,
                          "grade_mae": round(mean_absolute_error(val_true_g, gpreds), 2)})

# Test evaluation
all_tr = [test_results[s]["true_risk"]  for s in test_ids if s in test_results]
all_tg = [test_results[s]["true_grade"] for s in test_ids if s in test_results]
test_eval_rows = []
for T in WEEKS:
    probs  = [test_results[s]["risk_obs"][T]  for s in test_ids if s in test_results]
    gpreds = [test_results[s]["grade_obs"][T] for s in test_ids if s in test_results]
    preds  = [1 if p >= 0.5 else 0 for p in probs]
    racc   = accuracy_score(all_tr, preds)
    try:   rauc = round(roc_auc_score(all_tr, probs), 4)
    except: rauc = None
    test_eval_rows.append({"week": T, "risk_acc": round(racc,4), "risk_auc": rauc,
                           "grade_mae": round(mean_absolute_error(all_tg, gpreds), 2)})

# Per-group breakdown
group_eval = {}
for grp in ["Pass", "Fail", "Withdrawn", "Distinction"]:
    ids = [s for s in test_ids if s in test_results and test_results[s]["true_result"] == grp]
    if not ids: continue
    tr = [test_results[s]["true_risk"]  for s in ids]
    tg = [test_results[s]["true_grade"] for s in ids]
    rows = []
    for T in WEEKS:
        probs  = [test_results[s]["risk_obs"][T]  for s in ids]
        gpreds = [test_results[s]["grade_obs"][T] for s in ids]
        preds  = [1 if p >= 0.5 else 0 for p in probs]
        racc   = accuracy_score(tr, preds)
        try:   rauc = round(roc_auc_score(tr, probs), 4)
        except: rauc = None
        rows.append({"week": T, "risk_acc": round(racc,4), "risk_auc": rauc,
                     "grade_mae": round(mean_absolute_error(tg, gpreds), 2)})
    group_eval[grp] = {"count": len(ids), "eval": rows}

# Print summary
print(f"\n    {'Week':>5}  {'Test Acc':>9}  {'Test AUC':>9}  {'MAE':>6}  {'Val Acc':>8}")
print("    " + "─" * 48)
for r in test_eval_rows:
    if r["week"] in [1,5,10,15,20,25,30,35,39]:
        vr  = val_eval_rows[r["week"]-1]
        auc = f"{r['risk_auc']:.4f}" if r["risk_auc"] else "   N/A"
        print(f"    {r['week']:>5}  {r['risk_acc']:>8.0%}  {auc:>9}  "
              f"{r['grade_mae']:>6.2f}  {vr['risk_acc']:>7.0%}")

print("\n    Per-group at week 20:")
for grp, v in group_eval.items():
    r = next(x for x in v["eval"] if x["week"] == 20)
    print(f"      {grp:<12} n={v['count']}  acc={r['risk_acc']:.0%}  mae={r['grade_mae']:.1f}")

# ─── SAVE ─────────────────────────────────────────────────────────
save = {
    "meta": {
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "train_risk_rate": round(float(train_info["at_risk"].mean()), 4),
        "val_risk_rate":   round(float(val_info["at_risk"].mean()),   4),
        "test_risk_rate":  round(float(test_info["at_risk"].mean()),  4),
        "test_outcomes":   test_info["final_result"].value_counts().to_dict(),
        "features": FEAT_COLS,
        "new_features": ["reg_days_before", "weeks_since_any_click", "click_acceleration"],
        "best_params": {
            f"W{lo}-W{hi}": {"risk": best_params_r[(lo,hi)], "grade": best_params_g[(lo,hi)]}
            for lo, hi in BANDS
        }
    },
    "students": {
        str(sid): {
            k: ({str(kk): vv for kk,vv in v.items()} if isinstance(v, dict) else v)
            for k, v in d.items()
        }
        for sid, d in test_results.items()
    },
    "test_eval":  test_eval_rows,
    "val_eval":   val_eval_rows,
    "group_eval": {g: {"count": v["count"], "eval": v["eval"]} for g,v in group_eval.items()}
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(save, f, indent=2)

print(f"\n  ✓ Saved → {OUTPUT_FILE}")
print("  Next: python -m http.server 8000 → open dashboard.html")
print("=" * 62)
