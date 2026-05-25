"""
Student Risk & Grade Trajectory Predictor
==========================================
Run:    python 01_train_and_predict.py
Output: results.json  (read by dashboard.html)

KEY DESIGN — two separate data paths
  OBSERVED  (solid line):  at each week t, use real accumulated data → predict AT t
  PREDICTED (dashed line): freeze all features at snapshot week s,
                           only advance the week index for future weeks T.
                           Fresh data that arrives between s and T is unknown,
                           so the prediction diverges from the observed line.
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
DATA_DIR      = "."                      # folder with the 6 CSV files
EARLY_MODULES = ["AAA", "BBB", "CCC"]   # first 3 modules only
PRESENTATION  = "2013J"
N_TEST        = 8                        # test students (stratified)
DAYS_PER_WEEK = 7
WEEKS         = list(range(1, 40))
RANDOM_STATE  = 42
OUTPUT_FILE   = "results.json"

print("=" * 58)
print("  Student Risk & Grade Trajectory Predictor")
print("=" * 58)

# ─── 1. LOAD ──────────────────────────────────────────────────────
print("\n[1/6] Loading data...")
info   = pd.read_csv(f"{DATA_DIR}/studentInfo.csv")
svle   = pd.read_csv(f"{DATA_DIR}/studentVle.csv")
sa     = pd.read_csv(f"{DATA_DIR}/studentAssessment.csv")
assess = pd.read_csv(f"{DATA_DIR}/assessments.csv")

info_filt = info[
    info["code_module"].isin(EARLY_MODULES) &
    (info["code_presentation"] == PRESENTATION)
].copy()
info_filt["at_risk"] = info_filt["final_result"].isin(["Withdrawn", "Fail"]).astype(int)

# Weighted grade from TMA/CMA assessments
assess_filt = assess[
    assess["code_module"].isin(EARLY_MODULES) &
    (assess["code_presentation"] == PRESENTATION)
]
tma    = assess_filt[assess_filt["assessment_type"].isin(["TMA", "CMA"])]
sa_tma = sa.merge(tma[["id_assessment", "weight", "date"]], on="id_assessment", how="inner")
sa_tma["week"] = (sa_tma["date"] // DAYS_PER_WEEK + 1).clip(1, 39)

def compute_final_grade(sid):
    rows = sa_tma[sa_tma["id_student"] == sid].dropna(subset=["score"])
    if len(rows) == 0:
        return np.nan
    return float(np.average(rows["score"], weights=rows["weight"].fillna(1)))

info_filt["final_grade"] = info_filt["id_student"].map(compute_final_grade)
info_filt = info_filt.dropna(subset=["final_grade"])

print(f"    Students: {len(info_filt)}  |  at-risk: {info_filt['at_risk'].mean():.1%}")
print(f"    Grade range: {info_filt['final_grade'].min():.1f} – {info_filt['final_grade'].max():.1f}")

# ─── 2. STRATIFIED TRAIN/TEST SPLIT ───────────────────────────────
print(f"\n[2/6] Stratified split ({N_TEST} test students)...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=N_TEST, random_state=RANDOM_STATE)
X_ids = info_filt["id_student"].values
y_lbl = info_filt["at_risk"].values
for tr_idx, te_idx in sss.split(X_ids, y_lbl):
    train_students = X_ids[tr_idx]
    test_students  = X_ids[te_idx]

train_info = info_filt[info_filt["id_student"].isin(train_students)]
test_info  = info_filt[info_filt["id_student"].isin(test_students)]
print(f"    Train: {len(train_students)}  at-risk: {train_info['at_risk'].mean():.1%}")
print(f"    Test:  {len(test_students)}   at-risk: {test_info['at_risk'].mean():.1%}")
print("\n    Test students:")
for _, r in test_info.iterrows():
    print(f"      {r['id_student']:>10}  {r['final_result']:<12}  grade={r['final_grade']:.1f}")

# ─── 3. FEATURE BUILDER ───────────────────────────────────────────
print("\n[3/6] Preparing feature indices...")
svle_filt = svle[
    svle["code_module"].isin(EARLY_MODULES) &
    (svle["code_presentation"] == PRESENTATION)
].copy()
svle_filt["week"] = (svle_filt["date"] // DAYS_PER_WEEK + 1).clip(1, 39)
wclicks = svle_filt.groupby(["id_student", "week"])["sum_click"].sum().reset_index()
wc_idx  = wclicks.set_index("id_student")
sa_idx  = sa_tma.set_index("id_student")

FEAT_COLS = [
    "week", "total_clicks", "weekly_avg_clicks",
    "recent_clicks", "click_trend",
    "avg_score", "assessments_done", "days_active"
]

def get_features(sid, t):
    """
    Real feature vector for student sid using data available UP TO week t.
    This is the OBSERVED data path.
    """
    try:
        c      = wc_idx.loc[sid] if sid in wc_idx.index else pd.DataFrame()
        c      = c[c["week"] <= t] if len(c) > 0 else c
        recent = float(c[c["week"] == t]["sum_click"].sum())   if len(c) > 0 else 0
        prev   = float(c[c["week"] == t-1]["sum_click"].sum()) if t > 1 and len(c) > 0 else 0
        tot    = float(c["sum_click"].sum())  if len(c) > 0 else 0
        avg    = float(c["sum_click"].mean()) if len(c) > 0 else 0
        days   = int(c["week"].nunique())     if len(c) > 0 else 0
    except Exception:
        recent = prev = tot = avg = days = 0
    try:
        s      = sa_idx.loc[sid] if sid in sa_idx.index else pd.DataFrame()
        s      = s[s["week"] <= t] if len(s) > 0 else s
        ascore = float(np.average(s["score"], weights=s["weight"].fillna(1))) if len(s) > 0 else -1
        ndone  = len(s)
    except Exception:
        ascore = -1
        ndone  = 0
    # Feature vector — week index is position 0
    return [t, tot, avg, recent, recent - prev, ascore, ndone, days]

# ─── 4. TRAIN BAND MODELS ─────────────────────────────────────────
print("\n[4/6] Training models (one per 5-week band)...")
sample_weeks = list(range(1, 40, 2))   # every other week for speed
BANDS = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35),(36,39)]

rows_r, rows_g = [], []
for sid in train_students:
    ri = info_filt[info_filt["id_student"] == sid]
    if len(ri) == 0:
        continue
    lr = int(ri["at_risk"].values[0])
    lg = float(ri["final_grade"].values[0])
    for t in sample_weeks:
        f = get_features(sid, t)
        rows_r.append(f + [lr])
        rows_g.append(f + [lg])

df_r = pd.DataFrame(rows_r, columns=FEAT_COLS + ["label"])
df_g = pd.DataFrame(rows_g, columns=FEAT_COLS + ["label"])
print(f"    Training rows: {len(df_r)}")

risk_models, grade_models = {}, {}
for lo, hi in BANDS:
    sr = df_r[df_r["week"].between(lo, hi)]
    sg = df_g[df_g["week"].between(lo, hi)]
    if len(sr) < 20: sr = df_r
    if len(sg) < 20: sg = df_g

    mr = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=RANDOM_STATE)
    mr.fit(sr[FEAT_COLS].fillna(0), sr["label"])
    mg = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=RANDOM_STATE)
    mg.fit(sg[FEAT_COLS].fillna(0), sg["label"])

    risk_models[(lo, hi)]  = mr
    grade_models[(lo, hi)] = mg
    print(f"    Band W{lo:02d}–W{hi:02d} ✓")

def get_model(T, models):
    for (lo, hi), m in models.items():
        if lo <= T <= hi:
            return m
    return list(models.values())[-1]

# ─── 5. ROLLING PREDICTIONS ───────────────────────────────────────
print("\n[5/6] Generating trajectories...")

results = {}
for sid in test_students:
    ri = test_info[test_info["id_student"] == sid]
    if len(ri) == 0:
        continue
    true_risk   = int(ri["at_risk"].values[0])
    true_result = ri["final_result"].values[0]
    true_grade  = float(ri["final_grade"].values[0])

    # Pre-compute REAL features for every week (observed data path)
    real_feats = {t: get_features(sid, t) for t in WEEKS}

    # ── PATH A: OBSERVED (solid line) ─────────────────────────────
    # At each week t, use the real accumulated data to predict AT t.
    risk_obs, grade_obs = {}, {}
    for t in WEEKS:
        X = pd.DataFrame([real_feats[t]], columns=FEAT_COLS).fillna(0)
        risk_obs[t]  = round(float(get_model(t, risk_models).predict_proba(X)[0][1]), 4)
        grade_obs[t] = round(float(get_model(t, grade_models).predict(X)[0]), 2)

    # ── PATH B: PREDICTED (dashed line) ───────────────────────────
    # Freeze all features at a snapshot week s.
    # For each future week T, ONLY advance the week index — all other
    # features remain stale (no new clicks, no new scores).
    # When the real week T arrives on PATH A, it has accumulated more
    # data, so the two values differ — which is correct behaviour.
    def frozen_traj(snap_week):
        frozen = real_feats[snap_week].copy()  # locked at snap_week
        r_pred, g_pred = {}, {}
        for T in range(snap_week, 40):
            f = frozen.copy()
            f[0] = T   # only the week number changes; data stays stale
            X = pd.DataFrame([f], columns=FEAT_COLS).fillna(0)
            r_pred[T] = round(float(get_model(T, risk_models).predict_proba(X)[0][1]), 4)
            g_pred[T] = round(float(get_model(T, grade_models).predict(X)[0]), 2)
        return r_pred, g_pred

    rw1,  gw1  = frozen_traj(1)
    rw10, gw10 = frozen_traj(10)
    rw20, gw20 = frozen_traj(20)

    results[sid] = {
        "true_risk":   true_risk,
        "true_result": true_result,
        "true_grade":  round(true_grade, 1),
        # PATH A — observed
        "risk_obs":   risk_obs,
        "grade_obs":  grade_obs,
        # PATH B — predicted from frozen snapshots
        "risk_w1":    rw1,   "grade_w1":   gw1,
        "risk_w10":   rw10,  "grade_w10":  gw10,
        "risk_w20":   rw20,  "grade_w20":  gw20,
    }

    # Show divergence evidence
    diff_r = abs(risk_obs[20] - rw1.get(20, 0))
    diff_g = abs(grade_obs[20] - gw1.get(20, 0))
    print(f"    {sid} {true_result:<12}  "
          f"obs_W20={risk_obs[20]:.3f}  pred_from_W1_W20={rw1.get(20,0):.3f}  "
          f"Δrisk={diff_r:.3f}  Δgrade={diff_g:.1f}")

# ─── 6. EVALUATION ────────────────────────────────────────────────
print("\n[6/6] Evaluation (on observed trajectory)...")
all_true_risk  = [results[s]["true_risk"]  for s in test_students if s in results]
all_true_grade = [results[s]["true_grade"] for s in test_students if s in results]
eval_rows = []

for T in WEEKS:
    probs  = [results[s]["risk_obs"][T]  for s in test_students if s in results]
    gpreds = [results[s]["grade_obs"][T] for s in test_students if s in results]
    preds  = [1 if p >= 0.5 else 0 for p in probs]
    racc   = accuracy_score(all_true_risk, preds)
    try:
        rauc = round(roc_auc_score(all_true_risk, probs), 4)
    except Exception:
        rauc = None
    mae = round(mean_absolute_error(all_true_grade, gpreds), 2)
    eval_rows.append({"week": T, "risk_acc": round(racc, 4), "risk_auc": rauc, "grade_mae": mae})

print(f"\n    {'Week':>6}  {'Risk Acc':>9}  {'AUC':>8}  {'Grade MAE':>10}")
print("    " + "─" * 40)
for r in eval_rows:
    if r["week"] in [1, 5, 10, 15, 20, 25, 30, 35, 39]:
        auc_s = f"{r['risk_auc']:.4f}" if r["risk_auc"] else "   N/A"
        print(f"    {r['week']:>6}  {r['risk_acc']:>8.0%}  {auc_s:>8}  {r['grade_mae']:>10.2f}")

# ─── SAVE ─────────────────────────────────────────────────────────
save = {
    "students": {
        str(sid): {
            k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
            for k, v in d.items()
        }
        for sid, d in results.items()
    },
    "eval": eval_rows
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save, f, indent=2)

print(f"\n  ✓ Saved → {OUTPUT_FILE}")
print("  Next: python -m http.server 8000  then open dashboard.html")
print("=" * 58)
