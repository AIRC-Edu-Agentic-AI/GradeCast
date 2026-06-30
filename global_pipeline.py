"""
GradeCast — consolidated GLOBAL pipeline  (supersedes 03_ + 04_)
================================================================
Run:    python global_pipeline.py                  (K=20 ensemble)
        K=4 python global_pipeline.py               (fast dev)
        STRATIFY=1 python global_pipeline.py        (stratified bootstrap)
Output: results_global.json   (eval + coverage + FULL test trajectories)

One global model over all 22 OULAD cohorts, with:
  • RISK  — discrete-time survival hazard, bagged over a (cluster/stratified)
            bootstrap ensemble. The ensemble MEDIAN is the point estimate; the
            10/90 percentiles are the genuine epistemic band.
  • GRADE — one XGBoost quantile model (10/50/90) → grade cone.
  • Decision thresholds + evaluation PER MODULE (heterogeneous base rates).
  • Event-time quantiles (withdrawal timing) with Mondrian (per-module) conformal
    calibration; coverage reported SEPARATELY for Withdrawn (genuine timing) and
    Fail (resolves at course end → near-trivial), per the evaluation discussion.
Split is BY STUDENT. Features are cached to _features_global.pkl.
"""

import os, sys, json, math, warnings, pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

from hybrid_features_global import (HybridFeatureEngineerGlobal, STATIC_COLS,
                                    GROUP_COLS, TIMEVAR_COLS)

warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError): pass

DATA_DIR = r"C:\Users\quock\Documents\Projects\data\OULAD"
CACHE    = "_features_global.pkl"
K        = int(os.environ.get("K", 20))
STRATIFY = os.environ.get("STRATIFY", "0") == "1"
QUANTS   = [0.1, 0.5, 0.9]
TARGET_RECALL   = 0.80          # alert-threshold recall floor (per module)
TARGET_COVERAGE = 0.80          # event-time interval coverage (per module)
TEST_FRAC, VAL_FRAC = 0.10, 0.10
SAMPLE_WEEKS = list(range(1, 40, 2))
RANDOM_STATE = 42
OUTPUT_FILE  = "results_global.json"
rng = np.random.default_rng(RANDOM_STATE)

print("=" * 66)
print(f"  GradeCast — consolidated GLOBAL pipeline (K={K}, "
      f"{'stratified' if STRATIFY else 'cluster'} bootstrap)")
print("=" * 66)

# ─── 1. FEATURES (cached) ─────────────────────────────────────────
if os.path.exists(CACHE):
    print("\n[1/8] Loading cached features...")
    FULL, static = pickle.load(open(CACHE, "rb"))
else:
    print("\n[1/8] Building global features (all 22 cohorts)...")
    eng = HybridFeatureEngineerGlobal(DATA_DIR).build()
    static = eng.static
    FULL = eng.weekly.join(static[STATIC_COLS + GROUP_COLS], on="reg_id")[eng.feat_cols].sort_index()
    pickle.dump((FULL, static), open(CACHE, "wb"))
FEAT = STATIC_COLS + GROUP_COLS + TIMEVAR_COLS
reg_idx = FULL.index.get_level_values("reg_id").values
wk_idx  = FULL.index.get_level_values("week").values
Xall = FULL.values
print(f"    regs {static.shape[0]} | rows {len(FULL):,} | feats {len(FEAT)}")

# ─── 2. SPLIT BY STUDENT ──────────────────────────────────────────
print("\n[2/8] Split by student (80/10/10, stratified on at-risk)...")
stu = static.groupby("id_student")["at_risk"].max()
su, sy = stu.index.values, stu.values
hold = VAL_FRAC + TEST_FRAC
tr_i, rem_i = next(StratifiedShuffleSplit(1, test_size=hold, random_state=RANDOM_STATE).split(su, sy))
train_stu, rem_stu, rem_y = su[tr_i], su[rem_i], sy[rem_i]
v_i, te_i = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC/hold, random_state=RANDOM_STATE).split(rem_stu, rem_y))
val_stu, test_stu, train_stu = set(rem_stu[v_i]), set(rem_stu[te_i]), set(train_stu)
rs = static["id_student"]
train_regs = set(static.index[rs.isin(train_stu)])
val_regs   = set(static.index[rs.isin(val_stu)])
test_regs  = set(static.index[rs.isin(test_stu)])
print(f"    train {len(train_regs)} | val {len(val_regs)} | test {len(test_regs)}")

# ─── 3. PERSON-PERIOD (indexed by reg for bootstrap) ──────────────
print("\n[3/8] Person-period survival table...")
ev = static["event_week"].reindex(reg_idx).values
fl = static["event_flag"].reindex(reg_idx).values
keep = wk_idx <= ev
pp_X, pp_y, pp_reg = FULL.values[keep], ((wk_idx == ev) & (fl == 1)).astype(int)[keep], reg_idx[keep]
rows_by_reg = {}
for i, r in enumerate(pp_reg):
    rows_by_reg.setdefault(r, []).append(i)
rows_by_reg = {r: np.array(ix) for r, ix in rows_by_reg.items()}
train_cohorts, train_strata = {}, {}
for r in train_regs:
    c = (static.at[r, "code_module"], static.at[r, "code_presentation"])
    train_cohorts.setdefault(c, []).append(r)
    train_strata.setdefault(c, {}).setdefault(static.at[r, "final_result"], []).append(r)
cohort_keys = list(train_cohorts.keys())
print(f"    PP rows {len(pp_y):,} (+rate {pp_y.mean():.3%})")

# ─── 4. RISK ENSEMBLE (bagged hazard → point median + band) ───────
scheme = "stratified (all cohorts; within-cohort by outcome)" if STRATIFY else "cluster (resample cohorts→students)"
print(f"\n[4/8] Training {K} bootstrap hazard members — {scheme}...")
order = np.lexsort((wk_idx, reg_idx)); inv = np.empty(len(order), np.int64); inv[order] = np.arange(len(order))
ord_reg = reg_idx[order]
risk_members = np.empty((len(FULL), K), dtype=np.float32)
for k in range(K):
    boot = []
    if STRATIFY:
        for c, strata in train_strata.items():
            for o, regs_o in strata.items():
                a = np.asarray(regs_o); boot.extend(a[rng.integers(0, len(a), len(a))])
    else:
        for c in (cohort_keys[i] for i in rng.integers(0, len(cohort_keys), len(cohort_keys))):
            a = np.asarray(train_cohorts[c]); boot.extend(a[rng.integers(0, len(a), len(a))])
    idx = np.concatenate([rows_by_reg[r] for r in boot if r in rows_by_reg])
    m = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.06, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="logloss", random_state=k, n_jobs=0)
    m.fit(pp_X[idx], pp_y[idx])
    haz = m.predict_proba(Xall)[:, 1]
    df = pd.DataFrame({"r": ord_reg, "h": haz[order]})
    risk = (1 - (1 - df["h"]).groupby(df["r"]).cumprod()).to_numpy(dtype=np.float32)
    risk_members[:, k] = risk[inv]
    print(f"    member {k+1}/{K}", end="\r")
print()
q = np.percentile(risk_members, [25, 50, 75], axis=1)   # 50% "likely" band (teacher-useful)
RB = pd.DataFrame({"reg_id": reg_idx, "week": wk_idx, "r_lo": q[0], "r_med": q[1], "r_hi": q[2]})
# NOTE: RB is now WITHDRAWAL cumulative incidence (event = withdrawal only)

# ─── 4b. ACADEMIC-FAIL ENSEMBLE — P(Fail | survive), among exam-sitters ──
print(f"[4b/8] Training {K} academic members — P(Fail | survive to exam)...")
surv_set = set(r for r in static.index if int(static.at[r, "sat_exam"]) == 1)
acad_y_pp = np.array([1 if static.at[r, "final_result"] == "Fail" else 0 for r in pp_reg], dtype=int)
surv_train_cohorts = {}
for r in train_regs:
    if r in surv_set:
        c = (static.at[r, "code_module"], static.at[r, "code_presentation"])
        surv_train_cohorts.setdefault(c, []).append(r)
surv_keys = list(surv_train_cohorts.keys())
acad_members = np.empty((len(FULL), K), dtype=np.float32)
for k in range(K):
    boot = []
    for c in (surv_keys[i] for i in rng.integers(0, len(surv_keys), len(surv_keys))):
        a = np.asarray(surv_train_cohorts[c]); boot.extend(a[rng.integers(0, len(a), len(a))])
    idx = np.concatenate([rows_by_reg[r] for r in boot if r in rows_by_reg])
    m = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.06, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="logloss", random_state=1000 + k, n_jobs=0)
    m.fit(pp_X[idx], acad_y_pp[idx])
    acad_members[:, k] = m.predict_proba(Xall)[:, 1]
    print(f"    member {k+1}/{K}", end="\r")
print()
# Proper combined band: combine the withdrawal & academic ensembles PER MEMBER, then take
# percentiles — tighter and correct vs. combining the 10th/90th separately (which over-widens).
comb_members = risk_members + (1 - risk_members) * acad_members
# EWMA-smooth per-registration trajectories: the academic head re-estimates the FINAL outcome
# every week from noisy weekly features, so raw trajectories jitter though the outcome is fixed;
# the withdrawal cuminc is already smooth. Smoothing the estimate (not the model) de-jitters.
_mi = pd.MultiIndex.from_arrays([reg_idx, wk_idx], names=["reg", "wk"])
def _smooth(v):
    return (pd.Series(v, index=_mi).groupby(level="reg", sort=False)
            .transform(lambda c: c.ewm(alpha=0.4).mean()).to_numpy(dtype=np.float32))
aq = np.percentile(acad_members, [25, 50, 75], axis=1)   # 50% "likely" band
cq = np.percentile(comb_members, [25, 50, 75], axis=1)
a_lo, a_med, a_hi = _smooth(aq[0]), _smooth(aq[1]), _smooth(aq[2])
c_lo, c_med, c_hi = _smooth(cq[0]), _smooth(cq[1]), _smooth(cq[2])
row_mod = static["code_module"].reindex(reg_idx).values
y_acad_row = (static["final_result"].reindex(reg_idx).values == "Fail").astype(int)
test_surv = np.array([(r in test_regs) and (r in surv_set) for r in reg_idx])
def _ece(p, y, bins=10):
    if not len(p): return None
    e = 0.0
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins) if b < bins - 1 else (p >= b / bins)
        if m.sum(): e += abs(p[m].mean() - y[m].mean()) * m.sum()
    return round(float(e / len(p)), 3)
acad_ece = {mod: _ece(a_med[(row_mod == mod) & test_surv], y_acad_row[(row_mod == mod) & test_surv])
            for mod in sorted(set(row_mod))}
print("    academic reliability ECE per module (test, smoothed):", acad_ece)
AB = pd.DataFrame({"reg_id": reg_idx, "week": wk_idx, "a_lo": a_lo, "a_med": a_med, "a_hi": a_hi})
CB = pd.DataFrame({"reg_id": reg_idx, "week": wk_idx, "c_lo": c_lo, "c_med": c_med, "c_hi": c_hi})

# ─── 5. GRADE QUANTILE CONE ───────────────────────────────────────
print("[5/8] Training global grade quantile model...")
grade = static["final_grade"]
samp = FULL[np.isin(wk_idx, SAMPLE_WEEKS)]; sreg = samp.index.get_level_values("reg_id")
gmask = np.array([(r in train_regs) and pd.notna(grade[r]) for r in sreg])
gm = XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QUANTS),
                  n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=0)
gm.fit(samp.values[gmask], np.array([grade[r] for r in sreg[gmask]]))
gq = np.sort(gm.predict(Xall), axis=1)
GQ = pd.DataFrame({"reg_id": reg_idx, "week": wk_idx, "g_lo": gq[:, 0], "g_med": gq[:, 1], "g_hi": gq[:, 2]})

# ─── 6. EVAL FRAME + PER-MODULE THRESHOLDS ────────────────────────
print("[6/8] Per-module thresholds + evaluation...")
EV = RB.merge(AB, on=["reg_id", "week"]).merge(CB, on=["reg_id", "week"]).merge(
    GQ, on=["reg_id", "week"]).merge(
    static[["id_student", "code_module", "at_risk", "final_grade", "final_result",
            "event_week", "event_flag", "course_weeks"]].reset_index(), on="reg_id")
EV["split"] = np.where(EV["reg_id"].isin(train_regs), "train",
              np.where(EV["reg_id"].isin(val_regs), "val", "test"))

def pinball(y, p, qq):
    d = y - p
    return float(np.mean(np.maximum(qq * d, (qq - 1) * d)))

THRESH = {}                                   # module -> {week -> thr} (recall floor, on val)
for mod, mg in EV[EV.split == "val"].groupby("code_module"):
    THRESH[mod] = {}
    for t, wg in mg.groupby("week"):
        pos = wg[wg.at_risk == 1]["c_med"].values
        best = 0.0
        if len(pos):
            for thr in np.linspace(0, 1, 201):
                if (pos >= thr).mean() >= TARGET_RECALL:
                    best = max(best, thr)
        THRESH[mod][int(t)] = round(float(best), 4)

def thr_for(mod, t): return THRESH.get(mod, {}).get(int(t), 0.5)

def eval_block(df):
    rows = []
    for t, g in df.groupby("week"):
        y = g["at_risk"].values; r = g["c_med"].values
        thr = np.array([thr_for(m, t) for m in g["code_module"]])
        pred = (r >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        auc = None
        if len(np.unique(y)) > 1:
            a = roc_auc_score(y, r); auc = round(float(a), 4) if math.isfinite(a) else None
        # component AUCs: withdrawal (vs withdrew) and academic (Fail vs Pass/Dist among survivors)
        def _auc(yy, pp):
            return round(float(roc_auc_score(yy, pp)), 4) if len(np.unique(yy)) > 1 else None
        ywd = (g["final_result"] == "Withdrawn").astype(int).values
        auc_wd = _auc(ywd, g["r_med"].values)
        surv = g[g["final_result"] != "Withdrawn"]
        auc_ac = _auc((surv["final_result"] == "Fail").astype(int).values, surv["a_med"].values) if len(surv) else None
        gg = g[g["final_grade"].notna()]; mae = pin = cov = None
        if len(gg):
            yv = gg["final_grade"].values; lo, md, hi = gg["g_lo"].values, gg["g_med"].values, gg["g_hi"].values
            mae = round(float(np.mean(np.abs(md - yv))), 2)
            pin = round(float(np.mean([pinball(yv, qa, qq) for qa, qq in zip([lo, md, hi], QUANTS)])), 3)
            cov = round(float(np.mean((yv >= lo) & (yv <= hi))), 3)
        rows.append({"week": int(t), "n": len(g), "risk_acc": round((tp + tn) / len(g), 4),
                     "risk_auc": auc, "risk_auc_withdraw": auc_wd, "risk_auc_academic": auc_ac,
                     "risk_rmse": round(float(np.sqrt(np.mean((r - y) ** 2))), 4),
                     "fpr": round(fp / (fp + tn), 4) if (fp + tn) else None,
                     "fnr": round(fn / (fn + tp), 4) if (fn + tp) else None,
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "grade_mae": mae, "grade_pinball": pin, "grade_cov80": cov})
    return rows

test = EV[EV.split == "test"]
overall_eval = eval_block(test)
per_module = {mod: {"n": int(g["reg_id"].nunique()), "eval": eval_block(g)}
              for mod, g in test.groupby("code_module")}

# ─── 7. EVENT-TIME QUANTILES + MONDRIAN CONFORMAL (Withdrawn) ─────
print("[7/8] Event-time quantiles + per-module conformal (calibrated on Withdrawn)...")
cw = static["course_weeks"]
def crossing(g, p):
    h = g.loc[g["r_med"] >= p, "week"]
    return int(h.iloc[0]) if len(h) else int(cw[g.name])
te = RB.groupby("reg_id").apply(lambda g: pd.Series(
    {"te_lo": crossing(g, 0.10), "te_med": crossing(g, 0.50), "te_hi": crossing(g, 0.90)})).astype(int)
for col in ["code_module", "final_result", "event_week", "event_flag"]:
    te[col] = static[col]
te["split"] = np.where(te.index.isin(train_regs), "train",
              np.where(te.index.isin(val_regs), "val", "test"))

# CQR expansion tuned on validation WITHDRAWN events (genuine timing)
QADJ = {}
vw = te[(te.split == "val") & (te.final_result == "Withdrawn")]
for mod, g in vw.groupby("code_module"):
    e = g["event_week"].values
    score = np.maximum(g["te_lo"].values - e, e - g["te_hi"].values)
    n = len(score)
    QADJ[mod] = float(np.quantile(score, min(1.0, np.ceil((n + 1) * TARGET_COVERAGE) / n), method="higher")) if n else 0.0

def cov_by(group_result, use_adj):
    out = {}
    sub = te[(te.split == "test") & (te.final_result == group_result)]
    for mod, g in sub.groupby("code_module"):
        e = g["event_week"].values; adj = QADJ.get(mod, 0.0) if use_adj else 0.0
        c = float(np.mean((e >= g["te_lo"].values - adj) & (e <= g["te_hi"].values + adj))) if len(g) else None
        out[mod] = {"n": int(len(g)), "coverage": round(c, 3) if c is not None else None}
    return out

coverage = {
    "withdrawn_raw": cov_by("Withdrawn", False), "withdrawn_calibrated": cov_by("Withdrawn", True),
    "fail_raw": cov_by("Fail", False),  # reported for context (near-trivial: event=course end)
}

# ─── 8. FULL TEST TRAJECTORIES + SAVE ─────────────────────────────
print("[8/8] Full test trajectories (risk band + grade cone)...")
TRAJ = RB.merge(AB, on=["reg_id", "week"]).merge(CB, on=["reg_id", "week"]).merge(GQ, on=["reg_id", "week"])
TRAJ = TRAJ[TRAJ["reg_id"].isin(test_regs)].sort_values(["reg_id", "week"])
te_test = te[te.split == "test"]
trajectories = []
for r, band in TRAJ.groupby("reg_id"):
    mod = static.at[r, "code_module"]; g = te_test.loc[r]
    trajectories.append({"reg_id": r, "code_module": mod,
        "true_result": static.at[r, "final_result"],
        "true_grade": round(float(static.at[r, "final_grade"]), 1) if pd.notna(static.at[r, "final_grade"]) else None,
        "event_week": int(static.at[r, "event_week"]), "event_flag": int(static.at[r, "event_flag"]),
        "te_lo": int(g["te_lo"]), "te_med": int(g["te_med"]), "te_hi": int(g["te_hi"]),
        "te_lo_cal": round(float(g["te_lo"]) - QADJ.get(mod, 0.0), 1),
        "te_hi_cal": round(float(g["te_hi"]) + QADJ.get(mod, 0.0), 1),
        "weeks": band["week"].tolist(),
        # risk_* = COMBINED eventual at-risk (withdraw + academic); components emitted alongside
        "risk_lo": band["c_lo"].round(4).tolist(), "risk_med": band["c_med"].round(4).tolist(),
        "risk_hi": band["c_hi"].round(4).tolist(),
        "risk_wd_med": band["r_med"].round(4).tolist(), "risk_ac_med": band["a_med"].round(4).tolist(),
        "grade_lo": band["g_lo"].round(1).tolist(), "grade_med": band["g_med"].round(1).tolist(),
        "grade_hi": band["g_hi"].round(1).tolist()})

def clean(o):
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o

# console summary
o20 = next((x for x in overall_eval if x["week"] == 20), overall_eval[-1])
print(f"\n    Overall @ wk20: AUC {o20['risk_auc']} (withdraw {o20['risk_auc_withdraw']} / "
      f"academic {o20['risk_auc_academic']}) RMSE {o20['risk_rmse']} "
      f"FNR {o20['fnr']} grade-MAE {o20['grade_mae']} cov80 {o20['grade_cov80']}")
wcal = coverage["withdrawn_calibrated"]
print("    Withdrawn event-time coverage (calibrated):",
      {m: wcal[m]["coverage"] for m in sorted(wcal)})

save = {"meta": {"model": "consolidated global hybrid (bagged survival + quantile grade)",
                 "K": K, "stratified": STRATIFY, "quantiles": QUANTS,
                 "target_recall": TARGET_RECALL, "target_coverage": TARGET_COVERAGE,
                 "n_regs": static.shape[0], "n_students": int(static["id_student"].nunique()),
                 "n_train_regs": len(train_regs), "n_val_regs": len(val_regs), "n_test_regs": len(test_regs),
                 "group_features": GROUP_COLS, "features": FEAT,
                 "thresholds_per_module": THRESH, "q_adj_weeks_per_module": {m: round(v, 2) for m, v in QADJ.items()},
                 "split": "by id_student"},
        "overall_eval": overall_eval, "per_module_eval": per_module,
        "coverage": coverage, "test_trajectories": trajectories}
json.dump(clean(save), open(OUTPUT_FILE, "w", encoding="utf-8"), indent=2, allow_nan=False)
print(f"\n  ✓ Saved → {OUTPUT_FILE}  ({len(trajectories)} trajectories)")

# ─── 9. ROLE-BASED WEB DATA (teacher rosters + leadership aggregate) ──
print("[9/9] Emitting role-based web data...")
from collections import defaultdict
WEB = "web"; os.makedirs(WEB, exist_ok=True)
o10 = next((x for x in overall_eval if x["week"] == 10), overall_eval[0])
def rec(x): return None if x is None else round(float(1 - x), 3)   # recall from fnr

# ---- per-presentation rosters (ALL students) ----
DRV = ["weeks_since_last_click", "assess_n_missed", "assess_missed_wt_share",
       "assess_submission_ratio", "recent_vs_baseline_ratio", "cw_ceiling"]
NO_GRADE_MODULES = {"GGG"}   # coursework is unweighted/absent → no trainable grade target
ALLT = RB.merge(AB, on=["reg_id", "week"]).merge(CB, on=["reg_id", "week"]).merge(
    GQ, on=["reg_id", "week"]).merge(
    FULL[DRV].reset_index(), on=["reg_id", "week"]).sort_values(["reg_id", "week"])
split_of = {r: ("train" if r in train_regs else "val" if r in val_regs else "test")
            for r in static.index}
def rd(s, n): return [round(float(v), n) for v in s]
# raw assessment marks per registration (for the teacher chart bars) — model-independent
_asm = pd.read_csv(f"{DATA_DIR}/assessments.csv")
_sa = pd.read_csv(f"{DATA_DIR}/studentAssessment.csv")
_mk = _sa.merge(_asm[["id_assessment", "code_module", "code_presentation", "assessment_type", "date", "weight"]],
                on="id_assessment", how="inner").dropna(subset=["score", "date"])
_mk["w"] = (_mk["date"] // 7 + 1).clip(lower=1).astype(int)
marks_by_reg = defaultdict(list)
for rrow in _mk.itertuples(index=False):
    rid = f"{int(rrow.id_student)}_{rrow.code_module}_{rrow.code_presentation}"
    marks_by_reg[rid].append({"w": int(rrow.w), "score": round(float(rrow.score)),
                              "type": rrow.assessment_type,
                              "wt": round(float(rrow.weight), 1) if pd.notna(rrow.weight) else None})
cohorts = defaultdict(list)
for r in static.index:
    cohorts[(static.at[r, "code_module"], static.at[r, "code_presentation"])].append(r)
n_files = 0
for (mod, pres), regs in cohorts.items():
    regset = set(regs); students = []
    for r, g in ALLT[ALLT["reg_id"].isin(regset)].groupby("reg_id"):
        students.append({
            "id": str(int(static.at[r, "id_student"]))[-4:],
            "result": static.at[r, "final_result"],
            "assessments": marks_by_reg.get(r, []),
            "grade": round(float(static.at[r, "final_grade"]), 1) if pd.notna(static.at[r, "final_grade"]) else None,
            "event_week": int(static.at[r, "event_week"]), "event_flag": int(static.at[r, "event_flag"]),
            "in_sample": split_of.get(r, "train"), "registered_late": int(static.at[r, "registered_late"]),
            "demo": {"region": static.at[r, "dem_region"], "imd": static.at[r, "dem_imd"],
                     "age": static.at[r, "dem_age"], "education": static.at[r, "dem_education"],
                     "disability": static.at[r, "dem_disability"], "gender": static.at[r, "dem_gender"]},
            "weeks": g["week"].tolist(),
            # risk = COMBINED eventual at-risk; risk_wd = dropout, risk_ac = academic/exam fail
            "risk": rd(g["c_med"], 3), "risk_lo": rd(g["c_lo"], 3), "risk_hi": rd(g["c_hi"], 3),
            "risk_wd": rd(g["r_med"], 3), "risk_wd_lo": rd(g["r_lo"], 3), "risk_wd_hi": rd(g["r_hi"], 3),
            "risk_ac": rd(g["a_med"], 3),
            "grade_med": rd(np.minimum(g["g_med"], g["cw_ceiling"]), 0),
            "grade_lo": rd(np.minimum(g["g_lo"], g["cw_ceiling"]), 0),
            "grade_hi": rd(np.minimum(g["g_hi"], g["cw_ceiling"]), 0),
            "ceiling": rd(g["cw_ceiling"], 0),
            "inactive": [int(v) for v in g["weeks_since_last_click"]],
            "missed": [int(v) for v in g["assess_n_missed"]],
            "missed_wt": rd(g["assess_missed_wt_share"], 3),
            "subratio": rd(g["assess_submission_ratio"], 2), "engratio": rd(g["recent_vs_baseline_ratio"], 2)})
    out = {"module": mod, "presentation": pres, "threshold": THRESH.get(mod, {}),
           "grade_valid": mod not in NO_GRADE_MODULES, "students": students}
    json.dump(clean(out), open(f"{WEB}/roster_{mod}_{pres}.json", "w", encoding="utf-8"), allow_nan=False)
    n_files += 1
print(f"      {n_files} roster files")

# ---- leadership aggregate ----
def at_wk(mod, wk, key):
    e = next((x for x in per_module[mod]["eval"] if x["week"] == wk), None)
    return e[key] if e else None
def prec(mod, wk):
    e = next((x for x in per_module[mod]["eval"] if x["week"] == wk), None)
    return round(e["tp"] / (e["tp"] + e["fp"]), 3) if e and (e["tp"] + e["fp"]) else None
TERM = {"2013B": 0, "2013J": 1, "2014B": 2, "2014J": 3}
IMD_ORDER = ["0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
modules_agg, trends = {}, {}
for mod, mg in static.groupby("code_module"):
    pres_agg = {}
    for pres, pg in mg.groupby("code_presentation"):
        pres_agg[pres] = {"n": int(len(pg)), "at_risk_rate": round(float(pg["at_risk"].mean()), 3),
                          "outcomes": {k: int(v) for k, v in pg["final_result"].value_counts().items()}}
    modules_agg[mod] = {"n": int(len(mg)), "at_risk_rate": round(float(mg["at_risk"].mean()), 3),
                        "withdraw_rate": round(float((mg["final_result"] == "Withdrawn").mean()), 3),
                        "fail_rate": round(float((mg["final_result"] == "Fail").mean()), 3),
                        "recall_wk10": rec(at_wk(mod, 10, "fnr")), "recall_wk20": rec(at_wk(mod, 20, "fnr")),
                        "precision_wk20": prec(mod, 20), "presentations": pres_agg}
    trends[mod] = sorted([{"presentation": p, "n": int(len(pg)),
                           "at_risk_rate": round(float(pg["at_risk"].mean()), 3)}
                          for p, pg in mg.groupby("code_presentation")],
                         key=lambda x: TERM.get(x["presentation"], 9))
def equity(col, order=None):
    rows = []
    for grp, gg in static.groupby(col):
        if pd.isna(grp) or len(gg) < 10:
            continue
        rows.append({"group": str(grp), "n": int(len(gg)),
                     "at_risk_rate": round(float(gg["at_risk"].mean()), 3),
                     "pass_rate": round(float(gg["final_result"].isin(["Pass", "Distinction"]).mean()), 3)})
    if order:
        rows.sort(key=lambda x: order.index(x["group"]) if x["group"] in order else 99)
    return rows
lead = {"overall": {"n_students": int(static["id_student"].nunique()), "n_regs": int(len(static)),
                    "at_risk_rate": round(float(static["at_risk"].mean()), 3),
                    "withdraw_rate": round(float((static["final_result"] == "Withdrawn").mean()), 3),
                    "fail_rate": round(float((static["final_result"] == "Fail").mean()), 3),
                    "recall_wk10": rec(o10["fnr"]), "recall_wk20": rec(o20["fnr"]),
                    "precision_wk20": (round(o20["tp"] / (o20["tp"] + o20["fp"]), 3) if (o20["tp"] + o20["fp"]) else None),
                    "grade_mae": o20["grade_mae"], "grade_cov80": o20["grade_cov80"]},
        "modules": modules_agg, "trends": trends,
        "equity": {"imd": equity("dem_imd", IMD_ORDER), "age": equity("dem_age", ["0-35", "35-55", "55<="]),
                   "gender": equity("dem_gender"), "disability": equity("dem_disability")},
        "meta": {"n_test_regs": len(test_regs), "note": "catch-rate/precision from held-out test; counts from full roster"}}
json.dump(clean(lead), open(f"{WEB}/leadership.json", "w", encoding="utf-8"), indent=2, allow_nan=False)
print(f"      web/leadership.json")
print("=" * 66)
