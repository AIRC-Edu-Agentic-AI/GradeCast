"""
hybrid_features_global.py — Phase 1 global feature engineering
==============================================================
Generalises HybridFeatureEngineer to ALL OULAD module/presentation cohorts.

Key differences vs the single-cohort builder:
  - Unit = registration (id_student, code_module, code_presentation) → reg_id.
    id_student can appear in several cohorts; we keep id_student so the pipeline
    can split BY STUDENT (no person spans train/test).
  - Variable course length: weeks 1..n_weeks(cohort); frac_course_elapsed and
    unreg week are normalised by course length so cohorts are comparable.
  - GROUP features (module / presentation / course length) let one global model
    split on cohort instead of training 22 separate models.
  - cohort-relative click percentile is computed within each (cohort, week).

Reuses the proven per-cohort vectorised click/assessment/dynamics logic; just
loops cohorts and pools, loading the (large) CSVs only once.
"""

import numpy as np
import pandas as pd
from hybrid_features import (EDU_ORD, IMD_ORD, AGE_ORD, EWMA_ALPHA, DAYS_PER_WEEK,
                             STATIC_COLS, TIMEVAR_COLS)

WMAX = 40                       # global week grid ceiling (max course ≈ 39 wks)
MODULE_ORD = {m: i for i, m in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"])}
GROUP_COLS = ["module_ord", "pres_year", "pres_term_oct", "course_weeks"]


class HybridFeatureEngineerGlobal:
    def __init__(self, data_dir, modules=None, presentations=None, sample_frac=1.0):
        self.data_dir = data_dir
        self.modules = modules                 # None → all
        self.presentations = presentations
        self.sample_frac = sample_frac         # <1 → downsample students per cohort (quick runs)
        self.feat_cols = STATIC_COLS + GROUP_COLS + TIMEVAR_COLS

    # ── load once ────────────────────────────────────────────────────────────
    def _load(self):
        d = self.data_dir
        self.info_all = pd.read_csv(f"{d}/studentInfo.csv")
        self.svle_all = pd.read_csv(f"{d}/studentVle.csv")
        self.sa_all = pd.read_csv(f"{d}/studentAssessment.csv")
        self.assess_all = pd.read_csv(f"{d}/assessments.csv")
        self.sreg_all = pd.read_csv(f"{d}/studentRegistration.csv")
        self.vle_all = pd.read_csv(f"{d}/vle.csv")
        self.courses = (pd.read_csv(f"{d}/courses.csv")
                        .set_index(["code_module", "code_presentation"])
                        ["module_presentation_length"].to_dict())
        cohorts = self.info_all[["code_module", "code_presentation"]].drop_duplicates()
        if self.modules:
            cohorts = cohorts[cohorts["code_module"].isin(self.modules)]
        if self.presentations:
            cohorts = cohorts[cohorts["code_presentation"].isin(self.presentations)]
        self.cohorts = list(cohorts.itertuples(index=False, name=None))

    # ── one cohort → (weekly_df, static_df) ──────────────────────────────────
    def _process_cohort(self, mod, pres):
        info = self.info_all[(self.info_all.code_module == mod) &
                             (self.info_all.code_presentation == pres)].copy()
        if len(info) == 0:
            return None, None
        if self.sample_frac < 1.0 and len(info) > 30:
            info = info.sample(frac=self.sample_frac, random_state=42)
        sids = info["id_student"].tolist()
        sidx = {s: i for i, s in enumerate(sids)}
        n = len(sids)
        nwk = int(np.clip(self.courses.get((mod, pres), 255) // DAYS_PER_WEEK + 1, 1, WMAX))
        weeks = list(range(1, nwk + 1))

        svle = self.svle_all[(self.svle_all.code_module == mod) &
                             (self.svle_all.code_presentation == pres) &
                             (self.svle_all.id_student.isin(sids))].copy()
        vle = self.vle_all[(self.vle_all.code_module == mod) &
                           (self.vle_all.code_presentation == pres)]
        assess = self.assess_all[(self.assess_all.code_module == mod) &
                                 (self.assess_all.code_presentation == pres)].copy()
        sa = self.sa_all[self.sa_all.id_student.isin(sids)]
        sreg = self.sreg_all[(self.sreg_all.code_module == mod) &
                             (self.sreg_all.code_presentation == pres)]

        # ---- click matrix (n × nwk) ----
        svle["week"] = (svle["date"] // DAYS_PER_WEEK + 1).clip(1, nwk)
        M = (svle.groupby(["id_student", "week"])["sum_click"].sum().unstack(fill_value=0)
             .reindex(index=sids, columns=weeks, fill_value=0).values.astype(float))
        cum = np.cumsum(M, axis=1)
        active = (M > 0).astype(float)
        active_cum = np.cumsum(active, axis=1)
        pctile = pd.DataFrame(cum).rank(axis=0, pct=True).values

        # ---- activity entropy ----
        sv2 = svle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
        acts = sv2["activity_type"].dropna().unique()
        if len(acts):
            stack = []
            for a in acts:
                piv = (sv2[sv2.activity_type == a].groupby(["id_student", "week"])["sum_click"].sum()
                       .unstack(fill_value=0).reindex(index=sids, columns=weeks, fill_value=0)
                       .values.astype(float))
                stack.append(np.cumsum(piv, axis=1))
            stack = np.stack(stack, 0)
            tot = stack.sum(0) + 1e-9
            probs = stack / tot
            ent = -(np.where(probs > 0, probs * np.log2(probs + 1e-12), 0.0)).sum(0)
        else:
            ent = np.zeros((n, nwk))

        # ---- time-varying features (vectorised over weeks) ----
        F = {c: np.zeros((n, nwk)) for c in TIMEVAR_COLS}
        last_active = np.zeros(n); streak = np.zeros(n); longest = np.zeros(n)
        ewma = np.zeros(n); Sy = np.zeros(n); Sxy = np.zeros(n)
        for j in range(nwk):
            t = j + 1
            mw = M[:, j]
            ewma = EWMA_ALPHA * mw + (1 - EWMA_ALPHA) * ewma
            streak = np.where(mw == 0, streak + 1, 0); longest = np.maximum(longest, streak)
            last_active = np.where(mw > 0, t, last_active)
            Sy += mw; Sxy += t * mw
            Sx = t * (t + 1) / 2.0; Sxx = t * (t + 1) * (2 * t + 1) / 6.0
            denom = t * Sxx - Sx * Sx
            slope = np.where(denom != 0, (t * Sxy - Sx * Sy) / denom, 0.0)
            prev = M[:, j - 1] if j >= 1 else np.zeros(n)
            prev2 = M[:, j - 2] if j >= 2 else np.zeros(n)
            recent4 = cum[:, j] - (cum[:, j - 4] if j >= 4 else 0.0)
            baseline_mean = cum[:, j] / t
            last2 = (mw + prev) / 2.0 if t >= 2 else mw
            F["week"][:, j] = t
            F["frac_course_elapsed"][:, j] = t / nwk
            F["clicks_total"][:, j] = cum[:, j]
            F["clicks_per_active_week"][:, j] = np.where(active_cum[:, j] > 0, cum[:, j] / active_cum[:, j], 0.0)
            F["active_weeks_ratio"][:, j] = active_cum[:, j] / t
            F["clicks_recent4"][:, j] = recent4
            F["clicks_ewma"][:, j] = ewma
            F["weeks_since_last_click"][:, j] = np.where(last_active > 0, t - last_active, t)
            F["click_trend_slope"][:, j] = slope
            F["click_acceleration"][:, j] = (mw - prev) - (prev - prev2)
            F["recent_vs_baseline_ratio"][:, j] = last2 / (baseline_mean + 1e-5)
            F["longest_inactivity_streak"][:, j] = longest
            F["click_entropy"][:, j] = ent[:, j]
            F["clicks_pctile_vs_cohort_week"][:, j] = pctile[:, j]

        # ---- assessment features ----
        tcma = assess[assess.assessment_type.isin(["TMA", "CMA"])].copy()
        tcma["deadline_week"] = (tcma["date"] // DAYS_PER_WEEK + 1).clip(1, nwk)
        dls = np.array(sorted(tcma["deadline_week"].dropna().astype(int).tolist()), dtype=float)
        am = sa.merge(tcma[["id_assessment", "deadline_week", "weight", "date"]],
                      on="id_assessment", how="inner").dropna(subset=["deadline_week"])
        am["submit_week"] = (am["date_submitted"] // DAYS_PER_WEEK + 1).clip(1, nwk)
        am["days_early"] = am["date"] - am["date_submitted"]
        am["weight"] = am["weight"].fillna(1)
        # cohort cumulative graded-weight DUE by each week (for weight-aware "missed")
        tdw = tcma.dropna(subset=["deadline_week"]).copy()
        tdw["weight"] = tdw["weight"].fillna(1)
        due_wt_week = np.array([float(tdw.loc[tdw["deadline_week"] <= (j + 1), "weight"].sum())
                                for j in range(nwk)])
        # full coursework schedule, for policy grade (missed=0, late capped at 50)
        # and the max-still-achievable OCAS ceiling
        sched_ids = tdw["id_assessment"].values
        sched_dl = tdw["deadline_week"].values.astype(float)
        sched_w = tdw["weight"].values.astype(float)
        W_total = float(sched_w.sum())
        by_sid = {s: g for s, g in am.groupby("id_student")}
        for sid in sids:
            i = sidx[sid]
            g = by_sid.get(sid)
            s_week = g["submit_week"].values if g is not None else np.array([])
            score = g["score"].values if g is not None else np.array([])
            d_early = g["days_early"].values if g is not None else np.array([])
            w_sub = g["weight"].values if g is not None else np.array([])
            dl_sub = g["deadline_week"].values if g is not None else np.array([])
            # align this student's effective scores to the full schedule (policy applied)
            eff_arr = np.zeros(sched_ids.size); subw_arr = np.full(sched_ids.size, np.inf)
            if g is not None and sched_ids.size:
                gm = {int(r.id_assessment): (r.submit_week, r.score, r.days_early)
                      for r in g.itertuples()}
                for k, aid in enumerate(sched_ids):
                    v = gm.get(int(aid))
                    if v is not None:
                        sw, sc, de = v
                        e = 0.0 if pd.isna(sc) else float(sc)
                        if de < 0:           # submitted after deadline → raw cap at 50
                            e = min(e, 50.0)
                        eff_arr[k] = e; subw_arr[k] = sw
            for j in range(nwk):
                t = j + 1
                if dls.size:
                    F["n_assess_due_so_far"][i, j] = (dls <= t).sum()
                    F["is_assessment_week"][i, j] = float((dls == t).any())
                    fut = dls[dls >= t]; F["weeks_to_next_deadline"][i, j] = (fut.min() - t) if fut.size else 0.0
                due = int((dls <= t).sum()) if dls.size else 0
                # weight-aware missed: share of graded weight DUE that wasn't submitted
                due_wt = due_wt_week[j]
                if due_wt > 0:
                    msub = (s_week <= t) & (dl_sub <= t)
                    sub_wt = float(w_sub[msub].sum()) if w_sub.size else 0.0
                    F["assess_missed_wt_share"][i, j] = max(0.0, (due_wt - sub_wt) / due_wt)
                # max still-achievable OCAS: submitted→locked score, not-yet-due→up to 100,
                # past-due unsubmitted→0. At the final week this equals the policy grade.
                if W_total > 0:
                    sub_by_t = subw_arr <= t
                    contrib = np.where(sub_by_t, eff_arr * sched_w,
                                       np.where(sched_dl > t, 100.0 * sched_w, 0.0))
                    F["cw_ceiling"][i, j] = contrib.sum() / W_total
                else:
                    F["cw_ceiling"][i, j] = 100.0      # no graded coursework (GGG): non-binding
                sub = s_week <= t
                if sub.any():
                    sc = score[sub]
                    F["assess_mean_score"][i, j] = np.nanmean(sc) if np.isfinite(sc).any() else -1.0
                    F["has_any_assessment"][i, j] = 1.0
                    F["assess_score_std"][i, j] = np.nanstd(sc) if sc.size > 1 else 0.0
                    F["assess_submission_ratio"][i, j] = sub.sum() / due if due else 0.0
                    F["assess_mean_days_early"][i, j] = np.nanmean(d_early[sub])
                    F["assess_late_share"][i, j] = (d_early[sub] < 0).sum() / due if due else 0.0
                    F["assess_n_missed"][i, j] = max(0, due - int(sub.sum()))
                    if sub.sum() >= 2:
                        order = np.argsort(s_week[sub]); ys = sc[order]
                        F["assess_score_trend"][i, j] = np.polyfit(np.arange(len(ys)), ys, 1)[0]
                else:
                    F["assess_mean_score"][i, j] = -1.0
                    F["assess_n_missed"][i, j] = due

        # ---- weekly long frame ----
        reg_ids = [f"{s}_{mod}_{pres}" for s in sids]
        idx = pd.MultiIndex.from_product([reg_ids, weeks], names=["reg_id", "week"])
        weekly = pd.DataFrame({c: F[c].reshape(-1) for c in TIMEVAR_COLS}, index=idx)

        # ---- static + labels + group ----
        st = info.copy()
        # readable demographics (captured before the columns are re-encoded) —
        # used by the teacher/leadership dashboards, not as model features
        st["dem_region"] = st["region"]
        st["dem_imd"] = st["imd_band"]
        st["dem_age"] = st["age_band"]
        st["dem_education"] = st["highest_education"]
        st["dem_disability"] = st["disability"]
        st["dem_gender"] = st["gender"]
        st["prev_attempts"] = st["num_of_prev_attempts"]
        st["education_ord"] = st["highest_education"].map(EDU_ORD).fillna(-1)
        st["imd_ord"] = st["imd_band"].map(IMD_ORD).fillna(-1)
        st["age_ord"] = st["age_band"].map(AGE_ORD).fillna(-1)
        st["disability"] = (st["disability"] == "Y").astype(int)
        st["gender"] = (st["gender"] == "M").astype(int)
        st = st.merge(sreg[["id_student", "date_registration", "date_unregistration"]],
                      on="id_student", how="left")
        st["reg_days_before"] = st["date_registration"].fillna(0)
        st["registered_late"] = (st["date_registration"].fillna(0) > 0).astype(int)
        st["module_ord"] = MODULE_ORD.get(mod, -1)
        st["pres_year"] = int(pres[:4])
        st["pres_term_oct"] = 1 if pres.endswith("J") else 0     # J=Oct start, B=Feb
        st["course_weeks"] = nwk
        st["at_risk"] = st["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
        # academic outcome among students who sat the exam (Withdrawn → NaN):
        #   Fail(0) < Pass(1) < Distinction(2)  — these classes are the exam bands
        st["sat_exam"] = (st["final_result"] != "Withdrawn").astype(int)
        st["acad"] = st["final_result"].map({"Fail": 0, "Pass": 1, "Distinction": 2})
        # policy coursework grade = ceiling at the final week (missed=0, late capped at 50);
        # GGG has no graded coursework → fall back to the submitted-mean (cone suppressed in UI)
        if W_total > 0:
            st["final_grade"] = st["id_student"].map(
                lambda s: float(F["cw_ceiling"][sidx[s], nwk - 1]) if s in sidx else np.nan)
        else:
            st["final_grade"] = st["id_student"].map(self._grades(am))
        uw = (st["date_unregistration"] / DAYS_PER_WEEK).round()
        st["unreg_week"] = uw
        # survival event = WITHDRAWAL only; Fail/Pass/Distinction are censored at course end
        # (they survived to the exam — academic fail is handled by the academic head)
        ev, fl = [], []
        for res, w in zip(st["final_result"], uw):
            if res == "Withdrawn":
                ev.append(int(np.clip(w, 1, nwk)) if pd.notna(w) else nwk); fl.append(1)
            else:
                ev.append(nwk); fl.append(0)
        st["event_week"] = ev; st["event_flag"] = fl
        st["reg_id"] = reg_ids
        keep = (["reg_id", "id_student", "code_module", "code_presentation"] +
                STATIC_COLS + GROUP_COLS +
                ["dem_region", "dem_imd", "dem_age", "dem_education", "dem_disability", "dem_gender"] +
                ["final_result", "at_risk", "sat_exam", "acad",
                 "final_grade", "unreg_week", "event_week", "event_flag"])
        static = st[keep].set_index("reg_id")
        return weekly, static

    @staticmethod
    def _grades(am):
        m = am.dropna(subset=["score"])
        out = {}
        for sid, g in m.groupby("id_student"):
            w = g["weight"].values
            out[sid] = (float(np.average(g["score"], weights=w))
                        if w.sum() > 0 else float(g["score"].mean()))
        return out

    # ── public ───────────────────────────────────────────────────────────────
    def build(self):
        self._load()
        wks, sts = [], []
        for mod, pres in self.cohorts:
            w, s = self._process_cohort(mod, pres)
            if w is not None:
                wks.append(w); sts.append(s)
        self.weekly = pd.concat(wks).sort_index()
        self.static = pd.concat(sts)
        self.regs = self.static.index.tolist()
        self._static_vec = self.static[STATIC_COLS + GROUP_COLS]
        return self
