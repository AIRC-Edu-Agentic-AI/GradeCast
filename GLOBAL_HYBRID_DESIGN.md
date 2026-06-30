# GradeCast — Global + Local Hybrid with Calibrated Risk Quantiles

> **Historical design (Phases 1–4).** Phases 1–2 are implemented; the per-script names
> below (`03_*.py`, `04_*.py`) were **consolidated into `global_pipeline.py`** and removed.
> The **risk head has since been re-architected** — see `RISK_DECOMPOSITION.md`, which is the
> current living roadmap (withdrawal survival + academic head + calibration tasks). Read this
> file for the original phase rationale only.

Phased design for evolving the current single-cohort hybrid (survival risk +
quantile grade, AAA/BBB/CCC · 2013J) into a **global model with adaptive local
refinement and genuine, per-module-calibrated uncertainty**.

## Context & goal

Today's pipeline ([02_hybrid_train_and_predict.py](02_hybrid_train_and_predict.py))
is scoped to one presentation (2,620 students). Full OULAD is ~32,600 students
across 22 module/presentation cohorts. Two improvements are wanted and they
**compose** rather than compete:

1. **Global + local** — pool across all modules for statistical strength
   (especially weak early-week signal), while respecting genuine module
   heterogeneity (course length, assessment structure, base rates).
2. **Real risk quantiles** — a forecast band that is *earned* (estimation +
   future-trajectory uncertainty), not a fabricated envelope, matching the
   honesty of the grade cone.

### The unifying idea
- *Global vs local* = where the point estimate comes from (how much we pool).
- *Quantiles* = how uncertain that estimate is.
- They meet in exactly two places:
  - **(A)** the global↔local disagreement is itself a source of uncertainty, so
    it lives *inside* the quantile machinery (via group-level resampling) — not
    bolted on beside it (which would double-count).
  - **(B)** calibration must be **per-module** (Mondrian conformal); this is
    where "local" does its most important work for uncertainty.

### No-double-count principle (single generative story)
Each layer owns one distinct uncertainty source:
`cluster-bootstrap ensemble of (global+local)` = epistemic →
`Monte-Carlo forward simulation` = future-trajectory →
`per-module conformal` = calibration. Nothing is counted twice; every band has a
defined meaning.

## Target architecture (end state)

```
Level 0  Global base        hazard + grade-quantile on ALL cohorts,
                            group features + course-length-normalized time
Level 1  Local refinement   partial pooling (small modules → shrink to global),
                            residual fine-tune (large modules)
Uncert.  Risk band          cluster-bootstrap ensemble of L0+L1  → risk_q10/50/90(t)
                            + MC forward simulation (module-conditioned) for horizon
         Grade band         quantile head (already), locally adjusted
Calib.   Mondrian conformal  per-module band-width calibration on validation
Decision Per-module dial    per-module×week alert thresholds on a chosen quantile
```

## Cross-cutting requirements
- **Unit of analysis**: a registration `(id_student, code_module, code_presentation)`.
  Split **by `id_student`** so the same person never spans train/test.
- **Time normalization**: course length varies (`courses.module_presentation_length`).
  Use `frac_course_elapsed = week / n_weeks(course)`; scale `unreg_week` by course
  length; keep raw `week` too. Global week grid = max weeks across courses.
- **Cohort-relative features**: `clicks_pctile_vs_cohort_week` computed within each
  `(module, presentation, week)` — matters more globally (absolute clicks differ
  by module).
- **Group features**: module (ordinal/one-hot), presentation year + term (B=Feb /
  J=Oct), `module_presentation_length`.
- **Leakage discipline (unchanged)**: only data with `date ≤ cutoff`; assessments
  only once due; `date_unregistration` is the survival label only, never a feature.
- **Evaluation**: report **per module × presentation** as well as overall; the key
  honesty plot is **per-module `cov80`** for both grade and risk (event-time
  coverage for uncensored students).

---

## Phase 1 — Global base + group features + per-module thresholds  ← START HERE
**Goal:** one global hazard + grade-quantile model that matches or beats the local
model per module, with per-module operating points and a per-module eval breakdown.
This is the point-estimate foundation everything else wraps.

**Build**
- `hybrid_features_global.py` — `HybridFeatureEngineerGlobal`: process each
  `(module, presentation)` cohort with the existing vectorized click/assessment
  logic, append group + course-length features, pool into one frame keyed by
  `reg_id`. Course-length-normalized time + per-cohort cohort-percentile.
- `03_global_train_and_predict.py` — split by `id_student` (80/10/10, stratified
  on at-risk); train one global `XGBClassifier` hazard + one global
  `XGBRegressor` quantile grade with group features; per-module per-week alert
  thresholds tuned on validation; per-module + overall evaluation; write
  `results_global.json`.

**Success criteria**
- Runs on all 22 cohorts; per-module AUC/RMSE/grade-cov reported.
- Global model's per-module AUC ≥ local single-cohort baseline on AAA/BBB/CCC
  (sanity that pooling didn't hurt the cohort we already know).
- Per-module `cov80` and recall surfaced; per-module thresholds stored in meta.

## Phase 2 — Calibrated risk quantiles (ensemble + Mondrian conformal)
**Goal:** genuine risk band, per-module honest. Combines both ideas with no
explicit local models yet.
- **Cluster-bootstrap ensemble**: resample modules/presentations (block bootstrap)
  + students, refit the global pipeline K≈30 times → percentiles of cumulative
  incidence at each week = `risk_q10/50/90(t)`. Group resampling makes the band
  widen for under-represented modules automatically.
- **Mondrian (per-module) conformal** calibration on validation so per-module
  coverage hits target (global-only conformal mis-covers small vs large modules).
- Verify with **per-module event-time coverage** (does actual event week fall in
  `[q10,q90]` for uncensored students).

## Phase 3 — Explicit local refinement
Only where Phase 1 per-module residuals justify it:
- Partial pooling / shrinkage for small modules (auto-fallback to global).
- Residual fine-tune for large modules.
- Fold the global↔local gap into the Phase 2 ensemble (do **not** add a separate
  disagreement term — avoids double-counting).

## Phase 4 — Monte-Carlo forward simulation (true forecast cone)
- Module-conditioned next-week feature-evolution model (distributional /
  quantile regressor for engagement given recent state).
- Roll forward N sampled paths from the current week → forecast risk quantiles
  that widen with horizon. This is what turns the band into a real *forecast*
  rather than an interval on the observed curve.

## Sequencing rationale
Phase 1 fixes the foundation and tells us (via per-module residuals) whether
Phase 3 is even needed. Phase 2 already delivers both headline ideas (uncertainty
+ locality, through group resampling and Mondrian conformal) before any explicit
local model is built. Phase 4 is the largest new modeling effort and comes last.
