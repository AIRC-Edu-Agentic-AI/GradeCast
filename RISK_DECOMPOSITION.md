# Risk decomposition — withdrawal (survival) + academic fail (exam-gated)

> **Status: IMPLEMENTED & VERIFIED (2026-06-16).** This is the current living roadmap for
> the risk head; it supersedes the Phase 1–4 plan in `GLOBAL_HYBRID_DESIGN.md` for risk.
> Implemented across `hybrid_features_global.py` (labels) and `global_pipeline.py` (two
> heads + combination + emission); surfaced in `dashboard_teacher.html` (dropout vs academic
> split) and `dashboard_global.html` (component lines). Checks in `verify_decomp.py`.
>
> **Verified vs the pre-decomposition lumped baseline @ wk20 (held-out test):**
>
> | metric | lumped | decomposed | Δ |
> |---|---|---|---|
> | combined AUC | 0.892 | **0.945** | **+0.053** |
> | RMSE (Brier) | 0.472 | **0.302** | **−0.170** |
> | recall | 0.787 | 0.780 | −0.007 (held at the 0.80 target) |
>
> Component AUCs @ wk20: withdrawal **0.88**, academic **0.91** (both climb over the term).
> Combination identity `risk = wd + (1−wd)·ac` holds exactly (0 violations / 24,510 points).
> The large RMSE drop is the headline: lumped `r_med` was cumulative-incidence-*to-date*
> (underestimates the eventual ~33% base rate); the combined head estimates P(eventual
> at-risk) directly, so it is far better calibrated.

Supersedes the lumped `at_risk = Withdrawn OR Fail` survival event. Motivated by:
- **Two different processes.** Withdrawal is a *timed behavioural* event (detectable early
  from disengagement, exam-irrelevant). Academic fail is an *exam-gated* outcome among
  students who stay (decided ~23% by the unseen exam; see decomposition below).
- **Survivorship.** Predicting academic fail should be conditioned on *survival to the exam*
  — exactly the population that has coursework to predict from.
- **Honesty.** The exam is unobservable (KHZ17); its effect is bounded by the label
  (Pass⇒exam≥40, Distinction⇒exam≥80). The academic head should be honestly *wide* where
  the exam decides (40–60 and 80+ coursework bands; CCC worst at ~30% irreducible).

## Two heads

### A. Withdrawal head (re-scoped survival)
- **Event = Withdrawn only.** Fail / Pass / Distinction are all *censored at course end*
  (they survived to the exam). Removes the fake "Fail at course-end" pseudo-event.
- Output per week W: `P(withdraw by end | data ≤ W)`. For now = cumulative incidence to
  date (`risk_wd[W]`); forward projection past W is deferred (Phase-4-lite).
- Keeps the K-member cluster-bootstrap ensemble → 10/50/90 band, and event-time / Mondrian
  conformal for *withdrawal timing*.

### B. Academic head (new)
- **Population:** survivors only (`final_result != Withdrawn`, i.e. sat the exam).
- **Target:** ordinal academic outcome `Fail(0) < Pass(1) < Distinction(2)`. These classes
  *are* the exam bands given coursework (Pass⇒exam≥40, Distinction⇒exam≥80), so predicting
  class membership = an interval-censored forecast of the final (coursework+exam) outcome
  without inventing exam point values or the unknown combination weights.
- **Model:** per-week features → ordinal/multiclass, K-member ensemble for the band.
  Output: `P(Fail | survive, W)`, plus `P(Pass)`, `P(Distinction)`.
- Rolling + step-updating as discrete assessment marks resolve.

## Combination (coherent eventual at-risk)
```
P(at-risk | W) = P(withdraw by end)  +  (1 − P(withdraw by end)) · P(Fail | survive)
```
- First term: withdrawal head. Second term: academic head.
- `at_risk` (= Withdrawn OR Fail) stays the **ground-truth** label for *combined* evaluation
  and reporting; only the survival *event* changes.
- Surface the two terms **separately** to teachers ("dropout risk" vs "academic/exam risk")
  — different interventions.

## Leakage / honesty discipline
- Exam bounds are **label-derived** → may inform the academic *target*; never a *feature*.
- The academic head only predicts *among survivors*; combined risk uses survival to weight it.

## Evaluation
- **Withdrawal head**: AUC/recall vs actual withdrawal; event-time coverage (already have).
- **Academic head**: per-week Fail-vs-rest AUC among survivors; **per-band & per-module
  coverage** of the class forecast (expect wide CCC / 40–60 / 80+).
- **Combined**: AUC/recall/FNR vs `at_risk` — must match or beat the current lumped score,
  especially early-week recall. Per-module + overall.

## Decomposition evidence (why the academic band must be wide in places)
Outcome accuracy achievable from coursework alone ≈ **76.8%** → **~23% exam-attributable**.
Concentrated at the gates: 40–60 coursework = 38/61 Fail/Pass; 80+ = 58/41 Pass/Dist.
Per-module residual: AAA 16% … CCC **30%** (tracks exam weight).

## Build order — DONE
1. ✅ **Labels** (`hybrid_features_global.py`): event = withdrawal-only; added `sat_exam`, `acad`.
2. ✅ **Pipeline** (`global_pipeline.py`): re-scoped withdrawal hazard ensemble + academic
   ensemble (`P(Fail|survive)`, survivors only); combine; per-module thresholds re-tuned on
   combined; eval reports per-head + combined AUC.
3. ✅ **Emit**: `risk` (combined), `risk_wd`, `risk_ac` to rosters; `risk_wd_med`/`risk_ac_med`
   to trajectories.
4. ✅ **Dashboards**: teacher drill-down shows dropout-vs-academic split; researcher chart
   draws both component lines + combined band.

## Remaining roadmap (priority order; all compatible with the decomposition)

The decomposition did not break the Phase 1–4 plan — it *improved* it: the academic head
forecasts the endpoint directly (no MC needed), and withdrawal is now a clean time-to-event
(forward projection is well-defined). Two **new** calibration tasks are now the frontier.

1. **Academic-head calibration** *(new; highest leverage).* The head ranks well (AUC 0.91)
   but its band is raw ensemble spread. Add per-module / per-coursework-band conformal so the
   academic interval is *honestly wide where the exam decides* (40–60, 80+, CCC ~30%). This is
   the integrity claim behind the head — ranking ≠ calibrated coverage.
2. **Forward withdrawal projection** *(was Phase 4, now scoped to withdrawal only).* Extrapolate
   the withdrawal hazard past week W so the dropout line forecasts ahead and widens with
   horizon. The academic line already forecasts ahead, so this is the *only* Phase-4 piece left.
3. **Ordinal / interval-censored academic head** *(upgrade).* Replace binary Fail-vs-rest with
   `Fail<Pass<Distinction` under the exam bounds (Pass≥40, Dist≥80) → richer Pass/Distinction
   forecast; combination still consumes `P(Fail|survive)`.
4. **Combined-band calibration** *(new).* `c_lo/c_hi` is a heuristic combination of two
   epistemic ensemble spreads, not a coverage-calibrated interval — calibrate it once (1) lands.
5. **Leadership split + README.** Emit dropout-vs-academic rates to `leadership.json` and show
   them; refresh README for the decomposed architecture.
6. **Phase 3 (local refinement)** — only if post-calibration per-module residuals justify it.
   Low priority: GGG/CCC weakness is a *data* limit (exam weight), not a modeling one.

## Intersection with the unbuilt Phase 3 hierarchy
Both heads remain *global/pooled* with module features + per-module thresholds/conformal.
Explicit local refinement (Phase 3) is still optional and orthogonal; CCC's wide academic
band is a *data* limit (exam weight), not something local refinement would fix.
