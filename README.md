# GradeCast — Weekly Student Risk & Coursework Forecasting

Part of the **Agentic Teacher Dashboard** project.
Built by Chukwuka Christian (ML Engineer) · AIRC-Edu-Agentic-AI

GradeCast turns the OULAD dataset into a **weekly, decision-support** pipeline: for every
week of a course it forecasts a student's **risk** of failing/withdrawing and their likely
**coursework** trajectory, and presents each in the view a teacher, leader, or researcher needs.

---

## What it predicts

The risk side is **decomposed into the two real processes**, because they behave differently
and need different tools:

```
risk (eventual at-risk)  =  P(withdraw by end)  +  (1 − P(withdraw)) · P(fail | survive to exam)
                            └── withdrawal head ──┘   └────── academic head ──────┘
```

- **Withdrawal head** — discrete-time survival hazard (event = *withdrawal only*), K=20
  cluster-bootstrap ensemble → cumulative-incidence band + Mondrian-conformal event-time.
  A behavioural, early-detectable, *timed* signal. Projected forward in the UI.
- **Academic head** — `P(fail | survive)` among students who sit the exam, K=20 ensemble.
  Already well-calibrated (test ECE 0.01–0.05); when its probability sits near a coin-flip it
  is flagged **"hinges on the exam."**
- **Coursework forecast** — XGBoost **quantile** regression on a policy grade (missed = 0,
  late submissions capped at 50). The cone's upper bound is capped at the **maximum still
  achievable coursework** (arithmetic, leakage-safe).

### The exam is unobservable — and that shapes everything
Per Kuzilek et al. (KHZ17), final-exam scores are missing from OULAD. So the coursework
forecast is **continuous-assessment only** (not the official grade), and the exam-decided
pass/fail lives in the **risk** head's label. We measured that coursework alone determines
~77% of outcomes; **~23% is genuinely exam-decided** (up to ~30% in exam-heavy CCC). The
system is therefore *honestly uncertain* exactly where the exam decides, rather than
pretending to know. **GGG** has zero-weighted coursework → its coursework cone is suppressed.

---

## Results (held-out test, @ week 20)

| Metric | Value |
|---|---|
| Combined risk **AUC** | **0.945** |
| Combined risk **RMSE** (Brier) | **0.302** |
| Recall (recall-tuned threshold) | ~0.78 |
| Component AUC — withdrawal / academic | 0.88 / 0.91 |
| Coursework MAE / 80% coverage | 5.2 / 0.82 |
| Academic reliability (ECE per module) | 0.01–0.05 |
| Registrations / module-presentations | 32,593 / 22 |

The decomposition beat the prior lumped model substantially (AUC 0.892 → 0.945, RMSE
0.472 → 0.302) at the same recall — see `RISK_DECOMPOSITION.md`.

---

## Repository

```
global_pipeline.py          # the consolidated pipeline (features → 2 risk heads + grade → eval → web data)
hybrid_features_global.py   # HybridFeatureEngineerGlobal: all-cohort feature builder + labels
hybrid_features.py          # shared feature-column constants
verify_decomp.py            # post-run sanity checks for the risk decomposition

index.html                  # role landing → Teacher / Leadership / Researcher
dashboard_teacher.html      # watchlist, drill-down, forecast chart (forward-projected risk),
                            #   dropout-vs-academic split, grading-scheme control, pass outlook
dashboard_leadership.html   # programme health, trends, equity lens
dashboard_global.html       # researcher view: risk band + components + coursework cone, coverage
web/                        # generated: roster_<MOD>_<PRES>.json (22) + leadership.json
results_global.json         # generated: eval + per-module metrics + full test trajectories

RISK_DECOMPOSITION.md       # current living roadmap (architecture + verified results + next work)
GLOBAL_HYBRID_DESIGN.md     # historical phase plan (Phases 1–2 built; risk re-architected)
```

---

## Setup & run

```bash
# 1. point DATA_DIR at the OULAD CSVs (in global_pipeline.py / hybrid_features_global.py)
#    default: C:/Users/quock/Documents/Projects/data/OULAD
pip install -r requirements.txt          # xgboost>=2.0, scikit-learn, pandas, numpy

# 2. run the pipeline (~12–15 min cold; ~7 min with the cached feature matrix)
python global_pipeline.py                # writes results_global.json + web/*.json
python verify_decomp.py                  # sanity-checks the decomposition

# 3. serve the dashboards
python -m http.server 8000               # open http://localhost:8000/index.html
```

The feature matrix is cached to `_features_global.pkl`; delete it to force a rebuild after
changing the feature builder or labels.

---

## Teacher-controlled grading scheme

Grading weights are conceptually a **course-creation input**. The teacher dashboard exposes a
scheme panel (exam weight, pass/distinction thresholds, compensatory vs conjunctive rule);
editing it **recomputes the deterministic guidance live** — the coursework ceiling, the
point-of-no-return, and *"to pass, needs ≥ X% on the exam"* — with no retrain. An optional
"re-fit models to this scheme" action realigns the trained predictions (pipeline run).

---

## Honesty principles

- The **coursework forecast is coursework, not the grade** — labelled as such; the exam-decided
  outcome is the risk head's job.
- **Never** feed the exam (or anything imputed from `final_result`) back as a feature — leakage.
- Bands mean what they say: withdrawal cumulative-incidence is calibrated on event-time;
  academic probability is reliability-checked; the coursework cone is the quantile model's 10/90
  capped by what's arithmetically reachable.
- Figures shown in the dashboards are historical (hindsight) for demonstration; live use predicts
  ahead. Forecasts are decision support, not deterministic predictions.
