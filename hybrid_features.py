"""
hybrid_features.py — shared feature constants for the global hybrid pipeline
============================================================================
Ordinal encodings and the ordered model-input column lists, imported by
`hybrid_features_global.py` (the active `HybridFeatureEngineerGlobal` builder).

This module used to also hold a single-cohort `HybridFeatureEngineer` class; that
local builder and its driver script were removed once the global pipeline subsumed
them. Only the constants below remain in use.

  STATIC_COLS   — per-registration covariates (demographics, registration), broadcast
  TIMEVAR_COLS  — weekly time-varying features (engagement dynamics + assessment)
  EDU/IMD/AGE_ORD — ordinal encodings for naturally-ordered categoricals
"""

WEEKS_MAX = 39
DAYS_PER_WEEK = 7
EWMA_ALPHA = 0.4

# Ordinal encodings for naturally-ordered categoricals (monotone-friendly for trees)
EDU_ORD = {
    "No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
    "HE Qualification": 3, "Post Graduate Qualification": 4,
}
IMD_ORD = {
    "0-10%": 0, "10-20": 1, "20-30%": 2, "30-40%": 3, "40-50%": 4,
    "50-60%": 5, "60-70%": 6, "70-80%": 7, "80-90%": 8, "90-100%": 9,
}
AGE_ORD = {"0-35": 0, "35-55": 1, "55<=": 2}

STATIC_COLS = [
    "prev_attempts", "studied_credits", "education_ord", "imd_ord", "age_ord",
    "disability", "gender", "reg_days_before", "registered_late",
]
TIMEVAR_COLS = [
    # course-time structure
    "week", "frac_course_elapsed", "n_assess_due_so_far",
    "weeks_to_next_deadline", "is_assessment_week",
    # engagement level (exposure-normalised)
    "clicks_total", "clicks_per_active_week", "active_weeks_ratio",
    "clicks_recent4", "clicks_ewma",
    # engagement dynamics
    "weeks_since_last_click", "click_trend_slope", "click_acceleration",
    "recent_vs_baseline_ratio", "longest_inactivity_streak", "click_entropy",
    # relative to cohort
    "clicks_pctile_vs_cohort_week",
    # assessment / achievement
    "assess_mean_score", "has_any_assessment", "assess_score_trend",
    "assess_score_std", "assess_submission_ratio", "assess_mean_days_early",
    "assess_late_share", "assess_n_missed", "assess_missed_wt_share",
    # max still-achievable coursework score (OCAS ceiling) under the grading policy
    "cw_ceiling",
]
