# ── config.py ──────────────────────────────────────────────────────────────
# Central place for all constants. Change values here; nothing else needs
# to be touched.

# Number of days that make one "week" in the OULAD calendar
WEEK_DAYS = 7

# How many weeks from the start of the course to begin predicting.
# Week 0 has almost no data, so we start from week 1.
START_WEEK = 1

# Minimum number of students a cohort must have to be included.
MIN_COHORT_SIZE = 10

# Where the OULAD CSV files live (relative to the project root)
DATA_DIR = "data"

# Where to write output JSON files
OUTPUT_DIR = "outputs"

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
}