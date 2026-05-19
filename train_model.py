# ── train_model.py ──────────────────────────────────────────────────────────
# 1. Load OULAD data
# 2. Build weekly features (one row per student per week)
# 3. Train XGBoost on all weeks
# 4. Predict risk score for every (student, week) pair
# 5. Save output as JSON: one entry per student with a list of weekly scores

import os
import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from feature_engineering import WeeklyFeatureEngineer
from config import DATA_DIR, OUTPUT_DIR, XGBOOST_PARAMS

# ── 1. Load data ─────────────────────────────────────────────────────────────

print("Loading OULAD tables...")
tables = {
    "studentInfo":       pd.read_csv(f"{DATA_DIR}/studentInfo.csv"),
    "studentVle":        pd.read_csv(f"{DATA_DIR}/studentVle.csv"),
    "vle":               pd.read_csv(f"{DATA_DIR}/vle.csv"),
    "assessments":       pd.read_csv(f"{DATA_DIR}/assessments.csv"),
    "studentAssessment": pd.read_csv(f"{DATA_DIR}/studentAssessment.csv"),
    "courses":           pd.read_csv(f"{DATA_DIR}/courses.csv"),
}

# ── 2. Build weekly features ─────────────────────────────────────────────────

print("\nBuilding weekly features (this takes a few minutes)...")
engineer = WeeklyFeatureEngineer(tables)
df = engineer.build_weekly_features()

# ── 3. Prepare train/test split ──────────────────────────────────────────────
# IMPORTANT: We split by student ID, not by row.
# If we split randomly, the model would see week 3 of a student during
# training and week 5 during testing — that's data leakage.
# GroupShuffleSplit ensures all weeks for a student are in the same split.

print("\nSplitting data (by student, not by row)...")

KEY_COLS = ["id_student", "code_module", "code_presentation", "week"]
DROP_COLS = KEY_COLS + ["target", "final_result"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS and c in df.columns]

X = df[FEATURE_COLS]
y = df["target"]
groups = df["id_student"]   # ensures whole students go to train or test

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"  Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")

# ── 4. Train model ───────────────────────────────────────────────────────────

print("\nTraining XGBoost model...")
model = XGBClassifier(**XGBOOST_PARAMS)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# Evaluate
probs_test = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs_test)
print(f"\nTest AUC: {auc:.4f}")

# ── 5. Predict weekly risk for ALL students ───────────────────────────────────
# We predict on the full dataset (not just test) so the dashboard gets
# a complete weekly risk trace for every student.

print("\nGenerating weekly risk scores for all students...")
all_probs = model.predict_proba(X[FEATURE_COLS])[:, 1]
df["risk_score"] = all_probs

# ── 6. Save output as JSON ────────────────────────────────────────────────────
# Output format (one entry per student):
# {
#   "id_student": 12345,
#   "code_module": "BBB",
#   "code_presentation": "2013J",
#   "final_result": "Fail",
#   "weekly_risk": [
#     {"week": 1, "risk_score": 0.21},
#     {"week": 2, "risk_score": 0.34},
#     ...
#   ]
# }

print("Formatting and saving output...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

info = tables["studentInfo"][
    ["id_student", "code_module", "code_presentation", "final_result"]
]

output = []
for (sid, module, pres), group in df.groupby(["id_student", "code_module", "code_presentation"]):
    weekly_risk = (
        group.sort_values("week")[["week", "risk_score"]]
        .assign(risk_score=lambda x: x["risk_score"].round(4))
        .rename(columns={"risk_score": "risk_score"})
        .to_dict(orient="records")
    )
    student_info = info[
        (info["id_student"] == sid) &
        (info["code_module"] == module) &
        (info["code_presentation"] == pres)
    ]
    final_result = (
        student_info["final_result"].values[0]
        if len(student_info) > 0 else "Unknown"
    )
    output.append({
        "id_student": int(sid),
        "code_module": module,
        "code_presentation": pres,
        "final_result": final_result,
        "weekly_risk": weekly_risk,
    })

out_path = os.path.join(OUTPUT_DIR, "weekly_risk_predictions.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nDone! Output saved to: {out_path}")
print(f"Total students in output: {len(output)}")

# ── 7. Print a sample ────────────────────────────────────────────────────────

print("\n=== SAMPLE OUTPUT (first student) ===")
print(json.dumps(output[0], indent=2))