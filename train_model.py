import pandas as pd
from feature_engineering import FeatureEngineer

print("Loading data...")

tables = {
    "studentInfo": pd.read_csv("data/studentInfo.csv"),
    "studentVle": pd.read_csv("data/studentVle.csv"),
    "vle": pd.read_csv("data/vle.csv"),
    "assessments": pd.read_csv("data/assessments.csv"),
    "studentAssessment": pd.read_csv("data/studentAssessment.csv"),
    "courses": pd.read_csv("data/courses.csv")
}

print("Building features...")

fe = FeatureEngineer(tables, cutoff_pct=0.5)
features = fe.build_all_features()
print(features.columns)

print("Preparing data...")


features["week"] = features.groupby("id_student").cumcount()

id_cols = features[["id_student", "week"]]

print(features.columns)

X = features.drop(columns=[
    "target",
    "final_result",
    "code_module",
    "code_presentation"
])
y = features["target"]

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")

model = XGBClassifier()
model.fit(X_train, y_train)

print("Predicting...")

probs = model.predict_proba(X_test)

output = id_cols.loc[X_test.index].copy()
output["risk"] = probs[:, 1]

print("\n=== SAMPLE OUTPUT ===")
print(output[["id_student", "week", "risk"]].head())