import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import WEEK_DAYS


class FeatureEngineer:
    """
    Builds feature matrix for a given cutoff percentage.
    Features:
      (A) Demographic / static
      (B) Click-based behavioral
      (C) Trend (slope, window ratio, decline streaks, momentum)
      (D) Assessment
    """

    def __init__(self, tables: dict, cutoff_pct: float):
        self.tables = tables
        self.cutoff_pct = cutoff_pct
        self.course_lengths = (
            tables["courses"]
            .set_index(["code_module", "code_presentation"])["module_presentation_length"]
            .to_dict()
        )

    def _get_cutoff_day(self, module: str, presentation: str) -> int:
        length = self.course_lengths[(module, presentation)]
        return int(length * self.cutoff_pct)

    def build_demographic_features(self, info: pd.DataFrame) -> pd.DataFrame:
        df = info.copy()
        cat_cols = ["gender", "region", "highest_education",
                    "imd_band", "age_band", "disability"]
        for col in cat_cols:
            le = LabelEncoder()
            df[f"demo_{col}"] = le.fit_transform(df[col].astype(str))
        df["demo_num_prev_attempts"] = df["num_of_prev_attempts"]
        df["demo_studied_credits"] = df["studied_credits"]
        feature_cols = [c for c in df.columns if c.startswith("demo_")]
        key_cols = ["code_module", "code_presentation", "id_student"]
        return df[key_cols + feature_cols]

    def build_click_features(self, info: pd.DataFrame) -> pd.DataFrame:
        svle = self.tables["studentVle"].copy()
        vle = self.tables["vle"][["id_site", "code_module", "code_presentation", "activity_type"]]
        svle = svle.merge(vle, on=["id_site", "code_module", "code_presentation"], how="left")

        results = []
        for (module, pres), group in info.groupby(["code_module", "code_presentation"]):
            cutoff_day = self._get_cutoff_day(module, pres)
            student_ids = group["id_student"].values

            mask = (
                (svle["code_module"] == module) &
                (svle["code_presentation"] == pres) &
                (svle["id_student"].isin(student_ids)) &
                (svle["date"] <= cutoff_day)
            )
            cohort_vle = svle[mask].copy()
            cohort_vle["week"] = (cohort_vle["date"] // WEEK_DAYS).clip(lower=0)
            n_weeks = max(1, cutoff_day // WEEK_DAYS)

            for sid in student_ids:
                s_data = cohort_vle[cohort_vle["id_student"] == sid]
                feat = {"code_module": module, "code_presentation": pres, "id_student": sid}

                total_clicks = s_data["sum_click"].sum()
                n_active_days = s_data["date"].nunique()
                n_elapsed_days = max(1, cutoff_day + 20)

                feat["click_total"] = total_clicks
                feat["click_mean_per_day"] = total_clicks / n_elapsed_days
                feat["click_std_per_day"] = (
                    s_data.groupby("date")["sum_click"].sum().std()
                    if n_active_days > 1 else 0
                )
                feat["click_active_days"] = n_active_days
                feat["click_engagement_ratio"] = n_active_days / n_elapsed_days

                for act in ["forumng", "oucontent", "resource", "quiz", "homepage", "subpage"]:
                    feat[f"click_{act}"] = s_data[s_data["activity_type"] == act]["sum_click"].sum()

                act_counts = s_data.groupby("activity_type")["sum_click"].sum()
                if len(act_counts) > 0:
                    probs = act_counts / act_counts.sum()
                    feat["click_activity_entropy"] = -(probs * np.log2(probs + 1e-10)).sum()
                else:
                    feat["click_activity_entropy"] = 0

                weekly_clicks = (
                    s_data.groupby("week")["sum_click"].sum()
                    .reindex(range(n_weeks + 1), fill_value=0)
                )

                if len(weekly_clicks) >= 2:
                    weeks_arr = np.arange(len(weekly_clicks))
                    feat["trend_slope"] = (
                        np.polyfit(weeks_arr, weekly_clicks.values, 1)[0]
                        if np.std(weekly_clicks.values) > 0 else 0.0
                    )
                else:
                    feat["trend_slope"] = 0.0

                mid = max(1, len(weekly_clicks) // 2)
                feat["trend_window_ratio"] = (
                    weekly_clicks.iloc[mid:].mean() / (weekly_clicks.iloc[:mid].mean() + 1e-5)
                )

                diffs = weekly_clicks.diff().dropna()
                longest_decline = current_decline = 0
                for d in diffs:
                    if d < 0:
                        current_decline += 1
                        longest_decline = max(longest_decline, current_decline)
                    else:
                        current_decline = 0
                feat["trend_longest_decline"] = longest_decline

                longest_inactive = current_inactive = 0
                for c in weekly_clicks:
                    if c == 0:
                        current_inactive += 1
                        longest_inactive = max(longest_inactive, current_inactive)
                    else:
                        current_inactive = 0
                feat["trend_longest_inactivity"] = longest_inactive

                overall_avg = weekly_clicks.mean()
                recent_avg = weekly_clicks.iloc[-3:].mean() if len(weekly_clicks) >= 3 else overall_avg
                feat["trend_recent_momentum"] = recent_avg / (overall_avg + 1e-5)
                feat["trend_n_declining_weeks"] = int((diffs < 0).sum())

                results.append(feat)

        return pd.DataFrame(results)

    def build_assessment_features(self, info: pd.DataFrame) -> pd.DataFrame:
        sa = self.tables["studentAssessment"].copy()
        assess = self.tables["assessments"].copy()
        sa = sa.merge(assess, on="id_assessment", how="left")

        results = []
        for (module, pres), group in info.groupby(["code_module", "code_presentation"]):
            cutoff_day = self._get_cutoff_day(module, pres)
            student_ids = group["id_student"].values

            module_assess = assess[
                (assess["code_module"] == module) &
                (assess["code_presentation"] == pres) &
                (assess["date"] <= cutoff_day)
            ]
            n_assessments_due = len(module_assess)
            assess_ids_due = set(module_assess["id_assessment"])

            for sid in student_ids:
                s_assess = sa[
                    (sa["id_student"] == sid) &
                    (sa["id_assessment"].isin(assess_ids_due))
                ]
                feat = {"code_module": module, "code_presentation": pres, "id_student": sid}

                if len(s_assess) == 0:
                    feat.update({
                        "assess_mean_score": 0, "assess_std_score": 0,
                        "assess_n_submitted": 0, "assess_submission_ratio": 0,
                        "assess_mean_days_early": 0, "assess_n_late": 0,
                        "assess_score_trend": 0
                    })
                else:
                    feat["assess_mean_score"] = s_assess["score"].mean()
                    feat["assess_std_score"] = s_assess["score"].std() if len(s_assess) > 1 else 0
                    feat["assess_n_submitted"] = len(s_assess)
                    feat["assess_submission_ratio"] = len(s_assess) / max(1, n_assessments_due)

                    s_merged = s_assess[["id_assessment", "date_submitted", "score"]].merge(
                        module_assess[["id_assessment", "date"]].rename(columns={"date": "deadline"}),
                        on="id_assessment", how="left"
                    )
                    s_merged["days_early"] = s_merged["deadline"] - s_merged["date_submitted"]
                    feat["assess_mean_days_early"] = s_merged["days_early"].mean()
                    feat["assess_n_late"] = int((s_merged["days_early"] < 0).sum())

                    if len(s_assess) >= 2:
                        scores = s_assess.sort_values("date_submitted")["score"].values
                        feat["assess_score_trend"] = np.polyfit(np.arange(len(scores)), scores, 1)[0]
                    else:
                        feat["assess_score_trend"] = 0

                results.append(feat)

        return pd.DataFrame(results)

    def build_all_features(self) -> pd.DataFrame:
        info = self.tables["studentInfo"]
        key_cols = ["code_module", "code_presentation", "id_student"]

        print("  Building demographic features...")
        demo = self.build_demographic_features(info)

        print("  Building click + trend features...")
        clicks = self.build_click_features(info)

        print("  Building assessment features...")
        assess = self.build_assessment_features(info)

        features = demo.merge(clicks, on=key_cols, how="outer")
        features = features.merge(assess, on=key_cols, how="outer")

        target_map = {"Distinction": 0, "Pass": 0, "Fail": 1, "Withdrawn": 1}
        target = info[key_cols + ["final_result"]].copy()
        target["target"] = target["final_result"].map(target_map)
        features = features.merge(target, on=key_cols, how="left")
        features.fillna(0, inplace=True)

        print(f"  Feature matrix: {features.shape}")
        print(f"  Target distribution:\n{features['target'].value_counts().to_string()}")
        return features
