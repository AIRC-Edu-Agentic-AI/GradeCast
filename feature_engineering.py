# ── feature_engineering.py ─────────────────────────────────────────────────
# Builds ONE feature vector per (student, week).
# This replaces the old cutoff-based approach.
# The key change: we loop over weeks, not over cutoff percentages.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import WEEK_DAYS


class WeeklyFeatureEngineer:
    """
    For each student, produce one feature row per week.

    How it works:
    - Week N means: "using only data from day 0 up to day N*7"
    - We compute the same features as before, but at every weekly snapshot
    - The model then trains on all (student, week) pairs
    - At prediction time, we get a risk score for each week
    """

    def __init__(self, tables: dict):
        self.tables = tables
        # Build a lookup: (module, presentation) → total course length in days
        self.course_lengths = (
            tables["courses"]
            .set_index(["code_module", "code_presentation"])["module_presentation_length"]
            .to_dict()
        )

    def _n_weeks(self, module: str, presentation: str) -> int:
        """Total number of weeks in a course."""
        length = self.course_lengths.get((module, presentation), 0)
        return max(1, length // WEEK_DAYS)

    # ── Demographic features (same for every week — static) ─────────────────

    def _build_demo_features(self, info: pd.DataFrame) -> pd.DataFrame:
        df = info.copy()
        for col in ["gender", "region", "highest_education",
                    "imd_band", "age_band", "disability"]:
            le = LabelEncoder()
            df[f"demo_{col}"] = le.fit_transform(df[col].astype(str))
        df["demo_prev_attempts"] = df["num_of_prev_attempts"]
        df["demo_credits"] = df["studied_credits"]
        demo_cols = [c for c in df.columns if c.startswith("demo_")]
        keys = ["code_module", "code_presentation", "id_student"]
        return df[keys + demo_cols].copy()

    # ── Click features up to a given cutoff day ──────────────────────────────

    def _click_features_at_day(
        self, svle_cohort: pd.DataFrame, sid: int, cutoff_day: int, n_weeks_total: int
    ) -> dict:
        """
        Compute click-based features for student `sid` using data up to `cutoff_day`.
        """
        s = svle_cohort[
            (svle_cohort["id_student"] == sid) &
            (svle_cohort["date"] <= cutoff_day)
        ].copy()

        feat = {}

        total_clicks = s["sum_click"].sum()
        n_active_days = s["date"].nunique()
        elapsed = max(1, cutoff_day)

        feat["click_total"] = total_clicks
        feat["click_mean_per_day"] = total_clicks / elapsed
        feat["click_std_per_day"] = (
            s.groupby("date")["sum_click"].sum().std()
            if n_active_days > 1 else 0.0
        )
        feat["click_active_days"] = n_active_days
        feat["click_engagement_ratio"] = n_active_days / elapsed

        # Activity type breakdown
        for act in ["forumng", "oucontent", "resource", "quiz", "homepage", "subpage"]:
            feat[f"click_{act}"] = (
                s[s["activity_type"] == act]["sum_click"].sum()
            )

        # Entropy across activity types
        act_counts = s.groupby("activity_type")["sum_click"].sum()
        if len(act_counts) > 0:
            probs = act_counts / act_counts.sum()
            feat["click_entropy"] = -(probs * np.log2(probs + 1e-10)).sum()
        else:
            feat["click_entropy"] = 0.0

        # Weekly click series up to this cutoff
        n_weeks_elapsed = max(1, cutoff_day // WEEK_DAYS)
        s["week_num"] = (s["date"] // WEEK_DAYS).clip(lower=0)
        weekly = (
            s.groupby("week_num")["sum_click"].sum()
            .reindex(range(n_weeks_elapsed + 1), fill_value=0)
        )

        # Trend slope
        if len(weekly) >= 2 and np.std(weekly.values) > 0:
            feat["trend_slope"] = np.polyfit(np.arange(len(weekly)), weekly.values, 1)[0]
        else:
            feat["trend_slope"] = 0.0

        # Second-half vs first-half ratio
        mid = max(1, len(weekly) // 2)
        feat["trend_window_ratio"] = (
            weekly.iloc[mid:].mean() / (weekly.iloc[:mid].mean() + 1e-5)
        )

        # Longest consecutive declining weeks
        diffs = weekly.diff().dropna()
        longest_decline = current_streak = 0
        for d in diffs:
            if d < 0:
                current_streak += 1
                longest_decline = max(longest_decline, current_streak)
            else:
                current_streak = 0
        feat["trend_longest_decline"] = longest_decline

        # Longest inactive streak (zero-click weeks)
        longest_inactive = current_inactive = 0
        for c in weekly:
            if c == 0:
                current_inactive += 1
                longest_inactive = max(longest_inactive, current_inactive)
            else:
                current_inactive = 0
        feat["trend_longest_inactivity"] = longest_inactive

        # Recent momentum: last 3 weeks vs overall average
        overall_avg = weekly.mean()
        recent_avg = weekly.iloc[-3:].mean() if len(weekly) >= 3 else overall_avg
        feat["trend_recent_momentum"] = recent_avg / (overall_avg + 1e-5)
        feat["trend_n_declining_weeks"] = int((diffs < 0).sum())

        return feat

    # ── Assessment features up to a given cutoff day ─────────────────────────

    def _assess_features_at_day(
        self,
        sa_cohort: pd.DataFrame,
        module_assess: pd.DataFrame,
        sid: int,
        cutoff_day: int
    ) -> dict:
        """
        Assessment features for student `sid` using submissions up to `cutoff_day`.
        """
        due = module_assess[module_assess["date"] <= cutoff_day]
        n_due = len(due)
        due_ids = set(due["id_assessment"])

        s = sa_cohort[
            (sa_cohort["id_student"] == sid) &
            (sa_cohort["id_assessment"].isin(due_ids))
        ]

        feat = {}

        if len(s) == 0 or n_due == 0:
            feat.update({
                "assess_mean_score": 0.0,
                "assess_std_score": 0.0,
                "assess_n_submitted": 0,
                "assess_submission_ratio": 0.0,
                "assess_mean_days_early": 0.0,
                "assess_n_late": 0,
                "assess_score_trend": 0.0,
            })
        else:
            feat["assess_mean_score"] = s["score"].mean()
            feat["assess_std_score"] = s["score"].std() if len(s) > 1 else 0.0
            feat["assess_n_submitted"] = len(s)
            feat["assess_submission_ratio"] = len(s) / n_due

            merged = s[["id_assessment", "date_submitted", "score"]].merge(
                due[["id_assessment", "date"]].rename(columns={"date": "deadline"}),
                on="id_assessment", how="left"
            )
            merged["days_early"] = merged["deadline"] - merged["date_submitted"]
            feat["assess_mean_days_early"] = merged["days_early"].mean()
            feat["assess_n_late"] = int((merged["days_early"] < 0).sum())

            if len(s) >= 2:
                scores = s.sort_values("date_submitted")["score"].values
                feat["assess_score_trend"] = np.polyfit(
                    np.arange(len(scores)), scores, 1
                )[0]
            else:
                feat["assess_score_trend"] = 0.0

        return feat

    # ── Main builder ─────────────────────────────────────────────────────────

    def build_weekly_features(self) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per (student, week).
        Columns: id_student, week, <features...>, target
        """
        info = self.tables["studentInfo"]

        # Merge VLE interactions with activity type metadata once
        svle = self.tables["studentVle"].merge(
            self.tables["vle"][["id_site", "code_module", "code_presentation", "activity_type"]],
            on=["id_site", "code_module", "code_presentation"],
            how="left"
        )

        sa = self.tables["studentAssessment"].merge(
            self.tables["assessments"],
            on="id_assessment",
            how="left"
        )

        # Static demographic features (same value for all weeks)
        demo_df = self._build_demo_features(info)

        # Target label
        target_map = {"Distinction": 0, "Pass": 0, "Fail": 1, "Withdrawn": 1}
        info = info.copy()
        info["target"] = info["final_result"].map(target_map)

        all_rows = []

        for (module, pres), cohort in info.groupby(["code_module", "code_presentation"]):
            n_weeks_total = self._n_weeks(module, pres)
            student_ids = cohort["id_student"].values

            # Filter tables to this cohort once (fast)
            svle_c = svle[
                (svle["code_module"] == module) &
                (svle["code_presentation"] == pres) &
                (svle["id_student"].isin(student_ids))
            ]
            sa_c = sa[
                (sa["code_module"] == module) &
                (sa["code_presentation"] == pres) &
                (sa["id_student"].isin(student_ids))
            ]
            module_assess = self.tables["assessments"][
                (self.tables["assessments"]["code_module"] == module) &
                (self.tables["assessments"]["code_presentation"] == pres)
            ]

            # Demo features for this cohort
            demo_c = demo_df[
                (demo_df["code_module"] == module) &
                (demo_df["code_presentation"] == pres)
            ].set_index("id_student")

            # Target for this cohort
            target_c = cohort.set_index("id_student")["target"]

            print(f"  {module} {pres}: {len(student_ids)} students × {n_weeks_total} weeks")

            # ── The core loop: for each week, for each student ────────────
            for week in range(1, n_weeks_total + 1):
                cutoff_day = week * WEEK_DAYS

                for sid in student_ids:
                    row = {
                        "id_student": sid,
                        "code_module": module,
                        "code_presentation": pres,
                        "week": week,
                    }

                    # Add demographic features
                    if sid in demo_c.index:
                        demo_cols = {
                            k: v for k, v in demo_c.loc[sid].items()
                            if k not in ["code_module", "code_presentation"]
                        }
                        row.update(demo_cols)

                    # Add click + trend features
                    row.update(
                        self._click_features_at_day(svle_c, sid, cutoff_day, n_weeks_total)
                    )

                    # Add assessment features
                    row.update(
                        self._assess_features_at_day(sa_c, module_assess, sid, cutoff_day)
                    )

                    # Add target label
                    row["target"] = target_c.get(sid, np.nan)

                    all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df.fillna(0, inplace=True)
        print(f"\nFinal dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        return df