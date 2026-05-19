# GradeCast — Weekly Student Risk Prediction Pipeline

Part of the **Agentic Teacher Dashboard** project.  
Built by Chukwuka Christian (ML Engineer) · AIRC-Edu-Agentic-AI

---

## Overview

GradeCast converts static cutoff-based student risk prediction into a **weekly time-series pipeline** using the OULAD dataset.

Instead of a single risk label per student, the model produces a risk score for **every week** of the course — giving teachers an evolving view of each student's trajectory so they can intervene early.

---

## Results

| Metric | Value |
|---|---|
| Model | XGBoost Classifier |
| Test AUC | **0.9362** |
| Total students | 32,593 |
| Course presentations | 22 |
| Total weekly predictions | 1,179,984 |

### Sample output

```json
{
  "id_student": 3733,
  "code_module": "DDD",
  "code_presentation": "2013J",
  "final_result": "Withdrawn",
  "weekly_risk": [
    { "week": 1, "risk_score": 0.92 },
    { "week": 2, "risk_score": 0.94 },
    { "week": 3, "risk_score": 0.95 },
    { "week": 8, "risk_score": 0.99 }
  ]
}
```

A risk score near **1.0** = high risk of failure or withdrawal.  
A risk score near **0.0** = student is likely to pass.

---

## Features Used

| Category | Features |
|---|---|
| Demographic | Gender, region, education level, disability, prior attempts, credits |
| Clickstream | Total clicks, daily mean, engagement ratio, activity type breakdown, entropy |
| Trend | Weekly slope, window ratio, longest decline streak, inactivity streak, recent momentum |
| Assessment | Mean score, submission ratio, days early/late, score trend over time |

---

## Architecture