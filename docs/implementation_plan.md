Alright — let’s lock this in like a **production-grade blueprint**. Think of this as something you could literally hand to a team and say “ship this.”

No fluff, just execution clarity.

---

# 🚀 PROJECT PLAN: TNEA Admission Prediction System

---

# 🎯 1. Objective

Build a **Hybrid Admission Intelligence System** that:

### ✅ Predicts Top 20 College Options

Given:

* Cutoff
* Community

Returns:

* Top 20 (College + Branch) ranked by probability

---

### ✅ Computes Confidence Score

Given:

* Cutoff
* College Code
* Branch
* Community

Returns:

* Admission probability (0–100%)

---

# 🧠 2. System Philosophy

This is NOT pure ML.

It’s a:

> **Hybrid Decision System = ML + Rule Engine + Trend Intelligence**

---

# 🏗️ 3. System Architecture

## 🔷 High-Level Architecture

```
                ┌──────────────────────────┐
                │   Data Sources           │
                │                          │
                │  • 2020–2024 Allotments  │
                │  • 2025 Rank List        │
                │  • External Cutoffs      │
                └────────────┬─────────────┘
                             ↓
                ┌──────────────────────────┐
                │   Data Processing Layer  │
                │                          │
                │  • Schema Detection      │
                │  • Cleaning              │
                │  • Normalization         │
                └────────────┬─────────────┘
                             ↓
                ┌──────────────────────────┐
                │ Feature Engineering      │
                │                          │
                │  • Percentiles           │
                │  • Trends                │
                │  • Demand Scores         │
                └────────────┬─────────────┘
                             ↓
        ┌────────────────────┴────────────────────┐
        │                                         │
┌───────────────┐                      ┌────────────────────┐
│ ML Model      │                      │ Rule Engine (2025) │
│ (2020–2024)   │                      │                    │
│               │                      │ Cutoff Ranges      │
└──────┬────────┘                      └─────────┬──────────┘
       ↓                                         ↓
        └──────────────┬─────────────────────────┘
                       ↓
           ┌──────────────────────────┐
           │ Final Scoring Engine     │
           │ (Weighted Fusion)        │
           └────────────┬─────────────┘
                        ↓
           ┌──────────────────────────┐
           │ API Layer (FastAPI)      │
           └────────────┬─────────────┘
                        ↓
           ┌──────────────────────────┐
           │ UI (Streamlit/Web App)   │
           └──────────────────────────┘
```

---

# ⚙️ 4. Implementation Plan (Phase-wise)

---

## 🥇 Phase 1: Data Engineering

### 🎯 Goal:

Create a **clean, unified dataset**

---

### Step 1: Data Ingestion

Sources:

* CSV / PDF (historical)
* Rank list (2025)
* Web scraping (cutoff site)

---

### Step 2: Dynamic Schema Mapping

Problem:
Columns differ across years

### Solution:

```python
column_alias = {
    "cutoff": ["cutoff", "aggr mark"],
    "rank": ["rank"],
    "college": ["college code"],
    "branch": ["branch code"],
    "community": ["community"]
}
```

👉 Build auto-mapper

---

### Step 3: Data Cleaning

* Remove nulls
* Standardize:

  * Community labels
  * Branch codes
* Convert types

---

## 🥈 Phase 2: Feature Engineering

---

### Core Features:

* Cutoff
* Rank
* Community
* College
* Branch
* Year

---

### Derived Features:

#### 1. Percentiles

```python
cutoff_percentile = rank_based_percentile
```

---

#### 2. Demand Score

```python
branch_demand = frequency(branch)
college_popularity = frequency(college)
```

---

#### 3. Trend Feature

```python
trend = slope(cutoff over years)
```

---

## 🥉 Phase 3: Model Building

---

### Problem Type:

Binary Classification

---

### Training Data:

| Feature                            | Target |
| ---------------------------------- | ------ |
| Cutoff, College, Branch, Community | 1/0    |

---

### Negative Sampling:

For each student:

* Add 3–5 random colleges → label = 0

---

### Models:

* Logistic Regression (baseline)
* Random Forest
* XGBoost (final)

---

### Output:

```python
probability = model.predict_proba(X)
```

---

## 🟡 Phase 4: 2025 Intelligence Layer

---

### Input Sources:

* Rank List
* External Cutoff Website

---

### Extract:

* Opening cutoff
* Closing cutoff

---

### Rule Engine Logic:

```python
if cutoff >= opening:
    score = 0.9
elif cutoff >= closing:
    score = 0.6
else:
    score = 0.2
```

---

## 🔵 Phase 5: Final Scoring Engine

---

### Combine ML + Rule

```python
final_score = (0.7 * ml_score) + (0.3 * rule_score)
```

---

### Optional:

Add trend adjustment

---

# 🌐 5. API Design

---

## Endpoint 1: Top 20 Options

```
POST /predict/top-options
```

### Input:

```json
{
  "cutoff": 185,
  "community": "BC"
}
```

---

### Output:

```json
[
  {
    "college": 1315,
    "branch": "CS",
    "probability": 0.91
  }
]
```

---

## Endpoint 2: Confidence Score

```
POST /predict/confidence
```

### Input:

```json
{
  "cutoff": 185,
  "community": "BC",
  "college": 1315,
  "branch": "CS"
}
```

---

### Output:

```json
{
  "probability": 0.82,
  "ml_score": 0.78,
  "rule_score": 0.04
}
```

---

# 🔄 6. System Flow (End-to-End)

---

## 🧩 Flow 1: Top 20 Prediction

```
User Input → Cutoff + Community
        ↓
Generate all college-branch combos
        ↓
ML Model → probability
        ↓
Rule Engine (2025 adjustment)
        ↓
Final Score Calculation
        ↓
Sort Top 20
        ↓
Return Output
```

---

## 🧩 Flow 2: Confidence Score

```
User Input → Specific College + Branch
        ↓
ML Prediction
        ↓
Rule Adjustment
        ↓
Final Score
        ↓
Return Probability
```

---

# 🎯 7. Sample Scenarios

---

## 🔥 Scenario 1: Top 20 Prediction

### Input:

```
Cutoff = 190
Community = BC
```

---

### Output:

```
1. College 1315 - CS → 94%
2. College 2006 - IT → 91%
3. College 1399 - ECE → 89%
...
```

---

## 🔥 Scenario 2: Confidence Score

### Input:

```
Cutoff = 185
College = 1315
Branch = CS
Community = BC
```

---

### Output:

```
Probability = 0.87

Breakdown:
ML Score = 0.82
2025 Adjustment = +0.05
```

---

## 🔥 Scenario 3: Edge Case

### Input:

```
Cutoff = 150
Top Tier College
```

---

### Output:

```
Probability = 0.12

Insight:
Cutoff below historical + 2025 closing cutoff
```

---

# ⚠️ 8. Risks & Mitigation

---

### ❌ Missing 2025 Data

✔ Use rule engine

---

### ❌ Data Noise

✔ Normalize + validate

---

### ❌ Trend Shifts

✔ Recency weighting

---

# 🚀 9. Deployment Plan

---

### Backend:

* FastAPI

---

### ML:

* Scikit-learn / XGBoost

---

### UI:

* Streamlit

---
