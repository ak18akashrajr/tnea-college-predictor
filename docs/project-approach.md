Got it — let’s clean this up and make it actually implementation-ready, no clutter, no broken flow. Think of this like your system design doc + execution roadmap combined.

---

# 🎯 Problem Framing (What you’re building)

You’re essentially building a **TNEA Admission Prediction Engine** with two core capabilities:

### 1. 🎯 Top 20 College Predictions

> Input: Cut-off + Community
> Output: Top 20 **(College Code + Branch)** combinations with highest admission probability

---

### 2. 📊 Confidence Score Engine

> Input: Cut-off + College Code + Branch + Community
> Output: Probability (0–100%) of getting that seat

---

# 🧠 High-Level Strategy

Let’s not overcomplicate this with heavy ML initially. This is a **structured probabilistic ranking problem**, not a deep learning problem.

You’ll win by:

* Cleaning historical data properly
* Building a strong feature pipeline
* Using interpretable models

---

# 🏗️ System Architecture (Clean Flow)

```
Raw Data (CSV / PDFs / Website)
        ↓
Data Ingestion Layer
        ↓
Schema Detection (Dynamic Columns)
        ↓
Data Cleaning + Normalization
        ↓
Feature Engineering
        ↓
Model Layer (Prediction + Probability)
        ↓
API Layer (FastAPI)
        ↓
Frontend / Streamlit
```

---

# ⚙️ Step-by-Step Execution Plan

## 🥇 Step 1: Data Ingestion

You have:

* 5 years historical allotment data ✅
* 2025 Rank List ✅
* External site: tneacutoff.com ✅

### Strategy:

* Convert all CSV/PDF → unified dataframe
* Scrape:

  ```
  https://tneacutoff.com/college/{college_code}
  ```

Use:

* `pandas`
* `BeautifulSoup`

---

## 🥈 Step 2: Dynamic Column Handling (Important)

Your columns are not fixed → so build a **column mapper layer**

### Approach:

Use fuzzy matching / rules:

```python
column_map = {
    "cutoff": ["cutoff", "aggr mark", "aggregate"],
    "rank": ["rank"],
    "college_code": ["college code"],
    "branch": ["branch code", "dept"],
    "community": ["community", "category"]
}
```

Then auto-map columns dynamically.

👉 This makes your system scalable across years.

---

## 🥉 Step 3: Feature Engineering

This is where the real edge is.

### Core Features:

* Cut-off
* Rank
* Community (BC, OC, MBC, SC, etc.)
* College Code
* Branch Code
* Year

### Derived Features:

* Cut-off percentile (VERY IMPORTANT)
* Rank percentile
* College popularity (frequency of selection)
* Branch demand score

---

## 🧩 Step 4: Problem Transformation

### Convert into Binary Classification

Each row becomes:

```
Input:
(Cutoff, Community, College, Branch)

Output:
1 → Got seat
0 → Didn’t get seat
```

But you only have “allotted” data → so generate negative samples.

### Negative Sampling:

For each student:

* Add 3–5 random colleges they DIDN’T get → label = 0

---

## 🤖 Step 5: Model Selection

Start simple. Don’t jump to deep learning.

### Best Options:

* Logistic Regression ✅ (interpretable)
* Random Forest ✅ (handles non-linearity)
* XGBoost ✅ (best performance)

---

## 🎯 Step 6: Use Case Implementation

---

## 🔥 USE CASE 1: Top 20 Predictions

### Input:

```
Cutoff = 185
Community = BC
```

### Process:

* Generate all possible (college, branch) combinations
* Pass into model
* Get probability for each
* Sort descending
* Return Top 20

### Output:

```
1. College 1315 - CS → 92%
2. College 2006 - IT → 89%
...
```

---

## 🔥 USE CASE 2: Confidence Score

### Input:

```
Cutoff = 185
College = 1315
Branch = CS
Community = BC
```

### Output:

```
Probability = 0.87 → 87% chance
```

👉 This comes directly from model `predict_proba()`

---

# 📊 Model Output Logic

```python
prob = model.predict_proba(X)[0][1]
```

That’s your confidence score.

---

# 🌐 Step 7: API Design (FastAPI)

## Endpoint 1: Top Predictions

```
POST /predict/top-options
```

Input:

```json
{
  "cutoff": 185,
  "community": "BC"
}
```

---

## Endpoint 2: Confidence Score

```
POST /predict/confidence
```

Input:

```json
{
  "cutoff": 185,
  "community": "BC",
  "college_code": 1315,
  "branch": "CS"
}
```

---

# 🧠 Smart Enhancement (Your Differentiator)

## 📈 Trend Awareness (Advanced Layer)

Use last 5 years:

* If cutoff trend ↑ → increase probability
* If cutoff trend ↓ → decrease probability

---

## 🌍 External Data Integration

From:

```
tneacutoff.com
```

Extract:

* Opening cutoff
* Closing cutoff

👉 Use this to validate predictions

---

# ⚠️ Key Challenges (Don’t Ignore)

### 1. Data Imbalance

* Too many positive samples
* Solve using negative sampling

---

### 2. Changing Trends

* 2025 may differ → add weight to recent years

---

### 3. Community Impact

* Very critical → must encode properly

---

# 🧪 Evaluation Metrics

* Accuracy ❌ (not useful)
* ROC-AUC ✅
* Precision@Top20 ✅ (VERY IMPORTANT)

---

# 🚀 Suggested Tech Stack

* Backend → FastAPI
* ML → Scikit-learn / XGBoost
* Data → Pandas
* Scraping → BeautifulSoup
* Frontend → Streamlit

---

# 🧩 Future Scope (If You Want to Scale)

* Rank prediction instead of cutoff
* Preference optimization (like real counselling)
* Reinforcement learning for seat allocation simulation

---

# 🧠 Final Thought

This is not just a project — this is literally a **product-level idea**.

If you execute cleanly:

* You can deploy this as a web app
* Students will actually use it during counselling

---
