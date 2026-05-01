# Executive Summary

We propose a **data-driven ML system** that predicts TNEA college allotments from a student’s cutoff (normalized aggregate score), category, and preferences.  The solution ingests five years of historical allotment CSVs (with dynamic/variable columns), supplements them with the latest 2025 ranklist and cutoff data, and trains probabilistic models to output admission **likelihoods**.  The pipeline will intelligently **normalize** and validate inputs (handling shifting headers via regex or ML-based entity recognition【3†L148-L156】), **engineer features** (Rank, Marks, Category, Year, Branch, etc.), and learn from historical closing ranks.  During model training we’ll use multi-class softmax or one-vs-rest classifiers (e.g. logistic regression, random forests, XGBoost【40†L1-L4】) calibrated with Platt/sigmoid or isotonic methods【43†L125-L134】.  The final product is a REST API (Flask/FastAPI) where a student can input their normalized cutoff and get **top-20 college-branch recommendations with probabilities**, or query a specific college+branch to receive a confidence score for admission. 

This report details the full architecture, data flow, and implementation plan.  We compare modeling strategies (softmax vs binary vs learning-to-rank), outline evaluation metrics (Top-K accuracy, ROC, calibration curves), specify tech stack (Python, pandas, scikit-learn/CatBoost, BeautifulSoup/Selenium, Flask), and present example API payloads.  Mermaid diagrams illustrate the system and data pipelines.  A phase-by-phase timeline estimates effort for ingestion, scraping, modeling, deployment, and validation.  Finally, we list key data sources (official TNEA PDFs, tneacutoff.com pages), references on scraping and ML pipelines【9†L158-L163】【31†L184-L193】, and discuss risks (data quality, biases, privacy) with mitigation strategies.  

## 1. Data Ingestion & Normalization

- **Historical Allotment CSVs:** Collect 5 years of TNEA allotment data (CSV/XLS) from official archives. These files may have **dynamic columns** (varying names/order). We will load them with pandas and apply a **schema-mapping layer**: use regex/fuzzy matching on header names to align columns (e.g. “Aggr Mark” → “Aggregate Score”), or even train a simple Named Entity Recognition (NER) model (e.g. a CRF tagger【3†L148-L156】) on known column samples to auto-identify fields.  
- **Data Cleaning & Validation:** After loading, standardize data types (dates, numeric scores) and canonicalize categories (OC, BC, MBC, etc.). Use Pandas to fill missing values sensibly (e.g. median or mode) and drop irrelevant columns (Name, DOB, etc.). Implement data quality checks: verify rank-sorting consistency, score bounds, duplicate entries, etc.  This is the **Data Validation** stage of the ML pipeline【31†L175-L184】, catching anomalies before modeling.  
- **Normalization:** Normalize text fields (e.g. college/branch codes as integers) and ensure consistent categorical encoding (one-hot or label encoding for community and category). If branches or colleges changed over years, map legacy codes to current ones. Store the unified dataset in a database or a master CSV for downstream use.  

*Embedding an image here demonstrates a generalized ML pipeline:*  
【32†embed_image】 *Figure: A scalable ML pipeline ingests raw data and systematically delivers predictive outputs【31†L118-L127】【31†L184-L193】.*  

The output of this phase is a **clean, unified dataset** of student records with columns like `(Year, Rank, NormalizedScore, Community, Category, AllottedCollegeCode, BranchCode)`.  This data will feed feature engineering and model training.  

## 2. Web Scraping Latest Data (2025)

- **2025 Ranklist PDF/XLS:** Download the official 2025 rank list from the TNEA website (static.tneaonline.org). These documents (in PDF/Excel) list all candidates’ normalized scores and ranks. We can parse the files using tools like `tabula-py` or `pandas.read_excel`. This gives us the current distribution of marks → rank.  
- **tneacutoff.com Cutoff Pages:** The site https://tneacutoff.com provides latest cutoff data by college. For each college code, its `/college/{code}` page (web app) displays branch-wise cutoffs by category. As this is a **dynamic, JavaScript-rendered site**, we can attempt to access any underlying API or resort to Selenium headless browsing to render and scrape. For static content, one would use requests+BeautifulSoup, but for dynamic pages “you’ll often need tools like Scrapy or Selenium to handle JavaScript”【9†L158-L163】.  
- **Scraping Steps:** Write a scraper to iterate over college codes (e.g. 1–3000). For each:
  1. Access the college page (possibly via Selenium) and extract cutoff tables (or use any JSON API endpoint, if discovered).
  2. Parse branch codes and category-wise cutoffs (score or rank).  
  3. Append to a **cutoff lookup table**: `(CollegeCode, BranchCode, Category, CutoffScore2024, CutoffRank2024)`.  
- **Ingestion:** Integrate this scraped data into our database. It provides up-to-date benchmarks for 2024/2025 cutoff thresholds. Use it for feature creation (e.g. “how much a given score exceeds the last year’s cutoff for that branch”).  

This ensures our model sees the *current admission landscape*. We cite the approach of using Selenium for dynamic scraping【9†L158-L163】.  

## 3. Feature Engineering & Historical Analysis

- **Core Features:** Derive features from `(Rank, Score, Community, Category, Year)`. For example, **NormalizedScore** (0–200 scale) and **Rank** (1–~200k) are raw inputs. Include **Category** (OC/BC/MBC/SC/ST) and **Community** if relevant.  
- **Historical Trends:** Compute, for each (College, Branch, Category) and Year, the closing rank/score (last admitted rank). From historical data, compute statistics like *mean/median closing score per year*, *yearly change*, *branch popularity*. Add features like *“Score difference to last year’s closing score”*.  
- **Temporal Features:** Encode the application year, and maybe create features like *score percentile* (100–RankPercentile). If any student chose preferences, we could incorporate choice order.  
- **Feature Store:** Build a feature store table with a row per student query, including all engineered fields. This might include one-hot encodings or embeddings for categorical fields, as well as raw numeric features. According to ML best practices, raw data must be “cleaned, transformed, and shaped into informative features”【31†L184-L193】 (e.g. filling missing values, encoding labels).  
- **Feature Scaling:** For some models (e.g. logistic regression), scale numeric features (StandardScaler). Tree-based models don’t require scaling. 

Effective feature engineering often makes or breaks model performance【31†L191-L197】. By leveraging historical cutoffs, category-reservations, and score distributions, the feature set will capture the admissions context needed for accurate predictions.

## 4. Modeling Approaches & Calibration

We consider three general modeling strategies:

| **Approach**               | **Description**                                                                                 | **Pros**                                                                 | **Cons**                                                                | **Use Case / Verdict**      |
|----------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------|-----------------------------|
| **(A) Multiclass Softmax** | Single model (e.g. logistic regression, neural net) with softmax over all college-branches.     | Direct probability distribution; one model to maintain; captures correlations across classes. | Can struggle with **very many classes**; training could be heavy. Needs careful handling of thousands of classes (branch-college combos). | Good baseline if num_classes is moderate (e.g. a few hundred). |
| **(B) One-vs-Rest Binary** | Train one binary classifier per college-branch (e.g. logistic per branch)【38†L1-L4】.           | Flexibility to choose different models per branch; robust to class imbalance. | Potentially thousands of models (one per branch); complexity and maintenance overhead【38†L1-L4】. Harder to scale if classes ≫100. | Good if focusing on top N popular branches; use ensemble of models. |
| **(C) Ranking (LTR)**      | Treat as learning-to-rank or relevance (e.g. pairwise ranking of branch scores).            | Directly optimizes ordering of colleges. Can leverage ranking losses (e.g. LambdaMART). | More complex pipeline; less straightforward probability output; fewer off-the-shelf tools for multi-task ranking. | Might try for final refinement but likely overkill; not main choice. |

Approach (A) uses one multiclass model with softmax output.  This gives normalized probabilities over all classes.  Approach (B) uses many binary classifiers (often logistic or tree-based) – one per branch – and takes the top predictions.  Approach (B) is essentially One-vs-Rest (OvR).  A noted downside of OvR is “it requires one model to be created for each class”【38†L1-L4】, which can be heavy if classes are numerous.  Approach (C) would use a ranking algorithm (e.g. XGBoost’s rank objective), but it is less common for this use-case since we want calibrated probabilities.  

In practice, we will likely prototype (A) with scikit-learn’s logistic regression or a small neural net (softmax) to get end-to-end probabilities.  For better performance, a Gradient Boosting Machine (like XGBoost/CatBoost) can handle multiclass (via one-vs-all or softmax). We will **calibrate** the model outputs to ensure probabilities are meaningful【43†L125-L134】. This involves using techniques like Platt-scaling (`sigmoid` in scikit-learn) or `isotonic` regression (via `CalibratedClassifierCV`) to post-process probabilities. 

**Algorithm Choices:** We will evaluate:
- **Logistic Regression (Softmax):** Fast, interpretable, natively outputs calibrated probabilities (logistic has “canonical link”【43†L125-L134】).
- **Random Forest / XGBoost:** Can capture non-linear rank-score patterns (e.g. different slopes for different branches) and often outperform linear models. Use `predict_proba` and then calibrate.  
- **CatBoost:** Handles categorical features (like branch, category) well, and robust default settings.  

**Model Validation:** Use cross-year validation. E.g., train on 2019–2023 data, hold out 2024 for validation, and test on 2025. Key metrics:
  - **Top-K Accuracy:** e.g. is the actual allotted college/branch in the model’s top-5 or top-20 predicted list? This measures usefulness of recommendations.  
  - **ROC/AUC:** We can compute one-vs-rest ROC AUC per class or micro-average to gauge discriminative power.  
  - **Calibration:** Plot reliability diagrams (calibration curves) to verify probability quality. We ensure the model’s predicted probabilities align with empirical admit rates (well-calibrated models satisfy that “samples with prob ~0.8 are truly admitted ~80%”【43†L133-L141】).  
  - **Rank-based metrics:** If we use any ranking model, NDCG or MAP could be relevant (though optional).  

This thorough evaluation (including Platt/isotonic calibration【43†L125-L134】) ensures robust probabilities. We will document the chosen final model after comparing these approaches on validation data.

## 5. API & Service Design

We will build a REST API to serve predictions. Key components:
- **Framework:** Use Flask or FastAPI (lightweight Python web frameworks) to expose endpoints. Both integrate easily with a pre-trained scikit-learn/CatBoost model (saved via `joblib` or `pickle`). GeeksforGeeks notes that deploying an ML model with Flask “enables integration of trained models into web apps for real-time predictions”【24†L30-L34】.  
- **Endpoints:** 
  1. **`/predict_top20`** (POST): Input JSON includes `{ "score": 187.5, "category": "OC", "community": "General" }`. The API returns the **top 20 college-branch matches**, sorted by predicted probability, e.g. 
     ```json
     {
       "status": "success",
       "predictions": [
         {"college_code": 1399, "branch_code": "CS", "probability": 0.72},
         {"college_code": 1023, "branch_code": "EC", "probability": 0.65},
         … (up to 20) …
       ]
     }
     ```  
  2. **`/predict_chance`** (POST): Input JSON includes `{ "score": 187.5, "category": "SC", "college_code": 1399, "branch_code": "CS" }`. Returns: 
     ```json
     {
       "status": "success",
       "college_code": 1399,
       "branch_code": "CS",
       "category": "SC",
       "probability": 0.45
     }
     ```  
  These JSON examples illustrate the schema. The API will preprocess inputs (normalization, encoding), then use the model’s `predict_proba` to compute probabilities for all classes. For `/predict_top20`, we select the 20 highest probabilities. For `/predict_chance`, we index the specific (college,branch) class probability.  
- **Tech Stack:** Python 3.9+, with libraries: `pandas` (data loading), `numpy`, `scikit-learn`/`xgboost`/`catboost`, `requests`/`selenium` for scraping, `Flask` or `FastAPI` for serving. Use `gunicorn` or `uvicorn` for production. Docker can containerize the service.   
- **API Security & Validation:** Validate JSON schema, handle missing fields. Rate-limit as needed. 

**Project Structure:** Follow best practices【24†L69-L77】: separate `app.py` (Flask server) from the model (`model.pkl`), maintain a `requirements.txt`, and organize code into modules (data_ingest, model, api).  

## 6. System Architecture (Mermaid)

```mermaid
graph LR
  subgraph DataIngest
    CSVs[Historical CSV/Excel Files] --> Ingest["Data Ingestion & Cleaning"]
    RankList2025[2025 Ranklist (PDF/XLS)] --> Ingest
    CutoffScrape[tneacutoff.com Pages] --> Scrape["Web Scraping (Selenium)"] --> Ingest
  end
  Ingest --> Norm["Validation/Normalization"]
  Norm --> Features["Feature Engineering & Store"]
  Features --> Train["Model Training & Calibration"]
  Train --> Model["Deployed ML Model"]
  Model --> API["REST API (Flask/FastAPI)"]
  API --> Client[User/UI]
```

This diagram shows data sources feeding into a cleaning/normalization step, then feature engineering leading to model training. The trained model is served by the API to a client application.  

## 7. Data Flow (Mermaid)

```mermaid
flowchart LR
  A[Raw Allotment CSVs] --> B[Load & Clean (pandas)]
  C[2025 TNEA Ranklist] --> B
  D[tneacutoff.com Data] --> E[Scrape & Parse]
  E --> B
  B --> F[Normalized Dataset]
  F --> G[Feature Engineering]
  G --> H[Model Training (XGBoost, LR, etc.)]
  H --> I[Model Calibration (Platt/Isotonic)]
  I --> J[Final Predictive Model]
  J --> K[API Endpoint]
  K --> L[Client/UI]
```

This flowchart details how raw inputs (CSVs, rank list, cutoff scrape) are cleaned and integrated, then transformed into features. The training/calibration step yields the final model, which powers the API.  

## 8. Evaluation and Calibration Strategy

- **Cross-Year Validation:** Train on Years 2019–2023, validate on 2024, test on 2025. This simulates real future predictions.  
- **Metrics:**
  - *Top-K Accuracy:* Percentage of students whose true allotment appears in the model’s top-K suggestions. (We will report K=5, 10, 20.)  
  - *ROC AUC:* Compute multiclass AUC (e.g. macro/micro) or one-vs-rest AUC for each category.  
  - *Log-Loss/Brier Score:* For calibrated probabilities.  
  - *Calibration Curves:* Plot predicted probability vs actual admit frequency; check that a 0.8 score means ~80% admission. Scikit-learn’s calibration tools help here【43†L125-L134】.  
- **Calibration:** If probabilities are skewed, use scikit-learn’s `CalibratedClassifierCV` with `method="sigmoid"` (Platt) or `"isotonic"` to adjust. According to scikit-learn docs, Platt’s logistic model (`"sigmoid"`) or nonparametric isotonic regression can significantly improve predicted probabilities【43†L125-L134】【44†L1-L4】.  
- **Feature Importance:** Analyze feature weights/importance (e.g. using SHAP or model’s builtin importance) to ensure model makes sense (higher score → higher chance, etc.).  

## 9. Implementation Plan & Timeline

1. **Data Collection & Schema Mapping (5–7 days):** Gather 2019–2023 CSVs and 2025 ranklist. Write parsers and header-mapping code. Manual mapping of 2–3 initial files to establish patterns, then automate.  
2. **Web Scraping Setup (3–5 days):** Prototype scraping of one college page using Selenium or Reverse-Engineering (check if tneacutoff.com has a public API). Automate loop over all college codes, store results.  
3. **Data Cleaning & Integration (3 days):** Merge scraped data with historical CSVs. Clean anomalies, drop/unify fields.  
4. **Feature Engineering (5–7 days):** Develop code to compute historical cutoff stats, encode categories, etc. Build a feature store (could be a Pandas DataFrame or database). Validate feature distributions.  
5. **Model Development (5–7 days):** Train baseline multiclass logistic/regression. Evaluate. Train more complex models (RF, XGBoost, CatBoost). Compare via validation metrics.  
6. **Calibration & Evaluation (3 days):** Apply Platt/Isotonic calibration using held-out 2024. Generate ROC and calibration plots. Choose final model.  
7. **API Development (4–6 days):** Build Flask/FastAPI service with endpoints for Top-20 and single college queries. Connect to model. Write unit tests.  
8. **End-to-End Testing (3 days):** Test on real examples, ensure performance. Optimize for latency (<1s per query).  
9. **Deployment & Documentation (4 days):** Containerize (Docker) or deploy to cloud. Write README, API docs, and run risk/privacy checks.  

*Total estimated effort: ~30–40 work-days.*  This assumes a small team or developer; times can parallelize tasks.  

## 10. Example API Payloads

**Top-20 Prediction Request:**  
```json
POST /predict_top20
{
  "score": 187.4,
  "category": "OC",
  "community": "General",
  "gender": "Male"
}
```  
**Response (sample):**  
```json
{
  "status": "success",
  "predictions": [
    {"college_code": 1399, "branch_code": "CS", "probability": 0.729},
    {"college_code": 1023, "branch_code": "EC", "probability": 0.653},
    {"college_code": 1051, "branch_code": "ME", "probability": 0.542},
    // ... up to 20 entries
  ]
}
```

**Single College Confidence Request:**  
```json
POST /predict_chance
{
  "score": 187.4,
  "category": "OC",
  "college_code": 1023,
  "branch_code": "EC"
}
```  
**Response (sample):**  
```json
{
  "status": "success",
  "college_code": 1023,
  "branch_code": "EC",
  "category": "OC",
  "probability": 0.653
}
```

These payloads demonstrate the inputs and outputs in JSON for both endpoints.

## 11. Risk, Data Quality & Privacy

- **Dynamic Schema Risks:** Varied column names may lead to mis-parsing. *Mitigation:* Start with a robust mapping of known synonyms; log any unmapped columns for manual review. Use simple regex rules (e.g. `Rank|Position`) as a fallback.  
- **Incomplete or Noisy Data:** Past allotment records may have missing ranks or duplicates. *Mitigation:* Implement stringent cleaning (drop or impute missing ranks) and cross-check totals. Use audits (e.g. total students per year).  
- **Scraping Reliability:** tneacutoff.com is third-party and may change layout or block scraping. *Mitigation:* Check site’s terms of use, use exponential backoff. If an official API exists, prefer that. Cache results.  
- **Model Bias:** Reservation quotas (OC vs BC etc.) must be respected. The model may inadvertently discriminate if data is unbalanced. *Mitigation:* Ensure training dataset reflects the reservation categories. Possibly train separate models per category or include category as a key input.  
- **Data Privacy:** Source CSVs contain personal info (names, DOB). We must drop PII (only use anonymized fields like category and marks). Store only non-sensitive data. Use the data for aggregate modeling only.  
- **Overfitting to Historical Trends:** Admission patterns can change (new colleges, changed quotas). *Mitigation:* Keep models updated yearly. Use the latest cutoff scrape (2024) to adjust predictions for 2025 admissions.  
- **Calibration Errors:** Probability estimates might mislead if not calibrated. *Mitigation:* Use calibration plots and adjust with Platt/isotonic methods【43†L125-L134】. Provide confidence intervals or disclaimers if needed.  

## 12. Key Sources to Consult

- **Official TNEA Documents:** Rank lists and allotment PDFs (static.tneaonline.org) for raw data and format specifications.  
- **TNEA Cutoff Data:** **tneacutoff.com** (branchwise cutoff trends) for 2020–2025.  
- **Web Scraping Guides:** BeautifulSoup/Selenium documentation and tutorials. RealPython’s guide on scraping notes static vs dynamic site handling【9†L158-L163】.  
- **ML Pipeline References:** Articles on building ML pipelines (DSG.AI blog【31†L184-L193】), especially data validation and feature engineering steps.  
- **Modeling Techniques:** Scikit-learn docs on multiclass classification and calibration【43†L125-L134】, and literature on OvR vs softmax【38†L1-L4】.  
- **API Deployment Tutorials:** Guides like GeeksforGeeks on deploying ML models with Flask【24†L30-L34】, and FastAPI docs for production.  
- **Calibration Methods:** Sklearn’s **Probability Calibration** guide【43†L125-L134】 for details on Platt/sigmoid and isotonic regression.  

These references will ensure our approach is grounded in best practices and up-to-date techniques.

