from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pickle, pandas as pd, numpy as np, requests
from bs4 import BeautifulSoup

app = FastAPI()

# ── Allow Lovable frontend to call this API ───────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this after testing
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artifacts at startup (once) ─────────────────
print("Loading model artifacts...")

with open("artifacts/tnea_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("artifacts/encoders.pkl", "rb") as f:
    enc = pickle.load(f)
    le_community = enc["community"]
    le_college   = enc["college"]
    le_branch    = enc["branch"]

lookup  = pd.read_csv("artifacts/lookup_smart.csv")
df_hist = pd.read_csv("artifacts/df_clean.csv")

FEATURES = [
    "cutoff", "community_enc", "college_enc", "branch_enc",
    "hist_opening", "hist_closing", "hist_mean",
    "hist_count", "cutoff_trend", "rank"
]

print("✅ Artifacts loaded!")

# ── Request / Response Models ─────────────────────────
class DemandBoost(BaseModel):
    branch: str
    reason: str
    boost_marks: float

class PredictRequest(BaseModel):
    cutoff: float
    community: str
    college: str
    branch: str
    demand_boost_branches: Optional[List[DemandBoost]] = []
    use_manual_boost_only: Optional[bool] = False

# ── Scrape 2025 cutoff ────────────────────────────────
def scrape_2025_cutoff(college_code, branch_code, community):
    try:
        url  = f"https://tneacutoff.com/college/{college_code}"
        resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        if resp.status_code != 200: return None
        soup   = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2: continue
            headers_row = [th.get_text(strip=True) for th in rows[0].find_all(["th","td"])]
            if "Branch Name" not in headers_row and "Code" not in headers_row: continue
            communities = ["OC","BC","BCM","MBC","SC","SCA","ST"]
            for row in rows[1:]:
                cols = [td.get_text(strip=True) for td in row.find_all(["td","th"])]
                if len(cols) < 4: continue
                if cols[1].upper() != branch_code.upper(): continue
                for i, comm in enumerate(communities):
                    if comm == community and i + 2 < len(cols):
                        val = cols[i+2].strip()
                        if val and val != "-":
                            try: return float(val)
                            except: pass
        return None
    except: return None

# ── Main Prediction Endpoint ──────────────────────────
@app.post("/predict")
def predict(req: PredictRequest):

    # 1. Look up historical stats
    match = lookup[
        (lookup["college_code"] == str(req.college)) &
        (lookup["branch_code"]  == req.branch.upper()) &
        (lookup["community"]    == req.community)
    ]
    if match.empty:
        return {"error": f"No data found for College {req.college} | Branch {req.branch} | {req.community}"}

    row = match.iloc[0]

    try:
        comm_enc = le_community.transform([req.community])[0]
    except:
        return {"error": f"Community '{req.community}' not recognized"}

    # 2. ML probability
    inp = pd.DataFrame([{
        "cutoff"        : req.cutoff,
        "community_enc" : comm_enc,
        "college_enc"   : row["college_enc"],
        "branch_enc"    : row["branch_enc"],
        "hist_opening"  : row["hist_opening"],
        "hist_closing"  : row["hist_closing"],
        "hist_mean"     : row["hist_mean"],
        "hist_count"    : row["hist_count"],
        "cutoff_trend"  : row["cutoff_trend"],
        "rank"          : 0
    }])
    prob = round(float(model.predict_proba(inp[FEATURES])[0][1]) * 100, 1)

    # 3. Scrape 2025 cutoff
    cutoff_2025 = scrape_2025_cutoff(req.college, req.branch, req.community)

    # 4. Demand boost
    boost_map = {b.branch.upper(): b for b in req.demand_boost_branches}
    boost_info = boost_map.get(req.branch.upper())
    boost_marks   = boost_info.boost_marks if boost_info else 0.0
    boost_reason  = boost_info.reason if boost_info else None
    boost_priority = list(boost_map.keys()).index(req.branch.upper()) + 1 if boost_info else None

    # 5. Compute 2026 estimate
    base_ref     = cutoff_2025 if cutoff_2025 else row["last_closing"]
    trend_contrib = float(row["cutoff_trend"]) if not req.use_manual_boost_only else 0.0
    est_2026      = base_ref + trend_contrib + boost_marks
    gap_2026      = round(req.cutoff - est_2026, 2)

    # 6. Verdict
    if gap_2026 >= 3:    verdict = "HIGH"
    elif gap_2026 >= 0:  verdict = "MODERATE"
    else:                verdict = "LOW"

    # 7. Historical citations
    hist = df_hist[
        (df_hist["college_code"] == str(req.college)) &
        (df_hist["branch_code"]  == req.branch.upper()) &
        (df_hist["community"]    == req.community)
    ][["year","cutoff"]].copy()

    yearly_data = []
    if not hist.empty:
        yearly = hist.groupby("year")["cutoff"].agg(
            opening="max", closing="min", mean="mean", count="count"
        ).reset_index()
        for _, r in yearly.iterrows():
            yearly_data.append({
                "year"    : int(r["year"]),
                "opening" : round(r["opening"], 2),
                "closing" : round(r["closing"], 2),
                "mean"    : round(r["mean"], 2),
                "count"   : int(r["count"]),
                "qualified": bool(req.cutoff >= r["closing"])
            })

    # 8. Return full result
    return {
        "college"       : req.college,
        "branch"        : req.branch,
        "community"     : req.community,
        "your_cutoff"   : req.cutoff,
        "ml_probability": prob,
        "verdict"       : verdict,
        "cutoff_2025"   : cutoff_2025,
        "last_closing"  : float(row["last_closing"]),
        "weighted_closing": float(row["hist_closing"]),
        "trend"         : float(row["cutoff_trend"]),
        "trend_contribution": round(trend_contrib, 2),
        "boost_marks"   : boost_marks,
        "boost_priority": boost_priority,
        "boost_reason"  : boost_reason,
        "est_2026_closing": round(est_2026, 2),
        "gap_2026"      : gap_2026,
        "yearly_history": yearly_data
    }

@app.get("/health")
def health():
    return {"status": "ok"}