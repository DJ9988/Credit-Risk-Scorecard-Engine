import json
from fastapi import FastAPI
import math
from pathlib import Path

try:
    from api.schemas import BorrowerProfile, ScoreResponse
except ModuleNotFoundError:
    from schemas import BorrowerProfile, ScoreResponse

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

scorecard_path = BASE_DIR / "scorecard" / "scorecard.json"
reasons_path = BASE_DIR / "api" / "reasons.json"

with open (scorecard_path, 'r') as f:
    scorecard = json.load(f)

with open(reasons_path, 'r') as f:
    reasons_dict = json.load(f)

base_points = scorecard["base_points"]
cutoff_score = scorecard["cutoff_score"]
factor = scorecard["factor"]
offset = scorecard["offset"]
features = scorecard["features"]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "Scorecard-v1"
    }

@app.post("/predict")
def score(profile: BorrowerProfile) -> ScoreResponse:
    total_score = base_points
    feature_points = {}
    profile_data = profile.model_dump(by_alias=True)

    for feature in features:
        profile_feature = profile_data[feature]
        bins = features[feature]["bins"]
        labels = features[feature]["labels"]
        pts_dict = features[feature]["points"]
        pts = 0

        if profile_feature is None and 'Missing' in pts_dict:
            pts = pts_dict["Missing"]
        elif isinstance(profile_feature, int) or isinstance(profile_feature, float):
            for i, bin in enumerate(bins):
                if i < len(bins) - 1:
                    if bins[i] <= profile_feature < bins[i + 1]:
                        pts = pts_dict[labels[i]]
                        break
        feature_points[feature] = pts
        total_score += pts
    
    decision = "APPROVED" if total_score >= cutoff_score else "DENIED"

    log_odds = (total_score - offset) / factor
    odds = math.exp(log_odds)
    prob = 1 / (1 + odds)
    
    if decision == "DENIED":
        worst_features = sorted(feature_points.items(), key=lambda x: x[1])[:3]
        reasons = [reasons_dict[feature] for feature, pts in worst_features]
    else:
        reasons = []
    

    return ScoreResponse(
        credit_score = total_score,
        decision = decision,
        default_probability = prob,
        adverse_actions_reasons = reasons
    )



if __name__ == "__main__":
    sample_profile = BorrowerProfile(
        age=30,
        RevolvingUtilizationOfUnsecuredLines=0.1,
        NumberOfTime30_59DaysPastDueNotWorse=0,
        DebtRatio=0.8,
        MonthlyIncome=None,
        NumberOfOpenCreditLinesAndLoans=0,
        NumberOfTimes90DaysLate = 1,
        NumberRealEstateLoansOrLines = 3,
        NumberOfTime60_89DaysPastDueNotWorse = 1,
        NumberOfDependents = 1
    )
    print(score(sample_profile))