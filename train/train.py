import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
import os

from binning_config import BINNING_CONFIG
from train_utils import woe_mapping, bin_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATASET_PATH = "./data/raw/cs-training.csv"
SCORECARD_DIR = "./scorecard/"
TARGET_SCORE = 600
TARGET_ODDS = 50
PDO = 20
CUTOFF_SCORE = 580

def preprocessing(df: pd.DataFrame, binning_config: Dict[str, List[float]], target: str) -> pd.DataFrame:
    woe_map, IV_map = woe_mapping(df, binning_config)
    df_woe = pd.DataFrame(index = df.index)
    if target in df.columns:
        df_woe[target] = df[target]

    for feature, config in binning_config.items():
        bins = config["bins"]
        labels = config["labels"]

        group = bin_features(df[feature], bins, labels)
        group_name = f"{feature}_group"

        df_woe[group_name] = (group.astype(str).map(woe_map[group_name])).astype(float)

    return df_woe, woe_map

def build_scorecard(binning_config: Dict, woe_map: Dict, clf) -> Tuple[pd.DataFrame, Dict]:
    factor = PDO / np.log(2)
    offset = TARGET_SCORE - factor * np.log(TARGET_ODDS)
    base_points = round(offset + factor * clf.intercept_[0])

    coef_dict = dict(zip(clf.feature_names_in_, clf.coef_[0]))

    scorecard_rows = []
    features_json = {}

    for feature, config in binning_config.items():
        group_name = f"{feature}_group"
        coef = coef_dict[group_name]

        clean_bins = [
            999999.0 if b == np.inf else -999999.0 if b == -np.inf else b
            for b in config["bins"]
        ]

        points_map = {}
        for bin_label, woe in woe_map[group_name].items():
            pts = round(factor * coef * woe)
            points_map[bin_label] = pts

            scorecard_rows.append({
                "Feature": feature,
                "Bin": bin_label,
                "WoE": woe,
                "Points": pts
            })

        features_json[feature] = {
            "bins": clean_bins,
            "labels": config["labels"],
            "points": points_map
        }

    scorecard_df = pd.DataFrame(scorecard_rows)
    scorecard_json = {
        "base_points": base_points,
        "cutoff_score": CUTOFF_SCORE,
        "factor": factor,
        "offset": offset,
        "features": features_json
    }

    return scorecard_df, scorecard_json

def save_scorecard(scorecard_json: Dict, scorecard_dir = SCORECARD_DIR):
    os.makedirs(scorecard_dir, exist_ok=True)
    with open(os.path.join(scorecard_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard_json, f, indent=4)



def training(df: pd.DataFrame, binning_config: Dict[str, List[float]], target: str, model: LogisticRegression = None):
    if model is None:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)

    print("--- Starting Scorecard Training Pipeline ---")
    print("Step 1/4: Preprocessing and calculating WoE mappings...")
    df_woe, woe_map = preprocessing(df, binning_config, target)

    X = df_woe.drop(columns=[target])
    y = 1 - df_woe[target]

    print(f"Step 2/4: Fitting Logistic Regression on {len(X):,} records...")
    model.fit(X, y)
    train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"          Intercept: {model.intercept_[0]:.4f} | Train AUC: {train_auc:.4f}")

    print("Step 3/4: Scaling into FICO Scorecard points table...")
    scorecard, scorecard_json = build_scorecard(binning_config, woe_map, model)
    print(f"          Base Points: {scorecard_json['base_points']} | Features: {len(scorecard_json['features'])}")

    save_path = os.path.join(SCORECARD_DIR, "scorecard.json")
    print(f"Step 4/4: Saving production artifact to {save_path}...")
    save_scorecard(scorecard_json)
    print("--- Training Pipeline Completed Successfully! ---\n")

    return scorecard


if __name__ == "__main__":
    print(f"Loading raw dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df[df["age"] > 0]

    clf = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    scorecard = training(df, BINNING_CONFIG, "SeriousDlqin2yrs", clf)
