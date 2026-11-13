import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

def bin_features(series: pd.Series, bins: List, labels: List[str], right: bool = False, missing_label = "Missing"):
    group = pd.cut(series, bins = bins, labels = labels, right = right, include_lowest = True)

    group = group.cat.add_categories(missing_label).fillna(missing_label)

    return group.cat.remove_unused_categories()

def calculate_woe(df: pd.DataFrame, feature: str, target: str = "SeriousDlqin2yrs", epilson: float = 1e-6) -> tuple[pd.DataFrame, float]:
    total_goods = (df[target] == 0).sum()
    total_bads = (df[target] == 1).sum()

    summary = df.groupby(feature, observed = False)[target].agg(
        total = "count",
        bads = "sum"
    ).reset_index()

    summary["goods"] = summary["total"] - summary["bads"]
    summary["bad_rate"] = (summary["bads"] / summary["total"] * 100).round(2)

    summary["percentage_good"] = summary["goods"] / total_goods
    summary["percentage_bad"] = summary["bads"] / total_bads

    summary["percentage_good"] = summary["percentage_good"].replace(0, epilson)
    summary["percentage_bad"] = summary["percentage_bad"].replace(0, epilson)

    summary["WoE"] = np.log(summary["percentage_good"] / summary["percentage_bad"])
    summary["IV"] = (summary["percentage_good"] - summary["percentage_bad"]) * summary["WoE"]

    total_iv = summary["IV"].sum()
    
    return summary, total_iv

def woe_mapping(df: pd.DataFrame, config: Dict) -> tuple[Dict, Dict]:
    woe_map = {}
    IV_map = {}

    for feature, con in config.items():
        bin = con["bins"]
        labels = con["labels"]

        group = bin_features(df[feature], bin, labels)

        group_name = f"{feature}_group"
        df[group_name] = group

        summary, total_iv = calculate_woe(df, group_name)

        woe_map[group_name] = {
            row[group_name]:row["WoE"]
            for _, row in summary.iterrows()
        }

        IV_map[group_name] = total_iv
    return woe_map, IV_map
