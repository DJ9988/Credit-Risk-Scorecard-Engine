# Credit Risk Scorecard with Interpretability

Credit risk default prediction on the **Give Me Some Credit** dataset (150,000 training records, 101,503 test records). The target is `SeriousDlqin2yrs` (90+ days delinquent within 2 years), with a baseline default rate of **6.68%**.

---

## Phase 1: EDA & Data Quality Audit

### Key Discoveries
* **Informative Missingness (`MonthlyIncome`):** 29,731 records (~20%) have missing income with a **5.61% default rate** (lower than the 6.95% rate for reported income). Retained as an explicit `Missing` bin to preserve signal.
* **Mainframe Sentinel Code (`98`):** 264 records exhibit `98` across all three delinquency columns with a **54.17% default rate**. Identified as a credit bureau status code (e.g., frozen account), not literal missed payments.
* **Extreme Outliers (`RevolvingUtilization`):** Values reach up to 50,000 with a non-monotonic risk reversal above 150% due to \$0/\$1 limit accounts. Coarse binning into `>100%` enforeces monotonicity.
* **Data Errors (`age`):** Filtered out 1 invalid record with `age == 0`.

### Information Value (IV) Leaderboard
* `RevolvingUtilizationOfUnsecuredLines`: **1.136** (Dominant)
* `NumberOfTimes90DaysLate`: **0.882** (Strong)
* `NumberOfTime30-59DaysPastDueNotWorse`: **0.725** (Strong)
* `NumberOfTime60-89DaysPastDueNotWorse`: **0.599** (Strong)
* `age`: **0.231** (Medium)
* Supporting features (`NumberOfOpenCreditLinesAndLoans`, `MonthlyIncome`, `DebtRatio`, `NumberRealEstateLoansOrLines`, `NumberOfDependents`): $0.035 - 0.081$ (Weak, but retained).

---

## Phase 2: Logistic Scorecard (Champion Model)

### Methodology
1. **Stratified Split:** 80/20 train/test split preserving the 6.68% default rate to prevent data leakage.
2. **Coarse Binning & WoE:** Binned all 10 features to guarantee minimum bin sample size ($>2\%$) and strict monotonicity. Transformed data into standardized log-odds space ($\text{WoE} = \ln(\% \text{Goods} / \% \text{Bads})$).
3. **Constrained Model:** Fitted Logistic Regression predicting $P(\text{Good})$. All 10 model coefficients are strictly positive ($\beta > 0$).
4. **FICO Scaling Engine:** Calibrated to **Target Score 600 at 50:1 Odds**, with **PDO = 20 points** (Factor = 28.854, Offset = 487.123, Base Points = 562).

---

## Phase 3: XGBoost Challenger Model & SHAP Explanations

### Feature Engineering & Modeling
* Engineered financial domain features: `TotalDelinquency`, `IncomePerPerson`, `IsBureauFrozen`, and `EstimatedDollarDebt`.
* Trained an **XGBoost Classifier** (200 trees, depth 5, learning rate 0.05, subsample 0.8).
* Applied **SHAP (TreeExplainer)** to resolve the black-box dilemma:
  * **Global Interpretation (Beeswarm):** Confirmed model decisions align with economic risk (TotalDelinquency and Utilization drive risk up; Age drives risk down).
  * **Local Interpretation (Waterfall):** Successfully extracted top 3 adverse action reason codes per denied applicant for Regulation B compliance.

---

## Champion vs. Challenger Tournament Results

| Metric | Phase 2 Logistic Scorecard (Champion) | Phase 3 XGBoost (Challenger) | Lift ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Local Test AUC (20%)** | `0.8590` | **`0.8655`** | **$+0.0065$** |
| **Local Gini** | `0.7180` | **`0.7310`** | **$+0.0130$** |
| **Local KS Statistic** | `56.45%` | **`57.63%`** | **$+1.18\%$** |
| **Kaggle Public Leaderboard** | `0.85239` | **`0.85893`** | **$+0.00654$** |
| **Kaggle Private Leaderboard** | `0.85852` | **`0.86344`** | **$+0.00492$** |

> **Conclusion:** The XGBoost challenger delivers a consistent **+0.5% to +0.65% AUC lift** on both local and Kaggle holdout sets. With SHAP providing exact local feature attributions, the challenger meets regulatory explainability standards while capturing non-linear interactions.
