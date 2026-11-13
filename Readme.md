# Credit Risk Scorecard with Interpretability

Credit risk default prediction on the **Give Me Some Credit** dataset (150,000 training records, 101,503 test records). The target is `SeriousDlqin2yrs` (90+ days delinquent within 2 years), with a baseline default rate of **6.68%**.

---

## Phase 1: EDA & Data Quality Audit

### Key Discoveries
* **Informative Missingness (`MonthlyIncome`):** 29,731 records (~20%) have missing income. Their default rate is **5.61%**, lower than borrowers who reported income (6.95%). Preserved as an explicit `Missing` bin rather than imputed.
* **Mainframe Sentinel Code (`98`):** 264 records have `98` across all three delinquency columns with a **54.17% default rate** (bureau administrative freeze status code, not 98 literal late payments).
* **Extreme Outliers (`RevolvingUtilization`):** Max utilization reaches ~50,000. Risk drops above 150% due to \$0-limit card anomalies, requiring coarse binning into `>100%` to enforce monotonicity.
* **Data Errors (`age`):** 1 record with `age == 0` dropped.

### Information Value (IV) Leaderboard
* `RevolvingUtilizationOfUnsecuredLines`: **1.136** (Dominant)
* `NumberOfTimes90DaysLate`: **0.882** (Strong)
* `NumberOfTime30-59DaysPastDueNotWorse`: **0.725** (Strong)
* `NumberOfTime60-89DaysPastDueNotWorse`: **0.599** (Strong)
* `age`: **0.231** (Medium)
* Supporting features (`NumberOfOpenCreditLinesAndLoans`, `MonthlyIncome`, `DebtRatio`, `NumberRealEstateLoansOrLines`, `NumberOfDependents`): all $0.035 - 0.081$ (Weak, but retained).

---

## Phase 2: Logistic Scorecard & Scaling (Champion Model)

* **Architecture:** Coarse binning + WoE transformation + constrained Logistic Regression ($\beta > 0$ for all 10 features).
* **FICO Scaling Engine:** Calibrated to **Target Score 600 at 50:1 Odds**, with **PDO = 20 points** (Base Points = 562).
* **Scorecard:** Additive integer lookup table with zero black-box dependencies.

---

## Phase 3: XGBoost Challenger & SHAP Explanations

* **Feature Engineering:** Domain-specific ratios (`TotalDelinquency`, `IncomePerPerson`, `IsBureauFrozen`, `EstimatedDollarDebt`).
* **Model:** Gradient boosted tree ensemble (200 estimators, max depth 5, learning rate 0.05).
* **Interpretability (SHAP):** TreeExplainer used to extract global feature rankings and automated, individual-level Adverse Action reason codes for regulatory compliance (Equal Credit Opportunity Act).

---

## Champion vs. Challenger Tournament Results

| Model | Holdout Test AUC | Gini | KS Statistic | Kaggle Private LB | Kaggle Public LB | Explainability Method |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Scorecard (Champion)** | 0.8590 | 0.7180 | 56.45% | 0.8585 | 0.8524 | Exact Points Table |
| **XGBoost (Challenger)** | **0.8655** | **0.7310** | **57.63%** | **0.8634** | **0.8589** | TreeExplainer SHAP Values |
| **Lift / Delta** | $+0.0065$ | $+0.0130$ | $+1.18\%$ | $+0.0049$ | $+0.0065$ | — |
