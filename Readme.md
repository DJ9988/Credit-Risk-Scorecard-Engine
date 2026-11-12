# Credit Risk Scorecard with Interpretability

Credit risk default prediction on the **Give Me Some Credit** dataset (150,000 training records, 101,503 test records). The target is `SeriousDlqin2yrs` (90+ days delinquent within 2 years), with a baseline default rate of **6.68%**.

---

## Phase 1: EDA & Data Quality Audit

### Key Discoveries
* **Informative Missingness (`MonthlyIncome`):** 29,731 records (~20%) have missing income. Their default rate is **5.61%**, lower than borrowers who reported income (6.95%). Missingness carries signal and is preserved as an explicit `Missing` bin rather than imputed.
* **Mainframe Sentinel Code (`98`):** 264 records have `98` across all three delinquency columns with a **54.17% default rate**. This is a credit bureau system status code (e.g., frozen account), not a literal count of 98 missed payments.
* **Extreme Outliers (`RevolvingUtilization`):** Values reach up to 50,000. Risk drops above 150% due to \$0/\$1 limit accounts with minor trailing balances. Coarse binning is required to enforce monotonicity.
* **Data Errors (`age`):** Exactly 1 record has `age == 0` (entry error to drop).

### Information Value (IV) Leaderboard
* `RevolvingUtilizationOfUnsecuredLines`: **1.136** (Dominant)
* `NumberOfTimes90DaysLate`: **0.882** (Strong)
* `NumberOfTime30-59DaysPastDueNotWorse`: **0.725** (Strong)
* `NumberOfTime60-89DaysPastDueNotWorse`: **0.599** (Strong)
* `age`: **0.231** (Medium)
* Supporting features (`NumberOfOpenCreditLinesAndLoans`, `MonthlyIncome`, `DebtRatio`, `NumberRealEstateLoansOrLines`, `NumberOfDependents`): all $0.035 - 0.081$ (Weak, but retained).

---

## Phase 2: Logistic Scorecard & Scaling

### Methodology
1. **Stratified Split:** 80/20 train/test split preserving the 6.68% default rate to prevent target leakage.
2. **Coarse Binning & WoE:** Binned all 10 features to guarantee minimum bin sample size ($>2\%$) and strict monotonicity. Transformed data into standardized log-odds space ($\text{WoE} = \ln(\% \text{Goods} / \% \text{Bads})$).
3. **Constrained Model:** Fitted Logistic Regression predicting $P(\text{Good})$. All 10 model coefficients are strictly positive ($\beta > 0$), obeying economic risk intuition.
4. **FICO Scaling Engine:** Calibrated to **Target Score 600 at 50:1 Odds**, with **PDO = 20 points** (Factor = 28.854, Offset = 487.123, Base Points = 562).

### Performance & Validation

| Metric | Holdout Test (20%) | Kaggle Leaderboard |
| :--- | :---: | :---: |
| **AUC-ROC** | **0.8590** | **0.8585 (Private)** / **0.8524 (Public)** |
| **Gini ($2 \times \text{AUC} - 1$)** | **0.7180** | — |
| **KS Statistic** | **56.45%** ($p = 0.0$) | — |
| **Score Range** | **405 – 629** (Median: 592) | — |

> **Key Takeaway:** The linear scorecard achieved near-zero generalization gap (0.8590 local vs. 0.8585 Kaggle Private), placing it within ~0.01 AUC of complex black-box gradient-boosted ensembles while remaining 100% transparent and regulatory compliant.
