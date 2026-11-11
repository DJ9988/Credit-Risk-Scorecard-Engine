# Credit Risk Scorecard

Credit risk default prediction on the **Give Me Some Credit** dataset (150,000 records, 10 features). The target is `SeriousDlqin2yrs` (90+ days delinquent within 2 years), with a baseline default rate of **6.68%**.

---

## Phase 1: EDA & Data Quality Audit Findings

### Key Discoveries
* **Informative Missingness (`MonthlyIncome`):** 29,731 records (~20%) have missing income. Their default rate is **5.61%**, lower than borrowers who reported income (6.95%). Missingness carries signal and must be kept as an explicit bin rather than imputed.
* **Mainframe Sentinel Code (`98`):** 264 records have `98` across all three delinquency columns with a **54.17% default rate**. This is a credit bureau system status code (e.g., frozen account), not a literal count of 98 missed payments.
* **Extreme Outliers (`RevolvingUtilization`):** Values reach up to 50,000. Risk drops above 150% due to \$0/\$1 limit accounts with minor trailing balances. Coarse binning is required to enforce monotonicity.
* **Data Errors (`age`):** Exactly 1 record has `age == 0` (entry error to drop).

---

### Information Value (IV) Summary

Calculated using the underwriting convention ($\text{WoE} = \ln(\% \text{Goods} / \% \text{Bads})$):

| Feature | Total IV | Tier | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **`RevolvingUtilizationOfUnsecuredLines`** | **1.142** | Dominant | Borrowers with $<10\%$ utilization have a 1.8% default rate vs. ~40% for $>100\%$. |
| **`NumberOfTimes90DaysLate`** | **0.873** | Strong | Having even 1 late payment jumps default rate from 4.6% to 33.7%. |
| **`age`** | **0.243** | Medium | Strictly monotonic: risk drops steadily from 11.7% (18–29) to 2.3% (70+). |
| **`MonthlyIncome`** | **0.072** | Weak | Raw income alone is weakly predictive compared to repayment behavior. |
