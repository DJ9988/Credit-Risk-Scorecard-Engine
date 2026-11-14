# Credit Risk Scorecard & Explainable AI Engine

A production-style retail credit risk scoring system built on Kaggle's **Give Me Some Credit** dataset (150,000 borrower records). 

Instead of treating credit scoring as a black box, this project implements an institutional-grade **FICO-calibrated Weight of Evidence (WoE) Scorecard**, a **gradient-boosted Challenger (XGBoost) with SHAP explanations**, and a **sub-millisecond FastAPI web application** that automatically generates legal **Regulation B Adverse Action Notices** for denied applicants.

---

## Interactive Dashboard Showcase

The system includes a live web UI served directly by FastAPI. It supports one-click borrower presets, instant 300–850 credit score calculation, and dynamic legal adverse action reason disclosures.

### 1. Prime Borrower (Approved)
![Prime Borrower Approval](img/Screenshot%20From%202026-11-14%2012-33-29.png)

### 2. Subprime Borrower & Adverse Action Notice (Denied)
![Subprime Denial & Adverse Action Reasons](img/Screenshot%20From%202026-11-14%2012-34-06.png)

---

## What Was Built

### 1. FICO-Calibrated Logistic Scorecard (Champion Model)
* **Coarse Binning & WoE:** Binned all 10 continuous and discrete features to ensure strict monotonicity and handle missing values (`MonthlyIncome` preserved as an explicit informative bin).
* **Constrained Modeling:** Logistic regression trained on log-odds ($\beta > 0$ across all features), guaranteeing that lower risk always yields higher scores.
* **Score Scaling:** Calibrated to **Target Score 600 at 50:1 Odds** with **PDO = 20 points** (Base Points = 562). An underwriter can score any applicant via simple integer addition.

### 2. XGBoost Challenger & SHAP Interpretability
* **Feature Engineering:** Domain-specific ratios (`TotalDelinquency`, `IncomePerPerson`, `IsBureauFrozen`, `EstimatedDollarDebt`).
* **SHAP Explainability:** Uses `TreeExplainer` to extract global feature rankings and individual-level marginal risk contributions, automating compliance with the **Equal Credit Opportunity Act (ECOA)**.

### 3. Production Serving Architecture
* **Decoupled Pipeline:** Offline training script (`train/train.py`) exports a frozen configuration artifact (`scorecard/scorecard.json`).
* **Zero-Framework Inference:** The FastAPI service (`api/app.py`) scores applicants using pure dictionary lookup—executing in **$<1\text{ms}$** with zero heavy ML dependencies at runtime.

---

## Model Benchmark & Tournament Results

Both models were validated on an 80/20 holdout test set and Kaggle's unseen global test set (101,503 borrowers):

| Metric | Logistic Scorecard (Champion) | XGBoost (Challenger) | Lift / Delta |
| :--- | :---: | :---: | :---: |
| **Holdout Test AUC** | `0.8590` | **`0.8655`** | $+0.0065$ |
| **Gini Coefficient** | `0.7180` | **`0.7310`** | $+0.0130$ |
| **KS Statistic** | `56.45%` | **`57.63%`** | $+1.18\%$ |
| **Kaggle Private LB** | `0.8585` | **`0.8634`** | $+0.0049$ |
| **Kaggle Public LB** | `0.8524` | **`0.8589`** | $+0.0065$ |
| **Interpretability** | Exact Integer Points Table | TreeExplainer SHAP Values | — |
| **Inference Latency** | **$< 1\text{ms}$** | $\approx 15\text{ms}$ | — |

> **Key Insight:** The linear scorecard achieves **`0.8585` on Kaggle Private LB**—within ~0.005 AUC of the XGBoost challenger—proving why financial institutions prefer scorecards: near-identical predictive power with 100% mathematical transparency and instant auditability.

---

## Quickstart & Local Setup

### 1. Environment & Dependencies
```bash
git clone https://github.com/your-username/credit-risk-scorecard.git
cd credit-risk-scorecard

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Download (Optional)
The training pipeline expects `cs-training.csv` in `data/raw/`:

```bash
# Download and extract Give Me Some Credit from Kaggle
mkdir -p data/raw
kaggle competitions download -c GiveMeSomeCredit -p data/raw
unzip data/raw/GiveMeSomeCredit.zip -d data/raw
```
*Expected path:* `data/raw/cs-training.csv`

### 3. Retrain Scorecard Artifact (Optional)
Generates `scorecard/scorecard.json` from raw training data:
```bash
python train/train.py
```

### 4. Launch Web Dashboard & API
```bash
uvicorn api.app:app --reload
```
Open **`http://localhost:8000`** in your browser.

* **API Endpoints:**
  * `GET /`: Interactive web dashboard.
  * `GET /health`: Service health check.
  * `POST /predict`: Takes borrower financial profile JSON $\to$ returns credit score, decision, default probability, and adverse action reasons.
