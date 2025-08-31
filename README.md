# 💳 Credit Card Fraud Detection — A Pipeline Journey

> Less than **0.2%** of transactions are fraudulent, yet they cost banks **billions** every year.  
> Detecting fraud is like finding a needle in a haystack: only **492** frauds out of **284,807** transactions in this dataset.

This project builds a production-style pipeline that **learns time patterns and hidden signals** to flag fraud in real card payments.  
Think of it as a **first line of defense**: every decision threshold balances catching more fraudsters vs. minimizing false alarms.

---

## 📖 The Story
- Fraudulent transactions are extremely rare (**0.17% prevalence**) but have **huge financial impact**.  
- The challenge is the **severe class imbalance**: only a handful of fraudulent records hidden inside hundreds of thousands of genuine ones.  
- The system must balance **accuracy, cost, and customer experience**.  

---

## 📌 The Approach

- **Dataset**: `creditcard.csv` (284,807 rows, fraud ≈ **0.17%**).  
- **Time-aware split**: 80% train → 20% test (inner validation from the train set).  
- **Feature Engineering**:
  - `Hour` (from `Time`)
  - `Night_transaction` (22:00–05:00)
  - `Business_hours` (09:00–17:00)
  - `log(Amount)` as `_log_amount`
- **Models**:
  - Logistic Regression (with/without resampling: SMOTE / Under / SMOTE+Tomek)  
  - Random Forest  
  - XGBoost  
  - **Isotonic calibration** with `TimeSeriesSplit` (CV on train only)  
- **Decision Tuning**:
  - **Precision ≥ 90%** operating point (to reduce false alarms)  
  - **Min-Cost** operating point (FN = **$200**, FP = **$5**)  
- **Key Metrics**:  
  - AUPRC, ROC-AUC, Brier Score, Recall@P90, Realized Cost  

---

## ✅ Results

### Model Comparison (Test)

| Model                     | AUPRC  | ROC-AUC | Brier  |
|---------------------------|:------:|:------:|:------:|
| RandomForest (Pipeline)   | **0.8128** | 0.9338 | 0.0004 |
| XGB (Pipeline)            | 0.7917 | 0.9735 | 0.0005 |
| XGB-Calibrated (Pipeline) | 0.7623 | **0.9749** | 0.0006 |
| Logistic (Pipeline)       | 0.7415 | 0.9857 | 0.0337 |

> Calibration improves **probability quality** (Brier/ROC-AUC), which is crucial for threshold optimization.

---

### Thresholded Performance (XGB-Calibrated, Test)

- **Precision ≥ 90%** threshold (`~0.866`):  
  - Precision **0.92**, Recall **0.622**, F1 **0.742**  
  - Confusion matrix: `TP=46, FP=4, FN=28, TN=56668`  
  - **Realized Cost**: **$5,620**

- **Min-Cost** threshold (`~0.204`):  
  - Precision **0.484**, Recall **0.797**, F1 **0.602**  
  - Confusion matrix: `TP=59, FP=63, FN=15, TN=56609`  
  - **Realized Cost**: **$3,315**

💡 Business takeaway:  
- If **budget** is priority → use **Min-Cost**.  
- If **customer trust** is priority → use **P≥90%**.  

---

### Resampling Snapshot (Logistic Regression)

| Sampler        | VAL AUPRC | TEST AUPRC | TEST ROC-AUC | Recall@P90 |
|----------------|:---------:|:----------:|:------------:|:----------:|
| SMOTE          | 0.7714    | **0.7714** | 0.9659       | **0.7297** |
| SMOTE+Tomek    | 0.7714    | **0.7714** | 0.9659       | **0.7297** |
| None           | 0.6706    | 0.7072     | **0.9743**   | 0.4595     |
| Under          | 0.2766    | 0.4476     | 0.9827       | 0.7162     |

---

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
