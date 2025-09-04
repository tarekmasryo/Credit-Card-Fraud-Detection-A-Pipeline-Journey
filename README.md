# 💳 Credit Card Fraud Detection — A Pipeline Journey

> **TL;DR:** A practical, cost-aware pipeline for fraud detection with **time-aware split**, **probability calibration**, and **business-aligned thresholds**.

---

## 📖 The Story  
Less than **0.2% of transactions are fraudulent** — yet they cost banks **billions of dollars globally every year**.  
Detecting fraud is like finding a **needle in a haystack**: only **492 frauds out of 284,807 transactions** in this dataset.

This project builds a pipeline that **learns from time-based patterns and hidden signals** to flag fraud in real-world credit card payments.  
Think of it as a **first line of defense** for banks, where every decision threshold is a trade-off:  
- catching more fraudsters ✅  
- vs. minimizing false alarms ❌ that annoy customers.

---

## 🛠️ The Approach
- **Dataset:** 284,807 transactions with severe imbalance (~0.17% fraud).  
- **Methodology:**  
  - **EDA:** amount distributions & temporal patterns (hours, day-parts).  
  - **Feature Engineering:** `Hour`, `Night_transaction`, `Business_hours`, `log(Amount)`.  
  - **Models:** Logistic, Random Forest, XGBoost, plus **Isotonic Calibration** via **TimeSeriesSplit**.  
  - **Decision Tuning:** either **Precision ≥90%** (customer experience) or **Min-Cost** (loss vs. investigation).  
- **Key Metrics:** **AUPRC (AP)** as the main yardstick; plus **ROC-AUC**, **Brier**, **ECE**, and **Cost-based thresholds**.  

---

## 📊 Key Results (Held-out Test Window)

| Model   | AP(Test) | AP 95% CI       | ROC-AUC(Test) | Brier(Test) | ECE(15) | Thr@P90(val) | Thr@MinCost(val) | Cost@Test@P90 | Cost@Test@MinCost |
|---------|----------|-----------------|---------------|-------------|---------|---------------|------------------|---------------|-------------------|
| RF-Cal  | 0.7894   | [0.700, 0.867]  | 0.9549        | 0.0005      | 0.0003  | 0.6356        | 0.065            | 4030.0        | 3220.0            |
| XGB-Cal | 0.7775   | [0.695, 0.868]  | 0.9768        | 0.0006      | 0.0008  | 0.7718        | 0.169            | 5020.0        | 3190.0            |

➡️ **Calibrated XGBoost** achieved the highest ROC-AUC and strong calibration.  
➡️ **Calibrated Random Forest** had slightly higher AP and lower realized cost at Min-Cost threshold.  

---

## 🏦 Business Lens
Operating points are evaluated under illustrative costs:  
- **False Negative (FN):** $200 (average fraud loss).  
- **False Positive (FP):** $5 (investigation/alert cost).  

👉 Try sensitivity analysis with your own FN/FP costs — even small tweaks can flip which threshold, or even which model, is preferred.  

---

## 📦 Reproducibility
- **Split:** strict **time-based split** (no leakage).  
- **Calibration:** Isotonic regression using **TimeSeriesSplit(cv=3)**.  
- **Thresholds:** always chosen on **validation**, reported only on **test**.  
- **Random seed:** 42 fixed for reproducibility.  

---


## 📌 Insights
- Accuracy ≈ useless under 0.17% fraud prevalence.  
- Calibrated probabilities → thresholds that **actually mean something**.  
- Cost-sensitive analysis bridges the gap between **model metrics** and **business decisions**.  
- Fraud is not random — it clusters in time and amount ranges.  

---

## 📚 References
- Dataset: [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Calibration: Zadrozny & Elkan, *Transforming Classifier Scores into Accurate Multiclass Probability Estimates* (2002).  


