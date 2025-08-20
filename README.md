# 📊 Telco Customer Churn Prediction

## 🔍 Project Overview
This project analyzes customer churn in a telecommunications company using the **Telco Customer Churn dataset**.  
The goal is to build classification models that can predict whether a customer will churn (leave the company), and to identify the most important factors contributing to churn.

This project demonstrates key **data science skills**:
- Data cleaning and preprocessing
- Feature engineering
- Supervised learning (Logistic Regression, Random Forest)
- Model evaluation with ROC-AUC and confusion matrices
- Feature importance analysis

---

## 📁 Dataset
The dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) contains customer demographic information, services subscribed, account details, and whether they churned or not.

- **Rows:** ~7,000 customers  
- **Target Variable:** `Churn` (`Yes` = churned, `No` = retained)

---

## ⚙️ Methods
1. **Data Preprocessing**
   - Dropped irrelevant columns (`customerID`).
   - Converted `TotalCharges` to numeric, handling missing values.
   - Label-encoded categorical variables.
   - Standardized numerical features.

2. **Models Trained**
   - Logistic Regression (baseline model).
   - Random Forest Classifier (ensemble, more robust).

3. **Evaluation**
   - Classification report (precision, recall, F1-score).
   - ROC-AUC scores.
   - ROC Curves comparison.
   - Feature importance ranking.

---

## 📈 Results
- **Logistic Regression**
  - ROC AUC ≈ *0.83* (baseline model).
- **Random Forest**
  - ROC AUC ≈ *0.85–0.87* (stronger predictive performance).
- **Top Churn Predictors**
  - Contract type
  - Tenure (length of time with company)
  - Monthly charges
  - Internet service type

---

## 📊 Visuals
- ROC curves comparing Logistic Regression vs Random Forest  
- Feature importance bar plot (top 10 predictors of churn)

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/telco-customer-churn.git
   cd telco-customer-churn
