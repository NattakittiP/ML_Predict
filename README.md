# Machine Learning Pipeline for Postprandial Triglyceride Response Prediction  
*A leakage-controlled, calibrated, and explainable ML framework using fasting biomarkers and blood rheology*

---

## 📌 Overview
This repository provides a complete, reproducible machine learning pipeline for predicting **postprandial triglyceride response phenotypes (High TG4h vs. Normal)** using clinical, biochemical, and hemorheological features — including **Whole Blood Viscosity (WBV)**.

The entire workflow is implemented in a single script:

📄 **TG_Predict.py**

This pipeline follows modern best practices in clinical machine learning:
- Leakage-controlled cross-validation  
- Model comparison across 3 algorithms  
- Probability calibration (Isotonic Regression)  
- Test-set evaluation  
- SHAP explainability  
- Descriptive statistics and phenotype analysis  
- Full visualization suite (ROC, Calibration, SHAP, Correlation, Distributions)

---

## 🧪 Dataset
The dataset contains approximately **1,500 anonymized hospital records**, featuring:

- Demographics  
- Hematocrit, Total Protein, WBV  
- Fasting & 4-hour triglycerides (TG0h, TG4h)  
- HDL, LDL  
- BMI  

### **Label Definition**
A binary outcome (`High_TG4h`) is created using the **75th percentile** of TG4h:
- `1` = High postprandial TG response  
- `0` = Normal TG response  

---

## ⚙️ Machine Learning Pipeline

### **1. Preprocessing**
- Standardization of numerical features  
- One-Hot Encoding of categorical variables  
- Combined using `ColumnTransformer`

### **2. Models Evaluated**
- Logistic Regression  
- Random Forest  
- Gradient Boosting  

### **3. Leakage-Controlled Cross-Validation**
5-fold stratified CV is used to estimate performance while preventing information leakage.

Metrics computed:
- AUROC  
- Accuracy  
- F1 Score  
- Precision / Recall  
- Brier Score  

---

## 🏆 Best Model
Based on cross-validated AUROC, the best model was:

### **Logistic Regression (with isotonic calibration)**

**Test-set performance:**
- **AUROC:** 0.776  
- **Accuracy:** 0.781  
- **F1 Score:** 0.446  
- **Brier Score:** 0.156  
---

## 🔍 Statistical Outputs
Numerical reports (optional) include:

- **Descriptive statistics** (mean ± SD by phenotype)  
- **Cross-validated model performance**  
- **Calibrated test-set results**

These can be automatically generated when running the script.

---

## 🚀 Running the Pipeline

### **1. Install dependencies**
```bash
pip install numpy pandas matplotlib scikit-learn shap


