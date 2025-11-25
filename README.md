```markdown
# ML_Predict: A High-Fidelity Pipeline for Threshold-Based TG4h Risk Classification

This repository provides a fully reproducible, research-grade machine learning pipeline for predicting postprandial triglyceride response phenotypes using multi-threshold TG4h classification, calibrated machine learning models, and advanced model interpretability.  
The entire workflow (data cleaning → preprocessing → model zoo → threshold scanning → calibrated final model → explainability → robustness testing → baseline comparisons → pseudo-external validation) is implemented in a single script: **ML_Predict.py**.

The design philosophy follows **Q1-level methodological transparency** while keeping the documentation readable for practitioners (Hybrid Academic + Engineering style).

---

## 1. Overview

`ML_Predict.py` implements a comprehensive ML workflow for analyzing the relationship between fasting biomarkers and postprandial triglyceride response (TG4h).  
The script supports:

- Multi-threshold phenotype construction (TG4h 60th → 90th percentile)
- Extensive model zoo with 8 classifiers
- 5-fold cross-validation across thresholds
- Selection of the optimal threshold + model
- Train/test split with probability calibration
- SHAP-based explainability
- Partial Dependence analysis (1D/2D)
- Bootstrap confidence intervals
- Subgroup performance analysis (sex, age)
- Pseudo-external validation
- TG0h-only baseline comparisons
- Decision Curve Analysis
- Robustness via Repeated Stratified CV (5×10)

This pipeline is optimized for **high scientific rigor**, with clear reproducibility steps, structured flow, and domain-specific preprocessing.

---

## 2. Dataset Assumptions

The script expects a CSV containing fasting biomarkers and TG4h measurement.  
Default assumptions follow the structure used in cardiometabolic studies:

- **Features**
  - Age  
  - Sex (Male/Female)  
  - Hematocrit  
  - TotalProtein  
  - WBV (Whole Blood Viscosity)  
  - TG0h (fasting TG)  
  - HDL  
  - LDL  
  - BMI  

- **Target**
  - TG4h (postprandial triglycerides measured at 4 hours)

Required columns are validated on load.

---

## 3. Pipeline Summary

The workflow follows the schematic below:

```
Load Dataset
     ↓
Clean Missing Data
     ↓
Define TG4h Thresholds (60–90%)
     ↓
Create Binary Labels for Each Threshold
     ↓
Model Zoo (8 ML Models)
     ↓
5-fold CV Performance Comparison
     ↓
Select Best Threshold + Best Model
     ↓
Train/Test Split
     ↓
Probability Calibration (Isotonic/Sigmoid)
     ↓
Final Calibrated Model
     ↓
Explainability (SHAP + PDP)
     ↓
Bootstrapping (AUROC, Brier)
     ↓
Baseline Comparisons (TG0h-only)
     ↓
Pseudo-external Validation
     ↓
Repeated CV (5×10) Robustness Check
```

---

## 4. Model Zoo

The script benchmarks the following classifiers:

| Model Key | Algorithm | Notes |
|----------|-----------|-------|
| `logreg_l2` | Logistic Regression (L2) | Balanced weights, liblinear |
| `logreg_elastic` | Elastic Net Logistic | L1/L2 mix, saga solver |
| `rf` | Random Forest | 400 trees, balanced subsample |
| `gb` | Gradient Boosting | Default GBM |
| `xgb` | XGBoost | 600 trees, shallow depth, LR=0.03 |
| `lgbm` | LightGBM | Balanced class weights, 600 trees |
| `catboost` | CatBoost | Handles categorical features well |
| `svm_rbf` | RBF SVM | probability=True + balanced class weight |

Models are wrapped in a unified `Pipeline` with preprocessing.

---

## 5. Preprocessing

A rigorous preprocessing block is implemented using scikit-learn’s `ColumnTransformer`:

- **Numeric features** → `StandardScaler`
- **Categorical features** → `OneHotEncoder(drop="first")`

All transformations occur *inside the pipeline* to prevent leakage.

---

## 6. Multi-Threshold TG4h Classification

The script evaluates thresholds:

```
0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
```

For each threshold:
- Compute TG4h percentile  
- Convert TG4h ≥ cutoff → High responder (1), else (0)
- Run full model zoo
- Compute 5-fold CV metrics:
  - AUROC  
  - Brier  
  - F1  
  - Precision  
  - Recall  
  - Accuracy  

All results are stored in:
- `cv_results_all_thresholds.csv`
- `cv_auroc_summary_pivot.csv`

This allows a full methodological audit trail.

---

## 7. Model Selection

The script identifies the **main threshold** (default = 0.75) and selects the model with the highest mean AUROC.  
This becomes the **MAIN MODEL** used for all downstream analyses.

---

## 8. Train/Test Split and Calibration

After selecting the optimal threshold and model:

1. The dataset is split:  
   - **75% train**  
   - **25% test**  
   - Stratified  
   - Controlled random seed

2. Calibration methods evaluated:
   - **Isotonic**
   - **Platt scaling (sigmoid)**

3. Best calibration (lowest Brier score) becomes the **FINAL MODEL**.

Probability calibration is essential for medical ML and aligns with Q1 publication standards.

---

## 9. Evaluation Metrics

Evaluation on the held-out test includes:

- AUROC  
- Brier score  
- Precision, Recall, F1  
- ROC Curve  
- Calibration Curve  
- Calibration slope & intercept  

These metrics follow TRIPOD and ML reproducibility guidelines.

---

## 10. Explainability

### 10.1 SHAP (KernelExplainer)

- Global summary values  
- TG0h dependence plot  
- WBV interactions  
- Feature correlation structure  

### 10.2 Partial Dependence (PDP)

- 1D PDP for TG0h, WBV, BMI  
- 2D PDP for TG0h × BMI  
- High-resolution PDP for TG0h  

These methods quantify both average and interaction effects.

---

## 11. Robustness Testing

### 11.1 Bootstrap 95% CI  
For the final calibrated model:
- Approx. 1000 bootstrap samples  
- AUROC mean + CI  
- Brier mean + CI  

### 11.2 Repeated Stratified CV (5×10)

50 folds total, reporting:
- Mean AUROC ± SD  
- Min/Max AUROC  
- Mean Brier ± SD  

This ensures stability beyond a single train/test split.

---

## 12. Baseline Models

Two TG0h-only models are built as clinical baselines:

| Baseline | Description |
|----------|-------------|
| TG0h percentile rule | Threshold using 75th percentile on TG0h |
| Univariate logistic | TG0h → calibrated logistic regression |

This demonstrates the added value of the multivariable model.

---

## 13. Pseudo-External Validation

Two clinically relevant scenarios:

### (A) Train <55 yrs → Test ≥55 yrs  
### (B) Train Male → Test Female  

For each:
- Train on one demographic  
- Test on the other  
- Report AUROC & Brier

This simulates domain shift and demographic transportability.

---

## 14. Decision Curve Analysis (DCA)

The script computes net benefit across probability thresholds for:

- Final calibrated model  
- TG0h logistic baseline  
- TG0h cutoff rule  
- Treat-all  
- Treat-none  

DCA evaluates clinical utility, not just discrimination.

---

## 15. Outputs

The script generates:

- CV results tables  
- Train/test evaluation metrics  
- Calibration statistics  
- SHAP values  
- PDP summaries  
- Bootstrap distributions  
- Subgroup performance tables  
- DCA tables  

No figures are required, but these outputs can be formatted into tables for manuscripts.

---

## 16. Reproducibility

To align with Q1 and clinical-AI reproducibility requirements:

- Fixed `RANDOM_STATE = 42`
- Seeded numpy operations
- All preprocessing inside pipelines
- No leakage across folds
- All thresholds explicitly logged
- Every model evaluated with identical CV structure
- Final model calibrated on held-out data

The entire workflow is transparent and repeatable from raw CSV to final performance metrics.

---

## 17. Citation

If using this repository in academic work:

**Piyavechvirat, N.**  
*ML_Predict: A Calibrated Multi-Threshold Machine Learning Pipeline for Postprandial Lipid Response Analysis.*  
GitHub Repository, 2025.

---

## 18. Contact

Author: **Nattakitti Piyavechvirat**  
GitHub: https://github.com/NattakittiP  
For issues, questions, or collaboration proposals, please open an Issue or Pull Request.
```

