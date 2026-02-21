# ML_Predict

## A High-Fidelity, Leakage-Controlled Pipeline for Threshold-Based TG4h Risk Classification

ML_Predict is a fully reproducible, research-grade machine learning
framework for modeling postprandial triglyceride response phenotypes
using multi-threshold TG4h classification, calibrated logistic
regression models, and structured interpretability analysis.

The repository implements a coordinated, audit-ready workflow spanning:

-   Nested cross-validation with strict leakage control
-   Multi-threshold phenotype sensitivity
-   Probability calibration (OOF + held-out comparison)
-   SHAP and Partial Dependence explainability
-   Bootstrap reliability estimation
-   Decision Curve Analysis
-   Demographic pseudo-external transport
-   External synthetic stress testing
-   WBV misspecification sensitivity analysis

The design philosophy follows Q1-level methodological rigor, while
maintaining documentation clarity suitable for both academic researchers
and applied ML practitioners.

------------------------------------------------------------------------

# 1. Overview

The core pipeline (run_rebuild_pipeline_Rerun.py) implements a
comprehensive, leakage-controlled ML workflow to model the relationship
between fasting biomarkers and postprandial triglyceride response
(TG4h).

Supporting scripts:

-   summary_table_regen_Rerun.py
-   External_validation_Rerun.py
-   Sensitivity_of_the_null_Association_Rerun.py

Together, these provide a complete end-to-end framework for:

-   Multi-threshold phenotype construction (60th--90th percentiles;
    primary focus: 75th)
-   Nested 5×5 cross-validation
-   Threshold sensitivity analysis
-   Probability calibration (isotonic + sigmoid)
-   SHAP explainability
-   Partial Dependence analysis (1D & 2D)
-   Bootstrap confidence intervals (1,000 resamples)
-   Repeated stratified CV (5×10)
-   TG0h-only baseline comparisons
-   Decision Curve Analysis
-   Demographic pseudo-external transfer testing
-   External synthetic cohort stress testing
-   WBV misspecification sensitivity analysis

All phenotype definitions and preprocessing steps are strictly
training-fold confined to eliminate information leakage.

------------------------------------------------------------------------

# 2. Dataset Assumptions

All scripts expect a CSV file (e.g., Dataset.csv) containing fasting
biomarkers and TG4h measurements.

## Required Features

| Category         | Variables                         |
|------------------|-----------------------------------|
| Demographic      | Age, Sex                          |
| Hematologic      | Hematocrit, TotalProtein          |
| Derived          | WBV                               |
| Lipid            | TG0h, HDL, LDL                    |
| Anthropometric   | BMI                               |
| Target           | TG4h                              |

## WBV Derivation

WBV is computed using the de Simone low-shear surrogate:

WBV = 0.12 × Hct + 0.17 × TotalProtein − 0.3519

TG4h-derived labels are never used as predictors.

Flexible column names are automatically mapped during loading.

------------------------------------------------------------------------

# 3. End-to-End Workflow

```text
Load Dataset
      ↓
Normalize Sex coding
      ↓
Derive WBV and TGR
      ↓
Physiologic exclusions
      ↓
Define TG4h phenotype (primary: 75th percentile)
      ↓
Nested CV (5×5; fold-specific cutoff)
      ↓
OOF predictions + metrics
      ↓
Threshold sensitivity analysis
      ↓
Calibration (OOF + held-out comparison)
      ↓
Explainability (SHAP + PDP)
      ↓
Bootstrap CI (1,000×)
      ↓
Repeated CV (5×10)
      ↓
Baseline comparisons
      ↓
Decision Curve Analysis
      ↓
Demographic pseudo-external transfer
      ↓
External synthetic stress test
      ↓
WBV misspecification sensitivity
      ↓
LaTeX-ready outputs

All TG4h thresholds are computed from training indices only within CV
splits.
```
------------------------------------------------------------------------

# 4. Model Family

## Primary Model

L2-Penalized Logistic Regression

-   class_weight="balanced"
-   solver="lbfgs"
-   Nested hyperparameter tuning over C
-   Fully wrapped in scikit-learn Pipeline

## Optional Comparison Models

| Key     | Model                      |
|---------|----------------------------|
| svm_rbf | RBF SVM                    |
| rf      | Random Forest (600 trees)  |

The manuscript-aligned emphasis remains on the interpretable
L2-penalized logistic regression.

------------------------------------------------------------------------

# 5. Preprocessing

All preprocessing occurs within ColumnTransformer blocks inside CV
pipelines.

## Numeric

-   Median imputation
-   Fold-specific winsorization (1--99%)
-   StandardScaler

## Categorical

-   Most frequent imputation
-   OneHotEncoder (drop="first")

No preprocessing is performed outside CV folds.

------------------------------------------------------------------------

# 6. Multi-Threshold TG4h Classification

Evaluated percentiles:

60, 70, 75, 80, 90

For each threshold:

-   Compute fold-specific TG4h cutoff
-   Generate binary phenotype
-   Run nested CV
-   Save AUROC, Brier, F1, Precision, Recall, Accuracy

Outputs:

-   thresholdsensitivitysummary.csv
-   thresholdsensitivity.csv

------------------------------------------------------------------------

# 7. Primary Phenotype

Primary phenotype:\
TG4h ≥ 75th percentile

Within nested CV:

-   Cutoff computed from training fold only
-   Applied to validation fold
-   AUROC and Brier summarized across outer folds

This model supports:

-   Nested-CV reporting
-   Bootstrap CI
-   Repeated CV
-   SHAP & PDP analysis

------------------------------------------------------------------------

# 8. Calibration Strategy

## Nested CV Calibration

-   CalibratedClassifierCV inside outer folds
-   OOF calibrated probabilities saved

## Held-Out 80/20 Comparison

-   Stratified split
-   Isotonic vs sigmoid comparison
-   Report AUROC, Brier, slope, intercept

Isotonic selected for balanced calibration quality.

## External Synthetic Script

Three-step calibration:

1.  Hard-label training\
2.  Temperature scaling\
3.  Isotonic refinement

------------------------------------------------------------------------

# 9. Evaluation Metrics

Across nested CV, held-out splits, and external tests:

-   AUROC
-   Brier score
-   F1
-   Precision
-   Recall
-   Accuracy
-   ROC curves
-   PR curves
-   Calibration curves
-   Calibration slope & intercept

Outputs saved in CSV, JSON, and figure formats.

------------------------------------------------------------------------

# 10. Explainability

## SHAP

-   Global importance
-   TG0h dependence
-   TG0h × WBV interaction

## Partial Dependence

-   1D: TG0h, WBV, BMI
-   2D: TG0h × BMI

All derived from the final full-data LR model.

------------------------------------------------------------------------

# 11. Robustness

## Bootstrap (1,000×)

-   AUROC distribution
-   Brier distribution
-   Percentile-based 95% CI

## Repeated Stratified CV (5×10)

-   Mean ± SD
-   Min / Max
-   Total evaluated folds

------------------------------------------------------------------------

# 12. Baselines

| Baseline              | Description                     |
|-----------------------|---------------------------------|
| TG0h percentile rule  | 75th percentile threshold       |
| TG0h-only logistic    | Nested CV univariate LR         |

Used in performance tables and DCA.

------------------------------------------------------------------------

# 13. Pseudo-External Transport

Demographic transport simulations:

## Age

-   Train \<55 → Test ≥55
-   Train ≥55 → Test \<55

## Sex

-   Train Male → Test Female
-   Train Female → Test Male

Training-derived TG4h cutoff applied to test group.

Results saved to:\
pseudoexternaltransfer.csv

------------------------------------------------------------------------

# 14. Decision Curve Analysis

Computed from nested-CV OOF predictions.

Models included:

-   Multivariable LR
-   TG0h logistic
-   TG0h percentile rule
-   Treat-all
-   Treat-none

Probability grid: 0.01--0.99

Outputs:

-   decisioncurve.csv
-   figdecisioncurve.png

------------------------------------------------------------------------

# 15. Outputs

Generated artifacts include:

-   nestedcvsummary.json
-   oofpredictions.csv
-   thresholdsensitivitysummary.csv
-   Bootstrap summaries (CSV + JSON)
-   SHAP/PDP plots
-   DCA tables
-   Pseudo-external metrics
-   WBV sensitivity results
-   External stress test curves
-   summarytablerows.tex
-   summarytablevalues.json

All tables are manuscript-ready.

------------------------------------------------------------------------

# 16. Reproducibility Guarantees

-   Fixed RANDOM_STATE = 42
-   Seeded NumPy operations
-   All preprocessing inside pipelines
-   Training-fold-only threshold computation
-   Explicit logging of TG4h and TG0h cutoffs
-   Deterministic bootstrap
-   Transparent WBV computation
-   Script-level separation for auditability

From raw CSV to LaTeX tables, the workflow is fully repeatable.

------------------------------------------------------------------------

# 17. Citation

If using this repository:
```text
Piyavechvirat, N.
ML_Predict: A Calibrated Multi-Threshold Logistic Regression Pipeline
for Postprandial Lipid Response Analysis.
GitHub Repository, 2025.
```
------------------------------------------------------------------------

# 18. Contact

Author: Nattakitti Piyavechvirat\
GitHub: https://github.com/NattakittiP\
Email: Ohm19nattakitti@gmail.com

For collaboration, issues, or methodological discussion, please open an
Issue or Pull Request.
