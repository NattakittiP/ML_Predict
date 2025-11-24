# ============================================================
# STEP 0: Install dependencies (run once at top of Colab)
# ============================================================
!pip install xgboost lightgbm catboost shap --quiet
# ============================================================
# STEP 1: Imports & Global Config
# ============================================================
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, accuracy_score, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import shap
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
# ============================================================
# STEP 2: Load Dataset
#  - Upload your CSV to Colab (left sidebar -> Files -> upload)
# ============================================================

DATA_PATH = "/content/WBV_TCR_ICBCB2026_Synthetic_1500_precise_v2 (2).csv"

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.head()
# ============================================================
# STEP 3: Basic Cleaning & Column Definition
# ============================================================

df = df.dropna().reset_index(drop=True)

FEATURE_COLS = ["Age","Sex","Hematocrit","TotalProtein","WBV","TG0h","HDL","LDL","BMI"]

TARGET_TG4H = "TG4h"

NUMERIC_FEATURES = [
    "Age", "Hematocrit", "TotalProtein", "WBV",
    "TG0h", "HDL", "LDL", "BMI"
]
CATEGORICAL_FEATURES = ["Sex"]

missing_cols = [c for c in FEATURE_COLS + [TARGET_TG4H] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Columns not found in dataframe: {missing_cols}")
# ============================================================
# STEP 4: Preprocessor & Model Zoo Definitions
# ============================================================

def build_preprocessor():
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_models():
    models = {}

    # Logistic Regression L2
    models["logreg_l2"] = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    # Logistic Regression ElasticNet
    models["logreg_elastic"] = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        solver="saga",
        max_iter=4000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    # Random Forest
    models["rf"] = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    # Gradient Boosting
    models["gb"] = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    # XGBoost
    models["xgb"] = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.03,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    # LightGBM
    models["lgbm"] = LGBMClassifier(
        n_estimators=600,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.03,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # CatBoost (silent)
    models["catboost"] = CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.03,
        loss_function="Logloss",
        verbose=False,
        random_state=RANDOM_STATE
    )

    # SVM RBF
    models["svm_rbf"] = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    return models


def cross_val_eval(model, X, y, preprocessor, n_splits=5):
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )

    metrics = {
        "auroc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "brier": [],
        "acc": [],
    }

    pipe = Pipeline(
        steps=[("preprocess", preprocessor), ("model", model)]
    )

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics["auroc"].append(roc_auc_score(y_val, prob))
        metrics["f1"].append(f1_score(y_val, pred))
        metrics["precision"].append(precision_score(y_val, pred))
        metrics["recall"].append(recall_score(y_val, pred))
        metrics["brier"].append(brier_score_loss(y_val, prob))
        metrics["acc"].append(accuracy_score(y_val, pred))

    summary = {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
    return summary
# ============================================================
# STEP 5: Multi-threshold Model Zoo (60 → 90 by 5)
# ============================================================

threshold_list = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

all_results = []
preprocessor = build_preprocessor()
models = get_models()

X_full = df[FEATURE_COLS].copy()

for th in threshold_list:
    percentile = th * 100
    cutoff_value = np.percentile(df[TARGET_TG4H].values, percentile)

    y = (df[TARGET_TG4H] >= cutoff_value).astype(int)

    print(f"Running threshold {th:.2f} (TG4h ≥ {cutoff_value:.2f} mg/dL)")
    
    for model_name, model in models.items():
        res = cross_val_eval(model, X_full, y, preprocessor, n_splits=5)
        row = {
            "threshold": th,
            "percentile": percentile,
            "cutoff_TG4h_mgdl": cutoff_value,
            "model": model_name,
            "auroc_mean": res["auroc"][0],
            "auroc_std": res["auroc"][1],
            "brier_mean": res["brier"][0],
            "brier_std": res["brier"][1],
            "f1_mean": res["f1"][0],
            "recall_mean": res["recall"][0],
            "precision_mean": res["precision"][0],
            "acc_mean": res["acc"][0],
        }
        all_results.append(row)

cv_results_all_thresholds = pd.DataFrame(all_results)
cv_results_all_thresholds
# ============================================================
# STEP 6: ดูผลรวม + เลือก "Main Threshold & Main Model"
#  - ตัวอย่าง: เลือก threshold = 0.75, และโมเดลที่ AUROC สูงสุด
# ============================================================

summary_pivot = cv_results_all_thresholds.pivot_table(
    index=["threshold", "model"],
    values="auroc_mean"
).sort_values(by=["threshold", "auroc_mean"], ascending=[True, False])

summary_pivot

MAIN_THRESHOLD = 0.75

subset_75 = cv_results_all_thresholds[
    cv_results_all_thresholds["threshold"] == MAIN_THRESHOLD
].sort_values("auroc_mean", ascending=False)

subset_75
best_row = subset_75.iloc[0]
MAIN_MODEL_NAME = best_row["model"]

print("Main threshold:", MAIN_THRESHOLD)
print("Main model   :", MAIN_MODEL_NAME)
print("AUROC (CV)   :", best_row["auroc_mean"])
print("Cutoff TG4h  :", best_row["cutoff_TG4h_mgdl"])
# ============================================================
# STEP 7: Train/Test Split + Calibration (Isotonic & Sigmoid)
# ============================================================

main_percentile = MAIN_THRESHOLD * 100
main_cutoff = np.percentile(df[TARGET_TG4H].values, main_percentile)
y_main = (df[TARGET_TG4H] >= main_cutoff).astype(int)

X = X_full.copy()
y = y_main.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE
)

preprocessor_main = build_preprocessor()
models = get_models()
base_model = models[MAIN_MODEL_NAME]

base_pipe = Pipeline(
    steps=[("preprocess", preprocessor_main),
           ("model", base_model)]
)

base_pipe.fit(X_train, y_train)

def evaluate_probs(probs, y_true):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    return auc, brier, fpr, tpr

probs_base = base_pipe.predict_proba(X_test)[:, 1]
auc_base, brier_base, fpr_base, tpr_base = evaluate_probs(probs_base, y_test)

print("=== Uncalibrated ===")
print("AUROC:", auc_base)
print("Brier:", brier_base)

methods = ["isotonic", "sigmoid"]
calibrated_models = {}
calib_metrics = []

for m in methods:
    calib = CalibratedClassifierCV(
        estimator=base_pipe,
        method=m,
        cv="prefit"
    )
    calib.fit(X_test, y_test)

    prob_calib = calib.predict_proba(X_test)[:, 1]
    auc_c, brier_c, fpr_c, tpr_c = evaluate_probs(prob_calib, y_test)

    calibrated_models[m] = {
        "model": calib,
        "auc": auc_c,
        "brier": brier_c,
        "fpr": fpr_c,
        "tpr": tpr_c
    }
    calib_metrics.append((m, auc_c, brier_c))

print("\n=== Calibrated ===")
for m, auc_c, brier_c in calib_metrics:
    print(f"Method: {m:8s} | AUROC: {auc_c:.4f} | Brier: {brier_c:.4f}")

best_calib = min(calib_metrics, key=lambda x: x[2])[0]
FINAL_MODEL = calibrated_models[best_calib]["model"]
print("\n>>> FINAL MODEL uses calibration:", best_calib)
# ============================================================
# STEP 8: Plot ROC & Calibration Curves สำหรับ FINAL MODEL
# ============================================================

plt.figure(figsize=(6,5))
plt.plot([0,1], [0,1], linestyle="--", label="Chance")
for m in ["isotonic", "sigmoid"]:
    if m in calibrated_models:
        plt.plot(
            calibrated_models[m]["fpr"],
            calibrated_models[m]["tpr"],
            label=f"{m} (AUROC={calibrated_models[m]['auc']:.3f})"
        )
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({MAIN_MODEL_NAME}, threshold={MAIN_THRESHOLD})")
plt.legend()
plt.grid(True)
plt.show()

probs_final = FINAL_MODEL.predict_proba(X_test)[:, 1]
prob_true, prob_pred = calibration_curve(y_test, probs_final, n_bins=10)

plt.figure(figsize=(6,5))
plt.plot([0,1],[0,1],"--", label="Perfect calibration")
plt.plot(prob_pred, prob_true, marker="o", linestyle="-", label="Final model")
plt.xlabel("Predicted probability")
plt.ylabel("True fraction of positives")
plt.title("Calibration Curve (Final model)")
plt.legend()
plt.grid(True)
plt.show()
# ============================================================
# STEP 9: SHAP Explainability (KernelExplainer over FINAL_MODEL)
# ============================================================

X_train_shap = X_train.copy()
if len(X_train_shap) > 500:
    X_train_shap = X_train_shap.sample(300, random_state=RANDOM_STATE)

def model_predict_proba(data_as_array):
    df_temp = pd.DataFrame(data_as_array, columns=FEATURE_COLS)
    return FINAL_MODEL.predict_proba(df_temp)[:, 1]

explainer = shap.KernelExplainer(
    model_predict_proba,
    shap.sample(X_train_shap, 100, random_state=RANDOM_STATE)
)

X_shap_eval = shap.sample(X_train_shap, 200, random_state=RANDOM_STATE)
shap_values = explainer.shap_values(X_shap_eval)

shap.initjs()
shap.summary_plot(
    shap_values,
    X_shap_eval,
    feature_names=FEATURE_COLS
)

if "TG0h" in FEATURE_COLS:
    shap.dependence_plot(
        "TG0h",
        shap_values,
        X_shap_eval,
        feature_names=FEATURE_COLS
    )
else:
    print("Column 'TG0h' not found in FEATURE_COLS, please adjust name.")
# ============================================================
# STEP 10: Subgroup Analysis (Sex & Age)
# ============================================================

def subgroup_auc(model, X_test, y_test, mask, desc=""):
    X_sub = X_test[mask]
    y_sub = y_test[mask]
    if len(y_sub.unique()) < 2:
        return np.nan
    prob_sub = model.predict_proba(X_sub)[:, 1]
    auc = roc_auc_score(y_sub, prob_sub)
    print(f"{desc}: n={len(y_sub)}, AUROC={auc:.3f}")
    return auc

sex_series = X_test["Sex"]

male_mask = sex_series == "Male"
female_mask = sex_series == "Female"

print("=== Sex Subgroups ===")
auc_male = subgroup_auc(FINAL_MODEL, X_test, y_test, male_mask, "Male")
auc_female = subgroup_auc(FINAL_MODEL, X_test, y_test, female_mask, "Female")

age_series = X_test["Age"]
age_mask_low = age_series < 55
age_mask_high = age_series >= 55

print("\n=== Age Subgroups ===")
auc_age_low = subgroup_auc(FINAL_MODEL, X_test, y_test, age_mask_low, "Age < 55")
auc_age_high = subgroup_auc(FINAL_MODEL, X_test, y_test, age_mask_high, "Age ≥ 55")
# ============================================================
# STEP 11: Save key results to CSV (สำหรับเอาไปเขียน paper)
# ============================================================

cv_results_all_thresholds.to_csv("cv_results_all_thresholds.csv", index=False)
summary_pivot.to_csv("cv_auroc_summary_pivot.csv")

print("Saved:")
print("- cv_results_all_thresholds.csv")
print("- cv_auroc_summary_pivot.csv")
# ============================================================
# STEP 12: Bootstrap 95% CI for AUROC and Brier (Final Model)
# ============================================================
from sklearn.utils import resample

n_bootstrap = 1000
rng = np.random.RandomState(RANDOM_STATE)

aucs = []
briers = []

probs_full = FINAL_MODEL.predict_proba(X_test)[:, 1]

for i in range(n_bootstrap):
    indices = rng.randint(0, len(y_test), len(y_test))
    y_boot = y_test.iloc[indices]
    prob_boot = probs_full[indices]

    try:
        auc = roc_auc_score(y_boot, prob_boot)
        brier = brier_score_loss(y_boot, prob_boot)
        aucs.append(auc)
        briers.append(brier)
    except ValueError:
        continue

def ci_percentile(values, alpha=0.95):
    lower = np.percentile(values, (1-alpha)/2*100)
    upper = np.percentile(values, (1 + alpha)/2*100)
    return lower, upper

auc_mean = np.mean(aucs)
auc_ci = ci_percentile(aucs, 0.95)

brier_mean = np.mean(briers)
brier_ci = ci_percentile(briers, 0.95)

print("=== Bootstrap 95% CI (Final Calibrated Model) ===")
print(f"AUROC mean = {auc_mean:.3f}, 95% CI = [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
print(f"Brier mean = {brier_mean:.3f}, 95% CI = [{brier_ci[0]:.3f}, {brier_ci[1]:.3f}]")
print(f"n_bootstrap used = {len(aucs)}")
# ============================================================
# STEP 13: Baseline Models using TG0h only
#   (1) Simple percentile cut-off
#   (2) Univariate logistic regression
# ============================================================

from sklearn.pipeline import Pipeline

X_train_tg = X_train[["TG0h"]].copy()
X_test_tg  = X_test[["TG0h"]].copy()

tg_cutoff = np.percentile(X_train_tg["TG0h"].values, main_percentile)

y_pred_rule = (X_test_tg["TG0h"] >= tg_cutoff).astype(int).values

prob_rule = np.where(X_test_tg["TG0h"] >= tg_cutoff, 0.75, 0.25)

auc_rule  = roc_auc_score(y_test, prob_rule)
brier_rule = brier_score_loss(y_test, prob_rule)

print("=== Baseline 1: Simple TG0h cutoff ===")
print(f"TG0h cutoff (train 75th percentile) = {tg_cutoff:.2f} mg/dL")
print(f"AUROC  = {auc_rule:.3f}")
print(f"Brier  = {brier_rule:.3f}")
print()

baseline_lr = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    ))
])

baseline_lr.fit(X_train_tg, y_train)

prob_lr = baseline_lr.predict_proba(X_test_tg)[:, 1]
pred_lr = (prob_lr >= 0.5).astype(int)

auc_lr   = roc_auc_score(y_test, prob_lr)
brier_lr = brier_score_loss(y_test, prob_lr)
f1_lr    = f1_score(y_test, pred_lr)
recall_lr = recall_score(y_test, pred_lr)
prec_lr   = precision_score(y_test, pred_lr)

print("=== Baseline 2: Univariate Logistic Regression (TG0h only) ===")
print(f"AUROC   = {auc_lr:.3f}")
print(f"Brier   = {brier_lr:.3f}")
print(f"F1      = {f1_lr:.3f}")
print(f"Recall  = {recall_lr:.3f}")
print(f"Precision = {prec_lr:.3f}")
# ============================================================
# STEP 14: Repeated Stratified CV (5x10) for Robustness
# ============================================================
from sklearn.model_selection import RepeatedStratifiedKFold

X_full = df[FEATURE_COLS].copy()

main_percentile = MAIN_THRESHOLD * 100
main_cutoff = np.percentile(df[TARGET_TG4H].values, main_percentile)
y_main = (df[TARGET_TG4H] >= main_cutoff).astype(int)

preprocessor_rep = build_preprocessor()
models_all = get_models()
main_model = models_all[MAIN_MODEL_NAME]

pipe_main = Pipeline(steps=[
    ("preprocess", preprocessor_rep),
    ("model", main_model)
])

rskf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=RANDOM_STATE
)

aucs_rep = []
briers_rep = []

for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_full, y_main), start=1):
    X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_tr, y_val = y_main.iloc[train_idx], y_main.iloc[val_idx]

    pipe_main.fit(X_tr, y_tr)
    prob_val = pipe_main.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, prob_val)
    brier = brier_score_loss(y_val, prob_val)

    aucs_rep.append(auc)
    briers_rep.append(brier)

print("=== Repeated Stratified CV (5-fold x 10 repeats) ===")
print(f"Total folds: {len(aucs_rep)}")
print(f"AUROC: mean={np.mean(aucs_rep):.3f}, std={np.std(aucs_rep):.3f}, "
      f"min={np.min(aucs_rep):.3f}, max={np.max(aucs_rep):.3f}")
print(f"Brier: mean={np.mean(briers_rep):.3f}, std={np.std(briers_rep):.3f}, "
      f"min={np.min(briers_rep):.3f}, max={np.max(briers_rep):.3f}")

plt.figure(figsize=(6,4))
plt.hist(aucs_rep, bins=10, alpha=0.8)
plt.xlabel("AUROC")
plt.ylabel("Count")
plt.title("Distribution of AUROC across 5x10 repeated CV")
plt.grid(True)
plt.show()
# ============================================================
# STEP 15: Decision Curve Analysis (DCA)
#   Models:
#     - Final calibrated model (multivariable)
#     - TG0h-only logistic regression (baseline_lr)
#     - TG0h cutoff rule-based classifier
#     - Treat-all / Treat-none
# ============================================================

from sklearn.metrics import roc_auc_score

def net_benefit(y_true, prob, threshold):
    """
    y_true: 0/1 array
    prob: predicted probability
    threshold: pt in (0,1)
    """
    pred = (prob >= threshold).astype(int)
    TP = ((pred == 1) & (y_true == 1)).sum()
    FP = ((pred == 1) & (y_true == 0)).sum()
    n = len(y_true)
    if n == 0:
        return np.nan
    return (TP/n) - (FP/n) * (threshold / (1 - threshold))

probs_final = FINAL_MODEL.predict_proba(X_test)[:, 1]

probs_tg_lr = baseline_lr.predict_proba(X_test_tg)[:, 1]

thresholds = np.linspace(0.05, 0.80, 16)

nb_final = []
nb_tg_lr = []
nb_tg_rule = []
nb_treat_all = []

y_true = y_test.values

for pt in thresholds:
    nb_final.append(net_benefit(y_true, probs_final, pt))
    nb_tg_lr.append(net_benefit(y_true, probs_tg_lr, pt))
    nb_tg_rule.append(net_benefit(y_true, prob_rule, pt))
    prevalence = y_true.mean()
    nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
    nb_treat_all.append(nb_all)

plt.figure(figsize=(7,5))
plt.plot(thresholds, nb_final, marker="o", label="Multivariable model")
plt.plot(thresholds, nb_tg_lr, marker="s", linestyle="--", label="TG0h logistic")
plt.plot(thresholds, nb_tg_rule, marker="^", linestyle="--", label="TG0h cutoff rule")
plt.plot(thresholds, nb_treat_all, linestyle="-.", label="Treat-all")
plt.axhline(0.0, color="grey", linestyle=":", label="Treat-none")

plt.xlabel("Threshold probability")
plt.ylabel("Net benefit")
plt.title("Decision Curve Analysis")
plt.legend()
plt.grid(True)
plt.show()
# ============================================================
# STEP 16: Precision–Recall Curve & PR-AUC (Final Model)
# ============================================================
from sklearn.metrics import precision_recall_curve, average_precision_score

probs_final = FINAL_MODEL.predict_proba(X_test)[:, 1]
precision, recall, pr_thresholds = precision_recall_curve(y_test, probs_final)
pr_auc = average_precision_score(y_test, probs_final)

print(f"Average Precision (PR-AUC) of final model = {pr_auc:.3f}")

plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"Final model (AP={pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Final model)")
plt.grid(True)
plt.legend()
plt.show()
# ============================================================
# STEP 17: Pseudo-external validation
#   (A) Train on Age < 55, Test on Age ≥ 55
#   (B) Train on Male, Test on Female
# ============================================================

from sklearn.metrics import roc_auc_score, brier_score_loss

main_percentile = MAIN_THRESHOLD * 100
main_cutoff = np.percentile(df[TARGET_TG4H].values, main_percentile)
y_all = (df[TARGET_TG4H] >= main_cutoff).astype(int)

X_all = df[FEATURE_COLS].copy()

age_all = df["Age"]
mask_young = age_all < 55
mask_old   = age_all >= 55

def train_eval_pseudo_external(train_mask, test_mask, desc=""):
    X_train_pe = X_all[train_mask]
    y_train_pe = y_all[train_mask]
    X_test_pe  = X_all[test_mask]
    y_test_pe  = y_all[test_mask]

    if len(np.unique(y_test_pe)) < 2 or len(y_test_pe) < 20:
        print(f"[{desc}] Too few samples or only one class in test set. Skipping.")
        return None

    pre = build_preprocessor()
    models_all = get_models()
    base_model = models_all[MAIN_MODEL_NAME]

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", base_model)
    ])

    pipe.fit(X_train_pe, y_train_pe)
    prob_pe = pipe.predict_proba(X_test_pe)[:, 1]

    auc_pe = roc_auc_score(y_test_pe, prob_pe)
    brier_pe = brier_score_loss(y_test_pe, prob_pe)

    print(f"[{desc}] n_train={len(y_train_pe)}, n_test={len(y_test_pe)}")
    print(f"  AUROC = {auc_pe:.3f}")
    print(f"  Brier = {brier_pe:.3f}")
    print()
    return auc_pe, brier_pe

print("=== Pseudo-external validation: Age-based ===")
train_eval_pseudo_external(mask_young, mask_old, desc="Train <55, Test ≥55")
train_eval_pseudo_external(mask_old, mask_young, desc="Train ≥55, Test <55")

sex_all = df["Sex"].astype(str)
mask_male = sex_all.str.lower().str.contains("male")
mask_female = sex_all.str.lower().str.contains("female")

print("=== Pseudo-external validation: Sex-based ===")
train_eval_pseudo_external(mask_male, mask_female, desc="Train Male, Test Female")
train_eval_pseudo_external(mask_female, mask_male, desc="Train Female, Test Male")

# ===== Block 1: Calibration Slope & Intercept Plot =====
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

probs_final = FINAL_MODEL.predict_proba(X_test)[:, 1]
y_true = y_test.values

eps = 1e-6
probs_clipped = np.clip(probs_final, eps, 1 - eps)
logit_p = np.log(probs_clipped / (1 - probs_clipped)).reshape(-1, 1)

calib_lr = LogisticRegression(solver="lbfgs")
calib_lr.fit(logit_p, y_true)

slope = calib_lr.coef_[0][0]
intercept = calib_lr.intercept_[0]

print(f"Calibration slope  = {slope:.3f}")
print(f"Calibration intercept = {intercept:.3f}")

logit_range = np.linspace(logit_p.min(), logit_p.max(), 100).reshape(-1, 1)
p_perfect = 1 / (1 + np.exp(-logit_range))
logit_calib = intercept + slope * logit_range
p_calib = 1 / (1 + np.exp(-logit_calib))

plt.figure(figsize=(6,5))
plt.plot(p_perfect, p_perfect, linestyle="--", label="Perfect calibration")
plt.plot(p_perfect, p_calib, label=f"Fitted (slope={slope:.2f}, int={intercept:.2f})")
plt.xlabel("Predicted probability")
plt.ylabel("Observed probability (fitted)")
plt.title("Calibration Slope & Intercept")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure_S8_CalibrationSlopeIntercept.png", dpi=300)
plt.show()
# ===== Block 2: SHAP Association Heatmap (Feature x Feature) =====
import numpy as np
import matplotlib.pyplot as plt

shap_array = np.array(shap_values)

corr_mat = np.corrcoef(shap_array, rowvar=False)

plt.figure(figsize=(7,6))
im = plt.imshow(corr_mat, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(ticks=np.arange(len(FEATURE_COLS)), labels=FEATURE_COLS, rotation=90)
plt.yticks(ticks=np.arange(len(FEATURE_COLS)), labels=FEATURE_COLS)

plt.title("SHAP Value Correlation Between Features")
plt.tight_layout()
plt.savefig("Figure_S9_SHAP_FeatureCorrelationHeatmap.png", dpi=300)
plt.show()
# ===== Block 3: SHAP Interaction Plot TG0h × WBV =====
import shap

print("Features:", FEATURE_COLS)

if "TG0h" in FEATURE_COLS and "WBV" in FEATURE_COLS:
    shap.dependence_plot(
        "TG0h",
        shap_values,
        X_shap_eval,
        feature_names=FEATURE_COLS,
        interaction_index="WBV"
    )
else:
    print("โปรดตรวจสอบว่า 'TG0h' และ 'WBV' อยู่ใน FEATURE_COLS หรือไม่")

# ===== Block 4: AUROC vs Threshold Sensitivity =====
import matplotlib.pyplot as plt
import numpy as np

df_main = cv_results_all_thresholds[
    cv_results_all_thresholds["model"] == MAIN_MODEL_NAME
].copy()

df_main = df_main.sort_values("threshold")

thresholds = df_main["threshold"].values
auroc_mean = df_main["auroc_mean"].values
auroc_std  = df_main["auroc_std"].values

plt.figure(figsize=(6,5))
plt.errorbar(thresholds, auroc_mean, yerr=auroc_std, fmt="-o", capsize=4)
plt.xlabel("TG4h percentile threshold")
plt.ylabel("AUROC (mean ± SD)")
plt.title(f"Threshold Sensitivity of {MAIN_MODEL_NAME}")
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure_S11_AUROC_vs_Threshold.png", dpi=300)
plt.show()
# ===== Block 5: Bootstrap AUROC & Brier Histograms =====
import matplotlib.pyplot as plt
import numpy as np

print(f"Bootstrap samples AUROC: n={len(aucs)}, mean={np.mean(aucs):.3f}")
print(f"Bootstrap samples Brier: n={len(briers)}, mean={np.mean(briers):.3f}")

plt.figure(figsize=(6,4))
plt.hist(aucs, bins=20)
plt.xlabel("AUROC")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of AUROC (Final Model)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure_S12_Bootstrap_AUROC_Distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(6,4))
plt.hist(briers, bins=20)
plt.xlabel("Brier score")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of Brier Score (Final Model)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure_S13_Bootstrap_Brier_Distribution.png", dpi=300)
plt.show()

# ===== Block 6: Correlation Matrix of Fasting Biomarkers =====
import numpy as np
import matplotlib.pyplot as plt

num_cols = [c for c in FEATURE_COLS if c != "Sex"]
corr = df[num_cols].corr()

plt.figure(figsize=(7,6))
im = plt.imshow(corr.values, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(ticks=np.arange(len(num_cols)), labels=num_cols, rotation=90)
plt.yticks(ticks=np.arange(len(num_cols)), labels=num_cols)

plt.title("Correlation Matrix of Fasting Biomarkers")
plt.tight_layout()
plt.savefig("Figure_S14_Biomarker_CorrelationMatrix.png", dpi=300)
plt.show()
# ===== Block 7: TG0h Distribution (Normal vs High Responders) =====
import numpy as np
import matplotlib.pyplot as plt

main_percentile = MAIN_THRESHOLD * 100
main_cutoff = np.percentile(df[TARGET_TG4H].values, main_percentile)
y_main = (df[TARGET_TG4H] >= main_cutoff).astype(int)

tg0_normal = df.loc[y_main == 0, "TG0h"].values
tg0_high   = df.loc[y_main == 1, "TG0h"].values

plt.figure(figsize=(6,5))
plt.hist(tg0_normal, bins=30, density=True, alpha=0.6, label="Normal responders")
plt.hist(tg0_high, bins=30, density=True, alpha=0.6, label="High responders")

plt.xlabel("Fasting TG (TG0h)")
plt.ylabel("Density")
plt.title("TG0h Distribution by Postprandial Response Phenotype")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure_S15_TG0h_Distribution_Normal_vs_High.png", dpi=300)
plt.show()
# ===== Block 8: TG4h Distribution (Normal vs High Responders) =====
import numpy as np
import matplotlib.pyplot as plt

tg4_normal = df.loc[y_main == 0, TARGET_TG4H].values
tg4_high   = df.loc[y_main == 1, TARGET_TG4H].values

plt.figure(figsize=(6,5))
plt.hist(tg4_normal, bins=30, density=True, alpha=0.6, label="Normal responders")
plt.hist(tg4_high, bins=30, density=True, alpha=0.6, label="High responders")

plt.xlabel("Postprandial TG (TG4h)")
plt.ylabel("Density")
plt.title("TG4h Distribution by Phenotype")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure_S16_TG4h_Distribution_Normal_vs_High.png", dpi=300)
plt.show()

# ===== PDP Block 1: Prepare X_full / y_main (run once) =====
from copy import deepcopy
import numpy as np

X_full = df[FEATURE_COLS].copy()

main_percentile = MAIN_THRESHOLD * 100
main_cutoff = np.percentile(df[TARGET_TG4H].values, main_percentile)
y_main = (df[TARGET_TG4H] >= main_cutoff).astype(int)

print("X_full shape:", X_full.shape)
print("Positive rate:", y_main.mean())

# ===== PDP Block 2: 1D Partial Dependence (TG0h, WBV, BMI) =====
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

for feat in ["TG0h", "WBV", "BMI"]:
    if feat not in FEATURE_COLS:
        print(f"WARNING: '{feat}' ไม่อยู่ใน FEATURE_COLS, โปรดเช็คชื่อคอลัมน์")
        
features_1d = []
for feat in ["TG0h", "WBV", "BMI"]:
    if feat in FEATURE_COLS:
        features_1d.append(feat)

if len(features_1d) == 0:
    print("ไม่มีฟีเจอร์ที่ใช้ทำ PDP, โปรดตรวจ FEATURE_COLS")
else:
    fig, ax = plt.subplots(figsize=(7,5))
    disp = PartialDependenceDisplay.from_estimator(
        FINAL_MODEL,
        X_full,
        features=features_1d,
        kind="average",
        ax=ax
    )
    ax.set_title("1D Partial Dependence (TG0h, WBV, BMI)")
    plt.tight_layout()
    plt.savefig("Figure_S_PDP_1D_TG0h_WBV_BMI.png", dpi=300)
    plt.show()
    
# ===== PDP Block 3: 2D Partial Dependence TG0h × BMI =====
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

if ("TG0h" not in FEATURE_COLS) or ("BMI" not in FEATURE_COLS):
    print("Checked 'TG0h' and 'BMI' are in FEATURE_COLS or not?")
else:
   
    idx_tg0 = FEATURE_COLS.index("TG0h")
    idx_bmi = FEATURE_COLS.index("BMI")
    
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    
    disp2d = PartialDependenceDisplay.from_estimator(
        FINAL_MODEL,
        X_full,
        features=[(idx_tg0, idx_bmi)],
        kind="average",
        ax=ax
    )
    
    ax.set_title("2D Partial Dependence: TG0h × BMI")
    plt.tight_layout()
    plt.savefig("Figure_S_PDP_2D_TG0h_BMI.png", dpi=300)
    plt.show()
    
# ===== PDP Block 4 (optional): High-resolution PDP for TG0h =====
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

if "TG0h" not in FEATURE_COLS:
    print("'TG0h' ไม่อยู่ใน FEATURE_COLS")
else:
    fig, ax = plt.subplots(figsize=(6,4))
    disp_tg = PartialDependenceDisplay.from_estimator(
        FINAL_MODEL,
        X_full,
        features=["TG0h"],
        kind="average",
        grid_resolution=10000,
        ax=ax
    )
    ax.set_title("High-resolution Partial Dependence of TG0h")
    plt.tight_layout()
    plt.savefig("Figure_S_PDP_TG0h_HighRes.png", dpi=300)
    plt.show()

