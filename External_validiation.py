"""
External validation of leakage-controlled ML model
using original synthetic cohort (train) and an
independent external synthetic cohort (test).

Steps:
1. Load original synthetic cohort
2. Create high-responder label from TG4h 75th percentile
3. Train a standardized logistic regression model
4. Evaluate internal performance with stratified CV
5. Fit final model on full training cohort
6. Load external synthetic cohort
7. Create external labels in the same way (TG4h 75th percentile)
8. Evaluate external AUROC, Brier score, and other metrics
9. Plot ROC curve and calibration curve for external cohort
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)
from sklearn.calibration import calibration_curve
from joblib import dump

# =========================
# CONFIG
# =========================

TRAIN_PATH = "WBV_TCR_ICBCB2026_Synthetic_1500_precise_v2 (2).csv"
EXTERNAL_PATH = "external_synthetic_like.csv"
MODEL_OUTPUT_PATH = "midm_elastic_net_model.joblib"

FEATURE_COLS = [
    "TG0h",
    "TG4h",
    "Hematocrit",
    "TotalProtein",
    "HDL",
    "LDL",
    "BMI",
    "Age",
    "Sex_encoded",  # จะ encode จาก Sex ภายหลัง
]


# =========================
# HELPER FUNCTIONS
# =========================

def load_and_prepare_train(path: str) -> pd.DataFrame:
    """
    Load the original synthetic cohort and
    compute the high-responder label.
    """
    df = pd.read_csv(path)

    required_cols = [
        "Sex", "Age", "Hematocrit", "TotalProtein",
        "TG0h", "TG4h", "HDL", "LDL", "BMI"
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Training dataset missing columns: {missing}")

    # Encode Sex to binary
    df["Sex_encoded"] = df["Sex"].map({"Female": 1, "Male": 0}).astype(int)

    # Define high responder: TG4h >= 75th percentile of training cohort
    tg4h_75 = np.percentile(df["TG4h"], 75)
    df["HighResponder"] = (df["TG4h"] >= tg4h_75).astype(int)

    return df


def load_and_prepare_external(path: str) -> pd.DataFrame:
    """
    Load the external synthetic cohort and
    compute the high-responder label in the same way
    (based on its own TG4h 75th percentile).
    """
    df = pd.read_csv(path)

    required_cols = [
        "Sex", "Age", "Hematocrit", "TotalProtein",
        "TG0h", "TG4h", "HDL", "LDL", "BMI"
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"External dataset missing columns: {missing}")

    # Encode Sex
    df["Sex_encoded"] = df["Sex"].map({"Female": 1, "Male": 0}).astype(int)

    # Define high responder for external cohort
    tg4h_75_ext = np.percentile(df["TG4h"], 75)
    df["HighResponder"] = (df["TG4h"] >= tg4h_75_ext).astype(int)

    return df


def build_pipeline() -> Pipeline:
    """
    Build an elastic-net style logistic regression pipeline with scaling.
    Using 'saga' solver supports elastic-net penalty.
    """
    log_reg = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,         # mix L1/L2 (ปรับได้)
        C=1.0,
        solver="saga",
        max_iter=500,
        n_jobs=-1
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", log_reg),
        ]
    )
    return pipe


def evaluate_internal_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series):
    """
    Evaluate internal performance using stratified 5-fold CV
    with out-of-fold predicted probabilities.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # cross_val_predict with method='predict_proba' ให้ out-of-fold proba
    y_proba_oof = cross_val_predict(
        pipe,
        X,
        y,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    y_pred_oof = (y_proba_oof >= 0.5).astype(int)

    auc = roc_auc_score(y, y_proba_oof)
    brier = brier_score_loss(y, y_proba_oof)
    acc = accuracy_score(y, y_pred_oof)
    prec = precision_score(y, y_pred_oof)
    rec = recall_score(y, y_pred_oof)
    f1 = f1_score(y, y_pred_oof)

    print("=== INTERNAL (Cross-Validation on Training Synthetic) ===")
    print(f"AUROC      : {auc:.3f}")
    print(f"Brier score: {brier:.3f}")
    print(f"Accuracy   : {acc:.3f}")
    print(f"Precision  : {prec:.3f}")
    print(f"Recall     : {rec:.3f}")
    print(f"F1-score   : {f1:.3f}")

    return y_proba_oof


def fit_final_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Fit final model on full training dataset.
    """
    pipe.fit(X, y)
    return pipe


def evaluate_external(pipe: Pipeline, X_ext: pd.DataFrame, y_ext: pd.Series):
    """
    Evaluate external performance on independent synthetic cohort.
    """
    y_proba = pipe.predict_proba(X_ext)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_ext, y_proba)
    brier = brier_score_loss(y_ext, y_proba)
    acc = accuracy_score(y_ext, y_pred)
    prec = precision_score(y_ext, y_pred)
    rec = recall_score(y_ext, y_pred)
    f1 = f1_score(y_ext, y_pred)

    print("\n=== EXTERNAL (Independent Synthetic Cohort) ===")
    print(f"AUROC      : {auc:.3f}")
    print(f"Brier score: {brier:.3f}")
    print(f"Accuracy   : {acc:.3f}")
    print(f"Precision  : {prec:.3f}")
    print(f"Recall     : {rec:.3f}")
    print(f"F1-score   : {f1:.3f}")

    return y_proba


def plot_external_curves(y_true, y_proba, title_prefix="External synthetic"):
    """
    Plot ROC and calibration curve for external cohort.
    """
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label="Model ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("external_roc_curve.png", dpi=300)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Model calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(f"{title_prefix} Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("external_calibration_curve.png", dpi=300)


# =========================
# MAIN
# =========================

def main():
    # 1) Load and prepare training cohort
    print("Loading training synthetic cohort...")
    train_df = load_and_prepare_train(TRAIN_PATH)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["HighResponder"]

    # 2) Build pipeline
    pipe = build_pipeline()

    # 3) Internal CV evaluation
    _ = evaluate_internal_cv(pipe, X_train, y_train)

    # 4) Fit final model on full training data
    print("\nFitting final model on full training data...")
    pipe = fit_final_model(pipe, X_train, y_train)

    # 5) Save model
    dump(pipe, MODEL_OUTPUT_PATH)
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")

    # 6) Load and prepare external cohort
    print("\nLoading external synthetic cohort...")
    ext_df = load_and_prepare_external(EXTERNAL_PATH)

    X_ext = ext_df[FEATURE_COLS]
    y_ext = ext_df["HighResponder"]

    # 7) Evaluate external performance
    y_proba_ext = evaluate_external(pipe, X_ext, y_ext)

    # 8) Plot ROC and calibration curve for external cohort
    print("\nPlotting ROC and calibration curves for external cohort...")
    plot_external_curves(y_ext, y_proba_ext)

    print("\nDone. Check 'external_roc_curve.png' and 'external_calibration_curve.png'.")


if __name__ == "__main__":
    main()

#####################################################################################
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from joblib import dump

def build_uncalibrated_model():
    log_reg = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        C=1.0,
        solver="saga",
        max_iter=500,
        n_jobs=-1
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", log_reg),
    ])
    return pipe

def build_calibrated_model(X_train, y_train):
    base_model = build_uncalibrated_model()

    # Isotonic calibration with CV 
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        cv=5,
        method="isotonic"
    )

    calibrated.fit(X_train, y_train)
    return calibrated

def main():
    print("Loading synthetic training cohort...")
    train_df = load_and_prepare_train(TRAIN_PATH)
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["HighResponder"]

    # Build calibrated model
    print("Building calibrated model...")
    model = build_calibrated_model(X_train, y_train)

    # Save calibrated model
    dump(model, "midm_calibrated_model.joblib")

    print("Loading external synthetic cohort...")
    ext_df = load_and_prepare_external(EXTERNAL_PATH)
    X_ext = ext_df[FEATURE_COLS]
    y_ext = ext_df["HighResponder"]

    # Evaluate calibrated model
    print("\nEvaluating calibrated model on external cohort...")
    y_proba_ext = model.predict_proba(X_ext)[:, 1]

    auc = roc_auc_score(y_ext, y_proba_ext)
    brier = brier_score_loss(y_ext, y_proba_ext)

    print(f"External Calibrated AUROC: {auc:.3f}")
    print(f"External Calibrated Brier : {brier:.3f}")

    # Plot new ROC and calibration curve
    plot_external_curves(y_ext, y_proba_ext,
                         title_prefix="Calibrated External synthetic")

if __name__ == "__main__":
    main()
################################################################################

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    mean_squared_error,
)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt

# =========================
# PATH & FEATURE CONFIG
# =========================


TRAIN_PATH = "WBV_TCR_ICBCB2026_Synthetic_1500_precise_v2 (2).csv"
EXTERNAL_PATH = "external_synthetic_like.csv"

FEATURE_COLS = [
    "TG0h", "TG4h", "Hematocrit", "TotalProtein",
    "HDL", "LDL", "BMI", "Age", "Sex_encoded"
]

# =========================
# UTILITIES
# =========================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ---------- Soft threshold labels ----------

def soft_labeling(df, soft_temperature=30.0):
   
    p75 = np.percentile(df["TG4h"], 75)
    z = (df["TG4h"] - p75) / soft_temperature
    return sigmoid(z)

def add_labels_and_sex(df):
    
    df = df.copy()
    df["Sex_encoded"] = df["Sex"].map({"Female": 1, "Male": 0}).astype(int)

    p75 = np.percentile(df["TG4h"], 75)
    df["HighResponder"] = (df["TG4h"] >= p75).astype(int)
    df["SoftResponder"] = soft_labeling(df)

    return df

def load_train():
    df = pd.read_csv(TRAIN_PATH)
    return add_labels_and_sex(df)

def load_external():
    df = pd.read_csv(EXTERNAL_PATH)
    return add_labels_and_sex(df)

# ---------- Base classifier (uncalibrated) ----------

def build_base_model():
    
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            solver="saga",
            max_iter=500
        ))
    ])
    return base

# ---------- Temperature Scaling ----------

def fit_temperature(logits, y_true, T_min=0.5, T_max=3.0, num_grid=41):
    """
    หา temperature T ที่ minimize NLL บน hard labels
    ใช้ 1D grid search เพื่อความเสถียร
    """
    eps = 1e-15
    T_grid = np.linspace(T_min, T_max, num_grid)

    best_T = 1.0
    best_nll = np.inf

    for T in T_grid:
        p = sigmoid(logits / T)
        p = np.clip(p, eps, 1 - eps)
        nll = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        if nll < best_nll:
            best_nll = nll
            best_T = T

    return best_T

# ---------- Isotonic Regression บน softened probs ----------

def fit_isotonic(probs_T, y_soft):
    """
    Fit isotonic regression ให้ match probability ที่ผ่าน temperature scaling
    กับ soft labels
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_T, y_soft)
    return iso

def predict_soft_calibrated(base_model, T, iso, X):
    """
    X -> logits -> temperature scaling -> prob_T -> isotonic -> prob_soft
    """
    logits = base_model.decision_function(X)
    prob_T = sigmoid(logits / T)
    prob_soft = iso.predict(prob_T)
    return prob_soft

# ---------- Plot calibration (vs hard labels) ----------

def plot_calibration_hard(y_true, y_prob, filename="soft_calibrated_curve_v3.png"):
    """
    Reliability plot:
    y_true = binary HighResponder
    y_prob = prob หลัง 3 ชั้น calibration
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label="Soft-calibrated (3-layer)")
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("External Calibration Curve (HighResponder)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# =========================
# MAIN: Version 3
# =========================

def main():
    print("=== Version 3: Soft-Calibrated External Validation (3 layers) ===")

    # ----- Load data -----
    print("\n[1] Loading data...")
    train = load_train()
    ext = load_external()

    X_train = train[FEATURE_COLS]
    y_train_hard = train["HighResponder"].values
    y_train_soft = train["SoftResponder"].values

    X_ext = ext[FEATURE_COLS]
    y_ext_hard = ext["HighResponder"].values
    y_ext_soft = ext["SoftResponder"].values

    print(f"Train size:   {len(train)}")
    print(f"External size:{len(ext)}")

    # ----- Train base classifier -----
    print("\n[2] Training base classifier on HARD labels...")
    base_model = build_base_model()
    base_model.fit(X_train, y_train_hard)

    # Base logits / probs บน external (ไว้เทียบ)
    logits_ext = base_model.decision_function(X_ext)
    prob_base_ext = sigmoid(logits_ext)

    # ----- Temperature scaling -----
    print("\n[3] Fitting temperature scaling on TRAIN (hard NLL)...")
    train_logits = base_model.decision_function(X_train)
    T_opt = fit_temperature(train_logits, y_train_hard)
    print(f"   Optimal temperature T* = {T_opt:.3f}")

    prob_T_train = sigmoid(train_logits / T_opt)

    # ----- Isotonic บน softened probs + soft labels -----
    print("\n[4] Fitting isotonic regression on (prob_T_train vs soft labels)...")
    iso = fit_isotonic(prob_T_train, y_train_soft)

    # ----- External soft-calibrated prediction -----
    print("\n[5] Evaluating on EXTERNAL cohort...")
    prob_soft_ext = predict_soft_calibrated(base_model, T_opt, iso, X_ext)

    # ----- Metrics -----
    # Base (uncalibrated) vs hard labels
    auc_base = roc_auc_score(y_ext_hard, prob_base_ext)
    brier_base = brier_score_loss(y_ext_hard, prob_base_ext)

    # 3-layer soft-calibrated vs hard labels
    auc_soft = roc_auc_score(y_ext_hard, prob_soft_ext)
    brier_soft = brier_score_loss(y_ext_hard, prob_soft_ext)

    # Soft calibration quality vs soft labels
    mse_soft = mean_squared_error(y_ext_soft, prob_soft_ext)

    print("\n[6] Metrics (EXTERNAL):")
    print("  --- Base model (no calibration) ---")
    print(f"     AUROC (hard):        {auc_base:.3f}")
    print(f"     Brier (hard):        {brier_base:.3f}")

    print("  --- 3-layer soft-calibrated model ---")
    print(f"     AUROC (hard):        {auc_soft:.3f}")
    print(f"     Brier (hard):        {brier_soft:.3f}")
    print(f"     MSE (vs soft label): {mse_soft:.3f}")

    # ----- Plot calibration curve -----
    print("\n[7] Plotting external calibration curve (vs hard labels)...")
    plot_calibration_hard(y_ext_hard, prob_soft_ext,
                          filename="soft_calibrated_curve_v3.png")
    print("    Saved: soft_calibrated_curve_v3.png")

    print("\n=== Done. Version 3 pipeline finished. ===")

# =========================
# RUN DIRECTLY (Notebook-ready)
# =========================

main()