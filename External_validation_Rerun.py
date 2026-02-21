"""External_validation_aligned.py

Manuscript-consistent external-synthetic transfer (domain-shift stress test).

Key consistency requirements (per paper text):
- TG4h and derived response metrics (TGR) are used *only* to define the outcome label.
  They are *never* used as predictors.
- Predictors are restricted to baseline fasting variables:
    Age, Sex, BMI, TG0h, HDL, LDL, Hematocrit, TotalProtein, and WBV (derived from Hct+TP).
- WBV (de Simone low-shear surrogate):
    WBV = 0.12*Hct + 0.17*TP - 0.3519
- External cohort label is defined using the *external cohort's own* TG4h 75th percentile
  (this is a stress test, not clinical external validation).
- Model: L2-penalized logistic regression with per-fold standardization and one-hot sex encoding.
- Calibration (3-step procedure described in the manuscript):
    1) train on hard labels
    2) temperature scaling of logits (optimize NLL on CV logits)
    3) isotonic regression fitted to soft labels derived from proximity to TG4h threshold

Outputs:
- Prints internal CV AUROC/Brier (uncalibrated)
- Prints external AUROC/Brier (calibrated and uncalibrated)
- Saves ROC + calibration plots for external predictions

NOTE:
This script is intentionally "single-file" and explicit for auditability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


# =========================
# CONFIG
# =========================

TRAIN_PATH = "Dataset.csv"
EXTERNAL_PATH = "external_synthetic_like.csv"

RANDOM_SEED = 42
OUTER_FOLDS = 5

PRIMARY_PCTL = 75  # TG4h phenotype definition

# Soft-label temperature for proximity-to-threshold (used only for isotonic targets)
SOFT_TEMP = 30.0


# =========================
# DERIVED VARIABLES
# =========================

def compute_wbv(hct: pd.Series, tp: pd.Series) -> pd.Series:
    return 0.12 * hct + 0.17 * tp - 0.3519


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["WBV"] = compute_wbv(out["Hematocrit"], out["TotalProtein"])
    out["TGR"] = (out["TG4h"] - out["TG0h"]) / np.maximum(out["TG0h"], 1e-6) * 100.0
    return out


# =========================
# LABELS
# =========================

def tg4h_cutoff(df: pd.DataFrame, pctl: float) -> float:
    return float(np.percentile(df["TG4h"].to_numpy(dtype=float), pctl))


def hard_labels(df: pd.DataFrame, cutoff: float) -> np.ndarray:
    return (df["TG4h"].to_numpy(dtype=float) >= cutoff).astype(int)


def soft_labels(df: pd.DataFrame, cutoff: float, soft_temp: float = SOFT_TEMP) -> np.ndarray:
    # Soft target in [0,1] based on distance to the cutoff
    z = (df["TG4h"].to_numpy(dtype=float) - cutoff) / soft_temp
    return 1.0 / (1.0 + np.exp(-z))


# =========================
# MODEL
# =========================

def build_base_pipeline() -> Pipeline:
    num_cols = ["Age", "BMI", "TG0h", "HDL", "LDL", "Hematocrit", "TotalProtein", "WBV"]
    cat_cols = ["Sex"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=500,
        random_state=RANDOM_SEED,
    )

    return Pipeline([("pre", pre), ("clf", clf)])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def temperature_scale_logits(logits: np.ndarray, y: np.ndarray, iters: int = 2000, lr: float = 0.01) -> float:
    """Simple 1D temperature scaling (optimize NLL) on provided logits."""
    T = 1.0
    for _ in range(iters):
        z = logits / T
        p = 1.0 / (1.0 + np.exp(-z))
        # d/dT of NLL for logistic model
        grad = np.mean((p - y) * (-logits) / (T * T))
        T = max(1e-3, T - lr * grad)
    return float(T)


@dataclass
class Calibrator:
    T: float
    iso: IsotonicRegression

    def transform_proba(self, proba: np.ndarray) -> np.ndarray:
        logits = _logit(proba)
        logits_scaled = logits / self.T
        return np.clip(self.iso.predict(logits_scaled), 0.0, 1.0)


def fit_calibrator_cv(df_train: pd.DataFrame,
                      cutoff_train: float,
                      base_pipe: Pipeline) -> Calibrator:
    """Fit temperature scaling + isotonic regression using CV predictions on training data."""

    y_hard = hard_labels(df_train, cutoff_train)
    y_soft = soft_labels(df_train, cutoff_train, soft_temp=SOFT_TEMP)

    X = df_train

    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    proba_oof = np.zeros(len(df_train), dtype=float)

    for tr_idx, va_idx in cv.split(X, y_hard):
        pipe = build_base_pipeline()
        pipe.fit(X.iloc[tr_idx], y_hard[tr_idx])
        proba_oof[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1]

    logits_oof = _logit(proba_oof)

    # 1) Temperature scaling to hard labels
    T = temperature_scale_logits(logits_oof, y_hard)

    # 2) Isotonic regression to soft targets (on scaled logits)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(logits_oof / T, y_soft)

    return Calibrator(T=T, iso=iso)


# =========================
# EVALUATION + PLOTTING
# =========================

def plot_external_curves(y_true: np.ndarray, y_proba: np.ndarray, prefix: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label="Model ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix.lower().replace(' ', '_')}_roc_curve.png", dpi=300)

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Model calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(f"{prefix} Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix.lower().replace(' ', '_')}_calibration_curve.png", dpi=300)


def main() -> None:
    # --- Load ---
    train = pd.read_csv(TRAIN_PATH)
    ext = pd.read_csv(EXTERNAL_PATH)

    # Ensure required columns exist
    required = {"Sex", "Age", "Hematocrit", "TotalProtein", "TG0h", "TG4h", "HDL", "LDL", "BMI"}
    for name, df in [("train", train), ("external", ext)]:
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f"{name} dataset missing columns: {miss}")

    train = add_derived(train)
    ext = add_derived(ext)

    # --- Labels ---
    train_cut = tg4h_cutoff(train, PRIMARY_PCTL)
    y_train = hard_labels(train, train_cut)
    train["HighResponder"] = y_train
    train["Phenotype"] = np.where(train["HighResponder"]==1, "High", "Normal")

    ext_cut = tg4h_cutoff(ext, PRIMARY_PCTL)  # external cohort defines its own p75 (stress test)
    y_ext = hard_labels(ext, ext_cut)
    ext["HighResponder"] = y_ext
    ext["Phenotype"] = np.where(ext["HighResponder"]==1, "High", "Normal")

    # --- Internal CV (uncalibrated) ---
    base = build_base_pipeline()
    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    proba_oof = np.zeros(len(train), dtype=float)

    for tr_idx, va_idx in cv.split(train, y_train):
        pipe = build_base_pipeline()
        pipe.fit(train.iloc[tr_idx], y_train[tr_idx])
        proba_oof[va_idx] = pipe.predict_proba(train.iloc[va_idx])[:, 1]

    auc_int = roc_auc_score(y_train, proba_oof)
    brier_int = brier_score_loss(y_train, proba_oof)
    print("=== INTERNAL (5-fold CV on training cohort; uncalibrated) ===")
    print(f"TG4h p75 cutoff (train): {train_cut:.4f} mg/dL")
    print(f"AUROC: {auc_int:.4f}")
    print(f"Brier: {brier_int:.4f}")

    # --- Fit base model on full training ---
    base.fit(train, y_train)

    # --- Fit calibrator using CV logits on training ---
    cal = fit_calibrator_cv(train, train_cut, base)

    # --- External evaluation ---
    p_ext_uncal = base.predict_proba(ext)[:, 1]
    p_ext_cal = cal.transform_proba(p_ext_uncal)

    auc_ext_uncal = roc_auc_score(y_ext, p_ext_uncal)
    brier_ext_uncal = brier_score_loss(y_ext, p_ext_uncal)
    auc_ext_cal = roc_auc_score(y_ext, p_ext_cal)
    brier_ext_cal = brier_score_loss(y_ext, p_ext_cal)

    print("\n=== EXTERNAL (independent synthetic cohort; stress test) ===")
    print(f"TG4h p75 cutoff (external): {ext_cut:.4f} mg/dL")
    print(f"Uncalibrated  AUROC: {auc_ext_uncal:.4f} | Brier: {brier_ext_uncal:.4f}")
    print(f"Calibrated    AUROC: {auc_ext_cal:.4f} | Brier: {brier_ext_cal:.4f}")
    print(f"Temperature (T) used for scaling: {cal.T:.4f}")

    # --- Plots ---
    plot_external_curves(y_ext, p_ext_uncal, prefix="External Uncalibrated")
    plot_external_curves(y_ext, p_ext_cal, prefix="External Calibrated")

    print("\nSaved plots: *_roc_curve.png and *_calibration_curve.png")


if __name__ == "__main__":
    main()
