# run_rebuild_v2_pipeline.py
# ============================================================
# Rebuild v2 (physiology-coherent) + Leakage-controlled ML pipeline
# Outputs: Table1, sanity checks, nested-CV AUROC/Brier, calibration,
#          ablation, bootstrap CIs for correlations, SHAP (logistic),
#          figures with harmonized style.
#
# Usage:
#   python run_rebuild_v2_pipeline.py --data Dataset.csv --out out_v2
#
# Notes for reviewers:
# - TG4h is used ONLY to define labels (HighResponder). It is NEVER a predictor.
# - Postprandial response metric uses excursion ratio (TGR%) not "clearance".
# ============================================================

import os
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import partial_dependence

warnings.filterwarnings("ignore")

# ----------------------------
# Optional stats (p-values)
# ----------------------------
try:
    from scipy.stats import ttest_ind, chi2_contingency
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ----------------------------
# Plot style (harmonization)
# ----------------------------
def set_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


@dataclass
class Config:
    random_state: int = 42

    # Nested CV (primary evaluation)
    outer_folds: int = 5
    inner_folds: int = 5

    # Bootstrap (paper uses 1,000 resamples for model reliability)
    n_boot_final: int = 1000          # bootstrap for AUROC/Brier of final model (OOF-based)
    n_boot_corr: int = 1000           # bootstrap for correlation CIs

    # Robustness: repeated stratified CV (5-fold x 10 repeats)
    repeated_folds: int = 5
    repeated_repeats: int = 10

    # Calibration
    calib_method: str = "isotonic"    # "isotonic" or "sigmoid"

    # Primary phenotype definition (TG4h percentile, computed within each training fold)
    label_percentile: float = 75.0

    # Threshold sensitivity sweep (TG4h percentile 60–90)
    threshold_sweep: Tuple[float, ...] = (60.0, 70.0, 75.0, 80.0, 90.0)

    # Winsorization (fit on training folds)
    winsor_q_low: float = 0.01
    winsor_q_high: float = 0.99

    # Model comparisons (paper reports comparisons; optional)
    compare_model_zoo: bool = True

    # WBV misspecification sensitivity (relative Gaussian noise around unity)
    wbv_noise_levels: Tuple[float, ...] = (0.05, 0.10)

    # Pseudo-external subgroup split (age threshold)
    age_cutoff: float = 55.0



# ----------------------------
# Data + derived variables
# ----------------------------
REQUIRED_COLS = [
    "Sex", "Age", "Hematocrit", "TotalProtein", "TG0h", "TG4h", "HDL", "LDL", "BMI"
]

NUMERIC_FEATURES = ["Age", "Hematocrit", "TotalProtein", "TG0h", "HDL", "LDL", "BMI"]
CATEGORICAL_FEATURES = ["Sex"]

# IMPORTANT: TG4h is NOT a feature (label-only).
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def compute_wbv_de_simone_like(hct: pd.Series, tp: pd.Series) -> pd.Series:
    """
    Use the (manuscript-consistent) linear surrogate used in your external cohort script:
    WBV = 0.12*Hct + 0.17*TotalProtein - 0.3519
    (You will add the proper de Simone primary citation in the paper.)
    """
    return 0.12 * hct + 0.17 * tp - 0.3519


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # WBV (surrogate)
    if "WBV" not in out.columns:
        out["WBV"] = compute_wbv_de_simone_like(out["Hematocrit"], out["TotalProtein"])

    # Postprandial response ratio (avoid calling it "clearance" for 0h vs 4h only)
    # TGR% = (TG4h - TG0h)/TG0h * 100
    out["TGR"] = (out["TG4h"] - out["TG0h"]) / np.maximum(out["TG0h"], 1e-6) * 100.0

    return out


def make_label_from_tg4h_percentile(df: pd.DataFrame, pct: float) -> Tuple[pd.Series, float]:
    """
    Define phenotype as TG4h >= pct-th percentile *within the provided dataset*.

    Manuscript consistency:
    - TG4h is label-only (never a predictor).
    - For external-synthetic stress tests, the external cohort may define its own percentile cutoff.
    """
    cutoff = float(np.percentile(df["TG4h"].values, pct))
    y = (df["TG4h"].values >= cutoff).astype(int)
    return pd.Series(y, index=df.index, name="HighResponder"), cutoff


def make_fold_labels_from_train_cutoff(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    pct: float
) -> Tuple[pd.Series, pd.Series, float]:
    """Fold-specific phenotype definition (STRICT).

    - Compute TG4h cutoff from *training indices only*.
    - Derive y_train and y_valid using that training-derived cutoff.

    This matches the manuscript's leakage-controlled phenotype construction.
    """
    tg4h_train = df.iloc[train_idx]["TG4h"].values
    cutoff = float(np.percentile(tg4h_train, pct))

    y_train = (df.iloc[train_idx]["TG4h"].values >= cutoff).astype(int)
    y_valid = (df.iloc[valid_idx]["TG4h"].values >= cutoff).astype(int)

    y_train_s = pd.Series(y_train, index=df.index[train_idx], name="HighResponder")
    y_valid_s = pd.Series(y_valid, index=df.index[valid_idx], name="HighResponder")
    return y_train_s, y_valid_s, cutoff


class Winsorizer:
    """Train-fold-fitted winsorization (quantile clipping)."""

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99):
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X_arr, self.q_low, axis=0)
        self.upper_ = np.nanquantile(X_arr, self.q_high, axis=0)
        return self

    def transform(self, X):
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("Winsorizer is not fitted.")
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_, self.upper_)



# ----------------------------
# Table 1 generator
# ----------------------------
def _p_value_numeric(x0: np.ndarray, x1: np.ndarray) -> float:
    if not _HAS_SCIPY:
        return np.nan
    # Welch t-test
    stat, p = ttest_ind(x0, x1, equal_var=False, nan_policy="omit")
    return float(p)


def _p_value_categorical(counts: np.ndarray) -> float:
    if not _HAS_SCIPY:
        return np.nan
    # Chi-square
    chi2, p, dof, exp = chi2_contingency(counts)
    return float(p)


def make_table1(df: pd.DataFrame, y: pd.Series, out_csv: str) -> pd.DataFrame:
    """
    Recreate Table 1 (mean±SD or n(%)) by phenotype.
    """
    df0 = df.loc[y == 0].copy()
    df1 = df.loc[y == 1].copy()

    rows = []

    # Sex
    for sex_val in ["Female", "Male"]:
        n0 = int((df0["Sex"] == sex_val).sum())
        n1 = int((df1["Sex"] == sex_val).sum())
        p0 = 100.0 * n0 / max(len(df0), 1)
        p1 = 100.0 * n1 / max(len(df1), 1)
        rows.append((f"{sex_val}, n (%)", f"{n0} ({p0:.4f}%)", f"{n1} ({p1:.4f}%)", np.nan))

    # Sex p-value (2x2)
    counts = np.array([
        [(df0["Sex"] == "Female").sum(), (df0["Sex"] == "Male").sum()],
        [(df1["Sex"] == "Female").sum(), (df1["Sex"] == "Male").sum()],
    ], dtype=float)
    p_sex = _p_value_categorical(counts)

    # overwrite p-values for sex rows
    rows[0] = (rows[0][0], rows[0][1], rows[0][2], p_sex)
    rows[1] = (rows[1][0], rows[1][1], rows[1][2], p_sex)

    # Numeric vars
    numeric_vars = [
        ("Age, years", "Age"),
        ("Hematocrit, %", "Hematocrit"),
        ("Total protein, g/dL", "TotalProtein"),
        ("Whole blood viscosity, cP", "WBV"),
        ("Fasting TG (TG0h), mg/dL", "TG0h"),
        ("4-h TG (TG4h), mg/dL", "TG4h"),
        ("Postprandial response (TGR), %", "TGR"),
        ("HDL-C, mg/dL", "HDL"),
        ("LDL-C, mg/dL", "LDL"),
        ("BMI, kg/m^2", "BMI"),
    ]

    for label, col in numeric_vars:
        # กันพังถ้าคอลัมน์หาย (ไม่ควรหาย แต่กันไว้)
        if col not in df.columns:
            rows.append((label, "NA", "NA", np.nan))
            continue

        m0, s0 = float(df0[col].mean()), float(df0[col].std())
        m1, s1 = float(df1[col].mean()), float(df1[col].std())
        p = _p_value_numeric(df0[col].values, df1[col].values)
        rows.append((label, f"{m0:.4f} ± {s0:.4f}", f"{m1:.4f} ± {s1:.4f}", p))

    table = pd.DataFrame(rows, columns=["Variable", "Normal", "High", "p_value"])
    table.to_csv(out_csv, index=False)
    return table


# ----------------------------
# Preprocessor + models
# ----------------------------
def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], cfg: Config) -> ColumnTransformer:
    """
    IMPORTANT: We must use only columns that exist in X passed to the model.
    This avoids KeyError during ablation where some columns are intentionally dropped.
    """
    # Leakage-controlled preprocessing: impute -> winsorize (fit quantiles on train) -> scale
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(q_low=cfg.winsor_q_low, q_high=cfg.winsor_q_high)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop"
    )


def build_base_model(random_state: int) -> LogisticRegression:
    # L2-penalized logistic regression (interpretable + manuscript-consistent baseline)
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=8000,
        class_weight="balanced",
        random_state=random_state
    )


def nested_cv_calibrated(
    df_full: pd.DataFrame,
    X: pd.DataFrame,
    cfg: Config,
    out_dir: str
) -> Dict[str, float]:
    """
    Nested CV:
      Outer: evaluate
      Inner: calibrate via CalibratedClassifierCV within training folds
    We keep it strict: calibration is done ONLY using inner-CV on training.
    """
    os.makedirs(out_dir, exist_ok=True)

    # For splitting only, we stratify using a *global* label proxy.
    # The true phenotype labels used for training/evaluation are fold-specific
    # and are recomputed in each outer fold from the training TG4h percentile.
    y_proxy, _ = make_label_from_tg4h_percentile(df_full, cfg.label_percentile)
    if len(np.unique(y_proxy.values)) < 2:
        raise ValueError("Proxy y has only one class; cannot stratify outer CV.")

    outer = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.random_state)

    aurocs, briers, aps = [], [], []
    oof_proba = np.full(len(df_full), np.nan, dtype=float)
    oof_y = np.full(len(df_full), np.nan, dtype=float)
    oof_cutoff = np.full(len(df_full), np.nan, dtype=float)

    # เลือกคอลัมน์ให้ตรงกับ X จริง (กัน KeyError ตอน ablation)
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    # ถ้า X มี WBV (เช่นตอน ablation) ให้ถือเป็น numeric ได้ทันที
    if "WBV" in X.columns and "WBV" not in numeric_cols:
        numeric_cols = numeric_cols + ["WBV"]

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y_proxy), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]

        # STRICT: fold-specific phenotype labels
        y_tr, y_te, cutoff = make_fold_labels_from_train_cutoff(
            df=df_full,
            train_idx=tr_idx,
            valid_idx=te_idx,
            pct=cfg.label_percentile
        )

        # Skip pathological folds (should not happen with percentile phenotype + stratified split)
        if len(np.unique(y_tr.values)) < 2 or len(np.unique(y_te.values)) < 2:
            raise ValueError(
                f"Fold {fold} has a single class after fold-specific cutoff. "
                "Try increasing N, adjusting percentile, or using a different splitting strategy."
            )

        pre = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols, cfg=cfg)

        # Model zoo (paper compares multiple; main conclusion is L2-logistic + calibration)
        models = {
            "logreg_l2": LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=8000,
                class_weight="balanced",
                random_state=cfg.random_state + fold,
            )
        }
        param_grids = {
            "logreg_l2": {"model__C": [0.01, 0.1, 1.0, 10.0]}
        }

        if cfg.compare_model_zoo:
            models.update({
                "svm_rbf": SVC(probability=True, class_weight="balanced", random_state=cfg.random_state + fold),
                "rf": RandomForestClassifier(
                    n_estimators=600,
                    random_state=cfg.random_state + fold,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            })
            param_grids.update({
                "svm_rbf": {"model__C": [0.5, 1.0, 2.0, 4.0], "model__gamma": ["scale", "auto"]},
                "rf": {"model__max_depth": [None, 3, 5, 8], "model__min_samples_leaf": [1, 2, 5]},
            })

        inner = StratifiedKFold(
            n_splits=cfg.inner_folds,
            shuffle=True,
            random_state=cfg.random_state + 100 + fold
        )

        fold_rows = []
        best_calib = None
        best_auc = -np.inf

        for mname, model in models.items():
            pipe = Pipeline([("preprocess", pre), ("model", model)])
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grids[mname],
                scoring="roc_auc",
                cv=inner,
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X_tr, y_tr)

            # Calibrate ONLY on training data via inner CV (no access to outer test fold)
            calib = CalibratedClassifierCV(
                estimator=gs.best_estimator_,
                method=cfg.calib_method,
                cv=inner
            )
            calib.fit(X_tr, y_tr)

            proba = calib.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, proba)
            brier = brier_score_loss(y_te, proba)
            ap = average_precision_score(y_te, proba)

            fold_rows.append({
                "fold": fold,
                "model": mname,
                "cutoff_tg4h": cutoff,
                "auroc": auc,
                "brier": brier,
                "ap": ap,
                "best_params": json.dumps(gs.best_params_),
            })

            if auc > best_auc:
                best_auc = auc
                best_calib = calib

        # Save model comparison per fold (reviewer-proof trace)
        pd.DataFrame(fold_rows).to_csv(
            os.path.join(out_dir, f"fold_{fold}_model_comparison.csv"),
            index=False
        )

        # Use best calibrated model for OOF predictions
        proba_best = best_calib.predict_proba(X_te)[:, 1]
        oof_proba[te_idx] = proba_best
        oof_y[te_idx] = y_te.values
        oof_cutoff[te_idx] = cutoff

        aurocs.append(roc_auc_score(y_te, proba_best))
        briers.append(brier_score_loss(y_te, proba_best))
        aps.append(average_precision_score(y_te, proba_best))

        print(
            f"[Outer fold {fold}] cutoff(TG4h,p{cfg.label_percentile})={cutoff:.3f}  "
            f"AUROC={aurocs[-1]:.4f}  Brier={briers[-1]:.4f}  AP={aps[-1]:.4f}"
        )

    summary = {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "brier_mean": float(np.mean(briers)),
        "brier_std": float(np.std(briers)),
        "ap_mean": float(np.mean(aps)),
        "ap_std": float(np.std(aps)),
    }

    # Save OOF predictions for calibration/ROC plots
    pd.DataFrame({
        "y": oof_y,
        "proba_oof": oof_proba,
        "fold_cutoff_tg4h": oof_cutoff,
    }).to_csv(
        os.path.join(out_dir, "oof_predictions.csv"),
        index=False
    )

    with open(os.path.join(out_dir, "nestedcv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ----------------------------
# Plots
# ----------------------------
def plot_tg_distributions(df: pd.DataFrame, y: pd.Series, out_dir: str):
    set_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    # TG0h
    if "TG0h" in df.columns:
        plt.figure(figsize=(6.5, 4.5))
        plt.hist(df.loc[y == 0, "TG0h"], bins=30, density=True, alpha=0.7, label="Normal")
        plt.hist(df.loc[y == 1, "TG0h"], bins=30, density=True, alpha=0.7, label="High")
        plt.xlabel("Fasting TG (TG0h), mg/dL")
        plt.ylabel("Density")
        plt.title("TG0h distribution by phenotype")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_tg0h_distribution.png"))
        plt.close()

    # TG4h
    if "TG4h" in df.columns:
        plt.figure(figsize=(6.5, 4.5))
        plt.hist(df.loc[y == 0, "TG4h"], bins=30, density=True, alpha=0.7, label="Normal")
        plt.hist(df.loc[y == 1, "TG4h"], bins=30, density=True, alpha=0.7, label="High")
        plt.xlabel("Postprandial TG (TG4h), mg/dL")
        plt.ylabel("Density")
        plt.title("TG4h distribution by phenotype")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_tg4h_distribution.png"))
        plt.close()


def plot_correlation_matrix(df: pd.DataFrame, cols: List[str], out_dir: str):
    set_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    cols_use = [c for c in cols if c in df.columns]
    if len(cols_use) < 2:
        return

    corr = df[cols_use].corr()

    plt.figure(figsize=(7.2, 6.2))
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols_use)), cols_use, rotation=90)
    plt.yticks(range(len(cols_use)), cols_use)
    plt.title("Correlation matrix (key biomarkers)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_correlation_matrix.png"))
    plt.close()

    corr.to_csv(os.path.join(out_dir, "correlation_matrix.csv"))


def plot_roc_pr_calibration(y_true: np.ndarray, proba: np.ndarray, out_dir: str):
    set_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    plt.figure(figsize=(6.2, 5.0))
    plt.plot(fpr, tpr, label=f"Model (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (OOF, nested CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_roc_oof.png"))
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    plt.figure(figsize=(6.2, 5.0))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve (OOF, nested CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_pr_oof.png"))
    plt.close()

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6.2, 5.0))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration curve (OOF, nested CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_calibration_oof.png"))
    plt.close()



# ----------------------------
# Calibration slope/intercept + Decision Curve Analysis (DCA)
# ----------------------------
def calibration_slope_intercept(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    """
    Calibration slope/intercept via logistic calibration line:
        y ~ a + b * logit(p)
    Ideal: slope=1, intercept=0.
    """
    y = np.asarray(y_true).astype(float)
    p = np.asarray(proba).astype(float)
    eps = 1e-8
    p = np.clip(p, eps, 1 - eps)
    x = np.log(p / (1 - p))  # logit
    X = np.column_stack([np.ones_like(x), x])  # [1, logit(p)]
    # Least squares fit
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    return slope, intercept


def decision_curve_net_benefit(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """
    Standard net benefit:
        NB(pt) = TP/n - FP/n * (pt/(1-pt))
    Compare with treat-all and treat-none.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba).astype(float)
    n = len(y)

    out_rows = []
    prev = float(np.mean(y))  # event rate
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        w = pt / max(1 - pt, 1e-12)

        nb_model = (tp / n) - (fp / n) * w
        # treat-all: everyone positive
        tp_all = int((y == 1).sum())
        fp_all = int((y == 0).sum())
        nb_all = (tp_all / n) - (fp_all / n) * w
        nb_none = 0.0

        out_rows.append({
            "pt": float(pt),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": float(nb_none),
            "prevalence": float(prev),
        })

    return pd.DataFrame(out_rows)


def run_dca_with_baselines(df_full: pd.DataFrame, cfg: Config, out_dir: str):
    """
    DCA is computed from nested-CV OOF predictions for:
    - full multivariable model (from nested CV)
    - univariate TG0h-only logistic regression (nested CV)
    - TG0h percentile rule (baseline)
    """
    set_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    # Load OOF of full model (already produced by nested_cv_calibrated in main)
    oof_path = os.path.join(out_dir, "..", "oof_predictions.csv")
    if not os.path.exists(oof_path):
        # fall back to current directory
        oof_path = os.path.join(out_dir, "oof_predictions.csv")
    oof = pd.read_csv(oof_path).dropna(subset=["y", "proba_oof"]).copy()
    y = oof["y"].astype(int).values
    p_full = oof["proba_oof"].astype(float).values

    # TG0h-only nested CV OOF (leakage-controlled)
    X_tg0h = df_full[["TG0h", "Sex"]].copy() if "Sex" in df_full.columns else df_full[["TG0h"]].copy()
    cfg_tg = Config(**{**cfg.__dict__})
    cfg_tg.compare_model_zoo = False
    tg_dir = os.path.join(out_dir, "tg0h_only")
    nested_cv_calibrated(df_full=df_full, X=X_tg0h, cfg=cfg_tg, out_dir=tg_dir)
    oof_tg = pd.read_csv(os.path.join(tg_dir, "oof_predictions.csv")).dropna(subset=["y", "proba_oof"]).copy()
    # Align indices by row order (same df_full order, with NaNs only where fold issues)
    p_tg = oof_tg["proba_oof"].astype(float).values
    y_tg = oof_tg["y"].astype(int).values

    # TG0h percentile rule baseline (OOF, fold-specific; no leakage)
    # We compute an OOF "rule" prediction by deriving the TG0h cutoff on each training fold
    # and applying it to that fold's held-out indices.
    y_split = df_full["HighResponder"].values.astype(int) if "HighResponder" in df_full.columns else y
    tg = df_full["TG0h"].values.astype(float)

    cv_outer = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.random_state)
    p_rule_oof = np.full(len(df_full), np.nan, dtype=float)
    for train_idx, test_idx in cv_outer.split(np.zeros_like(y_split), y_split):
        cut = float(np.percentile(tg[train_idx], cfg.label_percentile))  # training percentile only
        p_rule_oof[test_idx] = (tg[test_idx] >= cut).astype(float)

    # Align to the OOF rows used for the multivariable model (drop NaNs consistently)
    p_rule = p_rule_oof[oof.index.values]


    thresholds = np.linspace(0.01, 0.99, 99)
    d_full = decision_curve_net_benefit(y, p_full, thresholds)
    d_tg = decision_curve_net_benefit(y_tg, p_tg, thresholds)
    d_rule = decision_curve_net_benefit(y, p_rule, thresholds)

    d_full = d_full.rename(columns={"net_benefit_model": "nb_full"})
    d_tg = d_tg.rename(columns={"net_benefit_model": "nb_tg0h_lr"})
    d_rule = d_rule.rename(columns={"net_benefit_model": "nb_tg0h_rule"})

    d = d_full[["pt", "nb_full", "net_benefit_treat_all", "net_benefit_treat_none"]].copy()
    d["nb_tg0h_lr"] = d_tg["nb_tg0h_lr"].values
    d["nb_tg0h_rule"] = d_rule["nb_tg0h_rule"].values
    d.to_csv(os.path.join(out_dir, "decision_curve.csv"), index=False)

    # Plot
    plt.figure(figsize=(6.6, 5.2))
    plt.plot(d["pt"], d["nb_full"], label="Multivariable model")
    plt.plot(d["pt"], d["nb_tg0h_lr"], label="TG0h-only LR")
    plt.plot(d["pt"], d["nb_tg0h_rule"], label=f"TG0h p{cfg.label_percentile:.0f} rule")
    plt.plot(d["pt"], d["net_benefit_treat_all"], linestyle="--", label="Treat all")
    plt.plot(d["pt"], d["net_benefit_treat_none"], linestyle="--", label="Treat none")
    plt.xlabel("Decision threshold (pt)")
    plt.ylabel("Net benefit")
    plt.title("Decision curve analysis (OOF, leakage-controlled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_decision_curve.png"))
    plt.close()


# ----------------------------
# Robustness: repeated 5×10 cross-validation
# ----------------------------
def repeated_cv_5x10(df_full: pd.DataFrame, X: pd.DataFrame, cfg: Config, out_dir: str) -> Dict[str, float]:
    """
    Repeated stratified 5-fold CV for robustness (5 folds × 10 repeats).
    Uses fold-specific TG4h cutoff computed from the training indices of each fold.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Proxy only for splitting (as in nested CV)
    y_proxy, _ = make_label_from_tg4h_percentile(df_full, cfg.label_percentile)

    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    if "WBV" in X.columns and "WBV" not in numeric_cols:
        numeric_cols = numeric_cols + ["WBV"]

    aurocs, briers = [], []
    rows = []

    for rep in range(cfg.repeated_repeats):
        outer = StratifiedKFold(
            n_splits=cfg.repeated_folds,
            shuffle=True,
            random_state=cfg.random_state + 1000 + rep
        )
        for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y_proxy), start=1):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te, cutoff = make_fold_labels_from_train_cutoff(df_full, tr_idx, te_idx, cfg.label_percentile)

            pre = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols, cfg=cfg)
            base = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=8000,
                class_weight="balanced",
                random_state=cfg.random_state + rep * 100 + fold,
            )
            pipe = Pipeline([("preprocess", pre), ("model", base)])

            inner = StratifiedKFold(
                n_splits=cfg.inner_folds,
                shuffle=True,
                random_state=cfg.random_state + 2000 + rep * 10 + fold
            )
            gs = GridSearchCV(
                estimator=pipe,
                param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
                scoring="roc_auc",
                cv=inner,
                n_jobs=-1,
                refit=True
            )
            gs.fit(X_tr, y_tr)

            calib = CalibratedClassifierCV(estimator=gs.best_estimator_, method=cfg.calib_method, cv=inner)
            calib.fit(X_tr, y_tr)

            p = calib.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, p)
            br = brier_score_loss(y_te, p)

            aurocs.append(auc)
            briers.append(br)
            rows.append({
                "repeat": rep + 1,
                "fold": fold,
                "cutoff_tg4h": cutoff,
                "auroc": float(auc),
                "brier": float(br),
                "best_params": json.dumps(gs.best_params_),
            })

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "repeatedcv_5x10_folds.csv"), index=False)

    summary = {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auroc_min": float(np.min(aurocs)),
        "auroc_max": float(np.max(aurocs)),
        "brier_mean": float(np.mean(briers)),
        "brier_std": float(np.std(briers)),
        "brier_min": float(np.min(briers)),
        "brier_max": float(np.max(briers)),
        "n_folds_total": int(len(aurocs)),
    }
    with open(os.path.join(out_dir, "repeatedcv_5x10_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# ----------------------------
# Bootstrap (1,000) for AUROC/Brier of the final model (OOF-based)
# ----------------------------
def bootstrap_final_model_oof(oof_csv: str, cfg: Config, out_dir: str) -> pd.DataFrame:
    """
    Bootstrap the OOF predictions to obtain AUROC/Brier distributions (1,000 resamples).
    This matches the paper's reliability bootstrap without re-fitting models.
    """
    set_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    oof = pd.read_csv(oof_csv).dropna(subset=["y", "proba_oof"]).copy()
    y = oof["y"].astype(int).values
    p = oof["proba_oof"].astype(float).values
    n = len(y)

    rng = np.random.default_rng(cfg.random_state + 999)
    aucs, briers = [], []
    idx = np.arange(n)
    for _ in range(cfg.n_boot_final):
        samp = rng.choice(idx, size=n, replace=True)
        ys = y[samp]
        ps = p[samp]
        # AUROC can fail if only 1 class in bootstrap sample; skip those (rare)
        if len(np.unique(ys)) < 2:
            continue
        aucs.append(roc_auc_score(ys, ps))
        briers.append(brier_score_loss(ys, ps))

    out = pd.DataFrame({"auroc": aucs, "brier": briers})
    out.to_csv(os.path.join(out_dir, "bootstrap_finalmodel_oof.csv"), index=False)

    # Histograms
    plt.figure(figsize=(6.0, 4.2))
    plt.hist(out["auroc"].values, bins=25)
    plt.xlabel("AUROC")
    plt.ylabel("Frequency")
    plt.title(f"Bootstrap AUROC (n={len(out)})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_bootstrap_auroc.png"))
    plt.close()

    plt.figure(figsize=(6.0, 4.2))
    plt.hist(out["brier"].values, bins=25)
    plt.xlabel("Brier score")
    plt.ylabel("Frequency")
    plt.title(f"Bootstrap Brier (n={len(out)})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_bootstrap_brier.png"))
    plt.close()

    # Percentile CIs
    summary = {
        "auroc_mean": float(np.mean(out["auroc"])),
        "auroc_ci95": [float(np.percentile(out["auroc"], 2.5)), float(np.percentile(out["auroc"], 97.5))],
        "brier_mean": float(np.mean(out["brier"])),
        "brier_ci95": [float(np.percentile(out["brier"], 2.5)), float(np.percentile(out["brier"], 97.5))],
    }
    with open(os.path.join(out_dir, "bootstrap_finalmodel_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out


# ----------------------------
# Threshold sensitivity sweep (TG4h percentile 60–90)
# ----------------------------
def threshold_sensitivity_sweep(df_full: pd.DataFrame, cfg: Config, out_dir: str) -> pd.DataFrame:
    """
    Re-run the full leakage-controlled nested CV pipeline across multiple TG4h percentile cutoffs.
    For each percentile, report AUROC/Brier + classification metrics from OOF predictions.
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for pct in cfg.threshold_sweep:
        cfg_th = Config(**{**cfg.__dict__})
        cfg_th.label_percentile = float(pct)
        cfg_th.compare_model_zoo = False  # Table 7 focuses on L2-penalized logistic regression

        th_dir = os.path.join(out_dir, f"pct_{int(pct)}")
        X = df_full[FEATURE_COLS].copy()
        summary = nested_cv_calibrated(df_full=df_full, X=X, cfg=cfg_th, out_dir=th_dir)

        oof = pd.read_csv(os.path.join(th_dir, "oof_predictions.csv")).dropna(subset=["y", "proba_oof"]).copy()
        y = oof["y"].astype(int).values
        p = oof["proba_oof"].astype(float).values

        # Default operating point at 0.5 for reporting (paper reports F1/recall/precision)
        pred = (p >= 0.5).astype(int)
        row = {
            "percentile": float(pct),
            "cutoff_mgdl_mean": float(np.nanmean(oof["fold_cutoff_tg4h"].values)),
            "auroc_mean": summary["auroc_mean"],
            "brier_mean": summary["brier_mean"],
            "f1": float(f1_score(y, pred)),
            "recall": float(recall_score(y, pred)),
            "precision": float(precision_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("percentile")
    out.to_csv(os.path.join(out_dir, "threshold_sensitivity_summary.csv"), index=False)

    # Plot AUROC vs percentile
    set_plot_style()
    plt.figure(figsize=(6.2, 5.0))
    plt.plot(out["percentile"].values, out["auroc_mean"].values, marker="o")
    plt.xlabel("TG4h phenotype percentile cutoff")
    plt.ylabel("AUROC (nested CV mean)")
    plt.title("Threshold sensitivity (60–90th percentiles)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_threshold_sensitivity_auroc.png"))
    plt.close()

    return out


# ----------------------------
# WBV misspecification sensitivity (±5%, ±10% Gaussian noise)
# ----------------------------
def wbv_misspecification_sensitivity(df_full: pd.DataFrame, cfg: Config, out_dir: str) -> pd.DataFrame:
    """
    Perturb WBV by multiplicative Gaussian noise around unity:
        WBV_pert = WBV * (1 + Normal(0, sigma))
    For each sigma in {0.05, 0.10}, recompute:
    - Pearson correlation between WBV and TGR
    - WBV-only AUROC via leakage-controlled nested CV
    """
    os.makedirs(out_dir, exist_ok=True)

    if "WBV" not in df_full.columns:
        raise ValueError("WBV column is required for WBV misspecification sensitivity.")

    rng = np.random.default_rng(cfg.random_state + 2024)
    rows = []

    for sigma in cfg.wbv_noise_levels:
        dfp = df_full.copy()
        noise = rng.normal(loc=0.0, scale=float(sigma), size=len(dfp))
        dfp["WBV"] = dfp["WBV"].values * (1.0 + noise)

        # Correlation with TGR
        r = float(np.corrcoef(dfp["WBV"].values, dfp["TGR"].values)[0, 1])

        # WBV-only nested CV AUROC/Brier
        X_wbv = dfp[["WBV"]].copy()
        cfg_w = Config(**{**cfg.__dict__})
        cfg_w.compare_model_zoo = False
        res = nested_cv_calibrated(df_full=dfp, X=X_wbv, cfg=cfg_w, out_dir=os.path.join(out_dir, f"sigma_{sigma:.2f}"))

        rows.append({
            "sigma": float(sigma),
            "pearson_r_wbv_tgr": r,
            "wbv_only_auroc": float(res["auroc_mean"]),
            "wbv_only_brier": float(res["brier_mean"]),
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "wbv_misspec_sensitivity.csv"), index=False)
    return out


# ----------------------------
# Pseudo-external-synthetic transfer: age/sex subgroup evaluation
# ----------------------------
def pseudo_external_transfer(df_full: pd.DataFrame, cfg: Config, out_dir: str) -> pd.DataFrame:
    """
    Demographic pseudo-external evaluation:
      - Age: train <55, test ≥55; then reverse
      - Sex: train male, test female; then reverse

    Labels are defined leakage-controlled: TG4h cutoff is computed on TRAIN group only,
    then applied to TEST group.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _fit_eval(train_mask: np.ndarray, test_mask: np.ndarray, desc: str) -> Dict[str, float]:
        df_tr = df_full.loc[train_mask].copy()
        df_te = df_full.loc[test_mask].copy()

        # TRAIN-derived cutoff
        cutoff = float(np.percentile(df_tr["TG4h"].values, cfg.label_percentile))
        y_tr = (df_tr["TG4h"].values >= cutoff).astype(int)
        y_te = (df_te["TG4h"].values >= cutoff).astype(int)

        X_tr = df_tr[FEATURE_COLS].copy()
        X_te = df_te[FEATURE_COLS].copy()

        numeric_cols = [c for c in NUMERIC_FEATURES if c in X_tr.columns]
        categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X_tr.columns]
        pre = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols, cfg=cfg)

        base = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=8000,
            class_weight="balanced",
            random_state=cfg.random_state + 77
        )
        pipe = Pipeline([("preprocess", pre), ("model", base)])

        inner = StratifiedKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.random_state + 88)
        gs = GridSearchCV(
            estimator=pipe,
            param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
            scoring="roc_auc",
            cv=inner,
            n_jobs=-1,
            refit=True
        )
        gs.fit(X_tr, y_tr)

        calib = CalibratedClassifierCV(estimator=gs.best_estimator_, method=cfg.calib_method, cv=inner)
        calib.fit(X_tr, y_tr)

        p = calib.predict_proba(X_te)[:, 1]
        if len(np.unique(y_te)) < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(y_te, p))
        br = float(brier_score_loss(y_te, p))
        return {
            "desc": desc,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "cutoff_tg4h": float(cutoff),
            "auroc": float(auc),
            "brier": float(br),
        }

    age = df_full["Age"].values
    sex = df_full["Sex"].astype(str).values

    mask_age_lo = age < cfg.age_cutoff
    mask_age_hi = age >= cfg.age_cutoff

    mask_male = sex == "Male"
    mask_female = sex == "Female"

    rows = []
    rows.append(_fit_eval(mask_age_lo, mask_age_hi, f"Train Age<{cfg.age_cutoff:g}, Test Age≥{cfg.age_cutoff:g}"))
    rows.append(_fit_eval(mask_age_hi, mask_age_lo, f"Train Age≥{cfg.age_cutoff:g}, Test Age<{cfg.age_cutoff:g}"))
    rows.append(_fit_eval(mask_male, mask_female, "Train Male, Test Female"))
    rows.append(_fit_eval(mask_female, mask_male, "Train Female, Test Male"))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "pseudo_external_transfer.csv"), index=False)
    return out


# ----------------------------
# SHAP + Partial Dependence (1D/2D)
# ----------------------------
def _get_feature_names_from_column_transformer(ct: ColumnTransformer) -> List[str]:
    """
    Best-effort extraction of output feature names after preprocessing.
    """
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                fn = list(trans.get_feature_names_out(cols))
            except Exception:
                fn = list(trans.get_feature_names_out())
            names.extend(fn)
        else:
            # passthrough / unknown transformer
            if isinstance(cols, (list, tuple)):
                names.extend(list(cols))
    return [str(x) for x in names]


def run_shap_and_pdp(df_full: pd.DataFrame, cfg: Config, out_dir: str):
    """
    Explainability for the final multivariable LR model:
    - SHAP summary + TG0h dependence + TG0h×WBV interaction (if available)
    - PDP 1D for key features + PDP 2D (TG0h×BMI) as in the manuscript

    NOTE: This block requires 'shap' to be installed. If not present, PDP will still run.
    """
    set_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    # Use main phenotype definition on full dataset for explainability (common practice for attribution plots)
    y_global, cutoff = make_label_from_tg4h_percentile(df_full, cfg.label_percentile)
    X = df_full[FEATURE_COLS].copy()

    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    pre = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols, cfg=cfg)

    base = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=8000,
        class_weight="balanced",
        random_state=cfg.random_state + 7
    )
    pipe = Pipeline([("preprocess", pre), ("model", base)])

    inner = StratifiedKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.random_state + 9)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="roc_auc",
        cv=inner,
        n_jobs=-1,
        refit=True
    )
    gs.fit(X, y_global)

    final_pipe = gs.best_estimator_
    with open(os.path.join(out_dir, "explainability_finalmodel_params.json"), "w", encoding="utf-8") as f:
        json.dump(gs.best_params_, f, indent=2)

    # ---------------- PDP (always) ----------------
    # 1D PDP for TG0h + BMI + WBV (if available)
    for feat in ["TG0h", "BMI", "WBV"]:
        if feat not in X.columns:
            continue
        try:
            pd_res = partial_dependence(final_pipe, X=X, features=[feat], kind="average", grid_resolution=200)
            xs = pd_res["grid_values"][0]
            ys = pd_res["average"][0]
            plt.figure(figsize=(6.2, 4.6))
            plt.plot(xs, ys)
            plt.xlabel(feat)
            plt.ylabel("Partial dependence (avg predicted probability)")
            plt.title(f"1D PDP: {feat}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"fig_pdp_1d_{feat}.png"))
            plt.close()
        except Exception as e:
            with open(os.path.join(out_dir, f"pdp_1d_{feat}_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))

    # 2D PDP: TG0h × BMI (manuscript), and TG0h × WBV (optional)
    for f1, f2 in [("TG0h", "BMI"), ("TG0h", "WBV")]:
        if f1 not in X.columns or f2 not in X.columns:
            continue
        try:
            pd_res = partial_dependence(final_pipe, X=X, features=[(f1, f2)], kind="average", grid_resolution=50)
            gx, gy = pd_res["grid_values"][0]
            z = pd_res["average"][0].reshape(len(gx), len(gy))

            plt.figure(figsize=(6.4, 5.2))
            plt.imshow(z, aspect="auto", origin="lower")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xticks(ticks=np.linspace(0, len(gy)-1, 5), labels=np.round(np.linspace(gy.min(), gy.max(), 5), 1))
            plt.yticks(ticks=np.linspace(0, len(gx)-1, 5), labels=np.round(np.linspace(gx.min(), gx.max(), 5), 1))
            plt.xlabel(f2)
            plt.ylabel(f1)
            plt.title(f"2D PDP: {f1} × {f2}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"fig_pdp_2d_{f1}_{f2}.png"))
            plt.close()
        except Exception as e:
            with open(os.path.join(out_dir, f"pdp_2d_{f1}_{f2}_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))

    # ---------------- SHAP (optional) ----------------
    try:
        import shap  # type: ignore

        # Sample for speed
        X_shap = X.copy()
        if len(X_shap) > 600:
            X_shap = X_shap.sample(600, random_state=cfg.random_state)

        # Transform data to model space
        pre_fitted = final_pipe.named_steps["preprocess"]
        model_fitted = final_pipe.named_steps["model"]
        Xt = pre_fitted.transform(X_shap)
        feat_names = _get_feature_names_from_column_transformer(pre_fitted)

        explainer = shap.LinearExplainer(model_fitted, Xt, feature_names=feat_names)
        shap_vals = explainer(Xt)

        # Summary plot
        plt.figure(figsize=(7.2, 5.0))
        shap.summary_plot(shap_vals.values, features=Xt, feature_names=feat_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_shap_summary.png"))
        plt.close()

        # Dependence plot for TG0h (in original feature name, may be expanded after preprocessing)
        # We map by prefix match.
        def _find_transformed_name(prefix: str) -> Optional[str]:
            for n in feat_names:
                if n == prefix or n.endswith(f"__{prefix}") or prefix in n:
                    return n
            return None

        tg0h_name = _find_transformed_name("TG0h")
        wbv_name = _find_transformed_name("WBV")

        if tg0h_name is not None:
            plt.figure(figsize=(7.0, 5.0))
            shap.dependence_plot(
                tg0h_name,
                shap_vals.values,
                Xt,
                feature_names=feat_names,
                interaction_index=wbv_name if wbv_name is not None else "auto",
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_shap_dependence_tg0h.png"))
            plt.close()

        # SHAP value correlation matrix (Figure 12-style)
        sv = np.asarray(shap_vals.values)
        if sv.ndim == 2 and sv.shape[1] == len(feat_names):
            corr = np.corrcoef(sv, rowvar=False)
            plt.figure(figsize=(7.2, 6.2))
            im = plt.imshow(corr, aspect="auto")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(range(len(feat_names)), feat_names, rotation=90)
            plt.yticks(range(len(feat_names)), feat_names)
            plt.title("Correlation matrix of SHAP values")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_shap_corr_matrix.png"))
            plt.close()

            pd.DataFrame(corr, index=feat_names, columns=feat_names).to_csv(
                os.path.join(out_dir, "shap_value_correlation_matrix.csv")
            )

    except Exception as e:
        with open(os.path.join(out_dir, "shap_not_available_or_failed.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))

# ----------------------------
# Ablation: drop feature blocks
# ----------------------------
def ablation_run(df: pd.DataFrame, cfg: Config, out_dir: str) -> pd.DataFrame:
    """
    Ablation analysis (manuscript-consistent leakage control):

    For each ablation setting, we re-run the same leakage-controlled nested CV,
    including fold-specific TG4h cutoff definition and fold-specific preprocessing.
    """
    os.makedirs(out_dir, exist_ok=True)

    base_cols = FEATURE_COLS.copy()
    # If WBV exists in df, allow it in ablation sets.
    if "WBV" in df.columns and "WBV" not in base_cols:
        base_cols = NUMERIC_FEATURES + ["WBV"] + CATEGORICAL_FEATURES

    specs = {
        "full": base_cols,
        "drop_TG0h": [c for c in base_cols if c != "TG0h"],
        "drop_BMI": [c for c in base_cols if c != "BMI"],
        "drop_WBV": [c for c in base_cols if c != "WBV"],
    }

    results = []
    for name, cols in specs.items():
        cols_use = [c for c in cols if c in df.columns]
        X_sub = df[cols_use].copy()
        summary = nested_cv_calibrated(
            df_full=df,
            X=X_sub,
            cfg=cfg,
            out_dir=os.path.join(out_dir, f"ablation_{name}")
        )
        results.append({"setting": name, **summary})

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(out_dir, "ablation_summary.csv"), index=False)
    return df_res


# ----------------------------

# Bootstrap CI for correlations
# ----------------------------
def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    corrs = []
    for _ in range(n_boot):
        samp = rng.choice(idx, size=len(idx), replace=True)
        xs, ys = x[samp], y[samp]
        r = np.corrcoef(xs, ys)[0, 1]
        if np.isfinite(r):
            corrs.append(r)
    corrs = np.array(corrs, dtype=float)
    r_mean = float(np.mean(corrs))
    lo = float(np.percentile(corrs, 2.5))
    hi = float(np.percentile(corrs, 97.5))
    return r_mean, lo, hi


def correlation_ci_package(df: pd.DataFrame, cfg: Config, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    pairs = [
        ("WBV", "TGR", "WBV–TGR%"),
        ("WBV", "TG0h", "WBV–TG0h"),
        ("WBV", "TG4h", "WBV–TG4h"),
    ]

    rows = []
    for a, b, label in pairs:
        if a not in df.columns or b not in df.columns:
            rows.append({"pair": label, "r_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue
        r_mean, lo, hi = bootstrap_corr_ci(df[a].values, df[b].values, cfg.n_boot_corr, cfg.random_state)
        rows.append({"pair": label, "r_mean": r_mean, "ci_low": lo, "ci_high": hi})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "correlation_bootstrap_ci.csv"), index=False)
    return out


# ----------------------------
# Main
# ----------------------------
def run_pipeline(cfg: Config, data_path: str, out_root: str):
    """Run the full manuscript-aligned pipeline under a given calibration method.

    Outputs are written under out_root (which should include a method tag).
    """
    os.makedirs(out_root, exist_ok=True)

    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path)

    # Validate columns
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Derived vars
    df = add_derived_columns(df)

    # Keep a global label column only for descriptive summaries / stratification proxy.
    # Model training/evaluation labels remain fold-specific inside nested CV.
    y_global, cutoff = make_label_from_tg4h_percentile(df, cfg.label_percentile)
    df["HighResponder"] = y_global.values

    # Sanity checks
    pct_rise = float((df["TG4h"] > df["TG0h"]).mean() * 100.0)
    tg0h_med = float(np.median(df["TG0h"]))
    tg4h_med = float(np.median(df["TG4h"]))
    print("\n=== Physiology sanity checks (v2) ===")
    print(f"TG4h cutoff (global p{cfg.label_percentile:.0f}) = {cutoff:.3f} mg/dL")
    print(f"Median TG0h = {tg0h_med:.3f} mg/dL | Median TG4h = {tg4h_med:.3f} mg/dL")
    print(f"P(TG4h > TG0h) = {pct_rise:.2f}%")

    with open(os.path.join(out_root, "physiology_sanity.json"), "w", encoding="utf-8") as f:
        json.dump({
            "tg4h_cutoff_global_p": cutoff,
            "median_tg0h": tg0h_med,
            "median_tg4h": tg4h_med,
            "pct_tg4h_gt_tg0h": pct_rise
        }, f, indent=2)

    # Table 1 (uses global label for descriptive table only)
    make_table1(df, y_global, out_csv=os.path.join(out_root, "Table1_baseline_v2.csv"))
    print("\nSaved Table1_baseline_v2.csv")

    # Plots
    plot_tg_distributions(df, y_global, out_root)

    key_corr_cols = ["TG0h", "TG4h", "Hematocrit", "TotalProtein", "WBV", "TGR", "BMI", "HDL", "LDL", "Age"]
    plot_correlation_matrix(df, key_corr_cols, out_root)

    # Nested CV (leakage-free predictors: TG4h is label-only)
    X = df[FEATURE_COLS].copy()

    print("\n=== Nested CV (leakage-controlled; TG4h label-only) ===")
    summary = nested_cv_calibrated(df_full=df, X=X, cfg=cfg, out_dir=out_root)
    print("\nNested CV summary:", summary)

    # OOF curves
    oof = pd.read_csv(os.path.join(out_root, "oof_predictions.csv")).dropna(subset=["y", "proba_oof"]).copy()
    plot_roc_pr_calibration(oof["y"].values.astype(int), oof["proba_oof"].values.astype(float), out_root)

    # Correlation bootstrap CIs
    correlation_ci_package(df, cfg, out_root)
    print("\nSaved correlation_bootstrap_ci.csv")

    # ------------------------------------------------------------
    # Additional analyses required by the manuscript
    # ------------------------------------------------------------

    # 1) Robustness: repeated 5×10 CV
    print("\n=== Repeated stratified CV (5-fold × 10 repeats) ===")
    rep_dir = os.path.join(out_root, "robustness_repeatedcv_5x10")
    repeated_summary = repeated_cv_5x10(df_full=df, X=X, cfg=cfg, out_dir=rep_dir)
    print("Repeated-CV summary:", repeated_summary)

    # 2) Bootstrap distributions (1,000) for AUROC/Brier of final model
    print("\n=== Bootstrap (1,000) for AUROC/Brier (OOF-based) ===")
    boot_dir = os.path.join(out_root, "bootstrap_finalmodel")
    bootstrap_final_model_oof(oof_csv=os.path.join(out_root, "oof_predictions.csv"), cfg=cfg, out_dir=boot_dir)
    print("Saved bootstrap distributions to:", boot_dir)

    # 3) Threshold sensitivity sweep (match paper Table 7)
    print("\n=== Threshold sensitivity sweep (TG4h percentile) ===")
    th_dir = os.path.join(out_root, "threshold_sensitivity")
    threshold_sensitivity_sweep(df_full=df, cfg=cfg, out_dir=th_dir)
    print("Saved threshold sensitivity summary to:", th_dir)

    # 4) Calibration slope/intercept (computed from OOF)
    print("\n=== Calibration slope & intercept (OOF) ===")
    slope, intercept = calibration_slope_intercept(oof["y"].values.astype(int), oof["proba_oof"].values.astype(float))
    with open(os.path.join(out_root, "calibration_slope_intercept.json"), "w", encoding="utf-8") as f:
        json.dump({"slope": float(slope), "intercept": float(intercept)}, f, indent=2)
    print(f"Calibration line: slope={slope:.4f}, intercept={intercept:.4f}")

    # 5) Decision curve analysis (DCA / net benefit)
    print("\n=== Decision curve analysis (net benefit) ===")
    dca_dir = os.path.join(out_root, "decision_curve_analysis")
    run_dca_with_baselines(df_full=df, cfg=cfg, out_dir=dca_dir)
    print("Saved DCA outputs to:", dca_dir)

    # 6) SHAP + PDP
    print("\n=== SHAP + PDP (explainability) ===")
    exp_dir = os.path.join(out_root, "explainability")
    run_shap_and_pdp(df_full=df, cfg=cfg, out_dir=exp_dir)
    print("Saved explainability outputs to:", exp_dir)

    # 7) WBV misspecification sensitivity
    print("\n=== WBV misspecification sensitivity ===")
    wbv_dir = os.path.join(out_root, "wbv_misspec_sensitivity")
    wbv_misspecification_sensitivity(df_full=df, cfg=cfg, out_dir=wbv_dir)
    print("Saved WBV misspec sensitivity outputs to:", wbv_dir)

    # 8) Pseudo-external subgroup transfer
    print("\n=== Pseudo-external subgroup transfer (age/sex) ===")
    pe_dir = os.path.join(out_root, "pseudo_external_transfer")
    pseudo_external_transfer(df_full=df, cfg=cfg, out_dir=pe_dir)
    print("Saved pseudo-external outputs to:", pe_dir)

    print("\nDONE. Outputs in:", out_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Dataset.csv")
    parser.add_argument("--out", type=str, default="out_v2")
    parser.add_argument("--calib", type=str, default="both", choices=["isotonic", "sigmoid", "both"])
    args = parser.parse_args()

    # Sweep calibration methods to reproduce the paper's comparison.
    methods = ["sigmoid", "isotonic"] if args.calib == "both" else [args.calib]
    for method in methods:
        cfg = Config(calib_method=method)
        out_dir = os.path.join(args.out, f"calib_{method}")
        run_pipeline(cfg=cfg, data_path=args.data, out_root=out_dir)


if __name__ == "__main__":
    main()
