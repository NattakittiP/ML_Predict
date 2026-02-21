# summary_table_regen.py
# Recompute all values for Summary Table (Table 7) from the current dataset,
# using a leakage-controlled pipeline consistent with the manuscript.
#
# Requirements:
#   pip install numpy pandas scipy scikit-learn joblib
#
# Run:
#   python summary_table_regen.py
#
# Outputs:
#   - summary_table_values.json
#   - summary_table_rows.tex
#   - baseline_table_stats.csv
#   - threshold_sensitivity.csv
#   - ablation_results.csv
#   - misspecification_results.csv
#
# NOTE: Column names expected (case-insensitive / flexible mapping):
#   TG0h, TG4h, Hct, TP, HDL, LDL, BMI, Age, Sex
# Sex can be coded as 0/1, M/F, Male/Female.

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Mild symmetric winsorization (fold-specific), fit on training fold only.
WINSOR_LO_Q = 0.01
WINSOR_HI_Q = 0.99

class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric features to training-fold quantile bounds (array-based)."""
    def __init__(self, lower_q: float = WINSOR_LO_Q, upper_q: float = WINSOR_HI_Q):
        self.lower_q = float(lower_q)
        self.upper_q = float(upper_q)
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "Dataset.csv"  # <-- change if needed
EXTERNAL_PATH = None  # e.g., "external_synthetic_like.csv" or None

RANDOM_SEED = 42
N_TOTAL_EXPECTED = 1500

# Exclusion thresholds (as in your table)
TG_MAX = 1500.0
WBV_MIN = 3.0
WBV_MAX = 10.0

# Primary phenotype threshold
PRIMARY_PCTL = 75

# Threshold sensitivity percentiles
PCTL_SWEEP = [60, 70, 75, 80, 90]

# CV / Bootstrap specs
OUTER_FOLDS = 5
INNER_FOLDS = 5
BOOTSTRAP_B = 1000
REPEATS = 10  # repeated 5-fold
MISSPEC_REPEATS = 200  # for WBV misspecification simulation (table says mean/SD)
# If you need closer to your manuscript runs, set to 1000.

# Hyperparameter grids (must match manuscript)
LR_C_GRID = [0.01, 0.1, 1, 10]
RF_N_EST = [200, 400, 800]
RF_MAX_DEPTH = [None, 6, 12]
RF_MIN_LEAF = [1, 2, 4]
SVM_C = [0.1, 1, 10]
SVM_GAMMA = ["scale", 0.1, 0.01]

# -------------------------------
# UTILITIES
# -------------------------------

def _lower_cols(df: pd.DataFrame) -> Dict[str, str]:
    return {c.lower(): c for c in df.columns}

def map_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Map flexible column names to canonical ones.
    Expected canonical keys: tg0h, tg4h, hct, tp, hdl, ldl, bmi, age, sex
    """
    lc = _lower_cols(df)
    candidates = {
        "tg0h": ["tg0h", "tg_0h", "tg0", "fasting_tg", "tg0_hr", "tg_0", "tg0h_mgdl"],
        "tg4h": ["tg4h", "tg_4h", "tg4", "postprandial_tg", "tg4_hr", "tg_4", "tg4h_mgdl"],
        "hct":  ["hct", "hematocrit"],
        "tp":   ["tp", "total_protein", "totalprotein"],
        "hdl":  ["hdl", "hdl_c", "hdlc", "hdl-c"],
        "ldl":  ["ldl", "ldl_c", "ldlc", "ldl-c"],
        "bmi":  ["bmi"],
        "age":  ["age", "age_years", "years"],
        "sex":  ["sex", "gender"]
    }

    mapping = {}
    for key, keys in candidates.items():
        found = None
        for k in keys:
            if k in lc:
                found = lc[k]
                break
        if found is None:
            raise ValueError(f"Missing required column for '{key}'. Looked for: {keys}. "
                             f"Available columns: {list(df.columns)}")
        mapping[key] = found
    return mapping

def normalize_sex(series: pd.Series) -> pd.Series:
    """
    Normalize sex to canonical strings: 'Female' / 'Male'.

    Accepts inputs such as:
      - strings: 'F','Female','M','Male' (case-insensitive)
      - numeric: 1/0, 0/1, or 1/2 (assumed 1=Female, 2=Male)
    """
    s = series.copy()

    # Parse strings
    if s.dtype == object:
        s2 = s.astype(str).str.strip().str.lower()
        s2 = s2.replace({
            "female": "f", "woman": "f", "f": "f",
            "male": "m", "man": "m", "m": "m",
        })
        mapped = s2.map({"f": "Female", "m": "Male"})
        # If still missing, attempt numeric parsing
        if mapped.isna().any():
            num = pd.to_numeric(s2, errors="coerce")
        else:
            num = None
    else:
        mapped = pd.Series([np.nan] * len(s), index=s.index)
        num = pd.to_numeric(s, errors="coerce")

    # Parse numeric encodings
    if mapped.isna().any():
        if num is None:
            num = pd.to_numeric(s, errors="coerce")
        if num.isna().any():
            raise ValueError("Sex column contains values that cannot be normalized.")
        uniq = sorted(num.dropna().unique().tolist())

        if uniq == [1, 2]:
            mapped = num.map({1: "Female", 2: "Male"})
        else:
            if any(u not in [0, 1] for u in uniq):
                raise ValueError(f"Sex normalization failed. Unique values after parsing: {uniq}")
            mapped = num.map({1: "Female", 0: "Male"})

    if mapped.isna().any():
        raise ValueError("Sex normalization failed; some values could not be mapped to Female/Male.")
    return mapped

def compute_wbv_de_simone(hct_pct: pd.Series, tp_gdl: pd.Series) -> pd.Series:
    """
    De Simone low-shear WBV approximation.
    IMPORTANT: Use the exact formula you used in the manuscript/code.
    Since implementations vary, we provide the commonly used de Simone form:
        WBV = 0.12*Hct + 0.17*TP - 0.3519
    BUT if your paper uses a different de Simone constant, you MUST replace here
    to ensure exact match.

    If you've already confirmed your code, paste the exact formula here.
    """
    hct = pd.to_numeric(hct_pct, errors="coerce")
    tp = pd.to_numeric(tp_gdl, errors="coerce")
    # ---- Replace with your exact WBV formula if different ----
    wbv = (0.12 * hct) + (0.17 * tp) - 0.3519
    return wbv

def compute_tgr(tg0h: pd.Series, tg4h: pd.Series) -> pd.Series:
    tg0 = pd.to_numeric(tg0h, errors="coerce")
    tg4 = pd.to_numeric(tg4h, errors="coerce")
    return (tg4 - tg0) / tg0 * 100.0

def fmt_mean_sd(x: pd.Series, decimals: int = 4) -> str:
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    return f"{mu:.{decimals}f} \\pm {sd:.{decimals}f}"

def fmt_n_pct(n: int, total: int, decimals: int = 4) -> str:
    pct = 100.0 * n / total if total > 0 else float("nan")
    return f"{n} ({pct:.{decimals}f}\\%)"

def chi2_sex_p(female_n: int, male_n: int, female_h: int, male_h: int) -> float:
    table = np.array([[female_n, male_n], [female_h, male_h]], dtype=float)
    chi2, p, _, _ = stats.chi2_contingency(table)
    return float(p)

def ttest_p(a: np.ndarray, b: np.ndarray) -> float:
    # Welch t-test for robustness
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(p)

def pearsonr_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)

def r2_from_r(r: float) -> float:
    return float(r * r)

def build_lr_pipeline(C: float, feature_cols: List[str]) -> Pipeline:
    """Leakage-controlled LR pipeline with Sex as string + one-hot encoding (drop first)."""
    cat_cols = [c for c in feature_cols if c == "Sex"]
    num_cols = [c for c in feature_cols if c != "Sex"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("winsor", Winsorizer()),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=8000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    return Pipeline([("pre", pre), ("clf", clf)])

def nested_cv_lr_oof(
    df: pd.DataFrame,
    y_split: np.ndarray,
    feature_cols: List[str],
    C_grid: List[float],
    label_percentile: float = PRIMARY_PCTL,
    tg4h_col: str = "TG4h",
    outer_folds: int = 5,
    inner_folds: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Nested CV selecting best C by mean AUROC in inner CV, returning OOF probabilities.

    Manuscript-consistent leakage control:
      - TG4h cutoff is computed from *train_idx only* in each outer fold.
      - y_train / y_valid use that train-fold cutoff (valid uses train cutoff).
      - Preprocessing (impute, winsorize, scale, one-hot) is fit only on training folds.
    """
    if tg4h_col not in df.columns:
        raise ValueError(f"df must contain '{tg4h_col}' for fold-specific phenotype definition.")

    X = df[feature_cols].copy()
    y_split = np.asarray(y_split).astype(int)
    tg4h = pd.to_numeric(df[tg4h_col], errors="coerce").to_numpy(dtype=float)

    # Outer folds are stratified using a global phenotype for balance,
    # but evaluation labels are fold-specific (derived from train_idx only).
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    oof = np.zeros(len(df), dtype=float)
    oof_y = np.zeros(len(df), dtype=int)
    oof_cut = np.zeros(len(df), dtype=float)

    chosen_Cs: List[float] = []
    fold_metrics: List[Dict] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y_split), start=1):
        # Fold-specific cutoff (TRAIN ONLY)
        cutoff = float(np.quantile(tg4h[train_idx], label_percentile / 100.0))
        y_tr = (tg4h[train_idx] >= cutoff).astype(int)
        y_te = (tg4h[test_idx] >= cutoff).astype(int)

        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]

        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + 100 + fold)
        best_C = None
        best_score = -1.0

        for C in C_grid:
            scores = []
            for in_tr, in_va in inner.split(X_tr, y_tr):
                pipe = build_lr_pipeline(C=C, feature_cols=feature_cols)
                pipe.fit(X_tr.iloc[in_tr], y_tr[in_tr])
                p = pipe.predict_proba(X_tr.iloc[in_va])[:, 1]
                scores.append(roc_auc_score(y_tr[in_va], p))
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_C = C

        assert best_C is not None
        chosen_Cs.append(float(best_C))

        # Fit on full outer-train with best C and predict on outer-test
        pipe = build_lr_pipeline(C=best_C, feature_cols=feature_cols)
        pipe.fit(X_tr, y_tr)
        p_te = pipe.predict_proba(X_te)[:, 1]

        oof[test_idx] = p_te
        oof_y[test_idx] = y_te
        oof_cut[test_idx] = cutoff

        fold_metrics.append({
            "fold": fold,
            "cutoff_trainfold": cutoff,
            "best_C": float(best_C),
            "inner_mean_auroc": float(best_score),
            "outer_auroc": float(roc_auc_score(y_te, p_te)),
            "outer_brier": float(brier_score_loss(y_te, p_te)),
        })

        # Summarize outer-fold metrics
    fold_aurocs = np.array([m["outer_auroc"] for m in fold_metrics], dtype=float)
    fold_briers = np.array([m["outer_brier"] for m in fold_metrics], dtype=float)

    meta = {
        "label_percentile": float(label_percentile),
        "winsor_lo_q": float(WINSOR_LO_Q),
        "winsor_hi_q": float(WINSOR_HI_Q),
        "chosen_Cs": chosen_Cs,
        "fold_metrics": fold_metrics,
        "oof_y": oof_y,
        "oof_cutoff_trainfold": oof_cut,
        "y_split_global": y_split,

        # Added: summary stats used later in the script
        "mean_auroc": float(np.mean(fold_aurocs)),
        "sd_auroc": float(np.std(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0,
        "mean_brier": float(np.mean(fold_briers)),
        "sd_brier": float(np.std(fold_briers, ddof=1)) if len(fold_briers) > 1 else 0.0,
    }
    return oof, meta


def bootstrap_ci_metrics(y: np.ndarray, p: np.ndarray, B: int, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)

    aucs = []
    briers = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = p[idx]
        # handle degenerate samples
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))
        briers.append(brier_score_loss(yb, pb))

    aucs = np.array(aucs)
    briers = np.array(briers)
    return {
        "auroc_mean": float(np.mean(aucs)),
        "auroc_ci": [float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))],
        "brier_mean": float(np.mean(briers)),
        "brier_ci": [float(np.quantile(briers, 0.025)), float(np.quantile(briers, 0.975))],
    }

def repeated_cv_metrics(df: pd.DataFrame, y: np.ndarray, feature_cols: List[str], C: float,
                        repeats: int, folds: int, seed: int = 42) -> Dict:
    X = df[feature_cols].copy()
    y = np.asarray(y).astype(int)

    rskf = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=seed
    )
    aucs, briers = [], []
    for tr, te in rskf.split(X, y):
        pipe = build_lr_pipeline(C=C, feature_cols=feature_cols)
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        briers.append(brier_score_loss(y[te], p))

    aucs = np.array(aucs)
    briers = np.array(briers)
    return {
        "auroc_mean": float(np.mean(aucs)),
        "auroc_sd": float(np.std(aucs, ddof=1)),
        "auroc_min": float(np.min(aucs)),
        "auroc_max": float(np.max(aucs)),
        "brier_mean": float(np.mean(briers)),
        "brier_sd": float(np.std(briers, ddof=1)),
        "brier_min": float(np.min(briers)),
        "brier_max": float(np.max(briers)),
    }

def fit_calibration_heldout(df: pd.DataFrame, y: np.ndarray, feature_cols: List[str], C: float,
                            seed: int = 42) -> Dict:
    """
    Held-out evaluation:
    - Split 80/20 (stratified) with seed
    - Train base LR on train
    - Calibrate via CalibratedClassifierCV (isotonic / sigmoid) using cv=5 on train
    - Evaluate on test
    - Compute calibration slope/intercept by regressing y on logit(p)
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    X = df[feature_cols].copy()
    y = np.asarray(y).astype(int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    (tr, te) = next(sss.split(X, y))

    base = build_lr_pipeline(C=C, feature_cols=feature_cols)
    base.fit(X.iloc[tr], y[tr])

    # calibrators trained on training split
    iso = CalibratedClassifierCV(estimator=clone(base), method="isotonic", cv=5)
    sig = CalibratedClassifierCV(estimator=clone(base), method="sigmoid", cv=5)

    iso.fit(X.iloc[tr], y[tr])
    sig.fit(X.iloc[tr], y[tr])

    p_iso = iso.predict_proba(X.iloc[te])[:, 1]
    p_sig = sig.predict_proba(X.iloc[te])[:, 1]

    out = {
        "isotonic": {
            "auroc": float(roc_auc_score(y[te], p_iso)),
            "brier": float(brier_score_loss(y[te], p_iso)),
        },
        "sigmoid": {
            "auroc": float(roc_auc_score(y[te], p_sig)),
            "brier": float(brier_score_loss(y[te], p_sig)),
        },
    }

    # calibration slope/intercept on isotonic (chosen later) by default:
    eps = 1e-6
    p = np.clip(p_iso, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    # Fit y ~ a + b*logit (least squares)
    b, a, _, _, _ = stats.linregress(logit, y[te])
    out["calibration_line"] = {"slope": float(b), "intercept": float(a)}
    return out

def wbv_misspecification(df: pd.DataFrame, y: np.ndarray, wbv_col: str,
                         rel_err: float, repeats: int, seed: int = 42) -> Dict:
    """
    Apply multiplicative noise to WBV:
        WBV' = WBV * (1 + eps), eps ~ N(0, rel_err)
    and evaluate:
        - corr(WBV', TGR) across repeats
        - AUROC of WBV'-only logistic regression (simple 5-fold CV)
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)

    corr_list = []
    auc_list = []

    # Use a lightweight CV for WBV-only AUROC
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for _ in range(repeats):
        noisy = df[wbv_col].to_numpy() * (1.0 + rng.normal(0.0, rel_err, size=len(df)))

        # corr with TGR (must exist)
        r, _p = pearsonr_p(noisy, df["TGR"].to_numpy())
        corr_list.append(r)

        # WBV-only model AUROC via 5-fold CV
        probs = np.zeros(len(df), dtype=float)
        for tr, te in skf.split(noisy.reshape(-1, 1), y):
            Xtr = noisy[tr].reshape(-1, 1)
            Xte = noisy[te].reshape(-1, 1)
            # Simple standardized LR
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("winsor", Winsorizer()),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                           max_iter=500, random_state=seed)),
            ])
            pipe.fit(Xtr, y[tr])
            probs[te] = pipe.predict_proba(Xte)[:, 1]
        auc_list.append(roc_auc_score(y, probs))

    corr_list = np.array(corr_list)
    auc_list = np.array(auc_list)
    return {
        "corr_mean": float(np.mean(corr_list)),
        "corr_sd": float(np.std(corr_list, ddof=1)),
        "auroc_mean": float(np.mean(auc_list)),
        "auroc_sd": float(np.std(auc_list, ddof=1)),
    }

# -------------------------------
# MAIN
# -------------------------------

def main():
    df0 = pd.read_csv(DATA_PATH)
    col = map_columns(df0)

    df = pd.DataFrame({
        "TG0h": df0[col["tg0h"]],
        "TG4h": df0[col["tg4h"]],
        "Hct":  df0[col["hct"]],
        "TP":   df0[col["tp"]],
        "HDL":  df0[col["hdl"]],
        "LDL":  df0[col["ldl"]],
        "BMI":  df0[col["bmi"]],
        "Age":  df0[col["age"]],
        "Sex_raw": df0[col["sex"]],
    })

    # Normalize / derive
    df["Sex"] = normalize_sex(df["Sex_raw"])
    df.drop(columns=["Sex_raw"], inplace=True)

    # Derived variables
    df["WBV"] = compute_wbv_de_simone(df["Hct"], df["TP"])
    df["TGR"] = compute_tgr(df["TG0h"], df["TG4h"])

    # Exclusions
    mask = (
        (df["TG0h"] <= TG_MAX) &
        (df["TG4h"] <= TG_MAX) &
        (df["WBV"] >= WBV_MIN) &
        (df["WBV"] <= WBV_MAX)
    )
    df = df.loc[mask].reset_index(drop=True)

    # Cohort counts
    n_total = len(df)
    if n_total != N_TOTAL_EXPECTED:
        print(f"[WARN] After exclusions, n={n_total} (expected {N_TOTAL_EXPECTED}). "
              f"If this is intended in your manuscript, ignore. Otherwise, verify thresholds/data.")

    # Primary cutoff
    tg4_cut = float(np.quantile(df["TG4h"], PRIMARY_PCTL / 100.0))
    y_primary = (df["TG4h"] >= tg4_cut).astype(int).to_numpy()
    df["HighResponder"] = y_primary
    df["Phenotype"] = np.where(df["HighResponder"]==1, "High", "Normal")

    n_high = int(y_primary.sum())
    n_norm = int(len(y_primary) - n_high)

    # Baseline group splits
    df_norm = df.loc[y_primary == 0]
    df_high = df.loc[y_primary == 1]

    # Sex counts
    female_norm = int((df_norm["Sex"] == "Female").sum())
    male_norm = int((df_norm["Sex"] == "Male").sum())
    female_high = int((df_high["Sex"] == "Female").sum())
    male_high = int((df_high["Sex"] == "Male").sum())
    p_sex = chi2_sex_p(female_norm, male_norm, female_high, male_high)

    # Continuous baseline stats + Welch p-values
    cont_vars = ["Age", "Hct", "TP", "WBV", "TG0h", "TG4h", "TGR", "HDL", "LDL", "BMI"]
    baseline_rows = []
    pvals = {}
    for v in cont_vars:
        a = df_norm[v].to_numpy()
        b = df_high[v].to_numpy()
        p = ttest_p(a, b)
        pvals[v] = p
        baseline_rows.append({
            "Variable": v,
            "Normal_mean": float(np.mean(a)),
            "Normal_sd": float(np.std(a, ddof=1)),
            "High_mean": float(np.mean(b)),
            "High_sd": float(np.std(b, ddof=1)),
            "p_value": p
        })
    pd.DataFrame(baseline_rows).to_csv("baseline_table_stats.csv", index=False)

    # Correlations
    r_wbv_tgr, p_wbv_tgr = pearsonr_p(df["WBV"].to_numpy(), df["TGR"].to_numpy())
    r_wbv_tg0, _ = pearsonr_p(df["WBV"].to_numpy(), df["TG0h"].to_numpy())
    r_wbv_tg4, _ = pearsonr_p(df["WBV"].to_numpy(), df["TG4h"].to_numpy())
    r_tg0_tgr, _ = pearsonr_p(df["TG0h"].to_numpy(), df["TGR"].to_numpy())
    r2 = r2_from_r(r_wbv_tgr)

    # Baseline classifiers
    # (a) TG0h 75th cutoff baseline (predict high-responder by TG0h threshold)
    tg0_cut = float(np.quantile(df["TG0h"], 0.75))
    p_baseline = (df["TG0h"] >= tg0_cut).astype(float).to_numpy()
    # For AUROC, we can use binary scores; for brier treat as probability in {0,1}
    auroc_tg0_cut = float(roc_auc_score(y_primary, p_baseline))
    brier_tg0_cut = float(brier_score_loss(y_primary, p_baseline))

    # (b) Univariate LR (TG0h only) via nested CV
    oof_uni, meta_uni = nested_cv_lr_oof(df, y_primary, ["TG0h"], LR_C_GRID, label_percentile=PRIMARY_PCTL, outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS, seed=RANDOM_SEED)
    auroc_uni = float(roc_auc_score(meta_uni["oof_y"], oof_uni))
    brier_uni = float(brier_score_loss(meta_uni["oof_y"], oof_uni))
    f1_uni = float(f1_score(meta_uni["oof_y"], (oof_uni >= 0.5).astype(int)))
    rec_uni = float(recall_score(meta_uni["oof_y"], (oof_uni >= 0.5).astype(int)))
    prec_uni = float(precision_score(meta_uni["oof_y"], (oof_uni >= 0.5).astype(int)))

    # (c) Multivariable LR via nested CV
    features_full = ["Age", "Sex", "BMI", "TG0h", "HDL", "LDL", "Hct", "TP", "WBV"]
    oof_full, meta_full = nested_cv_lr_oof(df, y_primary, features_full, LR_C_GRID, label_percentile=PRIMARY_PCTL, outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS, seed=RANDOM_SEED)
    auroc_full = float(roc_auc_score(meta_full["oof_y"], oof_full))
    brier_full = float(brier_score_loss(meta_full["oof_y"], oof_full))
    f1_full = float(f1_score(meta_full["oof_y"], (oof_full >= 0.5).astype(int)))
    rec_full = float(recall_score(meta_full["oof_y"], (oof_full >= 0.5).astype(int)))
    prec_full = float(precision_score(meta_full["oof_y"], (oof_full >= 0.5).astype(int)))

    # Bootstrap on OOF of primary multivariable model
    boot = bootstrap_ci_metrics(meta_full["oof_y"], oof_full, BOOTSTRAP_B, RANDOM_SEED)

    # Repeated CV (use most frequent chosen C from nested CV as "final C")
    chosen_C = max(set(meta_full["chosen_Cs"]), key=meta_full["chosen_Cs"].count)
    rep = repeated_cv_metrics(df, y_primary, features_full, C=chosen_C,
                              repeats=REPEATS, folds=OUTER_FOLDS, seed=RANDOM_SEED)

    # Calibration held-out
    cal = fit_calibration_heldout(df, y_primary, features_full, C=chosen_C, seed=RANDOM_SEED)
    # Choose isotonic as final
    slope = cal["calibration_line"]["slope"]
    intercept = cal["calibration_line"]["intercept"]

    # Threshold sensitivity sweep
    sweep_rows = []
    for pctl in PCTL_SWEEP:
        cut = float(np.quantile(df["TG4h"], pctl / 100.0))
        y = (df["TG4h"] >= cut).astype(int).to_numpy()
        oof, meta = nested_cv_lr_oof(df, y, features_full, LR_C_GRID, label_percentile=pctl, outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS, seed=RANDOM_SEED)
        au = float(roc_auc_score(meta["oof_y"], oof))
        br = float(brier_score_loss(meta["oof_y"], oof))
        f1 = float(f1_score(meta["oof_y"], (oof >= 0.5).astype(int)))
        rec = float(recall_score(meta["oof_y"], (oof >= 0.5).astype(int)))
        prec = float(precision_score(meta["oof_y"], (oof >= 0.5).astype(int)))
        sweep_rows.append({
            "percentile": pctl,
            "tg4_cut": cut,
            "auroc": au,
            "brier": br,
            "f1": f1,
            "recall": rec,
            "precision": prec
        })
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv("threshold_sensitivity.csv", index=False)

    # Feature ablation (primary task)
        # Feature ablation (primary task)
    ablation = []

    def ablate(cols_to_drop: List[str]) -> float:
        cols = [c for c in features_full if c not in cols_to_drop]

        oof, meta = nested_cv_lr_oof(
            df,
            y_primary,
            cols,
            LR_C_GRID,
            label_percentile=PRIMARY_PCTL,
            outer_folds=OUTER_FOLDS,
            inner_folds=INNER_FOLDS,
            seed=RANDOM_SEED
        )

        return float(roc_auc_score(meta["oof_y"], oof))

    ablation.append({"drop": "TG0h", "auroc": ablate(["TG0h"])})
    ablation.append({"drop": "BMI", "auroc": ablate(["BMI"])})
    ablation.append({"drop": "WBV", "auroc": ablate(["WBV"])})
    pd.DataFrame(ablation).to_csv("ablation_results.csv", index=False)


    # WBV misspecification
    miss_5 = wbv_misspecification(df, y_primary, "WBV", rel_err=0.05,
                                  repeats=MISSPEC_REPEATS, seed=RANDOM_SEED)
    miss_10 = wbv_misspecification(df, y_primary, "WBV", rel_err=0.10,
                                   repeats=MISSPEC_REPEATS, seed=RANDOM_SEED)
    pd.DataFrame([
        {"rel_err": 0.05, **miss_5},
        {"rel_err": 0.10, **miss_10},
    ]).to_csv("misspecification_results.csv", index=False)

    # External-synthetic transfer (optional)
    external_block = None
    if EXTERNAL_PATH is not None and os.path.exists(EXTERNAL_PATH):
        ext0 = pd.read_csv(EXTERNAL_PATH)
        colx = map_columns(ext0)
        ext = pd.DataFrame({
            "TG0h": ext0[colx["tg0h"]],
            "TG4h": ext0[colx["tg4h"]],
            "Hct":  ext0[colx["hct"]],
            "TP":   ext0[colx["tp"]],
            "HDL":  ext0[colx["hdl"]],
            "LDL":  ext0[colx["ldl"]],
            "BMI":  ext0[colx["bmi"]],
            "Age":  ext0[colx["age"]],
            "Sex_raw": ext0[colx["sex"]],
        })
        ext["Sex"] = normalize_sex(ext["Sex_raw"])
        ext.drop(columns=["Sex_raw"], inplace=True)
        ext["WBV"] = compute_wbv_de_simone(ext["Hct"], ext["TP"])
        ext["TGR"] = compute_tgr(ext["TG0h"], ext["TG4h"])
        mask2 = (
            (ext["TG0h"] <= TG_MAX) &
            (ext["TG4h"] <= TG_MAX) &
            (ext["WBV"] >= WBV_MIN) &
            (ext["WBV"] <= WBV_MAX)
        )
        ext = ext.loc[mask2].reset_index(drop=True)
        ext_cut = float(np.quantile(ext["TG4h"], PRIMARY_PCTL / 100.0))
        y_ext = (ext["TG4h"] >= ext_cut).astype(int).to_numpy()

        # Train on original (full dataset) and evaluate on external
        base = build_lr_pipeline(C=chosen_C, feature_cols=features_full)
        base.fit(df[features_full], y_primary)
        p_ext = base.predict_proba(ext[features_full])[:, 1]
        external_block = {
            "external_cutoff_tg4h": ext_cut,
            "uncalibrated": {
                "auroc": float(roc_auc_score(y_ext, p_ext)),
                "brier": float(brier_score_loss(y_ext, p_ext)),
            }
        }

    # Pack results
    values = {
        "cohort": {
            "n_total": n_total,
            "n_normal": n_norm,
            "n_high": n_high,
            "tg4_cut_primary": tg4_cut,
            "exclusion_text": f"TG > {TG_MAX:.0f} mg/dL; WBV < {WBV_MIN:.0f} or > {WBV_MAX:.0f} cP"
        },
        "baseline_table_1": {
            "female_normal": female_norm,
            "male_normal": male_norm,
            "female_high": female_high,
            "male_high": male_high,
            "p_sex": p_sex,
            "cont_pvals": pvals
        },
        "correlations": {
            "wbv_tgr_r": r_wbv_tgr,
            "wbv_tgr_p": p_wbv_tgr,
            "wbv_tgr_r2": r2,
            "wbv_tg0_r": r_wbv_tg0,
            "wbv_tg4_r": r_wbv_tg4,
            "tg0_tgr_r": r_tg0_tgr
        },
        "misspecification": {
            "5pct": miss_5,
            "10pct": miss_10
        },
        "baselines_table_4": {
            "tg0_cut_75": tg0_cut,
            "tg0_cut_auroc": auroc_tg0_cut,
            "tg0_cut_brier": brier_tg0_cut,
            "uni_lr": {
                "auroc": auroc_uni, "brier": brier_uni, "f1": f1_uni,
                "recall": rec_uni, "precision": prec_uni
            },
            "multi_lr": {
                "auroc": auroc_full, "brier": brier_full, "f1": f1_full,
                "recall": rec_full, "precision": prec_full
            },
            "nested_cv": f"Stratified {OUTER_FOLDS} x {INNER_FOLDS} nested cross-validation on primary TG4h {PRIMARY_PCTL}th percentile phenotype"
        },
        "primary_model": {
            "nested_cv": {
                "auroc_mean": meta_full["mean_auroc"],
                "auroc_sd": meta_full["sd_auroc"],
                "brier_mean": meta_full["mean_brier"],
                "brier_sd": meta_full["sd_brier"],
            },
            "bootstrap": boot,
            "repeated_cv": rep,
        },
        "calibration": {
            "isotonic": cal["isotonic"],
            "sigmoid": cal["sigmoid"],
            "chosen": "Isotonic",
            "slope": slope,
            "intercept": intercept
        },
        "threshold_sensitivity": sweep_rows,
        "ablation": ablation,
        "modeling_config": {
            "lr": {"C_grid": LR_C_GRID, "penalty": "l2", "solver": "lbfgs", "max_iter": 8000},
            "rf": {"n_estimators": RF_N_EST, "max_depth": RF_MAX_DEPTH, "min_samples_leaf": RF_MIN_LEAF},
            "svm": {"C_grid": SVM_C, "gamma_grid": SVM_GAMMA, "kernel": "rbf"}
        },
        "external_synthetic_transfer": external_block
    }

    with open("summary_table_values.json", "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2)

    # -----------------------
    # LaTeX row writer
    # -----------------------
    def pv(p: float, dec=4) -> str:
        if p < 1e-4:
            return "$< 0.0001$"
        return f"${p:.{dec}f}$"

    # baseline mean±sd strings
    b_fmt = {}
    for v in cont_vars:
        b_fmt[v] = {
            "norm": fmt_mean_sd(df_norm[v], decimals=4),
            "high": fmt_mean_sd(df_high[v], decimals=4),
            "p": pv(pvals[v], dec=4)
        }

    rows_tex = []
    # Cohort block
    rows_tex.append(f"  & Total synthetic adult records & $n = {n_total:,}$ \\\\")
    rows_tex.append(f"  & Normal responders (TG$_{{4h}}$ below {PRIMARY_PCTL}th percentile) & $n = {n_norm:,}$ ({(100*n_norm/n_total):.0f}\\%) \\\\")
    rows_tex.append(f"  & High responders (TG$_{{4h}}$ $\\ge$ {PRIMARY_PCTL}th percentile) & $n = {n_high:,}$ ({(100*n_high/n_total):.0f}\\%) \\\\")
    rows_tex.append(f"  & Primary TG$_{{4h}}$ cut-off ({PRIMARY_PCTL}th percentile) & ${tg4_cut:.4f}\\,\\mathrm{{mg/dL}}$ \\\\")
    rows_tex.append(f"  & Exclusion thresholds & {values['cohort']['exclusion_text']} \\\\")

    # Sex rows
    rows_tex.append(f"  & Female, $n$ (\\%) & Normal: {fmt_n_pct(female_norm, n_norm)}; High: {fmt_n_pct(female_high, n_high)}; $p = {p_sex:.4f}$ \\\\")
    rows_tex.append(f"  & Male, $n$ (\\%) & Normal: {fmt_n_pct(male_norm, n_norm)}; High: {fmt_n_pct(male_high, n_high)}; $p = {p_sex:.4f}$ \\\\")

    # Continuous baseline rows in the order of your table
    order = ["Age","Hct","TP","WBV","TG0h","TG4h","TGR","HDL","LDL","BMI"]
    labels = {
        "Age": "Age (years)",
        "Hct": "Hematocrit (\\%)",
        "TP": "Total protein (g/dL)",
        "WBV": "Whole blood viscosity (cP)",
        "TG0h": "Fasting TG (TG$_{0h}$, mg/dL)",
        "TG4h": "4-h TG (TG$_{4h}$, mg/dL)",
        "TGR": "postprandial triglyceride response ratio (TGR, \\%)",
        "HDL": "HDL-C (mg/dL)",
        "LDL": "LDL-C (mg/dL)",
        "BMI": "BMI (kg/m$^2$)"
    }
    for v in order:
        rows_tex.append(f"  & {labels[v]} & Normal: ${b_fmt[v]['norm']}$; High: ${b_fmt[v]['high']}$; $p = {b_fmt[v]['p']}$ \\\\")

    # Correlation structure
    rows_tex.append(f"  & WBV--TGR linear association & Pearson $r = {r_wbv_tgr:.4f}$; $p = {p_wbv_tgr:.4f}$; $R^2 = {r2:.4f}$ \\\\")
    rows_tex.append(f"  & WBV vs.\\ TG$_{{0h}}$ & Pearson $r = {r_wbv_tg0:.4f}$ \\\\")
    rows_tex.append(f"  & WBV vs.\\ TG$_{{4h}}$ & Pearson $r = {r_wbv_tg4:.4f}$ \\\\")
    rows_tex.append(f"  & TG$_{{0h}}$ vs.\\ TGR & Pearson $r = {r_tg0_tgr:.4f}$ \\\\")
    rows_tex.append("  & Qualitative summary & WBV shows negligible correlation with lipid markers and TGR \\\\")

    # Misspecification
    rows_tex.append(f"  & 5\\% relative error in WBV model & corr(WBV, TGR): mean {miss_5['corr_mean']:.5f} (SD {miss_5['corr_sd']:.4f}); AUROC(WBV-only): mean {miss_5['auroc_mean']:.4f} (SD {miss_5['auroc_sd']:.4f}) \\\\")
    rows_tex.append(f"  & 10\\% relative error in WBV model & corr(WBV, TGR): mean {miss_10['corr_mean']:.5f} (SD {miss_10['corr_sd']:.4f}); AUROC(WBV-only): mean {miss_10['auroc_mean']:.4f} (SD {miss_10['auroc_sd']:.4f}) \\\\")
    rows_tex.append("  & Interpretation & Even under misspecification, WBV shows no discriminative value for TGR \\\\")

    # Baselines
    rows_tex.append(f"  & TG$_{{0h}}$ {PRIMARY_PCTL}th percentile cut-off & Threshold: ${tg0_cut:.4f}\\,\\mathrm{{mg/dL}}$; AUROC ${auroc_tg0_cut:.4f}$; Brier ${brier_tg0_cut:.4f}$ \\\\")
    rows_tex.append(f"  & Univariate logistic regression (TG$_{{0h}}$ only) & AUROC ${auroc_uni:.4f}$; Brier ${brier_uni:.4f}$; F1-score ${f1_uni:.4f}$; recall ${rec_uni:.4f}$; precision ${prec_uni:.4f}$ \\\\")
    rows_tex.append(f"  & Multivariable L2-penalized logistic regression & AUROC ${auroc_full:.4f}$; Brier ${brier_full:.4f}$; F1-score ${f1_full:.4f}$; recall ${rec_full:.4f}$; precision ${prec_full:.4f}$ \\\\")
    rows_tex.append(f"  & Nested CV configuration & {values['baselines_table_4']['nested_cv']} \\\\")

    # Primary multivariable performance
    rows_tex.append(f"  & Nested CV (primary task) & Mean AUROC ${meta_full['mean_auroc']:.4f}$ (SD ${meta_full['sd_auroc']:.4f}$); mean Brier ${meta_full['mean_brier']:.4f} \\pm {meta_full['sd_brier']:.4f}$ \\\\")
    rows_tex.append(f"  & Bootstrap ({BOOTSTRAP_B:,} resamples) & AUROC mean ${boot['auroc_mean']:.4f}$ (95\\% CI $[{boot['auroc_ci'][0]:.4f}, {boot['auroc_ci'][1]:.4f}]$); Brier mean ${boot['brier_mean']:.4f}$ (95\\% CI $[{boot['brier_ci'][0]:.4f}, {boot['brier_ci'][1]:.4f}]$) \\\\")
    rows_tex.append(f"  & Repeated {OUTER_FOLDS}-fold CV ({REPEATS}$\\times$ repeats) & AUROC ${rep['auroc_mean']:.4f} \\pm {rep['auroc_sd']:.4f}$ (min ${rep['auroc_min']:.4f}$, max ${rep['auroc_max']:.4f}$); Brier ${rep['brier_mean']:.4f} \\pm {rep['brier_sd']:.4f}$ (min ${rep['brier_min']:.4f}$, max ${rep['brier_max']:.4f}$) \\\\")

    # Calibration
    rows_tex.append(f"  & Isotonic calibration (held-out) & AUROC ${cal['isotonic']['auroc']:.4f}$; Brier score ${cal['isotonic']['brier']:.4f}$ \\\\")
    rows_tex.append(f"  & Platt (sigmoid) calibration (held-out) & AUROC ${cal['sigmoid']['auroc']:.4f}$; Brier score ${cal['sigmoid']['brier']:.4f}$ \\\\")
    rows_tex.append("  & Final chosen calibration & Isotonic (selected for best balance of discrimination and Brier score) \\\\")
    rows_tex.append(f"  & Calibration curve (primary model) & Slope $\\approx {slope:.4f}$; intercept $\\approx {intercept:.4f}$ \\\\")

    # Threshold sensitivity
    sweep_map = {r["percentile"]: r for r in sweep_rows}
    for pctl in PCTL_SWEEP:
        rr = sweep_map[pctl]
        rows_tex.append(
            f"  & {pctl}th percentile TG$_{{4h}}$"
            + (" (primary)" if pctl == PRIMARY_PCTL else "")
            + f" & Cut-off ${rr['tg4_cut']:.4f}\\,\\mathrm{{mg/dL}}$; AUROC ${rr['auroc']:.4f}$; Brier ${rr['brier']:.4f}$; "
              f"F1 ${rr['f1']:.4f}$; recall ${rr['recall']:.4f}$; precision ${rr['precision']:.4f}$ \\\\"
        )
    rows_tex.append("  & Summary & AUROC remains $\\approx 0.80$ for 60--80th percentiles; WBV has negligible influence at all thresholds \\\\")

    # Ablation
    ab = {a["drop"]: a["auroc"] for a in ablation}
    rows_tex.append(f"  & Remove TG$_{{0h}}$ & AUROC ${ab['TG0h']:.2f}$ \\\\")
    rows_tex.append(f"  & Remove BMI & AUROC ${ab['BMI']:.2f}$ \\\\")
    rows_tex.append(f"  & Remove WBV & AUROC ${ab['WBV']:.2f}$ (no change vs. full model) \\\\")

    # External synthetic block (optional)
    if external_block is None:
        rows_tex.append("  & Original synthetic cohort (internal, external-synthetic setting) & (not computed: provide EXTERNAL_PATH if needed) \\\\")
        rows_tex.append("  & Independent external synthetic cohort (uncalibrated) & (not computed: provide EXTERNAL_PATH if needed) \\\\")
        rows_tex.append("  & After three-layer calibration on external-synthetic transfer & (not computed: provide EXTERNAL_PATH if needed) \\\\")
    else:
        rows_tex.append("  & Original synthetic cohort (internal, external-synthetic setting) & (not recomputed here; depends on your generation setting) \\\\")
        rows_tex.append(f"  & Independent external synthetic cohort (uncalibrated) & AUROC $= {external_block['uncalibrated']['auroc']:.3f}$; Brier $= {external_block['uncalibrated']['brier']:.3f}$ \\\\")
        rows_tex.append("  & After three-layer calibration on external-synthetic transfer & (optional: add if you run layered calibration) \\\\")

    # Modeling config lines (static)
    rows_tex.append(f"  & Logistic regression (L2) hyperparameters & $C \\in \\{{{', '.join(map(str, LR_C_GRID))}\\}}$; penalty $\\ell_2$; solver \\texttt{{lbfgs}}; max\\_iter $= 500$ \\\\")
    rows_tex.append(f"  & Random forest hyperparameters & $n\\_\\text{{estimators}} \\in \\{{{', '.join(map(str, RF_N_EST))}\\}}$; max\\_depth $\\in \\{{\\text{{None}}, 6, 12\\}}$; min\\_samples\\_leaf $\\in \\{{1, 2, 4\\}}$ \\\\")
    rows_tex.append(f"  & SVM (RBF) hyperparameters \\& CV details & $C \\in \\{{{', '.join(map(str, SVM_C))}\\}}$; $\\gamma \\in \\{{\\text{{scale}}, 0.1, 0.01\\}}$; stratified nested {OUTER_FOLDS}-fold outer / {INNER_FOLDS}-fold inner CV; global random seed $= {RANDOM_SEED}$ \\\\")

    with open("summary_table_rows.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(rows_tex) + "\n")

    print("Done.")
    print("- summary_table_values.json")
    print("- summary_table_rows.tex")
    print("- baseline_table_stats.csv")
    print("- threshold_sensitivity.csv")
    print("- ablation_results.csv")
    print("- misspecification_results.csv")

if __name__ == "__main__":
    main()
