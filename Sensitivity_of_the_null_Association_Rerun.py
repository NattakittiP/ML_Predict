"""Sensitivity_of_the_null_Association_aligned.py

Manuscript-consistent sensitivity analysis for possible WBV estimation error.

What this script does (as described in the paper):
- Ensure WBV is computed using the *de Simone low-shear surrogate*:
    WBV = 0.12*Hct + 0.17*TP - 0.3519
- Ensure postprandial response metric is *TGR (% excursion)*:
    TGR = (TG4h - TG0h)/TG0h * 100
- Define HighResponder by TG4h ≥ 75th percentile (within the current dataset).
- Perturb WBV multiplicatively with relative Gaussian noise (±5%, ±10% as SD).
- For each perturbation, recompute:
    * corr(WBV, TGR)
    * AUROC of a WBV-only score for predicting HighResponder

Notes:
- WBV-only AUROC uses z-scored WBV (monotone transform; AUROC unchanged).
- Increase MISSPEC_REPEATS to match the manuscript exactly if needed.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# =========================
# CONFIG
# =========================

PATH = "Dataset.csv"  # set to your current dataset file
REL_STD_LIST = (0.05, 0.10)
MISSPEC_REPEATS = 200
RANDOM_STATE = 42


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = {"Hematocrit", "TotalProtein", "TG0h", "TG4h"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # WBV (de Simone surrogate)
    out["WBV"] = 0.12 * out["Hematocrit"] + 0.17 * out["TotalProtein"] - 0.3519

    # TGR (% excursion)
    out["TGR"] = (out["TG4h"] - out["TG0h"]) / np.maximum(out["TG0h"], 1e-6) * 100.0

    # High-responder label (TG4h p75)
    cutoff = np.percentile(out["TG4h"], 75)
    out["HighResponder"] = (out["TG4h"] >= cutoff).astype(int)

    return out


def wbv_sensitivity(df: pd.DataFrame,
                    rel_std_list=REL_STD_LIST,
                    n_rep: int = MISSPEC_REPEATS,
                    random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    results = []

    y = df["HighResponder"].to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("HighResponder has a single class; AUROC is undefined.")

    wbv = df["WBV"].to_numpy(dtype=float)
    tgr = df["TGR"].to_numpy(dtype=float)

    for rel_std in rel_std_list:
        for rep in range(n_rep):
            # multiplicative noise around 1.0
            noise = rng.normal(loc=1.0, scale=rel_std, size=len(df))
            wbv_perturbed = wbv * noise

            r = np.corrcoef(wbv_perturbed, tgr)[0, 1]

            # z-score WBV (AUROC invariant under monotone transforms, but keep stable scale)
            s = wbv_perturbed
            s = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
            auc = roc_auc_score(y, s)

            results.append({
                "rel_std": float(rel_std),
                "rep": int(rep),
                "corr_WBV_TGR": float(r),
                "AUROC_WBV_only": float(auc),
            })

    return pd.DataFrame(results)


def main():
    df = pd.read_csv(PATH)
    df = ensure_columns(df)

    sens_df = wbv_sensitivity(df)
    summary = sens_df.groupby("rel_std")[["corr_WBV_TGR", "AUROC_WBV_only"]].agg(["mean", "std"])

    print(summary)


if __name__ == "__main__":
    main()
