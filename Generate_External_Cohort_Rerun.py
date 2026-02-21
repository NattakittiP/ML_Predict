"""
Generate an EXTERNAL synthetic cohort based on the primary synthetic dataset.

Manuscript consistency (Scientific Reports):
- Fit a Gaussian copula to the *baseline variables* joint distribution.
- Sample a new cohort (n=1500) and apply controlled covariate shifts.
- Recompute derived variables:
    * WBV (de Simone low-shear surrogate): WBV = 0.12*Hct + 0.17*TP - 0.3519
    * TGR (%): (TG4h - TG0h)/TG0h * 100
- Redefine the phenotype in the external cohort using the *external cohort's own*
  TG4h 75th percentile (domain-shift stress test).

Input columns expected in BASE_SYNTHETIC_PATH:
  Sex, Age, Hematocrit, TotalProtein, TG0h, TG4h, HDL, LDL, BMI

Outputs:
  external_synthetic_like.csv
"""

import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate

# =========================
# CONFIG
# =========================

BASE_SYNTHETIC_PATH = "Dataset.csv"
OUTPUT_EXTERNAL_PATH = "external_synthetic_like.csv"

FEATURE_COLS = [
    "Sex",          # string
    "Age",
    "Hematocrit",
    "TotalProtein",
    "TG0h",
    "TG4h",
    "HDL",
    "LDL",
    "BMI",
]

EXTERNAL_N = 1500
RANDOM_SEED = 42


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def preprocess_for_copula(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Sex to numeric for copula fitting (Female=1, Male=0)."""
    out = df.copy()
    out["Sex"] = out["Sex"].map({"Female": 1, "Male": 0}).astype(int)
    return out


def fit_copula(df_num: pd.DataFrame) -> GaussianMultivariate:
    model = GaussianMultivariate()
    model.fit(df_num[FEATURE_COLS])
    return model


def apply_covariate_shift(df_num: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Apply mild distributional shifts to emulate a different hospital population."""
    shifted = df_num.copy()

    # TG shifts
    shifted["TG0h"] *= rng.normal(1.07, 0.03, len(shifted))
    shifted["TG4h"] *= rng.normal(1.10, 0.04, len(shifted))

    # Hct / TP / BMI / Age shifts
    shifted["Hematocrit"] += rng.normal(-0.4, 0.6, len(shifted))
    shifted["TotalProtein"] += rng.normal(0.10, 0.10, len(shifted))
    shifted["BMI"] += rng.normal(0.5, 0.5, len(shifted))
    shifted["Age"] += rng.normal(1.2, 2.0, len(shifted))

    # Clip to physiologic ranges
    shifted["Hematocrit"] = shifted["Hematocrit"].clip(30, 55)
    shifted["TotalProtein"] = shifted["TotalProtein"].clip(5.5, 8.8)
    shifted["BMI"] = shifted["BMI"].clip(15, 45)
    shifted["Age"] = shifted["Age"].clip(20, 90)

    return shifted


def compute_derived(df_num: pd.DataFrame) -> pd.DataFrame:
    """Compute manuscript-consistent WBV and TGR, then phenotype by TG4h p75 (external-only)."""
    out = df_num.copy()

    out["WBV"] = 0.12 * out["Hematocrit"] + 0.17 * out["TotalProtein"] - 0.3519
    out["TGR"] = (out["TG4h"] - out["TG0h"]) / np.maximum(out["TG0h"], 1e-6) * 100.0

    cutoff = np.percentile(out["TG4h"], 75)
    out["HighResponder"] = (out["TG4h"] >= cutoff).astype(int)

    # Convert Sex back to string labels
    out["Sex"] = out["Sex"].map({1: "Female", 0: "Male"})
    out["ID"] = np.arange(1, len(out) + 1)

    return out


def generate_external_synthetic() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    print("Loading base synthetic dataset...")
    base = load_dataset(BASE_SYNTHETIC_PATH)

    prepared = preprocess_for_copula(base)

    print("Fitting Gaussian Copula...")
    copula = fit_copula(prepared)

    print(f"Sampling {EXTERNAL_N} new synthetic records...")
    sampled = copula.sample(EXTERNAL_N)

    # Copula may emit non-binary Sex; re-binarize
    sampled["Sex"] = (sampled["Sex"] > 0.5).astype(int)

    print("Applying covariate shifts...")
    shifted = apply_covariate_shift(sampled, rng)

    print("Computing derived variables and phenotype...")
    final_df = compute_derived(shifted)

    print(f"Saving external synthetic cohort to: {OUTPUT_EXTERNAL_PATH}")
    final_df.to_csv(OUTPUT_EXTERNAL_PATH, index=False)

    print("Done.")
    print(final_df.head())


if __name__ == "__main__":
    generate_external_synthetic()
