"""
Generate an EXTERNAL synthetic cohort based on your actual ICBCB 2026 synthetic dataset.

This script:
- Loads your real synthetic cohort (1500 rows)
- Fits a Gaussian Copula to learn realistic joint distributions
- Samples a NEW external cohort (with covariate shifts)
- Recomputes WBV and TCR
- Recomputes High Responder based on TG4h >= 75th percentile
- Saves external synthetic cohort to CSV

Author: ChatGPT (for MIDM external validation)
"""
#pip install copulas scikit-learn pandas numpy

import os
import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate


# =========================
# CONFIG
# =========================

BASE_SYNTHETIC_PATH = "WBV_TCR_ICBCB2026_Synthetic_1500_precise_v2 (2).csv"
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


# =========================
# LOAD ORIGINAL SYNTHETIC
# =========================

def load_dataset(path):
    df = pd.read_csv(path)
    if not all(col in df.columns for col in FEATURE_COLS):
        raise ValueError("Dataset missing required columns.")
    return df


# =========================
# PREPARE: Convert Sex to Binary
# =========================

def preprocess_for_copula(df):
    out = df.copy()
    out["Sex"] = out["Sex"].map({"Female": 1, "Male": 0}).astype(int)
    return out


# =========================
# FIT GAUSSIAN COPULA
# =========================

def fit_copula(df):
    model = GaussianMultivariate()
    model.fit(df[FEATURE_COLS])
    return model


# =========================
# GENERATE NEW EXTERNAL SYNTHETIC SAMPLES
# =========================

def apply_covariate_shift(df):
    shifted = df.copy()

    # TG shifts (+5 to +10%)
    shifted["TG0h"] *= np.random.normal(1.07, 0.03, len(df))
    shifted["TG4h"] *= np.random.normal(1.10, 0.04, len(df))

    # Hematocrit slight decrease (simulate different machine calibration)
    shifted["Hematocrit"] += np.random.normal(-0.4, 0.6, len(df))

    # Total Protein slight increase
    shifted["TotalProtein"] += np.random.normal(0.10, 0.10, len(df))

    # BMI slight shift upward
    shifted["BMI"] += np.random.normal(0.5, 0.5, len(df))

    # Age small shift
    shifted["Age"] += np.random.normal(1.2, 2.0, len(df))

    # Clip to realistic ranges
    shifted["Hematocrit"] = shifted["Hematocrit"].clip(30, 55)
    shifted["TotalProtein"] = shifted["TotalProtein"].clip(5.5, 8.8)
    shifted["BMI"] = shifted["BMI"].clip(15, 45)
    shifted["Age"] = shifted["Age"].clip(20, 90)

    return shifted


# =========================
# COMPUTE WBV + TCR + PHENOTYPE
# =========================

def compute_derived(df):
    out = df.copy()
    out["WBV"] = 0.12*out["Hematocrit"] + 0.17*out["TotalProtein"] - 0.3519
    out["TCR"] = (out["TG0h"] - out["TG4h"]) / out["TG0h"] * 100.0

    cutoff = np.percentile(out["TG4h"], 75)
    out["HighResponder"] = (out["TG4h"] >= cutoff).astype(int)
    out["Sex"] = out["Sex"].map({1: "Female", 0: "Male"})
    out["ID"] = np.arange(1, len(out) + 1)
    return out


# =========================
# MAIN FUNCTION
# =========================

def generate_external_synthetic():
    print("Loading base ICBCB synthetic dataset...")
    base = load_dataset(BASE_SYNTHETIC_PATH)

    # Convert Sex to binary 0/1
    prepared = preprocess_for_copula(base)

    print("Fitting Gaussian Copula...")
    copula = fit_copula(prepared)

    print(f"Sampling {EXTERNAL_N} new synthetic records...")
    sampled = copula.sample(EXTERNAL_N)

    # Convert Sex to int
    sampled["Sex"] = (sampled["Sex"] > 0.5).astype(int)

    print("Applying covariate shift...")
    shifted = apply_covariate_shift(sampled)

    print("Computing WBV, TCR, and phenotype...")
    final_df = compute_derived(shifted)

    print(f"Saving external synthetic cohort to: {OUTPUT_EXTERNAL_PATH}")
    final_df.to_csv(OUTPUT_EXTERNAL_PATH, index=False)

    print("Done!")
    print(final_df.head())


# =========================
# RUN
# =========================

if __name__ == "__main__":
    generate_external_synthetic()