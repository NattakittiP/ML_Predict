"""Rebuild_SyntheticCohort_v2.py

Purpose:
  Generate a physiologically coherent synthetic cohort for postprandial TG response modeling.
  Key properties enforced:
    - Realistic fasting TG scale (right-skewed; mg/dL).
    - Predominantly postprandial rise at 4h: TG4h = TG0h + ΔTG with ΔTG > 0.
    - Mildly realistic correlations: TG0h and ΔTG increase with BMI; HDL decreases with BMI/TG0h.
    - Hct and Total Protein constrained to physiologic bounds (defaults: Hct 30–55%, TP 5.5–8.8 g/dL).
    - WBV computed as a surrogate linear function of Hct and TP (keep consistent with manuscript formula).

Output:
  - Dataset_v2_physiologic.csv

NOTE:
  This generator is for methodological evaluation. It is not a substitute for real-world cohort validation.
"""

import math
import numpy as np
import pandas as pd

SEED = 42
N = 1500

# Physiologic bounds (update to match manuscript)
HCT_LO, HCT_HI = 30.0, 55.0
TP_LO, TP_HI = 5.5, 8.8

# WBV surrogate (update to match the manuscript + add proper citation there)
def compute_wbv(hct, tp):
    return 0.12*hct + 0.17*tp - 0.3519

def main():
    rng = np.random.default_rng(SEED)

    sex = rng.choice(["Female","Male"], size=N, p=[0.52,0.48])
    age = np.clip(rng.normal(53, 10, size=N), 25, 80).round(2)
    bmi = np.clip(rng.normal(24, 3, size=N), 16, 38).round(1)

    hct = np.where(sex=="Female", rng.normal(40.5,3,size=N), rng.normal(44.5,3,size=N))
    hct = np.clip(hct, HCT_LO, HCT_HI).round(2)

    tp = np.clip(rng.normal(7.0,0.5,size=N), TP_LO, TP_HI).round(2)

    # LDL: mild age trend
    ldl = np.clip(rng.normal(125 + 0.6*(age-53), 25, size=N), 50, 220).round(1)

    # Fasting TG: lognormal + BMI effect (right-skewed, realistic scale)
    sigma = 0.55
    median = 130.0
    mu = math.log(median)
    base_tg0 = rng.lognormal(mean=mu, sigma=sigma, size=N)
    tg0 = base_tg0 * (1 + 0.03*(bmi-24))
    tg0 = np.clip(tg0, 30, 800).round(2)

    # HDL: decreases with BMI and TG0h (plus noise)
    hdl = 52 - 0.6*(bmi-24) - 0.01*(tg0-130) + rng.normal(0,6,size=N)
    hdl = np.clip(hdl, 25, 95).round(1)

    # WBV surrogate
    wbv = compute_wbv(hct, tp)
    wbv = np.round(wbv, 5)

    # Postprandial excursion ΔTG: positive and depends on TG0h and BMI
    delta = rng.lognormal(mean=math.log(90), sigma=0.6, size=N)
    delta = delta * (tg0/130)**0.6 * (1 + 0.04*(bmi-24))
    delta = np.maximum(delta + rng.normal(0,15,size=N), 5.0)

    tg4 = np.clip(tg0 + delta, 40, 1200).round(2)

    # Postprandial response ratio (% increase)
    tgr_pct = ((tg4 - tg0)/tg0*100).round(2)

    df = pd.DataFrame({
        "ID": np.arange(1, N+1),
        "Sex": sex,
        "Age": age,
        "Hematocrit": hct,
        "TotalProtein": tp,
        "WBV": wbv,
        "TG0h": tg0,
        "TG4h": tg4,
        "TGR": tgr_pct,
        "TGR_pct": tgr_pct,
        "HDL": hdl,
        "LDL": ldl,
        "BMI": bmi
    })

    # Define phenotype using TG4h 75th percentile
    cutoff = float(df["TG4h"].quantile(0.75))
    df["HighResponder"] = (df["TG4h"] >= cutoff).astype(int)
    df["Phenotype"] = np.where(df["HighResponder"] == 1, "High", "Normal")

    df.to_csv("Dataset.csv", index=False)
    print("Saved: Dataset.csv")
    print(f"TG4h p75 cutoff = {cutoff:.2f} mg/dL; counts:", df["Phenotype"].value_counts().to_dict())

if __name__ == "__main__":
    main()
