import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# =========================
# 1) Preparing Dataset
# =========================

PATH = "WBV_TCR_ICBCB2026_Synthetic_1500_precise_v2 (2).csv"
df = pd.read_csv(PATH)
if "WBV" not in df.columns:
    if {"Hematocrit", "TotalProtein"}.issubset(df.columns):
        df["WBV"] = 1.89 * df["Hematocrit"] + 3.76 * df["TotalProtein"] - 4.55
    else:
        raise ValueError("ไม่มีคอลัมน์ WBV หรือ Hct/TP ให้คำนวณ WBV")
if "TCR" not in df.columns:
    if {"TG0h", "TG4h"}.issubset(df.columns):
        df["TCR"] = (df["TG4h"] - df["TG0h"]) / df["TG0h"]
    else:
        raise ValueError("ไม่มีคอลัมน์ TCR หรือ TG0h/TG4h ให้คำนวณ TCR")
if "HighResponder" not in df.columns:
    if "TG4h" in df.columns:
        cutoff = np.percentile(df["TG4h"], 75)
        df["HighResponder"] = (df["TG4h"] >= cutoff).astype(int)
    else:
        raise ValueError("ไม่มีคอลัมน์ HighResponder หรือ TG4h ให้นิยาม HighResponder")


# =========================
# 2) Sensitivity Analysis Function
# =========================

def wbv_sensitivity(df, rel_std_list=(0.05, 0.10), n_rep=50, random_state=42):
    rng = np.random.default_rng(random_state)
    results = []

    y = df["HighResponder"].values
    if len(np.unique(y)) < 2:
        raise ValueError("HighResponder มีแค่คลาสเดียว ทำ AUC ไม่ได้")

    for rel_std in rel_std_list:
        for rep in range(n_rep):
            
            noise = rng.normal(loc=1.0, scale=rel_std, size=len(df))
            wbv_perturbed = df["WBV"].values * noise
            r = np.corrcoef(wbv_perturbed, df["TCR"].values)[0, 1]
            std = wbv_perturbed.std()
            if std == 0:
                scores = wbv_perturbed
            else:
                scores = (wbv_perturbed - wbv_perturbed.mean()) / std
            auc = roc_auc_score(y, scores)
            results.append({
                "rel_std": rel_std,
                "rep": rep,
                "corr_WBV_TCR": r,
                "AUC_WBV_only": auc,
            })

    return pd.DataFrame(results)


# =========================
# 3) Running and Processing Process
# =========================

sens_df = wbv_sensitivity(df)
summary = sens_df.groupby("rel_std")[["corr_WBV_TCR", "AUC_WBV_only"]].agg(["mean", "std"])
print(summary)
