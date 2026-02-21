# compare_calibration_summary.py
# ------------------------------------------------------------
# Compare calibration methods (sigmoid vs isotonic) produced by run_rebuild_v2_pipeline.py
#
# Usage:
#   python compare_calibration_summary.py --root out_v2 --out out_v2/compare_calibration
#
# Expected structure under --root:
#   calib_sigmoid/
#   calib_isotonic/
#
# Minimum required files per method:
#   nestedcv_summary.json
#   oof_predictions.csv
#   calibration_slope_intercept.json
#
# Optional (used if present):
#   bootstrap_finalmodel/bootstrap_finalmodel_oof.csv
#   bootstrap_finalmodel/bootstrap_finalmodel_summary.json
#   robustness_repeatedcv_5x10/repeatedcv_5x10_summary.json
#   decision_curve_analysis/decision_curve.csv
#   threshold_sensitivity/threshold_sensitivity_summary.csv
# ------------------------------------------------------------

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.calibration import calibration_curve


# ----------------------------
# Helpers
# ----------------------------
def safe_read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


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
class MethodPaths:
    name: str
    root: str

    @property
    def nestedcv_summary(self) -> str:
        return os.path.join(self.root, "nestedcv_summary.json")

    @property
    def oof_predictions(self) -> str:
        return os.path.join(self.root, "oof_predictions.csv")

    @property
    def calib_line(self) -> str:
        return os.path.join(self.root, "calibration_slope_intercept.json")

    @property
    def bootstrap_oof(self) -> str:
        return os.path.join(self.root, "bootstrap_finalmodel", "bootstrap_finalmodel_oof.csv")

    @property
    def bootstrap_summary(self) -> str:
        return os.path.join(self.root, "bootstrap_finalmodel", "bootstrap_finalmodel_summary.json")

    @property
    def repeatedcv_summary(self) -> str:
        return os.path.join(self.root, "robustness_repeatedcv_5x10", "repeatedcv_5x10_summary.json")

    @property
    def decision_curve(self) -> str:
        return os.path.join(self.root, "decision_curve_analysis", "decision_curve.csv")

    @property
    def threshold_sensitivity(self) -> str:
        return os.path.join(self.root, "threshold_sensitivity", "threshold_sensitivity_summary.csv")


def compute_extra_metrics_from_oof(oof: pd.DataFrame) -> Dict[str, float]:
    """
    Recompute core metrics from OOF predictions (in case you want consistency).
    """
    oof2 = oof.dropna(subset=["y", "proba_oof"]).copy()
    y = oof2["y"].astype(int).values
    p = oof2["proba_oof"].astype(float).values

    out = {}
    out["auroc_oof_recalc"] = float(roc_auc_score(y, p))
    out["brier_oof_recalc"] = float(brier_score_loss(y, p))
    out["ap_oof_recalc"] = float(average_precision_score(y, p))

    # ECE (Expected Calibration Error) with quantile bins
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    # Approx ECE: average absolute difference across bins
    # (Note: sklearn doesn't return bin counts; this is an unweighted proxy ECE.)
    out["ece_10bin_unweighted"] = float(np.mean(np.abs(frac_pos - mean_pred)))

    return out


def summarize_bootstrap(boot: pd.DataFrame) -> Dict[str, float]:
    """
    Summarize bootstrap distributions if available.
    """
    out = {}
    if boot is None or len(boot) == 0:
        return out
    for col in ["auroc", "brier"]:
        if col not in boot.columns:
            continue
        vals = boot[col].dropna().values.astype(float)
        if len(vals) == 0:
            continue
        out[f"boot_{col}_mean"] = float(np.mean(vals))
        out[f"boot_{col}_ci95_lo"] = float(np.percentile(vals, 2.5))
        out[f"boot_{col}_ci95_hi"] = float(np.percentile(vals, 97.5))
    return out


def load_method(method: MethodPaths) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, float]]:
    """
    Returns:
      - summary_row: dict with lots of metrics in one row
      - oof_df: OOF predictions dataframe
      - dca_df: decision curve df (empty if missing)
    """
    # required
    s = safe_read_json(method.nestedcv_summary)
    oof = safe_read_csv(method.oof_predictions)
    cal = safe_read_json(method.calib_line)

    if s is None:
        raise FileNotFoundError(f"Missing required: {method.nestedcv_summary}")
    if oof is None:
        raise FileNotFoundError(f"Missing required: {method.oof_predictions}")
    if cal is None:
        raise FileNotFoundError(f"Missing required: {method.calib_line}")

    row: Dict[str, float] = {"method": method.name}

    # nestedcv_summary.json (from pipeline)
    for k, v in s.items():
        if isinstance(v, (int, float, np.number)):
            row[k] = float(v)

    # calibration slope/intercept
    if "slope" in cal:
        row["cal_slope"] = float(cal["slope"])
    if "intercept" in cal:
        row["cal_intercept"] = float(cal["intercept"])

    # recompute metrics from OOF for sanity
    row.update(compute_extra_metrics_from_oof(oof))

    # optional bootstrap
    boot = safe_read_csv(method.bootstrap_oof)
    row.update(summarize_bootstrap(boot))

    # optional repeated cv summary
    rep = safe_read_json(method.repeatedcv_summary)
    if rep is not None:
        # only pick a few key fields to avoid clutter
        for k in ["auroc_mean", "auroc_std", "auroc_min", "auroc_max",
                  "brier_mean", "brier_std", "brier_min", "brier_max",
                  "n_folds_total"]:
            if k in rep and isinstance(rep[k], (int, float, np.number)):
                row[f"repeated_{k}"] = float(rep[k])

    # decision curve
    dca = safe_read_csv(method.decision_curve)
    if dca is None:
        dca = pd.DataFrame()

    return row, oof, dca


def plot_overlay_calibration(oof_a: pd.DataFrame, oof_b: pd.DataFrame,
                             name_a: str, name_b: str, out_png: str,
                             n_bins: int = 10):
    set_plot_style()
    y_a = oof_a["y"].astype(int).values
    p_a = oof_a["proba_oof"].astype(float).values
    y_b = oof_b["y"].astype(int).values
    p_b = oof_b["proba_oof"].astype(float).values

    frac_a, mean_a = calibration_curve(y_a, p_a, n_bins=n_bins, strategy="quantile")
    frac_b, mean_b = calibration_curve(y_b, p_b, n_bins=n_bins, strategy="quantile")

    plt.figure(figsize=(6.2, 5.0))
    plt.plot(mean_a, frac_a, marker="o", label=name_a)
    plt.plot(mean_b, frac_b, marker="o", label=name_b)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(f"Calibration curve overlay (OOF, {n_bins} quantile bins)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_bootstrap_distributions(boot_a: Optional[pd.DataFrame], boot_b: Optional[pd.DataFrame],
                                 name_a: str, name_b: str, out_dir: str):
    """
    Save histograms for bootstrap AUROC and Brier, overlayed if both exist.
    """
    set_plot_style()
    ensure_dir(out_dir)

    for metric in ["auroc", "brier"]:
        if (boot_a is None or metric not in boot_a.columns) and (boot_b is None or metric not in boot_b.columns):
            continue

        plt.figure(figsize=(6.2, 4.6))
        if boot_a is not None and metric in boot_a.columns:
            plt.hist(boot_a[metric].dropna().values, bins=30, alpha=0.55, label=name_a)
        if boot_b is not None and metric in boot_b.columns:
            plt.hist(boot_b[metric].dropna().values, bins=30, alpha=0.55, label=name_b)
        plt.xlabel(metric.upper())
        plt.ylabel("Frequency")
        plt.title(f"Bootstrap distribution: {metric.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fig_bootstrap_overlay_{metric}.png"))
        plt.close()


def plot_decision_curve_overlay(dca_a: pd.DataFrame, dca_b: pd.DataFrame,
                                name_a: str, name_b: str, out_png: str):
    """
    Overlay net benefit curves if decision_curve.csv exists for both.
    Uses nb_full if present; falls back to net_benefit_model.
    """
    if dca_a is None or dca_b is None or dca_a.empty or dca_b.empty:
        return

    set_plot_style()

    # column selection
    def _pick_nb(df: pd.DataFrame) -> str:
        if "nb_full" in df.columns:
            return "nb_full"
        if "net_benefit_model" in df.columns:
            return "net_benefit_model"
        # some other naming? fallback
        for c in df.columns:
            if "net_benefit" in c and "treat" not in c:
                return c
        return ""

    nb_a = _pick_nb(dca_a)
    nb_b = _pick_nb(dca_b)
    if nb_a == "" or nb_b == "":
        return

    plt.figure(figsize=(6.6, 5.2))
    plt.plot(dca_a["pt"], dca_a[nb_a], label=f"{name_a}")
    plt.plot(dca_b["pt"], dca_b[nb_b], label=f"{name_b}")

    # baselines if exist in A (usually same in both)
    for base_col, style, label in [
        ("net_benefit_treat_all", "--", "Treat all"),
        ("net_benefit_treat_none", "--", "Treat none"),
    ]:
        if base_col in dca_a.columns:
            plt.plot(dca_a["pt"], dca_a[base_col], linestyle=style, label=label)

    plt.xlabel("Decision threshold (pt)")
    plt.ylabel("Net benefit")
    plt.title("Decision curve overlay (OOF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def compare_and_rank(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple "better/worse" cues:
      - higher AUROC/AP better
      - lower Brier/ECE better
      - slope closer to 1 better
      - intercept closer to 0 better
    """
    out = df_summary.copy()

    def closeness(x: float, target: float) -> float:
        return abs(float(x) - float(target))

    # score components (lower = better)
    comps = []
    if "auroc_mean" in out.columns:
        out["score_auroc"] = -out["auroc_mean"]  # negate so higher AUROC => lower score
        comps.append("score_auroc")
    if "brier_mean" in out.columns:
        out["score_brier"] = out["brier_mean"]
        comps.append("score_brier")
    if "ap_mean" in out.columns:
        out["score_ap"] = -out["ap_mean"]
        comps.append("score_ap")
    if "ece_10bin_unweighted" in out.columns:
        out["score_ece"] = out["ece_10bin_unweighted"]
        comps.append("score_ece")
    if "cal_slope" in out.columns:
        out["score_slope"] = out["cal_slope"].apply(lambda v: closeness(v, 1.0))
        comps.append("score_slope")
    if "cal_intercept" in out.columns:
        out["score_intercept"] = out["cal_intercept"].apply(lambda v: closeness(v, 0.0))
        comps.append("score_intercept")

    if comps:
        out["score_total_unweighted"] = out[comps].sum(axis=1)
        out = out.sort_values("score_total_unweighted", ascending=True)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing calib_sigmoid/ and calib_isotonic/")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for comparison artifacts")
    parser.add_argument("--sigmoid_dir", type=str, default="calib_sigmoid",
                        help="Folder name under root for sigmoid outputs")
    parser.add_argument("--isotonic_dir", type=str, default="calib_isotonic",
                        help="Folder name under root for isotonic outputs")
    parser.add_argument("--bins", type=int, default=10,
                        help="Number of quantile bins for calibration overlay")
    args = parser.parse_args()

    out_dir = args.out
    ensure_dir(out_dir)

    mp_sig = MethodPaths("sigmoid", os.path.join(args.root, args.sigmoid_dir))
    mp_iso = MethodPaths("isotonic", os.path.join(args.root, args.isotonic_dir))

    row_sig, oof_sig, dca_sig = load_method(mp_sig)
    row_iso, oof_iso, dca_iso = load_method(mp_iso)

    # Combine summaries
    df_summary = pd.DataFrame([row_sig, row_iso])
    df_summary = compare_and_rank(df_summary)

    # Save table
    df_summary.to_csv(os.path.join(out_dir, "calibration_methods_comparison.csv"), index=False)
    with open(os.path.join(out_dir, "calibration_methods_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(df_summary.to_dict(orient="records"), f, indent=2)

    # Also save a “wide” view for easy eyeballing
    wide = df_summary.set_index("method").T.reset_index().rename(columns={"index": "metric"})
    wide.to_csv(os.path.join(out_dir, "calibration_methods_comparison_wide.csv"), index=False)

    # Plots
    plot_overlay_calibration(
        oof_a=oof_sig.dropna(subset=["y", "proba_oof"]),
        oof_b=oof_iso.dropna(subset=["y", "proba_oof"]),
        name_a="sigmoid",
        name_b="isotonic",
        out_png=os.path.join(out_dir, "fig_calibration_overlay_oof.png"),
        n_bins=args.bins
    )

    # Bootstrap overlay if exists
    boot_sig = safe_read_csv(mp_sig.bootstrap_oof)
    boot_iso = safe_read_csv(mp_iso.bootstrap_oof)
    plot_bootstrap_distributions(boot_sig, boot_iso, "sigmoid", "isotonic",
                                 out_dir=os.path.join(out_dir, "bootstrap_overlay"))

    # Decision curve overlay if exists
    plot_decision_curve_overlay(dca_sig, dca_iso, "sigmoid", "isotonic",
                                out_png=os.path.join(out_dir, "fig_decision_curve_overlay.png"))

    # Quick console print
    print("\n=== Comparison saved to ===")
    print(out_dir)
    print("\nTop (best) by unweighted composite score (if available):")
    if "score_total_unweighted" in df_summary.columns:
        print(df_summary[["method", "score_total_unweighted"]].to_string(index=False))
    else:
        print(df_summary[["method"]].to_string(index=False))

    print("\nKey metrics (if present):")
    cols_show = [c for c in ["method", "auroc_mean", "brier_mean", "ap_mean",
                             "cal_slope", "cal_intercept", "ece_10bin_unweighted"] if c in df_summary.columns]
    if cols_show:
        print(df_summary[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
