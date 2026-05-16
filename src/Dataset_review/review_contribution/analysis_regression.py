from __future__ import annotations

"""
Regression of detection performance against PCA scores.

Answers the question: does ``PC1`` of the contribution proxies predict mAP,
and does adding ``PC2`` (channel-redundancy axis) add explanatory power?

Pipeline
--------
  1. Load per-method PCA scores (output of :mod:`analysis_pca`).
  2. Load detection training results (``raw_training_data.csv``).
  3. Filter training rows by ``--dataset`` + ``--condition``, aggregate per
     method (mean and P90 = μ + 1.28·σ, matching the user's convention).
  4. Merge on ``method`` / ``Type`` with an outer-key intersection check.
  5. Fit nested OLS models: ``target ~ PC1``, ``target ~ PC1 + PC2``,
     ``target ~ PC1 + PC2 + PC3``.  Report R², adjusted R², and nested
     F-tests for ΔR² significance.
  6. Emit a terminal summary + CSV + scatter plots.

Why P90?  The user consistently summarises noisy per-training distributions
by ``μ + 1.28·σ`` (approximate 90th percentile of the fitted normal).  The
same convention is used in this module for the detection target.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

from Dataset_review.review_contribution.training_results_check import (
    DEFAULT_CSV_PATH,
    build_dataset_metrics,
    load_training_results,
)


# Detection metrics to regress.  ``training_results_check`` exposes
# P, R, mAP50 and mAP50-95 filtered by Class=person and Group Key
# (equalization).  The user's stated regression targets are mAP50 and
# mAP50-95; P/R come along for free because the aggregator produces them.
DETECTION_TARGETS: tuple[tuple[str, str], ...] = (
    # (metric_in_training_check, reported_name)
    ("mAP50",     "mAP50"),
    ("mAP50-95",  "mAP50_95"),
)


# ---------------------------------------------------------------------------
# Data loading — delegates to training_results_check for CSV parsing.
# ---------------------------------------------------------------------------

def load_detection_per_method(training_csv: Path,
                              dataset: str,
                              condition: str,
                              equalization: str = "no_equalization",
                              method_names: list[str] | None = None,
                              ) -> pd.DataFrame:
    """Per-method detection metrics via :mod:`training_results_check`.

    P90 is computed here as ``μ + 1.28·σ_sample`` (ddof=1) to match the
    user's convention for noise-summary; the raw ``values`` array is used
    so we don't depend on ddof choices inside ``build_method_metrics``.
    """
    df = load_training_results(training_csv)
    dataset_tag = "llvip_80_20" if dataset == "llvip" else "kaist_80_20"

    if method_names is None:
        method_names = sorted(
            df[(df["Dataset"] == dataset_tag)
               & (df["Condition"] == condition)
               & (df["Group Key"] == equalization)]["Type"].unique().tolist()
        )

    metrics_by_method = build_dataset_metrics(
        df, method_names, condition, equalization, dataset,
    )

    rows: list[dict] = []
    for name, info in metrics_by_method.items():
        row: dict = {"method": name, "n_runs": info["n_runs"]}
        for metric, stats_ in info["metrics"].items():
            out_name = metric.replace("-", "_")  # "mAP50-95" -> "mAP50_95"
            values = np.asarray(stats_["values"], dtype=np.float64)
            mu = float(values.mean())
            sigma = float(values.std(ddof=1)) if values.size > 1 else 0.0
            row[f"{out_name}_mean"] = mu
            row[f"{out_name}_std"] = sigma
            row[f"{out_name}_median"] = float(np.median(values))
            row[f"{out_name}_p90"] = mu + 1.28 * sigma
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OLS regression
# ---------------------------------------------------------------------------

def _ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """Solve y = X·β by least squares and return fit diagnostics.

    *X* must already include an intercept column.
    """
    n, p = X.shape
    beta, residuals_sum_sq, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    # Adjusted R² penalises the extra coefficients (excluding the intercept).
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n - p > 0 else float("nan")
    return {"beta": beta, "rss": rss, "tss": tss, "r2": r2, "adj_r2": adj_r2,
            "n": n, "p": p, "y_hat": y_hat, "resid": resid}


def _nested_ftest(fit_small: dict, fit_large: dict) -> tuple[float, float]:
    """Return (F, p) for nested-model comparison fit_small ⊂ fit_large."""
    rss_s = fit_small["rss"]
    rss_l = fit_large["rss"]
    df_num = fit_large["p"] - fit_small["p"]
    df_den = fit_large["n"] - fit_large["p"]
    if df_num <= 0 or df_den <= 0 or rss_l <= 0:
        return float("nan"), float("nan")
    f = ((rss_s - rss_l) / df_num) / (rss_l / df_den)
    p = 1.0 - stats.f.cdf(f, df_num, df_den)
    return float(f), float(p)


def regress_target(df: pd.DataFrame, target: str,
                   pc_cols: tuple[str, ...] = ("PC1_mean", "PC2_mean", "PC3_mean")
                   ) -> pd.DataFrame:
    """Run nested regressions ``target ~ PC1``, ``~ PC1+PC2``, ``~ PC1+PC2+PC3``."""
    y = df[target].to_numpy(dtype=np.float64)
    mask = np.isfinite(y)
    for col in pc_cols:
        mask &= np.isfinite(df[col].to_numpy(dtype=np.float64))
    y = y[mask]
    base = df.loc[mask, list(pc_cols)].to_numpy(dtype=np.float64)
    n = y.size

    rows: list[dict] = []
    previous_fit: dict | None = None
    for k in range(1, len(pc_cols) + 1):
        X = np.column_stack([np.ones(n), base[:, :k]])
        fit = _ols_fit(X, y)
        entry = {
            "target": target,
            "predictors": ", ".join(pc_cols[:k]),
            "n": n,
            "intercept": float(fit["beta"][0]),
        }
        for j, col in enumerate(pc_cols[:k]):
            entry[f"beta_{col}"] = float(fit["beta"][j + 1])
        entry["r2"] = float(fit["r2"])
        entry["adj_r2"] = float(fit["adj_r2"])
        if previous_fit is not None:
            f, pval = _nested_ftest(previous_fit, fit)
            entry["delta_r2"] = float(fit["r2"] - previous_fit["r2"])
            entry["F_add"] = f
            entry["p_add"] = pval
        else:
            entry["delta_r2"] = float("nan")
            entry["F_add"] = float("nan")
            entry["p_add"] = float("nan")
        rows.append(entry)
        previous_fit = fit
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scatter_pc_vs_target(df: pd.DataFrame, pc_col: str, target: str,
                           out_path: Path, title: str) -> None:
    x = df[pc_col].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(x, y, s=45, alpha=0.85, color="tab:blue")
    for _, row in df.iterrows():
        ax.annotate(row["method"], (row[pc_col], row[target]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    # Simple linear fit line for visual reference.
    if np.isfinite(x).all() and np.isfinite(y).all() and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, "--", color="gray", lw=1,
                label=f"slope={slope:+.3f}")
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.02, 0.98, f"Pearson r = {r:+.3f}  (R² = {r*r:.3f})",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlabel(pc_col)
    ax.set_ylabel(target)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regress detection metrics against PCA scores of contribution proxies.",
    )
    p.add_argument("--dataset", choices=["kaist", "llvip"], default="llvip")
    p.add_argument("--condition", choices=["day", "night"], default="night")
    p.add_argument("--pca-scores", default=None,
                   help="Per-method PCA scores CSV (default: reports/pca/"
                        "pca_scores_per_method_<dataset>_<condition>[_with_cr].csv).")
    p.add_argument("--with-cr", action="store_true",
                   help="Load the PCA score file that includes channel_redundancy.")
    p.add_argument("--training-csv", default=str(DEFAULT_CSV_PATH),
                   help="Raw training results CSV (defaults to training_results_check.DEFAULT_CSV_PATH).")
    p.add_argument("--equalization",
                   choices=["no_equalization", "rgb_equalization",
                            "th_equalization", "rgb_th_equalization"],
                   default="no_equalization",
                   help="Group Key filter (default: no_equalization).")
    p.add_argument("--target-reducer", choices=["mean", "median", "p90"],
                   default="p90",
                   help="Per-method summary of the detection metric (default: p90).")
    p.add_argument("--report-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    cache_dir = Path.home() / ".cache" / "eeha_review_fusion_contribution"
    suffix = "_with_cr" if args.with_cr else ""
    pca_path = Path(args.pca_scores) if args.pca_scores else (
        cache_dir / "reports" / "pca"
        / f"pca_scores_per_method_{args.dataset}_{args.condition}{suffix}.csv"
    )
    report_dir = Path(args.report_dir) if args.report_dir else cache_dir / "reports" / "regression"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not pca_path.exists():
        print(f"[error] PCA scores not found: {pca_path}", file=sys.stderr)
        return 2

    pca_scores = pd.read_csv(pca_path)
    pca_methods = sorted(pca_scores["method"].unique().tolist())
    det = load_detection_per_method(
        Path(args.training_csv), args.dataset, args.condition,
        equalization=args.equalization, method_names=pca_methods,
    )
    merged = pca_scores.merge(det, on="method", how="inner")
    if merged.empty:
        print("[error] no overlap between PCA methods and training data", file=sys.stderr)
        return 3

    only_pca = set(pca_scores["method"]) - set(det["method"])
    only_det = set(det["method"]) - set(pca_scores["method"])
    print(f"[reg] merged methods: {len(merged)}  | PCA-only: {sorted(only_pca)}"
          f"  | training-only: {sorted(only_det)}")

    target_suffix = args.target_reducer
    all_results: list[pd.DataFrame] = []
    tag = f"{args.dataset}_{args.condition}{suffix}_{args.equalization}_{target_suffix}"

    for _, out_name in DETECTION_TARGETS:
        target_col = f"{out_name}_{target_suffix}"
        if target_col not in merged.columns or merged[target_col].isna().all():
            continue
        result = regress_target(merged, target_col)
        result.insert(0, "reducer", args.target_reducer)
        all_results.append(result)

        for pc in ("PC1_mean", "PC2_mean"):
            if pc in merged.columns:
                _scatter_pc_vs_target(
                    merged[["method", pc, target_col]].dropna(),
                    pc, target_col,
                    report_dir / f"scatter_{tag}_{out_name}_vs_{pc}.png",
                    title=f"{args.dataset}/{args.condition}  {out_name} vs {pc}",
                )

    if not all_results:
        print("[error] no regressions produced", file=sys.stderr)
        return 4

    combined = pd.concat(all_results, ignore_index=True)
    out_csv = report_dir / f"regression_{tag}.csv"
    combined.to_csv(out_csv, index=False)

    print(f"\n[reg] regression results — target_reducer={args.target_reducer}")
    cols = ["target", "predictors", "n", "r2", "adj_r2", "delta_r2", "F_add", "p_add"]
    print(combined[cols].round(4).to_string(index=False))
    print(f"\n[reg] written to {out_csv}")
    print(f"[reg] scatters in {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
