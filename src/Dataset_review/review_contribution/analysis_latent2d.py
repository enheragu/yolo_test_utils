from __future__ import annotations

"""
2D latent scatter: thermal axis × ``channel_redundancy`` across datasets.

Three axis options for the x-axis ("thermal axis"), trading off saturation
correction against cross-dataset portability:

  - ``latent_z``         (default): per-dataset PAVA-calibrated thermal share.
                         Best intra-dataset reading (saturation corrected,
                         IVW-weighted), but the calibration curves themselves
                         differ between datasets — so a method's value is
                         dataset-relative, not absolute.
  - ``cont_vis_raw``     (alias: ``raw``): unweighted mean of the 6 raw
                         proxies, mapped to thermal share as
                         ``1 − cont_vis_raw/100``.  No calibration; portable
                         cross-dataset; pays the price of proxy saturation.
  - ``reg``              per-channel NNLS thermal share (most calibration-
                         portable single proxy according to the curve-overlap
                         diagnostic).  Reductive — only one of the 6 proxies —
                         but very stable across datasets.

The y-axis is always ``channel_redundancy`` (inter-channel mean ``|corr|``).

Visual encoding:
  - color and marker = ``(dataset, condition)`` slice (3 redundant cues so the
    plot reads in B&W and for color-vision-deficient viewers).
  - size = uniform; ``contrib_confidence`` was tried as size but is hard to
    interpret without a key, and its information is largely captured by the
    inverse of ``contrib_std``.
  - small "+" markers at the visible/lwir endpoints anchor the eye to the
    extremes — the plot otherwise drops these to keep the comparison among
    real fusion methods.

The companion CSV ``latent2d_data_<tag>.csv`` carries every numeric used by
the figure (and a few extras like ``thermal_share_reg`` and
``calibration_lift``) so downstream analysis can replicate or recolor.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

from Dataset_review.review_contribution.training_results_check import (  # noqa: E402
    DEFAULT_CSV_PATH, load_training_results, build_dataset_metrics,
)


_DEFAULT_SLICES = ("kaist:day", "kaist:night", "llvip:night")
_DEFAULT_REPORTS = Path.home() / ".cache" / "eeha_review_fusion_contribution" / "reports"

# Marker per (dataset, condition).  Falls back to "P" for unknown combinations.
_MARKER_MAP = {
    ("kaist", "day"):    "o",
    ("kaist", "night"):  "s",
    ("llvip", "night"):  "^",
    ("llvip", "day"):    "v",
}

# Color per (dataset, condition).  Distinct hues, decent contrast.
_COLOR_MAP = {
    ("kaist", "day"):    "#1f77b4",   # blue
    ("kaist", "night"):  "#d62728",   # red
    ("llvip", "night"):  "#2ca02c",   # green
    ("llvip", "day"):    "#9467bd",   # purple
}

# X-axis selection — maps the CLI flag to (column-in-merged, axis label).
_X_AXIS_OPTIONS = {
    "latent_z":     ("latent_z_mean", "latent_z (calibrated thermal fraction)"),
    "raw":          ("thermal_share_raw",
                     "1 − cont_vis_raw/100 (uncalibrated thermal share)"),
    "cont_vis_raw": ("thermal_share_raw",
                     "1 − cont_vis_raw/100 (uncalibrated thermal share)"),
    "reg":          ("thermal_share_reg",
                     "1 − reg/100 (NNLS thermal share — calibration-portable)"),
}

# Anchor methods drawn as small grey "+" markers at the extremes of each axis
# so the eye keeps the [0, 1] context without crowding the legend.
_ANCHOR_METHODS = ("visible", "lwir")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _parse_slices(values: Iterable[str]) -> list[tuple[str, str]]:
    out = []
    for v in values:
        if ":" not in v:
            raise ValueError(f"Slice must be 'dataset:condition', got: {v!r}")
        ds, cond = v.split(":", 1)
        out.append((ds.strip(), cond.strip()))
    return out


def _load_overview(reports_dir: Path, dataset: str, condition: str,
                   eq_vis: str, eq_th: str) -> pd.DataFrame:
    csv_path = reports_dir / f"{dataset}_{condition}_methods_overview.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Overview CSV missing: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[(df["eq_vis"] == eq_vis) & (df["eq_th"] == eq_th)].copy()
    df["dataset"] = dataset
    df["condition"] = condition
    # Backfill derived columns when reading an older CSV that pre-dates the
    # pipeline patch.  Cheap and idempotent: a fresh CSV already has them.
    if "thermal_share_raw" not in df.columns and "cont_vis_raw_mean" in df.columns:
        df["thermal_share_raw"] = 1.0 - df["cont_vis_raw_mean"] / 100.0
    if "thermal_share_reg" not in df.columns and "reg_mean" in df.columns:
        df["thermal_share_reg"] = 1.0 - df["reg_mean"] / 100.0
    if "calibration_lift" not in df.columns and "latent_z_mean" in df.columns:
        df["calibration_lift"] = df["latent_z_mean"] - df["thermal_share_raw"]
    return df


def _load_training_metric(
    df_overview: pd.DataFrame,
    dataset: str,
    condition: str,
    equalization: str,
    metric: str,
    reducer: str,
    training_csv: Path,
) -> pd.Series:
    if not training_csv.exists():
        return pd.Series(np.nan, index=df_overview.index, name=metric)
    raw = load_training_results(training_csv)
    method_names = list(df_overview["method"].astype(str))
    metrics = build_dataset_metrics(raw, method_names, condition, equalization, dataset)
    out = []
    for name in method_names:
        entry = metrics.get(name)
        if entry is None or metric not in entry["metrics"]:
            out.append(np.nan)
            continue
        m = entry["metrics"][metric]
        if reducer == "mean":
            out.append(float(m["mean"]))
        elif reducer == "p90":
            out.append(float(m["mean"]) + 1.28 * float(m["std"]))
        elif reducer == "median":
            out.append(float(m["median"]))
        else:
            raise ValueError(f"Unknown reducer: {reducer}")
    return pd.Series(out, index=df_overview.index, name=metric)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _annotate_methods(ax, x, y, labels, fontsize=7, dy_lookup=None):
    for xv, yv, lab in zip(x, y, labels):
        if not (np.isfinite(xv) and np.isfinite(yv)):
            continue
        dy = dy_lookup(lab) if dy_lookup else 4
        ax.annotate(lab, (xv, yv), textcoords="offset points",
                    xytext=(4, dy), fontsize=fontsize, alpha=0.85)


def _plot_latent_2d_axis(ax, merged: pd.DataFrame, x_col: str, x_label: str,
                          anchors_df: pd.DataFrame | None,
                          show_legend: bool) -> list[tuple[str, str, str, str]]:
    """Render a single x-axis variant onto ``ax``.

    Returns the list of (dataset, condition, marker, color) tuples actually
    drawn so a caller can build a shared legend on the figure.
    """
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.45, zorder=1)
    ax.axhline(0.8, color="gray", linestyle=":", alpha=0.55, zorder=1)
    ax.axhline(1.0, color="gray", linestyle="-", alpha=0.30, linewidth=0.8, zorder=1)
    # Empty-quadrant marker: bottom-left (visible-dominant + distributed
    # channels).  No fusion method lands here in the surveyed datasets — the
    # shaded box surfaces it as a known gap rather than just empty space.
    ax.add_patch(plt.Rectangle((0.0, 0.0), 0.5, 0.6,
                                 facecolor="gray", alpha=0.06, zorder=0))
    ax.text(0.25, 0.30, "no fusion lands here\n(visible-dominant +\n distributed channels)",
             ha="center", va="center", fontsize=7, color="gray", alpha=0.85,
             style="italic", zorder=0)

    drawn = []
    for (ds, cond), sub in merged.groupby(["dataset", "condition"], sort=False):
        marker = _MARKER_MAP.get((ds, cond), "P")
        color = _COLOR_MAP.get((ds, cond), "#444444")
        drawn.append((ds, cond, marker, color))
        ax.scatter(sub[x_col].to_numpy(),
                    sub["channel_redundancy_mean"].to_numpy(),
                    s=80, c=color, marker=marker, edgecolors="black",
                    linewidths=0.5, alpha=0.85, zorder=3,
                    label=f"{ds}/{cond}")
        _annotate_methods(ax, sub[x_col].to_numpy(),
                           sub["channel_redundancy_mean"].to_numpy(),
                           sub["method"].to_numpy())

    # Anchors: visible & lwir per slice as small grey "+" symbols, no labels.
    if anchors_df is not None and not anchors_df.empty:
        ax.scatter(anchors_df[x_col].to_numpy(),
                    anchors_df["channel_redundancy_mean"].to_numpy(),
                    s=70, c="#888888", marker="+", linewidths=1.4,
                    alpha=0.65, zorder=2)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(x_label)
    ax.set_ylabel("channel_redundancy (inter-channel mean |corr|)")
    ax.grid(True, alpha=0.3)

    if show_legend:
        # Three stacked legends: dataset markers, reference lines, anchors.
        marker_handles = [
            Line2D([0], [0], marker=m, color="w", markerfacecolor=c,
                    markeredgecolor="black", markersize=9, label=f"{ds}/{cond}")
            for ds, cond, m, c in drawn
        ]
        ref_handles = [
            Line2D([0], [0], color="gray", linestyle="--", alpha=0.6,
                    label="thermal axis = 0.5"),
            Line2D([0], [0], color="gray", linestyle=":", alpha=0.7,
                    label="ch_redund ≈ 0.8 (natural RGB)"),
            Line2D([0], [0], marker="+", color="#888888", linestyle="",
                    markersize=10, markeredgewidth=1.4,
                    label="visible / lwir anchors"),
        ]
        leg1 = ax.legend(handles=marker_handles, loc="lower left", fontsize=8,
                          title="dataset / condition", framealpha=0.9)
        ax.add_artist(leg1)
        ax.legend(handles=ref_handles, loc="upper left", fontsize=7,
                   framealpha=0.85)
    return drawn


def plot_multi_axis(merged: pd.DataFrame, anchors_df: pd.DataFrame | None,
                    out_png: Path, x_axes: list[str]) -> None:
    """Side-by-side panels, one per x-axis variant."""
    n = len(x_axes)
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 7.5), sharey=True)
    if n == 1:
        axes = [axes]

    for i, ax_name in enumerate(x_axes):
        col, label = _X_AXIS_OPTIONS[ax_name]
        if col not in merged.columns:
            axes[i].set_title(f"{ax_name}: column missing")
            continue
        _plot_latent_2d_axis(axes[i], merged, col, label,
                              anchors_df, show_legend=(i == 0))
        axes[i].set_title(f"x = {ax_name}", fontsize=11)

    fig.suptitle("Latent contribution map  —  thermal axis × channel_redundancy",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interpretive summary
# ---------------------------------------------------------------------------

def _per_slice_top(merged: pd.DataFrame, x_col: str, k: int = 3) -> pd.DataFrame:
    """Top-k methods by mAP per slice; cheap, dataset-relative ranking."""
    if "metric_value" not in merged.columns:
        return pd.DataFrame()
    rows = []
    for (ds, cond), g in merged.groupby(["dataset", "condition"]):
        g2 = g.dropna(subset=["metric_value"]).sort_values("metric_value",
                                                            ascending=False)
        for rank, (_, r) in enumerate(g2.head(k).iterrows(), 1):
            rows.append({
                "dataset": ds, "condition": cond, "rank": rank,
                "method": r["method"],
                "x": float(r[x_col]),
                "ch_redund": float(r["channel_redundancy_mean"]),
                "metric_value": float(r["metric_value"]),
            })
    return pd.DataFrame(rows)


def _per_slice_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Pearson r within each slice between the two latent axes and mAP."""
    if "metric_value" not in merged.columns:
        return pd.DataFrame()
    rows = []
    for (ds, cond), g in merged.groupby(["dataset", "condition"]):
        g2 = g.dropna(subset=["metric_value"])
        if len(g2) < 3:
            continue
        def _corr(a, b):
            return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 else float("nan")
        rows.append({
            "dataset": ds, "condition": cond, "n": len(g2),
            "r(latent_z, mAP)":  _corr(g2["latent_z_mean"], g2["metric_value"]),
            "r(thermal_share_raw, mAP)": _corr(g2["thermal_share_raw"], g2["metric_value"]),
            "r(thermal_share_reg, mAP)": _corr(g2["thermal_share_reg"], g2["metric_value"]),
            "r(ch_redund, mAP)": _corr(g2["channel_redundancy_mean"], g2["metric_value"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D latent scatter (thermal axis × channel_redundancy) across datasets.",
    )
    p.add_argument("--slices", nargs="+", default=list(_DEFAULT_SLICES),
                    help="dataset:condition pairs (default: kaist:day kaist:night llvip:night).")
    p.add_argument("--reports-dir", default=str(_DEFAULT_REPORTS),
                    help="Directory with the *_methods_overview.csv files.")
    p.add_argument("--report-dir", default=None,
                    help="Output directory (default: <reports-dir>/latent2d).")
    p.add_argument("--equalization-vis", default="none",
                    help="Filter overview rows by eq_vis (default: none).")
    p.add_argument("--equalization-th", default="none",
                    help="Filter overview rows by eq_th (default: none).")
    p.add_argument("--x-axis", nargs="+",
                    choices=list(_X_AXIS_OPTIONS.keys()),
                    default=["latent_z", "raw", "reg"],
                    help="X-axis variant(s) to plot (one panel per axis). "
                         "Default emits all three side-by-side so the "
                         "calibration vs portable comparison is immediate.")
    p.add_argument("--metric", default="mAP50-95",
                    choices=["P", "R", "mAP50", "mAP50-95"],
                    help="Training metric used in the interpretive summary.")
    p.add_argument("--reducer", default="p90", choices=["mean", "p90", "median"],
                    help="Per-method training-metric reducer (default: p90).")
    p.add_argument("--training-csv", default=str(DEFAULT_CSV_PATH),
                    help="Raw training results CSV.")
    return p.parse_args()


def _equalization_tag(eq_vis: str, eq_th: str) -> str:
    if eq_vis == "none" and eq_th == "none":
        return "no_equalization"
    if eq_vis == "clahe" and eq_th == "clahe":
        return "rgb_th_equalization"
    if eq_vis == "clahe":
        return "rgb_equalization"
    if eq_th == "clahe":
        return "th_equalization"
    return f"{eq_vis}_{eq_th}"


def main() -> int:
    args = _parse_args()
    slices = _parse_slices(args.slices)
    reports_dir = Path(args.reports_dir)
    report_dir = Path(args.report_dir) if args.report_dir else reports_dir / "latent2d"
    report_dir.mkdir(parents=True, exist_ok=True)
    training_csv = Path(args.training_csv)
    eq_tag_overview = _equalization_tag(args.equalization_vis, args.equalization_th)

    real_frames: list[pd.DataFrame] = []
    anchor_frames: list[pd.DataFrame] = []
    for ds, cond in slices:
        try:
            df = _load_overview(reports_dir, ds, cond,
                                 args.equalization_vis, args.equalization_th)
        except FileNotFoundError as exc:
            print(f"[latent2d] skip: {exc}", file=sys.stderr)
            continue
        if df.empty:
            continue
        df["metric_value"] = _load_training_metric(
            df, ds, cond, eq_tag_overview, args.metric, args.reducer, training_csv,
        )
        anchor_mask = df["method"].isin(_ANCHOR_METHODS)
        anchor_frames.append(df[anchor_mask].reset_index(drop=True))
        real_frames.append(df[~anchor_mask].reset_index(drop=True))

    if not real_frames:
        print("[latent2d] no data — nothing to plot", file=sys.stderr)
        return 2

    merged = pd.concat(real_frames, ignore_index=True)
    anchors = pd.concat(anchor_frames, ignore_index=True) if anchor_frames else None
    needed = ["latent_z_mean", "thermal_share_raw", "thermal_share_reg",
               "channel_redundancy_mean"]
    merged = merged.dropna(subset=[c for c in needed if c in merged.columns])

    tag = f"{eq_tag_overview}_{args.reducer}_{args.metric}".replace("/", "_")
    out_png = report_dir / f"latent2d_scatter_{tag}.png"
    out_csv = report_dir / f"latent2d_data_{tag}.csv"
    out_top = report_dir / f"latent2d_top3_{tag}.csv"
    out_corr = report_dir / f"latent2d_corr_{tag}.csv"

    keep_cols = [
        "dataset", "condition", "method",
        "latent_z_mean", "latent_z_std",
        "latent_z_alpha_dep_mean",
        "thermal_share_raw", "thermal_share_reg", "calibration_lift",
        "channel_redundancy_mean", "channel_redundancy_std",
        "contrib_confidence_mean", "contrib_std_mean",
        "metric_value",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged[keep_cols].to_csv(out_csv, index=False, float_format="%.6g")

    plot_multi_axis(merged, anchors, out_png, args.x_axis)

    # Companion summaries — these power the HTML interpretive block and are
    # cheap to recompute, so we always emit them next to the figure.
    top_df = _per_slice_top(merged, x_col="latent_z_mean", k=3)
    if not top_df.empty:
        top_df.to_csv(out_top, index=False, float_format="%.4f")
    corr_df = _per_slice_correlations(merged)
    if not corr_df.empty:
        corr_df.to_csv(out_corr, index=False, float_format="%.3f")

    print(f"[latent2d] {len(merged)} method-rows from {len(real_frames)} slices "
          f"(+ {0 if anchors is None else len(anchors)} anchor rows)")
    print(f"[latent2d] CSV         -> {out_csv}")
    print(f"[latent2d] plot        -> {out_png}")
    if not top_df.empty:
        print(f"[latent2d] top-3 CSV   -> {out_top}")
    if not corr_df.empty:
        print(f"[latent2d] corr CSV    -> {out_corr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
