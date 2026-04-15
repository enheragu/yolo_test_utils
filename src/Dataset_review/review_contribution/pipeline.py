from __future__ import annotations

"""
Main entry point for RGB/LWIR contribution analysis.

Orchestrates dataset loading, calibration generation with multiple variants,
and fusion method evaluation via a simple CLI.
"""

import argparse
import os
import pickle
import hashlib
from pathlib import Path
import sys
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from Dataset_review.review_contribution.settings import PRESET_SETTINGS
from utils.log_utils import log, logCoolMessage, logTable
from .calibration import build_contribution_calibration_from_dataset
from .evaluation import CONTRIBUTION_METRIC_VERSION
from .contribution import (
    evaluate_fusion_methods_on_dataset,
    _iter_visible_images,
    _infer_condition,
    _split_calibration_eval,
    _get_default_fusion_methods,
)
from .training_results_check import (
    load_training_results,
    build_dataset_metrics,
    TRAINING_METRICS,
)


def _log_summary(summary: dict, output_path: str, filename: str) -> None:
    """Print and store a compact summary table using the shared logger."""
    if not summary:
        log("No samples to summarise.")
        return

    table_data = [["metric", "mean", "std", "n"]]
    for key, stats in summary.items():
        table_data.append([key, f"{stats['mean']:.4f}", f"{stats['std']:.4f}", str(stats['n'])])

    logTable(table_data, output_path, filename, screen=False, showindex=False)


def _summary_value(summary: dict, key: str) -> str:
    stats = summary.get(key)
    if not stats:
        return "-"
    return f"{stats['mean']:.4f}±{stats['std']:.4f}"


def _equalization_display_labels(variant: str) -> tuple[str, str]:
    """Map a combined equalization variant to visible/thermal labels."""
    mapping = {
        "no_equalization": ("none", "none"),
        "th_equalization": ("none", "clahe"),
        "rgb_equalization": ("clahe", "none"),
        "rgb_th_equalization": ("clahe", "clahe"),
    }
    return mapping.get(variant, (variant, variant))


def _path_bucket(visible_path: str, dataset_root: str) -> str:
    """Return the top-level dataset bucket used for stratified sampling."""
    root = Path(dataset_root).resolve()
    path = Path(visible_path).resolve()
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    return rel_parts[0] if rel_parts else "root"


def _stratified_sample_paths(visible_paths: list[str], dataset_root: str, max_images: int | None, seed: int) -> list[str]:
    """Select a deterministic, roughly balanced subset across top-level dataset buckets."""
    if max_images is None or max_images >= len(visible_paths):
        return list(visible_paths)

    buckets: dict[str, list[str]] = {}
    for visible_path in visible_paths:
        bucket = _path_bucket(visible_path, dataset_root)
        buckets.setdefault(bucket, []).append(visible_path)

    rng = np.random.default_rng(seed)
    for bucket_paths in buckets.values():
        rng.shuffle(bucket_paths)

    bucket_names = sorted(buckets.keys())
    sampled: list[str] = []
    while len(sampled) < max_images:
        progressed = False
        for bucket_name in bucket_names:
            bucket_paths = buckets[bucket_name]
            if not bucket_paths:
                continue
            sampled.append(bucket_paths.pop())
            progressed = True
            if len(sampled) >= max_images:
                break
        if not progressed:
            break

    return sampled


def _method_overview_rows(report: dict) -> list[list[str]]:
    """Build method overview table, sorted by latent_z (descending: highest first)."""
    rows = []
    for method_name, method_report in report["methods"].items():
        summary = method_report.get("summary", {})
        eq_vis = method_report.get("eq_vis", "-")
        eq_th = method_report.get("eq_th", "-")
        latent_z_str = _summary_value(summary, "latent_z")
        try:
            latent_z_val = float(latent_z_str.split('±')[0])
        except (ValueError, IndexError):
            latent_z_val = 0.0
        rows.append({
            "name": method_report.get("method_name", method_name),
            "eq_vis": eq_vis,
            "eq_th": eq_th,
            "n_images": str(method_report.get("n_images", 0)),
            "cont_vis": _summary_value(summary, "cont_vis"),
            "cont_vis_raw": _summary_value(summary, "cont_vis_raw"),
            "cont_vis_reg": _summary_value(summary, "cont_vis_reg"),
            "cont_vis_mi": _summary_value(summary, "cont_vis_mi"),
            "cont_vis_ssim": _summary_value(summary, "cont_vis_ssim"),
            "cont_vis_grad_combined": _summary_value(summary, "cont_vis_grad_combined"),
            "cont_vis_spectral": _summary_value(summary, "cont_vis_spectral"),
            "cont_vis_freq": _summary_value(summary, "cont_vis_freq"),
            "latent_z": latent_z_str,
            "latent_z_val": latent_z_val,
            "contrib_std": _summary_value(summary, "contrib_std"),
            "contrib_confidence": _summary_value(summary, "contrib_confidence"),
        })

    # Sort by latent_z in descending order (highest first: lwir at top, visible at bottom)
    rows.sort(key=lambda x: x["latent_z_val"], reverse=True)

    # Build final table with header
    table_data = [[
        "method",
        "eq_vis",
        "eq_th",
        "n_images",
        "cont_vis",
        "cont_vis_raw",
        "reg",
        "mi",
        "ssim",
        "grad",
        "spectral",
        "freq",
        "latent_z",
        "contrib_std",
        "conf",
    ]]
    for row in rows:
        table_data.append([
            row["name"],
            row["eq_vis"],
            row["eq_th"],
            row["n_images"],
            row["cont_vis"],
            row["cont_vis_raw"],
            row["cont_vis_reg"],
            row["cont_vis_mi"],
            row["cont_vis_ssim"],
            row["cont_vis_grad_combined"],
            row["cont_vis_spectral"],
            row["cont_vis_freq"],
            row["latent_z"],
            row["contrib_std"],
            row["contrib_confidence"],
        ])
    return table_data


def _fusion_timing_rows(report: dict) -> list[list[str]]:
    """Build a dedicated table for fusion execution times."""
    rows = []
    for method_name, method_report in report["methods"].items():
        summary = method_report.get("summary", {})
        timing_stats = summary.get("fusion_time_ms")
        if not timing_stats:
            continue
        eq_vis = method_report.get("eq_vis", "-")
        eq_th = method_report.get("eq_th", "-")
        rows.append({
            "name": method_report.get("method_name", method_name),
            "eq_vis": eq_vis,
            "eq_th": eq_th,
            "n_images": str(method_report.get("n_images", 0)),
            "fusion_time_ms_mean": f"{float(timing_stats.get('mean', 0.0)):.4f}",
            "fusion_time_ms_median": f"{float(timing_stats.get('median', timing_stats.get('mean', 0.0))):.4f}",
            "fusion_time_ms_std": f"{float(timing_stats.get('std', 0.0)):.4f}",
            "fusion_time_ms_mean_sort": float(timing_stats.get("mean", 0.0)),
        })

    rows.sort(key=lambda item: item["fusion_time_ms_mean_sort"])

    table_data = [["method", "eq_vis", "eq_th", "n_images", "ms_mean", "ms_median", "ms_std"]]
    for row in rows:
        table_data.append([
            row["name"],
            row["eq_vis"],
            row["eq_th"],
            row["n_images"],
            row["fusion_time_ms_mean"],
            row["fusion_time_ms_median"],
            row["fusion_time_ms_std"],
        ])
    return table_data


def _combine_variant_reports(variant_reports: dict[str, dict]) -> dict:
    """Merge per-equalization evaluation reports into one combined report."""
    combined_methods = {}
    combined_skipped = {}
    n_calibration_by_variant = {}
    calibrations = {}
    n_images_by_variant = {}
    total_images = 0

    for variant, report in variant_reports.items():
        eq_vis, eq_th = _equalization_display_labels(variant)
        n_calibration_by_variant[variant] = int(report.get("n_calibration", 0))
        calibrations[variant] = report.get("calibration")
        n_images_by_variant[variant] = int(report.get("n_images", 0))
        total_images += int(report.get("n_images", 0))

        for method_name, method_report in report.get("methods", {}).items():
            combined_key = f"{method_name}__{variant}"
            combined_methods[combined_key] = {
                **method_report,
                "method_name": method_name,
                "equalization_variant": variant,
                "eq_vis": eq_vis,
                "eq_th": eq_th,
            }

        for method_name, reason in report.get("skipped_methods", {}).items():
            combined_skipped[f"{method_name}__{variant}"] = f"{variant}: {reason}"

    return {
        "dataset": next(iter(variant_reports.values())).get("dataset", "unknown") if variant_reports else "unknown",
        "condition": next(iter(variant_reports.values())).get("condition", "all") if variant_reports else "all",
        "n_calibration": n_calibration_by_variant,
        "n_calibration_by_variant": n_calibration_by_variant,
        "n_images": total_images,
        "n_images_by_variant": n_images_by_variant,
        "calibration": calibrations,
        "equalization_variants": list(variant_reports.keys()),
        "methods": combined_methods,
        "skipped_methods": combined_skipped,
    }


def _print_method_metric_legend(output_path: str) -> None:
    """Print an interpretation legend for overview metrics."""
    legend_rows = [
        ["cont_vis", "Primary visible contribution in [0,100]. Disagreement-aware weighted aggregation of 6 proxies (shrunk toward 50 when proxies disagree strongly)."],
        ["cont_vis_raw", "Unweighted proxy mean in [0,100]. Useful for traceability and A/B comparisons against weighted aggregation."],
        ["reg", "Per-channel NNLS regression proxy. Estimates visible weight per channel independently (avoids flatten bias from 3-band visible vs 1-band LWIR)."],
        ["mi", "Per-channel unique mutual information proxy. Measures source-specific shared info per channel independently (avoids flatten bias). Uses 50k subsampled pixels for efficiency."],
        ["ssim", "Multichannel SSIM proxy. Compares SSIM(fused, visible) vs SSIM(fused, thermal) and maps difference via sigmoid."],
        ["grad", "Combined gradient proxy (50% magnitude + 50% orientation). Merged to avoid over-weighting structural texture (was 2 separate proxies = 40% of score, now 1 proxy = 1/6)."],
        ["spectral", "Inter-channel independence proxy. Measures whether fused channels carry independent info (like visible's 3 spectral bands) or redundant info (like replicated LWIR). Permutation-invariant."],
        ["freq", "FFT magnitude spectrum correlation proxy. Compares frequency-domain profile of fused vs each source. Complementary to spatial proxies (LWIR = low-freq, visible = high-freq)."],
        ["latent_z", "Calibrated thermal fraction in [0,1] from synthetic alpha mixtures using cont_vis. 0 = visible endpoint, 1 = thermal endpoint."],
        ["contrib_std", "Std deviation across 6 proxy contributions. Low = agreement/high confidence; high = disagreement/lower confidence."],
        ["conf", "Proxy-agreement confidence in [0,100] used by weighted aggregation. High = stable proxy consensus; low = stronger shrink toward neutral 50."],
        ["n_images", "Number of evaluated images for that method. Compare methods using similar n_images for fairness."],
    ]

    log("Metric overview info:")
    log("  · Scale note: cont_vis/cont_vis_raw/reg/mi/ssim/grad/spectral/freq are visible-oriented (higher => more visible), latent_z is thermal-oriented (higher => more thermal).")
    log("  · Confidence note: prefer methods with coherent proxies (lower contrib_std) and enough samples (n_images).")
    for row in legend_rows:
        log(f"  · {row[0]}: {row[1]}")
    

def _print_method_report(report: dict, output_path: str) -> None:
    """Print a compact report for all evaluated fusion methods."""
    report_tag = f"{report['dataset']}_{report['condition']}"
    logCoolMessage("Review Contribution Report")
    log(f"Dataset: {report['dataset']}")
    log(f"Dataset condition: {report['condition']}")
    if isinstance(report.get("n_calibration"), dict):
        log(f"Calibration samples by variant: {report['n_calibration']}")
    else:
        log(f"Calibration samples: {report['n_calibration']}")
    if isinstance(report.get("n_images_by_variant"), dict):
        log(f"Reference images by variant: {report['n_images_by_variant']}")
    else:
        log(f"Reference images: {report['n_images']}")
    calibration_value = report.get("calibration")
    if isinstance(calibration_value, dict):
        calibration_available = any(value is not None for value in calibration_value.values())
    else:
        calibration_available = calibration_value is not None
    log(f"Calibration available: {calibration_available}")
    log("")

    for method_name, method_report in report["methods"].items():
        _log_summary(method_report["summary"], output_path, f"{report_tag}_{method_name}_summary")

    timing_table = _fusion_timing_rows(report)
    if len(timing_table) > 1:
        logCoolMessage("Fusion timing overview")
        logTable(timing_table, output_path, f"{report_tag}_fusion_timing_overview", screen=True, showindex=False)

    logCoolMessage("Method overview")
    _print_method_metric_legend(output_path)
    logTable(_method_overview_rows(report), output_path, f"{report_tag}_method_overview", screen=True, showindex=False)

    if report.get("skipped_methods"):
        logCoolMessage("Skipped methods")
        for method_name, reason in report["skipped_methods"].items():
            log(f"{method_name}: {reason}")
        log("")


PROXY_KEYS = ["cont_vis_reg", "cont_vis_mi", "cont_vis_ssim",
              "cont_vis_grad_combined", "cont_vis_spectral", "cont_vis_freq"]
PROXY_LABELS = ["reg", "mi", "ssim", "grad", "spectral", "freq"]
PROXY_COLORS = plt.cm.Set2(np.linspace(0, 1, len(PROXY_KEYS)))


def _save_plot(fig, output_path: str, filename: str) -> None:
    """Save a figure to disk, close it, and log the path."""
    plot_path = os.path.join(output_path, f"{filename}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Plot saved: {plot_path}")


def _extract_calibration_per_alpha(calibration: dict) -> list[dict] | None:
    """Extract per-alpha-group statistics from calibration diagnostic data.

    Returns a list of dicts with keys: alpha, vis_ideal, mask, cont_vis,
    cont_vis_std, and each proxy key with its _std variant.
    Returns None if diagnostic data is missing.
    """
    alpha_points = calibration.get("raw_points_alpha")
    cont_vis_points = calibration.get("raw_points_cont_vis")
    if alpha_points is None or cont_vis_points is None:
        return None

    unique_alphas = np.sort(np.unique(alpha_points))
    nodes = []
    for alpha in unique_alphas:
        mask = np.abs(alpha_points - alpha) < 1e-8
        node = {
            "alpha": alpha,
            "vis_ideal": (1.0 - alpha) * 100,
            "n_samples": int(mask.sum()),
            "cont_vis": float(np.median(cont_vis_points[mask])),
            "cont_vis_std": float(np.std(cont_vis_points[mask])),
        }
        for pkey in PROXY_KEYS:
            pdata = calibration.get(f"raw_points_{pkey}")
            if pdata is not None:
                node[pkey] = float(np.median(pdata[mask]))
                node[pkey + "_std"] = float(np.std(pdata[mask]))
            else:
                node[pkey] = 0.0
                node[pkey + "_std"] = 0.0
        nodes.append(node)
    return nodes


def _extract_method_rows(report: dict) -> list[dict]:
    """Extract per-method proxy statistics from an evaluation report.

    Returns a list of dicts sorted by cont_vis with keys: name, cont_vis,
    each proxy key with its _std variant, and latent_z.
    """
    rows = []
    for method_key, method_report in report.get("methods", {}).items():
        summary = method_report.get("summary", {})
        cv_stats = summary.get("cont_vis")
        if not cv_stats:
            continue
        row = {
            "name": method_report.get("method_name", method_key),
            "cont_vis": cv_stats["mean"],
            "cont_vis_std": cv_stats.get("std", 0.0),
        }
        for pkey in PROXY_KEYS:
            stats = summary.get(pkey)
            row[pkey] = stats["mean"] if stats else float("nan")
            row[pkey + "_std"] = stats["std"] if stats else 0.0
        lz = summary.get("latent_z")
        row["latent_z"] = lz["mean"] if lz else float("nan")
        cvc = summary.get("cont_vis_calibrated")
        row["cont_vis_calibrated"] = cvc["mean"] if cvc else float("nan")
        row["cont_vis_calibrated_std"] = cvc["std"] if cvc else 0.0
        rows.append(row)
    rows.sort(key=lambda m: m["cont_vis"])
    return rows


def _plot_grouped_proxy_bars(ax, items: list[dict], labels: list[str],
                              ideal_vals: list[float] | None = None) -> None:
    """Draw grouped bar chart of proxy values on *ax*.

    Each item must have PROXY_KEYS (value) and PROXY_KEYS + '_std'.
    *labels* is the x-tick label per item.
    If *ideal_vals* is given, an ideal reference line is overlaid.
    """
    n = len(items)
    n_proxies = len(PROXY_KEYS)
    x = np.arange(n)
    bar_width = 0.8 / n_proxies

    for i, (pkey, plabel) in enumerate(zip(PROXY_KEYS, PROXY_LABELS)):
        vals = [it[pkey] for it in items]
        stds = [it[pkey + "_std"] for it in items]
        offset = (i - n_proxies / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, yerr=stds, label=plabel,
               color=PROXY_COLORS[i], edgecolor="white", linewidth=0.5,
               capsize=2, error_kw={"linewidth": 0.8})

    cv_vals = [it["cont_vis"] for it in items]
    ax.plot(x, cv_vals, "k-o", linewidth=2, markersize=6,
            label="cont_vis (weighted)", zorder=10)
    if ideal_vals is not None:
        ax.plot(x, ideal_vals, "r--D", linewidth=1.5, markersize=5,
                alpha=0.7, label="ideal (1-α)×100", zorder=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Visible contribution %")
    ax.legend(fontsize=8, ncol=4, loc="best")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)


# ---------- Public plot functions -----------------------------------------

def _plot_calibration_diagnostic(calibration: dict, report: dict, output_path: str,
                                  report_tag: str, variant: str = "no_equalization") -> None:
    """Plot calibration curves (blend + concat) with method positions overlaid."""
    by_type = calibration.get("calibrations_by_type", {})
    if not by_type:
        # Fallback: treat entire calibration as a single curve
        by_type = {"blend": calibration}

    n_types = len(by_type)
    fig, axes = plt.subplots(1, 1 + n_types, figsize=(8 * (1 + n_types), 7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # --- Left panel: Both calibration curves overlaid ---
    ax = axes[0]
    curve_styles = {"blend": ("red", "o", "Blend"), "concat": ("blue", "s", "Concat")}
    x_dense = np.linspace(0, 100, 500)

    for stype, cal in by_type.items():
        color, marker, label = curve_styles.get(stype, ("gray", "^", stype))
        raw_knots = np.asarray(cal.get("raw_knots", []))
        visible_knots = np.asarray(cal.get("visible_knots", []))
        if raw_knots.size == 0:
            continue
        thermal_knots = 1.0 - visible_knots
        ax.plot(raw_knots, thermal_knots, f"-{marker}", color=color, linewidth=2,
                markersize=7, label=f"{label} curve ({len(raw_knots)} knots)", zorder=4)
        y_dense = np.interp(x_dense, raw_knots, visible_knots)
        ax.plot(x_dense, 1.0 - y_dense, "--", color=color, alpha=0.3, linewidth=1, zorder=2)

    # Overlay method positions (using the combined latent_z which averages both curves)
    method_rows = _extract_method_rows(report) if report else []
    if method_rows:
        mx = [r["cont_vis"] for r in method_rows]
        my = [r["latent_z"] for r in method_rows]
        ax.scatter(mx, my, s=100, c="green", marker="*",
                   edgecolors="black", linewidths=0.5, label="Fusion methods", zorder=5)
        for r in method_rows:
            ax.annotate(r["name"], (r["cont_vis"], r["latent_z"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)

    ax.set_xlabel("cont_vis (raw visible contribution %)")
    ax.set_ylabel("latent_z (thermal fraction)")
    ax.set_title(f"Calibration curves [{variant}]")
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # --- Right panels: Per-alpha boxplot for each calibration type ---
    for i, (stype, cal) in enumerate(by_type.items()):
        ax2 = axes[1 + i]
        raw_points = cal.get("raw_points_cont_vis")
        alpha_points = cal.get("raw_points_alpha")
        if raw_points is None or alpha_points is None:
            ax2.set_title(f"{stype} — no data")
            continue
        color, _, label = curve_styles.get(stype, ("gray", "^", stype))
        unique_alphas = np.sort(np.unique(alpha_points))
        box_data = [raw_points[np.abs(alpha_points - a) < 1e-8] for a in unique_alphas]
        bp = ax2.boxplot(box_data, positions=unique_alphas.tolist(), widths=0.06, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        # Ideal line
        ax2.plot(unique_alphas, (1.0 - unique_alphas) * 100, "k--", alpha=0.4,
                 linewidth=1, label="ideal (1-α)×100")
        ax2.set_xlabel("alpha_lwir (thermal fraction)")
        ax2.set_ylabel("cont_vis (%)")
        title_kind = "blend fractions" if stype == "blend" else "channel-concat patterns"
        ax2.set_title(f"{label}: cont_vis per alpha ({title_kind})")
        ax2.set_xticks(unique_alphas.tolist())
        ax2.set_xticklabels([f"{a:.2f}" for a in unique_alphas], rotation=45, ha="right", fontsize=8)
        ax2.set_xlim(-0.1, 1.1)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_calibration_diagnostic_{variant}")


def _plot_per_proxy_calibration_curves(calibration: dict, report: dict, output_path: str,
                                        report_tag: str, variant: str = "no_equalization") -> None:
    """Plot individual calibration curves for each proxy metric.

    Shows how each proxy maps raw score → visible fraction, overlaid with the
    aggregate curve.  This visualises the per-proxy correction that compensates
    for metric saturation (e.g. grad/freq).
    """
    by_type = calibration.get("calibrations_by_type", {})
    active_cal = by_type.get("blend", calibration)
    proxy_cals = active_cal.get("proxy_calibrations")
    if not proxy_cals:
        return

    n_proxies = len(proxy_cals)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    method_rows = _extract_method_rows(report) if report else []

    for idx, (proxy_key, p_cal) in enumerate(proxy_cals.items()):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        label = PROXY_LABELS[PROXY_KEYS.index(proxy_key)] if proxy_key in PROXY_KEYS else proxy_key
        color = PROXY_COLORS[PROXY_KEYS.index(proxy_key)] if proxy_key in PROXY_KEYS else "gray"

        raw_knots = np.asarray(p_cal["raw_knots"])
        visible_knots = np.asarray(p_cal["visible_knots"])
        thermal_knots = 1.0 - visible_knots

        # Per-proxy calibration curve
        ivw_w = p_cal.get("ivw_weight_normalized", p_cal.get("ivw_weight"))
        cal_std = p_cal.get("calibrated_mean_std")
        extra_info = ""
        if ivw_w is not None:
            extra_info += f" w={ivw_w:.3f}"
        if cal_std is not None:
            extra_info += f" σ={cal_std:.3f}"
        ax.plot(raw_knots, visible_knots * 100, "-o", color=color, linewidth=2,
                markersize=5, label=f"{label}{extra_info}", zorder=4)

        # Ideal diagonal
        ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1, label="ideal (linear)")

        # Aggregate curve for reference
        agg_raw = np.asarray(active_cal.get("raw_knots", []))
        agg_vis = np.asarray(active_cal.get("visible_knots", []))
        if agg_raw.size > 0:
            ax.plot(agg_raw, agg_vis * 100, "--", color="gray", linewidth=1,
                    alpha=0.5, label="aggregate", zorder=2)

        # Per-alpha group medians for this proxy
        group_raw = p_cal.get("group_raw")
        group_vis = p_cal.get("group_visible")
        if group_raw is not None:
            ax.scatter(group_raw, np.asarray(group_vis) * 100, s=30, color=color,
                       alpha=0.4, zorder=3, marker="x")

        # Method positions on this proxy
        if method_rows:
            for m in method_rows:
                mx = m.get(proxy_key, m["cont_vis"])
                my = float(np.interp(mx, raw_knots, visible_knots)) * 100
                ax.scatter(mx, my, s=60, c="green", marker="*",
                           edgecolors="black", linewidths=0.3, zorder=5)
                ax.annotate(m["name"], (mx, my), textcoords="offset points",
                            xytext=(3, 3), fontsize=6, alpha=0.7)

        ax.set_xlabel(f"raw {label} (%)")
        ax.set_ylabel("calibrated visible (%)")
        ax.set_title(f"{label}: raw → calibrated")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_proxies, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(f"Per-proxy calibration curves [{variant}]", fontsize=14)
    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_per_proxy_calibration_curves_{variant}")


def _plot_proxy_overview(report: dict, output_path: str, report_tag: str) -> None:
    """Plot per-method proxy values (raw) + calibrated vs uncalibrated comparison.

    Panels:
      1. Grouped bars: raw per-proxy cont_vis per method (uncalibrated)
      2. Side-by-side raw cont_vis vs calibrated cont_vis (visible %) per method
      3. latent_z bar chart (final calibrated thermal fraction)
    """
    method_rows = _extract_method_rows(report)
    if not method_rows:
        return

    names = [m["name"] for m in method_rows]
    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(names) * 1.2), 16),
                              gridspec_kw={"height_ratios": [3, 2, 1]})

    _plot_grouped_proxy_bars(axes[0], method_rows, names)
    axes[0].set_title("Per-proxy visible contribution by fusion method (raw, uncalibrated)")

    # Panel 2: Raw vs calibrated visible % side-by-side
    ax_cmp = axes[1]
    x = np.arange(len(names))
    width = 0.38
    raw_vis = np.array([m["cont_vis"] for m in method_rows])
    cal_vis = np.array([m.get("cont_vis_calibrated", float("nan")) for m in method_rows])
    ax_cmp.bar(x - width / 2, raw_vis, width, label="raw cont_vis (uncalibrated)",
               color="#cccccc", edgecolor="black", linewidth=0.5)
    ax_cmp.bar(x + width / 2, cal_vis, width, label="calibrated cont_vis (IVW per-proxy)",
               color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax_cmp.set_ylabel("visible contribution (%)")
    ax_cmp.set_title("Raw vs calibrated visible contribution per method")
    ax_cmp.set_ylim(0, 105)
    ax_cmp.axhline(50, color="gray", linestyle="--", alpha=0.4)
    ax_cmp.legend(fontsize=8, loc="upper left")
    ax_cmp.grid(True, axis="y", alpha=0.3)

    # Panel 3: latent_z bar chart
    ax2 = axes[2]
    latent_vals = [m["latent_z"] for m in method_rows]
    bar_colors = plt.cm.RdYlBu_r(np.array(latent_vals))
    ax2.bar(x, latent_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("latent_z (thermal fraction)")
    ax2.set_title("Calibrated thermal fraction (latent_z)")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_proxy_overview")


def _plot_per_method_proxy_std(
    calibration: dict,
    report: dict,
    output_path: str,
    report_tag: str,
    variant: str,
) -> None:
    """Per-method proxy dispersion diagnostic.

    For each fusion method, plot the per-proxy σ of raw cont_vis values across
    images (how noisy that proxy is *for that method*).  Overlays a dashed
    horizontal reference equal to the calibration-derived σ (the σ that feeds
    the global IVW weight).  Methods whose bars sit above the reference line
    are noisier than the calibration assumed — a hint that the global IVW
    weight underestimates that proxy's uncertainty for that method.
    """
    method_rows = _extract_method_rows(report)
    if not method_rows:
        return
    by_type = calibration.get("by_type") or {}
    active_cal = by_type.get("blend") or calibration
    pcs = active_cal.get("proxy_calibrations") or {}
    if not pcs:
        return

    names = [m["name"] for m in method_rows]
    n = len(names)
    n_proxies = len(PROXY_KEYS)
    x = np.arange(n)
    bar_width = 0.9 / n_proxies

    fig, ax = plt.subplots(figsize=(max(14, n * 1.2), 6))
    for i, (pkey, plabel, pcolor) in enumerate(zip(PROXY_KEYS, PROXY_LABELS, PROXY_COLORS)):
        per_method_std = [m.get(pkey + "_std", 0.0) for m in method_rows]
        offsets = x + (i - n_proxies / 2) * bar_width + bar_width / 2
        ax.bar(offsets, per_method_std, bar_width, color=pcolor,
               edgecolor="black", linewidth=0.3, label=plabel)
        cal_std = pcs.get(pkey, {}).get("calibrated_mean_std")
        if cal_std is not None:
            # Dashed reference for this proxy's calibration σ
            left = offsets.min() - bar_width / 2
            right = offsets.max() + bar_width / 2
            ax.hlines(cal_std, left, right, colors=[pcolor], linestyles="--",
                      linewidth=1.2, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("σ of raw proxy cont_vis across images (per method)")
    ax.set_title(
        f"Per-method proxy dispersion  —  dashed line = calibration σ (global IVW) [{variant}]")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=n_proxies, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_per_method_proxy_std_{variant}")


def _plot_calibration_sigma_shape(
    calibration: dict,
    output_path: str,
    report_tag: str,
    variant: str,
) -> None:
    """Per-α σ curves for each proxy from calibration data.

    Surfaces the "variance shape" observation: for most proxies σ is smaller at
    the extremes (α≈0 or α≈1) and larger mid-range.  Shape differences between
    proxies are informative (flat σ ⇒ proxy is uniformly reliable across the
    whole α range; peaked σ ⇒ proxy loses resolution in mid-fusion).
    """
    by_type = calibration.get("by_type") or {}
    active_cal = by_type.get("blend") or calibration
    pcs = active_cal.get("proxy_calibrations") or {}
    if not pcs:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for pkey, plabel, pcolor in zip(PROXY_KEYS, PROXY_LABELS, PROXY_COLORS):
        entry = pcs.get(pkey)
        if not entry:
            continue
        alphas = entry.get("group_alphas_std")
        stds = entry.get("group_std_per_alpha")
        if alphas is None or stds is None or len(alphas) == 0:
            continue
        ivw_w = entry.get("ivw_weight_normalized", 0.0)
        ax.plot(alphas, stds, "o-", color=pcolor, linewidth=1.5,
                label=f"{plabel} (w={ivw_w:.2f})")

    ax.set_xlabel("α_lwir (thermal fraction in calibration blend)")
    ax.set_ylabel("σ of calibrated proxy within α-group")
    ax.set_title(f"Calibration σ shape vs α per proxy [{variant}]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_calibration_sigma_shape_{variant}")



def _plot_per_proxy_calibration(calibration: dict, output_path: str,
                                 report_tag: str, variant: str = "no_equalization") -> None:
    """Plot each proxy's response curve vs alpha to diagnose saturation."""
    alpha_points = calibration.get("raw_points_alpha")
    if alpha_points is None:
        return

    cont_vis_points = calibration.get("raw_points_cont_vis")
    all_keys = PROXY_KEYS + ["cont_vis"]
    all_labels = PROXY_LABELS + ["cont_vis"]
    all_colors = list(PROXY_COLORS) + [np.array([0.6, 0.9, 0.6, 1.0])]

    unique_alphas = np.sort(np.unique(alpha_points))
    ideal_visible = (1.0 - unique_alphas) * 100

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.flatten()

    for i, (pkey, plabel, color) in enumerate(zip(all_keys, all_labels, all_colors)):
        if i >= len(axes_flat) - 1:
            break
        ax = axes_flat[i]
        pdata = calibration.get(f"raw_points_{pkey}")
        if pdata is None and pkey == "cont_vis":
            pdata = cont_vis_points
        if pdata is None:
            ax.set_title(f"{plabel} — no data")
            continue

        box_data = []
        for alpha in unique_alphas:
            mask = np.abs(alpha_points - alpha) < 1e-8
            box_data.append(pdata[mask])

        bp = ax.boxplot(box_data, positions=unique_alphas.tolist(),
                        widths=0.06, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        medians = [float(np.median(d)) for d in box_data]
        ax.plot(unique_alphas, medians, "o-", color="black", linewidth=1.5,
                markersize=4, label="median")
        ax.plot(unique_alphas, ideal_visible, "r--", alpha=0.5, linewidth=1,
                label="ideal (1-α)×100")

        ax.set_xlabel("alpha (thermal fraction)")
        ax.set_ylabel("proxy value (%)")
        ax.set_title(plabel)
        ax.set_xticks(unique_alphas.tolist())
        ax.set_xticklabels([f"{a:.2f}" for a in unique_alphas], rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Dynamic range summary in last panel
    ax = axes_flat[-1]
    ax.axis("off")
    summary_lines = ["Dynamic range summary\n(median at α=0 → median at α=1)\n"]
    for plabel, pkey in zip(all_labels, all_keys):
        pdata = calibration.get(f"raw_points_{pkey}")
        if pdata is None and pkey == "cont_vis":
            pdata = cont_vis_points
        if pdata is None:
            continue
        mask_0 = np.abs(alpha_points) < 1e-8
        mask_1 = np.abs(alpha_points - 1.0) < 1e-8
        med_0 = float(np.median(pdata[mask_0])) if mask_0.any() else float("nan")
        med_1 = float(np.median(pdata[mask_1])) if mask_1.any() else float("nan")
        summary_lines.append(f"{plabel:>10s}: {med_0:5.1f} → {med_1:5.1f}  (Δ={abs(med_0 - med_1):.1f})")
    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"Per-proxy response to synthetic alpha mixtures [{variant}]",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_per_proxy_calibration_{variant}")


def _print_calibration_nodes_table(calibration: dict, output_path: str,
                                    report_tag: str, variant: str = "no_equalization") -> None:
    """Print a table with median proxy values per alpha step.

    When per-proxy calibrations are available, shows both the aggregate latent_z
    and the per-proxy calibrated latent_z (which corrects for metric saturation).
    """
    nodes = _extract_calibration_per_alpha(calibration)
    if not nodes:
        return

    raw_knots = np.asarray(calibration.get("raw_knots", []))
    visible_knots = np.asarray(calibration.get("visible_knots", []))
    proxy_cals = calibration.get("proxy_calibrations", {})

    has_proxy_cal = bool(proxy_cals)
    header = ["alpha", "vis_ideal", "cont_vis", *PROXY_LABELS, "lz_agg"]
    if has_proxy_cal:
        header.append("lz_per_proxy")
    rows = [header]
    for node in nodes:
        if raw_knots.size > 0 and visible_knots.size > 0:
            vis_frac = float(np.interp(node["cont_vis"], raw_knots, visible_knots))
            lz_agg = 1.0 - vis_frac
        else:
            lz_agg = float("nan")

        row = [
            f"{node['alpha']:.2f}",
            f"{node['vis_ideal']:.1f}",
            f"{node['cont_vis']:.1f}",
            *[f"{node[pk]:.1f}" for pk in PROXY_KEYS],
            f"{lz_agg:.4f}",
        ]

        if has_proxy_cal:
            # Per-proxy calibrated + IVW: calibrate each proxy, weight by reliability
            proxy_vis_fracs = []
            weights = []
            for pk in PROXY_KEYS:
                p_cal = proxy_cals.get(pk)
                if p_cal is None:
                    continue
                p_raw = np.asarray(p_cal["raw_knots"])
                p_vis = np.asarray(p_cal["visible_knots"])
                pval = node.get(pk, node["cont_vis"])
                proxy_vis_fracs.append(float(np.interp(pval, p_raw, p_vis)))
                weights.append(p_cal.get("ivw_weight_normalized",
                                          p_cal.get("ivw_weight", 1.0)))
            if proxy_vis_fracs:
                w = np.asarray(weights, dtype=np.float64)
                w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / w.size
                lz_pp = 1.0 - float(np.dot(w, np.asarray(proxy_vis_fracs)))
            else:
                lz_pp = lz_agg
            row.append(f"{lz_pp:.4f}")

        rows.append(row)

    n_samples = nodes[0]["n_samples"] if nodes else 0
    logCoolMessage(f"Calibration nodes [{variant}]  (median per alpha, n_images per alpha ≈ {n_samples})")
    logTable(rows, output_path, f"{report_tag}_calibration_nodes_{variant}", screen=True, showindex=False)


def _plot_calibration_nodes_overview(calibration: dict, output_path: str,
                                      report_tag: str, variant: str = "no_equalization") -> None:
    """Bar chart of proxy values per alpha step — like proxy_overview but for calibration nodes."""
    nodes = _extract_calibration_per_alpha(calibration)
    if not nodes:
        return

    labels = [f"α={n['alpha']:.2f}" for n in nodes]
    ideal_vals = [n["vis_ideal"] for n in nodes]

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(nodes) * 1.5), 12),
                              gridspec_kw={"height_ratios": [3, 1]})

    _plot_grouped_proxy_bars(axes[0], nodes, labels, ideal_vals=ideal_vals)
    axes[0].set_title(f"Per-proxy response at each calibration alpha [{variant}]")

    # Deviation from ideal
    ax2 = axes[1]
    x = np.arange(len(nodes))
    deviations = [n["cont_vis"] - n["vis_ideal"] for n in nodes]
    bar_colors = ["green" if d >= 0 else "red" for d in deviations]
    ax2.bar(x, deviations, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("cont_vis - ideal (%)")
    ax2.set_title("Deviation from ideal linear response")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_calibration_nodes_overview_{variant}")


def _print_calibration_report(calibrations: dict[str, dict | None], used_variants: set[str], output_path: str, report_tag: str) -> None:
    """Print and store a short calibration summary for the selected variants."""
    rows = [["variant", "status", "knots", "raw_range", "visible[min->max]", "thermal[min->max]"]]
    for variant, calibration in calibrations.items():
        if calibration is None:
            rows.append([variant, "missing", "-", "-", "-", "-"])
            continue

        raw_knots = np.asarray(calibration.get("raw_knots", []), dtype=np.float64)
        visible_knots = np.asarray(calibration.get("visible_knots", []), dtype=np.float64)
        thermal_knots = np.asarray(calibration.get("thermal_knots", []), dtype=np.float64)

        raw_range = f"{raw_knots.min():.2f}..{raw_knots.max():.2f}" if raw_knots.size else "-"
        vis_range = (
            f"{visible_knots[0]:.2f}->{visible_knots[-1]:.2f}"
            if visible_knots.size else "-"
        )
        thermal_range = (
            f"{thermal_knots[0]:.2f}->{thermal_knots[-1]:.2f}"
            if thermal_knots.size else "-"
        )
        status = "used" if variant in used_variants else "cache"
        rows.append([variant, status, str(raw_knots.size), raw_range, vis_range, thermal_range])

    logCoolMessage("Calibration variants")
    logTable(rows, output_path, f"{report_tag}_equalization_calibration_variants", screen=True, showindex=False)
    log(f"Evaluation used variants: {sorted(used_variants)}")
    log("")


def _plot_training_vs_latent(
    report: dict,
    output_path: str,
    report_tag: str,
    condition: str,
    equalization: str,
) -> None:
    """Plot training metrics (P, R, mAP50, mAP50-95) vs latent_z per fusion method.

    Creates a 2x2 grid with one panel per metric.  X-axis is latent_z (thermal
    fraction, 0=visible, 1=lwir), Y-axis is the training metric.  Each method
    is shown as a strip of individual-run points + mean marker.  This lets you
    see correspondence between the computed thermal/visible balance and the
    actual detection performance.

    Missing training data for a method (not in CSV) is silently skipped.
    """
    method_rows = _extract_method_rows(report)
    if not method_rows:
        return
    try:
        df = load_training_results()
    except FileNotFoundError as exc:
        log(f"Training results CSV not found, skipping training-vs-latent plot: {exc}")
        return

    method_names = [m["name"] for m in method_rows]
    metrics_by_method = build_dataset_metrics(df, method_names, condition, equalization)
    if not metrics_by_method:
        log(f"No training data matched condition={condition} eq={equalization}; skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, TRAINING_METRICS):
        xs_means, ys_means, names, xs_points, ys_points = [], [], [], [], []
        for m in method_rows:
            info = metrics_by_method.get(m["name"])
            if info is None or metric not in info["metrics"]:
                continue
            lz = m["latent_z"]
            if np.isnan(lz):
                continue
            mstats = info["metrics"][metric]
            xs_means.append(lz)
            ys_means.append(mstats["mean"])
            names.append(m["name"])
            for v in mstats["values"]:
                xs_points.append(lz)
                ys_points.append(v)

        # Individual runs as small points (strip)
        if xs_points:
            ax.scatter(xs_points, ys_points, s=18, color="#4c72b0", alpha=0.45,
                       edgecolors="none", zorder=2, label="individual runs")
        # Per-method mean markers
        if xs_means:
            ax.scatter(xs_means, ys_means, s=80, color="#c44e52", marker="D",
                       edgecolors="black", linewidths=0.5, zorder=4, label="mean per method")
            # Annotate
            for x, y, n in zip(xs_means, ys_means, names):
                ax.annotate(n, (x, y), textcoords="offset points",
                            xytext=(5, 4), fontsize=7, alpha=0.85)
            # Correlation line (Spearman/Pearson-informed trend)
            if len(xs_means) >= 3:
                order = np.argsort(xs_means)
                xs_sorted = np.array(xs_means)[order]
                ys_sorted = np.array(ys_means)[order]
                coeffs = np.polyfit(xs_sorted, ys_sorted, 1)
                xfit = np.linspace(xs_sorted.min(), xs_sorted.max(), 50)
                ax.plot(xfit, np.polyval(coeffs, xfit), "--", color="gray",
                        alpha=0.5, linewidth=1, zorder=3, label="linear fit")
                # Pearson r (linear) + Spearman ρ (rank-monotonic).
                # Disagreement between the two signals a non-linear or
                # outlier-driven relationship.
                xs_arr = np.asarray(xs_means, dtype=np.float64)
                ys_arr = np.asarray(ys_means, dtype=np.float64)
                pearson = float(np.corrcoef(xs_arr, ys_arr)[0, 1])
                if np.std(xs_arr) > 1e-12 and np.std(ys_arr) > 1e-12:
                    rx = np.argsort(np.argsort(xs_arr)).astype(np.float64)
                    ry = np.argsort(np.argsort(ys_arr)).astype(np.float64)
                    spearman = float(np.corrcoef(rx, ry)[0, 1])
                else:
                    spearman = 0.0
                ax.text(0.03, 0.97,
                        f"Pearson r = {pearson:+.3f}\nSpearman ρ = {spearman:+.3f}",
                        transform=ax.transAxes, fontsize=9, va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        ax.set_xlabel("latent_z (thermal fraction)")
        ax.set_ylabel(metric)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        f"Training metrics vs latent_z  —  {condition} / {equalization}",
        fontsize=14)
    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_training_vs_latent_{equalization}")


def main() -> None:
    """CLI entry point for KAIST/LLVIP dataset evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate visible/LWIR contribution on a dataset")
    parser.add_argument("--dataset", choices=["kaist", "llvip"], default="kaist")
    parser.add_argument("--dataset-root", default=None, help="Dataset root. If empty, uses defaults from Dataset.constants")
    parser.add_argument("--condition", choices=["all", "day", "night"], default="all")
    parser.add_argument("--preset", choices=["test", "fast", "balanced", "quality"], default="balanced", help="Runtime preset (test=10pct, fast=50pct, balanced=75pct, quality=100pct)")
    parser.add_argument("--methods", nargs="*", default=None, help="Subset of method names from Dataset.constants.dataset_options")
    parser.add_argument("--cache-file", default=None, help="Optional cache file for raw per-image per-method results")
    parser.add_argument("--reset-cache", action="store_true", help="Ignore existing cache and recompute everything")
    args = parser.parse_args()
    settings = PRESET_SETTINGS[args.preset]

    try:
        cv.setNumThreads(1)
    except Exception:
        pass

    from Dataset.constants import kaist_images_path, llvip_yolo_dataset_path

    if args.dataset == "llvip" and args.condition == "day":
        raise ValueError("LLVIP does not provide day split; use --condition night or --condition all")

    default_roots = {
        "kaist": kaist_images_path,
        "llvip": llvip_yolo_dataset_path,
    }
    dataset_root = args.dataset_root or default_roots[args.dataset]

    cache_file = args.cache_file
    if cache_file is None:
        cache_dir = Path.home() / ".cache" / "eeha_review_fusion_contribution"
        cache_file = str(cache_dir / f"review_image_contribution_{args.dataset}_{args.condition}.pkl")
    else:
        cache_dir = Path(cache_file).parent

    os.makedirs(str(cache_dir), exist_ok=True)
    report_dir = str(Path(cache_dir) / "reports")
    os.makedirs(report_dir, exist_ok=True)

    fusion_methods = _get_default_fusion_methods()
    if args.methods:
        fusion_methods = {name: fusion_methods[name] for name in args.methods if name in fusion_methods}

    # Calculate max_images from an absolute override first, then fall back to subsample ratio.
    visible_paths_all = []
    for visible_path in _iter_visible_images(dataset_root, args.dataset):
        if args.condition != "all" and _infer_condition(visible_path, args.dataset) != args.condition:
            continue
        visible_paths_all.append(visible_path)
    if settings.max_images is not None:
        max_images = max(1, min(int(settings.max_images), len(visible_paths_all)))
        sampling_note = f"absolute cap: {settings.max_images} (stratified by top-level folder, seed={settings.split_seed})"
    else:
        max_images = max(1, int(len(visible_paths_all) * settings.subsample_ratio)) if visible_paths_all else 0
        sampling_note = f"subsample ratio: {settings.subsample_ratio:.2f}"

    selected_visible_paths = _stratified_sample_paths(
        visible_paths_all,
        dataset_root=dataset_root,
        max_images=max_images,
        seed=settings.split_seed,
    )

    logCoolMessage("Run configuration")
    log(f"Dataset: {args.dataset} | condition: {args.condition} | preset: {args.preset}")
    log(f"Visible images found: {len(visible_paths_all)} | sampling: {sampling_note} | selected: {len(selected_visible_paths)}")
    log(f"Calibration ratio: {settings.calibration_ratio:.2f} | calibration_images: {settings.calibration_images}")
    log(f"Execution mode: {settings.execution_mode} | workers: {settings.workers} | alpha_steps: {settings.alpha_steps} | chunk: {settings.task_chunksize}")
    log(f"CPU count: {os.cpu_count()} | OpenCV threads forced to 1")
    log(f"Methods selected: {len(fusion_methods)}")
    log("")

    # Generate calibrations for each equalization variant
    log(f"Generating calibrations for variants: {settings.equalization_variants}")
    calibrations = {}
    for variant in settings.equalization_variants:
        calibration_protocol = {
            "metric_version": CONTRIBUTION_METRIC_VERSION,
            "dataset": args.dataset,
            "dataset_root": str(Path(dataset_root).resolve()),
            "condition": args.condition,
            "max_images": int(max_images),
            "calibration_images": settings.calibration_images,
            "calibration_ratio": float(settings.calibration_ratio),
            "split_seed": int(settings.split_seed),
            "alpha_steps": int(max(2, settings.alpha_steps)),
            "preset": args.preset,
            "equalization": variant,
        }
        protocol_raw = "|".join(f"{key}={value}" for key, value in sorted(calibration_protocol.items()))
        protocol_hash = hashlib.sha256(protocol_raw.encode("utf-8")).hexdigest()[:12]

        calibration_file = str(cache_dir / f"review_image_contribution_calibration_{args.dataset}_{args.condition}_{variant}_{protocol_hash}.pkl")
        calibration_samples_file = str(
            Path(cache_file).with_name(
                Path(cache_file).stem + f"_calibration_samples_{variant}_{protocol_hash}.pkl"
            )
        )

        if os.path.exists(calibration_file):
            with open(calibration_file, "rb") as file:
                calibrations[variant] = pickle.load(file)
            raw_knots = calibrations[variant].get("raw_knots", []) if calibrations[variant] else []
            log(f"[{variant}] Loaded calibration cache ({len(raw_knots)} knots): {calibration_file}")
        else:
            calibration_paths, _ = _split_calibration_eval(
                selected_visible_paths,
                calibration_ratio=settings.calibration_ratio,
                calibration_images=settings.calibration_images,
                seed=settings.split_seed,
            )

            calibrations[variant] = build_contribution_calibration_from_dataset(
                calibration_paths,
                alpha_grid=np.linspace(0.0, 1.0, max(2, settings.alpha_steps)),
                workers=settings.workers,
                execution_mode=settings.execution_mode,
                task_chunksize=settings.task_chunksize,
                calibration_samples_file=calibration_samples_file,
                equalization=variant,
            ) if calibration_paths else None

            if calibrations[variant] is not None:
                with open(calibration_file, "wb") as file:
                    pickle.dump(calibrations[variant], file)
                raw_knots = calibrations[variant].get("raw_knots", [])
                log(f"[{variant}] Saved calibration cache ({len(raw_knots)} knots): {calibration_file}")

    variant_reports = {}
    for variant in settings.equalization_variants:
        variant_reports[variant] = evaluate_fusion_methods_on_dataset(
            dataset_root=dataset_root,
            dataset=args.dataset,
            fusion_methods=fusion_methods,
            visible_paths_override=selected_visible_paths,
            condition=args.condition,
            max_images=None,
            calibration_images=settings.calibration_images,
            calibration_ratio=settings.calibration_ratio,
            split_seed=settings.split_seed,
            workers=settings.workers,
            execution_mode=settings.execution_mode,
            task_chunksize=settings.task_chunksize,
            calibration_samples_file=None,  # already cached during calibration phase
            cache_file=cache_file,
            reset_cache=args.reset_cache,
            calibration=calibrations.get(variant),
            equalization=variant,
        )

    combined_report = _combine_variant_reports(variant_reports)

    if len(calibrations) > 1:
        log(f"Note: Generated {len(calibrations)} calibration/evaluation variants: {list(calibrations.keys())}")
    _print_calibration_report(calibrations, set(variant_reports.keys()), report_dir, f"{args.dataset}_{args.condition}")

    log(f"Dataset root: {dataset_root}")
    _print_method_report(combined_report, report_dir)

    # Diagnostic plots
    report_tag = f"{args.dataset}_{args.condition}"
    for variant, calibration in calibrations.items():
        if calibration is None:
            continue
        # Combined calibration diagnostic (both curves overlaid)
        _plot_calibration_diagnostic(calibration, combined_report, report_dir, report_tag, variant)
        # Per-proxy calibration curves (shows individual metric corrections)
        _plot_per_proxy_calibration_curves(calibration, combined_report, report_dir, report_tag, variant)
        # Per-method proxy σ diagnostic (reveals method-specific proxy noise
        # that the global IVW weight cannot capture)
        _plot_per_method_proxy_std(calibration, combined_report, report_dir, report_tag, variant)
        # σ-vs-α shape per proxy (shows whether dispersion peaks mid-fusion)
        _plot_calibration_sigma_shape(calibration, report_dir, report_tag, variant)
        # Per calibration type: table, per-proxy, and node overview
        by_type = calibration.get("calibrations_by_type", {"blend": calibration})
        for stype, sub_cal in by_type.items():
            type_tag = f"{variant}_{stype}"
            _print_calibration_nodes_table(sub_cal, report_dir, report_tag, type_tag)
            _plot_per_proxy_calibration(sub_cal, report_dir, report_tag, type_tag)
            _plot_calibration_nodes_overview(sub_cal, report_dir, report_tag, type_tag)
    _plot_proxy_overview(combined_report, report_dir, report_tag)

    # Training-vs-latent correspondence plots.  One per (variant × condition).
    # If args.condition == "all" we emit both day and night when data exists.
    conditions_to_plot = ["day", "night"] if args.condition == "all" else [args.condition]
    for variant, report in variant_reports.items():
        for cond in conditions_to_plot:
            cond_tag = f"{args.dataset}_{cond}"
            _plot_training_vs_latent(report, report_dir, cond_tag, cond, variant)


if __name__ == "__main__":
    main()
