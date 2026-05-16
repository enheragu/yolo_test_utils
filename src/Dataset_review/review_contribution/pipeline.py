from __future__ import annotations

"""
Main entry point for RGB/LWIR contribution analysis.

Orchestrates dataset loading, calibration generation with multiple variants,
and fusion method evaluation via a simple CLI.
"""

import argparse
import csv
import json
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
from .data_loading import (
    iter_visible_images,
    infer_condition,
    resolve_dataset_root,
)
from .evaluation import (
    PROXY_VERSION,
    CALIBRATION_SAMPLES_VERSION,
    CALIBRATION_FIT_VERSION,
)
from .contribution import (
    evaluate_fusion_methods_on_dataset,
    _split_calibration_eval,
    _get_default_fusion_methods,
)
from .training_results_check import (
    load_training_results,
    build_dataset_metrics,
    TRAINING_METRICS,
)


def _log_summary(summary: dict, output_path: str, filename: str) -> None:
    """Print and store a compact summary table using the shared logger.

    Aggregated summaries contain heterogeneous entries:
      - standard {mean, median, std, n} stats dicts for scalar metrics;
      - per-proxy dict-of-dicts (e.g. ivw_weights_effective_mean) → expanded
        as one row per proxy with key "<metric>[<proxy>]";
      - plain scalars (e.g. ivw_converged_rate) → rendered as a single value.
    """
    if not summary:
        log("No samples to summarise.")
        return

    table_data = [["metric", "mean", "std", "n"]]
    for key, stats in summary.items():
        if isinstance(stats, dict) and "mean" in stats:
            table_data.append([key, f"{stats['mean']:.4f}", f"{stats['std']:.4f}", str(stats['n'])])
        elif isinstance(stats, dict):
            # Per-proxy nested stats (ivw_weights_effective_mean, ...)
            for proxy_key, proxy_stats in stats.items():
                if not isinstance(proxy_stats, dict) or "mean" not in proxy_stats:
                    continue
                table_data.append([
                    f"{key}[{proxy_key}]",
                    f"{proxy_stats['mean']:.4f}",
                    f"{proxy_stats['std']:.4f}",
                    str(proxy_stats['n']),
                ])
        elif isinstance(stats, (int, float)):
            table_data.append([key, f"{float(stats):.4f}", "-", "-"])

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
    """Return the stratification bucket for *visible_path*.

    Picks the deepest path component that is still *above* the ``visible`` leaf
    — for KAIST originals (``setXX/VNNN/visible/I*.jpg``) this yields
    ``setXX``; for LLVIP originals (``visible/{train,test}/*.jpg``) it yields
    ``train`` / ``test``.  Falls back to the first relative component when no
    ``visible`` segment is present, preserving old behavior for ad-hoc layouts.
    """
    root = Path(dataset_root).resolve()
    path = Path(visible_path).resolve()
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    if not rel_parts:
        return "root"
    # LLVIP originals start with ``visible/`` — skip it to bucket by split
    # (``train``/``test``).  KAIST originals start with ``setXX/`` directly.
    if rel_parts[0] == "visible" and len(rel_parts) > 1:
        return rel_parts[1]
    return rel_parts[0]


def _stratified_sample_paths(visible_paths: list[str], dataset_root: str, max_images: int | None, seed: int) -> list[str]:
    """Deterministic proportional-stratified subsample across top-level buckets.

    Per-bucket quota ``n_k`` ≈ ``max_images · |B_k| / N`` using largest-remainder
    rounding so ``sum(n_k) == max_images`` exactly.  Within each bucket paths
    are shuffled with a seeded RNG, then the first ``n_k`` entries are kept.
    If a bucket lacks stock for its quota, the surplus is redistributed to
    buckets with remaining capacity.
    """
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
    total = len(visible_paths)

    # Largest-remainder apportionment.
    raw = {b: max_images * len(buckets[b]) / total for b in bucket_names}
    quotas = {b: int(raw[b]) for b in bucket_names}
    leftover = max_images - sum(quotas.values())
    for b in sorted(bucket_names, key=lambda n: raw[n] - quotas[n], reverse=True):
        if leftover <= 0:
            break
        quotas[b] += 1
        leftover -= 1

    # Respect each bucket's capacity; redistribute any shortfall round-robin.
    for b in bucket_names:
        quotas[b] = min(quotas[b], len(buckets[b]))
    shortfall = max_images - sum(quotas.values())
    while shortfall > 0:
        progressed = False
        for b in bucket_names:
            if quotas[b] < len(buckets[b]):
                quotas[b] += 1
                shortfall -= 1
                progressed = True
                if shortfall == 0:
                    break
        if not progressed:
            break

    sampled: list[str] = []
    for b in bucket_names:
        sampled.extend(buckets[b][:quotas[b]])
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
            "latent_z_alpha_dep": _summary_value(summary, "latent_z_alpha_dep"),
            "contrib_std": _summary_value(summary, "contrib_std"),
            "contrib_confidence": _summary_value(summary, "contrib_confidence"),
            "channel_redundancy": _summary_value(summary, "channel_redundancy"),
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
        "latent_z_αdep",
        "contrib_std",
        "conf",
        "ch_redund",
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
            row["latent_z_alpha_dep"],
            row["contrib_std"],
            row["contrib_confidence"],
            row["channel_redundancy"],
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
        ["latent_z_αdep", "Same as latent_z but using α-dependent IVW: per-proxy σ is interpolated at the current α̂ estimate (iterative reweighting). Differences from latent_z reveal σ-heterogeneity along α."],
        ["contrib_std", "Std deviation across 6 proxy contributions. Low = agreement/high confidence; high = disagreement/lower confidence."],
        ["conf", "Proxy-agreement confidence in [0,100] used by weighted aggregation. High = stable proxy consensus; low = stronger shrink toward neutral 50."],
        ["ch_redund", "Channel redundancy in [0,1]: reliability-weighted composite of global Pearson, global Spearman, tile-robust Pearson and covariance effective-rank redundancy. 1 = duplicated channels, lower = more independent cross-channel content."],
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


# ---------------------------------------------------------------------------
# Tabular / structured artifact writers
# ---------------------------------------------------------------------------
# These CSV/JSON dumps mirror the plots so downstream review (including future
# Claude sessions) can inspect the data numerically without re-running the
# whole evaluation or trying to read raster images.
# ---------------------------------------------------------------------------

def _write_csv(output_path: str, filename: str, header: list[str],
               rows: list[list]) -> None:
    """Write a CSV file with given header + rows, log the path."""
    csv_path = os.path.join(output_path, f"{filename}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)
    log(f"CSV saved: {csv_path}")


def _fmt(x, nd: int = 6) -> str:
    """Stable numeric formatting for CSV (empty for None/NaN)."""
    if x is None:
        return ""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return ""
    return f"{xf:.{nd}g}"


def _dump_methods_overview_csv(report: dict, output_path: str, report_tag: str) -> None:
    """Per-method summary as CSV (numeric, one row per method).

    Adds three cross-dataset-portable columns derived from the per-key means:
      - ``thermal_share_raw`` = 1 − cont_vis_raw/100 (no calibration; honest
        cross-dataset reference because PAVA is dataset-specific).
      - ``thermal_share_reg`` = 1 − reg/100 (per-channel NNLS thermal share;
        the most calibration-portable proxy — see PAVA-portability diagnostic
        in the README).
      - ``calibration_lift`` = latent_z − thermal_share_raw (how much the
        calibration shifts the answer for this method; absolute value
        quantifies saturation correction strength).
    """
    rows = []
    for method_name, method_report in report.get("methods", {}).items():
        summary = method_report.get("summary", {})
        row = [
            method_report.get("method_name", method_name),
            method_report.get("eq_vis", ""),
            method_report.get("eq_th", ""),
            method_report.get("n_images", 0),
        ]
        # Scalar summary fields: mean, std, median per key
        for key in ["cont_vis", "cont_vis_raw",
                    "cont_vis_reg", "cont_vis_mi", "cont_vis_ssim",
                    "cont_vis_grad_combined", "cont_vis_spectral", "cont_vis_freq",
                    "latent_z", "latent_z_alpha_dep",
                    "contrib_std", "contrib_confidence",
                "channel_redundancy", "ivw_iters"]:
            stats = summary.get(key) or {}
            row.extend([_fmt(stats.get("mean")), _fmt(stats.get("std")), _fmt(stats.get("median"))])
        row.append(_fmt(summary.get("ivw_converged_rate")))
        # Derived portable columns (single scalar each).
        cv_raw = (summary.get("cont_vis_raw") or {}).get("mean")
        cv_reg = (summary.get("cont_vis_reg") or {}).get("mean")
        lz_mean = (summary.get("latent_z") or {}).get("mean")
        ts_raw = 1.0 - float(cv_raw) / 100.0 if cv_raw is not None else None
        ts_reg = 1.0 - float(cv_reg) / 100.0 if cv_reg is not None else None
        cal_lift = (float(lz_mean) - ts_raw) if (lz_mean is not None and ts_raw is not None) else None
        row.extend([_fmt(ts_raw), _fmt(ts_reg), _fmt(cal_lift)])
        rows.append(row)
    header = ["method", "eq_vis", "eq_th", "n_images"]
    for key in ["cont_vis", "cont_vis_raw", "reg", "mi", "ssim",
                "grad", "spectral", "freq",
                "latent_z", "latent_z_alpha_dep",
                "contrib_std", "contrib_confidence",
            "channel_redundancy", "ivw_iters"]:
        header.extend([f"{key}_mean", f"{key}_std", f"{key}_median"])
    header.append("ivw_converged_rate")
    header.extend(["thermal_share_raw", "thermal_share_reg", "calibration_lift"])
    _write_csv(output_path, f"{report_tag}_methods_overview", header, rows)


def _dump_calibration_sigma_csv(calibration: dict, output_path: str,
                                 report_tag: str, variant: str) -> None:
    """Long-format CSV of per-proxy σ(α) from calibration data."""
    by_type = calibration.get("by_type") or calibration.get("calibrations_by_type") or {}
    active_cal = by_type.get("blend") or calibration
    pcs = active_cal.get("proxy_calibrations") or {}
    if not pcs:
        return
    rows = []
    for pkey, p_cal in pcs.items():
        alphas = np.asarray(p_cal.get("group_alphas_std", []), dtype=np.float64)
        sigmas = np.asarray(p_cal.get("group_std_per_alpha", []), dtype=np.float64)
        mean_sigma = p_cal.get("calibrated_mean_std")
        ivw_w = p_cal.get("ivw_weight_normalized", p_cal.get("ivw_weight"))
        for a, s in zip(alphas.ravel(), sigmas.ravel()):
            rows.append([pkey, _fmt(a), _fmt(s), _fmt(mean_sigma), _fmt(ivw_w)])
    if rows:
        _write_csv(output_path, f"{report_tag}_calibration_sigma_shape_{variant}",
                   ["proxy", "alpha", "sigma", "sigma_mean_global", "ivw_weight_global"],
                   rows)


def _dump_method_sigma_csv(calibration: dict, report: dict, output_path: str,
                            report_tag: str, variant: str) -> None:
    """Per-method per-proxy σ of raw cont_vis, alongside calibration σ at the
    method's latent_z (so differences are visible numerically)."""
    method_rows = _extract_method_rows(report)
    by_type = calibration.get("by_type") or calibration.get("calibrations_by_type") or {}
    active_cal = by_type.get("blend") or calibration
    pcs = active_cal.get("proxy_calibrations") or {}
    rows = []
    for m in method_rows:
        lz = m.get("latent_z")
        for pkey in PROXY_KEYS:
            sigma_empirical = m.get(pkey + "_std", 0.0)
            p_cal = pcs.get(pkey, {})
            # Interpolate calibration σ at this method's latent_z
            alpha_grid = np.asarray(p_cal.get("group_alphas_std", []),
                                     dtype=np.float64).ravel()
            sigma_grid = np.asarray(p_cal.get("group_std_per_alpha", []),
                                     dtype=np.float64).ravel()
            sigma_cal_at_lz = None
            if (alpha_grid.size and sigma_grid.size and np.isfinite(lz)
                    and alpha_grid.size == sigma_grid.size):
                order = np.argsort(alpha_grid)
                sigma_cal_at_lz = float(np.interp(
                    np.clip(lz, 0.0, 1.0), alpha_grid[order], sigma_grid[order],
                    left=sigma_grid[order][0], right=sigma_grid[order][-1]))
            rows.append([
                m["name"], pkey, _fmt(lz),
                _fmt(sigma_empirical),
                _fmt(p_cal.get("calibrated_mean_std")),
                _fmt(sigma_cal_at_lz),
            ])
    if rows:
        _write_csv(output_path, f"{report_tag}_method_sigma_shape_{variant}",
                   ["method", "proxy", "latent_z",
                    "sigma_empirical", "sigma_calibration_global",
                    "sigma_calibration_at_latent_z"],
                   rows)


def _dump_ivw_weights_csv(report: dict, output_path: str,
                           report_tag: str, variant: str) -> None:
    """Per-method per-proxy effective (α-dep) vs global IVW weights."""
    rows = []
    for method_name, method_report in report.get("methods", {}).items():
        summary = method_report.get("summary", {})
        w_eff = summary.get("ivw_weights_effective_mean") or {}
        w_glob = summary.get("ivw_weights_global_mean") or {}
        proxy_keys_union = sorted(set(w_eff.keys()) | set(w_glob.keys()))
        display_name = method_report.get("method_name", method_name)
        for pk in proxy_keys_union:
            we = (w_eff.get(pk) or {}).get("mean")
            wg = (w_glob.get(pk) or {}).get("mean")
            delta_pct = None
            if we is not None and wg not in (None, 0.0):
                try:
                    delta_pct = 100.0 * (float(we) - float(wg)) / float(wg)
                except (TypeError, ValueError, ZeroDivisionError):
                    delta_pct = None
            rows.append([display_name, pk,
                         _fmt(wg), _fmt(we), _fmt(delta_pct)])
    if rows:
        _write_csv(output_path, f"{report_tag}_ivw_weights_{variant}",
                   ["method", "proxy",
                    "weight_global_mean", "weight_alpha_dep_mean",
                    "delta_pct"],
                   rows)


def _json_sanitize(obj, _seen: set | None = None):
    """Recursive conversion of numpy scalars/arrays and non-serializable types
    into JSON-friendly primitives. Uses an id()-based memo to break reference
    cycles (self-referencing dicts/lists would otherwise blow the stack)."""
    # Fast-path primitives before any allocation.
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return f"<cycle:{type(obj).__name__}>"

    if isinstance(obj, dict):
        _seen.add(oid)
        try:
            return {str(k): _json_sanitize(v, _seen) for k, v in obj.items()}
        finally:
            _seen.discard(oid)
    if isinstance(obj, (list, tuple)):
        _seen.add(oid)
        try:
            return [_json_sanitize(v, _seen) for v in obj]
        finally:
            _seen.discard(oid)
    if isinstance(obj, np.ndarray):
        # ndarray.tolist() yields primitives (no cycles possible), recurse without memo.
        return [_json_sanitize(v, _seen) for v in obj.tolist()]
    # Fall back to repr for opaque objects (shouldn't normally happen).
    return repr(obj)


def _dump_report_json(report: dict, output_path: str, report_tag: str) -> None:
    """Dump the combined evaluation report (summaries per method) as JSON.

    Excludes per-image raw results to keep the file small and focused on
    aggregated statistics — full raw results live in the pickle cache.
    """
    slim = {
        "report_tag": report_tag,
        "condition": report.get("condition"),
        "dataset": report.get("dataset"),
        "methods": {},
    }
    for mname, mrep in report.get("methods", {}).items():
        slim["methods"][mname] = {
            "method_name": mrep.get("method_name", mname),
            "eq_vis": mrep.get("eq_vis"),
            "eq_th": mrep.get("eq_th"),
            "n_images": mrep.get("n_images"),
            "summary": mrep.get("summary"),
        }
    path = os.path.join(output_path, f"{report_tag}_report.json")
    with open(path, "w") as fh:
        json.dump(_json_sanitize(slim), fh, indent=2)
    log(f"JSON saved: {path}")


def _dump_calibration_json(calibration: dict, output_path: str,
                            report_tag: str, variant: str) -> None:
    """Dump calibration structure (σ curves, knots, nodes) as JSON for review."""
    path = os.path.join(output_path, f"{report_tag}_calibration_{variant}.json")
    with open(path, "w") as fh:
        json.dump(_json_sanitize(calibration), fh, indent=2)
    log(f"JSON saved: {path}")


def _write_ivw_diagnostic_txt(calibration: dict, report: dict,
                                output_path: str, report_tag: str,
                                variant: str) -> None:
    """Plain-text diagnostic: σ_max/σ_min per proxy (justifying α-dep IVW),
    convergence rate, weight redistribution, methods with largest Δlatent_z."""
    by_type = calibration.get("by_type") or calibration.get("calibrations_by_type") or {}
    active_cal = by_type.get("blend") or calibration
    pcs = active_cal.get("proxy_calibrations") or {}

    lines = []
    lines.append(f"IVW α-dependent diagnostic — {report_tag} / {variant}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("[σ shape per proxy — ratio σ_max/σ_min along α]")
    lines.append("A ratio > 2 justifies α-dep IVW; ~1 means global σ is fine.")
    lines.append("")
    lines.append(f"  {'proxy':<28s} {'σ_min':>10s} {'σ_max':>10s} {'ratio':>10s} {'σ̄':>10s}")
    for pkey, p_cal in pcs.items():
        sigmas = np.asarray(p_cal.get("group_std_per_alpha", []), dtype=np.float64).ravel()
        if sigmas.size == 0:
            continue
        smin = float(np.min(sigmas))
        smax = float(np.max(sigmas))
        ratio = smax / smin if smin > 1e-12 else float("inf")
        smean = p_cal.get("calibrated_mean_std")
        lines.append(f"  {pkey:<28s} {smin:>10.4f} {smax:>10.4f} "
                     f"{ratio:>10.2f} {(smean or 0.0):>10.4f}")
    lines.append("")

    # Convergence + Δlatent_z per method
    lines.append("[IVW α-dep convergence and latent_z shift]")
    lines.append("")
    lines.append(f"  {'method':<30s} {'n':>5s} {'iters':>8s} {'conv':>6s} "
                 f"{'lz_glob':>8s} {'lz_αdep':>8s} {'Δlz':>8s}")
    shifts = []
    for mname, mrep in report.get("methods", {}).items():
        summary = mrep.get("summary", {}) or {}
        lz_g = (summary.get("latent_z") or {}).get("mean")
        lz_a = (summary.get("latent_z_alpha_dep") or {}).get("mean")
        it_m = (summary.get("ivw_iters") or {}).get("mean")
        conv = summary.get("ivw_converged_rate")
        n_img = mrep.get("n_images", 0)
        delta = None
        if lz_g is not None and lz_a is not None:
            delta = float(lz_a) - float(lz_g)
            shifts.append((mrep.get("method_name", mname), abs(delta), delta))
        lines.append(
            f"  {mrep.get('method_name', mname):<30s} {n_img:>5d} "
            f"{(it_m or 0.0):>8.2f} {(conv or 0.0):>6.2f} "
            f"{(lz_g if lz_g is not None else float('nan')):>8.4f} "
            f"{(lz_a if lz_a is not None else float('nan')):>8.4f} "
            f"{(delta if delta is not None else float('nan')):>8.4f}"
        )
    lines.append("")

    # Top shifts
    shifts.sort(reverse=True)
    lines.append("[Methods with largest |Δlatent_z| (α-dep vs global)]")
    for name, _absd, delta in shifts[:10]:
        lines.append(f"  {name:<30s} Δ = {delta:+.4f}")
    lines.append("")

    # Mean weights per proxy across methods
    lines.append("[Mean effective weights per proxy across all methods]")
    lines.append(f"  {'proxy':<28s} {'w_global':>10s} {'w_α-dep':>10s} {'Δ%':>10s}")
    proxy_w_glob: dict[str, list[float]] = {}
    proxy_w_adep: dict[str, list[float]] = {}
    for _, mrep in report.get("methods", {}).items():
        summary = mrep.get("summary", {}) or {}
        for pk, st in (summary.get("ivw_weights_global_mean") or {}).items():
            if st and st.get("mean") is not None:
                proxy_w_glob.setdefault(pk, []).append(float(st["mean"]))
        for pk, st in (summary.get("ivw_weights_effective_mean") or {}).items():
            if st and st.get("mean") is not None:
                proxy_w_adep.setdefault(pk, []).append(float(st["mean"]))
    for pk in sorted(set(proxy_w_glob) | set(proxy_w_adep)):
        wg = float(np.mean(proxy_w_glob.get(pk, [0.0])))
        wa = float(np.mean(proxy_w_adep.get(pk, [0.0])))
        dp = (100.0 * (wa - wg) / wg) if wg > 1e-12 else float("nan")
        lines.append(f"  {pk:<28s} {wg:>10.4f} {wa:>10.4f} {dp:>10.2f}")
    lines.append("")

    path = os.path.join(output_path, f"{report_tag}_ivw_diagnostic_{variant}.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"Diagnostic saved: {path}")


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
        cr = summary.get("channel_redundancy")
        row["channel_redundancy"] = cr["mean"] if cr else float("nan")
        row["channel_redundancy_std"] = cr["std"] if cr else 0.0
        lza = summary.get("latent_z_alpha_dep")
        row["latent_z_alpha_dep"] = lza["mean"] if lza else float("nan")
        row["latent_z_alpha_dep_std"] = lza["std"] if lza else 0.0
        cvr = summary.get("cont_vis_raw")
        row["cont_vis_raw"] = cvr["mean"] if cvr else float("nan")
        row["calibration_type"] = summary.get("calibration_type_mode", "blend")
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
    curve_styles = {
        "blend": ("red", "o", "Blend"),
        "concat": ("blue", "s", "Concat"),
        "freq_blend": ("purple", "D", "Freq-blend"),
        "nonlinear": ("orange", "^", "Nonlinear"),
    }
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

    # Overlay method positions, coloured by their best-fit calibration type
    method_rows = _extract_method_rows(report) if report else []
    if method_rows:
        # Group methods by selected calibration type for per-type legend entries
        by_sel_type: dict[str, list[dict]] = {}
        for r in method_rows:
            by_sel_type.setdefault(r.get("calibration_type", "blend"), []).append(r)
        for sel_type, items in by_sel_type.items():
            color = curve_styles.get(sel_type, ("gray", "^", sel_type))[0]
            ax.scatter([r["cont_vis"] for r in items],
                       [r["latent_z"] for r in items],
                       s=110, c=color, marker="*",
                       edgecolors="black", linewidths=0.6,
                       label=f"best-fit: {sel_type}", zorder=5)
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
        _title_kinds = {
            "blend": "blend fractions",
            "concat": "channel-concat patterns",
            "freq_blend": "wavelet-coeff blend",
            "nonlinear": "geometric-mean blend",
        }
        title_kind = _title_kinds.get(stype, stype)
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
        # Prefer multi-format averaged weight when available
        avg_weights = calibration.get("averaged_proxy_weights", {})
        ivw_w = avg_weights.get(proxy_key,
                    p_cal.get("ivw_weight_normalized", p_cal.get("ivw_weight")))
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
    # Saturation-naive reference: thermal fraction inferred directly from the
    # unweighted raw proxy mean (1 - cont_vis_raw/100).  The gap between this
    # marker and the calibrated bar exposes the saturation correction PAVA is
    # applying — large gaps mean proxies were saturating near the visible
    # endpoint and the calibration is pulling them back toward the centre.
    naive_lz = np.array([
        np.nan if not np.isfinite(m.get("cont_vis_raw", float("nan")))
        else 1.0 - float(m["cont_vis_raw"]) / 100.0
        for m in method_rows
    ])
    ax2.scatter(x, naive_lz, marker="_", s=140, color="gray", linewidths=2.0,
                label="naive 1 − cont_vis_raw/100", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("latent_z (thermal fraction)")
    ax2.set_title("Calibrated thermal fraction (latent_z)  —  gray ticks: saturation-naive reference")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="upper left")
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


def _plot_method_sigma_shape(
    calibration: dict,
    report: dict,
    output_path: str,
    report_tag: str,
    variant: str,
) -> None:
    """Per-proxy σ across fusion methods, plotted against latent_z.

    Analogous to `calibration_sigma_shape` but for the real fusion methods
    instead of synthetic blend samples.  Each method contributes one point
    per proxy: x = method's latent_z, y = σ of that proxy's raw cont_vis
    across images for that method.  The dashed horizontal line per proxy is
    the calibration-derived σ (the global IVW σ) for reference.

    Interpretation: if a method's point sits well above its proxy's dashed
    line, that proxy is noisier for that method than the calibration assumed
    — the global IVW weight is over-confident for that method.  If the
    point clouds follow the same shape as `calibration_sigma_shape` (peak
    mid-α, tails at extremes), the calibration σ generalises cleanly.
    """
    method_rows = _extract_method_rows(report)
    if not method_rows:
        return
    by_type = calibration.get("by_type") or {}
    active_cal = by_type.get("blend") or calibration
    pcs = active_cal.get("proxy_calibrations") or {}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for pkey, plabel, pcolor in zip(PROXY_KEYS, PROXY_LABELS, PROXY_COLORS):
        xs, ys = [], []
        for m in method_rows:
            lz = m.get("latent_z")
            sd = m.get(pkey + "_std")
            if lz is None or np.isnan(lz) or sd is None:
                continue
            xs.append(lz)
            ys.append(sd)
        if not xs:
            continue
        order = np.argsort(xs)
        xs_sorted = np.array(xs)[order]
        ys_sorted = np.array(ys)[order]
        ax.plot(xs_sorted, ys_sorted, "o-", color=pcolor, linewidth=1.2,
                markersize=5, label=plabel, alpha=0.85)
        # Calibration σ reference (global, from IVW) as dashed line
        cal_std = pcs.get(pkey, {}).get("calibrated_mean_std")
        if cal_std is not None:
            ax.axhline(cal_std, color=pcolor, linestyle="--",
                       linewidth=0.9, alpha=0.55)

    ax.set_xlabel("latent_z (thermal fraction)")
    ax.set_ylabel("σ of proxy cont_vis across images (per method)")
    ax.set_title(
        f"Per-method proxy σ shape along latent_z  —  dashed = calibration σ [{variant}]")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=len(PROXY_KEYS), loc="upper center",
              bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_method_sigma_shape_{variant}")


def _plot_channel_redundancy_vs_latent(
    report: dict,
    output_path: str,
    report_tag: str,
    variant: str,
) -> None:
    """Scatter plot of per-method channel redundancy vs latent_z.

    Helps distinguish fusions that reach high latent_z by carrying distributed
    thermal information across distinct channels (low redundancy) from those
    that replicate thermal across channels (high redundancy ≈ 1).  A dashed
    reference at ~0.8 marks the ballpark redundancy of natural RGB imagery.
    """
    method_rows = _extract_method_rows(report)
    rows = [m for m in method_rows
            if np.isfinite(m.get("channel_redundancy", float("nan")))
            and np.isfinite(m.get("latent_z", float("nan")))]
    if not rows:
        return

    xs = np.array([m["latent_z"] for m in rows], dtype=np.float64)
    ys = np.array([m["channel_redundancy"] for m in rows], dtype=np.float64)
    ystd = np.array([m.get("channel_redundancy_std", 0.0) for m in rows], dtype=np.float64)
    names = [m["name"] for m in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(xs, ys, yerr=ystd, fmt="o", color="tab:purple",
                ecolor="tab:purple", elinewidth=0.8, capsize=2, alpha=0.85,
                label="channel_redundancy")
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=8, alpha=0.8)

    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.6,
               label="natural RGB reference (~0.8)")
    ax.axhline(1.0, color="red", linestyle=":", alpha=0.5,
               label="full duplication (R=G=B)")
    ax.set_xlabel("latent_z (thermal fraction)")
    ax.set_ylabel("channel redundancy score [0,1]")
    ax.set_title(
        f"Channel redundancy vs latent_z  —  high latent_z + high redundancy = thermal duplication  [{variant}]")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_channel_redundancy_{variant}")


def _plot_ivw_weight_shape(
    calibration: dict,
    output_path: str,
    report_tag: str,
    variant: str,
) -> None:
    """Per-proxy effective weight w_p(α) = 1/σ_p(α) (soft IVW), normalized across proxies.

    Shows how the α-dependent IVW redistributes mass between proxies along
    the latent axis.  A proxy whose σ is low at α≈0 but high at α≈1 will have
    a high weight near visible and low weight near thermal — the global IVW
    collapses this to a single constant weight.  Accompanies
    `calibration_sigma_shape` (raw σ) and is more directly interpretable as
    "what is the α-dep IVW actually doing".
    """
    by_type = calibration.get("by_type") or calibration.get("calibrations_by_type") or {}
    active_cal = by_type.get("blend") or calibration
    pcs = active_cal.get("proxy_calibrations") or {}
    if not pcs:
        return

    # Build a common α grid from union of per-proxy grids, sampled densely
    # for smooth curves.
    alpha_grid = np.linspace(0.0, 1.0, 101)
    fig, ax = plt.subplots(figsize=(11, 6))

    # Compute per-proxy weights at each α, then normalize across proxies so
    # each α column sums to 1.
    proxy_keys = list(pcs.keys())
    w_matrix = np.zeros((len(proxy_keys), alpha_grid.size), dtype=np.float64)
    for i, pkey in enumerate(proxy_keys):
        p_cal = pcs[pkey]
        a_grid = np.asarray(p_cal.get("group_alphas_std", []), dtype=np.float64).ravel()
        s_grid = np.asarray(p_cal.get("group_std_per_alpha", []), dtype=np.float64).ravel()
        if a_grid.size == 0 or s_grid.size == 0 or a_grid.size != s_grid.size:
            fallback = float(p_cal.get("calibrated_mean_std", 1.0))
            sigmas = np.full_like(alpha_grid, max(fallback, 1e-8))
        else:
            order = np.argsort(a_grid)
            sigmas = np.interp(alpha_grid, a_grid[order], s_grid[order],
                                left=s_grid[order][0], right=s_grid[order][-1])
        sigmas = np.maximum(sigmas, 1e-8)
        w_matrix[i] = 1.0 / sigmas  # soft IVW (1/σ), consistent with evaluation.py

    col_sums = w_matrix.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums <= 0, 1.0, col_sums)
    w_norm = w_matrix / col_sums

    # Apply the MAX_PROXY_WEIGHT cap per α-column to match what the IVW
    # actually uses in evaluation.  Without this, the plot misrepresents
    # how much influence any single proxy really has.
    from .evaluation import _cap_proxy_weights
    for col in range(w_norm.shape[1]):
        w_norm[:, col] = _cap_proxy_weights(w_norm[:, col])

    label_map = dict(zip(PROXY_KEYS, PROXY_LABELS))
    color_map = dict(zip(PROXY_KEYS, PROXY_COLORS))
    for i, pkey in enumerate(proxy_keys):
        label = label_map.get(pkey, pkey)
        color = color_map.get(pkey, None)
        ax.plot(alpha_grid, w_norm[i], label=label, color=color, linewidth=2)
        # Horizontal reference: global (α-independent) normalized weight.
        # Prefer multi-format averaged weight for consistency with what the
        # IVW actually applies.
        avg_weights = calibration.get("averaged_proxy_weights", {})
        gw = avg_weights.get(pkey,
                pcs[pkey].get("ivw_weight_normalized",
                               pcs[pkey].get("ivw_weight")))
        if gw is not None:
            ax.axhline(float(gw), linestyle="--", linewidth=1.0,
                       color=color, alpha=0.4)

    ax.set_xlabel("α (latent_z)")
    ax.set_ylabel("normalized IVW weight")
    ax.set_title(
        f"IVW weight shape along α  —  solid = α-dep, dashed = global [{variant}]")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=len(proxy_keys), loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_ivw_weight_shape_{variant}")

    # CSV dump of the same curves for numerical inspection
    rows = []
    for i, pkey in enumerate(proxy_keys):
        for a, w in zip(alpha_grid, w_norm[i]):
            rows.append([pkey, _fmt(a), _fmt(w)])
    _write_csv(output_path, f"{report_tag}_ivw_weight_shape_{variant}",
               ["proxy", "alpha", "weight_normalized"], rows)


def _plot_global_vs_alphadep_latent(
    report: dict,
    output_path: str,
    report_tag: str,
    variant: str,
) -> None:
    """Scatter: latent_z (global IVW) vs latent_z_alpha_dep per method.

    Identifies methods whose estimate shifts under α-dependent reweighting —
    points far from the diagonal are where the global weighting was masking
    (or inflating) the thermal fraction due to σ-heterogeneity.
    """
    method_rows = _extract_method_rows(report)
    pairs = []
    for m in method_rows:
        lz_g = m.get("latent_z")
        # latent_z_alpha_dep is fetched from the summary; _extract_method_rows
        # doesn't pull it yet, but we can read it from the original report.
        pairs.append((m["name"], lz_g, m.get("latent_z_alpha_dep")))
    # Pull latent_z_alpha_dep directly from the report (as _extract_method_rows
    # returns only latent_z today).
    # Build name->summary map
    name_to_summary = {}
    for mname, mrep in report.get("methods", {}).items():
        name_to_summary[mrep.get("method_name", mname)] = mrep.get("summary", {})

    xs, ys, names = [], [], []
    for m in method_rows:
        summary = name_to_summary.get(m["name"], {})
        lz_g = (summary.get("latent_z") or {}).get("mean")
        lz_a = (summary.get("latent_z_alpha_dep") or {}).get("mean")
        if lz_g is None or lz_a is None:
            continue
        xs.append(float(lz_g))
        ys.append(float(lz_a))
        names.append(m["name"])
    if not xs:
        return

    xs = np.array(xs); ys = np.array(ys)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6,
            label="identity")
    ax.scatter(xs, ys, color="tab:blue", s=45, alpha=0.85, edgecolor="black",
               linewidth=0.4)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=8, alpha=0.85)
    ax.set_xlabel("latent_z (global IVW)")
    ax.set_ylabel("latent_z (α-dep IVW)")
    ax.set_title(f"Global vs α-dependent IVW latent_z per method  [{variant}]")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_ivw_global_vs_alphadep_{variant}")


def _plot_reg_vs_latent_crosscheck(
    report: dict,
    output_path: str,
    report_tag: str,
    variant: str,
) -> None:
    """Cross-check: raw per-channel regression (cont_vis_reg) vs IVW latent_z.

    cont_vis_reg is a direct per-channel NNLS visible share — independent of the
    IVW combination. If the weight cap is effective, latent_z should track reg
    closely. Large deviations flag IVW distortion.
    """
    name_to_summary: dict[str, dict] = {}
    for mname, mrep in report.get("methods", {}).items():
        name_to_summary[mrep.get("method_name", mname)] = mrep.get("summary", {})

    names, reg_vals, lz_glob, lz_adep, ch_redund = [], [], [], [], []
    for dname, summary in name_to_summary.items():
        reg_s = (summary.get("cont_vis_reg") or {}).get("mean")
        lz_g = (summary.get("latent_z") or {}).get("mean")
        lz_a = (summary.get("latent_z_alpha_dep") or {}).get("mean")
        cr = (summary.get("channel_redundancy") or {}).get("mean")
        if reg_s is None or lz_g is None or lz_a is None:
            continue
        names.append(dname)
        reg_vals.append(1.0 - float(reg_s) / 100.0)  # thermal share from reg
        lz_glob.append(float(lz_g))
        lz_adep.append(float(lz_a))
        ch_redund.append(float(cr) if cr is not None else None)
    if not names:
        return

    reg_arr = np.array(reg_vals)
    lz_g_arr = np.array(lz_glob)
    lz_a_arr = np.array(lz_adep)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, lz, label in (
        (axes[0], lz_g_arr, "latent_z (global IVW)"),
        (axes[1], lz_a_arr, "latent_z (α-dep IVW)"),
    ):
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="identity")
        sc = ax.scatter(reg_arr, lz, c="tab:blue", s=50, alpha=0.8,
                        edgecolor="black", linewidth=0.4)
        for i, n in enumerate(names):
            ax.annotate(n, (reg_arr[i], lz[i]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, alpha=0.85)
        ax.set_xlabel("thermal share from reg (1 − cont_vis_reg/100)")
        ax.set_ylabel(label)
        ax.set_title(f"Reg cross-check: {label}  [{variant}]")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    _save_plot(fig, output_path, f"{report_tag}_reg_crosscheck_{variant}")

    # CSV companion
    rows = []
    for i, n in enumerate(names):
        rows.append([n, _fmt(reg_vals[i]), _fmt(lz_glob[i]), _fmt(lz_adep[i]),
                      _fmt(ch_redund[i])])
    _write_csv(output_path, f"{report_tag}_reg_crosscheck_{variant}",
               ["method", "thermal_share_reg", "latent_z_global", "latent_z_alpha_dep",
                "channel_redundancy"],
               rows)


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
    dataset: str,
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
    metrics_by_method = build_dataset_metrics(df, method_names, condition, equalization, dataset)
    if not metrics_by_method:
        log(f"No training data matched condition={condition} eq={equalization} dataset={dataset}; skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    # Φ⁻¹(0.9) — z-score for the 90th percentile of a standard normal.
    # Using P90 of the fitted normal (μ + z·σ) as the per-method summary
    # keeps this plot consistent with how dispersion is reported elsewhere
    # in the user's work, and captures "achievable performance under this
    # method's distribution" rather than central tendency.  With small n
    # per method (typically 3–5 runs), σ is noisy and so is the P90, but
    # that is a property inherent to the sample size, not the choice of
    # summary.  Individual runs are still drawn as a strip so dispersion
    # remains visible at a glance.
    P90_Z = 1.2815515655446004
    for ax, metric in zip(axes_flat, TRAINING_METRICS):
        xs_centers, ys_centers, names, xs_points, ys_points = [], [], [], [], []
        for m in method_rows:
            info = metrics_by_method.get(m["name"])
            if info is None or metric not in info["metrics"]:
                continue
            lz = m["latent_z"]
            if np.isnan(lz):
                continue
            mstats = info["metrics"][metric]
            p90 = mstats["mean"] + P90_Z * mstats["std"]
            xs_centers.append(lz)
            ys_centers.append(p90)
            names.append(f"{m['name']} (n={len(mstats['values'])})")
            for v in mstats["values"]:
                xs_points.append(lz)
                ys_points.append(v)

        # Individual runs as small points (strip)
        if xs_points:
            ax.scatter(xs_points, ys_points, s=18, color="#4c72b0", alpha=0.45,
                       edgecolors="none", zorder=2, label="individual runs")
        # Per-method P90 markers (fitted normal, μ + 1.28·σ)
        if xs_centers:
            ax.scatter(xs_centers, ys_centers, s=80, color="#c44e52", marker="D",
                       edgecolors="black", linewidths=0.5, zorder=4,
                       label="P90 per method (fitted normal)")
            # Annotate
            for x, y, n in zip(xs_centers, ys_centers, names):
                ax.annotate(n, (x, y), textcoords="offset points",
                            xytext=(5, 4), fontsize=7, alpha=0.85)
            # Correlation line + Pearson/Spearman on the per-method P90
            # (one point per method — replicates at the same latent_z would
            # artificially inflate n and inject replicate noise into r/ρ).
            if len(xs_centers) >= 3:
                order = np.argsort(xs_centers)
                xs_sorted = np.array(xs_centers)[order]
                ys_sorted = np.array(ys_centers)[order]
                coeffs = np.polyfit(xs_sorted, ys_sorted, 1)
                xfit = np.linspace(xs_sorted.min(), xs_sorted.max(), 50)
                ax.plot(xfit, np.polyval(coeffs, xfit), "--", color="gray",
                        alpha=0.5, linewidth=1, zorder=3, label="linear fit")
                # Pearson r (linear) + Spearman ρ (rank-monotonic).
                # Disagreement between the two signals a non-linear or
                # outlier-driven relationship.
                xs_arr = np.asarray(xs_centers, dtype=np.float64)
                ys_arr = np.asarray(ys_centers, dtype=np.float64)
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

    if args.dataset == "llvip" and args.condition == "day":
        raise ValueError("LLVIP does not provide day split; use --condition night or --condition all")

    dataset_root = resolve_dataset_root(args.dataset, args.dataset_root)

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
    # Deduplicate by basename so overlapping splits (e.g. LLVIP train-full vs train-half1/half2,
    # or any future dataset layout with redundant folders) do not inflate the image count.
    # Both KAIST (setXX_VNNN_IMMMMM.png) and LLVIP (numeric IDs) guarantee globally unique basenames.
    visible_paths_all: list[str] = []
    seen_basenames: set[str] = set()
    duplicate_count = 0
    for visible_path in iter_visible_images(dataset_root, args.dataset):
        if args.condition != "all" and infer_condition(visible_path, args.dataset) != args.condition:
            continue
        basename = os.path.basename(visible_path)
        if basename in seen_basenames:
            duplicate_count += 1
            continue
        seen_basenames.add(basename)
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
    dedup_note = f" | duplicates filtered: {duplicate_count}" if duplicate_count else ""
    log(f"Visible images found: {len(visible_paths_all)}{dedup_note} | sampling: {sampling_note} | selected: {len(selected_visible_paths)}")
    log(f"Calibration ratio: {settings.calibration_ratio:.2f} | calibration_images: {settings.calibration_images}")
    log(f"Execution mode: {settings.execution_mode} | workers: {settings.workers} | alpha_steps: {settings.alpha_steps} | chunk: {settings.task_chunksize}")
    log(f"CPU count: {os.cpu_count()} | OpenCV threads forced to 1")
    log(f"Methods selected: {len(fusion_methods)}")
    log("")

    # Generate calibrations for each equalization variant
    log(f"Generating calibrations for variants: {settings.equalization_variants}")
    calibrations = {}
    from .evaluation import proxy_set_signature
    proxy_set_sig = proxy_set_signature()
    for variant in settings.equalization_variants:
        # Two protocol hashes — one for the *samples* cache (heavy: synthetic
        # mixture proxy computations) and one for the *fit* cache (light:
        # PAVA curves + IVW weights derived from the samples).  The samples
        # store the full proxy schema (all six values per (image, α, type))
        # so they are invariant to ENABLED_PROXIES; only the fit cache picks
        # up the active subset via ``active_proxy_set``.  This separation is
        # what makes "toggle a proxy on/off" cheap: the samples (the heavy
        # part) are reused, only the fit gets refit.
        samples_protocol = {
            "proxy_version": PROXY_VERSION,
            "calibration_samples_version": CALIBRATION_SAMPLES_VERSION,
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
        samples_raw = "|".join(f"{k}={v}" for k, v in sorted(samples_protocol.items()))
        samples_hash = hashlib.sha256(samples_raw.encode("utf-8")).hexdigest()[:12]

        # Fit-cache protocol = samples protocol + everything fit-specific.
        fit_protocol = {
            **samples_protocol,
            "calibration_fit_version": CALIBRATION_FIT_VERSION,
            "active_proxy_set": proxy_set_sig,
        }
        fit_raw = "|".join(f"{k}={v}" for k, v in sorted(fit_protocol.items()))
        fit_hash = hashlib.sha256(fit_raw.encode("utf-8")).hexdigest()[:12]

        calibration_file = str(cache_dir / f"review_image_contribution_calibration_{args.dataset}_{args.condition}_{variant}_{fit_hash}.pkl")
        calibration_samples_file = str(
            Path(cache_file).with_name(
                Path(cache_file).stem + f"_calibration_samples_{variant}_{samples_hash}.pkl"
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
        # Analogous σ-vs-latent_z shape but using real fusion methods
        # (compares method dispersion to the calibration reference)
        _plot_method_sigma_shape(calibration, combined_report, report_dir, report_tag, variant)
        # Channel-redundancy vs latent_z diagnostic: flags methods that reach
        # high latent_z by replicating thermal across channels.
        _plot_channel_redundancy_vs_latent(combined_report, report_dir, report_tag, variant)
        # α-dependent IVW diagnostics: weight-shape curve, global vs α-dep
        # scatter, and numeric diagnostic txt.
        _plot_ivw_weight_shape(calibration, report_dir, report_tag, variant)
        _plot_global_vs_alphadep_latent(combined_report, report_dir, report_tag, variant)
        # Cross-check: raw per-channel regression vs IVW-aggregated latent_z.
        _plot_reg_vs_latent_crosscheck(combined_report, report_dir, report_tag, variant)
        _write_ivw_diagnostic_txt(calibration, combined_report, report_dir, report_tag, variant)
        # CSV dumps so the data behind sigma-shape / ivw-weights plots can be
        # inspected numerically without re-running.
        _dump_calibration_sigma_csv(calibration, report_dir, report_tag, variant)
        _dump_method_sigma_csv(calibration, combined_report, report_dir, report_tag, variant)
        _dump_ivw_weights_csv(combined_report, report_dir, report_tag, variant)
        _dump_calibration_json(calibration, report_dir, report_tag, variant)
        # Per calibration type: table, per-proxy, and node overview
        by_type = calibration.get("calibrations_by_type", {"blend": calibration})
        for stype, sub_cal in by_type.items():
            type_tag = f"{variant}_{stype}"
            _print_calibration_nodes_table(sub_cal, report_dir, report_tag, type_tag)
            _plot_per_proxy_calibration(sub_cal, report_dir, report_tag, type_tag)
            _plot_calibration_nodes_overview(sub_cal, report_dir, report_tag, type_tag)
    _plot_proxy_overview(combined_report, report_dir, report_tag)
    # Overall method summary + full report snapshot as machine-readable dumps.
    _dump_methods_overview_csv(combined_report, report_dir, report_tag)
    _dump_report_json(combined_report, report_dir, report_tag)

    # Training-vs-latent correspondence plots.  One per (variant × condition).
    # If args.condition == "all" we emit both day and night when data exists.
    conditions_to_plot = ["day", "night"] if args.condition == "all" else [args.condition]
    for variant, report in variant_reports.items():
        for cond in conditions_to_plot:
            cond_tag = f"{args.dataset}_{cond}"
            _plot_training_vs_latent(report, report_dir, cond_tag, cond, variant, args.dataset)


if __name__ == "__main__":
    main()
