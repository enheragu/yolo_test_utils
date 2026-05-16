from __future__ import annotations

"""
Equalization effect analysis on training detection metrics.

Does histogram equalization (``no_equalization``, ``rgb_equalization``,
``th_equalization``, ``rgb_th_equalization``) systematically shift detection
quality, and does the effect interact with the fusion method?  Uses only the
existing ``raw_training_data.csv`` — no GPU cost.

Outputs (under ``~/.cache/eeha_review_fusion_contribution/reports/equalization/``):
  - ``pivot_<tag>.csv`` — method × equalization pivot of mean/std/n_runs.
  - ``delta_vs_no_eq_<tag>.csv`` — per-method Δmetric vs the ``no_equalization``
    baseline, with a noise-scale column (pooled within-cell σ).
  - ``anova_<tag>.csv`` — two-way ANOVA: ``metric ~ method + equalization +
    method:equalization``.
    - ``anova_grouped_<tag>.csv`` — two-way ANOVA with fusion families:
        ``metric ~ fusion_group + equalization + fusion_group:equalization``.
  - ``heatmap_<tag>.png`` — method × equalization mean heatmap.
  - ``delta_<tag>.png`` — stacked bar of Δmetric per method / equalization.
  - ``strip_<tag>.png`` — per-equalization strip-plot of all runs, all methods.

By default all methods present in the CSV slice are retained (``min_eqs=1``).
Methods with sparse equalization coverage still appear in pivot/heatmap/strip,
while delta-vs-baseline and ANOVA naturally use only rows/cells that satisfy
their own statistical requirements.

Heatmaps and delta bars use a fixed method order by fusion family (unless
``--method-order baseline`` is requested), so slice-to-slice comparison is
visually stable.

The family order is explicit and small: ``reference``, ``static_fusion``,
``reprojection_variance``, ``reprojection_freq``, ``alpha`` and
``dl_fusion``.

For the grouped view we also emit one ANOVA per family, so you can inspect
method × equalization interactions within each fusion group directly.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

from Dataset_review.review_contribution.training_results_check import (
    DEFAULT_CSV_PATH,
    TRAINING_METRICS,
    load_training_results,
)


EQUALIZATIONS = (
    "no_equalization",
    "rgb_equalization",
    "th_equalization",
    "rgb_th_equalization",
)
BASELINE_EQ = "no_equalization"

_FAMILY_SPECS = [
    ("reference", ["visible", "lwir"]),
    ("static_fusion", ["rgbt", "rgbt_v2", "vt", "vths", "vths_v2", "vths_v3", "hsvt"]),
    ("reprojection_variance", ["pca", "fa"]),
    ("reprojection_freq", ["wavelet", "wavelet_max", "curvelet", "curvelet_max"]),
    ("alpha", ["ssim", "ssim_v2", "sobel_weighted", "superpixel"]),
    ("dl_fusion", ["early_4ch", "middle_4ch", "late_4ch", "split_late_4ch"]),
]

_GROUP_ORDER = {group: idx for idx, (group, _) in enumerate(_FAMILY_SPECS)}
_METHOD_GROUP = {method: group for group, methods in _FAMILY_SPECS for method in methods}
_GROUP_METHOD_ORDER = {group: methods for group, methods in _FAMILY_SPECS}


def _method_key(method: str) -> str:
    return str(method).strip().lower()


def _fusion_group(method: str) -> str:
    return _METHOD_GROUP.get(_method_key(method), "other")


def _method_rank_tuple(method: str) -> tuple[int, int, str]:
    key = _method_key(method)
    group = _fusion_group(key)
    try:
        group_rank = _GROUP_ORDER[group]
    except KeyError:
        group_rank = _GROUP_ORDER["other"]
    try:
        within_group = _GROUP_METHOD_ORDER[group].index(key)
    except (KeyError, ValueError):
        within_group = 999
    return (group_rank, within_group, key)


def _fixed_method_order(methods: list[str]) -> list[str]:
    return sorted(methods, key=_method_rank_tuple)


def _display_number(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number == 0.0 or abs(number) < 1e-4:
            return f"{number:.3e}"
        return f"{number:.4f}"
    return str(value)


def _table_to_string(table: pd.DataFrame) -> str:
    formatters = {
        column: (lambda value: _display_number(value))
        for column in table.columns
        if pd.api.types.is_numeric_dtype(table[column])
    }
    return table.to_string(index=False, formatters=formatters)


# ---------------------------------------------------------------------------
# Data slicing
# ---------------------------------------------------------------------------

def _dataset_tag(dataset: str) -> str:
    return "llvip_80_20" if dataset == "llvip" else "kaist_80_20"


def load_runs_long(training_csv: Path,
                   dataset: str,
                   condition: str,
                   metric: str) -> pd.DataFrame:
    """Return a long DataFrame: one row per training run.

    Columns: ``method, equalization, value``.
    """
    df = load_training_results(training_csv)
    tag = _dataset_tag(dataset)
    subset = df[
        (df["Dataset"] == tag)
        & (df["Condition"] == condition)
        & (df["Group Key"].isin(EQUALIZATIONS))
    ].copy()
    if subset.empty:
        raise RuntimeError(
            f"No training rows for Dataset={tag}, Condition={condition}"
        )
    subset["value"] = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset.dropna(subset=["value"])
    subset = subset.rename(columns={"Type": "method", "Group Key": "equalization"})
    subset["fusion_group"] = subset["method"].map(_fusion_group)
    return subset[["method", "fusion_group", "equalization", "value"]].reset_index(drop=True)


def keep_well_covered(runs: pd.DataFrame, min_eqs: int = 1) -> pd.DataFrame:
    """Keep only methods that appear under ``>= min_eqs`` equalizations."""
    method_eqs = runs.groupby("method")["equalization"].nunique()
    keep = method_eqs[method_eqs >= min_eqs].index
    return runs[runs["method"].isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _p90(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    mu = float(values.mean())
    sigma = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return mu + 1.28 * sigma


def build_pivot(runs: pd.DataFrame, reducer: str) -> pd.DataFrame:
    """Pivot method × equalization with per-cell summary statistics."""
    reducers = {
        "mean":   lambda v: float(v.mean()),
        "median": lambda v: float(np.median(v)),
        "p90":    _p90,
    }
    pick = reducers[reducer]
    records: list[dict] = []
    for (method, eq), group in runs.groupby(["method", "equalization"]):
        values = group["value"].to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        fusion_group = _fusion_group(method)
        records.append({
            "method": method,
            "fusion_group": fusion_group,
            "equalization": eq,
            "n_runs": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "p90": _p90(values),
            "value": pick(values),
        })
    return pd.DataFrame(records)


def delta_vs_baseline(pivot: pd.DataFrame, reducer: str) -> pd.DataFrame:
    """Per-method delta of the reducer vs the ``no_equalization`` baseline.

    Adds a ``noise_scale`` column = the pooled within-cell σ across
    equalizations for the same method, giving a reference for when a Δ is
    meaningful vs run-to-run noise.
    """
    rows: list[dict] = []
    for method, g in pivot.groupby("method"):
        base_row = g[g["equalization"] == BASELINE_EQ]
        if base_row.empty:
            continue
        base_value = float(base_row["value"].iloc[0])
        # Pooled within-cell σ: sqrt(mean(var_i)) over cells with >1 run.
        vars_ = g.loc[g["n_runs"] > 1, "std"].to_numpy() ** 2
        noise_scale = float(np.sqrt(vars_.mean())) if vars_.size else float("nan")
        for _, row in g.iterrows():
            if row["equalization"] == BASELINE_EQ:
                continue
            rows.append({
                "method": method,
                "equalization": row["equalization"],
                "baseline": base_value,
                "value": float(row["value"]),
                "delta": float(row["value"]) - base_value,
                "noise_scale": noise_scale,
                "n_runs_eq": int(row["n_runs"]),
                "n_runs_base": int(base_row["n_runs"].iloc[0]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Two-way ANOVA
# ---------------------------------------------------------------------------

def run_anova(runs: pd.DataFrame) -> pd.DataFrame:
    """Two-way ANOVA: value ~ method + equalization + method:equalization.

    Returns a statsmodels-style ANOVA table as a DataFrame.  Requires at least
    one cell with >1 run for the interaction term to have residual df.
    """
    if runs["method"].nunique() < 2 or runs["equalization"].nunique() < 2:
        raise RuntimeError("ANOVA needs ≥2 methods and ≥2 equalizations")
    model = ols("value ~ C(method) + C(equalization) + C(method):C(equalization)",
                data=runs).fit()
    table = sm.stats.anova_lm(model, typ=2)
    table = table.reset_index().rename(columns={"index": "term"})
    return table


def run_grouped_anova(runs: pd.DataFrame) -> pd.DataFrame:
    """Two-way ANOVA replacing method by fusion family."""
    if runs["fusion_group"].nunique() < 2 or runs["equalization"].nunique() < 2:
        raise RuntimeError("Grouped ANOVA needs >=2 fusion groups and >=2 equalizations")
    model = ols(
        "value ~ C(fusion_group) + C(equalization) + C(fusion_group):C(equalization)",
        data=runs,
    ).fit()
    table = sm.stats.anova_lm(model, typ=2)
    table = table.reset_index().rename(columns={"index": "term"})
    return table


def run_intragroup_anova(runs: pd.DataFrame, fusion_group: str) -> pd.DataFrame:
    """Two-way ANOVA inside one fusion family: method vs equalization."""
    subset = runs[runs["fusion_group"] == fusion_group].copy()
    if subset["method"].nunique() < 2 or subset["equalization"].nunique() < 2:
        raise RuntimeError(
            f"Intragroup ANOVA for {fusion_group} needs >=2 methods and >=2 equalizations"
        )
    model = ols("value ~ C(method) + C(equalization) + C(method):C(equalization)",
                data=subset).fit()
    table = sm.stats.anova_lm(model, typ=2)
    table = table.reset_index().rename(columns={"index": "term"})
    return table


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _order_methods_by_baseline(pivot: pd.DataFrame) -> list[str]:
    base = pivot[pivot["equalization"] == BASELINE_EQ][["method", "value"]]
    return base.sort_values("value", ascending=False)["method"].tolist() + [
        m for m in pivot["method"].unique()
        if m not in base["method"].to_numpy()
    ]


def _order_methods(pivot: pd.DataFrame, method_order: str) -> list[str]:
    methods = pivot["method"].drop_duplicates().tolist()
    if method_order == "baseline":
        return _order_methods_by_baseline(pivot)
    return _fixed_method_order(methods)


def _group_boundaries(methods: list[str]) -> list[int]:
    boundaries: list[int] = []
    previous_group: str | None = None
    for index, method in enumerate(methods):
        current_group = _fusion_group(method)
        if previous_group is not None and current_group != previous_group:
            boundaries.append(index)
        previous_group = current_group
    return boundaries


def plot_heatmap(pivot: pd.DataFrame, metric: str, out_path: Path,
                 title: str, reducer: str, method_order: str = "fixed") -> None:
    methods = _order_methods(pivot, method_order)
    mat = pivot.pivot(index="method", columns="equalization", values="value")
    mat = mat.reindex(index=methods, columns=list(EQUALIZATIONS))

    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(methods) + 2))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([c.replace("_equalization", "") for c in mat.columns],
                       rotation=20, ha="right", fontsize=9)
    for boundary in _group_boundaries(methods):
        ax.hlines(boundary - 0.5, -0.5, len(mat.columns) - 0.5,
                  colors="black", linewidth=1, alpha=0.8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < np.nanmean(mat.to_numpy()) else "black",
                        fontsize=7)
    ax.set_title(f"{title}\n{metric} ({reducer})")
    fig.colorbar(im, ax=ax, shrink=0.8, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_delta(delta: pd.DataFrame, metric: str, out_path: Path,
               title: str, methods_ordered: list[str] | None = None) -> None:
    eqs = [e for e in EQUALIZATIONS if e != BASELINE_EQ]
    if methods_ordered is None:
        methods = delta["method"].drop_duplicates().tolist()
    else:
        methods = [m for m in methods_ordered if m in set(delta["method"].tolist())]
    x = np.arange(len(methods))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(methods) + 2), 5))
    noise_shown = False
    for i, eq in enumerate(eqs):
        sub = delta[delta["equalization"] == eq].set_index("method").reindex(methods)
        ax.bar(x + (i - 1) * width, sub["delta"].to_numpy(dtype=float),
               width=width, label=eq.replace("_equalization", ""))
        if not noise_shown:
            ax.errorbar(x, np.zeros(len(methods)),
                        yerr=sub["noise_scale"].to_numpy(dtype=float),
                        fmt="none", ecolor="black", alpha=0.4, capsize=2,
                        label="±σ (pooled within-cell)")
            noise_shown = True
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"Δ {metric} vs {BASELINE_EQ}")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_strip(runs: pd.DataFrame, metric: str, out_path: Path,
               title: str) -> None:
    eqs = list(EQUALIZATIONS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(0)
    for i, eq in enumerate(eqs):
        vals = runs.loc[runs["equalization"] == eq, "value"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=vals.size)
        ax.scatter(np.full_like(vals, i) + jitter, vals, s=16, alpha=0.55)
        ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, "k-", lw=2)
    ax.set_xticks(range(len(eqs)))
    ax.set_xticklabels([e.replace("_equalization", "") for e in eqs],
                       rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse the effect of equalization on detection metrics."
    )
    p.add_argument("--dataset", choices=["kaist", "llvip"], default="llvip")
    p.add_argument("--condition", choices=["day", "night"], default="night")
    p.add_argument("--metric", choices=TRAINING_METRICS, default="mAP50")
    p.add_argument("--target-reducer", choices=["mean", "median", "p90"], default="p90")
    p.add_argument("--min-eqs", type=int, default=1,
                   help="Keep only methods that appear under >= N equalizations "
                        "(default: 1, include all methods present in CSV).")
    p.add_argument("--method-order", choices=["fixed", "baseline"], default="fixed",
                   help="Method ordering for pivot/heatmap/delta. fixed=stable by "
                        "fusion family; baseline=dynamic by no_equalization value.")
    p.add_argument("--training-csv", default=str(DEFAULT_CSV_PATH))
    p.add_argument("--report-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    runs = load_runs_long(Path(args.training_csv),
                          args.dataset, args.condition, args.metric)
    runs = keep_well_covered(runs, min_eqs=args.min_eqs)
    if runs.empty:
        print(f"[eq] no methods cover >= {args.min_eqs} equalizations", file=sys.stderr)
        return 2

    cache_dir = Path.home() / ".cache" / "eeha_review_fusion_contribution"
    report_dir = (Path(args.report_dir) if args.report_dir
                  else cache_dir / "reports" / "equalization")
    report_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.dataset}_{args.condition}_{args.metric.replace('-', '_')}_{args.target_reducer}"

    pivot = build_pivot(runs, reducer=args.target_reducer)
    ordered_methods = _order_methods(pivot, args.method_order)
    method_rank = {m: i for i, m in enumerate(ordered_methods)}
    pivot["method_rank"] = pivot["method"].map(lambda m: method_rank.get(m, 10**9))
    pivot.sort_values(["method_rank", "method", "equalization"], inplace=True)
    pivot.drop(columns=["method_rank"], inplace=True)
    pivot.to_csv(report_dir / f"pivot_{tag}.csv", index=False)

    delta = delta_vs_baseline(pivot, reducer=args.target_reducer)
    delta.sort_values(["method", "equalization"], inplace=True)
    delta.to_csv(report_dir / f"delta_vs_no_eq_{tag}.csv", index=False)

    try:
        anova = run_anova(runs)
        anova.to_csv(report_dir / f"anova_{tag}.csv", index=False)
    except Exception as exc:
        print(f"[eq] ANOVA skipped: {exc}", file=sys.stderr)
        anova = None

    try:
        grouped_anova = run_grouped_anova(runs)
        grouped_anova.to_csv(report_dir / f"anova_grouped_{tag}.csv", index=False)
    except Exception as exc:
        print(f"[eq] grouped ANOVA skipped: {exc}", file=sys.stderr)
        grouped_anova = None

    intragroup_frames: list[pd.DataFrame] = []
    for fusion_group, _methods in _FAMILY_SPECS:
        try:
            intragroup = run_intragroup_anova(runs, fusion_group)
        except Exception as exc:
            print(f"[eq] intragroup ANOVA skipped for {fusion_group}: {exc}", file=sys.stderr)
            continue
        intragroup["fusion_group"] = fusion_group
        intragroup.to_csv(report_dir / f"anova_intragroup_{fusion_group}_{tag}.csv", index=False)
        intragroup_frames.append(intragroup)

    intragroup_all = pd.concat(intragroup_frames, ignore_index=True) if intragroup_frames else None

    plot_heatmap(pivot, args.metric, report_dir / f"heatmap_{tag}.png",
                 title=f"{args.dataset}/{args.condition}",
                 reducer=args.target_reducer,
                 method_order=args.method_order)
    if not delta.empty:
        plot_delta(delta, args.metric, report_dir / f"delta_{tag}.png",
                   title=f"{args.dataset}/{args.condition}  "
                         f"Δ{args.metric} vs {BASELINE_EQ}",
                   methods_ordered=ordered_methods)
    plot_strip(runs, args.metric, report_dir / f"strip_{tag}.png",
               title=f"{args.dataset}/{args.condition}  {args.metric} by equalization")

    # Terminal summary.
    print(f"[eq] methods: {runs['method'].nunique()}  equalizations: {runs['equalization'].nunique()}")
    print(f"\n[eq] pivot (rows=method, cols=eq, values={args.target_reducer}):")
    pivot_wide = pivot.pivot(index="method", columns="equalization", values="value")
    pivot_wide = pivot_wide.reindex(columns=list(EQUALIZATIONS))
    print(pivot_wide.round(4).to_string())

    print(f"\n[eq] mean across methods, per equalization:")
    print(pivot_wide.mean(axis=0).round(4).to_string())

    if anova is not None:
        print(f"\n[eq] two-way ANOVA ({args.metric}):")
        print(_table_to_string(anova))

    if grouped_anova is not None:
        print(f"\n[eq] grouped two-way ANOVA ({args.metric}, by fusion_group):")
        print(_table_to_string(grouped_anova))

    if intragroup_all is not None:
        print(f"\n[eq] intragroup two-way ANOVAs ({args.metric}):")
        print(_table_to_string(intragroup_all))

    print(f"\n[eq] artifacts in {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
