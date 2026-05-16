from __future__ import annotations

"""
Render a single self-contained HTML report for ``(dataset, condition)``.

Reads all existing artifacts from
``~/.cache/eeha_review_fusion_contribution/reports/`` (methods overview,
pipeline plots, PCA outputs, regression outputs, equalization outputs) and
renders them into ``reports/html/<dataset>_<condition>.html`` via the Jinja2
template at ``templates/report_template.html.j2``.

Missing artifacts are skipped gracefully with a placeholder — so the report
remains meaningful even when the toolchain has only been partially run.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader


_PKG_DIR = Path(__file__).resolve().parent
_SRC_DIR = _PKG_DIR.parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

from Dataset_review.review_contribution.evaluation import (  # noqa: E402
    PROXY_VERSION,
    CALIBRATION_SAMPLES_VERSION,
    CALIBRATION_FIT_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val):
    if isinstance(val, float):
        if val == 0.0 or abs(val) < 1e-4:
            return f"{val:.3e}"
        return f"{val:.4f}"
    return str(val)


def _table_from_csv(csv_path: Path,
                    max_rows: int | None = None,
                    float_cols: list[str] | None = None,
                    is_anova: bool = False) -> str | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    if float_cols:
        for c in float_cols:
            if c in df.columns:
                df[c] = df[c].map(lambda v: _fmt(v) if pd.notna(v) else v)
    formatters = {
        c: (lambda v: _fmt(v))
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    }
    
    # Apply ANOVA conditional coloring if requested and p-value column exists
    if is_anova and "PR(>F)" in df.columns:
        return _apply_anova_styling(df, formatters)
    
    return df.to_html(index=False, classes="data-table", border=0,
                      formatters=formatters)


def _apply_anova_styling(df: pd.DataFrame, formatters: dict) -> str:
    """Apply conditional row coloring to ANOVA table based on p-values.
    
    Green (reject H0, p < 0.05): Factor is significant
    Red (fail to reject H0, p >= 0.05): Factor is not significant
    Residual rows: no color (they're not hypothesis tests)
    """
    def row_color(row):
        # row is a Series with column names as index
        term_val = row["term"] if "term" in row.index else None
        p_val = row["PR(>F)"] if "PR(>F)" in row.index else None
        
        # Residual rows don't get colored
        if term_val is None or "Residual" in str(term_val):
            return [""] * len(row)
        
        # NaN p-values don't get colored
        if p_val is None or pd.isna(p_val):
            return [""] * len(row)
        
        try:
            p_value_float = float(p_val)
        except (ValueError, TypeError):
            return [""] * len(row)
        
        # Determine color: green if significant (p < 0.05), red otherwise
        color = "lightgreen" if p_value_float < 0.05 else "lightcoral"
        return [f"background-color: {color}"] * len(row)
    
    # Use Styler to apply row-level formatting and number formats
    styled = df.style.apply(row_color, axis=1).format(formatters)
    
    # Get HTML with minimal inline styles
    html = styled.to_html()
    
    # Add table class for consistency with non-ANOVA tables
    html = html.replace('<table', '<table class="data-table" border="0"', 1)
    
    return html


def _rel(path: Path, anchor: Path) -> str:
    """Relative URL from *anchor* directory to *path*."""
    try:
        return str(path.resolve().relative_to(anchor.resolve()))
    except ValueError:
        import os
        return os.path.relpath(str(path), str(anchor))


def _plots(png_paths: list[Path], anchor: Path,
           captions: list[str] | None = None) -> list[dict]:
    out: list[dict] = []
    captions = captions or [p.stem for p in png_paths]
    for p, cap in zip(png_paths, captions):
        if p.exists():
            out.append({"src": _rel(p, anchor), "caption": cap})
    return out


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_overview_section(reports_dir: Path, dataset: str, condition: str) -> dict:
    prefix = f"{dataset}_{condition}"
    overview_csv = reports_dir / f"{prefix}_methods_overview.csv"
    timing_html = reports_dir / f"{prefix}_fusion_timing_overview.html"

    blocks = []
    if overview_csv.exists():
        df = pd.read_csv(overview_csv)
        # Keep the most useful columns if the CSV is very wide.
        interesting = [c for c in df.columns if c == "method" or any(
            k in c for k in ("latent_z", "channel_redundancy", "cont_vis", "ivw_iters")
        )]
        if interesting:
            df = df[interesting]
        blocks.append({
            "kind": "table",
            "caption": "Methods overview (latent_z / channel_redundancy / cont_vis / iters)",
            "html": df.to_html(index=False, classes="data-table", border=0,
                               float_format=lambda v: f"{v:.4f}"),
        })
    else:
        blocks.append({"kind": "table", "caption": "Methods overview",
                       "html": None, "missing": f"{overview_csv.name} not found."})

    if timing_html.exists():
        blocks.append({"kind": "heading", "text": "Fusion timing"})
        blocks.append({"kind": "text", "html": timing_html.read_text()})

    return {
        "title": "Overview",
        "description": "High-level summary of every fusion method for this slice. "
                       "Other sections drill into proxies, calibration, PCA and "
                       "detection regression.",
        "open": True,
        "blocks": blocks,
    }


def build_proxy_calibration_section(reports_dir: Path, dataset: str,
                                    condition: str, equalization: str,
                                    anchor: Path) -> dict:
    prefix = f"{dataset}_{condition}"
    eq = equalization
    blocks = []

    # Proxies
    proxy_plots = _plots(
        [reports_dir / f"{prefix}_proxy_overview.png",
         reports_dir / f"{prefix}_per_method_proxy_std_{eq}.png",
         reports_dir / f"{prefix}_channel_redundancy_{eq}.png"],
        anchor,
        captions=["Raw + calibrated proxies per method",
                  "Per-method proxy dispersion (σ)",
                  "Channel redundancy per method"],
    )
    if proxy_plots:
        blocks.append({"kind": "plots", "caption": "Proxies", "plot_items": proxy_plots})

    # Calibration
    cal_plots = _plots(
        [reports_dir / f"{prefix}_per_proxy_calibration_curves_{eq}.png",
         reports_dir / f"{prefix}_calibration_sigma_shape_{eq}.png",
         reports_dir / f"{prefix}_method_sigma_shape_{eq}.png",
         reports_dir / f"{prefix}_ivw_weight_shape_{eq}.png",
         reports_dir / f"{prefix}_ivw_global_vs_alphadep_{eq}.png",
         reports_dir / f"{prefix}_calibration_diagnostic_{eq}.png",
         reports_dir / f"{prefix}_reg_crosscheck_{eq}.png",
         reports_dir / f"{prefix}_training_vs_latent_{eq}.png"],
        anchor,
        captions=["Per-proxy calibration curves",
                  "Calibration σ(α) shape",
                  "Method σ vs calibrated σ",
                  "IVW weight α-dep shape",
                  "IVW global vs α-dep",
                  "Calibration diagnostic",
                  "Thermal share: raw regression vs latent_z",
                  "Training metrics vs latent_z"],
    )
    if cal_plots:
        blocks.append({"kind": "plots", "caption": "Calibration",
                       "plot_items": cal_plots})

    sigma_csv = reports_dir / f"{prefix}_calibration_sigma_shape_{eq}.csv"
    if sigma_csv.exists():
        blocks.append({
            "kind": "table",
            "caption": "Calibration σ(α) (first 30 rows)",
            "html": _table_from_csv(sigma_csv, max_rows=30),
        })

    return {
        "title": "Proxies & Calibration",
        "description": "Per-proxy raw and calibrated values, plus the IVW weighting "
                       "that yields <code>latent_z</code>.",
        "open": False,
        "blocks": blocks,
    }


def build_pca_section(reports_dir: Path, dataset: str, condition: str,
                      anchor: Path) -> dict:
    pca_dir = reports_dir / "pca"
    blocks = []

    for suffix, label in [("", "Proxies only"), ("_with_cr", "Proxies + channel_redundancy")]:
        tag = f"{dataset}_{condition}{suffix}"
        var_csv = pca_dir / f"pca_explained_variance_{tag}.csv"
        loadings_csv = pca_dir / f"pca_loadings_{tag}.csv"
        scores_csv = pca_dir / f"pca_scores_per_method_{tag}.csv"
        scree_png = pca_dir / f"pca_scree_{tag}.png"
        loadings_png = pca_dir / f"pca_loadings_heatmap_{tag}.png"
        scatter_png = pca_dir / f"pca_pc1_pc2_scatter_{tag}.png"

        if not var_csv.exists():
            continue

        blocks.append({"kind": "heading", "text": label,
                       "description": f"Files tagged <code>{tag}</code>."})

        blocks.append({"kind": "table", "caption": "Explained variance",
                       "html": _table_from_csv(var_csv)})
        blocks.append({"kind": "table", "caption": "Loadings (proxy → PC)",
                       "html": _table_from_csv(loadings_csv)})
        blocks.append({"kind": "table",
                       "caption": "Per-method PC scores (mean ± std)",
                       "html": _table_from_csv(scores_csv)})

        plots = _plots([scree_png, loadings_png, scatter_png], anchor,
                       captions=["Scree + broken-stick",
                                 "Loadings heatmap",
                                 "PC1/PC2 scatter per method"])
        if plots:
            blocks.append({"kind": "plots", "caption": "Plots", "plot_items": plots})

    if not blocks:
        blocks.append({"kind": "text",
                       "html": "<i class='missing'>No PCA artifacts found; run "
                               "<code>analysis_pca.py</code> first.</i>"})

    return {
        "title": "PCA",
        "description": "Latent-dimensionality diagnostic on the proxy matrix. "
                       "Kaiser (eigenvalue > 1) and broken-stick references are "
                       "both reported per PC.",
        "open": False,
        "blocks": blocks,
    }


def build_regression_section(reports_dir: Path, dataset: str, condition: str,
                             equalization: str, target_reducer: str,
                             anchor: Path) -> dict:
    reg_dir = reports_dir / "regression"
    blocks = []

    for suffix, label in [("", "PCA without channel_redundancy"),
                          ("_with_cr", "PCA with channel_redundancy")]:
        tag = f"{dataset}_{condition}{suffix}_{equalization}_{target_reducer}"
        csv_path = reg_dir / f"regression_{tag}.csv"
        if not csv_path.exists():
            continue
        blocks.append({"kind": "heading", "text": label,
                       "description": f"Files tagged <code>{tag}</code>."})
        blocks.append({"kind": "table",
                       "caption": "Nested OLS: R², adjusted R², ΔR², F-test",
                       "html": _table_from_csv(csv_path)})

        scatters = sorted(reg_dir.glob(f"scatter_{tag}_*.png"))
        plots = [{"src": _rel(p, anchor), "caption": p.stem} for p in scatters]
        if plots:
            blocks.append({"kind": "plots", "caption": "Per-PC scatter plots",
                           "plot_items": plots})

    if not blocks:
        blocks.append({"kind": "text",
                       "html": "<i class='missing'>No regression artifacts found; "
                               "run <code>analysis_regression.py</code> first.</i>"})

    return {
        "title": "Regression (detection vs PC scores)",
        "description": "Nested OLS: <code>target ~ PC1</code>, "
                       "<code>+ PC2</code>, <code>+ PC3</code> with F-test on ΔR².",
        "open": False,
        "blocks": blocks,
    }


_EQUALIZATIONS = (
    "no_equalization",
    "rgb_equalization",
    "th_equalization",
    "rgb_th_equalization",
)


def _build_consolidated_eq_plot(eq_dir: Path, dataset: str, condition: str,
                                target_reducer: str) -> Path | None:
    """Synthesise a ``metric × equalization`` summary bar chart from pivot CSVs.

    Averages the reducer value across methods for each (metric, equalization)
    cell.  Written next to the per-metric plots so relative paths stay simple.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ("P", "R", "mAP50", "mAP50-95")
    data: dict[str, dict[str, float]] = {}
    for metric in metrics:
        mtag = metric.replace("-", "_")
        csv_path = eq_dir / f"pivot_{dataset}_{condition}_{mtag}_{target_reducer}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "equalization" not in df.columns or "value" not in df.columns:
            continue
        data[metric] = df.groupby("equalization")["value"].mean().to_dict()

    if not data:
        return None

    shown_metrics = [m for m in metrics if m in data]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(shown_metrics))
    n_eqs = len(_EQUALIZATIONS)
    width = 0.8 / n_eqs
    for i, eq in enumerate(_EQUALIZATIONS):
        vals = [data.get(m, {}).get(eq, np.nan) for m in shown_metrics]
        ax.bar(x + (i - (n_eqs - 1) / 2) * width, vals, width=width,
               label=eq.replace("_equalization", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(shown_metrics)
    ax.set_ylabel(f"{target_reducer} across methods")
    ax.set_title(f"{dataset}/{condition} — equalization × metric")
    ax.legend(fontsize=8, ncol=n_eqs, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = eq_dir / f"consolidated_{dataset}_{condition}_{target_reducer}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def build_equalization_section(reports_dir: Path, dataset: str, condition: str,
                               target_reducer: str, anchor: Path) -> dict:
    eq_dir = reports_dir / "equalization"
    blocks: list[dict] = []

    # Consolidated metric × equalization chart at the top — one plot that
    # answers "is the equalization preference consistent across metrics?".
    try:
        consolidated = _build_consolidated_eq_plot(
            eq_dir, dataset, condition, target_reducer)
    except Exception as exc:  # matplotlib / pandas edge cases shouldn't block the report
        print(f"[html] consolidated plot skipped: {exc}")
        consolidated = None
    if consolidated is not None and consolidated.exists():
        blocks.append({
            "kind": "plots",
            "caption": "Consolidated overview (all metrics)",
            "plot_items": [{
                "src": _rel(consolidated, anchor),
                "caption": "mean reducer per (metric, equalization)",
            }],
        })

    # Per-metric subsection — plots visible, tables collapsed.
    for metric in ("P", "R", "mAP50", "mAP50-95"):
        mtag = metric.replace("-", "_")
        tag = f"{dataset}_{condition}_{mtag}_{target_reducer}"
        pivot_csv = eq_dir / f"pivot_{tag}.csv"
        if not pivot_csv.exists():
            continue

        delta_csv = eq_dir / f"delta_vs_no_eq_{tag}.csv"
        anova_csv = eq_dir / f"anova_{tag}.csv"
        anova_grouped_csv = eq_dir / f"anova_grouped_{tag}.csv"
        heatmap_png = eq_dir / f"heatmap_{tag}.png"
        delta_png = eq_dir / f"delta_{tag}.png"
        strip_png = eq_dir / f"strip_{tag}.png"

        child_blocks: list[dict] = []
        plots = _plots([heatmap_png, delta_png, strip_png], anchor,
                       captions=["method × eq heatmap",
                                 f"Δ {metric} vs no_equalization",
                                 "Per-equalization strip plot"])
        if plots:
            child_blocks.append({"kind": "plots", "plot_items": plots})
        child_blocks.append({
            "kind": "table", "collapsed": True,
            "caption": f"Pivot: method × equalization ({target_reducer})",
            "html": _table_from_csv(pivot_csv),
        })
        child_blocks.append({
            "kind": "table", "collapsed": True,
            "caption": f"Δ vs {metric}@no_equalization",
            "html": _table_from_csv(delta_csv),
        })
        child_blocks.append({
            "kind": "table", "collapsed": True,
            "caption": "Two-way ANOVA",
            "html": _table_from_csv(anova_csv, is_anova=True),
        })
        child_blocks.append({
            "kind": "table", "collapsed": True,
            "caption": "Two-way ANOVA (fusion family)",
            "html": _table_from_csv(anova_grouped_csv, is_anova=True),
        })
        for family in ("reference", "static_fusion", "reprojection_variance",
                       "reprojection_freq", "alpha", "dl_fusion"):
            intragroup_csv = eq_dir / f"anova_intragroup_{family}_{tag}.csv"
            child_blocks.append({
                "kind": "table", "collapsed": True,
                "caption": f"Two-way ANOVA within {family}",
                "html": _table_from_csv(intragroup_csv, is_anova=True),
            })

        blocks.append({
            "kind": "subsection",
            "title": metric,
            "badge": f"tag={tag}",
            "open": False,
            "blocks": child_blocks,
        })

    if not blocks:
        blocks.append({"kind": "text",
                       "html": "<i class='missing'>No equalization artifacts; "
                               "run <code>analysis_equalization.py</code> first.</i>"})

    return {
        "title": "Equalization effect",
        "description": "Effect of <code>no / rgb / th / rgb_th</code> equalization on "
                       "training metrics.  Each metric below is collapsible; tables "
                       "are hidden by default so the plots stay visually dominant. "
                       "The consolidated chart at the top averages across methods "
                       "to show whether the equalization preference transfers "
                       "between metrics. Heatmaps use a fixed family-first method "
                       "order to keep cross-slice comparisons stable.",
        "open": False,
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def render_report(dataset: str, condition: str, equalization: str,
                  target_reducer: str,
                  reports_dir: Path, out_path: Path,
                  template_dir: Path | None = None) -> Path:
    anchor = out_path.parent
    template_dir = template_dir or (_PKG_DIR / "templates")
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,  # we hand-control escaping for tables and HTML fragments
        trim_blocks=True, lstrip_blocks=True,
    )
    template = env.get_template("report_template.html.j2")

    sections = [
        build_overview_section(reports_dir, dataset, condition),
        build_proxy_calibration_section(reports_dir, dataset, condition, equalization, anchor),
        build_pca_section(reports_dir, dataset, condition, anchor),
        build_regression_section(reports_dir, dataset, condition, equalization,
                                 target_reducer, anchor),
        build_equalization_section(reports_dir, dataset, condition,
                                   target_reducer, anchor),
    ]

    html = template.render(
        title=f"Review contribution — {dataset.upper()} / {condition}",
        subtitle=f"equalization: {equalization}  |  target reducer: {target_reducer}",
        version_tags={
            "proxy": PROXY_VERSION,
            "samples": CALIBRATION_SAMPLES_VERSION,
            "fit": CALIBRATION_FIT_VERSION,
        },
        generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        equalization=equalization,
        target_reducer=target_reducer,
        sections=sections,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a per-slice HTML report.")
    p.add_argument("--dataset", choices=["kaist", "llvip"], required=True)
    p.add_argument("--condition", choices=["day", "night"], required=True)
    p.add_argument("--equalization",
                   choices=["no_equalization", "rgb_equalization",
                            "th_equalization", "rgb_th_equalization"],
                   default="no_equalization")
    p.add_argument("--target-reducer", choices=["mean", "median", "p90"],
                   default="p90",
                   help="Reducer used by regression and equalization tags.")
    p.add_argument("--reports-dir", default=None,
                   help="Override reports dir (default: ~/.cache/eeha_review_fusion_contribution/reports).")
    p.add_argument("--out", default=None,
                   help="Output HTML path (default: <reports>/html/<dataset>_<condition>.html).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    reports_dir = (Path(args.reports_dir) if args.reports_dir
                   else Path.home() / ".cache" / "eeha_review_fusion_contribution" / "reports")
    out_path = (Path(args.out) if args.out
                else reports_dir / "html" / f"{args.dataset}_{args.condition}.html")

    rendered = render_report(args.dataset, args.condition, args.equalization,
                             args.target_reducer,
                             reports_dir=reports_dir, out_path=out_path)
    print(f"[html] wrote {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
