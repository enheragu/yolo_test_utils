from __future__ import annotations

"""
Render a comparison index HTML that puts the per-slice reports in tabs and
shows a ``Cross-slice`` section with the key artifacts side-by-side across
every slice.

Each per-slice report at ``reports/html/<dataset>_<condition>.html`` is
embedded via an ``<iframe>`` — those pages keep working standalone and are
not re-rendered here.  The cross-slice grids pull directly from the existing
files in ``reports/`` and degrade gracefully when an artifact is missing.

Invocation (typically from the orchestrator, after all per-slice reports
exist):

    python -m Dataset_review.review_contribution.build_comparison_html \\
        --slice kaist:day --slice kaist:night --slice llvip:night \\
        --equalization no_equalization --target-reducer p90
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader


_PKG_DIR = Path(__file__).resolve().parent


def _fmt(val):
    if isinstance(val, float):
        if val == 0.0 or abs(val) < 1e-4:
            return f"{val:.3e}"
        return f"{val:.4f}"
    return str(val)


def _slice_id(dataset: str, condition: str) -> str:
    return f"{dataset}_{condition}"


def _rel(path: Path, anchor: Path) -> str:
    try:
        return str(path.resolve().relative_to(anchor.resolve()))
    except ValueError:
        import os
        return os.path.relpath(str(path), str(anchor))


def _table_from_csv(csv_path: Path,
                    keep_cols: list[str] | None = None,
                    max_rows: int | None = None,
                    sort_by: str | None = None,
                    ascending: bool = False,
                    is_anova: bool = False) -> str | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if keep_cols:
        cols = [c for c in keep_cols if c in df.columns]
        df = df[cols] if cols else df
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
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


def _image_cell(path: Path, anchor: Path, caption: str,
                missing_msg: str = "missing") -> dict:
    if path.exists():
        return {"kind": "image", "src": _rel(path, anchor), "caption": caption}
    return {"kind": "image", "src": None, "caption": caption, "missing": missing_msg}


def _table_cell(html: str | None, missing_msg: str = "missing") -> dict:
    if html:
        return {"kind": "table", "html": html}
    return {"kind": "table", "html": None, "missing": missing_msg}


# ---------------------------------------------------------------------------
# Cross-slice row builders
# ---------------------------------------------------------------------------

def _methods_overview_row(reports_dir: Path, slices: list[dict], anchor: Path) -> dict:
    cells = []
    for sl in slices:
        prefix = _slice_id(sl["dataset"], sl["condition"])
        csv_path = reports_dir / f"{prefix}_methods_overview.csv"
        html = _table_from_csv(
            csv_path,
            keep_cols=["method", "latent_z", "channel_redundancy_mean",
                       "cont_vis", "cont_vis_raw"],
            sort_by="latent_z",
        )
        cell = _table_cell(html, missing_msg=f"{csv_path.name} missing")
        cell["slice_label"] = sl["label"]
        cells.append(cell)
    return {"caption": "Methods overview (latent_z / CR / cont_vis)", "cells": cells}


def _proxy_plot_row(reports_dir: Path, slices: list[dict], anchor: Path,
                    filename_template: str, caption: str,
                    equalization: str) -> dict:
    cells = []
    for sl in slices:
        prefix = _slice_id(sl["dataset"], sl["condition"])
        fname = filename_template.format(prefix=prefix, equalization=equalization)
        cell = _image_cell(reports_dir / fname, anchor, caption,
                           missing_msg=f"{fname} missing")
        cell["slice_label"] = sl["label"]
        cells.append(cell)
    return {"caption": caption, "cells": cells}


def _pca_plot_row(pca_dir: Path, slices: list[dict], anchor: Path,
                  filename_template: str, caption: str, with_cr: bool) -> dict:
    cells = []
    suffix = "_with_cr" if with_cr else ""
    for sl in slices:
        prefix = _slice_id(sl["dataset"], sl["condition"]) + suffix
        fname = filename_template.format(prefix=prefix)
        cell = _image_cell(pca_dir / fname, anchor, caption,
                           missing_msg=f"{fname} missing")
        cell["slice_label"] = sl["label"]
        cells.append(cell)
    return {"caption": caption, "cells": cells}


def _pca_variance_table_row(pca_dir: Path, slices: list[dict]) -> dict:
    cells = []
    for sl in slices:
        prefix = _slice_id(sl["dataset"], sl["condition"])
        csv_path = pca_dir / f"pca_explained_variance_{prefix}.csv"
        html = _table_from_csv(csv_path)
        cell = _table_cell(html, missing_msg=f"{csv_path.name} missing")
        cell["slice_label"] = sl["label"]
        cells.append(cell)
    return {"caption": "PCA explained variance (proxies only)", "cells": cells}


def _regression_table_row(reg_dir: Path, slices: list[dict],
                          equalization: str, target_reducer: str,
                          with_cr: bool) -> dict:
    cells = []
    cr_suffix = "_with_cr" if with_cr else ""
    for sl in slices:
        prefix = _slice_id(sl["dataset"], sl["condition"])
        fname = f"regression_{prefix}{cr_suffix}_{equalization}_{target_reducer}.csv"
        html = _table_from_csv(reg_dir / fname)
        cell = _table_cell(html, missing_msg=f"{fname} missing")
        cell["slice_label"] = sl["label"]
        cells.append(cell)
    label = "with CR" if with_cr else "no CR"
    return {"caption": f"Nested OLS ({label})", "cells": cells}


def _equalization_plot_row(eq_dir: Path, slices: list[dict], anchor: Path,
                           metric: str, target_reducer: str,
                           kind: str) -> dict:
    """kind in {"heatmap", "delta", "strip"}"""
    cells = []
    mtag = metric.replace("-", "_")
    for sl in slices:
        prefix = _slice_id(sl["dataset"], sl["condition"])
        fname = f"{kind}_{prefix}_{mtag}_{target_reducer}.png"
        cell = _image_cell(eq_dir / fname, anchor, f"{metric} — {kind}",
                           missing_msg=f"{fname} missing")
        cell["slice_label"] = sl["label"]
        cells.append(cell)
    titles = {"heatmap": "heatmap (method × equalization)",
              "delta": "Δ vs no_equalization",
              "strip": "strip plot"}
    return {"caption": f"{metric} — {titles[kind]}", "cells": cells}

    
def _equalization_anova_row(eq_dir: Path, slices: list[dict],
                            metric: str, target_reducer: str) -> dict:
    cells = []
    mtag = metric.replace("-", "_")
    for sl in slices:
        prefix = _slice_id(sl["dataset"], sl["condition"])
        fname = f"anova_{prefix}_{mtag}_{target_reducer}.csv"
        html = _table_from_csv(eq_dir / fname, is_anova=True)
        cell = _table_cell(html, missing_msg=f"{fname} missing")
        cell["slice_label"] = sl["label"]
        cells.append(cell)
    return {"caption": f"{metric} — Two-way ANOVA", "cells": cells}


def _equalization_grouped_anova_row(eq_dir: Path, slices: list[dict],
                                    metric: str, target_reducer: str) -> dict:
    cells = []
    mtag = metric.replace("-", "_")
    for sl in slices:
        prefix = _slice_id(sl["dataset"], sl["condition"])
        fname = f"anova_grouped_{prefix}_{mtag}_{target_reducer}.csv"
        html = _table_from_csv(eq_dir / fname, is_anova=True)
        cell = _table_cell(html, missing_msg=f"{fname} missing")
        cell["slice_label"] = sl["label"]
        cells.append(cell)
    return {"caption": f"{metric} — Two-way ANOVA (fusion family)", "cells": cells}


def _equalization_tag_from_overview(equalization: str) -> tuple[str, str]:
    """Map an overview-CSV equalization tag back to (eq_vis, eq_th) pair."""
    return {
        "no_equalization":     ("none", "none"),
        "rgb_equalization":    ("clahe", "none"),
        "th_equalization":     ("none", "clahe"),
        "rgb_th_equalization": ("clahe", "clahe"),
    }.get(equalization, ("none", "none"))


def _ensure_latent2d_artifacts(reports_dir: Path, slices: list[dict],
                               equalization: str, target_reducer: str,
                               metric: str = "mAP50-95"
                               ) -> tuple[Path, Path, Path, Path]:
    """Ensure ``analysis_latent2d`` artifacts exist and are not stale.

    Regenerates when the PNG is missing or older than any of the source
    ``*_methods_overview.csv`` files in the requested slices.  Failure to
    regenerate is non-fatal: the section will simply render the missing-state.

    Returns ``(scatter_png, data_csv, top3_csv, corr_csv)``.  The two summary
    CSVs (top-3 and per-slice correlations) power the interpretive block;
    they may not exist on older runs.
    """
    latent_dir = reports_dir / "latent2d"
    latent_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{equalization}_{target_reducer}_{metric}"
    out_png = latent_dir / f"latent2d_scatter_{tag}.png"
    out_csv = latent_dir / f"latent2d_data_{tag}.csv"
    out_top = latent_dir / f"latent2d_top3_{tag}.csv"
    out_corr = latent_dir / f"latent2d_corr_{tag}.csv"

    overview_csvs = [
        reports_dir / f"{_slice_id(sl['dataset'], sl['condition'])}_methods_overview.csv"
        for sl in slices
    ]
    overview_csvs = [p for p in overview_csvs if p.exists()]
    needs_regen = (
        not out_png.exists()
        or not out_csv.exists()
        or any(p.stat().st_mtime > out_png.stat().st_mtime for p in overview_csvs)
    )
    if not needs_regen:
        return out_png, out_csv, out_top, out_corr

    import subprocess
    eq_vis, eq_th = _equalization_tag_from_overview(equalization)
    cmd = [
        sys.executable, "-m",
        "Dataset_review.review_contribution.analysis_latent2d",
        "--slices", *[f"{sl['dataset']}:{sl['condition']}" for sl in slices],
        "--reports-dir", str(reports_dir),
        "--report-dir", str(latent_dir),
        "--equalization-vis", eq_vis,
        "--equalization-th", eq_th,
        "--metric", metric,
        "--reducer", target_reducer,
    ]
    src_dir = _PKG_DIR.parents[1]
    env = {**__import__("os").environ, "PYTHONPATH": str(src_dir)}
    try:
        subprocess.run(cmd, check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        tail = " | ".join(msg[-3:]) if msg else "(no stderr)"
        print(f"[html] latent2d regen failed: {tail}")
    return out_png, out_csv, out_top, out_corr


def _latent2d_rows(reports_dir: Path, slices: list[dict], anchor: Path,
                   equalization: str, target_reducer: str) -> list[dict]:
    """Build cross-slice rows for the ``latent2d`` section.

    Four rows: scatter, per-slice top-3 by mAP, per-slice Pearson correlations
    (latent variants × mAP), and the tidy joint coordinate table.  Top-3 and
    correlations are the interpretive block — they convert the plot into
    something actionable without requiring the reader to inspect CSVs.
    """
    out_png, out_csv, out_top, out_corr = _ensure_latent2d_artifacts(
        reports_dir, slices, equalization, target_reducer, metric="mAP50-95")

    scatter_caption = (
        "thermal axis × channel_redundancy  —  three side-by-side panels: "
        "<code>latent_z</code> (PAVA-calibrated), <code>1 − cont_vis_raw/100</code> "
        "(uncalibrated), <code>1 − reg/100</code> (NNLS, calibration-portable)"
    )
    img_cell = _image_cell(out_png, anchor, scatter_caption,
                           missing_msg=f"{out_png.name} missing")
    img_cell["slice_label"] = "joint"
    img_cell["full_width"] = True

    top_html = _table_from_csv(out_top, sort_by=None)
    top_cell = _table_cell(top_html,
                           missing_msg=f"{out_top.name} missing")
    top_cell["slice_label"] = "joint"
    top_cell["full_width"] = True

    corr_html = _table_from_csv(out_corr, sort_by=None)
    corr_cell = _table_cell(corr_html,
                            missing_msg=f"{out_corr.name} missing")
    corr_cell["slice_label"] = "joint"
    corr_cell["full_width"] = True

    table_html = _table_from_csv(
        out_csv,
        keep_cols=["dataset", "condition", "method",
                   "latent_z_mean", "thermal_share_raw", "thermal_share_reg",
                   "calibration_lift", "channel_redundancy_mean",
                   "metric_value"],
        sort_by="latent_z_mean",
    )
    table_cell = _table_cell(table_html,
                             missing_msg=f"{out_csv.name} missing")
    table_cell["slice_label"] = "joint"
    table_cell["full_width"] = True

    return [
        {"caption": scatter_caption, "cells": [img_cell]},
        {"caption": "Top 3 methods per slice (by mAP50-95 P90)",
         "cells": [top_cell]},
        {"caption": "Pearson r of each thermal-axis variant vs mAP, per slice "
                    "(stable r across variants ⇒ proxy choice insensitive; "
                    "diverging r ⇒ calibration is amplifying or hiding signal)",
         "cells": [corr_cell]},
        {"caption": "Per-method joint coordinates "
                    "(<code>calibration_lift</code> = latent_z − thermal_share_raw)",
         "cells": [table_cell]},
    ]


def _equalization_intragroup_anova_row(eq_dir: Path, slices: list[dict],
                                       metric: str, target_reducer: str,
                                       fusion_group: str) -> dict:
    cells = []
    mtag = metric.replace("-", "_")
    for sl in slices:
        prefix = _slice_id(sl["dataset"], sl["condition"])
        tag = f"{prefix}_{mtag}_{target_reducer}"
        fname = f"anova_intragroup_{fusion_group}_{tag}.csv"
        html = _table_from_csv(eq_dir / fname, is_anova=True)
        cell = _table_cell(html, missing_msg=f"{fname} missing")
        cell["slice_label"] = sl["label"]
        cells.append(cell)
    return {"caption": f"{metric} — Two-way ANOVA within {fusion_group}", "cells": cells}


# ---------------------------------------------------------------------------
# Cross-slice section assembly
# ---------------------------------------------------------------------------

def build_cross_sections(reports_dir: Path, slices: list[dict],
                         equalization: str, target_reducer: str,
                         anchor: Path) -> list[dict]:
    pca_dir = reports_dir / "pca"
    reg_dir = reports_dir / "regression"
    eq_dir = reports_dir / "equalization"

    sections: list[dict] = []

    # Overview — one row only but critical for visual anchor.
    sections.append({
        "title": "Methods overview",
        "description": "Side-by-side of latent_z, channel redundancy and "
                       "cont_vis across slices.  Lets you spot when a method "
                       "ranks differently between conditions or datasets.<br>"
                       "<b>Quick read:</b> high latent_z + high channel redundancy "
                       "often indicates thermal dominance via channel duplication; "
                       "methods with similar latent_z but lower redundancy tend to "
                       "carry more distributed cross-channel information.",
        "open": True,
        "rows": [_methods_overview_row(reports_dir, slices, anchor)],
    })

    # Joint latent map: thermal axis × channel_redundancy across all slices.
    sections.append({
        "title": "Joint latent map (thermal axis × channel_redundancy)",
        "description": "Two latent axes for the contribution analysis: a "
                       "thermal‐share axis on x and "
                       "<code>channel_redundancy</code> (inter-channel mean |corr|) "
                       "on y.  Three x-axis variants are shown side-by-side "
                       "because they trade off saturation correction against "
                       "cross-dataset portability:<ul>"
                       "<li><code>latent_z</code> — PAVA-calibrated, IVW-weighted. "
                       "Best <i>intra</i>-dataset reading; calibration is fitted "
                       "per dataset so absolute values are <i>dataset-relative</i> "
                       "(direct numerical comparison across slices is unsafe).</li>"
                       "<li><code>1 − cont_vis_raw/100</code> — unweighted mean of "
                       "the 6 raw proxies. No calibration; portable cross-dataset; "
                       "saturates near the endpoints.</li>"
                       "<li><code>1 − reg/100</code> — per-channel NNLS thermal "
                       "share.  The most calibration-portable single proxy "
                       "(curve overlap diagnostic gives max|Δ| ≈ 0.06 across "
                       "datasets vs ~0.20–0.29 for grad/spectral/freq).  "
                       "Reductive but stable.</li></ul>"
                       "<b>PAVA portability:</b> <code>reg</code> ≈ <code>ssim</code> "
                       "&lt; <code>mi</code> &lt; <code>freq</code> ≈ "
                       "<code>grad</code> ≈ <code>spectral</code>.  When the "
                       "interpretive correlations row shows a methods's r flipping "
                       "sign between <code>latent_z</code> and "
                       "<code>thermal_share_raw</code>, calibration is the cause.<br>"
                       "<b>Quick read of the 2D map:</b> top band = channel "
                       "duplication; bottom-right = thermal distributed across "
                       "structurally distinct channels (informative for a 3-plane "
                       "detector); bottom-left is shaded because no real fusion "
                       "lands there. <code>latent_z_αdep</code> correlates "
                       "0.92–0.98 with <code>latent_z</code> and is kept as a "
                       "σ-shape diagnostic, not a second axis.",
        "open": True,
        "rows": _latent2d_rows(reports_dir, slices, anchor,
                                equalization, target_reducer),
    })

    # Proxies / calibration plots.
    sections.append({
        "title": "Proxies & calibration",
        "description": "Core per-method proxy and calibration artifacts.<br>"
                       "<b>Quick read:</b> use proxy-overview for level differences, "
                       "proxy-σ for stability (lower is more stable), and "
                       "training-vs-latent for whether proxy ranking aligns with "
                       "detection ranking.",
        "open": False,
        "rows": [
            _proxy_plot_row(reports_dir, slices, anchor,
                            "{prefix}_proxy_overview.png",
                            "Proxy overview (raw + calibrated)", equalization),
            _proxy_plot_row(reports_dir, slices, anchor,
                            "{prefix}_channel_redundancy_{equalization}.png",
                            "Channel redundancy per method", equalization),
            _proxy_plot_row(reports_dir, slices, anchor,
                            "{prefix}_per_method_proxy_std_{equalization}.png",
                            "Per-method proxy σ", equalization),
            _proxy_plot_row(reports_dir, slices, anchor,
                            "{prefix}_ivw_global_vs_alphadep_{equalization}.png",
                            "IVW global vs α-dep", equalization),
            _proxy_plot_row(reports_dir, slices, anchor,
                            "{prefix}_training_vs_latent_{equalization}.png",
                            "Training metrics vs latent_z", equalization),
        ],
    })

    # PCA.
    sections.append({
        "title": "PCA",
        "description": "Dimensionality diagnostic: scree, loadings and "
                       "first-two-PC scatter.  Explained-variance rows let "
                       "you check whether the latent manifold is equally "
                       "rich in each slice.<br>"
                       "<b>Quick read:</b> scree says how many dimensions matter; "
                       "loadings tell which proxies define each PC; scatter shows "
                       "whether methods cluster/separate in PC space.",
        "open": False,
        "rows": [
            _pca_variance_table_row(pca_dir, slices),
            _pca_plot_row(pca_dir, slices, anchor, "pca_scree_{prefix}.png",
                          "Scree + broken-stick (proxies only)", with_cr=False),
            _pca_plot_row(pca_dir, slices, anchor,
                          "pca_loadings_heatmap_{prefix}.png",
                          "Loadings heatmap (proxies only)", with_cr=False),
            _pca_plot_row(pca_dir, slices, anchor,
                          "pca_pc1_pc2_scatter_{prefix}.png",
                          "PC1/PC2 scatter (proxies only)", with_cr=False),
            _pca_plot_row(pca_dir, slices, anchor, "pca_scree_{prefix}.png",
                          "Scree (proxies + CR)", with_cr=True),
        ],
    })

    # Regression.
    sections.append({
        "title": "Regression (detection vs PC scores)",
        "description": "Nested OLS tables (R², adj R², ΔR², F-test) side by "
                       "side — easy to see if PC1/PC2 carry the same "
                       "predictive load across slices.<br>"
                       "<b>Quick read:</b> prioritize ΔR² and nested F-test when "
                       "adding PCs; gains in R² without support in adjusted R² or "
                       "F-test are often overfit on small method counts.",
        "open": False,
        "rows": [
            _regression_table_row(reg_dir, slices, equalization,
                                  target_reducer, with_cr=False),
            _regression_table_row(reg_dir, slices, equalization,
                                  target_reducer, with_cr=True),
        ],
    })

    # Equalization.
    eq_rows: list[dict] = []
    for metric in ("P", "R", "mAP50", "mAP50-95"):
        eq_rows.append(_equalization_plot_row(eq_dir, slices, anchor, metric,
                                              target_reducer, "heatmap"))
        eq_rows.append(_equalization_plot_row(eq_dir, slices, anchor, metric,
                                              target_reducer, "delta"))
        eq_rows.append(_equalization_anova_row(eq_dir, slices, metric,
                                               target_reducer))
        eq_rows.append(_equalization_grouped_anova_row(eq_dir, slices, metric,
                                                        target_reducer))
        for fusion_group in ("reference", "static_fusion", "reprojection_variance",
                             "reprojection_freq", "alpha", "dl_fusion"):
            eq_rows.append(_equalization_intragroup_anova_row(
                eq_dir, slices, metric, target_reducer, fusion_group
            ))
    sections.append({
        "title": "Equalization effect",
        "description": "method × equalization heatmap + Δ vs baseline for "
                       "P / R / mAP50 / mAP50-95, with ANOVA tables under each "
                       "metric block. The grouped ANOVA is also shown, so you can "
                       "compare families directly.<br><b>Quick read:</b> <br>1) check method×equalization "
                       "interaction first (if significant, effect is method-dependent), "
                       "<br>2) then check equalization main effect (if significant without "
                       "strong interaction, preference is more consistent), <br>3) use Δ plots "
                       "for direction/magnitude (positive = gain vs no_equalization), "
                       "<br>4) read sum_sq shares as explained-variance contribution by term.",
        "open": False,
        "rows": eq_rows,
    })

    return sections


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def render_index(slices: list[dict], equalization: str, target_reducer: str,
                 reports_dir: Path, out_path: Path,
                 template_dir: Path | None = None) -> Path:
    anchor = out_path.parent
    template_dir = template_dir or (_PKG_DIR / "templates")

    # Fill in href per slice (relative to anchor).  ``?embed=1`` tells the
    # per-slice template to hide its header + tag-row so the iframe only shows
    # content — avoids duplicating title/subtitle that the index already carries.
    for sl in slices:
        sl.setdefault("id", _slice_id(sl["dataset"], sl["condition"]))
        sl.setdefault("label", f"{sl['dataset'].upper()} / {sl['condition']}")
        sl.setdefault("href", f"{sl['id']}.html?embed=1")

    cross_sections = build_cross_sections(reports_dir, slices,
                                          equalization, target_reducer, anchor)

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True, lstrip_blocks=True,
    )
    template = env.get_template("index_template.html.j2")

    html = template.render(
        title="Review contribution — index",
        subtitle=f"comparison across {len(slices)} slice(s)",
        slices=slices,
        n_slices=len(slices),
        cross_sections=cross_sections,
        equalization=equalization,
        target_reducer=target_reducer,
        generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_slice(value: str) -> dict:
    """Parse ``dataset:condition`` into a slice dict."""
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--slice expects dataset:condition, got {value!r}")
    return {"dataset": parts[0], "condition": parts[1]}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render the index HTML with tabs + cross-slice grids.")
    p.add_argument("--slice", action="append", type=_parse_slice, default=[],
                   help="Add a slice in dataset:condition form. "
                        "Repeat to add more. "
                        "Default: kaist:day kaist:night llvip:night.")
    p.add_argument("--equalization",
                   choices=["no_equalization", "rgb_equalization",
                            "th_equalization", "rgb_th_equalization"],
                   default="no_equalization")
    p.add_argument("--target-reducer", choices=["mean", "median", "p90"],
                   default="p90")
    p.add_argument("--reports-dir", default=None)
    p.add_argument("--out", default=None,
                   help="Output path (default: <reports>/html/index.html).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    slices = args.slice or [
        {"dataset": "kaist", "condition": "day"},
        {"dataset": "kaist", "condition": "night"},
        {"dataset": "llvip", "condition": "night"},
    ]

    reports_dir = (Path(args.reports_dir) if args.reports_dir
                   else Path.home() / ".cache" / "eeha_review_fusion_contribution" / "reports")
    out_path = (Path(args.out) if args.out
                else reports_dir / "html" / "index.html")

    rendered = render_index(slices, args.equalization, args.target_reducer,
                            reports_dir=reports_dir, out_path=out_path)
    print(f"[html-index] wrote {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
