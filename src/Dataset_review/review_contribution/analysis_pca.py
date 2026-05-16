from __future__ import annotations

"""
PCA analysis of raw contribution proxies.

Reads the evaluation cache produced by :mod:`pipeline` / :mod:`contribution`
and asks a single question: *how many independent dimensions do the 6 proxies
actually describe?*

Outputs (under the provided ``--report-dir``, default ``<cache>/reports/pca``):
  - ``pca_explained_variance.csv``  : per-PC variance, cumulative, broken-stick
                                      reference, Kaiser rule (eigenvalue > 1
                                      on z-scored data ⇔ variance ratio > 1/k).
  - ``pca_loadings.csv``            : proxy × PC loadings (correlation-scaled).
  - ``pca_scores_per_entry.csv``    : method, visible_path, PC1..PCk.
  - ``pca_scores_per_method.csv``   : method mean/std for PC1..PCk.
  - ``pca_scree.png``               : scree plot + broken-stick overlay.
  - ``pca_loadings_heatmap.png``    : loadings visualisation.
  - ``pca_pc1_pc2_scatter.png``     : PC1 vs PC2 colored by method.

Rule of thumb for interpreting the scree plot:
  - If PC1 explains > 80% variance: a single latent is sufficient.
  - If PC1 + PC2 > 90% but PC1 < 80%: two dimensions are real.
  - Broken-stick test: keep PCs whose eigenvalue exceeds the broken-stick
    reference (a uniform-null baseline).  More conservative than Kaiser.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))


PROXY_KEYS: tuple[str, ...] = (
    "cont_vis_reg",
    "cont_vis_mi",
    "cont_vis_ssim",
    "cont_vis_grad_combined",
    "cont_vis_spectral",
    "cont_vis_freq",
)

# When `--include-channel-redundancy` is set, the per-method channel-redundancy
# mean is broadcast to every row of that method and appended as a 7th feature.
CHANNEL_REDUNDANCY_KEY = "channel_redundancy"


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def _default_cache_path(dataset: str, condition: str) -> Path:
    return (Path.home() / ".cache" / "eeha_review_fusion_contribution"
            / f"review_image_contribution_{dataset}_{condition}.pkl")


def load_proxy_table(cache_file: Path,
                     method_filter: set[str] | None = None,
                     condition_filter: str | None = None,
                     drop_anchors: bool = True,
                     include_channel_redundancy: bool = False) -> pd.DataFrame:
    """Flatten an evaluation-cache pickle into a tidy DataFrame.

    One row per (method, image).  Columns: method, visible_path, condition,
    + the 6 proxies in :data:`PROXY_KEYS`.  If ``include_channel_redundancy``
    is true and the entry carries a native per-image value (eval cache at
    ``PROXY_VERSION >= v16``), it is appended as :data:`CHANNEL_REDUNDANCY_KEY`.
    """
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)

    rows: list[dict] = []
    for entry in cache.values():
        if not isinstance(entry, dict):
            continue
        method = entry.get("source_method")
        if method is None:
            continue
        if method_filter is not None and method not in method_filter:
            continue
        if condition_filter is not None and entry.get("condition") != condition_filter:
            continue
        row = {
            "method": method,
            "visible_path": entry.get("visible_path"),
            "condition": entry.get("condition"),
        }
        any_missing = False
        for key in PROXY_KEYS:
            value = entry.get(key)
            if value is None or not np.isfinite(value):
                any_missing = True
                break
            row[key] = float(value)
        if any_missing:
            continue
        if include_channel_redundancy:
            cr = entry.get(CHANNEL_REDUNDANCY_KEY)
            if cr is not None and np.isfinite(cr):
                row[CHANNEL_REDUNDANCY_KEY] = float(cr)
        rows.append(row)

    df = pd.DataFrame(rows)
    if drop_anchors:
        df = df[~df["method"].isin({"visible", "lwir"})].copy()
    if include_channel_redundancy:
        if CHANNEL_REDUNDANCY_KEY not in df.columns:
            # Cache predates per-image CR; caller must fall back to the
            # overview broadcast via ``merge_channel_redundancy``.
            df.attrs["channel_redundancy_native"] = False
        else:
            # Mixed caches can contain old entries (no CR) plus new entries
            # (with CR). Only treat as native when every retained row has CR.
            cr_ok = np.isfinite(df[CHANNEL_REDUNDANCY_KEY].to_numpy(dtype=np.float64))
            if bool(cr_ok.all()):
                df.attrs["channel_redundancy_native"] = True
            else:
                # Drop partial per-image CR and force a clean per-method merge.
                df = df.drop(columns=[CHANNEL_REDUNDANCY_KEY])
                df.attrs["channel_redundancy_native"] = False
    return df


# ---------------------------------------------------------------------------
# PCA computation
# ---------------------------------------------------------------------------

def _broken_stick(k: int) -> np.ndarray:
    """Broken-stick reference eigenvalues for *k* dimensions (sum = 1)."""
    return np.array([np.sum(1.0 / np.arange(j, k + 1)) for j in range(1, k + 1)]) / k


def merge_channel_redundancy(df: pd.DataFrame, overview_csv: Path) -> pd.DataFrame:
    """Broadcast per-method ``channel_redundancy_mean`` from the overview CSV.

    The evaluation cache (pre-``channel_redundancy`` addition) does not store
    the diagnostic per-image.  Since channel-redundancy is a property of the
    fusion method — its inter-method variance dominates by orders of
    magnitude (~1e-4 for PCA vs ~0.94 for wavelet) — broadcasting the
    method-level mean to every row is a reasonable approximation for
    screening a second PCA dimension.
    """
    overview = pd.read_csv(overview_csv, usecols=["method", "channel_redundancy_mean"])
    overview = overview.rename(columns={"channel_redundancy_mean": CHANNEL_REDUNDANCY_KEY})
    merged = df.merge(overview, on="method", how="left")
    missing = merged[CHANNEL_REDUNDANCY_KEY].isna()
    if missing.any():
        absent = sorted(merged.loc[missing, "method"].unique())
        raise ValueError(
            f"channel_redundancy_mean missing for methods: {absent}. "
            f"Check overview CSV: {overview_csv}"
        )
    return merged


def _drop_nonfinite_feature_rows(df: pd.DataFrame,
                                 features: tuple[str, ...]) -> tuple[pd.DataFrame, int]:
    """Drop rows containing NaN/Inf in any selected feature.

    This protects the SVD step from non-finite inputs, which can happen when
    channel-redundancy is only partially present in older or mixed caches.
    """
    X = df[list(features)].to_numpy(dtype=np.float64)
    keep = np.isfinite(X).all(axis=1)
    dropped = int((~keep).sum())
    if dropped == 0:
        return df, 0
    return df.loc[keep].copy(), dropped


def run_pca(df: pd.DataFrame,
            features: tuple[str, ...] = PROXY_KEYS) -> dict:
    """Run PCA on ``features`` columns of *df* and return a structured report."""
    X = df[list(features)].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        sample = bad[:5].tolist()
        raise ValueError(
            "PCA input contains non-finite values after filtering. "
            f"Sample bad positions (row, col): {sample}"
        )
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1.0
    Z = (X - mu) / sigma

    # SVD-based PCA (no sklearn dependency).
    n = Z.shape[0]
    try:
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        eigenvalues = (S ** 2) / (n - 1)                 # on z-scored data, sum ≈ k
        # Loadings = V scaled by sqrt(eigenvalue) → proxy-PC correlation.
        loadings = Vt.T * np.sqrt(eigenvalues)
        scores = Z @ Vt.T
    except np.linalg.LinAlgError:
        # Fallback path for rare LAPACK non-convergence on large/ill-conditioned
        # matrices: eigendecomposition of the covariance matrix.
        cov = np.cov(Z, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        eigenvalues = np.clip(evals[order], 0.0, None)
        V = evecs[:, order]
        loadings = V * np.sqrt(eigenvalues)
        scores = Z @ V

    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)

    k = len(features)
    broken = _broken_stick(k)
    kaiser = eigenvalues > 1.0

    return {
        "mu": mu,
        "sigma": sigma,
        "eigenvalues": eigenvalues,
        "explained": explained,
        "cumulative": cumulative,
        "broken_stick": broken,
        "kaiser_keep": kaiser,
        "loadings": loadings,
        "scores": scores,
        "proxy_keys": list(features),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _write_variance_csv(report: dict, out_path: Path) -> None:
    k = len(report["proxy_keys"])
    df = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(k)],
        "eigenvalue": report["eigenvalues"],
        "explained_ratio": report["explained"],
        "cumulative_ratio": report["cumulative"],
        "broken_stick_ref": report["broken_stick"],
        "above_broken_stick": report["explained"] > report["broken_stick"],
        "kaiser_keep": report["kaiser_keep"],
    })
    df.to_csv(out_path, index=False)


def _write_loadings_csv(report: dict, out_path: Path) -> None:
    k = len(report["proxy_keys"])
    df = pd.DataFrame(
        report["loadings"],
        index=report["proxy_keys"],
        columns=[f"PC{i + 1}" for i in range(k)],
    )
    df.index.name = "proxy"
    df.to_csv(out_path)


def _write_scores_csv(df_meta: pd.DataFrame, report: dict,
                      per_entry_path: Path, per_method_path: Path,
                      k_keep: int) -> None:
    k = min(k_keep, report["scores"].shape[1])
    score_cols = [f"PC{i + 1}" for i in range(k)]
    scores = pd.DataFrame(report["scores"][:, :k], columns=score_cols)
    entry = pd.concat(
        [df_meta[["method", "visible_path", "condition"]].reset_index(drop=True), scores],
        axis=1,
    )
    entry.to_csv(per_entry_path, index=False)

    agg = entry.groupby("method")[score_cols].agg(["mean", "std"])
    agg.columns = [f"{pc}_{stat}" for pc, stat in agg.columns]
    agg.reset_index().to_csv(per_method_path, index=False)


def _plot_scree(report: dict, out_path: Path) -> None:
    k = len(report["proxy_keys"])
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, report["explained"], color="tab:blue", alpha=0.75, label="Explained ratio")
    ax.plot(x, report["broken_stick"], "o--", color="tab:red",
            label="Broken-stick reference")
    ax.plot(x, report["cumulative"], "s-", color="tab:green", alpha=0.7,
            label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Ratio")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    ax.set_title("PCA scree: explained variance vs broken-stick")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_loadings(report: dict, out_path: Path) -> None:
    k = len(report["proxy_keys"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(report["loadings"], aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"PC{i + 1}" for i in range(k)])
    ax.set_yticks(range(len(report["proxy_keys"])))
    ax.set_yticklabels(report["proxy_keys"])
    for i in range(len(report["proxy_keys"])):
        for j in range(k):
            val = report["loadings"][i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="Loading (proxy-PC correlation)")
    ax.set_title("PCA loadings")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_pc1_pc2(df_meta: pd.DataFrame, report: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    scores = report["scores"]
    methods = df_meta["method"].to_numpy()
    uniq = sorted(set(methods))
    cmap = plt.get_cmap("tab20")
    for idx, method in enumerate(uniq):
        mask = methods == method
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   s=12, alpha=0.45, color=cmap(idx % 20), label=method)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(f"PC1 ({report['explained'][0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({report['explained'][1] * 100:.1f}%)")
    ax.legend(loc="best", fontsize=7, ncols=2, framealpha=0.9)
    ax.set_title("PC1 vs PC2 (colored by method)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PCA over raw contribution proxies from the evaluation cache.",
    )
    p.add_argument("--dataset", choices=["kaist", "llvip"], default="llvip")
    p.add_argument("--condition", choices=["all", "day", "night"], default="night")
    p.add_argument("--cache-file", default=None,
                   help="Override evaluation-cache path (.pkl).")
    p.add_argument("--report-dir", default=None,
                   help="Output directory (default: <cache-dir>/reports/pca).")
    p.add_argument("--methods", nargs="*", default=None,
                   help="Restrict to these source methods (default: all except visible/lwir).")
    p.add_argument("--keep-anchors", action="store_true",
                   help="Keep visible/lwir in the analysis (off by default).")
    p.add_argument("--k-keep", type=int, default=3,
                   help="How many PCs to emit in score CSVs (default: 3).")
    p.add_argument("--include-channel-redundancy", action="store_true",
                   help="Append per-method channel_redundancy as 7th feature "
                        "(broadcast from methods-overview CSV).")
    p.add_argument("--overview-csv", default=None,
                   help="Path to methods-overview CSV. "
                        "Default: <cache-dir>/reports/<dataset>_<condition>_methods_overview.csv.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    cache_file = Path(args.cache_file) if args.cache_file else _default_cache_path(args.dataset, args.condition)
    if not cache_file.exists():
        print(f"[error] cache file not found: {cache_file}", file=sys.stderr)
        return 2

    report_dir = Path(args.report_dir) if args.report_dir else cache_file.parent / "reports" / "pca"
    report_dir.mkdir(parents=True, exist_ok=True)

    method_filter = set(args.methods) if args.methods else None
    condition_filter = args.condition if args.condition != "all" else None

    print(f"[pca] loading {cache_file}")
    df = load_proxy_table(
        cache_file,
        method_filter=method_filter,
        condition_filter=condition_filter,
        drop_anchors=not args.keep_anchors,
        include_channel_redundancy=args.include_channel_redundancy,
    )
    if df.empty:
        print("[error] no rows survived filters", file=sys.stderr)
        return 3
    print(f"[pca] rows: {len(df)}  methods: {df['method'].nunique()}")

    features = PROXY_KEYS
    tag_suffix = ""
    if args.include_channel_redundancy:
        if df.attrs.get("channel_redundancy_native", False):
            features = PROXY_KEYS + (CHANNEL_REDUNDANCY_KEY,)
            tag_suffix = "_with_cr"
            print("[pca] channel_redundancy read natively per image from cache")
        else:
            overview_csv = Path(args.overview_csv) if args.overview_csv else (
                cache_file.parent / "reports"
                / f"{args.dataset}_{args.condition}_methods_overview.csv"
            )
            if not overview_csv.exists():
                print(f"[error] overview CSV not found: {overview_csv}", file=sys.stderr)
                return 4
            df = merge_channel_redundancy(df, overview_csv)
            features = PROXY_KEYS + (CHANNEL_REDUNDANCY_KEY,)
            tag_suffix = "_with_cr"
            print(f"[pca] appended channel_redundancy via broadcast from {overview_csv.name}")

    # Guard against mixed caches where one or more selected features are NaN/Inf.
    df, dropped_nonfinite = _drop_nonfinite_feature_rows(df, features)
    if dropped_nonfinite:
        print(f"[pca] warning: dropped {dropped_nonfinite} rows with non-finite feature values")
    if df.empty:
        print("[error] no finite rows left after feature filtering", file=sys.stderr)
        return 5

    report = run_pca(df, features=features)

    tag = f"{args.dataset}_{args.condition}{tag_suffix}"
    _write_variance_csv(report, report_dir / f"pca_explained_variance_{tag}.csv")
    _write_loadings_csv(report, report_dir / f"pca_loadings_{tag}.csv")
    _write_scores_csv(df, report,
                      report_dir / f"pca_scores_per_entry_{tag}.csv",
                      report_dir / f"pca_scores_per_method_{tag}.csv",
                      k_keep=args.k_keep)
    _plot_scree(report, report_dir / f"pca_scree_{tag}.png")
    _plot_loadings(report, report_dir / f"pca_loadings_heatmap_{tag}.png")
    _plot_pc1_pc2(df, report, report_dir / f"pca_pc1_pc2_scatter_{tag}.png")

    # Terminal summary — first verdict.
    print("\n[pca] explained variance")
    for i, (e, c, bs) in enumerate(zip(report["explained"], report["cumulative"], report["broken_stick"])):
        flag = " *" if e > bs else "  "
        print(f"  PC{i + 1}: {e * 100:6.2f}%   cum {c * 100:6.2f}%   broken-stick {bs * 100:6.2f}%{flag}")
    n_above = int(np.sum(report["explained"] > report["broken_stick"]))
    n_kaiser = int(np.sum(report["kaiser_keep"]))
    print(f"\n[pca] components above broken-stick: {n_above}")
    print(f"[pca] components by Kaiser rule (eigenvalue > 1): {n_kaiser}")
    print(f"\n[pca] outputs written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
