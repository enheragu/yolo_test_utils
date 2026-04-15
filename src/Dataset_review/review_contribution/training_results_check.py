from __future__ import annotations

"""
Parse training results CSV and correlate them with computed latent_z.

Encapsulates CSV parsing so that changes in column naming or format only
require editing this module.  Produces per-method training-metric statistics
keyed by (fusion_type, condition, equalization) that the pipeline can overlay
on latent_z for correspondence analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_CSV_PATH = Path(__file__).parent / "raw_data" / "raw_training_data.csv"

# Training metrics we track (matches CSV column names).
TRAINING_METRICS = ["P", "R", "mAP50", "mAP50-95"]

# Map fusion-method names as produced by the contribution pipeline to the
# simpler type tags found in the training CSV.  When a method has no entry in
# the CSV (e.g. vths_v2 not yet trained), it is silently skipped when plotting.
_METHOD_NAME_TO_CSV_TYPE = {
    "sobel_weighted": "sobel",
    "ssim_v2": "ssim",
    "rgbt_v2": "rgbt",
    "curvelet_max": "curvelet",
    "wavelet_max": "wavelet",
}


def _canonical_type(method_name: str) -> str:
    """Return the CSV ``Type`` tag that corresponds to *method_name*."""
    return _METHOD_NAME_TO_CSV_TYPE.get(method_name, method_name)


def load_training_results(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    target_class: str = "person",
) -> pd.DataFrame:
    """Load training results CSV, filter to a target class, drop bad rows.

    Parameters
    ----------
    csv_path : str | Path
        Path to the raw CSV.  Defaults to raw_data/raw_training_data.csv.
    target_class : str
        Only keep rows for this class (default ``person``).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["Class"] == target_class].copy()
    # Drop rows with unknown equalization or unusable types
    df = df[df["Group Key"].isin(
        ["no_equalization", "rgb_equalization", "th_equalization", "rgb_th_equalization"])]
    df = df[df["Condition"].isin(["day", "night"])]
    return df.reset_index(drop=True)


def get_runs(
    df: pd.DataFrame,
    fusion_type: str,
    condition: str,
    equalization: str,
) -> pd.DataFrame:
    """Return all runs matching ``(fusion_type, condition, equalization)``."""
    return df[
        (df["Type"] == fusion_type)
        & (df["Condition"] == condition)
        & (df["Group Key"] == equalization)
    ]


def build_method_metrics(
    df: pd.DataFrame,
    method_name: str,
    condition: str,
    equalization: str,
) -> dict | None:
    """Summarise training metrics for one fusion method × condition × equalization.

    Returns a dict with per-metric ``values`` list (all runs) and ``mean``/``std``,
    or ``None`` if no matching runs were found.
    """
    csv_type = _canonical_type(method_name)
    runs = get_runs(df, csv_type, condition, equalization)
    if runs.empty:
        return None
    info = {"method_name": method_name, "csv_type": csv_type, "n_runs": len(runs),
            "metrics": {}}
    for metric in TRAINING_METRICS:
        vals = runs[metric].dropna().to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        info["metrics"][metric] = {
            "values": vals,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
        }
    return info


def build_dataset_metrics(
    df: pd.DataFrame,
    method_names: list[str],
    condition: str,
    equalization: str,
) -> dict[str, dict]:
    """Return a mapping ``{method_name -> metrics dict}`` for a given slice."""
    out = {}
    for name in method_names:
        entry = build_method_metrics(df, name, condition, equalization)
        if entry is not None:
            out[name] = entry
    return out
