from __future__ import annotations

"""
Calibration building from synthetic RGB/LWIR mixtures with caching support.

Generates calibration curves by creating synthetic mixtures of known alpha
(thermal contribution), computing metrics on them, and fitting monotonic curves.
"""

import os
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import cv2 as cv
from tqdm import tqdm

try:
    cv.setNumThreads(1)
except Exception:
    pass

from utils.file_lock import FileLock
from utils.log_utils import log
from .evaluation import compute_contribution_rgb, _align_shapes, CONTRIBUTION_METRIC_VERSION


def _calibration_sample_key(visible_path: str, alpha: float, sample_type: str = "blend") -> str:
    """Stable key for one calibration sample point."""
    lwir_path = visible_path.replace(os.sep + "visible" + os.sep, os.sep + "lwir" + os.sep)
    try:
        vis_mtime = os.path.getmtime(visible_path)
    except OSError:
        vis_mtime = -1.0
    try:
        lwir_mtime = os.path.getmtime(lwir_path)
    except OSError:
        lwir_mtime = -1.0
    raw = (
        f"{CONTRIBUTION_METRIC_VERSION}|{visible_path}|{vis_mtime:.6f}|"
        f"{lwir_path}|{lwir_mtime:.6f}|{float(alpha):.8f}|{sample_type}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _save_cache(cache_file: str | None, cache_data: dict) -> None:
    """Persist cache dictionary to pickle."""
    if cache_file is None:
        return
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    lock_path = f"{cache_file}.lock"
    with FileLock(lock_path):
        with open(cache_file, "wb") as file:
            pickle.dump(cache_data, file)


def _load_cache(cache_file: str | None) -> dict:
    """Load cache dictionary from pickle if available."""
    if cache_file is None or not os.path.exists(cache_file):
        return {}
    lock_path = f"{cache_file}.lock"
    try:
        with FileLock(lock_path):
            with open(cache_file, "rb") as file:
                data = pickle.load(file)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def _load_rgb_lwir_pair(visible_path: str, equalization: str = "no_equalization") -> tuple[np.ndarray, np.ndarray]:
    """Load a visible/LWIR pair from a visible image path with optional equalization."""
    visible_bgr = cv.imread(visible_path, cv.IMREAD_COLOR)
    lwir_path = visible_path.replace(os.sep + "visible" + os.sep, os.sep + "lwir" + os.sep)
    lwir = cv.imread(lwir_path, cv.IMREAD_GRAYSCALE)

    if visible_bgr is None or lwir is None:
        raise FileNotFoundError(f"Could not read visible/LWIR pair for {visible_path}")

    if equalization in ("th_equalization", "rgb_th_equalization", "rgb_equalization"):
        from Dataset.th_equalization import th_equalization, rgb_equalization
        if equalization in ("th_equalization", "rgb_th_equalization"):
            lwir = th_equalization(lwir, "clahe")
        if equalization in ("rgb_equalization", "rgb_th_equalization"):
            visible_bgr = rgb_equalization(visible_bgr, "clahe")

    visible = cv.cvtColor(visible_bgr, cv.COLOR_BGR2RGB)

    if visible.ndim == 2:
        visible = np.repeat(visible[..., None], 3, axis=2)
    if visible.shape[-1] > 3:
        visible = visible[..., :3]
    if lwir.ndim == 3:
        lwir = lwir[..., 0]

    return visible, lwir


_PROXY_CACHE_KEYS = ("cont_vis_reg", "cont_vis_mi", "cont_vis_ssim",
                     "cont_vis_grad_combined", "cont_vis_spectral", "cont_vis_freq")


# Channel-concatenation patterns for calibration.  Each entry is
# (channel_sources, alpha_thermal) where channel_sources is a 3-tuple of
# "v0","v1","v2" (visible channels) or "t" (thermal replicated).
_CHANNEL_CONCAT_PATTERNS: list[tuple[tuple[str, str, str], float]] = [
    (("v0", "v1", "v2"), 0.0),        # 3/3 visible
    (("v0", "v1", "t"),  1.0 / 3.0),  # 2/3 visible
    (("v0", "t",  "v2"), 1.0 / 3.0),
    (("t",  "v1", "v2"), 1.0 / 3.0),
    (("v0", "t",  "t"),  2.0 / 3.0),  # 1/3 visible
    (("t",  "v1", "t"),  2.0 / 3.0),
    (("t",  "t",  "v2"), 2.0 / 3.0),
    (("t",  "t",  "t"),  1.0),        # 0/3 visible
]


def _build_channel_concat_fused(visual: np.ndarray, lwir: np.ndarray,
                                 pattern: tuple[str, str, str]) -> np.ndarray:
    """Build a 3-channel fused image by concatenating visible/thermal channels.

    *visual* is (H,W,3) float64 [0,1], *lwir* is (H,W) float64 [0,1].
    *pattern* is a 3-tuple of "v0","v1","v2","t".
    """
    source = {"v0": visual[..., 0], "v1": visual[..., 1], "v2": visual[..., 2], "t": lwir}
    return np.stack([source[p] for p in pattern], axis=2)


def _calibration_samples_for_visible_path(
    visible_path: str,
    alpha_grid_list: list[float],
    equalization: str = "no_equalization",
) -> tuple[list[float], list[float], dict[str, list[float]], list[str]]:
    """Compute synthetic-mixture calibration samples for one image pair.

    Generates two types of calibration points:
      1. ``blend``: Linear alpha blending  fused = (1-α)·vis + α·lwir_rgb
      2. ``concat``: Channel concatenation  e.g. [R, G, T] = 1/3 thermal

    Returns (raw_visible_percent, alpha_lwir, per_proxy_values, sample_types).
    """
    visual, lwir = _load_rgb_lwir_pair(visible_path, equalization=equalization)
    visual = visual.astype(np.float64)
    lwir = lwir.astype(np.float64)

    if visual.max() > 1.0:
        visual = visual / 255.0
    if lwir.max() > 1.0:
        lwir = lwir / 255.0

    visual, lwir = _align_shapes(visual, lwir)
    lwir_rgb = np.repeat(lwir[..., None], 3, axis=2)

    raw_visible_percent: list[float] = []
    alpha_lwir: list[float] = []
    per_proxy_values: dict[str, list[float]] = {}
    sample_types: list[str] = []

    def _record_sample(metrics: dict, alpha: float, stype: str) -> None:
        raw_visible_percent.append(float(metrics["cont_vis"]))
        alpha_lwir.append(float(alpha))
        sample_types.append(stype)
        for key in _PROXY_CACHE_KEYS:
            per_proxy_values.setdefault(key, []).append(float(metrics.get(key, 0.0)))

    # --- Type 1: Linear alpha blending ---
    for alpha in alpha_grid_list:
        fused = (1.0 - alpha) * visual + alpha * lwir_rgb
        _record_sample(compute_contribution_rgb(visual, lwir, fused), alpha, "blend")

    # --- Type 2: Channel concatenation ---
    for pattern, alpha in _CHANNEL_CONCAT_PATTERNS:
        fused = _build_channel_concat_fused(visual, lwir, pattern)
        _record_sample(compute_contribution_rgb(visual, lwir, fused), alpha, "concat")

    return raw_visible_percent, alpha_lwir, per_proxy_values, sample_types


def _calibration_samples_for_path_specs_batch(
    specs: list[tuple[str, list[float]]],
    equalization: str = "no_equalization",
) -> list[tuple[str, float, float, dict[str, float], str]]:
    """Compute calibration rows (path, raw_visible, alpha, proxy_dict, sample_type)."""
    rows = []
    for visible_path, alpha_list in specs:
        raw_vals, alpha_vals, per_proxy, stypes = _calibration_samples_for_visible_path(
            visible_path, alpha_list, equalization=equalization)
        for i, (raw_value, alpha) in enumerate(zip(raw_vals, alpha_vals)):
            proxy_row = {k: v[i] for k, v in per_proxy.items()}
            rows.append((visible_path, float(raw_value), float(alpha), proxy_row, stypes[i]))
    return rows


def _batched(items: list, batch_size: int) -> list[list]:
    """Split a list into fixed-size batches."""
    size = max(1, int(batch_size))
    return [items[index:index + size] for index in range(0, len(items), size)]


def build_contribution_calibration_from_dataset(
    visible_paths: list[str],
    alpha_grid: np.ndarray | None = None,
    max_images: int | None = None,
    workers: int = 1,
    execution_mode: str = "thread",
    task_chunksize: int = 1,
    calibration_samples_file: str | None = None,
    equalization: str = "no_equalization",
) -> dict:
    """Build a calibration from synthetic alpha mixtures generated on dataset pairs."""
    from .evaluation import fit_contribution_calibration

    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 11)
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64)

    sample_paths = list(visible_paths)
    if max_images is not None:
        sample_paths = sample_paths[:max_images]

    # --- Accumulators per sample type ---
    all_raw: dict[str, list[float]] = {"blend": [], "concat": []}
    all_alpha: dict[str, list[float]] = {"blend": [], "concat": []}
    all_proxy: dict[str, dict[str, list[float]]] = {"blend": {}, "concat": {}}
    samples_cache = _load_cache(calibration_samples_file) if calibration_samples_file else {}
    if not isinstance(samples_cache, dict):
        samples_cache = {}
    cached_images = samples_cache.get("images", {}) if isinstance(samples_cache.get("images"), dict) else {}

    def _image_cache_key(visible_path: str) -> str:
        return _calibration_sample_key(visible_path, alpha=0.0, sample_type="image_all")

    def _collect(raw_vals, alpha_vals, per_proxy, stypes):
        for rv, av, st in zip(raw_vals, alpha_vals, stypes):
            all_raw[st].append(rv)
            all_alpha[st].append(av)
        for pkey in _PROXY_CACHE_KEYS:
            pvals = per_proxy.get(pkey, [])
            for pv, st in zip(pvals, stypes):
                all_proxy[st].setdefault(pkey, []).append(pv)

    pending_paths: list[str] = []
    reused_points = 0
    for visible_path in sample_paths:
        img_key = _image_cache_key(visible_path)
        cached = cached_images.get(img_key)
        if isinstance(cached, dict) and "raw_visible" in cached and "sample_types" in cached:
            stypes = cached["sample_types"]
            _collect(cached["raw_visible"], cached["alpha"],
                     {pk: cached.get(pk, []) for pk in _PROXY_CACHE_KEYS}, stypes)
            reused_points += len(cached["raw_visible"])
        else:
            pending_paths.append(visible_path)

    alpha_grid_list = alpha_grid.tolist()
    pending_specs = [(p, alpha_grid_list) for p in pending_paths]
    new_points = 0

    max_workers = max(1, int(workers))
    n_concat = len(_CHANNEL_CONCAT_PATTERNS)
    points_per_image = len(alpha_grid) + n_concat
    log(
        f"Calibration setup: images={len(sample_paths)} alpha_steps={len(alpha_grid)} "
        f"concat_patterns={n_concat} points_per_image={points_per_image} "
        f"equalization={equalization} workers={max_workers} mode={execution_mode}"
    )
    total_points = len(sample_paths) * points_per_image
    pending_points = len(pending_paths) * points_per_image
    log(
        f"Calibration workload: total_points={total_points} reused_points={reused_points} "
        f"pending_points={pending_points} pending_images={len(pending_paths)}"
    )

    def _store_image_results(visible_path: str, raw_vals: list, alpha_vals: list,
                              per_proxy: dict[str, list[float]], stypes: list[str]) -> None:
        nonlocal new_points
        _collect(raw_vals, alpha_vals, per_proxy, stypes)
        img_key = _image_cache_key(visible_path)
        cache_entry = {
            "raw_visible": [float(v) for v in raw_vals],
            "alpha": [float(a) for a in alpha_vals],
            "sample_types": list(stypes),
            "visible_path": visible_path,
        }
        for pkey, pvals in per_proxy.items():
            cache_entry[pkey] = [float(v) for v in pvals]
        cached_images[img_key] = cache_entry
        new_points += len(raw_vals)

    tqdm_desc = (
        f"Calibration images ({len(sample_paths)} images, {points_per_image} pts/img, eq={equalization})"
    )
    if max_workers == 1:
        iterator = tqdm(pending_specs, total=len(pending_specs), desc=tqdm_desc, leave=False)
        for visible_path, alpha_list in iterator:
            raw_vals, alpha_vals, per_proxy, stypes = _calibration_samples_for_visible_path(
                visible_path, alpha_list, equalization=equalization,
            )
            _store_image_results(visible_path, raw_vals, alpha_vals, per_proxy, stypes)
    else:
        executor_cls = ProcessPoolExecutor if execution_mode == "process" else ThreadPoolExecutor
        failed_batches = 0
        with executor_cls(max_workers=max_workers) as executor:
            spec_batches = _batched(pending_specs, task_chunksize)
            log(f"Calibration batching: batch_size={max(1, int(task_chunksize))} total_batches={len(spec_batches)}")
            futures = {
                executor.submit(
                    _calibration_samples_for_path_specs_batch,
                    batch,
                    equalization,
                ): batch
                for batch in spec_batches
            }
            future_desc = f"Calibration batches ({len(futures)} batches, eq={equalization}, mode={execution_mode})"
            for future in tqdm(as_completed(futures), total=len(futures), desc=future_desc, leave=False):
                try:
                    sample_rows = future.result()
                except Exception as exc:
                    failed_batches += 1
                    if failed_batches <= 5:
                        log(f"Calibration batch failed ({failed_batches}): {exc}")
                    continue
                by_path: dict[str, tuple[list, list, dict, list]] = {}
                for visible_path, raw_value, alpha, proxy_row, stype in sample_rows:
                    if visible_path not in by_path:
                        by_path[visible_path] = ([], [], {k: [] for k in _PROXY_CACHE_KEYS}, [])
                    rv, av, pp, st = by_path[visible_path]
                    rv.append(raw_value)
                    av.append(alpha)
                    st.append(stype)
                    for pkey, pval in proxy_row.items():
                        pp[pkey].append(pval)
                for visible_path, (rv, av, pp, st) in by_path.items():
                    _store_image_results(visible_path, rv, av, pp, st)
                if failed_batches > 0:
                    log(f"Calibration batch failures: {failed_batches} batches")

    if calibration_samples_file:
        samples_cache["images"] = cached_images
        samples_cache["version"] = 2
        _save_cache(calibration_samples_file, samples_cache)

    if reused_points > 0 or new_points > 0:
        log(f"Calibration sample cache: reused={reused_points} new={new_points} total={reused_points + new_points}")

    n_blend = len(all_raw["blend"])
    n_concat = len(all_raw["concat"])
    if n_blend == 0 and n_concat == 0:
        raise RuntimeError("Calibration failed: no valid synthetic samples were generated")

    # --- Fit independent calibrations for each type ---------------------------
    calibrations_by_type = {}
    for stype in ("blend", "concat"):
        if not all_raw[stype]:
            continue
        # Build per-proxy arrays for per-proxy calibration
        per_proxy = {pk: np.asarray(pv, dtype=np.float64)
                     for pk, pv in all_proxy[stype].items() if pv}
        cal = fit_contribution_calibration(
            np.asarray(all_raw[stype]), np.asarray(all_alpha[stype]),
            per_proxy_values=per_proxy)
        # Attach per-proxy raw data for diagnostic plots
        cal["raw_points_cont_vis"] = np.asarray(all_raw[stype], dtype=np.float64)
        cal["raw_points_alpha"] = np.asarray(all_alpha[stype], dtype=np.float64)
        for pkey, pvals in all_proxy[stype].items():
            cal[f"raw_points_{pkey}"] = np.asarray(pvals, dtype=np.float64)
        calibrations_by_type[stype] = cal
        knots = cal.get("raw_knots", [])
        log(f"Calibration [{stype}]: knots={len(knots)} "
            f"raw_range={knots[0] if len(knots) else None}..{knots[-1] if len(knots) else None}")

    # --- Combined calibration: average of both types --------------------------
    # Use blend as primary; if concat exists, store it alongside for comparison.
    calibration = calibrations_by_type.get("blend", calibrations_by_type.get("concat", {}))
    calibration["calibrations_by_type"] = calibrations_by_type
    calibration["n_blend"] = n_blend
    calibration["n_concat"] = n_concat
    log(f"Calibration done: blend_points={n_blend} concat_points={n_concat}")
    return calibration
