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
import pywt
from tqdm import tqdm

try:
    cv.setNumThreads(1)
except Exception:
    pass

from utils.file_lock import FileLock
from utils.log_utils import log
from .data_loading import load_rgb_lwir_pair, visible_to_lwir_path
from .evaluation import (
    compute_contribution_rgb,
    _align_shapes,
    PROXY_VERSION,
    CALIBRATION_SAMPLES_VERSION,
)


def _calibration_sample_key(visible_path: str, alpha: float, sample_type: str = "blend") -> str:
    """Stable key for one calibration sample point."""
    lwir_path = visible_to_lwir_path(visible_path)
    try:
        vis_mtime = os.path.getmtime(visible_path)
    except OSError:
        vis_mtime = -1.0
    try:
        lwir_mtime = os.path.getmtime(lwir_path)
    except OSError:
        lwir_mtime = -1.0
    raw = (
        f"{PROXY_VERSION}|{CALIBRATION_SAMPLES_VERSION}|{visible_path}|{vis_mtime:.6f}|"
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


_PROXY_CACHE_KEYS = ("cont_vis_reg", "cont_vis_mi", "cont_vis_ssim",
                     "cont_vis_grad_combined", "cont_vis_spectral", "cont_vis_freq")


# Channel-concatenation patterns for calibration.  Each entry is
# (channel_sources, alpha_thermal) where channel_sources is a 3-tuple of
# "v0","v1","v2" (visible channels), "t" (thermal replicated), or
# "m0","m1","m2" (50/50 mix of that visible channel with thermal).
# Alpha reflects the effective thermal fraction: v=0, m=0.5, t=1 per channel.
_CHANNEL_CONCAT_PATTERNS: list[tuple[tuple[str, str, str], float]] = [
    # Pure substitution (original set)
    (("v0", "v1", "v2"), 0.0),          # α = 0/3
    (("v0", "v1", "t"),  1.0 / 3.0),    # α = 1/3
    (("v0", "t",  "v2"), 1.0 / 3.0),
    (("t",  "v1", "v2"), 1.0 / 3.0),
    (("v0", "t",  "t"),  2.0 / 3.0),    # α = 2/3
    (("t",  "v1", "t"),  2.0 / 3.0),
    (("t",  "t",  "v2"), 2.0 / 3.0),
    (("t",  "t",  "t"),  1.0),          # α = 3/3
    # Intra-channel 50/50 mix — fills gaps between substitution points
    (("v0", "v1", "m2"), 1.0 / 6.0),    # α = 0.5/3  (one channel half-mixed)
    (("v0", "m1", "v2"), 1.0 / 6.0),
    (("m0", "v1", "v2"), 1.0 / 6.0),
    (("v0", "m1", "t"),  1.0 / 2.0),    # α = 1.5/3  (one replaced + one mixed)
    (("m0", "v1", "t"),  1.0 / 2.0),
    (("m0", "t",  "v2"), 1.0 / 2.0),
    (("m0", "t",  "t"),  5.0 / 6.0),    # α = 2.5/3  (two replaced + one mixed)
    (("t",  "m1", "t"),  5.0 / 6.0),
    (("t",  "t",  "m2"), 5.0 / 6.0),
]


def _build_freq_blend_fused(visual: np.ndarray, lwir: np.ndarray,
                            alpha: float, wavelet: str = "db1") -> np.ndarray:
    """Frequency-domain blend via nonlinear detail selection.

    Linear blending of DWT coefficients is equivalent to linear pixel blending
    because DWT is a linear operator (blend is a no-op in the transform
    domain).  To produce a genuinely different signal we apply a nonlinear
    rule on the detail subbands:

      - Approximation (cA): linear blend (1-α)·cA_v + α·cA_t  — controls
        luminance, must remain smooth.
      - Details (cH, cV, cD): α-weighted max-abs selection, i.e. at each
        coefficient location we pick the source with larger magnitude, with
        α biasing the choice toward thermal.  This mirrors the `wavelet_max`
        fusion method and exercises `freq`/`spectral` proxies differently
        from the linear blend.

    Result is (H, W, 3) float64 [0, 1].
    """
    H, W = visual.shape[:2]
    a = float(alpha)
    fused_channels = []
    for ch in range(3):
        v_ch = visual[..., ch]
        cA_v, (cH_v, cV_v, cD_v) = pywt.dwt2(v_ch, wavelet)
        cA_t, (cH_t, cV_t, cD_t) = pywt.dwt2(lwir, wavelet)
        cA = (1.0 - a) * cA_v + a * cA_t

        def _detail_select(c_v: np.ndarray, c_t: np.ndarray) -> np.ndarray:
            # α-biased max-abs: compare |c_t|*α to |c_v|*(1-α) and pick the
            # coefficient with larger effective magnitude.
            mask = (np.abs(c_t) * a) > (np.abs(c_v) * (1.0 - a))
            return np.where(mask, c_t, c_v)

        cH = _detail_select(cH_v, cH_t)
        cV = _detail_select(cV_v, cV_t)
        cD = _detail_select(cD_v, cD_t)
        rec = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
        fused_channels.append(rec[:H, :W])
    fused = np.stack(fused_channels, axis=2)
    return np.clip(fused, 0.0, 1.0)


def _build_nonlinear_fused(visual: np.ndarray, lwir: np.ndarray,
                           alpha: float) -> np.ndarray:
    """Nonlinear (geometric mean) blend: V^(1-α) · T_rgb^α.

    Both inputs assumed float64 [0, 1].  Epsilon floor avoids 0^x = NaN.
    Result is (H, W, 3) float64 [0, 1].
    """
    eps = 1e-8
    a = float(alpha)
    lwir_rgb = np.repeat(lwir[..., None], 3, axis=2)
    fused = np.power(visual + eps, 1.0 - a) * np.power(lwir_rgb + eps, a)
    return np.clip(fused, 0.0, 1.0)


def _build_channel_concat_fused(visual: np.ndarray, lwir: np.ndarray,
                                 pattern: tuple[str, str, str]) -> np.ndarray:
    """Build a 3-channel fused image by concatenating visible/thermal channels.

    *visual* is (H,W,3) float64 [0,1], *lwir* is (H,W) float64 [0,1].
    *pattern* is a 3-tuple where each element is one of:
      "v0","v1","v2" — visible channel
      "t"            — thermal (replicated)
      "m0","m1","m2" — 50/50 mix of visible channel with thermal
    """
    source = {
        "v0": visual[..., 0], "v1": visual[..., 1], "v2": visual[..., 2],
        "t": lwir,
        "m0": 0.5 * visual[..., 0] + 0.5 * lwir,
        "m1": 0.5 * visual[..., 1] + 0.5 * lwir,
        "m2": 0.5 * visual[..., 2] + 0.5 * lwir,
    }
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
    visual, lwir = load_rgb_lwir_pair(visible_path, equalization=equalization)
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

    # --- Type 3: Frequency-domain (wavelet) blend ---
    for alpha in alpha_grid_list:
        fused = _build_freq_blend_fused(visual, lwir, alpha)
        _record_sample(compute_contribution_rgb(visual, lwir, fused), alpha, "freq_blend")

    # --- Type 4: Nonlinear (geometric mean) blend ---
    for alpha in alpha_grid_list:
        fused = _build_nonlinear_fused(visual, lwir, alpha)
        _record_sample(compute_contribution_rgb(visual, lwir, fused), alpha, "nonlinear")

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


def _average_proxy_weights_across_formats(
    calibrations_by_type: dict[str, dict],
) -> dict[str, float]:
    """Average normalised IVW proxy weights across all calibration formats.

    Each format contributes equally (simple mean of normalised weight vectors).
    This reduces the bias that any single calibration format introduces.
    Returns a dict {proxy_name: averaged_weight} summing to ~1.0.
    """
    weight_vectors: list[dict[str, float]] = []
    for stype, cal in calibrations_by_type.items():
        # Key matches `fit_contribution_calibration` writer in evaluation.py
        # (was previously `per_proxy_calibrations`, which silently fell back to
        # the uniform 1/n branch and bypassed the soft-IVW averaging).
        per_proxy_cal = cal.get("proxy_calibrations", {})
        if not per_proxy_cal:
            continue
        raw_weights = {}
        for pkey, pcal in per_proxy_cal.items():
            w = pcal.get("ivw_weight", 0.0)
            if w > 0:
                raw_weights[pkey] = w
        if not raw_weights:
            continue
        # Normalise within this format
        wsum = sum(raw_weights.values())
        weight_vectors.append({k: v / wsum for k, v in raw_weights.items()})

    if not weight_vectors:
        # Fallback: uniform over known proxies
        n = len(_PROXY_CACHE_KEYS)
        return {k: 1.0 / n for k in _PROXY_CACHE_KEYS}

    # Average across formats
    all_keys = set()
    for wv in weight_vectors:
        all_keys.update(wv.keys())
    averaged = {}
    for k in all_keys:
        averaged[k] = sum(wv.get(k, 0.0) for wv in weight_vectors) / len(weight_vectors)
    # Re-normalise
    total = sum(averaged.values())
    if total > 0:
        averaged = {k: v / total for k, v in averaged.items()}
    return averaged


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
    _SAMPLE_TYPES = ("blend", "concat", "freq_blend", "nonlinear")
    all_raw: dict[str, list[float]] = {st: [] for st in _SAMPLE_TYPES}
    all_alpha: dict[str, list[float]] = {st: [] for st in _SAMPLE_TYPES}
    all_proxy: dict[str, dict[str, list[float]]] = {st: {} for st in _SAMPLE_TYPES}
    samples_cache = _load_cache(calibration_samples_file) if calibration_samples_file else {}
    if not isinstance(samples_cache, dict):
        samples_cache = {}
    cached_images = samples_cache.get("images", {}) if isinstance(samples_cache.get("images"), dict) else {}

    def _image_cache_key(visible_path: str) -> str:
        return _calibration_sample_key(visible_path, alpha=0.0, sample_type="image_all")

    def _collect(raw_vals, alpha_vals, per_proxy, stypes):
        for rv, av, st in zip(raw_vals, alpha_vals, stypes):
            if st not in all_raw:
                all_raw[st] = []
                all_alpha[st] = []
                all_proxy[st] = {}
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
    # blend + freq_blend + nonlinear each produce len(alpha_grid) points; concat has its own count
    points_per_image = 3 * len(alpha_grid) + n_concat
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

        def _flush_calibration_cache() -> None:
            if not calibration_samples_file:
                return
            samples_cache["images"] = cached_images
            samples_cache["version"] = 3
            _save_cache(calibration_samples_file, samples_cache)

        with executor_cls(max_workers=max_workers) as executor:
            spec_batches = _batched(pending_specs, task_chunksize)
            # 5% of total batches with an absolute cap of 25.  Caps worst-case
            # lost work to 5% on small/medium runs and to a fixed ≤25 batches
            # on very large runs (where 5% would be excessive).
            checkpoint_every_batches = max(1, min(len(spec_batches) // 20, 25))
            log(f"Calibration batching: batch_size={max(1, int(task_chunksize))} "
                f"total_batches={len(spec_batches)} checkpoint_every={checkpoint_every_batches}")
            futures = {
                executor.submit(
                    _calibration_samples_for_path_specs_batch,
                    batch,
                    equalization,
                ): batch
                for batch in spec_batches
            }
            future_desc = f"Calibration batches ({len(futures)} batches, eq={equalization}, mode={execution_mode})"
            batches_since_checkpoint = 0
            try:
                for future in tqdm(as_completed(futures), total=len(futures), desc=future_desc, leave=False):
                    try:
                        sample_rows = future.result()
                    except Exception as exc:
                        failed_batches += 1
                        if failed_batches <= 5:
                            log(f"Calibration batch failed ({failed_batches}): {exc}")
                        batches_since_checkpoint += 1
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
                    batches_since_checkpoint += 1
                    if batches_since_checkpoint >= checkpoint_every_batches:
                        _flush_calibration_cache()
                        batches_since_checkpoint = 0
            except BaseException:
                # Guarantees KeyboardInterrupt / OOM / crash still flushes
                # partial progress before re-raising.
                log("[warn] calibration interrupted; flushing cache before re-raise")
                _flush_calibration_cache()
                raise
            if failed_batches > 0:
                log(f"Calibration batch failures: {failed_batches} batches")

    if calibration_samples_file:
        samples_cache["images"] = cached_images
        samples_cache["version"] = 3
        _save_cache(calibration_samples_file, samples_cache)

    if reused_points > 0 or new_points > 0:
        log(f"Calibration sample cache: reused={reused_points} new={new_points} total={reused_points + new_points}")

    sample_counts = {st: len(all_raw[st]) for st in all_raw}
    total_samples = sum(sample_counts.values())
    if total_samples == 0:
        raise RuntimeError("Calibration failed: no valid synthetic samples were generated")

    # --- Fit independent calibrations for each type ---------------------------
    calibrations_by_type = {}
    for stype, raw_list in all_raw.items():
        if not raw_list:
            continue
        per_proxy = {pk: np.asarray(pv, dtype=np.float64)
                     for pk, pv in all_proxy[stype].items() if pv}
        cal = fit_contribution_calibration(
            np.asarray(raw_list), np.asarray(all_alpha[stype]),
            per_proxy_values=per_proxy)
        cal["raw_points_cont_vis"] = np.asarray(raw_list, dtype=np.float64)
        cal["raw_points_alpha"] = np.asarray(all_alpha[stype], dtype=np.float64)
        for pkey, pvals in all_proxy[stype].items():
            cal[f"raw_points_{pkey}"] = np.asarray(pvals, dtype=np.float64)
        calibrations_by_type[stype] = cal
        knots = cal.get("raw_knots", [])
        log(f"Calibration [{stype}]: knots={len(knots)} "
            f"raw_range={knots[0] if len(knots) else None}..{knots[-1] if len(knots) else None}")

    # --- Multi-format proxy weight averaging ----------------------------------
    # Average IVW weights across all available formats to reduce single-format
    # bias (e.g. blend favoring reg).  Each format contributes equally.
    averaged_proxy_weights = _average_proxy_weights_across_formats(calibrations_by_type)

    # --- Combined calibration dict -------------------------------------------
    # Use blend as primary calibration for scale correction (backward-compat);
    # best-fit matching per method happens downstream in evaluation.
    calibration = calibrations_by_type.get("blend",
                  calibrations_by_type.get("freq_blend",
                  calibrations_by_type.get("nonlinear",
                  calibrations_by_type.get("concat", {}))))
    calibration["calibrations_by_type"] = calibrations_by_type
    calibration["averaged_proxy_weights"] = averaged_proxy_weights
    for st, cnt in sample_counts.items():
        calibration[f"n_{st}"] = cnt
    log(f"Calibration done: {' '.join(f'{st}={cnt}' for st, cnt in sample_counts.items() if cnt > 0)}")
    log(f"Averaged proxy weights: { {k: f'{v:.4f}' for k, v in averaged_proxy_weights.items()} }")
    return calibration
