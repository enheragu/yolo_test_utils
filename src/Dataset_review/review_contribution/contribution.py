from __future__ import annotations

"""
RGB/LWIR fusion method evaluation and aggregation.

Orchestrates multi-method evaluation on datasets with caching, parallelization,
and result aggregation.
"""

import os
import hashlib
import inspect
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import cv2 as cv
from tqdm import tqdm

try:
    cv.setNumThreads(1)
except Exception:
    pass

from utils.file_lock import FileLock
from utils.log_utils import log, logCoolMessage
from .evaluation import (
    compute_contribution_rgb,
    aggregate_contribution_results,
    apply_contribution_calibration,
    _with_calibration,
    CONTRIBUTION_METRIC_VERSION,
)
from .calibration import build_contribution_calibration_from_dataset, _load_cache, _save_cache, _batched


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

KAIST_DAY_SETS = {"set00", "set01", "set02", "set06", "set07", "set08"}
KAIST_NIGHT_SETS = {"set03", "set04", "set05", "set09", "set10", "set11"}


def _infer_kaist_condition(image_path: str) -> str:
    """Infer day/night from the KAIST set folder name in the image path."""
    for part in os.path.normpath(image_path).split(os.sep):
        if part in KAIST_DAY_SETS:
            return "day"
        if part in KAIST_NIGHT_SETS:
            return "night"
    return "unknown"


def _iter_kaist_visible_images(dataset_root: str):
    """Yield visible image paths under a KAIST-style dataset tree."""
    for folder_set in sorted(os.listdir(dataset_root)):
        folder_path = os.path.join(dataset_root, folder_set)
        if not os.path.isdir(folder_path):
            continue
        for subfolder_set in sorted(os.listdir(folder_path)):
            visible_folder = os.path.join(folder_path, subfolder_set, "visible", "images")
            if not os.path.isdir(visible_folder):
                continue
            for filename in sorted(os.listdir(visible_folder)):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    yield os.path.join(visible_folder, filename)


def _iter_llvip_visible_images(dataset_root: str):
    """Yield visible image paths under an LLVIP YOLO-style dataset tree."""
    for split in sorted(os.listdir(dataset_root)):
        split_path = os.path.join(dataset_root, split)
        if not os.path.isdir(split_path):
            continue
        visible_folder = os.path.join(split_path, "visible", "images")
        if not os.path.isdir(visible_folder):
            continue
        for filename in sorted(os.listdir(visible_folder)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                yield os.path.join(visible_folder, filename)


def _iter_visible_images(dataset_root: str, dataset: str):
    """Dataset-aware visible image iterator."""
    if dataset == "kaist":
        yield from _iter_kaist_visible_images(dataset_root)
    elif dataset == "llvip":
        yield from _iter_llvip_visible_images(dataset_root)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")


def _infer_condition(image_path: str, dataset: str) -> str:
    """Infer condition label for a given dataset."""
    if dataset == "kaist":
        return _infer_kaist_condition(image_path)
    if dataset == "llvip":
        return "night"
    return "unknown"


def _split_calibration_eval(
    visible_paths: list[str],
    calibration_ratio: float,
    calibration_images: int | None,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split paths into disjoint calibration and evaluation subsets."""
    paths = list(visible_paths)
    if not paths:
        return [], []

    rng = np.random.default_rng(seed)
    rng.shuffle(paths)

    if calibration_images is not None:
        n_cal = max(0, min(calibration_images, len(paths)))
    else:
        n_cal = int(round(len(paths) * calibration_ratio))
        n_cal = max(1, min(n_cal, len(paths)))

    # Keep at least one evaluation sample when possible.
    if len(paths) > 1 and n_cal >= len(paths):
        n_cal = len(paths) - 1

    calibration_paths = paths[:n_cal]
    evaluation_paths = paths[n_cal:] if n_cal < len(paths) else paths
    return calibration_paths, evaluation_paths


# ---------------------------------------------------------------------------
# Caching and fingerprinting
# ---------------------------------------------------------------------------

def _method_fingerprint(method_name: str, fusion_function) -> str:
    """Best-effort fingerprint of method implementation for cache invalidation."""
    module_name = getattr(fusion_function, "__module__", "unknown")
    qualname = getattr(fusion_function, "__qualname__", method_name)
    try:
        source = inspect.getsource(fusion_function)
    except Exception:
        source = ""
    raw = f"{module_name}|{qualname}|{source}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_key(method_name: str, method_fp: str, visible_path: str, equalization: str) -> str:
    """Stable cache key that tracks source image mtimes."""
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
        f"{CONTRIBUTION_METRIC_VERSION}|{method_name}|{method_fp}|{equalization}|"
        f"{visible_path}|{vis_mtime:.6f}|{lwir_mtime:.6f}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fusion method registry
# ---------------------------------------------------------------------------

_WORKER_METHODS: dict | None = None
_SHARED_MULTI_OUTPUT_PRIMARY = {
    "wavelet_max": "wavelet",
    "curvelet_max": "curvelet",
}


def _get_worker_methods() -> dict:
    """Lazy method registry for process workers."""
    global _WORKER_METHODS
    if _WORKER_METHODS is None:
        _WORKER_METHODS = _get_default_fusion_methods()
    return _WORKER_METHODS


def _get_default_fusion_methods() -> dict:
    """Load fusion method registry from Dataset.constants to avoid duplication."""
    from Dataset.constants import dataset_options

    def _identity_visible(visible_bgr, lwir):
        return visible_bgr

    def _identity_lwir(visible_bgr, lwir):
        return cv.cvtColor(lwir, cv.COLOR_GRAY2BGR)

    methods = {}
    methods["visible"] = _identity_visible
    methods["lwir"] = _identity_lwir
    for method_name, cfg in dataset_options.items():
        merge_fn = cfg.get("merge")
        if callable(merge_fn):
            methods[method_name] = merge_fn
    return methods


# ---------------------------------------------------------------------------
# Image loading utilities
# ---------------------------------------------------------------------------

def _load_bgr_lwir_pair(visible_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a visible/LWIR pair in OpenCV BGR order for fusion functions."""
    visible = cv.imread(visible_path, cv.IMREAD_COLOR)
    lwir_path = visible_path.replace(os.sep + "visible" + os.sep, os.sep + "lwir" + os.sep)
    lwir = cv.imread(lwir_path, cv.IMREAD_GRAYSCALE)

    if visible is None or lwir is None:
        raise FileNotFoundError(f"Could not read visible/LWIR pair for {visible_path}")

    return visible, lwir


def _extract_fused_image(fused_output):
    """Normalize fusion outputs that may be a single image or a tuple of images."""
    if isinstance(fused_output, tuple) or isinstance(fused_output, list):
        for item in fused_output:
            if isinstance(item, np.ndarray):
                if item.ndim == 3:
                    return item
        return fused_output[0]
    return fused_output


def _prepare_fused_for_metric(fused_output) -> np.ndarray | None:
    """Convert fusion output to a 3-channel image suitable for the metric."""
    fused_image = _extract_fused_image(fused_output)
    if not isinstance(fused_image, np.ndarray):
        return None

    if fused_image.ndim == 2:
        return np.repeat(fused_image[..., None], 3, axis=2)

    if fused_image.ndim != 3:
        return None

    channels = fused_image.shape[-1]
    if channels == 1:
        return np.repeat(fused_image, 3, axis=2)
    if channels == 2:
        ch3 = np.mean(fused_image, axis=2, keepdims=True)
        return np.concatenate([fused_image, ch3], axis=2)
    if channels == 3:
        return fused_image[..., :3]

    return None


def _split_fused_outputs(fused_output) -> list[tuple[str | None, np.ndarray]]:
    """Split a fusion return value into named image outputs when possible.

    Expected pattern for multi-output methods is a tuple/list containing images
    and optional trailing method labels, e.g. (img_a, img_b, 'a', 'b').
    """
    if isinstance(fused_output, (tuple, list)):
        images = [item for item in fused_output if isinstance(item, np.ndarray)]
        labels = [item for item in fused_output if isinstance(item, str)]
        if images and len(images) == len(labels):
            return list(zip(labels, images))
        if images:
            return [(None, images[0])]
    if isinstance(fused_output, np.ndarray):
        return [(None, fused_output)]
    return []


def _call_fusion_for_metric(method_name: str, fusion_function, visible_bgr: np.ndarray, lwir: np.ndarray):
    """Call fusion without decorator side effects when possible.

    In contribution review we only need in-memory outputs, so we prefer the
    undecorated function to avoid wrappers that can alter return format.
    """
    if hasattr(fusion_function, "__wrapped__"):
        return fusion_function.__wrapped__(visible_bgr, lwir)
    return fusion_function(visible_bgr, lwir)


# ---------------------------------------------------------------------------
# Raw result computation
# ---------------------------------------------------------------------------

def _compute_raw_method_results(
    method_name: str,
    fusion_function,
    visible_path: str,
    dataset: str,
    equalization: str = "no_equalization",
    selected_methods: set[str] | None = None,
) -> dict[str, dict]:
    """Compute one or more raw contribution results from a single fusion call."""
    visible_bgr, lwir = _load_bgr_lwir_pair(visible_path)
    if equalization in ("th_equalization", "rgb_equalization", "rgb_th_equalization"):
        from Dataset.th_equalization import th_equalization, rgb_equalization
        if equalization in ("th_equalization", "rgb_th_equalization"):
            lwir = th_equalization(lwir, "clahe")
        if equalization in ("rgb_equalization", "rgb_th_equalization"):
            visible_bgr = rgb_equalization(visible_bgr, "clahe")
    return _compute_raw_method_results_from_inputs(
        method_name=method_name,
        fusion_function=fusion_function,
        visible_bgr=visible_bgr,
        lwir=lwir,
        visible_path=visible_path,
        dataset=dataset,
        selected_methods=selected_methods,
    )


def _compute_raw_method_results_from_inputs(
    method_name: str,
    fusion_function,
    visible_bgr: np.ndarray,
    lwir: np.ndarray,
    visible_path: str,
    dataset: str,
    selected_methods: set[str] | None = None,
) -> dict[str, dict]:
    """Compute one or more raw contribution results from preloaded/equalized inputs."""
    t0 = time.perf_counter()
    fused_output = _call_fusion_for_metric(method_name, fusion_function, visible_bgr, lwir)
    fusion_time_ms = (time.perf_counter() - t0) * 1000.0
    outputs = _split_fused_outputs(fused_output)
    if not outputs:
        raise ValueError("fusion output is not a 2D/3D ndarray compatible with metric")

    raw_results: dict[str, dict] = {}
    for output_name, fused_image in outputs:
        output_method = output_name or method_name
        if selected_methods is not None and output_method not in selected_methods:
            continue
        prepared_fused = _prepare_fused_for_metric(fused_image)
        if prepared_fused is None:
            continue
        raw = compute_contribution_rgb(visible_bgr, lwir, prepared_fused, calibration=None)
        raw["visible_path"] = visible_path
        raw["condition"] = _infer_condition(visible_path, dataset)
        raw["source_method"] = method_name
        raw["output_method"] = output_method
        raw["fusion_time_ms"] = float(fusion_time_ms)
        raw_results[output_method] = raw

    return raw_results


def _primary_method_for(method_name: str) -> str:
    return _SHARED_MULTI_OUTPUT_PRIMARY.get(method_name, method_name)


def _group_methods_by_primary(method_names: list[str]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for method_name in method_names:
        primary = _primary_method_for(method_name)
        grouped.setdefault(primary, set()).add(method_name)
    return grouped


def _compute_raw_results_for_image_with_methods(
    visible_path: str,
    dataset: str,
    equalization: str,
    method_names: list[str],
    fusion_methods: dict,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Compute pending method outputs for one image loading/equalizing inputs once."""
    visible_bgr, lwir = _load_bgr_lwir_pair(visible_path)
    if equalization in ("th_equalization", "rgb_equalization", "rgb_th_equalization"):
        from Dataset.th_equalization import th_equalization, rgb_equalization
        if equalization in ("th_equalization", "rgb_th_equalization"):
            lwir = th_equalization(lwir, "clahe")
        if equalization in ("rgb_equalization", "rgb_th_equalization"):
            visible_bgr = rgb_equalization(visible_bgr, "clahe")

    grouped_methods = _group_methods_by_primary(method_names)
    raw_results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    for primary_method, requested_methods in grouped_methods.items():
        fusion_function = fusion_methods.get(primary_method)
        if fusion_function is None:
            for method_name in requested_methods:
                errors[method_name] = f"missing fusion function for primary method '{primary_method}'"
            continue
        try:
            outputs = _compute_raw_method_results_from_inputs(
                method_name=primary_method,
                fusion_function=fusion_function,
                visible_bgr=visible_bgr,
                lwir=lwir,
                visible_path=visible_path,
                dataset=dataset,
                selected_methods=requested_methods,
            )
        except Exception as exc:
            for method_name in requested_methods:
                errors[method_name] = str(exc)
            continue

        for method_name in requested_methods:
            if method_name not in outputs:
                errors[method_name] = f"method output '{method_name}' not returned by '{primary_method}'"

        raw_results.update(outputs)

    return raw_results, errors


def _compute_raw_results_for_image_by_names(
    visible_path: str,
    dataset: str,
    equalization: str,
    method_names: list[str],
) -> tuple[str, dict[str, dict], dict[str, str]]:
    """Process-safe wrapper to compute pending methods for one image."""
    fusion_methods = _get_worker_methods()
    raw_results, errors = _compute_raw_results_for_image_with_methods(
        visible_path=visible_path,
        dataset=dataset,
        equalization=equalization,
        method_names=method_names,
        fusion_methods=fusion_methods,
    )
    return visible_path, raw_results, errors


def _compute_raw_results_for_image_specs_batch_by_names(
    specs: list[tuple[str, list[str]]],
    dataset: str,
    equalization: str,
) -> list[tuple[str, dict[str, dict], dict[str, str]]]:
    """Process-safe batch wrapper for image-first evaluation."""
    fusion_methods = _get_worker_methods()
    batch_results: list[tuple[str, dict[str, dict], dict[str, str]]] = []
    for visible_path, method_names in specs:
        try:
            raw_results, errors = _compute_raw_results_for_image_with_methods(
                visible_path=visible_path,
                dataset=dataset,
                equalization=equalization,
                method_names=method_names,
                fusion_methods=fusion_methods,
            )
        except Exception as exc:
            raw_results = {}
            errors = {method_name: str(exc) for method_name in method_names}
        batch_results.append((visible_path, raw_results, errors))
    return batch_results


def _compute_raw_results_for_image_specs_batch_with_methods(
    specs: list[tuple[str, list[str]]],
    dataset: str,
    equalization: str,
    fusion_methods: dict,
) -> list[tuple[str, dict[str, dict], dict[str, str]]]:
    """Thread-mode batch wrapper for image-first evaluation."""
    batch_results: list[tuple[str, dict[str, dict], dict[str, str]]] = []
    for visible_path, method_names in specs:
        try:
            raw_results, errors = _compute_raw_results_for_image_with_methods(
                visible_path=visible_path,
                dataset=dataset,
                equalization=equalization,
                method_names=method_names,
                fusion_methods=fusion_methods,
            )
        except Exception as exc:
            raw_results = {}
            errors = {method_name: str(exc) for method_name in method_names}
        batch_results.append((visible_path, raw_results, errors))
    return batch_results


def _compute_raw_method_result_by_name(method_name: str, visible_path: str, dataset: str, equalization: str = "no_equalization") -> dict:
    """Process-friendly wrapper that resolves fusion function by method name."""
    fusion_methods = _get_worker_methods()
    if method_name not in fusion_methods:
        raise KeyError(f"Unknown fusion method '{method_name}'")
    raw_results = _compute_raw_method_results(method_name, fusion_methods[method_name], visible_path, dataset, equalization=equalization, selected_methods={method_name})
    return raw_results[method_name]


def _compute_raw_method_result_batch_by_name(
    method_name: str,
    visible_paths: list[str],
    dataset: str,
    equalization: str = "no_equalization",
) -> list[tuple[str, str, dict]]:
    """Process-friendly batch wrapper for method evaluation."""
    fusion_methods = _get_worker_methods()
    if method_name not in fusion_methods:
        raise KeyError(f"Unknown fusion method '{method_name}'")

    fusion_function = fusion_methods[method_name]
    outputs = []
    for visible_path in visible_paths:
        raw_results = _compute_raw_method_results(method_name, fusion_function, visible_path, dataset, equalization=equalization)
        for output_method, raw in raw_results.items():
            outputs.append((visible_path, output_method, raw))
    return outputs


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------


def _apply_batch_results(
    batch_results,
    *,
    cache_data: dict,
    method_results_map: dict,
    skipped_pairs: dict,
    method_fps: dict,
    equalization: str,
    calibration,
) -> None:
    """Fold one batch of per-image results into cache, results map, and skip log.

    ``batch_results`` is an iterable of ``(visible_path, raw_results, errors)``
    triples.  The sequential path adapts its single-image output to this shape
    with a one-element list, so all three execution modes share this helper.
    """
    for visible_path, raw_results, errors in batch_results:
        for method_name, reason in errors.items():
            skipped_pairs[method_name].append((visible_path, str(reason)))
        for output_method, raw in raw_results.items():
            output_fp = method_fps.get(output_method)
            if output_fp is None:
                continue
            cache_data[_cache_key(output_method, output_fp, visible_path, equalization)] = dict(raw)
            method_results_map[output_method].append(_with_calibration(raw, calibration))


def _drain_futures_with_checkpoints(
    future_map: dict,
    future_desc: str,
    checkpoint_every_batches: int,
    *,
    cache_file,
    cache_data: dict,
    method_results_map: dict,
    skipped_pairs: dict,
    method_fps: dict,
    equalization: str,
    calibration,
) -> None:
    """Consume a ``{future: batch_spec}`` map with progress, checkpoints and
    crash-safe cache flushing.  Used by both process- and thread-pool paths.
    """
    batches_since_checkpoint = 0
    try:
        for future in tqdm(as_completed(future_map), total=len(future_map),
                           desc=future_desc, leave=False):
            try:
                batch_results = future.result()
            except Exception as exc:
                # Whole batch failed: record per-pair, keep going.
                log(f"[warn] batch failed ({exc}); continuing with remaining batches")
                for visible_path, missing_methods in future_map[future]:
                    for method_name in missing_methods:
                        skipped_pairs[method_name].append((visible_path, f"batch failed: {exc}"))
            else:
                _apply_batch_results(
                    batch_results,
                    cache_data=cache_data,
                    method_results_map=method_results_map,
                    skipped_pairs=skipped_pairs,
                    method_fps=method_fps,
                    equalization=equalization,
                    calibration=calibration,
                )
            batches_since_checkpoint += 1
            if batches_since_checkpoint >= checkpoint_every_batches:
                _save_cache(cache_file, cache_data)
                batches_since_checkpoint = 0
    except BaseException:
        log("[warn] evaluation interrupted; flushing cache before re-raise")
        _save_cache(cache_file, cache_data)
        raise


def evaluate_fusion_methods_on_dataset(
    dataset_root: str,
    dataset: str,
    fusion_methods: dict,
    visible_paths_override: list[str] | None = None,
    condition: str = "all",
    max_images: int | None = None,
    calibration_images: int | None = None,
    calibration_ratio: float = 0.2,
    split_seed: int = 42,
    workers: int = 1,
    execution_mode: str = "thread",
    task_chunksize: int = 1,
    calibration_samples_file: str | None = None,
    cache_file: str | None = None,
    reset_cache: bool = False,
    calibration: dict | None = None,
    equalization: str = "no_equalization",
) -> dict:
    """Evaluate multiple fusion methods and aggregate contribution metrics per method."""
    if visible_paths_override is None:
        visible_paths = []
        for visible_path in _iter_visible_images(dataset_root, dataset):
            if condition != "all" and _infer_condition(visible_path, dataset) != condition:
                continue
            visible_paths.append(visible_path)
    else:
        visible_paths = list(visible_paths_override)

    if max_images is not None:
        visible_paths = visible_paths[:max_images]

    calibration_paths, evaluation_paths = _split_calibration_eval(
        visible_paths,
        calibration_ratio=calibration_ratio,
        calibration_images=calibration_images,
        seed=split_seed,
    )

    if calibration is None and calibration_paths:
        calibration = build_contribution_calibration_from_dataset(
            calibration_paths,
            workers=workers,
            execution_mode=execution_mode,
            task_chunksize=task_chunksize,
            calibration_samples_file=calibration_samples_file,
            equalization=equalization,
        )

    cache_data = {} if reset_cache else _load_cache(cache_file)
    per_method_results = {}
    max_workers = max(1, int(workers))
    method_fps = {method_name: _method_fingerprint(method_name, fusion_function) for method_name, fusion_function in fusion_methods.items()}
    method_names = list(fusion_methods.keys())
    # Per-pair failure log: method -> [(visible_path, reason), ...].
    # Failures are scoped to a single (method, image); they do NOT disqualify
    # the method globally, so other successful images still get cached and used.
    skipped_pairs: dict[str, list[tuple[str, str]]] = {m: [] for m in method_names}
    # How many batches between cache checkpoints (bounds worst-case lost work).
    checkpoint_every_batches = 10
    method_results_map: dict[str, list[dict]] = {method_name: [] for method_name in method_names}
    method_initial_cache_hits: dict[str, int] = {method_name: 0 for method_name in method_names}
    pending_by_image: dict[str, list[str]] = {}

    logCoolMessage("Evaluation setup")
    log(f"Methods: {len(fusion_methods)} | evaluation images: {len(evaluation_paths)} | calibration images: {len(calibration_paths)}")
    log(f"Equalization: {equalization}")
    log(f"Execution mode: {execution_mode} | workers: {max_workers} | chunk size: {task_chunksize}")
    log(f"Cache file: {cache_file if cache_file else 'disabled'}")
    log("")

    # Cache pass: reuse available results and track what is still pending per image.
    cache_hits_total = 0
    for visible_path in evaluation_paths:
        missing_methods = []
        for method_name in method_names:
            method_fp = method_fps[method_name]
            cache_key = _cache_key(method_name, method_fp, visible_path, equalization)
            cached = cache_data.get(cache_key)
            if cached is None:
                missing_methods.append(method_name)
            else:
                method_results_map[method_name].append(_with_calibration(dict(cached), calibration))
                method_initial_cache_hits[method_name] += 1
                cache_hits_total += 1
        if missing_methods:
            pending_by_image[visible_path] = missing_methods

    log(f"Cache warm-start: hits={cache_hits_total} pending_pairs={sum(len(v) for v in pending_by_image.values())}")

    # Compute pending method/image pairs with image-first execution.
    if pending_by_image:
        pending_images_count = len(pending_by_image)
        pending_pairs_count = sum(len(missing) for missing in pending_by_image.values())
        avg_pending_methods = pending_pairs_count / max(1, pending_images_count)
        log(
            f"Evaluation workload: pending_images={pending_images_count} "
            f"pending_pairs={pending_pairs_count} avg_methods_per_image={avg_pending_methods:.2f}"
        )
        image_tqdm_desc = (
            f"Evaluation images ({len(pending_by_image)} images, {len(method_names)} methods, eq={equalization})"
        )
        if max_workers == 1:
            # Sequential path: one image at a time, no pool.  Adapt each
            # single-image output to the (visible_path, raw, errors) tuple
            # shape so we reuse the same result-folding helper as the
            # parallel paths.
            iterator = tqdm(pending_by_image.items(), total=len(pending_by_image),
                            desc=image_tqdm_desc, leave=False)
            processed_since_checkpoint = 0
            try:
                for visible_path, missing_methods in iterator:
                    try:
                        raw_results, errors = _compute_raw_results_for_image_with_methods(
                            visible_path=visible_path,
                            dataset=dataset,
                            equalization=equalization,
                            method_names=missing_methods,
                            fusion_methods=fusion_methods,
                        )
                    except Exception as exc:
                        # Image-level failure (corrupt file, unreadable pair):
                        # record per-pair and keep going — other images still run.
                        log(f"[warn] image skipped entirely: {visible_path} ({exc})")
                        for method_name in missing_methods:
                            skipped_pairs[method_name].append((visible_path, f"image load failed: {exc}"))
                        continue
                    _apply_batch_results(
                        [(visible_path, raw_results, errors)],
                        cache_data=cache_data,
                        method_results_map=method_results_map,
                        skipped_pairs=skipped_pairs,
                        method_fps=method_fps,
                        equalization=equalization,
                        calibration=calibration,
                    )
                    processed_since_checkpoint += 1
                    if processed_since_checkpoint >= checkpoint_every_batches:
                        _save_cache(cache_file, cache_data)
                        processed_since_checkpoint = 0
            except BaseException:
                log("[warn] evaluation interrupted; flushing cache before re-raise")
                _save_cache(cache_file, cache_data)
                raise
        else:
            # Parallel path: process pool vs thread pool differ only in the
            # executor class and the worker function.  Everything downstream
            # (future draining, checkpointing, error handling) is shared via
            # _drain_futures_with_checkpoints.
            pending_specs = list(pending_by_image.items())
            spec_batches = _batched(pending_specs, task_chunksize)
            log(f"Evaluation batching: batch_size={max(1, int(task_chunksize))} total_batches={len(spec_batches)}")

            if execution_mode == "process":
                executor_cls = ProcessPoolExecutor

                def _submit(executor, batch):
                    return executor.submit(
                        _compute_raw_results_for_image_specs_batch_by_names,
                        batch, dataset, equalization,
                    )
            else:
                executor_cls = ThreadPoolExecutor

                def _submit(executor, batch):
                    return executor.submit(
                        _compute_raw_results_for_image_specs_batch_with_methods,
                        batch, dataset, equalization, fusion_methods,
                    )

            with executor_cls(max_workers=max_workers) as executor:
                future_map = {_submit(executor, batch): batch for batch in spec_batches}
                future_desc = (
                    f"Evaluation futures ({len(future_map)} batches, "
                    f"{len(method_names)} methods, eq={equalization}, mode={execution_mode})"
                )
                _drain_futures_with_checkpoints(
                    future_map, future_desc, checkpoint_every_batches,
                    cache_file=cache_file,
                    cache_data=cache_data,
                    method_results_map=method_results_map,
                    skipped_pairs=skipped_pairs,
                    method_fps=method_fps,
                    equalization=equalization,
                    calibration=calibration,
                )

    # Persist progress once after the image-major pass.
    _save_cache(cache_file, cache_data)

    total_images = len(evaluation_paths)
    # Build a summary of methods that failed on every image (totally skipped),
    # preserved under the legacy "skipped_methods" key for downstream consumers.
    skipped_methods: dict[str, str] = {}
    for method_name in method_names:
        method_results = method_results_map[method_name]
        n_skipped_pairs = len(skipped_pairs.get(method_name, []))
        if not method_results:
            sample = skipped_pairs[method_name][0][1] if skipped_pairs.get(method_name) else "no results"
            reason = f"no successful evaluations ({n_skipped_pairs} images failed); sample: {sample}"
            skipped_methods[method_name] = reason
            log(f"[{method_name}] skipped: {reason}")
            continue
        summary = aggregate_contribution_results(method_results)

        # Correct latent_z to avoid Jensen's inequality distortion.
        # Per-image calibration + averaging (mean(f(x))) can invert the ranking
        # of methods with different variance, because the calibration curve is
        # non-linear. The principled summary value is f(mean(x)): calibrate the
        # aggregated mean cont_vis, not average the per-image calibrated values.
        if calibration is not None:
            for src_key, z_key in [
                ("cont_vis_weighted", "latent_z"),
                ("cont_vis_weighted", "latent_z_weighted"),
                ("cont_vis_raw", "latent_z_raw"),
            ]:
                src_stats = summary.get(src_key)
                if src_stats and "mean" in src_stats:
                    z_from_mean = float(apply_contribution_calibration(src_stats["mean"], calibration))
                    if z_key in summary:
                        summary[z_key]["mean"] = z_from_mean
                    else:
                        summary[z_key] = {"mean": z_from_mean, "median": z_from_mean, "std": 0.0, "n": src_stats["n"]}
                    # Also update derived calibrated fields
                    if z_key == "latent_z":
                        cont_lw_cal = 100.0 * z_from_mean
                        if "cont_lw_calibrated" in summary:
                            summary["cont_lw_calibrated"]["mean"] = cont_lw_cal
                        if "cont_vis_calibrated" in summary:
                            summary["cont_vis_calibrated"]["mean"] = 100.0 - cont_lw_cal

        per_method_results[method_name] = {
            "n_images": len(method_results),
            "summary": summary,
            "results": method_results,
            "equalization": equalization,
        }
        cache_hits = method_initial_cache_hits[method_name]
        computed = max(0, len(method_results) - cache_hits)
        extra = f" skipped_pairs={n_skipped_pairs}" if n_skipped_pairs else ""
        log(f"[{method_name}] done: total={total_images} cache_hits={cache_hits} computed={computed}{extra}")

    return {
        "dataset": dataset,
        "condition": condition,
        "n_calibration": len(calibration_paths),
        "n_images": len(evaluation_paths),
        "calibration": calibration,
        "equalization": equalization,
        "methods": per_method_results,
        "skipped_methods": skipped_methods,
        "skipped_pairs": skipped_pairs,
    }
