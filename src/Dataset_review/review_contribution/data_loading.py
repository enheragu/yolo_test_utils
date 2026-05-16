from __future__ import annotations

"""
Shared data-loading utilities for the review_contribution package.

Single source of truth for:
  - Default dataset roots (sourced from ``Dataset.constants``).
  - The visible/LWIR filesystem convention (``visible/ → lwir/``).
  - Directory iteration for KAIST and LLVIP YOLO-style trees.
  - Condition inference (day/night) from an image path.
  - Image pair loading in either BGR (OpenCV-native) or RGB (metric-ready).

Any other module in this package that needs to discover or read image pairs
must import from here instead of re-implementing path transforms or walks.
"""

import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np


_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

from Dataset.constants import kaist_images_path, llvip_images_path


# --- Dataset root resolution ------------------------------------------------
#
# Point at the ORIGINAL (unprocessed) datasets, not the YOLO-annotated copies.
# The generated trees are derived artifacts; contribution analysis must run on
# pristine visible/LWIR pairs so proxies and calibration are comparable across
# runs and independent of any preprocessing applied to the training split.

DEFAULT_DATASET_ROOTS: dict[str, str] = {
    "kaist": kaist_images_path,      # /home/.../kaist-cvpr15/images
    "llvip": llvip_images_path,      # /home/.../LLVIP
}


def default_dataset_root(dataset: str) -> str:
    try:
        return DEFAULT_DATASET_ROOTS[dataset]
    except KeyError:
        raise ValueError(f"Unsupported dataset '{dataset}'") from None


def resolve_dataset_root(dataset: str, override: str | None = None) -> str:
    """Return the effective dataset root, honoring an explicit override."""
    if override:
        return override
    return default_dataset_root(dataset)


# --- Visible ↔ LWIR path convention -----------------------------------------

def visible_to_lwir_path(visible_path: str) -> str:
    """Translate a visible image path to its paired LWIR path.

    Both KAIST and LLVIP YOLO-style trees mirror ``.../visible/images/...``
    as ``.../lwir/images/...``.  All callers must route through this
    function so the convention lives in one place.
    """
    return visible_path.replace(os.sep + "visible" + os.sep,
                                os.sep + "lwir" + os.sep)


# --- KAIST day/night sets ---------------------------------------------------

KAIST_DAY_SETS = {"set00", "set01", "set02", "set06", "set07", "set08"}
KAIST_NIGHT_SETS = {"set03", "set04", "set05", "set09", "set10", "set11"}


def _infer_kaist_condition(image_path: str) -> str:
    for part in os.path.normpath(image_path).split(os.sep):
        if part in KAIST_DAY_SETS:
            return "day"
        if part in KAIST_NIGHT_SETS:
            return "night"
    return "unknown"


def infer_condition(image_path: str, dataset: str) -> str:
    """Infer the condition label (``day``/``night``) for *image_path*."""
    if dataset == "kaist":
        return _infer_kaist_condition(image_path)
    if dataset == "llvip":
        return "night"
    return "unknown"


# --- Visible image iteration ------------------------------------------------

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _iter_kaist_visible_images(dataset_root: str):
    # Original KAIST layout: <root>/setXX/VNNN/visible/I*.jpg
    for folder_set in sorted(os.listdir(dataset_root)):
        folder_path = os.path.join(dataset_root, folder_set)
        if not os.path.isdir(folder_path):
            continue
        for subfolder_set in sorted(os.listdir(folder_path)):
            visible_folder = os.path.join(folder_path, subfolder_set, "visible")
            if not os.path.isdir(visible_folder):
                continue
            for filename in sorted(os.listdir(visible_folder)):
                if filename.lower().endswith(_IMAGE_SUFFIXES):
                    yield os.path.join(visible_folder, filename)


def _iter_llvip_visible_images(dataset_root: str):
    # Original LLVIP layout: <root>/visible/{train,test}/*.jpg
    visible_root = os.path.join(dataset_root, "visible")
    if not os.path.isdir(visible_root):
        return
    for split in sorted(os.listdir(visible_root)):
        split_path = os.path.join(visible_root, split)
        if not os.path.isdir(split_path):
            continue
        for filename in sorted(os.listdir(split_path)):
            if filename.lower().endswith(_IMAGE_SUFFIXES):
                yield os.path.join(split_path, filename)


def iter_visible_images(dataset_root: str, dataset: str):
    """Dataset-aware iterator over visible image paths."""
    if dataset == "kaist":
        yield from _iter_kaist_visible_images(dataset_root)
    elif dataset == "llvip":
        yield from _iter_llvip_visible_images(dataset_root)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")


# --- Image loaders ----------------------------------------------------------

def load_bgr_lwir_pair(visible_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a visible/LWIR pair in OpenCV BGR order (for fusion functions)."""
    visible = cv.imread(visible_path, cv.IMREAD_COLOR)
    lwir_path = visible_to_lwir_path(visible_path)
    lwir = cv.imread(lwir_path, cv.IMREAD_GRAYSCALE)
    if visible is None or lwir is None:
        raise FileNotFoundError(f"Could not read visible/LWIR pair for {visible_path}")
    return visible, lwir


def load_rgb_lwir_pair(
    visible_path: str,
    equalization: str = "no_equalization",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a visible/LWIR pair with optional equalization; visible as RGB."""
    visible_bgr = cv.imread(visible_path, cv.IMREAD_COLOR)
    lwir_path = visible_to_lwir_path(visible_path)
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


__all__ = [
    "DEFAULT_DATASET_ROOTS",
    "KAIST_DAY_SETS",
    "KAIST_NIGHT_SETS",
    "default_dataset_root",
    "resolve_dataset_root",
    "visible_to_lwir_path",
    "iter_visible_images",
    "infer_condition",
    "load_bgr_lwir_pair",
    "load_rgb_lwir_pair",
]
