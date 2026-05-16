from __future__ import annotations

"""
Latent contribution scoring: metrics computation and calibration.

This module provides core functions for computing visible-to-LWIR contribution
scores via multiple proxy methods and for calibrating raw scores to a
reproducible latent axis via isotonic regression.

Proxy methods (6 total):
  - reg:      Per-channel NNLS regression
  - mi:       Per-channel unique mutual information
  - ssim:     Multichannel SSIM comparison
  - grad:     Combined gradient magnitude + orientation
  - spectral: Inter-channel independence structure
  - freq:     FFT magnitude spectrum correlation

Design decisions:
  - Per-channel computation (reg, mi): Each channel is compared independently
    (1D vs 1D) to avoid a structural bias where visible's 3 independent spectral
    bands would dominate over LWIR's replicated single band when all channels are
    concatenated into one vector. With flatten, NNLS and MI inherently assign more
    weight to visible because it spans a richer subspace — per-channel averaging
    puts both sources on equal footing.
  - Subsampled MI: Mutual information uses spatial subsampling (50k pixels) for
    efficiency. For 128-bin joint histograms, convergence analysis shows ~20k
    samples suffice for <1% estimation error vs full-image computation. We use
    50k as comfortable margin with a fixed seed for reproducibility.
  - Combined gradient: Magnitude and orientation gradient proxies are merged into
    a single composite (50/50 by default) to avoid over-weighting structural
    texture in the aggregate. As separate proxies they contributed 2/5 = 40% of
    the score, biasing toward whichever modality had richer fine-grained texture
    (typically visible). As one proxy they contribute 1/6 alongside the others.
  - Spectral independence: Measures inter-channel correlation structure of the
    fused output compared to visible (diverse channels) vs LWIR (identical
    channels). This captures spectral-band information preservation regardless of
    whether the output is true-color or false-color (e.g. PCA/FA descriptors).
    It does not assume perceptual color — it measures whether fused channels carry
    independent information (like visible's 3 spectral bands at ~470/530/620nm)
    or redundant information (like LWIR's single ~8-14um band replicated).
    Permutation-invariant by construction (pairwise correlations are a set, not
    ordered).
  - Frequency proxy: Compares FFT magnitude spectra of fused vs each source.
    Complementary to spatial-domain proxies: LWIR typically has more low-frequency
    energy while visible has richer high-frequency content. Permutation-invariant
    (operates on channel-average grayscale).
  - Permutation pre-screening: Only the top 2 channel permutations (by sum of
    per-channel correlation with visible) are evaluated with the full metric suite.
    This reduces computation from 6x to ~2x while preserving channel-order
    robustness for descriptor-based methods. Permutation-invariant proxies (grad,
    spectral, freq) are computed once, outside the permutation loop.
"""

import numpy as np
import cv2 as cv
from itertools import permutations
from scipy.optimize import nnls
from skimage.metrics import structural_similarity as skimage_ssim


# Bump when contribution logic changes so caches can be invalidated safely.
# --- Cache versioning ------------------------------------------------------
# Three independent version tags control three independent caches.  Bump the
# narrowest one that applies to avoid unnecessary recomputation.
#
#   PROXY_VERSION
#     Governs the evaluation cache (~23MB: proxies raw per method/image).
#     Bump when: `compute_contribution_rgb`, any proxy formula (reg/mi/ssim/
#     grad/spectral/freq), or the shape of per-image proxy output changes.
#     Bumping this also invalidates everything downstream (samples, fit).
#
#   CALIBRATION_SAMPLES_VERSION
#     Governs the calibration-samples cache (synthetic mixture proxies).
#     Bump when: synthetic generators (`_build_freq_blend_fused`,
#     `_build_nonlinear_fused`, `_build_channel_concat_fused`,
#     `_CHANNEL_CONCAT_PATTERNS`) change.  Does NOT invalidate evaluation.
#
#   CALIBRATION_FIT_VERSION
#     Governs the calibration-object cache (PAVA curves + IVW weights).
#     Bump when: `fit_contribution_calibration`, `_cap_proxy_weights`,
#     `MAX_PROXY_WEIGHT`, `_average_proxy_weights_across_formats`, or
#     `_select_best_fit_calibration` change.  Does NOT invalidate samples
#     or evaluation.
PROXY_VERSION = "v18_cr_reliability_weighted"
CALIBRATION_SAMPLES_VERSION = "s2_freq_maxabs"
CALIBRATION_FIT_VERSION = "f2_multi_format_avg_fix"

# Full set of proxy keys that ``compute_contribution_rgb`` emits.  This is the
# *cache schema* — every per-image cache stores all six values regardless of
# which subset is currently enabled for aggregation.  Disabling a proxy via
# ``settings.ENABLED_PROXIES`` only changes calibration / aggregation, not
# the per-image cache (so re-enabling it later doesn't require recomputing).
_CALIBRATION_PROXY_KEYS = (
    "cont_vis_reg", "cont_vis_mi", "cont_vis_ssim",
    "cont_vis_grad_combined", "cont_vis_spectral", "cont_vis_freq",
)


def _active_proxy_keys() -> tuple[str, ...]:
    """Subset of ``_CALIBRATION_PROXY_KEYS`` enabled for calibration/aggregation.

    Reads from :data:`settings.ENABLED_PROXIES` at *call time*, so a single
    process can pick up changes when settings.py is re-imported (e.g. tests).
    Order follows ``_CALIBRATION_PROXY_KEYS`` for reproducibility.  Unknown
    keys in ``ENABLED_PROXIES`` are silently dropped — the cache schema is
    authoritative.
    """
    from .settings import ENABLED_PROXIES
    enabled = set(ENABLED_PROXIES)
    return tuple(k for k in _CALIBRATION_PROXY_KEYS if k in enabled)


def proxy_set_signature() -> str:
    """Stable short hash of the active proxy set for cache keying.

    Hashing the active subset rather than the full schema means flipping a
    proxy on/off auto-invalidates the calibration cache without manual
    ``CALIBRATION_FIT_VERSION`` bumps.  6 hex chars is plenty for a 6-element
    universe (collision-free in practice).
    """
    import hashlib
    payload = ",".join(_active_proxy_keys()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:6]

# Maximum normalized weight any single proxy can hold in IVW combination.
# Prevents a low-σ proxy (e.g. cont_vis_reg) from monopolising the aggregate
# and masking informative signals from the remaining proxies.
# With 6 proxies, equal weight = 16.7%; cap at 0.25 allows ≈1.5× advantage.
MAX_PROXY_WEIGHT = 0.25

# Channel-redundancy uses multiple components combined by per-image reliability
# (same philosophy as IVW for contribution proxies, but without synthetic-alpha
# calibration because no monotonic ground-truth target exists for redundancy).
_CR_COMPONENT_PRIORS = {
    "pearson_global": 0.45,
    "spearman_global": 0.20,
    "pearson_tiles": 0.20,
    "effective_rank_redundancy": 0.15,
}
_SPEARMAN_MAX_SAMPLES = 25_000


def _cap_proxy_weights(weights: np.ndarray, cap: float = MAX_PROXY_WEIGHT) -> np.ndarray:
    """Clip normalized proxy weights so no single proxy exceeds *cap*.

    Excess weight is redistributed proportionally among uncapped proxies.
    Iterates until no proxy exceeds the cap (handles cascading overflow when
    redistribution pushes another proxy past the cap).
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    w /= s
    for _ in range(len(w)):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        under_sum = w[under].sum()
        if under_sum > 0:
            w[under] += w[under] / under_sum * excess
        else:
            break
    # Final normalization for numerical safety.
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _mutual_information(im1: np.ndarray, im2: np.ndarray, bins: int = 128,
                        max_samples: int = 50_000) -> float:
    """
    Mutual information between two arrays (normalised to [0, 1]).

    Uses spatial subsampling for efficiency: with 128 bins, the joint histogram
    has 128x128 = 16384 cells, and ~20k samples are sufficient for <1%
    estimation error compared to full-image computation (see Kraskov et al. for
    convergence bounds on plug-in MI estimators). We use 50k samples as a
    comfortable margin. Subsampling is deterministic (seed=42) for
    reproducibility across runs.
    """
    a = im1.ravel()
    b = im2.ravel()
    if a.size > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(a.size, max_samples, replace=False)
        a = a[idx]
        b = b[idx]
    h, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])
    h = h + 1e-10  # Laplace smoothing avoids log(0)
    pxy = h / h.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mi = np.sum(pxy * np.log(pxy / (px * py)))
    return float(np.clip(mi, 0.0, None))


def _normalise(img: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns 0.5 constant if range is zero."""
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return np.full_like(img, 0.5, dtype=np.float64)
    return (img - lo) / (hi - lo)


def _align_shapes(*arrays: np.ndarray) -> tuple:
    """Crop all arrays to the minimum common (H, W)."""
    h = min(a.shape[0] for a in arrays)
    w = min(a.shape[1] for a in arrays)
    return tuple(a[:h, :w] for a in arrays)


def _to_color_for_metric(image: np.ndarray) -> np.ndarray:
    """Convert an image to a 3-channel float64 array for channel-aware scoring."""
    if image.ndim == 2:
        return np.repeat(image.astype(np.float64)[..., None], 3, axis=2)
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return np.repeat(image.astype(np.float64), 3, axis=2)
        return image.astype(np.float64)[..., :3]
    raise ValueError(f"Unsupported image shape for color conversion: {image.shape}")


def _normalise_color(img: np.ndarray) -> np.ndarray:
    """Normalise each channel independently to [0, 1]."""
    if img.ndim != 3:
        raise ValueError(f"Expected a 3-channel image, got {img.shape}")
    return np.stack([_normalise(img[..., idx]) for idx in range(img.shape[2])], axis=2)


def _channelwise_average(img: np.ndarray) -> np.ndarray:
    """Collapse a colour image to grayscale via channel mean."""
    if img.ndim == 2:
        return img
    return img.mean(axis=2)


def _channel_permutations(img: np.ndarray) -> list[np.ndarray]:
    """Return all channel-order permutations for a 3-channel image."""
    if img.ndim != 3 or img.shape[2] != 3:
        return [img]
    return [img[..., perm] for perm in permutations((0, 1, 2))]


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that degrades gracefully for near-constant signals."""
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    if np.std(a_flat) < 1e-12 or np.std(b_flat) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(a_flat, b_flat)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; robust to outliers and non-linear monotonic
    relationships.  Reported only as a diagnostic — proxy aggregation still
    uses Pearson where linearity is assumed by construction (reg/NNLS) or
    where signals have been isotonically calibrated upstream."""
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    if a_flat.size < 2 or np.std(a_flat) < 1e-12 or np.std(b_flat) < 1e-12:
        return 0.0
    # np.argsort-based rank (average ranks for ties are ignored for speed —
    # acceptable for a diagnostic, and images rarely have exact ties anyway).
    ranks_a = np.argsort(np.argsort(a_flat)).astype(np.float64)
    ranks_b = np.argsort(np.argsort(b_flat)).astype(np.float64)
    corr = float(np.corrcoef(ranks_a, ranks_b)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def _safe_spearman_sampled(a: np.ndarray, b: np.ndarray,
                           max_samples: int = _SPEARMAN_MAX_SAMPLES) -> float:
    """Sampled Spearman rank correlation for large vectors.

    Keeps compute bounded for high-resolution images while preserving robust
    monotonic-correlation diagnostics.
    """
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    if a_flat.size == 0:
        return 0.0
    if a_flat.size > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(a_flat.size, size=max_samples, replace=False)
        a_flat = a_flat[idx]
        b_flat = b_flat[idx]
    return _safe_spearman(a_flat, b_flat)


def _normalise_positive_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize non-negative weights; fallback to uniform if sum is zero."""
    keys = list(weights.keys())
    vals = np.array([max(0.0, float(weights[k])) for k in keys], dtype=np.float64)
    s = float(np.sum(vals))
    if s <= 1e-12:
        vals = np.ones_like(vals) / float(len(vals))
    else:
        vals = vals / s
    return {k: float(v) for k, v in zip(keys, vals)}


def _effective_rank_redundancy(img: np.ndarray) -> float:
    """Map channel covariance effective rank to a redundancy score in [0, 1].

    For a 3-channel image, rank_eff ~= 1 implies duplicated channels
    (high redundancy), while rank_eff ~= 3 implies more independent channels.
    """
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    X = arr.reshape(-1, 3).astype(np.float64)
    X -= np.mean(X, axis=0, keepdims=True)
    cov = np.cov(X, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(evals, 0.0, None)
    tr = float(np.sum(evals))
    if tr <= 1e-12:
        return 1.0
    p = evals / tr
    p = p[p > 1e-12]
    if p.size == 0:
        return 1.0
    entropy = -float(np.sum(p * np.log(p)))
    r_eff = float(np.exp(entropy))
    # r_eff in [1, 3] -> redundancy in [1, 0]
    red = (3.0 - r_eff) / 2.0
    return float(np.clip(red, 0.0, 1.0))


def _channel_redundancy_components(img: np.ndarray, tile_grid: int = 4) -> dict[str, float]:
    """Compute channel-redundancy components and reliability-weighted score.

    Components:
      - pearson_global: mean |Pearson| across channel pairs (legacy v1)
      - spearman_global: mean |Spearman| across channel pairs
      - pearson_tiles: robust spatial median of local mean |Pearson|
            - effective_rank_redundancy: covariance effective-rank mapping

        Combination:
            effective_weight_k ∝ prior_k * reliability_k
            channel_redundancy = sum_k effective_weight_k * component_k
    """
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return {
            "pearson_global": 0.0,
            "spearman_global": 0.0,
            "pearson_tiles": 0.0,
            "effective_rank_redundancy": 0.0,
            "channel_redundancy": 0.0,
            "w_pearson_global": 0.0,
            "w_spearman_global": 0.0,
            "w_pearson_tiles": 0.0,
            "w_effective_rank_redundancy": 0.0,
        }

    c0 = arr[..., 0].astype(np.float64)
    c1 = arr[..., 1].astype(np.float64)
    c2 = arr[..., 2].astype(np.float64)
    pairs = ((c0, c1), (c1, c2), (c0, c2))

    pearson_pair_vals = [abs(_safe_corrcoef(a, b)) for a, b in pairs]
    pearson_global = float(np.mean(pearson_pair_vals))
    spearman_pair_vals = [abs(_safe_spearman_sampled(a, b)) for a, b in pairs]
    spearman_global = float(np.mean(spearman_pair_vals))

    h, w = c0.shape
    gh = max(1, tile_grid)
    gw = max(1, tile_grid)
    tile_scores = []
    for yi in range(gh):
        y0 = int(round(yi * h / gh))
        y1 = int(round((yi + 1) * h / gh))
        if y1 <= y0:
            continue
        for xi in range(gw):
            x0 = int(round(xi * w / gw))
            x1 = int(round((xi + 1) * w / gw))
            if x1 <= x0:
                continue
            vals = []
            for a, b in pairs:
                vals.append(abs(_safe_corrcoef(a[y0:y1, x0:x1], b[y0:y1, x0:x1])))
            tile_scores.append(float(np.mean(vals)))
    pearson_tiles = float(np.median(tile_scores)) if tile_scores else pearson_global

    eff_rank_red = _effective_rank_redundancy(arr)

    # Reliability terms in [0,1]. Lower dispersion => higher trust.
    rel_pearson = float(np.clip(1.0 - np.std(pearson_pair_vals) / 0.25, 0.15, 1.0))
    rel_spearman = float(np.clip(1.0 - np.std(spearman_pair_vals) / 0.25, 0.15, 1.0))
    if tile_scores:
        rel_tiles = float(np.clip(1.0 - np.std(tile_scores) / 0.25, 0.15, 1.0))
    else:
        rel_tiles = rel_pearson
    # Effective-rank reliability drops if covariance is near-degenerate.
    X = arr.reshape(-1, 3).astype(np.float64)
    X -= np.mean(X, axis=0, keepdims=True)
    cov = np.cov(X, rowvar=False)
    evals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    rel_effrank = float(np.clip(np.sum(evals > 1e-10) / 3.0, 0.2, 1.0))

    raw_weights = {
        "pearson_global": _CR_COMPONENT_PRIORS["pearson_global"] * rel_pearson,
        "spearman_global": _CR_COMPONENT_PRIORS["spearman_global"] * rel_spearman,
        "pearson_tiles": _CR_COMPONENT_PRIORS["pearson_tiles"] * rel_tiles,
        "effective_rank_redundancy": _CR_COMPONENT_PRIORS["effective_rank_redundancy"] * rel_effrank,
    }
    norm_weights = _normalise_positive_weights(raw_weights)
    score = (
        norm_weights["pearson_global"] * pearson_global
        + norm_weights["spearman_global"] * spearman_global
        + norm_weights["pearson_tiles"] * pearson_tiles
        + norm_weights["effective_rank_redundancy"] * eff_rank_red
    )

    return {
        "pearson_global": float(np.clip(pearson_global, 0.0, 1.0)),
        "spearman_global": float(np.clip(spearman_global, 0.0, 1.0)),
        "pearson_tiles": float(np.clip(pearson_tiles, 0.0, 1.0)),
        "effective_rank_redundancy": float(np.clip(eff_rank_red, 0.0, 1.0)),
        "channel_redundancy": float(np.clip(score, 0.0, 1.0)),
        "w_pearson_global": norm_weights["pearson_global"],
        "w_spearman_global": norm_weights["spearman_global"],
        "w_pearson_tiles": norm_weights["pearson_tiles"],
        "w_effective_rank_redundancy": norm_weights["effective_rank_redundancy"],
    }


def _channel_redundancy(img: np.ndarray) -> float:
    """Mean absolute Pearson correlation between the three channels of a
    pseudo-RGB image.  Interpretation: 1.0 means channels are duplicates
    (e.g. R=G=B, as in a fusion that replicates thermal three times); ~0.8
    is typical for natural RGB (channels structurally distinct but highly
    correlated); lower values indicate channels that carry more distinct
    information.  Permutation-invariant, so computed once outside the
    permutation loop.

    Returns 0.0 for non-3-channel inputs (nothing to correlate)."""
    return _channel_redundancy_components(img)["channel_redundancy"]


def _gradient_maps(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel gradient maps: gx, gy, magnitude and orientation."""
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    return gx, gy, mag, ang


def _map_delta_with_anchors(delta: float, visible_anchor: float, thermal_anchor: float) -> float:
    """Map a proxy delta to [0,100] using per-pair visible/lwir anchor endpoints."""
    denom = visible_anchor - thermal_anchor
    if abs(denom) < 1e-8:
        return 50.0
    mapped = 100.0 * (delta - thermal_anchor) / denom
    return float(np.clip(mapped, 0.0, 100.0))


def _ssim_multichannel(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM score between two images, supporting multichannel via channel_axis.

    Uses skimage's channel_axis parameter when available (scikit-image >= 0.19)
    for efficient multichannel computation. Falls back to per-channel averaging
    on older versions.
    """
    if img1.ndim == 3 and img2.ndim == 3:
        try:
            return float(skimage_ssim(img1, img2, data_range=1.0, channel_axis=2))
        except TypeError:
            return float(np.mean([
                skimage_ssim(img1[..., c], img2[..., c], data_range=1.0)
                for c in range(img1.shape[2])
            ]))
    return float(skimage_ssim(img1, img2, data_range=1.0))


def _build_gradient_source_cache(v: np.ndarray, lw: np.ndarray) -> dict:
    """Precompute gradient terms that depend only on visible/LWIR sources.

    These anchor values are reused for every fused image evaluated against the
    same source pair, avoiding redundant Sobel + correlation work.
    """
    _, _, mag_v, ang_v = _gradient_maps(v)
    _, _, mag_lw, ang_lw = _gradient_maps(lw)

    def _ori_agreement(ang_src: np.ndarray, ang_ref: np.ndarray, weight_map: np.ndarray) -> float:
        delta = ang_src - ang_ref
        agree_map = (np.cos(delta) + 1.0) * 0.5
        return float(np.sum(agree_map * weight_map) / (np.sum(weight_map) + 1e-12))

    weight_v = np.clip(mag_v / (np.mean(mag_v) + 1e-12), 0.0, 10.0)
    weight_lw = np.clip(mag_lw / (np.mean(mag_lw) + 1e-12), 0.0, 10.0)

    mag_delta_visible_anchor = _safe_corrcoef(mag_v, mag_v) - _safe_corrcoef(mag_lw, mag_v)
    mag_delta_thermal_anchor = _safe_corrcoef(mag_v, mag_lw) - _safe_corrcoef(mag_lw, mag_lw)
    ori_delta_visible_anchor = _ori_agreement(ang_v, ang_v, weight_v) - _ori_agreement(ang_lw, ang_v, weight_v)
    ori_delta_thermal_anchor = _ori_agreement(ang_v, ang_lw, weight_lw) - _ori_agreement(ang_lw, ang_lw, weight_lw)

    return {
        "mag_v": mag_v,
        "mag_lw": mag_lw,
        "ang_v": ang_v,
        "ang_lw": ang_lw,
        "mag_delta_visible_anchor": mag_delta_visible_anchor,
        "mag_delta_thermal_anchor": mag_delta_thermal_anchor,
        "ori_delta_visible_anchor": ori_delta_visible_anchor,
        "ori_delta_thermal_anchor": ori_delta_thermal_anchor,
        "ori_agreement_fn": _ori_agreement,
    }


# ---------------------------------------------------------------------------
# Proxy 1 — Non-negative Least Squares regression (per-channel)
# ---------------------------------------------------------------------------

def _contrib_regression(v: np.ndarray, lw: np.ndarray, f: np.ndarray) -> float:
    """NNLS regression on 1-D vectors: f ~ a*v + b*lw, returns 100*a/(a+b)."""
    A = np.stack([v.ravel(), lw.ravel()], axis=1)
    b_vec = f.ravel()
    try:
        coeffs, _ = nnls(A, b_vec, maxiter=max(100, 10 * A.shape[1]))
    except RuntimeError:
        coeffs, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
        coeffs = np.clip(coeffs, 0.0, None)
    total = coeffs[0] + coeffs[1]
    if total < 1e-8:
        return 50.0
    return float(100.0 * coeffs[0] / total)


def _contrib_regression_perchannel(v: np.ndarray, lw: np.ndarray, f: np.ndarray) -> float:
    """
    Per-channel NNLS regression, averaged across channels.

    Each channel is compared independently (1D vs 1D) so that visible and LWIR
    compete on equal footing. Without per-channel separation, the flattened
    visible vector (3 independent spectral bands) spans a richer subspace than
    the replicated LWIR vector, biasing NNLS toward higher visible coefficients
    regardless of actual content contribution.
    """
    if v.ndim == 2:
        return _contrib_regression(v.ravel(), lw.ravel(), f.ravel())
    scores = []
    for ch in range(v.shape[2]):
        scores.append(_contrib_regression(
            v[..., ch].ravel(), lw[..., ch].ravel(), f[..., ch].ravel()
        ))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Proxy 2 — Unique mutual information (per-channel)
# ---------------------------------------------------------------------------

def _contrib_unique_mi_from_scores(mi_vf: float, mi_lwf: float, corr_cross: float) -> float:
    """Unique-MI contribution from precomputed MI scores and cross-correlation."""
    unique_vis = max(mi_vf - mi_lwf * corr_cross, 0.0)
    unique_lw = max(mi_lwf - mi_vf * corr_cross, 0.0)
    total = unique_vis + unique_lw
    if total < 1e-8:
        return 50.0
    return float(100.0 * unique_vis / total)


def _contrib_unique_mi_perchannel(v: np.ndarray, lw: np.ndarray, f: np.ndarray) -> float:
    """
    Per-channel unique MI, averaged across channels.

    Same rationale as per-channel regression: comparing 1D vs 1D per channel
    eliminates the flatten bias where 3-channel visible would have structurally
    higher MI than single-band LWIR replicated to 3 identical channels. Each
    channel independently measures how much unique information the fused channel
    shares with visible vs thermal, then results are averaged.
    """
    if v.ndim == 2:
        mi_vf = _mutual_information(v, f)
        mi_lwf = _mutual_information(lw, f)
        corr_cross = abs(_safe_corrcoef(v.ravel(), lw.ravel()))
        return _contrib_unique_mi_from_scores(mi_vf, mi_lwf, corr_cross)
    scores = []
    for ch in range(v.shape[2]):
        v_ch = v[..., ch]
        lw_ch = lw[..., ch]
        f_ch = f[..., ch]
        mi_vf = _mutual_information(v_ch, f_ch)
        mi_lwf = _mutual_information(lw_ch, f_ch)
        corr_cross = abs(_safe_corrcoef(v_ch.ravel(), lw_ch.ravel()))
        scores.append(_contrib_unique_mi_from_scores(mi_vf, mi_lwf, corr_cross))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Proxy 3 — SSIM-based (multichannel)
# ---------------------------------------------------------------------------

def _contrib_ssim_from_scores(s_vis: float, s_lw: float) -> float:
    """SSIM contribution from precomputed visible/lwir SSIM values."""
    delta = s_vis - s_lw
    return float(100.0 / (1.0 + np.exp(-delta * 5.0)))


# ---------------------------------------------------------------------------
# Proxy 4 — Combined gradient (magnitude + orientation)
# ---------------------------------------------------------------------------

def _contrib_gradient_combined(v_gray: np.ndarray, lw_gray: np.ndarray,
                               f_gray: np.ndarray,
                               source_cache: dict | None = None,
                               mag_weight: float = 0.5) -> tuple[float, dict]:
    """
    Combined gradient proxy merging magnitude correlation and orientation
    agreement into a single score.

    Magnitude and orientation are merged (default 50/50 weight) to avoid
    over-weighting structural texture in the final proxy aggregation. When they
    were separate proxies they contributed 2/5 = 40% of the aggregate, biasing
    toward whichever modality had richer fine texture (typically visible).
    As a single proxy they contribute 1/N alongside the other proxies.

    Returns the combined score and a diagnostics dict with individual components.
    """
    if source_cache is None:
        source_cache = _build_gradient_source_cache(v_gray, lw_gray)

    mag_v = source_cache["mag_v"]
    mag_lw = source_cache["mag_lw"]
    ang_v = source_cache["ang_v"]
    ang_lw = source_cache["ang_lw"]
    _ori_agreement = source_cache["ori_agreement_fn"]

    _, _, mag_f, ang_f = _gradient_maps(f_gray)

    # Magnitude component: edge-strength pattern correlation
    grad_mag_corr_vis = _safe_corrcoef(mag_v, mag_f)
    grad_mag_corr_lw = _safe_corrcoef(mag_lw, mag_f)
    mag_delta = grad_mag_corr_vis - grad_mag_corr_lw
    cont_vis_grad_mag = _map_delta_with_anchors(
        mag_delta,
        visible_anchor=source_cache["mag_delta_visible_anchor"],
        thermal_anchor=source_cache["mag_delta_thermal_anchor"],
    )

    # Orientation component: local edge direction agreement
    weight_f = np.clip(mag_f / (np.mean(mag_f) + 1e-12), 0.0, 10.0)
    grad_ori_agree_vis = _ori_agreement(ang_v, ang_f, weight_f)
    grad_ori_agree_lw = _ori_agreement(ang_lw, ang_f, weight_f)
    ori_delta = grad_ori_agree_vis - grad_ori_agree_lw
    cont_vis_grad_ori = _map_delta_with_anchors(
        ori_delta,
        visible_anchor=source_cache["ori_delta_visible_anchor"],
        thermal_anchor=source_cache["ori_delta_thermal_anchor"],
    )

    combined = mag_weight * cont_vis_grad_mag + (1.0 - mag_weight) * cont_vis_grad_ori

    diagnostics = {
        "cont_vis_grad_mag": float(cont_vis_grad_mag),
        "cont_vis_grad_ori": float(cont_vis_grad_ori),
        "grad_mag_corr_vis": float(grad_mag_corr_vis),
        "grad_mag_corr_lw": float(grad_mag_corr_lw),
        "grad_ori_agree_vis": float(grad_ori_agree_vis),
        "grad_ori_agree_lw": float(grad_ori_agree_lw),
    }
    return float(combined), diagnostics


# ---------------------------------------------------------------------------
# Proxy 5 — Spectral independence (inter-channel structure)
# ---------------------------------------------------------------------------

def _contrib_spectral_independence(v: np.ndarray, lw: np.ndarray, f: np.ndarray) -> float:
    """
    Measures how much the fused image's inter-channel correlation structure
    resembles visible (diverse channels) vs LWIR (identical channels).

    This proxy captures spectral-band information preservation regardless of
    whether the output is true-color or false-color (e.g. PCA/FA descriptors).
    It does not assume perceptual color meaning: it measures whether the fused
    channels carry independent information — as visible's 3 spectral bands at
    approximately 470nm (B), 530nm (G), 620nm (R) do — or redundant information
    — as LWIR's single 8-14um band replicated to 3 channels does.

    The metric computes pairwise Pearson correlations between channels of each
    image and uses the Frobenius distance between correlation matrices. This is
    inherently permutation-invariant: channel reordering preserves the set of
    pairwise correlations, so no permutation search is needed.

    Returns cont_vis in [0, 100].
    """
    def _channel_corr_matrix(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return np.ones((1, 1), dtype=np.float64)
        n_ch = img.shape[2]
        flat = img.reshape(-1, n_ch).T  # (n_ch, n_pixels)
        stds = np.std(flat, axis=1)
        if np.any(stds < 1e-12):
            mat = np.eye(n_ch, dtype=np.float64)
            for i in range(n_ch):
                for j in range(i + 1, n_ch):
                    if stds[i] > 1e-12 and stds[j] > 1e-12:
                        c = float(np.corrcoef(flat[i], flat[j])[0, 1])
                        mat[i, j] = mat[j, i] = c if np.isfinite(c) else 0.0
            return mat
        mat = np.corrcoef(flat)
        mat = np.where(np.isfinite(mat), mat, 0.0)
        return mat

    corr_f = _channel_corr_matrix(f)
    corr_v = _channel_corr_matrix(v)
    corr_lw = _channel_corr_matrix(lw)

    dist_to_vis = np.linalg.norm(corr_f - corr_v, 'fro')
    dist_to_lw = np.linalg.norm(corr_f - corr_lw, 'fro')

    total = dist_to_vis + dist_to_lw
    if total < 1e-8:
        return 50.0
    # Closer to visible (smaller dist) => higher visible contribution
    return float(100.0 * dist_to_lw / total)


# ---------------------------------------------------------------------------
# Proxy 6 — Frequency proxy (FFT magnitude correlation)
# ---------------------------------------------------------------------------

def _contrib_frequency(v_gray: np.ndarray, lw_gray: np.ndarray, f_gray: np.ndarray,
                       source_cache: dict | None = None) -> float:
    """
    Compares the frequency-domain content of fused image against each source
    via correlation of FFT magnitude spectra, anchored with endpoint references.

    Complementary to spatial-domain proxies: LWIR typically concentrates energy
    in low frequencies (smooth thermal gradients) while visible has richer
    high-frequency content (texture, edges, fine patterns). This proxy measures
    which source's overall frequency profile the fused image more closely
    matches, independent of spatial alignment or pixel-level similarity.

    Uses endpoint anchoring (same approach as gradient proxy): the delta for
    fused=visible and fused=lwir defines the [0, 100] range, ensuring exact
    0 and 100 at the endpoints. Without anchoring, visible and LWIR images of
    the same scene share enough spatial structure that their FFT spectra are
    naturally correlated (~0.7+), compressing the sigmoid output range and
    preventing the proxy from reaching the expected 0/100 extremes.

    Operates on grayscale (channel average) and is permutation-invariant.

    Returns cont_vis in [0, 100].
    """
    if source_cache is None:
        source_cache = _build_freq_source_cache(v_gray, lw_gray)

    mag_f = source_cache["_fft_mag_fn"](f_gray)

    corr_vis = _safe_corrcoef(source_cache["mag_v"], mag_f)
    corr_lw = _safe_corrcoef(source_cache["mag_lw"], mag_f)
    delta = corr_vis - corr_lw

    return _map_delta_with_anchors(
        delta,
        visible_anchor=source_cache["freq_delta_visible_anchor"],
        thermal_anchor=source_cache["freq_delta_thermal_anchor"],
    )


def _build_freq_source_cache(v_gray: np.ndarray, lw_gray: np.ndarray) -> dict:
    """Precompute FFT terms that depend only on visible/LWIR sources."""
    def _fft_magnitude(img: np.ndarray) -> np.ndarray:
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        return np.log1p(np.abs(f_shift))

    mag_v = _fft_magnitude(v_gray)
    mag_lw = _fft_magnitude(lw_gray)

    # Anchor endpoints: delta when fused = visible, delta when fused = lwir
    freq_delta_visible_anchor = _safe_corrcoef(mag_v, mag_v) - _safe_corrcoef(mag_lw, mag_v)
    freq_delta_thermal_anchor = _safe_corrcoef(mag_v, mag_lw) - _safe_corrcoef(mag_lw, mag_lw)

    return {
        "mag_v": mag_v,
        "mag_lw": mag_lw,
        "freq_delta_visible_anchor": freq_delta_visible_anchor,
        "freq_delta_thermal_anchor": freq_delta_thermal_anchor,
        "_fft_mag_fn": _fft_magnitude,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_proxy_contributions(proxies: np.ndarray) -> dict:
    """Aggregate proxy scores into raw and disagreement-aware weighted estimates."""
    proxies = np.asarray(proxies, dtype=np.float64).ravel()
    if proxies.size == 0:
        return {
            "cont_vis_raw": 50.0,
            "cont_vis_weighted": 50.0,
            "cont_lw_weighted": 50.0,
            "contrib_std": 0.0,
            "contrib_confidence": 100.0,
        }

    cont_vis_raw = float(np.mean(proxies))
    contrib_std = float(np.std(proxies))

    median = float(np.median(proxies))
    mad = float(np.median(np.abs(proxies - median)))
    robust_scale = max(1.4826 * mad, 1e-8)
    robust_z = np.abs(proxies - median) / robust_scale
    robust_weights = 1.0 / (1.0 + robust_z ** 2)
    robust_weights = np.clip(robust_weights, 0.05, 1.0)
    weighted_center = float(np.average(proxies, weights=robust_weights))

    confidence = float(np.clip(1.0 - (contrib_std / 35.0), 0.0, 1.0))
    cont_vis_weighted = float(confidence * weighted_center + (1.0 - confidence) * 50.0)

    return {
        "cont_vis_raw": cont_vis_raw,
        "cont_vis_weighted": cont_vis_weighted,
        "cont_lw_weighted": 100.0 - cont_vis_weighted,
        "contrib_std": contrib_std,
        "contrib_confidence": 100.0 * confidence,
    }


# ---------------------------------------------------------------------------
# Calibration Functions
# ---------------------------------------------------------------------------

def _pava_non_decreasing(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Fit a non-decreasing sequence using the pool-adjacent-violators algorithm."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if weights is None:
        weights = np.ones_like(values, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()

    if values.size != weights.size:
        raise ValueError("values and weights must have the same length")

    block_values: list[float] = []
    block_weights: list[float] = []
    block_sizes: list[int] = []

    for value, weight in zip(values, weights):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_sizes.append(1)

        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            left_weight = block_weights[-2]
            right_weight = block_weights[-1]
            merged_weight = left_weight + right_weight
            merged_value = (
                block_values[-2] * left_weight + block_values[-1] * right_weight
            ) / merged_weight

            block_values[-2] = merged_value
            block_weights[-2] = merged_weight
            block_sizes[-2] += block_sizes[-1]
            block_values.pop()
            block_weights.pop()
            block_sizes.pop()

    return np.repeat(np.asarray(block_values, dtype=np.float64), np.asarray(block_sizes, dtype=np.int64))


def _fit_monotonic_knots(
    raw_values: np.ndarray,
    alpha_lwir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit monotonic knots from raw metric values and known alpha.

    Returns (raw_knots, visible_knots, group_raw, group_visible).
    """
    unique_alphas = np.unique(alpha_lwir)
    group_raw = []
    group_visible = []

    for alpha in unique_alphas:
        mask = np.abs(alpha_lwir - alpha) < 1e-8
        group_vals = raw_values[mask]
        if group_vals.size == 0:
            continue
        group_raw.append(float(np.median(group_vals)))
        group_visible.append(float(1.0 - alpha))

    group_raw = np.array(group_raw, dtype=np.float64)
    group_visible = np.array(group_visible, dtype=np.float64)

    order = np.argsort(group_raw, kind="mergesort")
    raw_sorted = group_raw[order]
    visible_sorted = group_visible[order]

    fitted_visible = _pava_non_decreasing(visible_sorted)
    fitted_visible = np.clip(fitted_visible, 0.0, 1.0)

    return raw_sorted, fitted_visible, group_raw, group_visible


def fit_contribution_calibration(
    raw_visible_percent: np.ndarray,
    alpha_lwir: np.ndarray,
    per_proxy_values: dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Fit monotonic calibrations from raw contribution scores to a latent axis.

    Fits one calibration curve per proxy metric (reg, mi, ssim, grad, spectral,
    freq) so that each proxy's non-linearity is corrected independently. At
    application time, each proxy is calibrated through its own curve and the
    results are averaged — this handles metric saturation (e.g. grad/freq)
    without manual rescaling.

    Also fits an aggregate curve on the mean cont_vis for backward compatibility
    and diagnostics.

    Parameters
    ----------
    raw_visible_percent : np.ndarray
        Raw visible contribution scores (aggregate) in [0, 100].
    alpha_lwir : np.ndarray
        Known LWIR mixture factors in [0, 1].
    per_proxy_values : dict[str, np.ndarray] | None
        Per-proxy raw values keyed by proxy name (e.g. "cont_vis_reg").
        When provided, an independent calibration curve is fitted per proxy.
    """
    raw_visible_percent = np.asarray(raw_visible_percent, dtype=np.float64).ravel()
    alpha_lwir = np.clip(np.asarray(alpha_lwir, dtype=np.float64).ravel(), 0.0, 1.0)

    if raw_visible_percent.size == 0:
        raise ValueError("Calibration requires at least one sample")
    if raw_visible_percent.size != alpha_lwir.size:
        raise ValueError("raw_visible_percent and alpha_lwir must have the same length")

    # --- Aggregate calibration (backward compat / diagnostics) ---------------
    raw_knots, visible_knots, group_raw, group_visible = _fit_monotonic_knots(
        raw_visible_percent, alpha_lwir)

    result = {
        "raw_knots": raw_knots,
        "visible_knots": visible_knots,
        "thermal_knots": 1.0 - visible_knots,
        "raw_points_cont_vis": raw_visible_percent.copy(),
        "raw_points_alpha": alpha_lwir.copy(),
        "group_raw": group_raw.copy(),
        "group_visible": group_visible.copy(),
    }

    # --- Per-proxy calibrations ----------------------------------------------
    if per_proxy_values:
        proxy_calibrations = {}
        unique_alphas = np.unique(alpha_lwir)

        # Only fit curves for proxies currently enabled by settings.  The
        # disabled ones still exist in `per_proxy_values` (cache schema is
        # full) but we skip them so they don't enter the IVW averaging.
        for proxy_key in _active_proxy_keys():
            pvals = per_proxy_values.get(proxy_key)
            if pvals is None:
                continue
            pvals = np.asarray(pvals, dtype=np.float64).ravel()
            if pvals.size != alpha_lwir.size:
                continue
            p_raw_knots, p_vis_knots, p_group_raw, p_group_vis = _fit_monotonic_knots(
                pvals, alpha_lwir)

            # Reliability estimate: intra-group dispersion in calibrated space.
            # Calibrate each raw sample through its own curve, then measure how
            # tight the per-alpha-group distribution is (ideally a proxy maps all
            # samples at alpha=k to the same calibrated value).  Lower dispersion
            # ⇒ higher reliability ⇒ larger IVW weight.
            calibrated = np.clip(np.interp(
                pvals, p_raw_knots, p_vis_knots,
                left=p_vis_knots[0], right=p_vis_knots[-1]), 0.0, 1.0)
            group_stds = []
            group_alphas_list = []
            for alpha in unique_alphas:
                mask = np.abs(alpha_lwir - alpha) < 1e-8
                if mask.sum() > 1:
                    group_stds.append(float(np.std(calibrated[mask])))
                    group_alphas_list.append(float(alpha))
            mean_std = float(np.mean(group_stds)) if group_stds else 0.0
            # Std floor avoids exploding weight on degenerate zero-variance
            # proxies (e.g. constant output) and caps the maximum weight ratio.
            std_floored = max(mean_std, 1e-2)

            proxy_calibrations[proxy_key] = {
                "raw_knots": p_raw_knots,
                "visible_knots": p_vis_knots,
                "group_raw": p_group_raw,
                "group_visible": p_group_vis,
                "calibrated_mean_std": mean_std,
                "std_floored": std_floored,
                "ivw_weight": 1.0 / std_floored,  # soft IVW (1/σ); consumer normalises + caps
                "group_alphas_std": np.asarray(group_alphas_list, dtype=np.float64),
                "group_std_per_alpha": np.asarray(group_stds, dtype=np.float64),
            }

        if proxy_calibrations:
            # Normalise IVW weights so they sum to 1 across available proxies
            total_w = sum(pc["ivw_weight"] for pc in proxy_calibrations.values())
            if total_w > 0:
                for pc in proxy_calibrations.values():
                    pc["ivw_weight_normalized"] = pc["ivw_weight"] / total_w
            result["proxy_calibrations"] = proxy_calibrations

    return result


def _apply_single_calibration(
    raw_value: np.ndarray,
    cal: dict,
) -> np.ndarray:
    """Interpolate raw cont_vis through one set of calibration knots."""
    raw_knots = np.asarray(cal["raw_knots"], dtype=np.float64)
    visible_knots = np.asarray(cal["visible_knots"], dtype=np.float64)
    return np.clip(np.interp(
        raw_value, raw_knots, visible_knots,
        left=visible_knots[0], right=visible_knots[-1],
    ), 0.0, 1.0)


def _select_best_fit_calibration(
    calibration: dict,
    proxy_values: dict[str, float] | None = None,
) -> tuple[dict, str]:
    """Select the calibration format that produces lowest inter-proxy disagreement.

    For each available calibration type, calibrate all proxies independently
    and measure the **weighted** std of the resulting visible fractions, using
    the same multi-format averaged proxy weights the aggregator will use.  The
    type with the smallest weighted std (most agreement under the actual
    aggregator) is the best fit for this method.  Coherent with the soft-IVW
    aggregator in `apply_contribution_calibration` — both use the same notion
    of dispersion.

    Returns (active_cal_dict, selected_type_name).
    Falls back to blend when proxy_values is missing or only one type exists.
    """
    by_type = calibration.get("calibrations_by_type")
    if not by_type:
        return calibration, "blend"

    # If no proxy values or only one format, use blend as default
    if not proxy_values or len(by_type) <= 1:
        for preferred in ("blend", "freq_blend", "nonlinear", "concat"):
            if preferred in by_type:
                return by_type[preferred], preferred
        return calibration, "unknown"

    averaged_weights = calibration.get("averaged_proxy_weights") or {}

    best_type = None
    best_std = float("inf")
    best_cal = None

    for stype, cal in by_type.items():
        proxy_cals = cal.get("proxy_calibrations")
        if not proxy_cals:
            continue
        keys, fracs, weights = [], [], []
        for pk, pcal in proxy_cals.items():
            pval = proxy_values.get(pk)
            if pval is None:
                continue
            pval_arr = np.asarray(pval, dtype=np.float64)
            keys.append(pk)
            fracs.append(float(_apply_single_calibration(pval_arr, pcal)))
            weights.append(float(averaged_weights.get(
                pk, pcal.get("ivw_weight_normalized",
                              pcal.get("ivw_weight", 0.0)))))
        if len(fracs) < 2:
            continue
        fracs_arr = np.asarray(fracs, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.sum() <= 0:
            weights_arr = np.ones_like(weights_arr)
        weights_arr = _cap_proxy_weights(weights_arr)
        # Weighted std (same notion of dispersion as the aggregator)
        mean_w = float(np.dot(weights_arr, fracs_arr))
        var_w = float(np.dot(weights_arr, (fracs_arr - mean_w) ** 2))
        contrib_std = float(np.sqrt(max(var_w, 0.0)))
        if contrib_std < best_std:
            best_std = contrib_std
            best_type = stype
            best_cal = cal

    if best_cal is None:
        for preferred in ("blend", "freq_blend", "nonlinear", "concat"):
            if preferred in by_type:
                return by_type[preferred], preferred
        return calibration, "unknown"

    return best_cal, best_type


def apply_contribution_calibration(
    raw_visible_percent: float | np.ndarray,
    calibration: dict,
    return_visible: bool = False,
    proxy_values: dict[str, float] | None = None,
) -> float | np.ndarray:
    """
    Map raw contribution scores into the calibrated latent axis.

    When ``proxy_values`` is provided and the calibration contains per-proxy
    curves (``proxy_calibrations``), each proxy is calibrated independently
    through its own monotonic curve, and the results are averaged.  This
    corrects for metric saturation (e.g. grad/freq saturate quickly toward
    visible) without manual rescaling — each proxy's non-linearity is handled
    by its own calibration curve.

    Falls back to the aggregate cont_vis calibration when per-proxy curves are
    not available.

    By default the returned value is the LWIR fraction in [0, 1]:
    0 = visible-only, 1 = LWIR-only.
    """
    # --- Resolve which calibration format to use (best-fit for this method) --
    active_cal, _selected_type = _select_best_fit_calibration(calibration, proxy_values)

    # --- Multi-format averaged proxy weights (preferred over single-format) --
    averaged_weights = calibration.get("averaged_proxy_weights")

    # --- Per-proxy calibration (preferred path) ------------------------------
    proxy_cals = active_cal.get("proxy_calibrations")
    if proxy_values and proxy_cals:
        fractions = []
        weights = []
        for proxy_key, p_cal in proxy_cals.items():
            pval = proxy_values.get(proxy_key)
            if pval is None:
                continue
            pval_arr = np.asarray(pval, dtype=np.float64)
            fractions.append(_apply_single_calibration(pval_arr, p_cal))
            # Use multi-format averaged weight if available, else single-format
            if averaged_weights and proxy_key in averaged_weights:
                weights.append(averaged_weights[proxy_key])
            else:
                weights.append(p_cal.get("ivw_weight_normalized",
                                          p_cal.get("ivw_weight", 1.0)))
        if fractions:
            fractions_arr = np.asarray(fractions, dtype=np.float64)
            weights_arr = np.asarray(weights, dtype=np.float64)
            if weights_arr.sum() > 0:
                weights_arr = _cap_proxy_weights(weights_arr)
                visible_fraction = np.tensordot(weights_arr, fractions_arr, axes=1)
            else:
                visible_fraction = np.mean(fractions_arr, axis=0)
        else:
            visible_fraction = _apply_single_calibration(
                np.asarray(raw_visible_percent, dtype=np.float64), active_cal)
    else:
        # Fallback: aggregate cont_vis calibration
        visible_fraction = _apply_single_calibration(
            np.asarray(raw_visible_percent, dtype=np.float64), active_cal)

    if return_visible:
        return visible_fraction
    return 1.0 - visible_fraction


def _interp_proxy_sigma(alpha_hat: float, p_cal: dict) -> float:
    """Interpolate a proxy's σ at the current α̂ estimate.

    Uses the calibration-time per-α-group σ curve (`group_alphas_std` /
    `group_std_per_alpha`) with linear interpolation and flat extrapolation
    beyond the calibrated α range.  Falls back to the global mean σ
    (`calibrated_mean_std`) when the per-α curve is unavailable.
    """
    alpha_grid = p_cal.get("group_alphas_std")
    sigma_grid = p_cal.get("group_std_per_alpha")
    fallback = float(p_cal.get("calibrated_mean_std", 1.0))
    if alpha_grid is None or sigma_grid is None:
        return max(fallback, 1e-8)
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64).ravel()
    sigma_grid = np.asarray(sigma_grid, dtype=np.float64).ravel()
    if alpha_grid.size == 0 or sigma_grid.size == 0 or alpha_grid.size != sigma_grid.size:
        return max(fallback, 1e-8)
    # np.interp requires monotonically increasing x — sort if needed
    order = np.argsort(alpha_grid)
    alpha_sorted = alpha_grid[order]
    sigma_sorted = sigma_grid[order]
    sigma_val = float(np.interp(np.clip(alpha_hat, 0.0, 1.0),
                                 alpha_sorted, sigma_sorted,
                                 left=sigma_sorted[0], right=sigma_sorted[-1]))
    return max(sigma_val, 1e-8)


def apply_contribution_calibration_alpha_dep(
    calibration: dict,
    proxy_values: dict[str, float],
    max_iter: int = 4,
    tol: float = 1e-3,
) -> dict:
    """α-dependent Inverse-Variance Weighting of per-proxy calibrated fractions.

    Unlike `apply_contribution_calibration` which uses a single global σ per
    proxy (mean over α-groups), this routine iterates:

      1. Start from the global-IVW estimate of α̂ (LWIR fraction).
      2. For each proxy, look up σ_p(α̂) via the calibration's per-α σ curve.
      3. Recompute weights w_p = 1 / σ_p(α̂) (soft IVW, then cap+normalise).
      4. Combine the per-proxy calibrated visible fractions → new α̂.
      5. Repeat until |Δα̂| < tol or `max_iter` reached.

    Returns a dict with the final α̂ (LWIR fraction), iteration trace, final
    weights, and diagnostic bookkeeping.  Falls back to the global result
    when per-proxy σ curves are missing.

    **TODO (debug once we have results)**: verify convergence is stable and
    `max_iter=4, tol=1e-3` is adequate across the method population; tighten
    tolerance or raise max_iter if empirical `ivw_iters` distribution hits
    the ceiling too often.
    """
    # Best-fit calibration format for this method's proxy values
    active_cal, selected_type = _select_best_fit_calibration(calibration, proxy_values)
    proxy_cals = active_cal.get("proxy_calibrations")
    averaged_weights = calibration.get("averaged_proxy_weights")

    # Global (non α-dep) fallback: reuse existing routine.
    def _global_result():
        vf_global = float(apply_contribution_calibration(
            50.0, calibration, proxy_values=proxy_values, return_visible=True))
        alpha_global = 1.0 - vf_global
        weights_global_dict = {}
        if proxy_cals:
            for pk, pc in proxy_cals.items():
                weights_global_dict[pk] = float(
                    pc.get("ivw_weight_normalized", pc.get("ivw_weight", 0.0)))
        return {
            "alpha_hat": alpha_global,
            "visible_fraction": vf_global,
            "iters": 0,
            "alpha_trace": [alpha_global],
            "weights_effective": weights_global_dict,
            "weights_global": dict(weights_global_dict),
            "converged": True,
            "fallback": True,
        }

    if not proxy_cals or not proxy_values:
        return _global_result()

    # Pre-compute per-proxy calibrated visible fractions — these don't depend
    # on α̂, only the *weights* do, so we can factor this out of the loop.
    proxy_fractions = {}
    global_weights = {}
    for proxy_key, p_cal in proxy_cals.items():
        pval = proxy_values.get(proxy_key)
        if pval is None:
            continue
        pval_arr = np.asarray(pval, dtype=np.float64)
        proxy_fractions[proxy_key] = float(_apply_single_calibration(pval_arr, p_cal))
        # Prefer multi-format averaged weight over single-format IVW
        if averaged_weights and proxy_key in averaged_weights:
            global_weights[proxy_key] = averaged_weights[proxy_key]
        else:
            global_weights[proxy_key] = float(
                p_cal.get("ivw_weight_normalized", p_cal.get("ivw_weight", 0.0)))

    if not proxy_fractions:
        return _global_result()

    # Seed α̂ with the global-IVW result so we start close to the solution.
    keys = list(proxy_fractions.keys())
    fracs = np.array([proxy_fractions[k] for k in keys], dtype=np.float64)
    gw = np.array([global_weights.get(k, 0.0) for k in keys], dtype=np.float64)
    if gw.sum() > 0:
        gw_norm = _cap_proxy_weights(gw)
        vf = float(np.dot(gw_norm, fracs))
    else:
        vf = float(np.mean(fracs))
    alpha_hat = 1.0 - vf
    alpha_trace = [alpha_hat]

    effective_weights = dict(zip(keys, gw_norm if gw.sum() > 0 else np.ones(len(keys)) / len(keys)))
    converged = False
    iters_done = 0

    for it in range(max_iter):
        # Refresh σ_p at current α̂ → recompute weights.
        sigmas = np.array(
            [_interp_proxy_sigma(alpha_hat, proxy_cals[k]) for k in keys],
            dtype=np.float64,
        )
        w = 1.0 / sigmas  # soft IVW: 1/σ instead of 1/σ²
        w_sum = float(w.sum())
        if w_sum <= 0 or not np.isfinite(w_sum):
            break
        w_norm = _cap_proxy_weights(w)
        vf_new = float(np.dot(w_norm, fracs))
        alpha_new = 1.0 - vf_new

        effective_weights = dict(zip(keys, w_norm))
        iters_done = it + 1
        alpha_trace.append(alpha_new)

        if abs(alpha_new - alpha_hat) < tol:
            alpha_hat = alpha_new
            vf = vf_new
            converged = True
            break
        alpha_hat = alpha_new
        vf = vf_new

    return {
        "alpha_hat": float(alpha_hat),
        "visible_fraction": float(vf),
        "iters": int(iters_done),
        "alpha_trace": [float(a) for a in alpha_trace],
        "weights_effective": effective_weights,
        "weights_global": global_weights,
        "converged": bool(converged),
        "fallback": False,
        "calibration_type": selected_type,
    }


def _with_calibration(raw_result: dict, calibration: dict | None) -> dict:
    """Attach calibrated fields to a raw contribution result."""
    result = dict(raw_result)
    if calibration is None:
        result["latent_z"] = None
        result["latent_z_raw"] = None
        result["latent_z_weighted"] = None
        result["latent_z_alpha_dep"] = None
        result["cont_vis_calibrated"] = None
        result["cont_lw_calibrated"] = None
        result["ivw_iters"] = None
        result["ivw_converged"] = None
        result["ivw_alpha_trace"] = None
        result["ivw_weights_effective"] = None
        result["ivw_weights_global"] = None
        return result

    # Extract per-proxy values for per-proxy calibration.  Restrict to the
    # enabled subset so disabled proxies don't influence the apply path
    # (they're still in `result` because the per-image cache is unfiltered).
    proxy_values = {k: result[k] for k in _active_proxy_keys() if k in result}

    cont_vis_weighted = float(result.get("cont_vis_weighted", result["cont_vis"]))
    cont_vis_raw = float(result.get("cont_vis_raw", result["cont_vis"]))

    latent_z_weighted = float(apply_contribution_calibration(
        cont_vis_weighted, calibration, proxy_values=proxy_values))
    latent_z_raw = float(apply_contribution_calibration(
        cont_vis_raw, calibration, proxy_values=proxy_values))

    # α-dependent IVW (iterative), using the same per-proxy calibrated fractions.
    alpha_dep = apply_contribution_calibration_alpha_dep(calibration, proxy_values)

    result["latent_z"] = latent_z_weighted
    result["latent_z_weighted"] = latent_z_weighted
    result["latent_z_raw"] = latent_z_raw
    result["latent_z_alpha_dep"] = float(alpha_dep["alpha_hat"])
    result["ivw_iters"] = alpha_dep["iters"]
    result["ivw_converged"] = alpha_dep["converged"]
    result["ivw_alpha_trace"] = alpha_dep["alpha_trace"]
    result["ivw_weights_effective"] = alpha_dep["weights_effective"]
    result["ivw_weights_global"] = alpha_dep["weights_global"]
    result["calibration_type"] = alpha_dep.get("calibration_type", "blend")
    result["cont_lw_calibrated"] = 100.0 * latent_z_weighted
    result["cont_vis_calibrated"] = 100.0 - result["cont_lw_calibrated"]
    return result


# ---------------------------------------------------------------------------
# Permutation pre-screening
# ---------------------------------------------------------------------------

def _prescreen_permutations(fused_permutations: list[np.ndarray], v: np.ndarray,
                            max_perms: int = 2, min_delta: float = 0.01) -> list[np.ndarray]:
    """
    Select the top channel permutations by per-channel correlation with visible.

    For standard RGB fusion methods, identity is typically best. For descriptor-
    based methods (PCA, FA), the best channel mapping may differ. Pre-screening
    with a cheap metric (sum of per-channel correlations) avoids running the
    full expensive metric suite on all 6 permutations.

    If the top permutations are very similar (delta < min_delta), keeps only the
    best one to avoid redundant computation.
    """
    if len(fused_permutations) <= max_perms:
        return fused_permutations

    scores = []
    for f_perm in fused_permutations:
        score = sum(
            _safe_corrcoef(v[..., ch].ravel(), f_perm[..., ch].ravel())
            for ch in range(min(v.shape[2], f_perm.shape[2]))
        )
        scores.append(score)

    sorted_indices = np.argsort(scores)[::-1]

    # If top-1 and top-2 are very close, just use the best one
    if len(sorted_indices) >= 2:
        if abs(scores[sorted_indices[0]] - scores[sorted_indices[1]]) < min_delta:
            return [fused_permutations[sorted_indices[0]]]

    return [fused_permutations[sorted_indices[i]] for i in range(min(max_perms, len(sorted_indices)))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_contribution_rgb(
    visual: np.ndarray,
    lwir:   np.ndarray,
    fused:  np.ndarray,
    calibration: dict | None = None,
) -> dict:
    """
    Compute the latent visible-to-LWIR contribution score for one triplet.

    Uses 6 proxy methods:
      - reg:      Per-channel NNLS regression (avoids flatten bias)
      - mi:       Per-channel unique mutual information (avoids flatten bias)
      - ssim:     Multichannel SSIM comparison
      - grad:     Combined gradient magnitude + orientation (single proxy)
      - spectral: Inter-channel independence structure
      - freq:     FFT magnitude spectrum correlation

    Permutation-sensitive proxies (reg, mi, ssim) are evaluated on the top 2
    channel permutations selected by pre-screening. Permutation-invariant proxies
    (grad, spectral, freq) are computed once without permutation search.

    Parameters
    ----------
    visual : np.ndarray
        Visible image, shape (H, W, 3) or (H, W).
    lwir : np.ndarray
        LWIR grayscale image, shape (H, W).
    fused : np.ndarray
        Fused pseudo-RGB image, shape (H, W, 3) or (H, W).
    calibration : dict | None
        Optional calibration knots from fit_contribution_calibration.

    Returns
    -------
    dict
        Raw similarity metrics, per-proxy visible contribution estimates, the
        final averaged visible/LWIR percentages, inter-proxy spread, and
        optional calibrated latent-axis values when a calibration is provided.
    """

    # --- Convert to channel-aware float images ------------------------------
    vis_color = _to_color_for_metric(visual)
    fused_color = _to_color_for_metric(fused)
    lwir_color = _to_color_for_metric(lwir)

    # --- Align shapes -------------------------------------------------------
    vis_color, lwir_color, fused_color = _align_shapes(vis_color, lwir_color, fused_color)

    # --- Normalise each channel independently to [0, 1] ---------------------
    v = _normalise_color(vis_color)
    lw = _normalise_color(lwir_color)
    f = _normalise_color(fused_color)

    # --- Grayscale versions for permutation-invariant proxies ---------------
    v_gray = _channelwise_average(v)
    lw_gray = _channelwise_average(lw)
    f_gray = _channelwise_average(f)

    # --- Permutation-invariant proxies (computed once) ----------------------
    grad_source_cache = _build_gradient_source_cache(v_gray, lw_gray)
    cv_grad_combined, grad_diag = _contrib_gradient_combined(
        v_gray, lw_gray, f_gray, source_cache=grad_source_cache,
    )
    cv_spectral = _contrib_spectral_independence(v, lw, f)
    freq_source_cache = _build_freq_source_cache(v_gray, lw_gray)
    cv_freq = _contrib_frequency(v_gray, lw_gray, f_gray, source_cache=freq_source_cache)

    # --- Channel redundancy diagnostic (permutation-invariant) -------------
    # Mean |corr| between the three fused-image channels.  Flags methods that
    # achieve high latent_z by duplicating thermal across channels (e.g.
    # combine_lwir_npy does T,T,T → redundancy ≈ 1) vs methods that carry
    # genuinely distributed thermal content across distinct channels.
    # Reference: natural RGB images typically sit around 0.8.
    cr_components = _channel_redundancy_components(f)
    channel_redundancy = cr_components["channel_redundancy"]

    # --- Pre-screen channel permutations ------------------------------------
    all_permutations = _channel_permutations(f)
    selected_permutations = _prescreen_permutations(all_permutations, v)

    # --- Permutation-sensitive proxies (per-channel reg, mi, ssim) ----------
    cv_reg_candidates = []
    cv_mi_candidates = []
    cv_ssim_candidates = []
    ssim_vis_candidates = []
    ssim_lw_candidates = []
    cc_vis_candidates = []
    cc_lw_candidates = []
    sp_vis_candidates = []
    sp_lw_candidates = []
    mi_vis_candidates = []
    mi_lw_candidates = []
    rmse_vis_candidates = []
    rmse_lw_candidates = []

    for f_perm in selected_permutations:
        # Per-channel NNLS regression
        cv_reg_candidates.append(_contrib_regression_perchannel(v, lw, f_perm))

        # Per-channel unique MI
        cv_mi_candidates.append(_contrib_unique_mi_perchannel(v, lw, f_perm))

        # Multichannel SSIM
        s_vis = _ssim_multichannel(v, f_perm)
        s_lw = _ssim_multichannel(lw, f_perm)
        ssim_vis_candidates.append(s_vis)
        ssim_lw_candidates.append(s_lw)
        cv_ssim_candidates.append(_contrib_ssim_from_scores(s_vis, s_lw))

        # Diagnostic raw metrics (for traceability, not used in proxy aggregation)
        v_vec = v.reshape(-1)
        lw_vec = lw.reshape(-1)
        f_vec = f_perm.reshape(-1)
        cc_vis_candidates.append(_safe_corrcoef(v_vec, f_vec))
        cc_lw_candidates.append(_safe_corrcoef(lw_vec, f_vec))
        sp_vis_candidates.append(_safe_spearman(v_vec, f_vec))
        sp_lw_candidates.append(_safe_spearman(lw_vec, f_vec))

        mi_vis_ch = []
        mi_lw_ch = []
        for ch in range(v.shape[2]):
            mi_vis_ch.append(_mutual_information(v[..., ch], f_perm[..., ch]))
            mi_lw_ch.append(_mutual_information(lw[..., ch], f_perm[..., ch]))
        mi_vis_candidates.append(float(np.mean(mi_vis_ch)))
        mi_lw_candidates.append(float(np.mean(mi_lw_ch)))
        rmse_vis_candidates.append(float(np.sqrt(np.mean((v - f_perm) ** 2))))
        rmse_lw_candidates.append(float(np.sqrt(np.mean((lw - f_perm) ** 2))))

    # --- Aggregate across permutations -------------------------------------
    # Use the mean over all channel permutations.  Rationale: most fusion
    # methods in this project do not preserve a canonical R/G/B channel
    # mapping (the output channels are just "fused channels" without fixed
    # semantics), so no single permutation is "the correct alignment".
    # Averaging treats all permutations as equally valid views of the same
    # content and removes the upward bias of `max` (best-of-6) and the
    # downward bias of `min`.  Kept consistent across all proxies (including
    # concat) so that calibration and evaluation use the same reducer.
    cc_vis = float(np.mean(cc_vis_candidates))
    cc_lw = float(np.mean(cc_lw_candidates))
    sp_vis = float(np.mean(sp_vis_candidates))
    sp_lw = float(np.mean(sp_lw_candidates))
    mi_vis = float(np.mean(mi_vis_candidates))
    mi_lw = float(np.mean(mi_lw_candidates))
    ssim_vis = float(np.mean(ssim_vis_candidates))
    ssim_lw = float(np.mean(ssim_lw_candidates))
    rmse_vis = float(np.mean(rmse_vis_candidates))
    rmse_lw = float(np.mean(rmse_lw_candidates))
    cv_reg = float(np.mean(cv_reg_candidates))
    cv_mi = float(np.mean(cv_mi_candidates))
    cv_ssim = float(np.mean(cv_ssim_candidates))

    # --- Final estimate (6 proxies) ----------------------------------------
    proxies = np.array(
        [cv_reg, cv_mi, cv_ssim, cv_grad_combined, cv_spectral, cv_freq],
        dtype=np.float64,
    )
    aggregation = _aggregate_proxy_contributions(proxies)
    cont_vis_raw = aggregation["cont_vis_raw"]
    cont_vis = aggregation["cont_vis_weighted"]
    cont_lw = aggregation["cont_lw_weighted"]
    contrib_std = aggregation["contrib_std"]
    contrib_confidence = aggregation["contrib_confidence"]

    latent_z = None
    latent_z_raw = None
    latent_z_weighted = None
    latent_z_alpha_dep = None
    cont_vis_calibrated = None
    cont_lw_calibrated = None
    ivw_iters = None
    ivw_converged = None
    ivw_alpha_trace = None
    ivw_weights_effective = None
    ivw_weights_global = None
    if calibration is not None:
        proxy_values = {
            "cont_vis_reg": cv_reg,
            "cont_vis_mi": cv_mi,
            "cont_vis_ssim": cv_ssim,
            "cont_vis_grad_combined": cv_grad_combined,
            "cont_vis_spectral": cv_spectral,
            "cont_vis_freq": cv_freq,
        }
        latent_z_raw = float(apply_contribution_calibration(
            cont_vis_raw, calibration, proxy_values=proxy_values))
        latent_z_weighted = float(apply_contribution_calibration(
            cont_vis, calibration, proxy_values=proxy_values))
        latent_z = latent_z_weighted
        cont_lw_calibrated = 100.0 * latent_z_weighted
        cont_vis_calibrated = 100.0 - cont_lw_calibrated

        alpha_dep = apply_contribution_calibration_alpha_dep(
            calibration, proxy_values)
        latent_z_alpha_dep = float(alpha_dep["alpha_hat"])
        ivw_iters = alpha_dep["iters"]
        ivw_converged = alpha_dep["converged"]
        ivw_alpha_trace = alpha_dep["alpha_trace"]
        ivw_weights_effective = alpha_dep["weights_effective"]
        ivw_weights_global = alpha_dep["weights_global"]

    return {
        # Raw similarity metrics (diagnostic, not used in proxy aggregation)
        "cc_vis":    cc_vis,
        "cc_lw":     cc_lw,
        "sp_vis":    sp_vis,
        "sp_lw":     sp_lw,
        "mi_vis":    mi_vis,
        "mi_lw":     mi_lw,
        "ssim_vis":  ssim_vis,
        "ssim_lw":   ssim_lw,
        "rmse_vis":  rmse_vis,
        "rmse_lw":   rmse_lw,
        "grad_mag_corr_vis": grad_diag["grad_mag_corr_vis"],
        "grad_mag_corr_lw":  grad_diag["grad_mag_corr_lw"],
        "grad_ori_agree_vis": grad_diag["grad_ori_agree_vis"],
        "grad_ori_agree_lw":  grad_diag["grad_ori_agree_lw"],
        # Per-proxy contributions
        "cont_vis_reg":  cv_reg,
        "cont_vis_mi":   cv_mi,
        "cont_vis_ssim": cv_ssim,
        "cont_vis_grad_combined": cv_grad_combined,
        "cont_vis_grad_mag": grad_diag["cont_vis_grad_mag"],
        "cont_vis_grad_ori": grad_diag["cont_vis_grad_ori"],
        "cont_vis_spectral": cv_spectral,
        "cont_vis_freq": cv_freq,
        # Final estimate
        "cont_vis_raw": cont_vis_raw,
        "cont_vis_weighted": cont_vis,
        "cont_vis":     cont_vis,
        "cont_lw":      cont_lw,
        "contrib_std":  contrib_std,
        "contrib_confidence": contrib_confidence,
        "latent_z":     latent_z,
        "latent_z_raw": latent_z_raw,
        "latent_z_weighted": latent_z_weighted,
        "latent_z_alpha_dep": latent_z_alpha_dep,
        "cont_vis_calibrated": cont_vis_calibrated,
        "cont_lw_calibrated":  cont_lw_calibrated,
        # Channel-redundancy diagnostic (0 = fully independent channels,
        # 1 = channels are duplicates).  Separates thermal-rich fusions that
        # carry distributed info from those that simply replicate thermal.
        "channel_redundancy": channel_redundancy,
        "channel_redundancy_pearson": cr_components["pearson_global"],
        "channel_redundancy_spearman": cr_components["spearman_global"],
        "channel_redundancy_tile": cr_components["pearson_tiles"],
        "channel_redundancy_effrank": cr_components["effective_rank_redundancy"],
        "channel_redundancy_w_pearson": cr_components["w_pearson_global"],
        "channel_redundancy_w_spearman": cr_components["w_spearman_global"],
        "channel_redundancy_w_tile": cr_components["w_pearson_tiles"],
        "channel_redundancy_w_effrank": cr_components["w_effective_rank_redundancy"],
        # IVW α-dependent diagnostics: iterative reweighting of proxies using
        # σ_p(α̂).  Stored as bookkeeping to inspect convergence / weight
        # redistribution in downstream plots.  Not included in numeric_keys
        # aggregation beyond `latent_z_alpha_dep` and `ivw_iters`.
        "ivw_iters": ivw_iters,
        "ivw_converged": ivw_converged,
        "ivw_alpha_trace": ivw_alpha_trace,
        "ivw_weights_effective": ivw_weights_effective,
        "ivw_weights_global": ivw_weights_global,
    }


def aggregate_contribution_results(results: list[dict]) -> dict:
    """Aggregate per-image results into a mean/std summary."""
    if not results:
        return {}

    summary = {}
    numeric_keys = [
        "fusion_time_ms",
        "cc_vis", "cc_lw", "sp_vis", "sp_lw", "mi_vis", "mi_lw", "ssim_vis", "ssim_lw",
        "rmse_vis", "rmse_lw",
        "grad_mag_corr_vis", "grad_mag_corr_lw", "grad_ori_agree_vis", "grad_ori_agree_lw",
        "cont_vis_reg", "cont_vis_mi", "cont_vis_ssim",
        "cont_vis_grad_combined", "cont_vis_grad_mag", "cont_vis_grad_ori",
        "cont_vis_spectral", "cont_vis_freq",
        "cont_vis_raw", "cont_vis_weighted",
        "cont_vis", "cont_lw", "contrib_std",
        "contrib_confidence",
        "latent_z", "latent_z_raw", "latent_z_weighted", "cont_vis_calibrated", "cont_lw_calibrated",
        "latent_z_alpha_dep",
        "channel_redundancy",
        "channel_redundancy_pearson",
        "channel_redundancy_spearman",
        "channel_redundancy_tile",
        "channel_redundancy_effrank",
        "channel_redundancy_w_pearson",
        "channel_redundancy_w_spearman",
        "channel_redundancy_w_tile",
        "channel_redundancy_w_effrank",
        "ivw_iters",
    ]

    for key in numeric_keys:
        values = [item[key] for item in results if item.get(key) is not None]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }

    # --- Aggregate per-proxy IVW weights (dict-valued) ---------------------
    # Record mean effective α-dep weight and mean global weight per proxy so
    # we can quantify how much the iterative reweighting redistributes mass.
    for field, out_key in (("ivw_weights_effective", "ivw_weights_effective_mean"),
                             ("ivw_weights_global", "ivw_weights_global_mean")):
        per_proxy: dict[str, list[float]] = {}
        for item in results:
            wdict = item.get(field)
            if not isinstance(wdict, dict):
                continue
            for pk, wv in wdict.items():
                if wv is None:
                    continue
                per_proxy.setdefault(pk, []).append(float(wv))
        if per_proxy:
            summary[out_key] = {
                pk: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
                for pk, vals in per_proxy.items()
            }

    # Convergence rate of IVW α-dep
    conv_flags = [item.get("ivw_converged") for item in results
                  if item.get("ivw_converged") is not None]
    if conv_flags:
        summary["ivw_converged_rate"] = float(np.mean([1.0 if c else 0.0 for c in conv_flags]))

    # Best-fit calibration type: mode (most frequently selected) + distribution
    cal_types = [item.get("calibration_type") for item in results
                 if item.get("calibration_type")]
    if cal_types:
        from collections import Counter
        counts = Counter(cal_types)
        summary["calibration_type_mode"] = counts.most_common(1)[0][0]
        summary["calibration_type_counts"] = dict(counts)

    return summary
