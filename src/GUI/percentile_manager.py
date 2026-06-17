#!/usr/bin/env python3
# encoding: utf-8
"""
Percentile estimation for groups of variance trials.

For each group of N repeated trials, fits a normal distribution to the metrics
of those trials and returns the requested percentile.

Scalars  (mAP50, MR, LAMR, …)  →  norm.ppf(p, mean, std)
Curves   (PR, F1, MR-FPPI, …)  →  norm.ppf(p) applied point-wise after
                                    stacking the N trial curves (NaN-safe).
"""

import warnings
import numpy as np
from scipy.stats import norm as scipy_norm

from utils import log, bcolors


# pr_data_best fields that are curves (list-of-class-lists, each 1000 pts).
# Note: bare 'p'/'r'/'f1' in YOLO output are per-class SCALARS (max precision/recall/F1),
# NOT curves — those live in 'p_plot'/'r_plot'/'f1_plot'. We treat them separately below.
_CURVE_TAGS = ('py', 'p_plot', 'r_plot', 'f1_plot', 'mr_plot', 'mrfppi_plot')

# pr_data_best fields that are per-class scalar lists ([first_class_value, ...]).
_SCALAR_TAGS = ('mr', 'lamr', 'fppi', 'p', 'r', 'f1')


def _ppf_scalar(values, p):
    if not values:
        return 0.0
    if len(values) < 2:
        return float(values[0])
    mean = np.mean(values)
    std  = np.std(values, ddof=1)
    if std == 0:
        return float(mean)
    return float(scipy_norm.ppf(p, mean, std))


def _ppf_curve(trial_curves, p):
    """
    trial_curves : list of N matrices  (num_classes x 1000 pts each)
    Returns       : one matrix of the same shape with p-th percentile values.

    Points where all trials are NaN remain NaN. Points where only one trial
    contributed get the mean back (std=0, so the normal collapses to a delta).
    """
    nc = len(trial_curves[0])
    result = []
    for c in range(nc):
        arr = np.array([trial[c] for trial in trial_curves], dtype=float)  # (N, 1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(arr, axis=0)
            stds  = np.nanstd(arr, axis=0, ddof=1 if arr.shape[0] > 1 else 0)
            # nanstd with ddof=1 on a single non-NaN sample returns NaN — replace
            # with 0 so norm.ppf(p, mean, 0) = mean (no spread to estimate).
            # Points with all-NaN means stay NaN and propagate as expected.
            stds = np.where(np.isnan(stds), 0.0, stds)
            # Where std=0 the distribution is a delta — ppf collapses to mean.
            # Calling scipy with scale=0 triggers "-inf * 0 = NaN" internally.
            ppf  = np.where(stds == 0, means, scipy_norm.ppf(p, means, stds))
        result.append(ppf.tolist())
    return result


def _nan_mask_pr_curve(y_raw, r_list_entry, n_pts):
    """
    Fill the tail of a PR curve with NaN where recall never reached.
    Mirrors the logic in variance_compare_tab.py.
    """
    y = list(y_raw)
    if r_list_entry:
        px_grid = np.linspace(0, 1, n_pts)
        max_r = max(r_list_entry) if r_list_entry else 1.0
        idx_max = next((i for i, x in enumerate(px_grid) if x > max_r), n_pts - 1)
        y = y[:idx_max] + [np.nan] * (n_pts - idx_max)
    return y


class PercentileManager:
    """
    Compute and temporarily inject percentile-based synthetic data into the
    DataSetHandler so that existing plot code works without modification.

    Usage:
        mgr = PercentileManager()
        keys = mgr.inject_groups(groups, dataset_handler, percentile, class_key)
        # ... render plots / CSV with those keys ...
        mgr.cleanup(dataset_handler)
    """

    def __init__(self):
        self._injected_keys = []

    def inject_groups(self, groups, dataset_handler, percentile, class_key=None):
        """
        groups : {group_name: [key1, key2, …]}  — all checked groups with their trial keys
        Returns list of synthetic keys injected into dataset_handler.
        """
        # Remove any stale synthetic data from a previous call
        self.cleanup(dataset_handler)

        p = percentile / 100.0
        for group, keys in groups.items():
            synth_data, synth_info = self._compute(group, keys, dataset_handler, p, class_key)
            if synth_data is None:
                continue
            synth_key = f"__p{percentile}_{group}"
            dataset_handler.inject(synth_key, synth_data, synth_info)
            self._injected_keys.append(synth_key)

        return list(self._injected_keys)

    def cleanup(self, dataset_handler):
        for k in self._injected_keys:
            dataset_handler.eject(k)
        self._injected_keys = []

    # ------------------------------------------------------------------
    def _compute(self, group, group_keys, dataset_handler, p, class_key):
        reference   = None
        val_scalars = {}   # metric -> [float, …]
        pr_scalars  = {}   # 'mr'/'lamr'/'fppi' -> [float, …]
        pr_curves   = {}   # tag -> list-of-N trial matrices

        for key in group_keys:
            data = dataset_handler[key]
            if not data or 'validation_best' not in data:
                continue
            if reference is None:
                reference = data

            # ── Scalars from validation_best ─────────────────────────
            cls = class_key or 'all'
            val_classes = data['validation_best'].get('data', {})
            if cls not in val_classes:
                cls = next(iter(val_classes), None)
            if cls:
                for metric, val in val_classes[cls].items():
                    if isinstance(val, (int, float)):
                        val_scalars.setdefault(metric, []).append(float(val))

            # ── Scalars from pr_data_best (per-class lists, take [0]) ──
            pr = data.get('pr_data_best', {})
            for metric in _SCALAR_TAGS:
                raw = pr.get(metric)
                if raw and isinstance(raw[0], (int, float)):
                    pr_scalars.setdefault(metric, []).append(float(raw[0]))

            # ── Curves from pr_data_best ──────────────────────────────
            for tag in _CURVE_TAGS:
                curves = pr.get(tag)
                if not curves or not isinstance(curves[0], (list, np.ndarray)):
                    continue

                n_pts = len(curves[0])

                if tag == 'py':
                    # Mask where recall didn't reach (same logic as variance_compare_tab)
                    r_list = pr.get('r_plot') or []
                    masked = []
                    for i, y in enumerate(curves):
                        r_entry = r_list[i] if i < len(r_list) else []
                        masked.append(_nan_mask_pr_curve(y, r_entry, n_pts))
                    pr_curves.setdefault(tag, []).append(masked)
                else:
                    pr_curves.setdefault(tag, []).append([list(c) for c in curves])

        if reference is None:
            log(f"[PercentileManager] No valid data for group '{group}'", bcolors.WARNING)
            return None, None

        # ── Build synthetic validation class data ─────────────────────
        cls = class_key or 'all'
        val_classes = reference['validation_best'].get('data', {})
        if cls not in val_classes:
            cls = next(iter(val_classes), 'all')
        synth_class = {m: _ppf_scalar(v, p) for m, v in val_scalars.items()}

        # ── Build synthetic pr_data (scalars + curves) ────────────────
        synth_pr = dict(reference.get('pr_data_best', {}))  # copy non-computed fields
        for metric in _SCALAR_TAGS:
            if metric in pr_scalars:
                synth_pr[metric] = [_ppf_scalar(pr_scalars[metric], p)]
        for tag, trial_list in pr_curves.items():
            if trial_list:
                synth_pr[tag] = _ppf_curve(trial_list, p)

        # train_data: copy reference and blank out fields that have no percentile meaning
        # (a "best epoch index" of a fitted normal distribution is not meaningful).
        synth_train = dict(reference['train_data'])
        synth_train['epoch_best_fit_index'] = '-'

        synth_data = {
            'validation_best': {**reference['validation_best'], 'data': {cls: synth_class}},
            'pr_data_best':    synth_pr,
            'train_data':      synth_train,
            'n_images':        reference['n_images'],
            'pretrained':      reference['pretrained'],
            'n_classes':       reference.get('n_classes', 0),
            'dataset_tag':     reference.get('dataset_tag', ''),
            'device_type':     reference.get('device_type', ''),
            'deterministic':   reference.get('deterministic', ''),
            'batch':           reference.get('batch', ''),
        }

        ref_info  = dataset_handler.getInfo().get(group_keys[0], {})
        synth_info = {**ref_info,
                      'key':   f'__p{int(p*100)}_{group}',
                      'label': ref_info.get('label', group)}

        return synth_data, synth_info
