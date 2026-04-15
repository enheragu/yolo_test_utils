from __future__ import annotations

"""Centralized settings for review contribution pipeline.

Keep CLI short and maintain advanced knobs here.
"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ReviewContributionSettings:
    calibration_images: int | None
    calibration_ratio: float
    split_seed: int
    workers: int
    execution_mode: str
    alpha_steps: int
    task_chunksize: int
    max_images: int | None
    subsample_ratio: float
    equalization_variants: list[str]  # which eq. regimes to generate calibration for


DEFAULT_SETTINGS = ReviewContributionSettings(
    calibration_images=None,
    calibration_ratio=0.25,
    split_seed=42,
    workers=max(1, (os.cpu_count() or 2) // 2),
    execution_mode="process",
    alpha_steps=15,
    task_chunksize=2,
    max_images=None,
    subsample_ratio=0.75,
    equalization_variants=["no_equalization", "th_equalization", "rgb_equalization", "rgb_th_equalization"],
)


PRESET_SETTINGS: dict[str, ReviewContributionSettings] = {
    "test": ReviewContributionSettings(
        calibration_images=None,
        calibration_ratio=0.2,
        split_seed=42,
        workers=max(1, (os.cpu_count() or 3) // 3),
        execution_mode="process",
        alpha_steps=11,
        task_chunksize=4,
        max_images=500,
        subsample_ratio=0.025,
        equalization_variants=["no_equalization"],
    ),
    "fast": ReviewContributionSettings(
        calibration_images=None,
        calibration_ratio=0.2,
        split_seed=42,
        workers=max(1, (os.cpu_count() or 3) // 3),
        execution_mode="process",
        alpha_steps=10,
        task_chunksize=4,
        max_images=None,
        subsample_ratio=0.5,
        equalization_variants=["no_equalization"],
    ),
    "balanced": DEFAULT_SETTINGS,
    "quality": ReviewContributionSettings(
        calibration_images=None,
        calibration_ratio=0.3,
        split_seed=42,
        workers=max(1, (os.cpu_count() or 3) // 3),
        execution_mode="process",
        alpha_steps=20,
        task_chunksize=2,
        max_images=None,
        subsample_ratio=1.0,
        equalization_variants=["no_equalization", "th_equalization", "rgb_equalization", "rgb_th_equalization"],
    ),
}
