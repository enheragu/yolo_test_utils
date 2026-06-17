#!/usr/bin/env bash
#
# End-to-end rerun of contribution evaluation + analyses + HTML report.
#
# Order per dataset:
#   1. pipeline.py --condition all   (cache-idempotent; recomputes if
#      PROXY_VERSION / CALIBRATION_*_VERSION bumped, else fast)
#   2. For each concrete condition:
#        analysis_pca.py         (no CR, then with CR)
#        analysis_regression.py  (no CR, then with CR)
#        analysis_equalization.py
#        build_report_html.py    (per-condition HTML)
#
# Edit the CONFIG block below and run:
#   bash scripts/run_all_analysis.sh

set -euo pipefail

# ============================================================================
# CONFIG — edit these before running
# ============================================================================

# Datasets to process in order.
DATASETS=(kaist llvip)
# Runtime preset for pipeline.py: test (10%) | fast (50%) | balanced (75%) | quality (100%)
PRESET="test"
# Equalization slice used by regression + HTML report.
# One of: no_equalization | rgb_equalization | th_equalization | rgb_th_equalization
EQUALIZATION="no_equalization"
# Target reducer for regression and equalization: mean | median | p90 | best
TARGET_REDUCER="best"
# Detection metrics for the equalization analysis.
EQUALIZATION_METRICS=(P R mAP50 mAP50-95)

# Fusion methods to evaluate.  Leave empty to use pipeline defaults.
METHODS=(
    wavelet wavelet_max
    pca
    curvelet curvelet_max
    vt vths_v2
    sobel_weighted superpixel
    ssim_v2
    visible lwir
    hsvt rgbt_v2
)

# Python interpreter.  Override with PYTHON=/path/to/python when needed.
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    else
        echo "[toolchain] no Python interpreter found in PATH; set PYTHON explicitly" >&2
        return 1 2>/dev/null || exit 1
    fi
fi

# Where to write the tee'd run log (timestamped).  Set to empty to disable.
LOG_DIR="${HOME}/.cache/eeha_review_fusion_contribution/logs"

# ============================================================================
# End of CONFIG
# ============================================================================

HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PKG_DIR="$(cd -P "${HERE}/.." >/dev/null 2>&1 && pwd)"   # review_contribution/
SRC_DIR="$(cd -P "${PKG_DIR}/../.." >/dev/null 2>&1 && pwd)"  # src/

# Mirror stdout + stderr to a timestamped log file so partial runs can be
# diagnosed after the fact.  Exit codes of commands are preserved because the
# process substitution only forks a background ``tee``.
if [[ -n "${LOG_DIR}" ]]; then
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "[toolchain] logging to ${LOG_FILE}"
fi

run_py() {
    # All modules are invoked with cwd=src/ so relative imports resolve.
    (cd "${SRC_DIR}" && "${PYTHON}" -m "$@")
}

conditions_for() {
    # LLVIP only has night; KAIST has day and night.
    case "$1" in
        llvip) echo "night" ;;
        kaist) echo "day night" ;;
        *)     echo "day night" ;;
    esac
}

# Pipeline expects --methods as a list of positional values after the flag.
# Build the optional arg array once.
if [[ "${#METHODS[@]}" -gt 0 ]]; then
    METHODS_ARG=(--methods "${METHODS[@]}")
else
    METHODS_ARG=()
fi

echo "[toolchain] CONFIG:"
echo "  datasets        = ${DATASETS[*]}"
echo "  preset          = ${PRESET}"
echo "  equalization    = ${EQUALIZATION}"
echo "  target_reducer  = ${TARGET_REDUCER}"
echo "  equalization_metrics = ${EQUALIZATION_METRICS[*]}"
echo "  methods (${#METHODS[@]}) = ${METHODS[*]:-<defaults>}"
echo "  python          = ${PYTHON}"
echo

for dataset in "${DATASETS[@]}"; do
    echo "========================================"
    echo "[toolchain] dataset=${dataset}"
    echo "========================================"

    for condition in $(conditions_for "${dataset}"); do
        tag="${dataset}-${condition}"
        echo "----------------------------------------"
        echo "[toolchain] ${tag}"
        echo "----------------------------------------"

        # echo "[toolchain] pipeline (condition=${condition}, preset=${PRESET})"
        # run_py Dataset_review.review_contribution.pipeline \
        #     --dataset "${dataset}" --condition "${condition}" --preset "${PRESET}" \
        #     "${METHODS_ARG[@]}"

        # echo "[toolchain] pca (${tag})"
        # run_py Dataset_review.review_contribution.analysis_pca \
        #     --dataset "${dataset}" --condition "${condition}"

        # echo "[toolchain] pca + channel_redundancy (${tag})"
        # run_py Dataset_review.review_contribution.analysis_pca \
        #     --dataset "${dataset}" --condition "${condition}" \
        #     --include-channel-redundancy

        # echo "[toolchain] regression (${tag})"
        # run_py Dataset_review.review_contribution.analysis_regression \
        #     --dataset "${dataset}" --condition "${condition}" \
        #     --equalization "${EQUALIZATION}" --target-reducer "${TARGET_REDUCER}"

        # echo "[toolchain] regression + CR (${tag})"
        # run_py Dataset_review.review_contribution.analysis_regression \
        #     --dataset "${dataset}" --condition "${condition}" \
        #     --equalization "${EQUALIZATION}" --target-reducer "${TARGET_REDUCER}" --with-cr

        echo "[toolchain] equalization (${tag})"
        for metric in "${EQUALIZATION_METRICS[@]}"; do
            run_py Dataset_review.review_contribution.analysis_equalization \
                --dataset "${dataset}" --condition "${condition}" \
                --metric "${metric}" --target-reducer "${TARGET_REDUCER}"
        done

        # echo "[toolchain] html report (${tag})"
        # run_py Dataset_review.review_contribution.build_report_html \
        #     --dataset "${dataset}" --condition "${condition}" \
        #     --equalization "${EQUALIZATION}" \
        #     --target-reducer "${TARGET_REDUCER}"
    done
done

# # Cross-slice artifacts and index — must run AFTER all per-slice
# # `*_methods_overview.csv` files exist.
# echo "========================================"
# echo "[toolchain] cross-slice analyses"
# echo "========================================"
# INDEX_SLICES=()
# LATENT2D_SLICES=()
# for dataset in "${DATASETS[@]}"; do
#     for condition in $(conditions_for "${dataset}"); do
#         INDEX_SLICES+=(--slice "${dataset}:${condition}")
#         LATENT2D_SLICES+=("${dataset}:${condition}")
#     done
# done

# # Decode the overview CSV equalization tag into the (eq_vis, eq_th) pair the
# # analysis_latent2d filter expects.  The orchestrator uses the overview-tag
# # convention; the analyser uses the original column values.
# case "${EQUALIZATION}" in
#     rgb_equalization)    EQ_VIS=clahe; EQ_TH=none  ;;
#     th_equalization)     EQ_VIS=none;  EQ_TH=clahe ;;
#     rgb_th_equalization) EQ_VIS=clahe; EQ_TH=clahe ;;
#     *)                   EQ_VIS=none;  EQ_TH=none  ;;
# esac

# echo "[toolchain] latent2d (joint latent_z × channel_redundancy)"
# run_py Dataset_review.review_contribution.analysis_latent2d \
#     --slices "${LATENT2D_SLICES[@]}" \
#     --equalization-vis "${EQ_VIS}" --equalization-th "${EQ_TH}" \
#     --reducer "${TARGET_REDUCER}"

# echo "[toolchain] building comparison index"
# run_py Dataset_review.review_contribution.build_comparison_html \
#     "${INDEX_SLICES[@]}" \
#     --equalization "${EQUALIZATION}" \
#     --target-reducer "${TARGET_REDUCER}"

echo "========================================"
echo "[toolchain] done"
echo "========================================"
