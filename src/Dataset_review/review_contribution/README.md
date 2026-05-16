ReviewContribution
==================

RGB/LWIR contribution analysis pipeline. Places each fused method on a latent
axis of visible vs thermal information content.

- Primary score: `latent_z` ∈ `[0, 1]` (`0` = visible endpoint, `1` = thermal
  endpoint).
- Diagnostic scores: `cont_vis` (weighted), `cont_vis_raw` (plain mean), and
  six proxies (`reg`, `mi`, `ssim`, `grad`, `spectral`, `freq`) explain *why*
  a method lands at a given `latent_z`.
- Scope: this is an information-content attribution metric, not a detector
  quality metric (mAP/recall may differ).
- Practical caveat: the gradient proxy is structural and can favor visible in
  scenes with richer fine texture even when LWIR is semantically stronger
  (hot-target detection). Frequency and spectral proxies complement it.

Structure
---------

- `pipeline.py`: CLI entrypoint, orchestration, report output.
- `contribution.py`: Dataset traversal, method execution, parallelization.
- `calibration.py`: Synthetic-mixture calibration and sample caching.
- `evaluation.py`: Core metrics, gradient proxies, calibration mapping.
- `settings.py`: Presets (`test`, `fast`, `balanced`, `quality`) and advanced
  knobs.  The `ENABLED_PROXIES` tuple controls which proxies enter
  calibration / IVW / best-fit selection — comment out a line to drop a
  proxy from the aggregate (the per-image cache stays full so re-enabling
  is free).  The active subset is hashed into the calibration cache key, so
  toggling auto-invalidates only what depends on the proxy set.
- `analysis_pca.py`: PCA over the 6 raw proxies (asks how many real dimensions).
- `analysis_regression.py`: PC scores → training mAP regression.
- `analysis_latent2d.py`: Cross-dataset `latent_z × channel_redundancy` map.
- `__init__.py`: Package entrypoint helper.

Quick Launch
------------

```bash
cd /home/arvc/eeha/yolo_test_utils
PYTHONPATH=src python3 -m Dataset_review.review_contribution.pipeline \
    --dataset llvip --condition night --preset test \
    --methods wavelet wavelet_max pca curvelet_max curvelet vt fa \
              sobel_weighted superpixel vths_v2 ssim_v2 visible lwir hsvt rgbt_v2
```

The CLI is intentionally minimal (dataset / condition / preset / methods / cache).
Advanced tuning (workers, split, alpha steps, equalization variants, chunk
size, `max_images`, `subsample_ratio`) lives in `settings.py`. Sample selection
is deterministic and stratified across top-level dataset folders, seeded by
`split_seed`.

### Presets

| Preset | Subsample | Alpha steps | Variants | Use case |
|--------|-----------|-------------|----------|----------|
| `test` | 10% | 5 | 1 (none) | Quick validation |
| `fast` | 50% | 7 | 1 (none) | Fast eval on many methods |
| `balanced` (default) | 75% | 11 | 4 (none, th, rgb, rgb_th) | Standard eval with equalization analysis |
| `quality` | 100% | 15 | 4 (none, th, rgb, rgb_th) | Exhaustive coverage |

`balanced` and `quality` run 4 independent calibration/evaluation variants
(equalization regimes: none / th CLAHE / rgb CLAHE / both), each fit from the
same synthetic mixtures but evaluated under its own preprocessing. Results
appear side-by-side via `eq_vis` / `eq_th` columns.

Method Overview Columns
-----------------------

The overview table mixes final scores and proxy diagnostics:

| Column | Meaning |
|--------|---------|
| `eq_vis` / `eq_th` | Equalization used (`none` or `clahe`) |
| `cont_vis` | Weighted visible contribution (disagreement-aware, 6 proxies) |
| `cont_vis_raw` | Plain unweighted mean of 6 proxies (auditability) |
| `reg` | Per-channel NNLS visible share |
| `mi` | Per-channel unique-MI visible share |
| `ssim` | Multichannel SSIM-based visible estimate |
| `grad` | Combined gradient (50% magnitude + 50% orientation) |
| `spectral` | Inter-channel independence estimate |
| `freq` | FFT magnitude-spectrum correlation |
| `latent_z` | Calibrated thermal fraction (global IVW) |
| `latent_z_αdep` | Same, using α-dependent IVW (see below) |
| `contrib_std` | Proxy dispersion (disagreement) |
| `contrib_confidence` | Consensus confidence used by weighted aggregator |
| `ch_redund` | Mean \|corr\| between output channels (1 = duplicates, ~0.8 ≈ natural RGB) |
| `thermal_share_raw` | `1 − cont_vis_raw/100` — uncalibrated thermal share. Portable cross-dataset. |
| `thermal_share_reg` | `1 − reg/100` — NNLS thermal share. Most calibration-portable single proxy. |
| `calibration_lift` | `latent_z − thermal_share_raw` — magnitude of saturation correction PAVA applies for this method. |

Proxy Formulas
--------------

Let $V$ be the visible reference, $T$ the LWIR reference, and $F$ the fused
image (all normalized to $[0, 1]$). Subscript $c$ denotes per-channel computation.

### Per-channel regression (`reg`)

For each channel $c$:

$$
F_c \approx a_c V_c + b_c T_c,\quad a_c, b_c \ge 0,
\qquad
\mathrm{reg}_c = 100 \cdot \frac{a_c}{a_c + b_c},
\qquad
\mathrm{reg} = \tfrac{1}{3} \sum_c \mathrm{reg}_c
$$

**Design decision**: per-channel (not flattened) to avoid structural bias —
flattening gives visible (3 independent spectral bands) a richer subspace than
LWIR (1 band replicated to 3 channels).

**Cross-channel blind spot**: each output channel is regressed against *only*
its homologous visible channel ($V_c$).  Methods that mix across visible
channels (PCA / FA decompose by covariance, hsvt converts to HSV) leave the
non-homologous visible content unmodelled — the residual is absorbed by
$b_c$, inflating the apparent thermal share.  `reg` therefore underestimates
the true visible contribution of cross-channel methods; `mi`, `spectral` and
`freq` capture them better but are less calibration-portable across datasets.
Read `reg` as a *structural* metric (linear amplitude composition), not as a
universal "thermal share".

### Per-channel unique-MI (`mi`)

For each channel $c$, discount cross-source MI to isolate unique contribution:

$$
u_{V,c} = \max(\mathrm{MI}(V_c, F_c) - \rho_c\,\mathrm{MI}(T_c, F_c), 0),
\quad
u_{T,c} = \max(\mathrm{MI}(T_c, F_c) - \rho_c\,\mathrm{MI}(V_c, F_c), 0),
$$

$$
\rho_c = |\mathrm{corr}(V_c, T_c)|,
\quad
\mathrm{mi}_c = 100 \cdot \frac{u_{V,c}}{u_{V,c} + u_{T,c}},
\quad
\mathrm{mi} = \tfrac{1}{3} \sum_c \mathrm{mi}_c
$$

**Design decision**: same per-channel rationale as `reg`. Spatial subsampling
(50k pixels, seed 42) keeps cost bounded; with 128-bin joint histograms that
gives <1% estimation error vs full-image MI.

### Multichannel SSIM (`ssim`)

$$
\Delta_{ssim} = \mathrm{SSIM}(V, F) - \mathrm{SSIM}(T, F),
\qquad
\mathrm{ssim} = \frac{100}{1 + e^{-5\Delta_{ssim}}}
$$

SSIM is multichannel (scikit-image `channel_axis`), averaging per-channel
structural similarity internally, so it does not suffer flatten bias.

### Combined gradient (`grad`)

Magnitude and orientation, anchored per image pair to endpoints:

$$
\Delta_{mag}(F) = \mathrm{corr}(\|\nabla V\|, \|\nabla F\|) - \mathrm{corr}(\|\nabla T\|, \|\nabla F\|),
$$

$$
\Delta_{ori}(F) = A(V, F; w_F) - A(T, F; w_F),
\qquad
\mathrm{grad} = \tfrac{1}{2}(\mathrm{grad\_mag} + \mathrm{grad\_ori})
$$

**Design decision**: magnitude and orientation were previously two separate
proxies (40% of the aggregate, biasing toward fine-texture modalities). Merged
into one proxy (1/6 weight) so structural signal is preserved without
dominating. Still leans visible when visible has richer micro-edges; this is a
real signal, not a bug.

### Inter-channel independence (`spectral`)

$$
C_X = \text{pairwise Pearson correlation matrix of channels of } X,
$$

$$
\mathrm{spectral} = 100 \cdot \frac{\|C_F - C_T\|_F}{\|C_F - C_V\|_F + \|C_F - C_T\|_F}
$$

**Design decision**: measures whether fused channels carry independent info
(like visible's 3 spectral bands) or redundant info (like LWIR replicated).
Permutation-invariant. Works for true-color, false-color, and descriptor
outputs (PCA/FA).

### Frequency (`freq`)

$$
M_X = \log(1 + |\mathrm{FFT}_{\text{shifted}}(X_{\text{gray}})|),
\qquad
\Delta_{freq}(F) = \mathrm{corr}(M_V, M_F) - \mathrm{corr}(M_T, M_F)
$$

Anchored per pair:

$$
\mathrm{freq} = \mathrm{clip}_{[0, 100]}\!
\left( 100 \cdot \frac{\Delta_{freq}(F) - \Delta_{freq}^{T}}{\Delta_{freq}^{V} - \Delta_{freq}^{T}} \right)
$$

**Design decision**: LWIR concentrates energy in low frequencies, visible has
richer high-frequency content. Endpoint anchoring ensures exact 0 / 100 at the
endpoints; without it V/T FFT spectra share enough structure to compress the
range. Grayscale, permutation-invariant.

### Raw and weighted aggregates

$$
\mathrm{cont\_vis\_raw} = \tfrac{1}{6}(\mathrm{reg} + \mathrm{mi} + \mathrm{ssim}
+ \mathrm{grad} + \mathrm{spectral} + \mathrm{freq})
$$

Weighted `cont_vis` uses robust MAD weights and shrinks toward 50 when proxies
disagree. Let $p_i$ be proxy scores and $\tilde{p}$ their median:

$$
z_i = \frac{|p_i - \tilde{p}|}{1.4826\,\mathrm{MAD}(p) + \epsilon},
\qquad
w_i = \mathrm{clip}_{[0.05, 1]}\!\left(\frac{1}{1 + z_i^2}\right),
$$

$$
\hat{p}_{\mathrm{rob}} = \frac{\sum_i w_i p_i}{\sum_i w_i},
\qquad
c = \mathrm{clip}_{[0,1]}\!\left(1 - \frac{\mathrm{std}(p)}{35}\right),
$$

$$
\mathrm{cont\_vis} = c\,\hat{p}_{\mathrm{rob}} + (1 - c)\cdot 50,
\qquad
\mathrm{contrib\_confidence} = 100\,c
$$

$\mathrm{contrib\_std} = \mathrm{std}(p)$ reports dispersion across proxies.

### Channel redundancy (`ch_redund`, v18)

$$
\mathrm{ch\_redund} = \tfrac{1}{3}\big(
|\mathrm{corr}(F_0, F_1)| + |\mathrm{corr}(F_1, F_2)| + |\mathrm{corr}(F_0, F_2)|
\big)
$$

Permutation-invariant, computed once per image. Complements `latent_z`:

- `latent_z` high + `ch_redund ≈ 1` → thermal dominance via **channel duplication**
  (e.g. replicated T,T,T).
- `latent_z` high + `ch_redund ≈ 0.7–0.85` → thermal dominance with **distributed**
  content across distinct channels.
- `latent_z` low + `ch_redund ≈ 0.8` → visible-centric with natural-RGB redundancy.

Since `PROXY_VERSION = v18_cr_reliability_weighted`, `ch_redund` is a single
robust composite score:

$$
\mathrm{ch\_redund} = \sum_k w_k\,r_k,
\qquad
w_k \propto \pi_k\,\rho_k,
\qquad
\sum_k w_k = 1
$$

where:

- $r_{\mathrm{pearson,global}}$: legacy mean $|\mathrm{corr}|$ over channel pairs.
- $r_{\mathrm{spearman,global}}$: rank-correlation counterpart (sampled for speed).
- $r_{\mathrm{pearson,tiles}}$: median local redundancy over a $4\times4$ grid
  (robust to spatial artifacts).
- $r_{\mathrm{effrank}}$: covariance effective-rank mapped to redundancy
  ($1\Rightarrow$ duplicated channels, $3\Rightarrow$ independent channels).
- $\pi_k$: fixed prior importance per component.
- $\rho_k$: per-image reliability estimate (lower component dispersion
  $\Rightarrow$ higher reliability).

Design intent: keep one metric only (no v1/v2 split) while preserving
interpretability through component diagnostics and effective weights.

Calibration note: unlike `cont_vis -> latent_z`, there is no synthetic
monotonic ground-truth axis for redundancy, so we do not run a PAVA/IVW
calibration stage for `ch_redund`. Robustness is handled via the
reliability-weighted component fusion above.

Computational cost: low relative to existing proxies. The added operations are
small-channel ($3\times3$) covariance/eigendecomposition and a bounded-sample
Spearman computation (max 25k points), which are negligible versus MI, SSIM,
and gradient/FFT proxy steps already computed per image.

### Expected redundancy by construction

The following table summarises the expected `ch_redund` for each fusion method
based on structural analysis of its channel construction logic. It serves as
ground truth for validating the measured metric.

| Method | Ch 0 | Ch 1 | Ch 2 | Expected `ch_redund` | Why |
|---|---|---|---|---|---|
| wavelet | `(B+T)/2` (wavelet coeffs) | `(G+T)/2` | `(R+T)/2` | **≈ 0.95** | V_{RGB} very correlated → 3 outputs ≈ same blend |
| wavelet_max | same + max-abs detail | same | same | **≈ 0.95** | thermal dominates detail → 3 nearly identical |
| curvelet / curvelet_max | analogous to wavelet in curvelet domain | — | — | **≈ 0.97** | same structural pattern |
| ssim_v2 | `ssim_R·R + (1-ssim_R)·T` | `ssim_G·G + …` | `ssim_B·B + …` | **≈ 0.96** | 3 SSIM maps (R/G/B vs T) are very correlated |
| hsvt | `cv.COLOR_HSV2BGR` after V←mean(V,T) | (H,S from visible) | — | **≈ 0.90** | color transform decorrelates somewhat |
| rgbt_v2 | `B·T` (product) | `G·T` | `R·T` | **≈ 0.97** | multiplicative; B/G/R modulated by same T |
| superpixel | `mask·R + (1-mask)·T` | `mask·G + …` | `mask·B + …` | **≈ 0.94** | single mask from superpixel means → near-identical blend |
| sobel_weighted | `(1-α·∇T)·ch + α·∇T·T` per ch | — | — | **≈ 0.94** | one gradient mask across all channels |
| vt | V (luminance) | T | mean(V,T) | **≈ 0.5** | ch2 = linear combo of ch0,ch1 → moderate |
| vths_v2 | V (luminance) | T | HS packed (4b H + 4b S) | **≈ 0.13** | ch2 orthogonal to luminance |
| pca | PC₁ | PC₂ | PC₃ | **≈ 0** | orthogonal by construction |
| fa | FA₁ | FA₂ | FA₃ | **≈ 0.1–0.3** | nearly orthogonal; FA retains some covariance (noise model) |

**Design decision**: methods with `ch_redund > 0.9` achieve high `latent_z`
by replicating thermal across channels. Methods with similar `latent_z` but
lower redundancy distribute thermal content across structurally distinct
channels — this is more informative for a downstream detector that benefits
from 3 independent feature planes.

Calibrated Thermal Axis (`latent_z`)
------------------------------------

Since v10 the pipeline fits **one monotonic calibration curve per proxy** (PAVA
isotonic regression on per-α-group medians). Raw proxy values map through their
own curves to visible fractions $\hat{v}_p \in [0, 1]$. The six calibrated
fractions are combined by **Inverse-Variance Weighting (IVW)**:

$$
\hat{v} = \frac{\sum_p w_p\,\hat{v}_p(p)}{\sum_p w_p},
\qquad
w_p = \frac{1}{\sigma_p + \epsilon},
\qquad
z = \mathrm{latent\_z} = 1 - \hat{v}
$$

$\sigma_p$ is the proxy's **calibrated intra-group dispersion**: calibration
samples are pushed through the proxy's own curve, and the std of each α-group's
distribution is measured. Low $\sigma_p$ → proxy reacts consistently → higher
weight.

**Soft IVW (v15)**: weights use $1/\sigma_p$ (inverse-std) instead of
$1/\sigma_p^2$ (inverse-variance). Inverse-variance amplifies moderate σ
differences quadratically — with the current proxy set, `reg` would reach 93%
weight, effectively silencing 5 of 6 proxies. Inverse-std is a deliberate
choice: all proxies have the same between-group signal after calibration
(PAVA normalises to [0, 1]), so the only differentiator is within-group noise;
$1/\sigma$ gives consistent proxies a moderate advantage without the extreme
nonlinearity of $1/\sigma^2$.

After normalisation, weights are capped at `MAX_PROXY_WEIGHT` (default 0.25 ≈
1.5× uniform) and excess redistributed proportionally among uncapped proxies.
A std floor $\epsilon$ prevents degenerate zero-σ proxies from exploding the
weight.

An aggregate calibration on averaged `cont_vis` is still fitted for
backward-compatibility (column `lz_agg` in the calibration table).

### α-dependent IVW (v14)

The global IVW uses one σ_p per proxy (mean over α-groups). This is fine when
σ_p(α) is roughly flat along α, but many proxies have **peaked** or **monotone**
σ shapes (e.g. inverted parabola with peak at α ≈ 0.5). In that case the global
σ̄ over-penalises the proxy at the endpoints and under-penalises it in the middle.

The α-dependent IVW iterates:

1. Seed with the global-IVW estimate α̂₀.
2. For each proxy, interpolate σ_p(α̂) over the stored per-α-group σ curve
   (`group_std_per_alpha` vs `group_alphas_std`).
3. Recompute weights `w_p = 1 / σ_p(α̂)` normalised across proxies (+ cap).
4. Re-combine per-proxy calibrated fractions → new α̂.
5. Repeat until `|Δα̂| < 1e-3` or `max_iter = 4`.

The result is stored as `latent_z_αdep` alongside the global `latent_z`.
Per-image bookkeeping (iterations, convergence flag, α̂ trace, effective vs
global weights per proxy) is persisted for downstream inspection.

**When it helps**: inspect `calibration_sigma_shape`. If two or more proxies
show `σ_max / σ_min > 2` or a clearly picuda/monotone shape, α-dep IVW
redistributes weight meaningfully and `latent_z_αdep` diverges from `latent_z`
for methods in the high-σ regions. If σ(α) is flat, both variants converge to
the same result.

Channel-Concatenation Calibration (Diagnostic)
----------------------------------------------

### Calibration bias and multi-format calibration (v15)

**Fundamental bias**: the primary calibration uses pixel-wise linear blends
`F = (1 - α)·V + α·T_{rgb}`. Proxies that structurally match this model
(e.g. `reg`, which fits `F ≈ a·V + b·T` via NNLS) achieve artificially low σ
and high F-ratio because they are solving the generating model. Real fusion
methods (wavelet decomposition, channel substitution, nonlinear blending) do
NOT produce pixel-wise linear blends. A proxy's calibration-measured σ therefore
reflects its affinity to the calibration format, not its general reliability.

To mitigate this, the pipeline fits **four calibration formats** that exercise
proxies in complementary ways:

| Format | Synthetic | α range | Exercises |
|---|---|---|---|
| **blend** | `(1-α)·V + α·T_{rgb}` pixel-wise | [0, 1] continuous | `reg` (linear model), `mi`, `ssim` |
| **concat** | Channel substitution + intra-channel mix `[R,G,T]`, `[R,mix,T]`, … | {0, ⅙, ⅓, ½, ⅔, ⅚, 1} | `spectral` (channel independence), `grad` |
| **freq_blend** | Wavelet coefficient blend + reconstruct | [0, 1] continuous | `freq`, `spectral` (frequency-domain mixing) |
| **nonlinear** | `V^{1-α} · T_{rgb}^α` (geometric mean) | [0, 1] continuous | `mi`, `ssim`, `grad` (nonlinear interaction) |

**R vs G vs B are not perceptually weighted in our metrics, but they are
structurally distinct.** `[R, R, T]` has a duplicated visible channel plus
thermal, which `spectral` picks up as LWIR-like redundancy; `[R, G, T]` has two
independent visibles and preserves more visible info even though both have
α = 1/3. The concat calibration uses patterns where listed visible channels
are distinct, plus intra-channel 50/50 mixes (`m` = `(V_ch + T)/2`) to fill
gaps between the pure-substitution points:

```
α = 0:    (R,G,B)                              — pure visible
α = 1/6:  (R,G,m_B), (R,m_G,B), (m_R,G,B)     — one channel half-mixed
α = 1/3:  (R,G,T), (R,T,B), (T,G,B)           — one channel replaced
α = 1/2:  (R,m_G,T), (m_R,G,T), (m_R,T,B)     — one replaced + one mixed
α = 2/3:  (R,T,T), (T,G,T), (T,T,B)           — two channels replaced
α = 5/6:  (m_R,T,T), (T,m_G,T), (T,T,m_B)     — two replaced + one mixed
α = 1:    (T,T,T)                              — pure thermal
```

The intermediate mix points (`m`) stress the proxies differently from pure
substitution: `grad_combined` and `ssim` respond to partial blending within a
channel differently than to complete replacement. Each α group averages over
its permutations.

#### Proxy weighting across formats

Per-format proxy weights (soft IVW: 1/σ + cap) are **averaged across the four
formats** before normalisation. A proxy that dominates in blend (e.g. `reg`)
but is mediocre in freq_blend and concat ends up with a moderate combined
weight. This eliminates the single-format bias without discarding consistency
information.

#### Scale correction: best-fit calibration per method

For mapping raw proxy values → latent_z, the pipeline selects the calibration
format that **best fits each evaluated method**. Selection criterion: apply
each format's per-proxy curves to the method's raw proxy values; the format
producing the **lowest inter-proxy disagreement** (`contrib_std` of the
calibrated fractions) is the best match. Intuition: if a calibration's curves
"understand" how the method mixes, all six proxies should converge after
calibration. High disagreement signals a mismatch between format and method.

This separates two roles that were previously conflated:
1. **Proxy weighting** — multi-format average, no single format dominates.
2. **Scale correction** — per-method best-fit, uses the most appropriate
   reference for that fusion type.  Since `f2_multi_format_avg_fix`, best-fit
   selection minimises the **weighted** inter-proxy std (using the same
   averaged proxy weights the aggregator applies), so the format chosen for
   scale correction is the one most consistent with the actual aggregation —
   not just the unweighted mean.

Calibration Protocol
--------------------

The pipeline fits `z` in two stages: (1) compute the six raw proxies per image,
(2) calibrate to the latent axis via synthetic mixtures. The calibration is
fitted **once per protocol** from the synthetic subset, then reused for every
evaluated image sharing that protocol. All four formats are generated from the
same image subset and share the same protocol hash.

Protocol fields (if any changes, a fresh calibration is generated and cached
separately): dataset, root path, condition, split settings (ratio/images + seed),
`alpha_steps`, equalization variant.

Correlation Diagnostics
-----------------------

Each image emits **both** Pearson (`cc_vis`, `cc_lw`) and Spearman (`sp_vis`,
`sp_lw`) correlations between fused and each source. Pearson is the primary
coefficient — calibration handles non-linearity per proxy, so by the time
values reach the combination they live in a roughly linear calibrated-fraction
space. Spearman is emitted as a rank-based diagnostic; large Pearson/Spearman
disagreement flags non-linear or outlier-driven behaviour for that method.

Training Results Correspondence
-------------------------------

`training_results_check.py` encapsulates CSV parsing (slicing by fusion type,
condition, equalization) so format changes only require editing that module.
Fusion method names in the pipeline (`sobel_weighted`, `ssim_v2`, `rgbt_v2`,
`curvelet_max`, `wavelet_max`) are mapped to shorter CSV tags (`sobel`, `ssim`,
`rgbt`, `curvelet`, `wavelet`) via `_METHOD_NAME_TO_CSV_TYPE` — extend it when
adding non-matching names.

With only ~5 runs per method, a classical boxplot is noisy; we use a strip-plot
style (individual points + marker) which is more honest for small n. The
method-level summary for training-vs-latent plots uses the **P90 of the fitted
normal** (μ + 1.28·σ) per method, consistent with the user's other work.

PCA and Regression Diagnostics
------------------------------

**Motivation.** `latent_z` collapses the six proxies into a single dimension by
IVW-weighted aggregation.  That is *useful* but not necessarily *complete*: if
two methods share the same `latent_z` but differ in, say, how redundant visible
and thermal channels are, the 1-D summary loses that distinction.  PCA on the
proxy matrix is the cheapest way to ask: *how many latent dimensions do the
proxies actually span?*

**Artifacts.** `analysis_pca.py` reads the evaluation cache (per method × image
× proxy), z-scores the feature table, runs SVD-based PCA and emits:

- `pca_explained_variance_<tag>.csv` — eigenvalues, % explained, cumulative, and
  the **broken-stick** reference.  Broken-stick is the expected eigenvalue of
  each PC under random unit-variance data; a component above the line is more
  informative than noise.
- `pca_loadings_<tag>.csv` + heatmap — how each proxy weights each PC.
- `pca_scores_per_entry_<tag>.csv` — per-(method, image) projection.
- `pca_scores_per_method_<tag>.csv` — per-method mean/std of PC scores.
- `pca_pc1_pc2_scatter_<tag>.png` — visual of method layout on the top-2 axes.

Pass `--include-channel-redundancy --overview-csv <methods_overview>` to inject
the per-method mean `channel_redundancy` as a 7th feature (until the cache is
regenerated with `PROXY_VERSION` bumped, the cache does not carry per-image
`channel_redundancy`).

**Diagnostic criteria.**

- **Kaiser rule** — with z-scored inputs, any PC with eigenvalue > 1 captures
  more variance than a single original feature.
- **Broken-stick** — stricter null model; surviving a broken-stick threshold
  is stronger evidence the PC is real.
- We keep the union of both in the CSVs and flag which PCs pass which.

**Regression against detection.** `analysis_regression.py` answers the follow-up
question: *do the PC scores predict training mAP?*  Pipeline:

1. Load per-method PC scores (`pca_scores_per_method_<tag>.csv`).
2. Load detection metrics via `training_results_check.load_training_results` +
   `build_dataset_metrics` (same module used by the correspondence plots, so
   `Class=person` and `Group Key` filtering are shared).
3. Inner-join on `method`.  Per-method detection summary is either `mean` or
   `p90` (μ + 1.28·σ) per the `--target-reducer` flag.
4. Fit nested OLS: `target ~ PC1`, `target ~ PC1 + PC2`, `target ~ PC1 + PC2 + PC3`.
5. Report R², adjusted R², ΔR² and a **nested F-test** (`F = ((RSS_s − RSS_l)/Δdf) /
   (RSS_l/(n−p_l))`) for each added PC.

**Reading the output.** A large ΔR² with non-significant p-value usually just
means n is too small (≤ ~10 methods overlap between PCA cache and training CSV);
treat the point estimate as a signal-suggestion, not a conclusion.  An R² that
jumps only once PC3 is added, when ΔR²(PC2) ≈ 0, is almost always overfitting
at small n.

Runtime and Cache Notes
-----------------------

- Three independent version tags gate three independent caches so you only
  invalidate what actually changed:
  - `PROXY_VERSION` → evaluation cache (per method/image proxies). Bump when
    proxy formulas or `compute_contribution_rgb` output shape change.
    Invalidates everything downstream (samples + fit too).
  - `CALIBRATION_SAMPLES_VERSION` → calibration-samples cache (synthetic
    mixture proxies at `(image, α, sample_type)`). Bump when synthetic
    generators or `_CHANNEL_CONCAT_PATTERNS` change. Does **not** invalidate
    evaluation.
  - `CALIBRATION_FIT_VERSION` → calibration object (PAVA curves + IVW
    weights). Bump when `fit_contribution_calibration`, `_cap_proxy_weights`,
    `_average_proxy_weights_across_formats`, or `_select_best_fit_calibration`
    change. Does **not** invalidate samples or evaluation.
  Current tags: proxy `v16_cr_per_image`, samples `s2_freq_maxabs`,
  fit `f2_multi_format_avg_fix` (bumped from `f1_soft_ivw_cap25` after fixing
  the key mismatch that made `_average_proxy_weights_across_formats` always
  fall back to uniform 1/n; the soft-IVW averaged weights are now the ones
  the aggregator actually uses).
- Calibration has incremental sample caching for `(image, α, sample_type)`
  points; protocol fields drive the cache key.
- **Crash-safe checkpointing**: both calibration and evaluation flush the cache
  every `max(1, min(total_batches // 20, 25))` batches (≈5% of the run, capped
  at 25 between flushes). On `BaseException` (ctrl+C, OOM, crash) partial
  progress is flushed before re-raise. Failed batches do not poison the cache;
  their images are simply recomputed on the next run.

Machine-Readable Artifacts
--------------------------

Every run emits structured dumps alongside PNG plots, so results can be inspected
numerically without re-executing:

- `<tag>_methods_overview.csv` — per-method numeric summary (mean/std/median per
  column, including `latent_z`, `latent_z_alpha_dep`, `channel_redundancy`,
  `ivw_iters`, …).
- `<tag>_calibration_sigma_shape_<variant>.csv` — long-format
  `proxy, alpha, sigma, sigma_mean_global, ivw_weight_global`.
- `<tag>_method_sigma_shape_<variant>.csv` — per-method per-proxy empirical σ
  alongside calibration σ interpolated at the method's `latent_z`.
- `<tag>_ivw_weights_<variant>.csv` — per-(method, proxy) global vs α-dep
  weights and their Δ%.
- `<tag>_ivw_weight_shape_<variant>.csv` — α-dep weight curves backing the
  corresponding plot.
- `<tag>_reg_crosscheck_<variant>.csv` — per-method thermal share from raw
  regression vs `latent_z` (global and α-dep) and `channel_redundancy`.
- `<tag>_report.json` — full per-method summary snapshot (no raw per-image data).
- `<tag>_calibration_<variant>.json` — calibration structure (σ curves, knots,
  nodes).
- `<tag>_ivw_diagnostic_<variant>.txt` — plaintext diagnostic: σ_max/σ_min per
  proxy (justification for α-dep IVW), convergence rate, Δlatent_z per method,
  mean weight redistribution.

Plot Interpretation Guide
-------------------------

All diagnostic plots are written to the report folder as
`<dataset>_<condition>_<plotname>_<variant>.png`.

### `proxy_overview`

Three stacked panels:

1. **Top (grouped bars)**: raw per-proxy `cont_vis` per method. Spots proxy
   disagreement.
2. **Middle (side-by-side bars)**: raw vs calibrated `cont_vis` per method. Net
   effect of calibration (methods at `cont_vis ≈ 75` may drop to `≈ 55` once
   saturating proxies are corrected).
3. **Bottom**: final `latent_z` per method, blue→red colormap (visible→thermal).
   The final ranking.

### `calibration_diagnostic`

Left: blend and concat calibration curves overlaid. Right (one per sample type):
per-α boxplot of `cont_vis` across images. Per-α medians should track the
dashed ideal `(1 - α)·100`; spread reports alpha-slice noise.

### `per_proxy_calibration_curves`

Six panels (one per proxy). Each shows the per-proxy monotonic calibration
(solid), the aggregate curve (dashed gray reference), the ideal linear response
(dashed black), the per-α knots fitted by PAVA (`x` markers), and fusion methods
projected onto that proxy's curve (green stars). Legend label `w=0.xxx σ=0.yyy`
is the IVW weight and calibrated intra-group std.

### `per_proxy_calibration`

Per-proxy boxplot of raw proxy values vs α with ideal line. Complementary to
`per_proxy_calibration_curves`: shows pre-calibration saturation, not the
calibration curve itself.

### `calibration_nodes_overview`

Bar chart, one group per α. Top: each proxy's median raw value at that α.
Bottom: deviation of aggregated `cont_vis` from ideal `(1 - α)·100`; red bars
below zero mean under-reporting of visible.

### `calibration_sigma_shape_<variant>`

Per-proxy σ(α) curves from calibration data. Flat curves → global σ is
adequate; picuda/monotone shapes justify α-dep IVW. Inverted-parabola
(peak at α ≈ 0.5) is typical.

### `per_method_proxy_std_<variant>`

Per-method per-proxy σ of raw `cont_vis` across images. Dashed references =
global calibration σ. Bars above the dashed line → method's proxy dispersion
exceeds what global IVW assumes.

### `method_sigma_shape_<variant>`

Real-methods counterpart of `calibration_sigma_shape`: per-proxy σ vs method's
`latent_z`. Sawtoothy because methods cluster at specific `latent_z` values;
use to confirm/contrast with calibration curves at the levels where methods
actually live.

### `channel_redundancy_<variant>`

Scatter of `channel_redundancy` vs `latent_z` per method. Dashed at ~0.8
(natural-RGB baseline), dotted at 1.0 (full duplication). Upper-right points
= high thermal dominance via channel duplication; high `latent_z` with
redundancy near 0.7-0.85 = distributed thermal content.

### `ivw_weight_shape_<variant>`

Per-proxy normalised IVW weight as a function of α. Solid = α-dependent weights
`w_p(α)/Σ_q w_q(α)`; dashed horizontals = corresponding global weights. Drift
between solid and dashed measures how much α-dep IVW redistributes mass from
that proxy. Companion to `calibration_sigma_shape` in weight space.

### `ivw_global_vs_alphadep_<variant>`

Per-method `latent_z` (global) vs `latent_z_αdep`. Points far from the 1:1
diagonal = methods whose estimate moves under α-dependent reweighting. Diagonal
pileup → global IVW is effectively equivalent to α-dep for the current method
population.

### `reg_crosscheck_<variant>`

Two-panel scatter: X = thermal share from raw per-channel NNLS regression
(`1 − cont_vis_reg/100`), Y = `latent_z` (left panel: global IVW, right:
α-dep). Points near the diagonal confirm that the IVW aggregate tracks the
direct regression; deviations reveal IVW distortion. Companion CSV with
`channel_redundancy` for quick triaging.

### Cross-dataset portability (PAVA)

PAVA calibration is fitted **per dataset** from that dataset's synthetic
mixtures.  An empirical curve-overlap diagnostic on the three default slices
(kaist_day / kaist_night / llvip_night) reports max|Δ| between curves at the
same raw input value:

| Proxy | Max\|Δ\| (KAIST↔LLVIP) | Portability |
|---|---|---|
| `reg`      | ≈ 0.075 | High — the structural V/T linear share is dataset-invariant |
| `ssim`     | ≈ 0.13  | Moderate |
| `mi`       | ≈ 0.20  | Low at KAIST↔LLVIP (high information-content sensitivity) |
| `freq`     | ≈ 0.20  | Low |
| `grad`     | ≈ 0.25  | Low — varies even within KAIST_day↔KAIST_night |
| `spectral` | ≈ 0.29  | Low |

Implication: `latent_z` absolute values are **dataset-relative** — a method's
0.7 in KAIST and 0.7 in LLVIP do not refer to the same V/T mixture, because
the calibration endpoints differ.

**Which metric for which comparison?**  No single column wins all three cases.

| Comparison | Use | Why |
|---|---|---|
| Same method, different datasets | `thermal_share_reg` | The structural cross-channel bias of `reg` is the *same* in every dataset, so it cancels under method-vs-itself comparison.  Most calibration-portable. |
| Different methods, same dataset | `latent_z` | All six proxies + saturation correction + IVW reliability weighting; the aggregate handles the per-method proxy biases that any single column would carry. |
| Different methods, different datasets | No clean comparison | `latent_z` mixes calibration drift across datasets; `thermal_share_reg` mixes per-method bias.  Rank within each dataset and compare *rankings*, not absolute values. |

`thermal_share_raw` (unweighted mean of the 6 raw proxies) sits in between:
reasonable cross-dataset, less biased than `reg` cross-method, but saturates
near the endpoints.  All three columns are exposed in `methods_overview.csv`
and as side-by-side panels in `analysis_latent2d.py`.

### `latent2d_scatter_<eq>_<reducer>_<metric>`

Cross-dataset scatter combining one or more `*_methods_overview.csv` files
along two latent axes.  By default emits **three side-by-side panels** — one
per x-axis variant — to make the calibration vs portable comparison
immediate:

- **x panel 1 — `latent_z`** (PAVA-calibrated, IVW-weighted).  Best
  intra-dataset reading; *not directly comparable across datasets*.
- **x panel 2 — `1 − cont_vis_raw/100`** (uncalibrated mean of 6 proxies).
  Portable cross-dataset; saturates near the endpoints.
- **x panel 3 — `1 − reg/100`** (NNLS thermal share).  Calibration-portable;
  reductive (single proxy).
- **y (all panels) — `channel_redundancy`** (axis 2 — inter-channel mean |corr|).

Visual encoding:
- Marker shape *and* color per `(dataset, condition)` slice — redundant cues
  so the figure reads in B&W and for color-vision-deficient viewers.
- Marker size uniform (the previous `contrib_confidence` size encoding was
  hard to read without a key).
- Small grey "+" markers at the four corners are the
  `visible`/`lwir` anchors per dataset, kept off-legend.
- A shaded box marks the bottom-left quadrant ("visible-dominant +
  distributed channels") — currently empty across all surveyed methods.

`latent_z_alpha_dep` is **not** plotted as a second axis: empirically it
correlates 0.92–0.98 with `latent_z` across the three default slices, so it
adds almost no orthogonal information.  It remains a diagnostic of σ-shape
heterogeneity (see `ivw_global_vs_alphadep`), not a second latent.

Companion artifacts emitted alongside the figure:
- `latent2d_data_<tag>.csv` — per-method joint coordinates including
  `thermal_share_raw`, `thermal_share_reg`, `calibration_lift`, and the
  training metric.
- `latent2d_top3_<tag>.csv` — top-3 methods per slice by mAP, for the
  "where do the winners land?" reading.
- `latent2d_corr_<tag>.csv` — per-slice Pearson r between each thermal-axis
  variant and mAP, plus r(`channel_redundancy`, mAP).  When a method's r
  flips sign between `latent_z` and `thermal_share_raw`, calibration is
  amplifying or hiding the underlying signal.

Run via `python -m Dataset_review.review_contribution.analysis_latent2d`
(see `--help`).  Outputs land under `<reports>/latent2d/`.

### `training_vs_latent_<equalization>`

2×2 grid, one panel per training metric (P, R, mAP50, mAP50-95). X = `latent_z`
per method under that equalization, Y = observed training metric. Each method:
faint blue dots = individual runs, red marker = P90-of-fitted-normal summary,
dashed gray = linear fit. Pearson r and Spearman ρ printed. A monotonic trend
suggests the latent axis captures something useful about the fusion; a flat
cloud means the axis and the detection metric are uncorrelated under the tested
training conditions.
