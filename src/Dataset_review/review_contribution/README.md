ReviewContribution
==================

This directory encapsulates the RGB/LWIR contribution analysis pipeline and keeps
the original entrypoint compatible.

Objective & Scope
-----------------

Goal: place each fused method on a latent axis of **visible information vs thermal
information** contained in the fused output.

- Primary score: `latent_z` in `[0,1]`. `0` means visible-endpoint behavior and `1` means LWIR-endpoint behavior.
- Diagnostic scores: `cont_vis` (weighted), `cont_vis_raw` (plain mean), and proxy columns (`reg`, `mi`, `ssim`, `grad`, `spectral`, `freq`) explain *why* a method lands at a given `latent_z`.
- Scope: this is an information-content attribution metric, not a direct detector
	quality metric (mAP/recall may differ).
- Practical caveat: gradient proxy is structural; it can favor visible in
	scenes with richer fine texture, even when LWIR is semantically stronger for
	hot-target detection. Frequency and spectral proxies complement it.

Structure
---------

- `pipeline.py`: CLI entrypoint and runtime orchestration (preset selection, cache paths, report output).
- `contribution.py`: Dataset traversal, method execution, parallelization, and per-method aggregation.
- `calibration.py`: Synthetic-mixture calibration building and calibration-sample caching.
- `evaluation.py`: Core contribution metrics, gradient proxies, and calibration mapping utilities.
- `settings.py`: Centralized advanced knobs and presets (`test`, `fast`, `balanced`, `quality`).
- `__init__.py`: Package entrypoint helper.

Quick Launch
------------

Run a fast smoke test with the full method list:

```bash
cd /home/arvc/eeha/yolo_test_utils
PYTHONPATH=src python3 -m Dataset_review.review_contribution.pipeline \
	--dataset llvip \
	--condition night \
	--preset test \
	--methods wavelet wavelet_max pca curvelet_max curvelet vt fa sobel_weighted superpixel vths_v2 ssim_v2 visible lwir hsvt rgbt_v2
```

This keeps the run short (`preset test`) while validating all requested methods.

CLI Philosophy
--------------

The CLI is intentionally short to reduce confusion. Only operational options are
kept as arguments (dataset/condition/preset/methods/cache). Advanced tuning
parameters (workers, split ratio, alpha steps, chunk size, `max_images`, and
equalization variants) live in `settings.py`.

There are only two ways to set the sample size:

- `max_images`: absolute image budget.
- `subsample_ratio`: relative fraction of the available images.

The selection policy is always the same in both cases: deterministic and
stratified across the top-level dataset folders, using `split_seed` for
reproducibility.

If `max_images` is set, it takes precedence. If it is `None`, the pipeline uses
`subsample_ratio`.

### Runtime Presets & Automatic Calibration Variants

Each preset bundles subsample ratio, calibration strategy, and parallelization tuning.
More importantly, **balanced and quality presets automatically generate 4
calibration/evaluation variants** (one per equalization regime), while test/fast
generate only 1.

| Preset | Subsample | Alpha Steps | Variants | Use Case |
|--------|-----------|-------------|----------|----------|
| `test` | 10% | 5 | 1 (none) | Quick validation; check pipeline is working |
| `fast` | 50% | 7 | 1 (none) | Fast eval; good for 50+ methods on medium datasets |
| `balanced` (default) | 75% | 11 | 4 (none, th_equalization, rgb_equalization, rgb_th_equalization) | Standard eval; measure equalization impact |
| `quality` | 100% | 15 | 4 (none, th_equalization, rgb_equalization, rgb_th_equalization) | Exhaustive eval; all images, all regimes |

Choose `--preset test` to iterate quickly, `fast` for large method suites,
`balanced` for publication-ready numbers with equalization analysis, and
`quality` when you want exhaustive coverage.

Equalization Impact via Automatic Calibration Variants
-------------------------------------------------------

When you run with `--preset balanced` or `--preset quality`, the pipeline automatically
generates 4 independent calibration/evaluation variants:

1. **none**: No preprocessing on either image.
2. **th_equalization**: CLAHE applied to LWIR only.
3. **rgb_equalization**: CLAHE applied to visible only.
4. **rgb_th_equalization**: CLAHE applied to both images.

Each variant is fit independently from the same synthetic mixtures, then evaluated
on the same selected image subset and cached.
This lets you compare how equalization affects both the calibration map and the
final method scores, while keeping the sampled images fixed and reproducible.

All variants are reported, and the overview table keeps them side by side via
`eq_vis` and `eq_th` columns.

At runtime, the report prints a short one-line summary per variant:
which variant was used, how many knots it has, and the fitted raw/visible/thermal
range.

The method report is also compact: it shows a single overview table in the
terminal and saves the full per-method summaries into the report folder.

Method Overview Columns
-----------------------

The terminal overview table includes both final and proxy metrics:

- `eq_vis`: visible-side equalization used for that run (`none` or `clahe`)
- `eq_th`: thermal-side equalization used for that run (`none` or `clahe`)
- `cont_vis`: primary visible contribution estimate (disagreement-aware weighted aggregation of 6 proxies)
- `cont_vis_raw`: plain unweighted mean of all 6 proxies, kept for traceability
- `reg`: per-channel **NNLS** (Non-Negative Least Squares) visible estimate. For
  each channel we solve `F_c ≈ a·V_c + b·T_c` with `a,b ≥ 0` and report the
  share attributable to visible: `100·a/(a+b)`. Non-negativity rules out
  physically impossible "negative mixtures".
- `mi`: per-channel unique-mutual-information visible estimate. For each
  channel, **Mutual Information** (a measure of statistical dependence
  between two random variables) of `(V_c, F_c)` and `(T_c, F_c)` is
  computed, then the *unique* contribution of each source is the MI
  discounted by the cross-source MI (so shared information is not double
  counted). The visible share is reported as a percentage.
- `ssim`: multichannel **SSIM** (Structural Similarity Index, which captures
  luminance/contrast/structure similarity, not just pixel differences). We
  compare `SSIM(V,F)` and `SSIM(T,F)` and map their difference to a visible
  percentage via a logistic squashing.
- `grad`: combined gradient proxy (50% gradient magnitude correlation +
  50% gradient orientation agreement). Structural-texture evidence.
- `spectral`: inter-channel independence structure visible estimate.
  Measures whether fused channels carry *independent* spectral information
  (like visible's 3 distinct bands) or *redundant* information (like LWIR
  replicated across 3 channels).
- `freq`: **FFT** (Fast Fourier Transform) magnitude spectrum correlation.
  Captures the energy distribution over frequencies; LWIR concentrates
  energy in low frequencies, visible has richer high-frequency content.
- `latent_z`: calibrated thermal-axis score in `[0, 1]`
- `contrib_std`: dispersion across proxies (proxy disagreement)
- `contrib_confidence`: proxy-consensus confidence used by the weighted aggregator

This makes it easier to detect methods that look similar in the final score but
behave differently across proxy families.

Proxy Formulas (Implementation-Level)
-------------------------------------

Let $V$ be the visible reference, $T$ the LWIR reference, and $F$ the fused image
(all normalized to $[0,1]$ in the metric pipeline). Let subscript $c$ denote
per-channel computation when applicable.

### Per-channel regression proxy (`reg`)

For each channel $c \in \{0, 1, 2\}$:

$$
F_c \approx a_c V_c + b_c T_c,\quad a_c,b_c \ge 0
$$

$$
\mathrm{reg}_c = 100 \cdot \frac{a_c}{a_c+b_c}
$$

$$
\mathrm{reg} = \frac{1}{3}\sum_{c} \mathrm{reg}_c
$$

**Design decision**: Per-channel computation avoids a structural bias where
flattening all channels into one vector gives visible (3 independent spectral
bands at ~470/530/620nm) a richer subspace than LWIR (single ~8-14um band
replicated to 3 identical channels), making NNLS inherently favor visible
regardless of actual content contribution.

### Per-channel unique-MI proxy (`mi`)

For each channel $c$:

$$
u_{V,c} = \max\big(\mathrm{MI}(V_c,F_c) - \rho_c\,\mathrm{MI}(T_c,F_c),\;0\big),
$$

$$
u_{T,c} = \max\big(\mathrm{MI}(T_c,F_c) - \rho_c\,\mathrm{MI}(V_c,F_c),\;0\big),
$$

$$
\rho_c = |\mathrm{corr}(V_c,T_c)|,
\qquad
\mathrm{mi}_c = 100 \cdot \frac{u_{V,c}}{u_{V,c} + u_{T,c}}
$$

$$
\mathrm{mi} = \frac{1}{3}\sum_{c} \mathrm{mi}_c
$$

**Design decision**: Same per-channel rationale as regression. MI is computed
with spatial subsampling (50k pixels, seed=42) for efficiency: with 128-bin
joint histograms (16384 cells), ~20k samples suffice for <1% estimation error
vs full-image computation (see Kraskov et al. convergence bounds for plug-in
MI estimators). 50k provides comfortable margin.

### Multichannel SSIM proxy (`ssim`)

$$
\Delta_{ssim} = \mathrm{SSIM}(V,F) - \mathrm{SSIM}(T,F)
$$

$$
\mathrm{ssim} = \frac{100}{1 + e^{-5\Delta_{ssim}}}
$$

SSIM is computed as multichannel via scikit-image's `channel_axis` parameter
when available. This is inherently per-channel internally (SSIM averages
structural similarity across channels), so it does not suffer from the
flatten bias.

### Combined gradient proxy (`grad`)

Magnitude component:

$$
\Delta_{mag}(F) = \mathrm{corr}(\|\nabla V\|,\|\nabla F\|)
- \mathrm{corr}(\|\nabla T\|,\|\nabla F\|)
$$

Mapped to $[0, 100]$ with visible/thermal anchor endpoints.

Orientation component:

$$
\Delta_{ori}(F) = A(V,F;w_F) - A(T,F;w_F)
$$

Mapped to $[0, 100]$ with the same anchor scheme.

Combined:

$$
\mathrm{grad} = 0.5 \cdot \mathrm{grad\_mag} + 0.5 \cdot \mathrm{grad\_ori}
$$

**Design decision**: Magnitude and orientation were previously separate proxies
contributing 2/5 = 40% of the aggregate score, biasing toward whichever
modality had richer fine-grained texture (typically visible, with its
micro-edges, background patterns, etc.). Merging them into one proxy reduces
their combined weight to 1/6 while retaining both structural signals. The
gradient proxy correctly reports visible dominance when the fused image
preserves visible textures, and thermal dominance otherwise — the fix is about
aggregate weighting, not suppressing the signal.

### Inter-channel independence proxy (`spectral`)

$$
C_X = \text{pairwise Pearson correlation matrix of channels of } X
$$

$$
\mathrm{spectral} = 100 \cdot \frac{\|C_F - C_T\|_F}{\|C_F - C_V\|_F + \|C_F - C_T\|_F}
$$

**Design decision**: This captures spectral-band information preservation
regardless of whether the fused output is true-color or false-color (e.g.
PCA/FA descriptors). It does not assume perceptual color meaning — it measures
whether the fused channels carry independent information (like visible's 3
spectral bands at ~470nm B, ~530nm G, ~620nm R) or redundant information
(like LWIR's single ~8-14um band replicated to 3 channels). Permutation-
invariant by construction: pairwise correlations form an unordered set.

### Frequency proxy (`freq`)

$$
M_X = \log(1 + |\mathrm{FFT}_{\text{shifted}}(X_{\text{gray}})|)
$$

$$
\Delta_{freq}(F) = \mathrm{corr}(M_V, M_F) - \mathrm{corr}(M_T, M_F)
$$

Anchors (same image pair):

$$
\Delta_{freq}^{V} = \Delta_{freq}(F{=}V),
\qquad
\Delta_{freq}^{T} = \Delta_{freq}(F{=}T)
$$

$$
\mathrm{freq} = \mathrm{clip}_{[0,100]}
\left(
100\cdot\frac{\Delta_{freq}(F)-\Delta_{freq}^{T}}{\Delta_{freq}^{V}-\Delta_{freq}^{T}}
\right)
$$

**Design decision**: Complementary to spatial-domain proxies. LWIR typically
concentrates energy in low frequencies (smooth thermal gradients) while visible
has richer high-frequency content (texture, edges, fine patterns). Uses endpoint
anchoring (like the gradient proxy): the delta for fused=visible and fused=lwir
defines the [0, 100] range, ensuring exact 0 and 100 at the endpoints. Without
anchoring, visible and LWIR images of the same scene share enough spatial
structure that their FFT spectra are naturally correlated (~0.7+), compressing
the output range and preventing the proxy from reaching the expected extremes.
Operates on grayscale (channel average) and is permutation-invariant.

### Final raw contribution (`cont_vis_raw`)

$$
\mathrm{cont\_vis\_raw} = \frac{1}{6}
\left(
\mathrm{reg} + \mathrm{mi} + \mathrm{ssim} + \mathrm{grad} + \mathrm{spectral} + \mathrm{freq}
\right)
$$

### Disagreement-aware weighted contribution (`cont_vis`)

Let $p_i$ be the 6 proxy scores and $\tilde{p}$ their median. Define robust weights
from median absolute deviation (MAD):

$$
z_i = \frac{|p_i - \tilde{p}|}{1.4826\,\mathrm{MAD}(p) + \epsilon},
\qquad
w_i = \mathrm{clip}_{[0.05,1]}\!\left(\frac{1}{1+z_i^2}\right)
$$

$$
\hat{p}_{\mathrm{rob}} = \frac{\sum_i w_i p_i}{\sum_i w_i}
$$

A disagreement confidence is derived from proxy spread:

$$
\mathrm{contrib\_confidence} = 100\cdot\mathrm{clip}_{[0,1]}\left(1-\frac{\mathrm{std}(p)}{35}\right)
$$

The final score shrinks toward neutral 50 when disagreement grows:

$$
\mathrm{cont\_vis} = c\,\hat{p}_{\mathrm{rob}} + (1-c)\,50,
\quad c=\frac{\mathrm{contrib\_confidence}}{100}
$$

### Dispersion across proxies (`contrib_std`)

$$
\mathrm{contrib\_std} = \mathrm{std}
\left[
\mathrm{reg},\mathrm{mi},\mathrm{ssim},\mathrm{grad},\mathrm{spectral},\mathrm{freq}
\right]
$$

### Calibrated thermal axis (`latent_z`)

Since v10, the pipeline fits **one monotonic calibration curve per proxy**
(reg, mi, ssim, grad, spectral, freq), not just one aggregate curve. Each raw
proxy value is mapped through its own curve to a visible fraction in `[0,1]`,
and the six calibrated values are combined by **Inverse-Variance Weighting
(IVW)**:

$$
\hat{v} = \frac{\sum_p w_p\,\hat{v}_p(p)}{\sum_p w_p},
\qquad
w_p = \frac{1}{\sigma_p^2 + \epsilon}
$$

- $\hat{v}_p(\cdot)$ is the per-proxy monotonic interpolator fitted during
  calibration (PAVA isotonic regression on per-alpha-group medians).
- $\sigma_p$ is the proxy's **calibrated intra-group dispersion** — computed
  by pushing the calibration samples through their own curve and measuring
  how tight the per-alpha-group distribution is. Low $\sigma_p$ ⇒ the proxy
  reacts consistently to known mixtures ⇒ larger weight.
- A variance floor $\epsilon$ prevents a degenerate zero-variance proxy
  from collapsing the combination onto itself.

$$
z = \text{latent\_z} = 1 - \hat{v}
$$

**Why per-proxy + IVW?** Proxies saturate at different rates (e.g. `grad` and
`freq` reach 90-100% visible at alpha ≈ 0.3, while `reg` stays almost linear).
Calibrating each proxy independently *corrects the non-linearity* automatically
without reshuffling raw values. IVW then down-weights proxies whose calibrated
output is noisy (they had wide intra-group spread during calibration),
preferring proxies that proved stable.

An **aggregate calibration** on the averaged `cont_vis` is still fitted for
backward-compatibility and diagnostics (column `lz_agg` in the calibration
table) — it is what would happen if you applied a single curve to the mean
proxy score. Compare with `lz_per_proxy` to see the effect of per-proxy IVW.

### Channel-concatenation calibration (diagnostic)

In addition to the main **blend** calibration
(`F = (1-α)·V + α·T_rgb`), the pipeline also generates a **concat**
calibration using channel-concatenation ground truths:

```
[R, G, T]     → α = 1/3        (2 visible channels out of 3)
[R, G, B]     → α = 0          (fully visible)
[T, T, T]     → α = 1          (fully thermal)
```

With 3 channels we can only obtain α ∈ {0, 1/3, 2/3, 1}. Introducing
intra-channel mixing (e.g. `T/2 + R/2`) would break the structural "whole
channel from one source" property, so concat stays at 4 anchor points and
serves only as a **sanity check**, not as a second calibration to be averaged
with blend. The `apply_contribution_calibration` function uses the blend
curves to produce `latent_z`; concat is plotted side-by-side for verification.

**A note on RTT/GTT/BTT and R vs G vs B.** There is no *perceptual*
weighting in our metrics (no `0.299·R + 0.587·G + 0.114·B`), so we do not
privilege one visible channel over another on luminance grounds. But R, G
and B are *not* structurally equivalent either — they are correlated but
distinct channels carrying different information about the scene. In
particular:

- `[R, R, T]` has a *duplicated* visible channel plus thermal. Structurally
  it looks closer to LWIR than one might expect because `spectral` picks up
  on the redundancy (R correlates perfectly with R, like T does with T).
- `[R, G, T]` has two *independent* visible channels plus thermal.
  Structurally this preserves more visible information than `[R, R, T]`
  even though both have α = 1/3.

For this reason the concat calibration uses patterns where each listed
visible channel is **distinct**:

```
α = 1/3:  (R,G,T), (R,T,B), (T,G,B)        # 2 distinct visibles + thermal
α = 2/3:  (R,T,T), (T,G,T), (T,T,B)        # 1 distinct visible + 2 thermals
```

The calibration knot at each α is the median of the metric across the 3
patterns — this averages over "which specific channel was used" without
assuming the channels are interchangeable. We do *not* use RRT/GGT/BBT
because they introduce channel redundancy that would bias the knot toward
LWIR-like behavior for no good reason.

Design Decisions
------------------------------------

- Why per-channel proxies: flattening 3-channel visible and replicated-LWIR into
	single vectors creates a structural bias toward visible because it has 3
	independent spectral bands vs 1 replicated band. Per-channel computation
	ensures 1D-vs-1D comparisons on equal footing.
- Why subsampled MI: with 128x128=16384 histogram bins, ~20k samples give <1%
	estimation error. Using 50k gives comfortable margin. Fixed seed (42) ensures
	reproducibility. Full-image MI on large images is unnecessarily expensive.
- Why combined gradient: as two separate proxies they contributed 40% of the
	aggregate, over-weighting structural texture signals. As one proxy they
	contribute 1/6, better balanced against semantic proxies.
- Why spectral independence: captures information diversity across channels
	without assuming color semantics. Works for true-color, false-color, and
	descriptor outputs (PCA/FA). The question is not "does the fused image look
	colorful?" but "do the fused channels carry independent spectral information?"
- Why frequency proxy: provides an orthogonal view to spatial-domain metrics.
	Low-frequency dominance suggests LWIR contribution; high-frequency richness
	suggests visible. Not biased by fine-texture density in the same way as
	gradient proxies.
- Why weighted aggregation: plain mean is easy but can be unstable when one proxy
	disagrees strongly. The weighted score improves robustness while keeping the
	pipeline simple and deterministic.
- Why keep `cont_vis_raw`: auditability and comparability. You can still inspect
	the original unweighted behavior and quantify how much weighting changed each method.
- Why shrink toward 50 under disagreement: when proxies conflict, certainty is lower.
	Moving toward neutral avoids overconfident extremes.
- Why no semantic proxy yet: semantic encoders (LPIPS/ViT/CLIP-like) add model bias,
	compute overhead, and domain adaptation cost. This pipeline stays signal-centric for
	now and reports structural uncertainty explicitly.
- Why permutation pre-screening: for standard fusion methods, channel identity is
	best. For descriptor methods (PCA, FA), channel order may vary. Pre-screening
	cheaply selects the top 1-2 permutations by per-channel correlation, reducing
	computation from 6x to ~1-2x without loss of robustness. Permutation-invariant
	proxies (grad, spectral, freq) skip this entirely.

Calibration Protocol (What "split/alpha/protocol" means)
---------------------------------------------------------

The pipeline estimates a latent thermal axis `z` in `[0, 1]`:

- `z = 0`: visible-only endpoint.
- `z = 1`: LWIR-only endpoint.

It does this in two stages:

1. Compute a raw per-image contribution score from six proxies
   (per-channel NNLS, per-channel MI, multichannel SSIM, combined gradient,
   spectral independence, frequency correlation).
2. Calibrate that raw score to the latent axis using synthetic mixtures:
	 `I_alpha = (1 - alpha) * V + alpha * T`.

Protocol fields are parameters that define which calibration is valid:

- dataset and root path
- condition (day/night/all)
- split settings (ratio/images + seed)
- alpha grid density (`alpha_steps`)
- equalization variant (none/th_equalization/rgb_equalization/rgb_th_equalization)

If any of these changes, a fresh calibration is generated and cached separately.

How calibration is applied
--------------------------

The calibration is **not** computed per image. It is fitted once from the
synthetic calibration subset, then reused for every evaluated image that shares
the same protocol.

In practice:

- one calibration curve is built per preset and equalization variant
- each fused image gets its raw `cont_vis` mapped through that shared curve
- the same calibration is reused for all images in the run

So the evaluation compares images on a common scale; it does not search for a
different optimal calibration for each individual image.

Metric Design Principles
------------------------

Contribution scoring is intentionally designed to be:

1. **Generic across methods**: no method-specific heuristics are injected into
	the metric.
2. **Representation-aware**: visible information is evaluated using per-channel
	signals (not a single early grayscale collapse), while LWIR remains a single
	source replicated to 3 channels for fair comparison.
3. **Channel-order robust**: fused 3-channel outputs are evaluated with
	permutation pre-screening for channel-sensitive proxies, while spectral,
	gradient, and frequency proxies are permutation-invariant by construction.
4. **Multi-domain**: spatial (reg, mi, ssim), structural (grad), spectral
	(spectral independence), and frequency (FFT) domains are all represented
	for comprehensive characterization.

In practice, this keeps the score meaningful for classic RGB-like fusions,
pseudo-color outputs, and descriptor-like fusion encodings without tailoring the
metric to any specific method family.

Runtime and Cache Notes
-----------------------

- Evaluation metrics are cached per method/image with implementation fingerprint.
- Calibration now supports incremental sample caching for `(image, alpha)` points.
- Calibration is auto-saved/auto-loaded using a protocol-derived hash path in cache.
- Metric logic changes are versioned in code and included in cache keys/hash
	inputs, so old cache entries are not mixed with new metric definitions.

Plot Interpretation Guide
-------------------------

All diagnostic plots are written to the report folder with filenames of the
form `<dataset>_<condition>_<plotname>_<variant>.png`. Here is what each
template plot tells you:

### `proxy_overview`

Three stacked panels per dataset/condition:

1. **Top (grouped bars)**: raw per-proxy `cont_vis` per fusion method. Each
   method has six bars (reg/mi/ssim/grad/spectral/freq). Useful for spotting
   methods where one proxy disagrees with the rest.
2. **Middle (side-by-side bars)**: raw `cont_vis` (uncalibrated) vs
   calibrated `cont_vis` (IVW per-proxy) per method. Shows the net effect of
   calibration: a method at `cont_vis≈75` may drop to `cont_vis_calibrated≈55`
   once saturating proxies are corrected.
3. **Bottom (single-color bars)**: final `latent_z` per method, blue→red
   colormap (blue = visible-dominant, red = LWIR-dominant). Read this as the
   final ranking.

### `calibration_diagnostic`

Left panel overlays the blend and concat calibration curves. Right panels
(one per sample type) show the per-alpha boxplot of `cont_vis` across images.
Ideally the per-alpha medians should track the dashed ideal line
`(1-α)·100`; the spread of each box reports how noisy that alpha slice is.

### `per_proxy_calibration_curves`

Six panels, one per proxy. Each panel shows:
- solid colored curve: the per-proxy monotonic calibration
- dashed gray: the aggregate curve, for reference
- dashed black: the ideal linear response
- colored 'x' markers: per-alpha group medians (the knots PAVA fits)
- green stars: fusion methods projected onto that proxy's curve

The legend label `w=0.xxx  σ=0.yyy` shows the IVW weight and the calibrated
intra-group std used to compute it. A proxy with `σ` near zero dominates the
combination; a noisy proxy contributes almost nothing.

### `per_proxy_calibration`

Per-proxy boxplot of raw proxy values vs alpha, with ideal line overlaid.
Complementary to `per_proxy_calibration_curves`: shows pre-calibration
saturation, not the calibration curve itself.

### `calibration_nodes_overview`

Bar chart with one group of bars per α step. Top panel: each proxy's median
raw value at that α. Bottom panel: deviation of aggregated `cont_vis` from
the ideal `(1-α)·100`. Red bars below zero mean the method under-reports
visible at that α.

### `training_vs_latent_<equalization>`

2x2 grid, one panel per training metric (P, R, mAP50, mAP50-95). X-axis is
the computed `latent_z` for the method (under that equalization), Y-axis is
the observed training metric. Each method shows:
- faint blue dots: individual training runs from `raw_data/raw_training_data.csv`
- red diamond: per-method mean
- dashed gray line: linear fit
- Pearson r printed in the upper-left

Use this to check whether the computed latent axis tracks actual detection
performance. A monotonic trend (positive or negative slope) suggests the
latent axis is capturing something useful about the fusion. A flat cloud
means the axis and the detection metric are essentially uncorrelated — which
may simply mean that, under the tested training conditions, the
visible/thermal mixture matters less than we assumed.

Correlation Choice: Pearson vs Spearman
---------------------------------------

- **Pearson** correlation assumes a *linear* relationship. We use it in the
  regression proxy (`reg`, NNLS is explicitly linear) and in the training-vs-
  latent summary plot (easy to read as a coefficient). It is the right choice
  when the underlying model is linear.
- **Spearman** correlation uses *ranks* and only assumes a *monotonic*
  relationship. It is more robust to outliers and to saturating metrics,
  and makes fewer assumptions about the shape of the response.

In practice we compute **both**:

- Pearson r is the primary coefficient because calibration already handles
  non-linearity (each proxy has its own monotonic PAVA curve), so by the
  time values reach the final combination they live in a roughly linear
  "calibrated visible-fraction" space. Pearson is also what the NNLS `reg`
  proxy optimises.
- Spearman ρ is emitted alongside (`sp_vis`, `sp_lw` in the per-image
  result dict) as a rank-based diagnostic. It is more honest when
  inspecting raw saturating proxies or when an outlier image would
  dominate Pearson. Large Pearson/Spearman disagreement is a red flag
  that the relationship is non-linear or outlier-driven for that method.

Training Results Correspondence
-------------------------------

`training_results_check.py` encapsulates CSV parsing so that format changes
only require editing that one module. It exposes:

- `load_training_results(csv_path, target_class)` — read and filter.
- `get_runs(df, fusion_type, condition, equalization)` — slice rows.
- `build_method_metrics(df, method_name, condition, equalization)` — mean /
  std / median / raw values for P, R, mAP50, mAP50-95 for one slice.
- `build_dataset_metrics(df, method_names, condition, equalization)` — dict
  of the above across several methods.

Fusion method names used in the pipeline (e.g. `sobel_weighted`, `ssim_v2`,
`rgbt_v2`, `curvelet_max`, `wavelet_max`) are mapped internally to the
shorter CSV tags (e.g. `sobel`, `ssim`, `rgbt`, `curvelet`, `wavelet`). Add
entries to `_METHOD_NAME_TO_CSV_TYPE` if you add more fusion methods with
non-matching names.

With only ~5 runs per method, a classical boxplot is noisy; we instead use
a strip-plot style (individual points + mean marker) which is more honest
for small n.

Gradient Proxy Notes
--------------------

- The gradient proxy is a 50/50 composite of magnitude correlation and orientation
	agreement, computed on grayscale (channel average) and mapped to [0, 100] with
	per-pair anchor endpoints.
- This anchoring keeps gradient scales interpretable and bounded.
- In night scenes, visible and LWIR can still share strong edge structure, so
	the gradient proxy may remain less extreme than expected for some methods.
- The gradient proxy correctly reports visible dominance when fused images
	preserve visible texture, and thermal dominance otherwise. It can still lean
	toward visible when visible has richer fine texture, but as a single proxy
	(1/6 weight vs the former 2/5 = 40%) this no longer dominates the aggregate.
- Interpretation guideline: use the gradient proxy as structural-detail evidence,
	complemented by frequency and spectral proxies for a fuller picture.
