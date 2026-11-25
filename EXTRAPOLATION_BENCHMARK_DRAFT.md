# Draft PR — fixed ABRIM extrapolation benchmark and weighted checkpoint score (3a/3b)

Repository: https://github.com/AugFabCode/WGAN_SMEG-Synthetic-MRI-Echo-Generator-Lab

**Implementation status:** This PR provides the deterministic scenario definitions, a pure-Python metric reporting/ranking scaffold, and unit tests. **Model inference and image-similarity metric extraction are not yet implemented.** The online training loop, fixed_v2 MAE-based best-checkpoint selection, resume handling, and early stopping are deliberately unchanged.

## 3a — Fixed extrapolation scenarios

Each enabled scenario is to be evaluated over **every** validation sample from the existing independent subject-level partition (manifest split = eval). No randomly selected echo configurations or validation subset caps are permitted for this benchmark.

| Scenario ID | Model input | Ground-truth output | Difficulty group |
|---|---|---|---|
| TE1_to_TE6 | TE1 | Acquired TE6 | hard |
| TE3_to_TE9 | TE3 | Acquired TE9 | hard |
| TE1_TE2_to_TE6 | TE1 + TE2 | Acquired TE6 | medium |
| TE3_TE4_to_TE9 | TE3 + TE4 | Acquired TE9 | hard |
| TE1_TE2_TE3_to_TE6 | TE1 + TE2 + TE3 | Acquired TE6 | medium |
| TE3_TE4_TE5_to_TE9 | TE3 + TE4 + TE5 | Acquired TE9 | medium |
| TE1_TE2_TE3_to_TE9 | TE1 + TE2 + TE3 | Acquired TE9 | hard |

These medium/hard labels are qualitative and can be adjusted after observing results; true difficulty depends on both extrapolation distance (in milliseconds) and number of input echoes. Use actual per-scan TE values from the manifest, not fixed nominal TE values.

The reporting CLI supports all seven (default), medium, hard, or an explicit comma-separated subset of scenario IDs. A separate multiselect in the training dashboard is proposed but is **not included** in this draft.

### Planned inference implementation

Use the established model interfaces and the preprocessed fast NumPy/HDF5 cache:

1. Load the active manifest with load_active_ceaset_manifest(modality=...), filtering strictly split = eval. Validate the subject-level split and artifacts before inference; **do not** cap validation samples through EVAL_MAX_CHUNKS.
2. For each scan/chunk and selected scenario, resolve zero-based echo indices from TE labels. Call make_ceaset_sample_from_configuration with those input indices, the physically acquired target echo, augment_rng=None and continuous_auxiliary=False. Never supply target echo values to input normalization or physics baseline.
3. Inference runs with generator(inputs, training=False), using the checkpoint's matching saved configuration and model architecture. Run magnitude and phase separately, with no optimizer update.
4. Calculate and save per-chunk metrics with checkpoint identity, modality, subject/case, chunk, scenario, input and target TEs, mask method, scale method and software versions.
5. Produce per-scan, per-scenario, and checkpoint-comparison reports; retain individual metrics alongside the weighted score. Do not write MRI preview images by default.

**Important leakage limitation:** The existing brain mask comes from the acquired TE3 magnitude. In the TE1-only scenario, that means the mask uses data from an echo outside the declared input set. The current benchmark must therefore be labeled a **retrospective reference-mask evaluation**, not proof of fully prospective TE1-only acquisition-time savings. A later input-only brain-mask protocol and phase preprocessing dependency audit are needed for a true shorter-acquisition experiment.

## Image-similarity metrics: scientific definitions (planned, not coded)

Let $x$ denote the acquired reference echo, $y$ the predicted echo, and $\Omega$ the common valid brain ROI containing $N$ voxels. The images must share spatial coordinates and a declared intensity domain. The following definitions describe the proposed evaluation protocol, not completed metric extraction.

### LPIPS — Learned Perceptual Image Patch Similarity (25%)

LPIPS measures the distance between features of two images extracted from a pretrained convolutional neural network. At selected layers, feature vectors are normalized across channels, their differences are weighted using learned channel coefficients, squared and spatially averaged: $d(x,y)=\sum_l (H_lW_l)^{-1}\sum_{h,w}\|w_l\odot(\hat{f}_l(x)_{hw}-\hat{f}_l(y)_{hw})\|_2^2$. Lower is better; identical images yield zero and there is no universal upper bound of one. LPIPS can capture texture and appearance differences beyond pointwise errors, potentially complementing assessment of late-echo vessel detail. However, natural-image pretraining does not guarantee sensitivity to clinically significant MRI abnormalities or correct susceptibility information. Fix the backbone, weights, grayscale-to-RGB adaptation, intensity scaling, native slice orientation and ROI policy. The existing RadImageNet training perceptual loss is not LPIPS. Isolate optional PyTorch metric computation from the TensorFlow training GPU context. Sources: [Zhang et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html); [original LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity).

### SSIM — Structural Similarity Index Measure (20%)

SSIM compares local luminance, contrast and structural correspondence using means $\mu_x,\mu_y$, variances $\sigma_x^2,\sigma_y^2$ and covariance $\sigma_{xy}$. A common form is $\mathrm{SSIM}(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}$, with stabilizers $C_1=(K_1L)^2$ and $C_2=(K_2L)^2$ for a prescribed intensity range $L$. Local scores are aggregated over valid windows. Higher is better; one represents perfect similarity, and negative values are possible. For echo synthesis, SSIM describes local anatomical organization and contrast, but can penalize noise differences or overlook localized vessel errors. Freeze the window size, Gaussian weighting, data range, ROI inclusion and 2-D/3-D definition across checkpoints. Sources: [Wang, Bovik, Sheikh and Simoncelli, IEEE Transactions on Image Processing 13(4), 600–612 (2004)](https://doi.org/10.1109/TIP.2003.819861); [scikit-image 0.24 metrics documentation](https://scikit-image.org/docs/0.24.x/api/skimage.metrics.html).

### MS-SSIM — Multi-Scale Structural Similarity (20%)

MS-SSIM evaluates structural agreement at several resolutions obtained by low-pass filtering and downsampling. It combines contrast $c_j$ and structure $s_j$ at each scale $j$ with luminance $l_M$ at the coarsest scale $M$: $\mathrm{MS\!-\!SSIM}(x,y)=l_M^{\alpha_M}\prod_{j=1}^{M}c_j^{\beta_j}s_j^{\gamma_j}$, with protocol-defined scale weights. Higher is better; identical images yield one, and conventional nonnegative implementations give values between zero and one. In late-echo MRI, MS-SSIM complements SSIM by assessing fine and coarse anatomical patterns but cannot guarantee preservation of small veins or pathology. Declare scale count, weights, filtering, voxel anisotropy, dimensionality and mask policy. Small native patches that cannot support the required scales must be flagged rather than artificially upsampled or evaluated with an undisclosed approximation. Source: [Wang, Simoncelli and Bovik, Multiscale structural similarity for image quality assessment, 37th Asilomar Conference, 1398–1402 (2003)](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Wang03b).

### NCC — Normalized Cross-Correlation (15%)

NCC quantifies linear correspondence between co-located intensities after mean subtraction and normalization by centered energy. The proposed global signed Pearson-style definition is $\mathrm{NCC}(x,y)=\frac{\sum_{i\in\Omega}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i\in\Omega}(x_i-\bar{x})^2\sum_{i\in\Omega}(y_i-\bar{y})^2}}$. Values range from minus one to one when both images have nonzero variance; one is perfect positive linear association, zero is no linear association, and minus one is perfect negative association. Higher is better, without squaring or taking the absolute value. NCC is useful for comparing MRI intensity patterns despite positive global gain or offset changes, but those invariances mean high NCC does not establish quantitative magnitude or phase fidelity. Constant images and empty masks make the expression undefined and must be flagged, not scored as perfect. Do not optimize spatial shifts between already aligned echoes. Source: [J. P. Lewis, Fast Normalized Cross-Correlation, Vision Interface (1995), expanded author version](https://scribblethink.org/Work/nvisionInterface/nip.pdf).

### PSNR — Peak Signal-to-Noise Ratio (10%)

PSNR expresses the ratio of a prescribed squared intensity range to the mean squared reference error, in decibels: $\mathrm{PSNR}(x,y)=10\log_{10}(L^2/\mathrm{MSE})$, with $\mathrm{MSE}=N^{-1}\sum_{i\in\Omega}(x_i-y_i)^2$ and declared data range $L>0$. Higher is better; exact matches have mathematically infinite PSNR. PSNR measures overall voxel-level agreement, not physical scanner SNR or denoising performance: noise in the acquired reference contributes to the error, and smooth images can score well without retaining subtle vessels. Use a frozen modality-specific data range rather than each target or prediction's maximum. The future implementation must define a documented finite scoring convention for exact matches while preserving their identity in raw reports; the current scaffold rejects non-finite values. Sources: [scikit-image 0.24, peak_signal_noise_ratio](https://scikit-image.org/docs/0.24.x/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio); [review of cross-contrast MRI image synthesis](https://doi.org/10.1007/s44352-025-00012-3).

### MAE — Mean Absolute Error (5%)

MAE is the arithmetic average of absolute voxel differences within the common valid ROI: $\mathrm{MAE}(x,y)=N^{-1}\sum_{i\in\Omega}|x_i-y_i|$. It is nonnegative, has the units of the evaluated intensities and reaches zero for exact agreement. Lower is better. Each residual contributes linearly, making MAE less dominated by isolated large errors than RMSE. It provides an interpretable measure of average intensity error for synthetic magnitude and phase echoes, but does not distinguish a localized vessel error from dispersed noise with the same total absolute deviation. Report the fixed intensity domain and mask; linear MAE is not a circular distance for wrapped phase. Sources: [Willmott and Matsuura, Climate Research 30, 79–82 (2005)](https://doi.org/10.3354/cr030079); [Chai and Draxler, Geoscientific Model Development 7, 1247–1250 (2014)](https://doi.org/10.5194/gmd-7-1247-2014).

### RMSE — Root Mean Squared Error (5%)

RMSE is the square root of the average squared voxel residual: $\mathrm{RMSE}(x,y)=\sqrt{N^{-1}\sum_{i\in\Omega}(x_i-y_i)^2}$. It is nonnegative, shares the intensity units of the images and equals zero for identical images. Lower is better. Squaring residuals before averaging makes RMSE sensitive to large local synthesis failures, potentially revealing errors near susceptibility-related boundaries, but also emphasizing noisy or artifact-affected voxels. Compute RMSE in the declared evaluation domain before averaging sample-level results. RMSE and PSNR are mathematically linked for a fixed data range, so their combined weighting emphasizes squared intensity error rather than independent quality dimensions. Wrapped phase requires an explicit circular residual policy where appropriate. Source: [Chai and Draxler, Root mean square error (RMSE) or mean absolute error (MAE)? (2014)](https://doi.org/10.5194/gmd-7-1247-2014).

### Interpretation and MRI-specific validity

Magnitude and phase require distinct reports. For wrapped phase, specify whether to evaluate the original phase or a circular/angular error representation; do not automatically interpret branch cuts as anatomical mismatch. Normalization must not depend on synthetic output or held-out target data. All metric implementations require synthetic numerical fixtures and representative ABRIM validation before scientific reporting. The proposed weights are research choices, not literature-validated optimal weights: structural and intensity-error metrics overlap, and similarity to noisy acquired data does not establish denoising or clinical equivalence. Supplement the score with vessel-sensitive assessment, blinded radiological review and downstream SWI/QSM comparison. See [Agreement of image quality metrics with radiological evaluation in the presence of motion artifacts](https://doi.org/10.1007/s10334-025-01266-y) for MRI-specific evidence that metric agreement with human evaluation must be assessed empirically.

## 3b — Fixed-weight quality score

Weights sum to 100%:

| Image similarity metric | Weight | Better |
|---|---:|---|
| LPIPS | 25% | lower |
| SSIM | 20% | higher |
| MS-SSIM | 20% | higher |
| NCC | 15% | higher |
| PSNR | 10% | higher |
| MAE | 5% | lower |
| RMSE | 5% | lower |

The unspecified remaining 20% is provisionally divided as PSNR 10%, MAE 5%, RMSE 5%.

**A direct weighted average of raw values is invalid.** To compare dimensionless quality, each metric must be mapped to [0,1] using a versioned, frozen low/high reference interval. Define u = clip((metric - low)/(high - low), 0, 1); invert u for LPIPS, MAE and RMSE. Sum the weighted quality values, so **higher composite score is better**.

The prototype intentionally requires an explicit calibration JSON with:

- version: nonempty version identifier
- source_split: training or external
- frozen: true
- bounds: one set of seven [low, high] intervals per requested modality

Calibration bounds must be derived from the **training cohort or external reference**, checked before evaluating checkpoint performance, and frozen for the entire comparison. Never fit bounds from the evaluation/test cohort or re-estimate per checkpoint. There are no arbitrary operational default bounds. If LPIPS or any other metric is missing/non-finite, the report fails rather than silently reweighting the remaining metrics.

### Scan-balanced aggregation

One row in the input CSV means one patch/chunk x scenario x checkpoint x modality. Compute:

1. Average each metric over chunks **within each case**.
2. Compute the normalized weighted quality score **per case and scenario** using the frozen calibration.
3. Average case metrics and case scores with equal case weights **within the scenario**.
4. Macro-average over selected scenarios with equal scenario weights.

The scaffold enforces **identical (case_id, chunk_id) coverage** across scenarios and compared checkpoints within each modality, rejects duplicates, and refuses any train/test row. Thus dropping difficult samples cannot artificially inflate a checkpoint. If the dataset contains several runs/sessions for one patient, future work should add a patient-key aggregation layer so patient contributions are equal, not just case contributions.

Report the full metric vector and per-combination score, not just the composite. Investigate confidence intervals, clinically relevant anatomy, vessels/noise, and downstream SWI/QSM outputs before final model selection. The independent test cohort is reserved exclusively for evaluation **after** the chosen checkpoint and calibration are frozen.

## Implemented: dependency-free scoring report

Already-measured input CSV must have columns:

~~~text
checkpoint,modality,case_id,chunk_id,combination,split,lpips,ssim,ms_ssim,ncc,psnr,mae,rmse
~~~

Examples:

~~~bash
# All seven scenarios, raw metric report only (no arbitrary score)
python echo_gen_extrapolation_benchmark.py \
  --csv evaluation/metrics.csv \
  --combinations all \
  --output evaluation/extrapolation_report.json

# Frozen calibration is needed to compute and rank weighted scores
python echo_gen_extrapolation_benchmark.py \
  --csv evaluation/metrics.csv \
  --combinations medium \
  --calibration evaluation/calibration_frozen.json \
  --output evaluation/extrapolation_scored.json

python -m unittest -v test_echo_gen_extrapolation_benchmark.py
~~~

**Example JSON structure only. The numeric ranges here are placeholders and MUST NOT be used as scientifically validated calibration:**

~~~json
{
  "version": "example-replace-with-training-calibration",
  "source_split": "training",
  "frozen": true,
  "bounds": {
    "magnitude": {
      "lpips": [0, 1],
      "ssim": [-1, 1],
      "ms_ssim": [0, 1],
      "ncc": [-1, 1],
      "psnr": [0, 60],
      "mae": [0, 1],
      "rmse": [0, 1]
    }
  }
}
~~~

Include a separately validated phase bounds object before reporting phase scores.

### Outstanding work before the draft is merge-ready

- Implement deterministic TensorFlow model-inference runner for all validation chunks and the seven explicit echo configurations, with preprocessing-identity and checkpoint validation.
- Compute image metrics with version-pinned dependencies, mask/ROI rules, native-resolution handling, phase policy, and LPIPS backend; validate against analytical and real MRI examples.
- Persist independent per-scan metric tables and provenance (TE values, protocol version, calibration SHA, generator checkpoint, subject/patient key, mask/scale policy); add uncertainty and error review.
- Add optional seven-scenario selection and results panel to the dashboard. Avoid running seven costly inferences every epoch without an explicit user opt-in.
- Decide **after calibration and scientific review** whether an opt-in score-based best-checkpoint and early-stopping policy is justified; maintain backwards-compatible MAE-based resume and existing checkpoint paths.
- Run the frozen final checkpoint on the held-out test partition without re-optimizing the score, then evaluate downstream SWI/QSM reconstructions and blinded clinical reviews.

The intent of this draft PR is review of scenario definitions, scientific validity and a safe reporting interface **before** changing production checkpoint selection.
