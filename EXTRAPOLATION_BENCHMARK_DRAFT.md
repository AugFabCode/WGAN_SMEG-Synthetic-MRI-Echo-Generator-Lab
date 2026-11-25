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

## Image-similarity metrics to calculate (planned, not coded)

- **LPIPS, lower is better:** Declare a pinned model/backbone, pretrained weights, native-resolution 2-D slice orientation, handling of 1-channel MR images and intensity scaling. A natural-image LPIPS network is not automatically MRI-validated. The existing optional training perceptual loss is not itself LPIPS. Keep optional PyTorch metric computation isolated from the TensorFlow training GPU context.
- **SSIM and MS-SSIM, higher is better:** Use a version-pinned published definition. Declare 2-D vs 3-D evaluation, mask treatment, window/sigma, number of scales and how slices are aggregated. Small native MRI patches may not support standard multi-scale implementations: do not upsample MRI data merely to satisfy a library's minimum image size, and do not mislabel a custom approximation as standard MS-SSIM.
- **NCC, higher is better:** Masked normalized correlation with explicit constant/empty-image rules. Correlation alone can mask intensity errors.
- **PSNR, higher is better:** Specify a **frozen** per-modality data range. Do not use individual acquired targets' maxima, which makes cross-TE/checkpoint comparisons unstable.
- **MAE and RMSE, lower is better:** Same brain ROI/mask, normalization domain and physical interpretation as above; compute RMSE before averaging per scan.

Magnitude and phase need distinct reports. For wrapped phase, specify whether to evaluate the original phase or a circular/angular error representation; do not automatically treat branch cuts as anatomical mismatch. Reference normalization must not depend on synthetic output or held-out target data. All metric code requires synthetic numerical fixtures and representative ABRIM validation before scientific reporting.

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
