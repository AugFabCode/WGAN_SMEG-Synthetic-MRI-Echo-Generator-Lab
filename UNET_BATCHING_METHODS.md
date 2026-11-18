# 3-D U-Net input batching: grid patches versus full-volume ROI

**Audience:** MRI / medical-image analysis supervisor and students.
**Project:** [WGAN_SMEG — Synthetic MRI Echo Generator](https://github.com/AugFabCode/WGAN_SMEG-Synthetic-MRI-Echo-Generator-Lab)

## What problem does this solve?

Our generator synthesizes a missing/late multi-echo GRE MRI echo from earlier acquired magnitude or phase echoes and their echo times (TE). The **generator is a 3-D U-Net-like, TE-conditioned model embedded in a WGAN-GP pipeline**. The training target is the *acquired target echo*, not a segmentation label. A brain mask restricts image losses and prevents non-brain background from entering the synthesis output.

An acquired volume frequently has a large, entirely empty border. Sending empty voxels through a convolutional network costs memory and compute without providing anatomical signal. There are two ways to package the 3-D images for training:

1. **Grid patches (existing method)**: subdivide each image into overlapping small 3-D training samples.
2. **Full volume (new option)**: pass one entire 3-D brain volume per subject, optionally cropped tightly to the brain mask.

**Brain bounding-box cropping is an *independent* preprocessing option**, applicable to either method. Thus the dashboard provides four combinations, not two entirely separate model implementations.

## Comparison at a glance

| Property | Grid patches | One full volume per subject |
|---|---|---|
| Training unit | One spatial chunk from a regular 3-D grid | One 3-D sample containing the whole retained field of view |
| Existing implementation | Reuses `build_chunk_specs()`, `extract_chunk()` and the direct TensorFlow loop | Reuses exactly the same functions with grid `(1,1,1)` and overlap `0` |
| Brain mask ROI crop | Optional, before calculating the grid | Optional, before calculating the single volume |
| Sample spatial shape | Uniform within an individual subject's grid; depends on ROI dimensions in crop mode | May differ between subjects |
| GPU memory per optimizer step | Usually lower; recommended for 3-D WGAN-GP | Usually higher than an individual patch, even after ROI cropping |
| Spatial context | Local neighborhood plus configurable overlap | Full retained 3-D anatomy in a single network call |
| Work per epoch | Many chunks per patient, potentially many steps | One volume per patient (plus configured echo-sampling repeats) |
| Border effects | Convolutions at patch edges; overlap provides context | No artificial internal grid boundaries |
| Sampling coverage | Grid covers entire ROI; chunks with no foreground are skipped | Whole retained ROI per subject |
| Batch dimension | **Fixed at 1** in this codebase | **Fixed at 1** in this codebase |
| Suitable first experiment | **Yes** (including on the A6000) | Only after confirming GPU memory headroom |

**Important:** Here “batching mode” means *how a 3-D spatial training sample is formed*. The optimizer's minibatch size is **not** being changed. The existing direct-Python pipeline deliberately fixes `TRAINING_BATCH_SIZE=1` because each sample draws a variable number of input echoes (K = 1–4), without introducing dummy echo padding.

## Why crop to the brain-mask bounding box?

For each subject, the already-produced BET/eroded brain mask is loaded exactly once. Its nonzero foreground (> 0.5) determines a half-open, axis-aligned 3-D voxel bounding box. The optional safety margin extends every side by a user-selected number of voxels (default: 4) and is clamped to the acquired image limits:

```text
Original image:  [256, 256, 176]              Acquired field of view
Mask foreground: bounds of brain anatomy      One box per subject
Margin:         +4 voxels on each side        No change in voxel spacing
ROI crop:       image[H0:H1, W0:W1, D0:D1]    Applied identically to all data
Grid mode:      subdivide cropped ROI         Same existing grid algorithm
Full mode:      keep entire cropped ROI       One sample per patient
Padding:        round sample axes to 8/16/32  Zero padding, no interpolation
```

The crop is computed once from the **mask**, never separately from individual echoes. Consequently, all nine magnitude echoes, all nine phase echoes, the brain mask and the acquired target echo have identical spatial indexing. The selected input echoes and target echoes remain aligned and use their original TE values. The code uses NumPy slicing, equivalent in purpose to `fslstats -w` followed by spatial `fslroi`, but **does not start redundant FSL processes or resample intensities**. In this repository the processing arrays swap source axes 0 and 1; coordinates in the chunk manifest are therefore recorded as **H/W/D processing axes**. A new `chunk_nifti_affine()` maps optional NIfTI chunk exports back into the source image's scanner/world coordinates, including crop offset and zero padding.

**Clinical caution:** The ROI is only as complete as the selected brain mask. Aggressive BET/erosion can eliminate boundary tissues or superficial vessels. Inspect the mask and use a sufficient margin; do not interpret cropped-out tissue as model reconstruction failure. Padding produces *zero values*, not interpolated pixels.

## Detailed method A — original regular grid patches

The default remains `UNET_BATCHING_MODE="grid_patches"`, `UNET_BRAIN_CROP_ENABLED=false`.

- Existing default grid: `CHUNK_GRID=[5,5,3]`, or up to 75 nominal chunks per case.
- Existing default overlap: `CHUNK_OVERLAP_VOXELS=20`.
- The splitter calculates complete voxel coverage, including division remainders, and pads chunks to a multiple of the configured stride (8 by default).
- Chunks with no brain voxels are omitted from the manifest.
- Optionally cropping first creates a smaller ROI and often reduces the amount of background and the required padded chunk dimensions.

Unlike random patch extraction, the current grid is **deterministic**. Randomness comes from selecting input/target echo configurations, spatial augmentations, and shuffled chunk order. `TRAINING_SAMPLES_PER_CHUNK_PER_EPOCH` controls how many echo-configuration samples are drawn for each selected chunk per epoch.

For a 3-D WGAN-GP generator and critic, grid patches are the safer starting point when memory is constrained. The trade-off is limited anatomical context and possible boundary effects.

## Detailed method B — full-volume sampling

Set `UNET_BATCHING_MODE="full_volume"`. It internally selects `CHUNK_GRID=(1,1,1)` and `overlap=0`, regardless of the grid controls. The same `extract_chunk()`, HDF5 writing, fast NumPy caching, sample assembly, normalization, physics baseline, WGAN-GP losses, preview and validation remain in use.

For meaningful savings, also set `UNET_BRAIN_CROP_ENABLED=true`. Each patient has an individually sized ROI; the axes are padded to a multiple of 8, 16 or 32 so the three stride-2 encoder stages and decoder skip connections line up. **No rescaling to a common image size** occurs. Because the minibatch size remains 1, different patient volumes can be processed sequentially; the GPU still needs enough memory for the *largest* single patient sample.

For example, a `256×256×176` source contains about 11.53 million voxels. A hypothetical `180×210×154` retained ROI, padded to `184×216×160`, contains about 6.36 million voxels (55% of the original). This is a **voxel-count illustration, not a promised 45% VRAM or runtime saving**: intermediate 3-D feature maps, K input echoes, the adversarial critic and gradient penalty dominate peak allocations. Full-volume training can still exceed a 48 GB A6000's memory budget, and a smaller patch may remain much faster.

## Dashboard controls and meanings

| Setting | Default | Explanation |
|---|---|---|
| `UNET_BATCHING_MODE` | `grid_patches` | Existing grid chunks vs. one full-volume sample per subject |
| `UNET_BRAIN_CROP_ENABLED` | `false` | Opt in to foreground mask-based spatial cropping |
| `UNET_BRAIN_CROP_MARGIN_VOXELS` | `4` | Additional context on each of the six ROI faces, clipped to image limits |
| `UNET_PAD_MULTIPLE` | `8` | 8 matches current U-Net's 3-stage spatial downsampling; 16/32 add more padding |
| `CHUNK_GRID` | `[5,5,3]` | Active for grid mode only |
| `CHUNK_OVERLAP_VOXELS` | `20` | Active for grid mode only |
| `TRAINING_SAMPLES_PER_CHUNK_PER_EPOCH` | `1` | Independent input/target TE configurations sampled per spatial unit per epoch |
| `TRAINING_BATCH_SIZE` | `1` | Intentionally fixed for existing variable-K direct training |
| `OVERWRITE_DERIVED` | `false` | Opt in to rebuilding derived HDF5 chunks and fast NumPy cache after geometry changes |

**Recommended workflow:** In the dashboard select *Grid patches*, enable *Crop background to mask bounding box*, set 4-voxel margin and 8-voxel alignment, select **Rebuild derived training chunks and fast NumPy cache**, and click **Preprocess data**. This reuses existing prepared NIfTI files when their signatures are unchanged. After preprocessing completes, start the training job. Do **not** check *Rebuild changed/legacy partitions* unless patient allocations themselves need revision: that setting is unrelated to the ROI crop.

A sample advanced JSON override (the visible controls are preferable):

```json
{
  "UNET_BATCHING_MODE": "grid_patches",
  "UNET_BRAIN_CROP_ENABLED": true,
  "UNET_BRAIN_CROP_MARGIN_VOXELS": 4,
  "UNET_PAD_MULTIPLE": 8,
  "OVERWRITE_DERIVED": true
}
```

Changing the mode, crop flag, margin, grid, overlap or padding alters the **derived-data signature**. The existing cache is rejected as stale until rebuilt. Previously prepared image intensity normalization and the original nine echoes remain unaffected. No new directory naming scheme or independent data loader is introduced. Per-chunk metadata in the shared manifest includes `roi_start_*`, `roi_end_*`, `source_start_*`, `source_end_*` and `pad_before/after_*` for reconstruction in source coordinates.

## Fair scientific comparison

To compare the methods in an experiment, retain **the same patient-level train/evaluation/test partition** (default 70/15/15, or 80/20/0 in the explicitly labelled legacy two-way mode), normalization, selected TE distributions, random seed, network architecture and stopping criteria. Change only the spatial sampling method and/or ROI flag, and start separate run/checkpoint identities.

Report:

1. **Data efficiency:** source voxels, retained ROI voxels, fraction of zero-mask voxels, chunks per subject and training samples per epoch.
2. **Systems performance:** peak GPU memory, data-load time, seconds per optimizer step, total epoch time and any out-of-memory failures. Full-volume and grid runs may have different numbers of optimizer steps; compare wall-clock as well as step counts.
3. **Image fidelity:** masked magnitude/phase MAE and RMSE on fixed validation; held-out test metrics only after selection. For the project goal, also evaluate resulting SWI/QSM reconstruction and vessel visibility on original-versus-hybrid multi-echo GRE stacks.
4. **Reproducibility:** store effective dashboard configuration, source dataset version, preprocessing signatures, split assignments and trained checkpoint IDs. No patient is allowed in more than one partition.

**Hypothesis:** Cropping should reduce wasted computations in both approaches. Patches should retain the lower peak memory requirement; full volumes may improve long-range consistency, but at higher activation-memory cost. This is a hypothesis to measure, not an established performance claim.

## Implementation map (code reuse)

- `echo_gen_config.py` and `echo_gen_defaults.json`: one shared, validated configuration.
- `echo_gen_dashboard.html`: controls in the existing preprocessing/training form.
- `echo_gen_preprocessing.py`: `brain_mask_bounding_box()` → `build_training_spatial_specs()` → existing `build_chunk_specs()` / `extract_chunk()` for *both* HDF5 chunks and the fast NumPy cache.
- `echo_gen_training.py`: **unchanged** dynamic input-echo sampling, direct batch-size-one loop, generator, critic and losses.
- `test_echo_gen_unet_batching.py`: isolated, synthetic-array tests for crop bounds, identical spatial alignment, padding, world transforms and variable cache sizes.

## Limitations

The cropped chunk NIfTI affine is preserved, but raw HDF5 and NPY arrays do not store a NIfTI header; their coordinate reconstruction depends on the manifest and original prepared NIfTIs. Crop mode does not create a new whole-volume NIfTI for downstream SWI/QSM; those reconstruction pipelines should continue using their correctly registered source/hybrid volume assembly. This change does not introduce random coordinate patches, multi-subject minibatches, automatic tiling/stitching for inference, or batch-size >1; these would require separate design and validation.
