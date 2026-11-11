# Configuration reference

[README](README.md) · [Getting started](GETTING_STARTED.md) · [Pipeline](PIPELINE.md)

## Configuration files and precedence

[echo_gen_defaults.json](echo_gen_defaults.json) lists accepted JSON keys and their defaults. [echo_gen_config.py](echo_gen_config.py) resolves paths, applies overrides and validates settings. The smaller [example configuration](echo_gen_config.example.json) is a starting point; replace its dataset placeholder before processing.

Commands accept `--config PATH`. Omitted keys retain defaults. Unknown keys are rejected. The dashboard builds a configuration from its fields and then applies Advanced JSON on top. **Nested objects are replaced as a whole**, not recursively merged. For example, overriding `TRAINING_HYPERPARAMETERS` must supply all seven fields for both magnitude and phase.

JSON settings are not arbitrary Python code. Use arrays for tuples and JSON `true`, `false` and `null`. Input-count probability keys are converted from JSON strings to integers.

`--project-root` controls the project location. Relative source/cache paths resolve against it. Output roots and `PIPELINE_PROFILE` are derived by the Python configuration and are not editable JSON keys. `ECHO_GEN_PROJECT_ROOT`, `ECHO_GEN_CONFIG` and `ECHO_GEN_ACTION` are internal/environment hooks used by the runner; prefer the documented CLI flags.

Some keys remain for notebook compatibility rather than providing independent active controls. In particular, the current device setup always selects float32, so `TRAINING_ENABLE_MIXED_PRECISION` does not enable a mixed-precision policy. Logical device placement follows `TRAINING_DEVICE_MODE`; the direct Python loop does not activate tf.data prefetch/parallel-I/O knobs. Verify the relevant implementation before using advanced compatibility settings in experiments.

## Data and preprocessing

| Setting | Default | Purpose |
|---|---|---|
| `SOURCE_ROOT` | Original notebook's external ABRIM path | Replace with your dataset root |
| `PHASE_MODE` | `"auto"` | `existing`, `compute`, or existing-first automatic selection |
| `ALLOW_PROTOCOL_TE_FALLBACK` | `true` | Permit the declared nine-echo protocol when TE metadata is absent |
| `FILTERED_PHASE_OVERRIDES` | `{}` | Per-case path to an existing 4D filtered-phase volume |
| `PHASE_4D_OVERRIDES` | `{}` | Per-case path to a genuine raw 4D phase volume |
| `CLEAR_SWI_REQUIRES_X` | `false` | Enable the modified CLEAR-SWI CLI's `-x` phase argument |
| `CLEAR_SWI_EXTRA_ARGS` | `[]` | Additional CLEAR-SWI arguments |
| `BET_REFERENCE_ECHO` | `"TE3"` | Magnitude echo used for masking |
| `BET_FRACTION` | `0.20` | BET fractional threshold |
| `BET_EROSION_RADIUS_MM` | `2.0` | Erosion radius in millimetres |
| `CHUNK_GRID` | `[5,5,3]` | Chunk counts along spatial axes |
| `CHUNK_OVERLAP_VOXELS` | `20` | Chunk overlap/padding setting |
| `SMALL_DATASET_MODE` | `false` | Opt-in original two-way experiment (80% training / 20% validation / 0% independent test) |
| `TRAINING_SPLIT_PERCENT` | `70.0` | Percentage of subjects assigned to training (defaults to 80 in small-dataset mode if omitted) |
| `EVALUATION_SPLIT_PERCENT` | `15.0` | Percentage assigned to evaluation/checkpoint selection (defaults to 20 in small-dataset mode if omitted) |
| `TEST_SPLIT_PERCENT` | `15.0` | Held-out test percentage, forced to 0 in small-dataset mode |
| `REBUILD_PATIENT_SPLIT` | `false` | Explicitly rebuild legacy/changed partitions during preprocessing; also requires `OVERWRITE_DERIVED=true` |
| `TRAIN_EVAL_SPLIT_SEED` | `42` | Seed for a newly created split |
| `OVERWRITE_PREPROCESSING` | `false` | Explicitly rebuild prepared image outputs |
| `OVERWRITE_DERIVED` | `false` | Explicitly rebuild stale statistics/chunks/cache |

Per-case override keys omit the initial `sub-`, for example `001_run-1` or `001_ses-01_run-1`. Use absolute paths in override maps; they are not normalized through the source/cache path resolver. The split workbook stores percentages and seed in a `split_settings` sheet alongside `patient_split`. Matching settings reuse its subject assignments. Changing percentages/seed or migrating a legacy workbook requires an explicit preprocessing rebuild. Percentages must sum to 100; training and evaluation must be positive, while test may be zero. Every nonzero partition needs at least one subject; integer counts approximate the requested percentages deterministically. All sessions/runs from a subject stay together. The retired `TRAIN_EVAL_SPLIT_FRACTION` key must be removed from old configuration JSON. With `SMALL_DATASET_MODE=true`, omit the three percentage keys to use original 80/20/0 defaults; the second partition is **validation** (including best-checkpoint selection and early stopping), not an unbiased final test set. Only two independent subjects are required. Legacy workbooks containing only `patient_split` with matching original 80/20 allocations are reused read-only, including `train/test` aliases (legacy `test` becomes internal `eval`). They are not silently repartitioned; changing the allocation requires explicit rebuild. Existing derived caches may still require `OVERWRITE_DERIVED=true` to refresh signatures. Do not resume checkpoints across split-mode changes. New percentages and workbook fingerprints isolate model run identities from old splits.

Preprocessing cache reuse is signature-checked. A changed source, cohort, chunk geometry or normalization setup may require a rebuild. Only enable overwrite flags when that rebuild is intended. The high-level preparation function rebuilds derived outputs together when needed; lower-level compatibility flags such as `OVERWRITE_CHUNKS` are not a substitute for `OVERWRITE_DERIVED`.

## Cache layout selection

| Override | Notebook folder | Repository folder |
|---|---|---|
| `CEASET_CHUNK_ROOT` | `preprocess/training_chunks` | `preprocess/09_chunks_553` |
| `CEASET_FAST_NPY_CACHE_ROOT` | `preprocess/training_cache` | `preprocess/10_training_cache_npy` |

Leave these overrides absent or `null` to detect existing folders. A folder containing a CSV manifest takes priority. If both names for a cache contain CSV manifests, choose an explicit path. If neither contains a manifest, an existing notebook folder takes priority, then an existing repository folder; otherwise the notebook name is used.

The resolver does not move folders. Known stale cache-root names in manifests can be mapped to the selected root in memory. Arrays must actually exist; path resolution cannot reconstruct missing cache files. `TRAINING_USE_FAST_NPY_CACHE` and `TRAINING_REQUIRE_FAST_NPY_CACHE` both default to `true`.

## Model settings

| Setting | Default | Purpose |
|---|---|---|
| `MODEL_FILTERS` | `[16,24,32,48]` | Four positive encoder filter counts |
| `MODEL_TE_EMBED_DIM` | `48` | Learned TE embedding dimension |
| `MODEL_L2` | `0.00005` | Model L2 regularization |
| `MODEL_LEAKY_RELU` | `0.2` | Leaky ReLU slope |
| `RAW_TE_FILM_ENABLED` | `true` | Additional physical TE conditioning |
| `RAW_TE_PHYSICAL_SCALE_MS` | `40.0` | Raw-TE path scaling |
| `MIN_INPUT_ECHOES`, `MAX_INPUT_ECHOES` | `1`, `4` | Dynamic echo-set range |
| `INPUT_COUNT_PROBABILITIES` | `1: .32, 2: .30, 3: .23, 4: .15` | Selected input-count distribution |

If changing the input-count range, provide a matching probability map covering that range and summing to one. Such changes also need scientific/memory validation; a successful configuration check is not a completed training test.

## Training settings

| Setting | Default | Purpose |
|---|---|---|
| `TRAINING_MODALITIES_TO_RUN` | `["magnitude","phase"]` | Sequential training selection |
| `TRAINING_DEVICE_MODE` | `"single_gpu"` | `single_gpu`, `dual_gpu`, or `cpu` |
| `TRAINING_EPOCHS` | `40` | Final epoch limit, not additional epochs on resume |
| `TRAINING_BATCH_SIZE` | `1` | Required by the current direct loop |
| `TRAINING_EVAL_BATCH_SIZE` | `1` | Required by the current direct loop |
| `TRAINING_DYNAMIC_ECHO_AXIS` | `true` | Required by the current direct loop |
| `TRAINING_USE_DIRECT_PYTHON_LOOP` | `true` | Required stable execution path |
| `TRAINING_SEED` | `42` | Training random seed |
| `WGAN_GP_LAMBDA` | `10.0` | Gradient-penalty coefficient |
| `WGAN_GP_DOWNSAMPLE_FACTOR` | `1` | Full-resolution gradient penalty; larger values approximate it |
| `CRITIC_UPDATES_BY_MODALITY` | Magnitude `2`, phase `1` | Critic steps per generator batch |
| `TRAINING_SAVE_EVERY_EPOCHS` | `1` | Periodic checkpoint interval |
| `ENABLE_EARLY_STOPPING` | `true` | Stop on stagnant fixed validation |
| `EARLY_STOPPING_START_EPOCH` | `8` | First epoch where stopping is considered |
| `EARLY_STOPPING_PATIENCE` | `6` | Epochs without meaningful improvement |
| `EARLY_STOPPING_MIN_DELTA` | `0.0002` | Minimum meaningful fixed-validation MAE improvement |

`TRAINING_HYPERPARAMETERS` has complete magnitude and phase objects:

| Field | Magnitude | Phase |
|---|---:|---:|
| `generator_lr` | `0.00003` | `0.00003` |
| `critic_lr` | `0.0001` | `0.0001` |
| `adv_weight` | `0.08` | `0.06` |
| `rec_weight` | `5.6` | `5.6` |
| `edge_weight` | `0.1` | `0.1` |
| `perc_weight` | `0.06` | `0.03` |
| `physics_residual_weight` | `0.35` | `0.35` |

The legacy `physics_residual_weight` name controls the coarse structural physics term. `perc_weight` has no effect while `TRAINING_ENABLE_PERCEPTUAL_LOSS=false` (the default). Enabling perceptual loss requires `models/RadImageNet-ResNet50_notop.h5`.

## Naming, saving and resume

Default run-name pattern:

```text
wgangp_f16-24-32-48_te48_<12-character-configuration-hash>
```

The hash includes resolved configuration and available preprocessing signature, split workbook, statistics and cache-manifest fingerprints. It does not hash every MRI/cache array. The integration revision is included, but it is not an automatic hash of all source code. Record the Git commit separately when comparing experiments.

`PIPELINE_ACTION`, `TRAINING_EPOCHS`, `TRAINING_RESUME` and `RUN_INSTANCE` are excluded from the configuration hash. `RUN_INSTANCE`, when set, is appended to the run name. Paths and most other settings remain part of identity, so relocating data or changing a runtime setting may select a different run. The run hash shown for preprocessing can differ from a later training hash once the prepared manifests exist.

| Intent | Settings |
|---|---|
| Resume a matching default run | `TRAINING_RESUME=true`, `RUN_INSTANCE=""` |
| Extend that run to epoch 60 | Same configuration, `TRAINING_EPOCHS=60` |
| Start a separate fresh run | `TRAINING_RESUME=false`, `RUN_INSTANCE=""` |
| Resume a previously named fresh run | Its saved `RUN_INSTANCE`, `TRAINING_RESUME=true`, otherwise matching settings |

Fresh runs receive a unique instance suffix. The dashboard fills the assigned instance field. Reusing an instance with resume disabled is rejected when matching weights/state already exist. `RUN_INSTANCE` permits up to 64 letters, digits, underscores or hyphens.

Resume prefers the optimizer-inclusive state. If only matching generator epoch weights exist, the existing fallback loads the generator and resets critic/optimizer state. Old untagged notebook checkpoints are not automatically selected. When the epoch limit has already been reached, raise the limit if more training is intended.

Within `models/SWI-CEASET-ABRIM-V8/<modality>/`:

| Artifact | Meaning |
|---|---|
| `generator_<run>_epochNNN.weights.h5` | Periodic generator weights |
| `critic_<run>_epochNNN.weights.h5` | Periodic critic weights |
| `generator_<run>_best_fixed_eval.weights.h5` | Best generator by fixed-validation MAE |
| `configuration_<run>.json` | Resolved settings, fingerprints, paths and launch metadata |
| `best_fixed_eval_<run>.txt` | Best epoch and MAE |
| `training_state_<run>/` | TensorFlow generator/critic/optimizer state; last five managed checkpoints |

History lives at `training/SWI-CEASET-ABRIM-V8/<modality>/history_<run>.csv`. Dashboard launch configs and logs live directly under the training profile directory. The last completed epoch is saved even if early stopping occurs between periodic save intervals. Forced termination can still lose work since the last successfully saved checkpoint.

A resolved `configuration_<run>.json` contains derived fields and metadata; it is **not** directly accepted as a CLI override file. Use the dashboard-exported config or saved `dashboard_config_*.json` to rerun, and consult the resolved record for provenance. For CLI-created fresh runs, transfer the recorded `RUN_INSTANCE` into the original override file and enable resume.

## Live training MRI preview

`TRAINING_PREVIEW_ENABLED=true` enables a preview after the first training batch, then at most once per `TRAINING_PREVIEW_INTERVAL_SECONDS` (default 60; minimum 10). These display-only settings are excluded from checkpoint identity, so they do not select new model weights. No additional generator inference is performed.

The preview displays **all active K=1–4 acquired input echo slices** from the current training batch, not only the first echo. `INPUT_COUNT_PROBABILITIES` controls the *probability* of sampling K input echoes, not a permanent number of input panels; e.g. `{1: .20, 2: .20, 3: .20, 4: .40}` gives four input panels in about 40% of sampled batches. Each input card is labeled with its source echo identifier and **physical TE in milliseconds**; generator output, acquired target, and the absolute-error panel are labeled with the target TE. Mixed input/target TEs and changes across preview frames are normal due to training sampling.

The preview is a central **transversal slice of the current training chunk**, not a whole-brain mid-slice. The source affine and preprocessing transpose define the slice axis. Captured panels are bounded to 256 pixels per side. The input slices are extracted on the generator device and transferred only after slicing and masking, avoiding full-volume transfers to the critic device. The generator output comes from the first existing critic update; preview capture neither calls the generator again nor changes the optimization steps.

Each display panel is independently enlarged by exactly **3× pixel replication** (nearest-neighbor): each input pixel becomes a uniform 3×3 block of **nine identical pixels**. There is no bilinear/bicubic interpolation, resampling, or artificial medical detail. The panels are transmitted as one RAM-only lossless PNG and displayed in responsive individual canvas cards using the original pixel values. The dashboard holds only the latest preview per modality; no preview images/history files are written. Existing checkpoints, raw process logs and training metrics are still saved as normal.

All acquired inputs, generated image and target share one sample-normalized grayscale display window (phase uses a symmetric window); the orange absolute-error panel has an independent error scale. Preview TE scale (e.g. `TE_EMBEDDING_SCALE_MS=38.0`) affects model conditioning, **not** the acquisition TE labels or number of preview panels. Different sampled echo counts and spatial flips are expected; this is not a fixed longitudinal evaluation. An empty central masked slice is displayed as empty rather than selecting a different slice.
