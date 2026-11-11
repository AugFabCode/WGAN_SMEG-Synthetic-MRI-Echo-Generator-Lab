# Troubleshooting

[README](README.md) · [Getting started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md)

## Diagnose configuration first

```bash
python train_echo_gen.py --config echo_gen_config.json --dry-run
```

Check the resolved source/project/cache paths, modality, device and run name. Dry-run does not validate a complete dataset or GPU environment. Full worker errors appear in the dashboard log and saved `training/SWI-CEASET-ABRIM-V8/dashboard_*.log` files.

## Data preparation and cache errors

| Symptom | What to check |
|---|---|
| Unprocessed data directory does not exist | Enter the host's dataset directory; remote dashboard paths are not local laptop paths |
| No matching cases or incomplete echoes | Check BIDS `sub-*`, session/run entities, `part-mag_MEGRE` names and all nine echoes |
| Missing filtered phase/genuine phase source | Select a valid phase mode; provide actual phase or an existing filtered-phase override |
| Affine/grid mismatch | Confirm matching acquisitions and spatial alignment; renaming files does not correct geometry |
| Derived data has no completion marker | Run preprocessing to completion; do not fabricate `derived_data_signature.json` |
| Dataset/split/chunk settings changed | Confirm the intended configuration; if a rebuild is intended, enable `OVERWRITE_DERIVED` and rerun preprocessing |
| Both cache layouts contain manifests | Set explicit `CEASET_CHUNK_ROOT` and/or `CEASET_FAST_NPY_CACHE_ROOT` |
| Incomplete fast cache or missing arrays | Check that selected folders contain arrays for both modalities and all echoes; rebuild if required |
| Cohort/split mismatch | Use statistics, manifests and split workbook from the same prepared cohort |

The split scripts do not migrate a legacy `ceaset_abrim_v8` tree. Preserve that data, inspect the selected paths, and deliberately choose or rebuild the current artifacts. Do not move folders merely to suppress a path error.

Some retained error messages mention notebook `PIPELINE_ACTION="prepare"` or a cache cell. In the script workflow, run `preprocess_echo_gen.py` or the dashboard's **Preprocess data** action.

## TensorFlow and GPU issues

Check the interpreter and TensorFlow-visible GPUs in the same environment used for the dashboard:

```bash
python -c "import sys, tensorflow as tf; print(sys.executable); print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

`single_gpu` requires one visible GPU; `dual_gpu` requires two. `CUDA_VISIBLE_DEVICES` can change the logical device mapping. Selecting dual-GPU mode does not create another device. Use `cpu` only for deliberately small checks.

For out-of-memory errors, inspect chunk dimensions and the chosen filter counts. Changing chunk geometry requires corresponding preprocessing; changing architecture starts a distinct configuration. Batch size is already fixed to one. A downsampled gradient penalty is an approximation, not an equivalent memory-only optimization. There is no validated minimum GPU-memory requirement for every possible source volume/configuration.

Each CLI worker is a fresh Python process, allowing memory growth to be configured before training initialization. If manually importing modules into a notebook after TensorFlow has initialized, restart that kernel before changing device setup.

If perceptual loss reports missing weights, either supply the expected local RadImageNet file or keep the default `TRAINING_ENABLE_PERCEPTUAL_LOSS=false`.

## Dashboard behavior

| Symptom | Explanation/action |
|---|---|
| Page does not open | Check the terminal for the actual port and startup error; use `--port` if occupied |
| No loss curves yet | Curves are written after completed validation epochs; inspect batch logs during the first epoch |
| Unexpected parameter values | Advanced JSON overrides visible controls |
| Job fails immediately | Inspect the process log for dependency, source-path, cache, configuration or GPU errors |
| Connection says disconnected | Confirm the dashboard server is still running and the SSH tunnel is open |
| Dashboard restarted but job still running | Reattachment is not implemented; inspect the existing saved log/process |
| Same settings create another run | Check paths, data fingerprints, other runtime settings and `RUN_INSTANCE` |
| Resume does no additional epochs | `TRAINING_EPOCHS` is the final epoch limit; increase it |
| Fresh run rejected due to checkpoints | Enable resume for that instance or clear/select a new instance |

The log panel displays only the most recent portion of the log; the on-disk log contains earlier output. It may include local paths and filenames, so review it before sharing publicly.

## Project lock and interruptions

The runner writes `.echo_gen_job.lock` with a PID, token and action to prevent simultaneous workers from mutating/reading one project inconsistently. Normal completion and handled Python errors remove it. Abrupt termination may leave it behind.

If starting another job is blocked:

1. Read the lock and identify its PID/action.
2. Verify on the training host whether that worker is still running. Do not assume it stopped because the browser closed.
3. Only after confirming the worker has stopped, remove the stale lock and start a new job.

Do not remove an active worker's lock. There is no automatic stale-lock deletion or dashboard stop action. Interrupted training resumes from the last successfully saved matching checkpoint, not necessarily the most recent batch. Preserve the saved configuration and instance when resuming.
