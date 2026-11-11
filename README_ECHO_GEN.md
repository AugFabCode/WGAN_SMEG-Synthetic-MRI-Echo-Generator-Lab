# Echo Gen: preprocessing, training and local dashboard

Repository: https://github.com/AugFabCode/WGAN_SMEG-Synthetic-MRI-Echo-Generator-Lab

Extracted from the supplied `pp_to_train_echo_generator_RECENT(5).ipynb`.
The original notebook is unchanged. This package is self-contained and does not execute notebook cells at runtime.

## Install into the existing project

Extract the files directly into your **existing Echo_Gen project root**, beside `preprocess/`, `training/`, `models/`, `evaluation/`, `architecture/`, `synthetic/`, and `inference/`. Do not extract your MRI data or move existing folders. Existing data folders and files are preserved.

Use your working notebook environment. The dashboard itself needs only Python's standard library. For a new environment, use Python 3.11 and the package versions retained from the notebook:

```bash
python -m pip install -r requirements-echo-gen-training.txt
```

For preprocessing alone, install `requirements-echo-gen-preprocess.txt`. FSL (`bet`, `fslmaths`) and, when recomputing filtered phase, CLEAR-SWI must already be installed/configured as in the notebook. GPU drivers are external dependencies.

## Dashboard

From the project root in your training environment:

```bash
python echo_gen_dashboard.py
```

Open http://127.0.0.1:8765. If launching from somewhere else:

```bash
python /path/to/Echo_Gen/echo_gen_dashboard.py --project-root /path/to/Echo_Gen --port 8765
```

1. Enter the unprocessed ABRIM directory. It contains the `sub-*` BIDS folders, optionally with sessions.
2. Set preprocessing parameters or select existing cache directories.
3. Set generator filters, TE embedding, regularization, FiLM, device, epochs, and modality-specific learning rates/loss weights/critic updates. Advanced JSON exposes the remaining settings in `echo_gen_defaults.json`.
4. Click **Preprocess data** for new data or changed chunk settings. Wait for completion.
5. Click **Start training** to consume the prepared data.

The diagram shows the actual CEASET generator/critic flow and WGAN-GP objective. Six curves display completed-epoch generator loss, critic loss, Wasserstein gap, gradient penalty, reconstruction loss and fixed-validation MAE. Batch progress streams into the log. Empty plots are intentional until validation completes; no example metrics are displayed as real results.

The current notebook's direct training loop requires batch size 1 and a dynamic echo axis. The dashboard displays this constraint; incompatible overrides are rejected rather than silently changing the algorithm. Mixed precision and compiled-step settings retain their notebook behavior.

Only one processing/training job may own a project at a time. Closing the browser does not stop it. Restarting the dashboard does not reattach its monitor to an older worker: inspect the saved log and process before starting another job. If a worker is forcibly killed, verify the recorded PID has stopped before removing `.echo_gen_job.lock`. No automatic stale-lock deletion is performed.

The server binds to localhost. For a remote training machine use an SSH tunnel, for example `ssh -L 8765:127.0.0.1:8765 user@training-host`, then open the same local URL. Do not expose this local process-control dashboard directly to the internet.

## Independent scripts

Export a configuration from the dashboard, then:

```bash
python preprocess_echo_gen.py --config echo_gen_config.json
python train_echo_gen.py --config echo_gen_config.json
```

Either command accepts `--project-root /path/to/Echo_Gen`. Relative dataset/cache paths resolve against that project root. Configuration validation without MRI processing, TensorFlow or output creation:

```bash
python train_echo_gen.py --config echo_gen_config.json --dry-run
```

Training never invokes `prepare_training_data`. Preprocessing never imports TensorFlow or the training module. The shared data readers live in `echo_gen_preprocessing.py`; the network, sampling and losses live in `echo_gen_training.py`.

## Folder preservation

The existing top-level folders and notebook profile/modality layout are retained. Nothing is moved, deleted or renamed. The scripts use the original `preprocess/` outputs including physical images, masks, metadata, split and statistics. Cache selection supports both:

| Cache | Notebook layout | Repository layout |
|---|---|---|
| Chunks | `preprocess/training_chunks` | `preprocess/09_chunks_553` |
| Fast NumPy | `preprocess/training_cache` | `preprocess/10_training_cache_npy` |

Existing manifests take priority. If both versions contain manifests, explicitly choose the intended cache path in the dashboard. On an empty project the notebook defaults are used unless the repository folders already exist. Stale manifest paths are resolved in memory against the selected cache root, without moving arrays or rewriting manifests.

Normal preprocessing still generates required outputs within these roots, and explicit overwrite settings retain the original notebook semantics. It does not migrate legacy `ceaset_abrim_v8` datasets automatically.

## Automatic model naming and saving

Example name (the hash depends on your resolved configuration):

`wgangp_f16-24-32-48_te48_<12-character-hash>`

Within `models/SWI-CEASET-ABRIM-V8/<magnitude-or-phase>/`:

- `generator_<run-name>_epochNNN.weights.h5`
- `critic_<run-name>_epochNNN.weights.h5`
- `generator_<run-name>_best_fixed_eval.weights.h5`
- `configuration_<run-name>.json` (resolved settings, paths, data-manifest fingerprints and repository)
- `best_fixed_eval_<run-name>.txt`
- `training_state_<run-name>/` (the existing optimizer-inclusive checkpoint mechanism, isolated by configuration)

Epoch histories remain in `training/SWI-CEASET-ABRIM-V8/<modality>/history_<run-name>.csv`. Dashboard launch configurations and logs are saved under the same existing profile root. A checkpoint is always retained for the final completed epoch even when it falls between save intervals. Best-model selection remains fixed-validation MAE.

The hash covers resolved architecture, loss, sampling, preprocessing and runtime settings, plus available preprocessing signature/split/statistics/cache-manifest contents. Paths are part of the identity. It does not hash every MRI array. The existing artifact validation still checks the prepared dataset. Changing configuration or relocating data can intentionally select a different run.

With **Resume matching configuration** checked, the same settings load that run's latest full state. Epoch limit is excluded from identity, so it can be extended. Old untagged notebook checkpoints are not automatically resumed.

For a fresh experiment, uncheck resume and clear **Run instance**. A unique suffix is assigned and saved; the dashboard fills the instance field. To resume this fresh experiment later, use its saved configuration/instance and set `TRAINING_RESUME` to `true`. `RUN_INSTANCE` is excluded from the hash but appended to the name. An explicitly reused instance with resume disabled is rejected if checkpoints already exist; clear it when you want a separate run.

## Checks and limitations

```bash
python -m unittest -v test_echo_gen.py test_echo_gen_data.py
```

Checks cover configuration identity and resume, fresh-run separation, invalid settings, both cache layouts, dry-run side effects, TensorFlow-free preprocessing import, synthetic chunk coverage/padding, a known exponential magnitude baseline, and NumPy cache round-trip. HTTP smoke checks cover status/defaults, invalid launch rejection, worker failure logs and project-lock cleanup. Python compilation and dashboard JavaScript syntax were checked.

Full ABRIM preprocessing and GPU training were **not executed** in the delivery environment: the real dataset, FSL/CLEAR-SWI, TensorFlow and a compatible GPU were unavailable. Browser rendering could not be verified because the test browser was blocked by the execution environment. The optional notebook inference/qualitative-evaluation cells remain in the unchanged original notebook; they are not run by these entry points. To use newly named weights there, supply the explicit checkpoint path.
