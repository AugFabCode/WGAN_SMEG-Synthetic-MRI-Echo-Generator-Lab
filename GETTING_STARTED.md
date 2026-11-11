# Getting started

[README](README.md) · [Configuration](CONFIGURATION.md) · [Troubleshooting](TROUBLESHOOTING.md)

## 1. Prepare the environment

Run these commands from an existing checkout of [the repository](https://github.com/AugFabCode/WGAN_SMEG-Synthetic-MRI-Echo-Generator-Lab). To obtain a new checkout:

```bash
git clone https://github.com/AugFabCode/WGAN_SMEG-Synthetic-MRI-Echo-Generator-Lab.git
cd WGAN_SMEG-Synthetic-MRI-Echo-Generator-Lab
```

Reuse your working notebook environment if available. For a separate environment on Linux:

```bash
python3.11 -m venv .venv-ceaset
source .venv-ceaset/bin/activate
python -m pip install -r requirements-echo-gen-training.txt
```

The requirements retain the notebook's TensorFlow 2.16.1 and Keras 3.3.3 versions. Installing Python dependencies does not install the NVIDIA driver, FSL, CLEAR-SWI or the dataset. The preprocessing backend calls Bash and FSL; the documented workflow targets Linux.

For preprocessing alone:

```bash
python -m pip install -r requirements-echo-gen-preprocess.txt
```

The dashboard uses Python's standard library and launches workers with **the same Python interpreter** that started the dashboard. Start it from the environment in which you installed the scientific/training dependencies.

## 2. Configure external preprocessing tools

FSL must provide `bet` and `fslmaths`. The pipeline sources the first existing FSL setup file from `$FSLDIR/etc/fslconf/fsl.sh`, `/etc/fsl/fsl.sh`, or `~/fsl/etc/fslconf/fsl.sh` (with `/usr/local/fsl` as the default FSLDIR location). It also sources `~/.bash_aliases` and prepends `~/CLEARSWI` to the subprocess PATH.

If the required filtered-phase volume already exists, choose `PHASE_MODE="existing"`. Otherwise, `auto` uses an existing volume when available and computes it when necessary. `compute` explicitly recomputes phase through CLEAR-SWI. Genuine phase inputs are required for computation; SWI images cannot substitute for raw phase. See [Configuration](CONFIGURATION.md) for per-case overrides.

## 3. Identify the dataset root

`SOURCE_ROOT` points to the directory containing BIDS-style `sub-*` folders. Supported inputs use patterns such as:

```text
sub-001/anat/sub-001_run-1_echo-1_part-mag_MEGRE.nii.gz
sub-001/anat/sub-001_run-1_echo-2_part-mag_MEGRE.nii.gz
```

There must be nine acquired magnitude echoes per accepted case, plus a suitable filtered-phase volume or genuine phase source. Session directories/entities and uncompressed `.nii` files are supported. JSON `EchoTime` metadata supplies TE values, converted to milliseconds; protocol fallback is enabled by default and recorded per case. The default fallback is 6.14, 10.14, 14.14, 18.14, 22.14, 26.14, 30.14, 34.14 and 38.14 ms. Disable fallback if missing metadata should be a hard error.

The default cohort must contain at least three independent subjects for separate training, evaluation and test partitions. For experiments with only two or more subjects, enable **Small-dataset experiment** under **01 · Data & preprocessing**. This selects the original 80% training / 20% validation split (0% independent test) and automatically enables the explicit derived-data rebuild checkbox in the dashboard. The second partition drives validation/early stopping/checkpoint selection and is not an unbiased final test set. All sessions/runs from a subject stay in the same partition. Preprocessing prepares both magnitude and phase; selecting magnitude-only training does not remove the current requirement for complete prepared data in both modalities.

## 4. Launch the dashboard

```bash
python echo_gen_dashboard.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

1. Enter the unprocessed data directory and choose phase handling, BET fraction and chunk geometry.
2. Set **Training (%)**, **Evaluation (%)** and **Test (%)** under **01 · Data & preprocessing** (defaults **70 / 15 / 15**, total 100). Leave cache paths blank for automatic selection, or specify the existing folders when both layouts are present.
3. Configure the generator filters, TE embedding and regularization.
4. Select modality/device, epoch limit, learning rates, loss weights and critic update counts.
5. Run **Preprocess data**, wait for completion, then run **Start training**.

For the original two-way workbook, small-dataset mode accepts matching legacy 80/20 training/evaluation or train/test assignments without changing the workbook. The dashboard enables derived-cache rebuild when you switch to this mode, because prior cache signatures may not be compatible; already-prepared source volumes remain reusable unless their preprocessing settings changed. If source images or cache manifests are missing, the original raw ABRIM source is still needed. Never interpret the legacy second partition's validation metrics as held-out test results.

For existing prepared data, enable **Rebuild changed/legacy partitions and derived caches** and run **Preprocess data** when changing percentages/seed or migrating the old two-way split. This checkbox also sets `OVERWRITE_DERIVED=true`; Advanced JSON can override it. The previous split workbook is archived with a unique `.backup-…xlsx` name in the same folder, and derived caches are rebuilt under the existing roots. Matching settings preserve assignments even with the checkbox enabled. Old cache files may remain but only the regenerated manifests define the active cohort. Do not resume a model trained on the old split; the new split changes run identity. Test data is held out from training and checkpoint selection; final test scoring is a separate future workflow.

Advanced JSON values override visible fields. Nested objects are replaced, not merged: supplying a `TRAINING_HYPERPARAMETERS` object requires complete entries for both modalities. **Export configuration** downloads the current form settings; it does not start a job.

The page is titled **WGAN_SMEG - Synthetic Multi Echo Generator - Training Suite**. Its left column contains **01 · Data & preprocessing**, **02 · Model architecture**, and **03 · Training setup**. Magnitude and phase hyperparameters are in expandable sections; magnitude is expanded initially. Batch size is fixed at 1 by the current training loop.

The right column contains **WGAN-GP architecture**, **Active Run**, **Live training MRI output · transversal**, **Event feed**, and **Process log**. The diagram is an architecture schematic. Active Run shows generator loss, critic loss, Wasserstein gap, fixed validation MAE, gradient penalty and reconstruction loss. Curves update after each completed training-and-validation epoch. During a long first epoch, empty charts are expected; use batch messages in the log to check progress.

### Configure live MRI previews before starting training

Both the controls and the empty preview panel are visible before any job starts. In **03 · Training setup**, below early-stopping settings:

1. Enable **Live training MRI preview (central transversal slice)**.
2. Set **Preview interval (seconds)** to **20**. The repository default is 60 seconds; the minimum is 10 seconds.
3. Click **Start training** after preparing the data. Editing the form does not change a running worker's settings.
4. Under **Active Run**, choose **Display modality** to select magnitude or phase for both charts and previews. A modality without a preview yet shows an empty panel.

The four panels show input echo, generator output, acquired target and absolute error. They use the central transversal slice of the current training chunk, not necessarily the middle of the full volume. The generated image is the existing output used in the first critic update, published after that batch completes; no extra inference pass is requested. Input/output/target share a grayscale window; the orange error map uses its own scale. Case, echo times, epoch, batch and capture time identify the current sample. Training samples and augmentation can change between updates.

Each captured panel is limited to 256 pixels per side and upscaled **2× with exact pixel replication (nearest-neighbor) on a background CPU thread** before display. Each original display pixel becomes four identical pixels in a 2×2 block. The image is displayed at its intrinsic size in a scrollable area, with pixelated browser rendering to avoid smoothing. Upscaling adds no acquired detail. The plane follows native acquisition orientation, without oblique resampling or anatomical left/right labels.

Previews are **memory-only**: the dashboard keeps the latest image per modality and replaces it on update. No preview image files are saved. A new launch clears previews; stopping the server loses them. Configuration files, process logs and trained-model checkpoints are still saved normally. Standalone CLI training without a dashboard receiver skips preview capture.

Twenty seconds is a capture interval, not a guaranteed display deadline: batch completion, rendering and dashboard polling can delay an update. The design avoids additional generator passes and bounds the transferred slice size, but an A6000 slowdown below 5% has not been measured or guaranteed. Compare representative training throughput with previews enabled and disabled on your server.

If these controls are missing after updating the checkout, restart the dashboard from the updated directory and refresh the browser. An already-running worker must finish or be stopped separately before launching with updated settings; restarting the dashboard does not reconnect it.

The dashboard permits one active job per project. Browser closure does not stop the worker. Restarting the dashboard loses its in-memory monitor connection to that worker; saved logs and the project lock remain. Use **Stop training** to request a cooperative interruption of the active training worker. The status changes to **stopping**, then **stopped** when the worker exits. A current GPU operation/data load or checkpoint write may need to finish before a safe Python boundary is reached. The unfinished epoch is discarded; resume uses the last successfully saved checkpoint (which can be earlier than the last completed epoch according to the save interval). If no checkpoint exists yet, training starts over. Keep the same configuration and run instance and enable **Resume matching configuration**. The project lock is released on handled interruption. The button does not stop preprocessing, and job reattachment after a dashboard restart is not implemented.

## 5. Use the independent CLI

Create a local configuration:

```bash
cp echo_gen_config.example.json echo_gen_config.json
```

Edit `SOURCE_ROOT` to your dataset path. Check resolved configuration before launching a real workload:

```bash
python preprocess_echo_gen.py --config echo_gen_config.json --dry-run
python train_echo_gen.py --config echo_gen_config.json --dry-run
```

Dry-run does not import TensorFlow or start preprocessing. It validates settings and prints resolved paths/run identity; it does **not** prove the dataset, external tools, GPU or caches are ready.

Then execute each stage separately:

```bash
python preprocess_echo_gen.py --config echo_gen_config.json
python train_echo_gen.py --config echo_gen_config.json
```

Both entry points accept `--project-root`. By default the project root is the script directory, rather than the terminal's working directory. Relative dataset/cache settings resolve against that root. The `--config` file path itself is resolved from the current shell directory.

```bash
python /path/to/Echo_Gen/train_echo_gen.py \
  --project-root /path/to/Echo_Gen \
  --config /path/to/echo_gen_config.json
```

## Remote access

On the training host:

```bash
python echo_gen_dashboard.py --project-root /path/to/Echo_Gen --port 8765
```

On your own computer:

```bash
ssh -L 8765:127.0.0.1:8765 user@training-host
```

Open the same localhost URL locally. Dataset paths entered into the dashboard refer to the **training host's filesystem**. Keep the dashboard bound to localhost; it is a local process-control interface, not a public multi-user service.

### Event feed and original process log

The event feed is visible by default and presents recent output as categorized events: training progress, completed epochs, checkpoints, evaluation, preprocessing, setup, warnings and errors. Unrecognized lines appear as **Output**. It is a best-effort interpretation of text, not an authoritative training-state protocol. Entries appear newest first, up to 100 from the recent 32 KB of output. Original timestamps are retained if present; the feed does not invent event times. Repeated polling does not append duplicates, and switching to a new job replaces the feed.

The original **Process log** section remains available. **Show original process log** is unchecked by default on each page load; check it to display the raw tail, or uncheck it to opt out again. Hidden raw output is not included in status responses. The existing full process log continues to be saved normally, so opt-out affects dashboard display only. No separate event-feed file is written and no MRI previews are saved.
