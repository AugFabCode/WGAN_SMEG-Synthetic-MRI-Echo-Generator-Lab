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

The cohort must support separate training and evaluation subjects. All sessions/runs from a subject stay in the same partition. Preprocessing prepares both magnitude and phase; selecting magnitude-only training does not remove the current requirement for complete prepared data in both modalities.

## 4. Launch the dashboard

```bash
python echo_gen_dashboard.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

1. Enter the unprocessed data directory and choose phase handling, BET fraction and chunk geometry.
2. Leave cache paths blank for automatic selection, or specify the existing folders when both layouts are present.
3. Configure the generator filters, TE embedding and regularization.
4. Select modality/device, epoch limit, learning rates, loss weights and critic update counts.
5. Run **Preprocess data**, wait for completion, then run **Start training**.

Advanced JSON values override visible fields. Nested objects are replaced, not merged: supplying a `TRAINING_HYPERPARAMETERS` object requires complete entries for both modalities. **Export configuration** downloads the current form settings; it does not start a job.

The monitor shows process status and a bounded tail of the log. Curves update after each completed training-and-validation epoch. During a long first epoch, an empty chart is expected; use batch messages in the log to check progress. The diagram is an architecture schematic, not a live tensor or generated-MRI preview.

The dashboard permits one active job per project. Browser closure does not stop the worker. Restarting the dashboard loses its in-memory monitor connection to that worker; saved logs and the project lock remain. There is currently no dashboard stop button or job reattachment interface.

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
