# Echo Gen · Synthetic MRI Echo Generator

[Repository](https://github.com/AugFabCode/WGAN_SMEG-Synthetic-MRI-Echo-Generator-Lab)

Echo Gen is a research pipeline for generating MRI echoes from a variable set of acquired echoes and a requested echo time (TE). Its core is a **physics-conditioned 3D Wasserstein GAN with gradient penalty (WGAN-GP)**, trained separately for magnitude and phase on ABRIM multi-echo gradient-echo (MEGRE) data.

The research goal is to investigate whether synthetic late echoes, combined with acquired early echoes, can support SWI and QSM reconstruction with shorter acquisitions. Reduced acquisition time, improved vessel visibility, and reduced noise are **hypotheses to evaluate**, not established results of this implementation.

## What is included

- Independent preprocessing and training commands, extracted from the source notebook.
- ABRIM discovery, filtered-phase handling, TE3 masks, physical-value normalization statistics, subject-level splits, chunks and a fast NumPy cache.
- A local dashboard with dataset paths, model parameters, training hyperparameters, a WGAN-GP diagram, epoch curves, low-overhead training MRI previews an event feed and optional process logs.
- Configuration-based checkpoint names, fixed-validation best-model selection and optimizer-inclusive resume.
- Compatibility with the existing project folders and both supported cache layouts.

## Quick start

Use your working Echo Gen environment, or create the Python 3.11 environment described in [Getting started](GETTING_STARTED.md). From the repository root:

```bash
python -m pip install -r requirements-echo-gen-training.txt
python echo_gen_dashboard.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765), enter the ABRIM directory, run **Preprocess data**, and then **Start training**. FSL is required for preprocessing; CLEAR-SWI is also required when filtered phase must be computed. The MRI dataset and trained weights are not bundled.

Set subject percentages under **01 · Data & preprocessing** (default **70% training / 15% evaluation / 15% test**). For small/legacy datasets, the opt-in **Small-dataset experiment** uses the original **80% training / 20% validation** split with no independent test; the 20% is used for checkpoint selection and early stopping, not unbiased final testing. The test partition is held out from training and model selection. **Stop training** interrupts at a safe boundary; resume uses the last saved checkpoint.

The dashboard's **03 · Training setup** includes the live MRI preview checkbox and **Preview interval (seconds)** before training starts. Set the interval to **20** for the requested cadence (default: 60). The preview displays all K=1–4 actual sampled acquired input echoes with individual echo/TE labels, plus generator output, acquired target and error. Each 2-D slice is enlarged exactly 3× by nearest-neighbor pixel replication (nine identical display pixels per source pixel). No previews are saved; the latest snapshot per modality remains in memory.

For command-line execution, copy [echo_gen_config.example.json](echo_gen_config.example.json) to `echo_gen_config.json`, replace `SOURCE_ROOT`, then run:

```bash
python preprocess_echo_gen.py --config echo_gen_config.json
python train_echo_gen.py --config echo_gen_config.json
```

The training command uses already prepared data; it does not launch preprocessing. The current direct training loop requires **batch size 1**. Full-volume 3D workloads may require substantial GPU memory; CPU mode is intended for small checks.

## Documentation

| Guide | Contents |
|---|---|
| [Getting started](GETTING_STARTED.md) | Environment, input data, dashboard, CLI and remote access |
| [Pipeline and architecture](PIPELINE.md) | Preprocessing, normalization, generator, critic, losses and validation |
| [Configuration reference](CONFIGURATION.md) | Supported settings, defaults, overrides, checkpoint identity and resume |
| [Troubleshooting](TROUBLESHOOTING.md) | Dataset/cache errors, GPU setup, run monitoring and locks |
| [Development and validation](DEVELOPMENT.md) | Module responsibilities, tests, provenance and verification limits |

## Project layout

No existing data directories are relocated. Documentation files live in the repository root.

| Location | Purpose |
|---|---|
| `preprocess/` | Prepared volumes, masks, metadata, split, statistics and caches |
| `models/SWI-CEASET-ABRIM-V8/<modality>/` | Generator/critic weights, best model, configuration and training state |
| `training/SWI-CEASET-ABRIM-V8/` | Dashboard launch configurations/logs and per-modality epoch histories |
| `architecture/`, `evaluation/` | Existing locations retained for architecture/evaluation work |
| `synthetic/`, `inference/` | Existing locations retained for downstream generation/reconstruction work |

The new entry points do not automatically run the source notebook's optional inference or qualitative-evaluation cells, nor a complete SWI/QSM reconstruction pipeline. See [Pipeline and architecture](PIPELINE.md) for scope and normalization details.

## Validation status

Configuration, synthetic-data, and preprocessing/training-validation regression tests are available:

```bash
python -m unittest -v test_echo_gen.py test_echo_gen_data.py test_echo_gen_preview.py test_echo_gen_partition.py test_echo_gen_events.py
```

Full ABRIM preprocessing and GPU training have not been verified in the delivery environment. Browser rendering also remains unverified. See [Development and validation](DEVELOPMENT.md) for the checks performed and their limits.

The **Event feed** summarizes recent process output. The original **Process log** is hidden by default; enable **Show original process log** when you need raw details. Full on-disk logs continue to be recorded.
