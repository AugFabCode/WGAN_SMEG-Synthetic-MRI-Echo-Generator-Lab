# Development and validation

[README](README.md) · [Pipeline](PIPELINE.md) · [Configuration](CONFIGURATION.md)

## Module responsibilities

| File | Responsibility |
|---|---|
| `preprocess_echo_gen.py` | Preprocessing CLI entry point |
| `train_echo_gen.py` | Training CLI entry point |
| `echo_gen_runner.py` | CLI parsing, project lock, stage dispatch and model configuration records |
| `echo_gen_config.py` | Override loading, path resolution, validation and run identity |
| `echo_gen_defaults.json` | Accepted override keys/defaults |
| `echo_gen_preprocessing.py` | ABRIM preparation, artifact validation, cache readers and NumPy physical baselines |
| `echo_gen_training.py` | TensorFlow models, sampling, losses, evaluation and epoch/checkpoint loop |
| `echo_gen_dashboard.py` | Local HTTP server, subprocess launch and monitoring |
| `echo_gen_dashboard.html` | Configuration form, architecture schematic, curves, MRI preview panel and logs |
| `echo_gen_preview.py` | Bounded central-slice extraction, background CPU rendering, 3× bicubic upscaling and in-memory delivery |

The preprocessing module must remain importable without TensorFlow. Its artifact validator accepts an optional `sample_builder` callback. Preprocessing validates prepared artifacts without that callback; the training runner supplies `make_ceaset_sample_from_configuration` so actual model-input checks still run before training. This avoids relying on another module's globals after the notebook split.

The dashboard intentionally does not load MRI arrays or models itself. Jobs execute in separate Python processes, using the dashboard's interpreter. A project lock prevents concurrent jobs. The server binds to localhost and validates its host/request token; it is not a public authentication or multi-user scheduling system.

## Tests

From the repository root with preprocessing dependencies installed:

```bash
python -m unittest -v test_echo_gen.py test_echo_gen_data.py test_echo_gen_preview.py
```

`test_echo_gen.py` checks configuration identity/resume, fresh instances, overwrite rejection, both cache layouts, invalid settings, dry-run side effects, data-manifest identity and separation from training imports.

`test_echo_gen_data.py` uses small synthetic arrays and artifact fixtures to check chunk coverage/padding, the exponential magnitude baseline, NumPy round-trip, TensorFlow-free imports and validation with/without a training sampler. It also checks rejection of non-finite sampled data.

These tests require no ABRIM dataset, FSL execution or TensorFlow model training. The artifact-validation regression fixtures mock discovery/signature results; they are not evidence of a successful end-to-end NIfTI preparation run.

For syntax validation:

```bash
python -m compileall -q echo_gen_config.py echo_gen_preprocessing.py echo_gen_training.py echo_gen_runner.py echo_gen_dashboard.py echo_gen_preview.py preprocess_echo_gen.py train_echo_gen.py
```

## Checks performed and remaining verification

The implementation has passed configuration/synthetic-array tests and targeted HTTP smoke checks for defaults/status, invalid launches, worker-error logs and lock cleanup. Python compilation and dashboard JavaScript syntax were checked. Documentation links and the example JSON are checked when preparing the documentation change.

Full ABRIM preprocessing, GPU training, live loss-curve behavior during a real training run and browser rendering remain unverified in the delivery environment. A successful dry-run is configuration validation only. There is no claim that scientific performance, GPU stability or all advanced notebook-compatibility switches have been validated.

Before relying on results, run the stages on a representative local dataset, inspect physical-value preprocessing outputs, confirm subject separation, complete a short GPU training/validation session, test checkpoint resume, and evaluate held-out reconstructions with the intended downstream SWI/QSM pipeline.

## Provenance and contributions

[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records the supplied notebook filename, SHA-256 and extracted cell groups. The source notebook itself is not required at runtime and is not included in this repository change. The split preserves its core model/loss/sampling implementation; integration changes include configuration overrides, path resolution, entry points, naming, final checkpoint saving and the dashboard.

When changing the pipeline:

- Preserve the existing project folders and scientific data; avoid committing datasets, caches, logs, credentials or trained weights.
- Keep preprocessing independent of TensorFlow and test both CLI stages.
- Update defaults/reference/example together when introducing supported settings.
- Check whether a model/loss change requires an integration-revision change to avoid unintended checkpoint reuse. Run identity does not automatically hash source code.
- Record the Git commit, effective configuration, data provenance and evaluation protocol for experiments.
- Distinguish synthetic/unit checks from real data and GPU verification in pull requests.

No dataset license or project software license is inferred here. Check the repository's license files and the applicable dataset/tool terms before redistribution; this documentation does not grant additional rights.

Preview tests cover source-axis mapping, central slice extraction and transfer bounds, PNG decoding, throttling, run isolation and critic-loop metric/call-count preservation. These use synthetic arrays and an instrumented TensorFlow adapter; GPU execution and compiled-step tracing remain unverified.

The preview renderer uses Pillow for independent 3× bicubic panel upscaling on the background CPU thread. Tests check exact bicubic pixels, no interpolation across panel boundaries, unchanged source arrays, and maximum-size in-memory JSON delivery. Publisher, authentication, latest-only replacement and disabled-without-receiver tests verify the transient preview path. GPU capture remains bounded to 256 pixels per side; encoded image payloads and browser image memory increase with display resolution.
