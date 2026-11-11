"""CLI runner shared by independent preprocessing and training entry points."""
import argparse
import json
import os
from pathlib import Path
import sys
import signal
from echo_gen_control import request_stop, reset_stop, check_training_stop, TrainingStopped
import uuid
from datetime import datetime, timezone


def main(action):
    parser = argparse.ArgumentParser(description=f'Echo Gen {action}')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--config', type=Path, help='JSON settings (see echo_gen_defaults.json)')
    parser.add_argument('--dry-run', action='store_true', help='Resolve/validate settings without loading MRI or TensorFlow')
    args = parser.parse_args()
    os.environ['ECHO_GEN_PROJECT_ROOT'] = str(args.project_root.expanduser().resolve())
    os.environ['ECHO_GEN_ACTION'] = action
    if args.config:
        os.environ['ECHO_GEN_CONFIG'] = str(args.config.expanduser().resolve())
    import echo_gen_config as cfg
    if args.dry_run:
        print(json.dumps(cfg.RESOLVED_CONFIGURATION, indent=2, default=str))
        return
    # One job per project: preprocessing must not rewrite data while training reads it.
    lock = cfg.PROJECT_ROOT / '.echo_gen_job.lock'
    token = uuid.uuid4().hex
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError(f'Another job owns {lock}. If a process crashed, verify it has stopped before removing this lock.')
    with os.fdopen(fd, 'w') as f:
        json.dump({'pid':os.getpid(),'token':token,'action':action}, f)
    old_handler = None
    if action == 'train':
        reset_stop()
        old_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        if action == 'prepare':
            import echo_gen_preprocessing as p
            if not cfg.SOURCE_ROOT.is_dir():
                raise FileNotFoundError(f'Unprocessed data directory does not exist: {cfg.SOURCE_ROOT}')
            p.prepare_training_data(p.CFG)
            p.validate_training_artifacts(p.CFG)
        else:
            import echo_gen_training as t
            check_training_stop()
            t.assert_training_hardware()
            t.validate_training_artifacts(
                t.CFG, sample_builder=t.make_ceaset_sample_from_configuration
            )
            cfg.assert_direct_loop_runtime_config()
            for modality in cfg.TRAINING_MODALITIES_TO_RUN:
                check_training_stop()
                root = cfg.TRAINING_MODEL_ROOT / modality
                root.mkdir(parents=True, exist_ok=True)
                metadata = root / f'configuration_{cfg.RUN_NAME}.json'
                resolved = dict(cfg.RESOLVED_CONFIGURATION)
                resolved['started_at_utc'] = datetime.now(timezone.utc).isoformat()
                resolved['repository'] = 'https://github.com/AugFabCode/WGAN_SMEG-Synthetic-MRI-Echo-Generator-Lab'
                metadata.write_text(json.dumps(resolved, indent=2, default=str)+'\n')
                t.train_ceaset_modality(modality)
    except TrainingStopped as exc:
        print(f'[STOPPED] {exc}', flush=True)
        raise SystemExit(130)
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
        if lock.exists() and json.loads(lock.read_text()).get('token') == token:
            lock.unlink()

if __name__ == '__main__':
    raise SystemExit('Use preprocess_echo_gen.py or train_echo_gen.py')
