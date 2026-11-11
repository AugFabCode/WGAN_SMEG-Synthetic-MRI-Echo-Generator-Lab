"""ABRIM preprocessing and training-data readers extracted from the supplied notebook."""
from echo_gen_config import *
import csv
import json
import math
import gc
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence, Mapping, Any

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.orientations import (
    aff2axcodes,
    apply_orientation,
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

# Keep BLAS/OpenMP thread use conservative during volume preprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.seterr(invalid="warn", divide="warn", over="warn", under="ignore")

print(f"Python executable: {sys.executable}")
print(f"Python version:    {sys.version.split()[0]}")
print(f"Notebook cwd:      {Path.cwd()}")


from itertools import combinations

from IPython.display import display


@dataclass
class PipelineConfig:
    source_nifti_root: Path = SOURCE_ROOT
    preprocessing_data_root: Path = REFERENCE_ROOT
    project_root: Path = PROJECT_ROOT
    profile_name: str = PIPELINE_PROFILE
    echo_labels: tuple = ECHO_LABELS
    default_echo_times_ms: tuple = DEFAULT_ECHO_TIMES_MS
    bet_reference_echo: str = BET_REFERENCE_ECHO
    bet_fraction: float = BET_FRACTION
    erosion_radius_mm: float = BET_EROSION_RADIUS_MM
    affine_tolerance: float = 1e-4
    chunk_grid: tuple = CHUNK_GRID
    chunk_overlap_voxels: int = CHUNK_OVERLAP_VOXELS
    write_chunk_nifti: bool = WRITE_CHUNK_NIFTI
    write_chunk_hdf5: bool = WRITE_CHUNK_HDF5
    clearswi_path_prefix: Path = CLEARSWI_PATH_PREFIX
    bash_aliases: Path = BASH_ALIASES
    fsl_setup_candidates: tuple = FSL_SETUP_CANDIDATES

    @property
    def raw_root(self):
        return self.preprocessing_data_root / 'metadata'

    @property
    def swi_temp_root(self):
        return self.preprocessing_data_root / 'work'

    @property
    def filtered_phase_root(self):
        return self.preprocessing_data_root / 'filtered_phase'

    @property
    def magnitude_raw_root(self):
        return self.preprocessing_data_root / 'mag_raw'

    @property
    def magnitude_bet_root(self):
        return self.preprocessing_data_root / 'mag_bet'

    @property
    def phase_bet_root(self):
        return self.preprocessing_data_root / 'phase_bet'

    @property
    def betmask_root(self):
        return self.preprocessing_data_root / 'mask' / self.bet_reference_echo


CFG = PipelineConfig()


def _json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + '\n')
    tmp.replace(path)


def _csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def _nii_candidates(stem):
    stem = str(stem)
    return [Path(stem + '.nii.gz'), Path(stem + '.nii')]


def _one_existing(paths, *, required=False, description='NIfTI'):
    found = list(dict.fromkeys(Path(p) for p in paths if Path(p).is_file()))
    if len(found) > 1:
        raise ValueError(f'Ambiguous {description}: {found}. Keep one or set an explicit override.')
    if not found:
        if required:
            raise FileNotFoundError(f'Missing {description}; checked: {list(map(str, paths))}')
        return None
    return found[0]


def _file_signature(path):
    path = Path(path)
    st = path.stat()
    return {'path': str(path.resolve()), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _sidecar(path):
    text = str(path)
    return Path(text[:-7] + '.json' if text.endswith('.nii.gz') else text[:-4] + '.json')


def echo_time_sidecar_path(case_id, cfg=CFG):
    return cfg.raw_root / f'{case_id}_EchoTimes.json'


def echo_times_for_case(case_id, cfg=CFG):
    path = echo_time_sidecar_path(case_id, cfg)
    if not path.is_file():
        raise FileNotFoundError(f'Missing verified TE metadata: {path}; run preparation first.')
    values = tuple(json.loads(path.read_text())['echo_times_ms'])
    if (len(values) != len(cfg.echo_labels) or not np.all(np.isfinite(values))
            or min(values) <= 0 or np.any(np.diff(values) <= 0)):
        raise ValueError(f'Invalid ordered echo times in {path}: {values}')
    return values


def echo_time_source_for_case(case_id, cfg=CFG):
    return json.loads(echo_time_sidecar_path(case_id, cfg).read_text())['source']


def _read_echo_times(mag_paths, phase_paths, cfg=CFG):
    values, metadata = [], []
    for idx, mag_path in enumerate(mag_paths):
        path = _sidecar(mag_path)
        payload = json.loads(path.read_text()) if path.is_file() else {}
        if path.is_file():
            metadata.append(path)
        # BIDS EchoTime is in seconds. Never guess units from the number.
        value = payload.get('EchoTime')
        if value is not None:
            if isinstance(value, (list, dict, bool)):
                raise ValueError(f'{path}: expected scalar EchoTime in seconds')
            value = float(value) * 1000.0
        values.append(value)
        if phase_paths:
            pp = _sidecar(phase_paths[idx])
            if pp.is_file():
                metadata.append(pp)
                pv = json.loads(pp.read_text()).get('EchoTime')
                if pv is not None and value is not None and not np.isclose(float(pv)*1000, value, atol=0.01, rtol=0):
                    raise ValueError(f'Magnitude/phase TE mismatch: {mag_path} / {pp}')
    source = 'bids_json_seconds'
    if any(v is None for v in values):
        if not ALLOW_PROTOCOL_TE_FALLBACK:
            raise ValueError('Missing EchoTime JSON metadata. Confirm the ABRIM protocol and enable ALLOW_PROTOCOL_TE_FALLBACK.')
        for i, value in enumerate(values):
            if value is not None and not np.isclose(value, cfg.default_echo_times_ms[i], atol=0.01, rtol=0):
                raise ValueError('Partial TE metadata disagrees with the fallback protocol; refusing to mix protocols.')
        all_missing = all(v is None for v in values)
        values = [cfg.default_echo_times_ms[i] if v is None else v for i, v in enumerate(values)]
        source = 'configured_abrim_fallback' if all_missing else 'bids_json_with_confirmed_fallback'
    # Also validate phase metadata when magnitude EchoTime required the fallback.
    for i, phase_path in enumerate(phase_paths):
        pp = _sidecar(phase_path)
        if pp.is_file():
            pv = json.loads(pp.read_text()).get('EchoTime')
            if pv is not None and not np.isclose(float(pv)*1000, values[i], atol=0.01, rtol=0):
                raise ValueError(f'Magnitude/phase TE mismatch: {mag_paths[i]} / {pp}')
    if (len(values) != len(cfg.echo_labels) or not np.all(np.isfinite(values))
            or min(values) <= 0 or np.any(np.diff(values) <= 0)):
        raise ValueError(f'Invalid echo times: {values}')
    return tuple(map(float, values)), source, metadata


def discover_abrim_cases(cfg=CFG):
    """Discover separate BIDS magnitude echoes; never treat an SWI image as raw phase."""
    root = Path(cfg.source_nifti_root)
    if not root.is_dir():
        raise FileNotFoundError(f'ABRIM input directory not found: {root}')
    groups = {}
    pattern = re.compile(r'^(?P<prefix>sub-[A-Za-z0-9]+(?:_ses-[A-Za-z0-9]+)?_run-[A-Za-z0-9]+)_echo-(?P<echo>\d+)_part-mag_MEGRE\.nii(?:\.gz)?$')
    anat_dirs = sorted(set(root.glob('sub-*/anat')) | set(root.glob('sub-*/ses-*/anat')))
    for anat in anat_dirs:
        for path in sorted(anat.glob('*_part-mag_MEGRE.nii*')):
            match = pattern.fullmatch(path.name)
            if not match:
                continue
            prefix, index = match['prefix'], int(match['echo'])
            key = (anat, prefix)
            group = groups.setdefault(key, {})
            if index in group:
                raise ValueError(f'Duplicate echo {index}: {group[index]} and {path}')
            group[index] = path
    if not groups:
        raise RuntimeError(f'No sub-*/anat/sub-*_run-*_echo-*_part-mag_MEGRE.nii[.gz] files found under {root}')
    cases, seen = [], set()
    for (anat, prefix), by_echo in sorted(groups.items(), key=lambda item: str(item[0])):
        expected = set(range(1, len(cfg.echo_labels) + 1))
        if set(by_echo) != expected:
            raise ValueError(f'{prefix}: expected echoes {sorted(expected)}, found {sorted(by_echo)}')
        case_id = prefix.removeprefix('sub-')
        if case_id in seen:
            raise ValueError(f'Duplicate case ID in different directories: {case_id}')
        seen.add(case_id)
        subject = prefix.split('_')[0]
        subject_dir = anat.parent.parent if anat.parent.name.startswith('ses-') else anat.parent
        if subject_dir.name != subject:
            raise ValueError(f'Subject entity disagrees with directory: {anat / prefix}')
        run = next(s for s in prefix.split('_') if s.startswith('run-'))
        mags = [by_echo[i] for i in sorted(expected)]
        phases = [_one_existing(_nii_candidates(anat / f'{prefix}_echo-{i}_part-phase_MEGRE')) for i in sorted(expected)]
        if any(p is not None for p in phases) and not all(p is not None for p in phases):
            raise ValueError(f'{prefix}: incomplete raw phase echo set')
        phases = phases if all(p is not None for p in phases) else []
        filtered = None
        if case_id in FILTERED_PHASE_OVERRIDES:
            filtered = _one_existing([Path(FILTERED_PHASE_OVERRIDES[case_id])], required=True, description=f'{case_id} filtered-phase override')
        else:
            candidates = []
            for steps in [anat / f'swi_steps_{run}', anat / 'swi_steps' / run,
                          anat / 'swi_steps' / prefix, anat / f'{prefix}_swi_steps']:
                candidates += _nii_candidates(steps / 'filteredphase_orig')
            filtered = _one_existing(candidates, description=f'{case_id} run-specific filtered phase')
            legacy = _one_existing(_nii_candidates(anat / 'swi_steps' / 'filteredphase_orig'))
            if filtered is None and legacy is not None:
                prefixes = {p for (a, p) in groups if a == anat}
                swi_runs = {re.search(r'(run-[A-Za-z0-9]+)_swi', p.name)[1]
                            for p in anat.glob('*_run-*_swi.nii*') if re.search(r'(run-[A-Za-z0-9]+)_swi', p.name)}
                if len(prefixes) == 1 and len(swi_runs) <= 1:
                    filtered = legacy
                elif not phases and case_id not in PHASE_4D_OVERRIDES:
                    raise ValueError(f'{case_id}: shared {legacy} is ambiguous across runs. Set FILTERED_PHASE_OVERRIDES per case or supply genuine raw phase.')
        phase4d = _one_existing([Path(PHASE_4D_OVERRIDES[case_id])], required=True) if case_id in PHASE_4D_OVERRIDES else None
        if PHASE_MODE == 'compute':
            filtered = None
        if PHASE_MODE == 'existing' and filtered is None:
            raise FileNotFoundError(f'{case_id}: PHASE_MODE=existing requires filteredphase_orig')
        if filtered is None and not phases and phase4d is None:
            raise FileNotFoundError(f'{case_id}: no filtered phase or genuine phase source. Provide part-phase echoes or a PHASE_4D_OVERRIDES entry. The *_swi image is not a phase substitute.')
        tes, te_source, metadata = _read_echo_times(mags, phases, cfg)
        cases.append({'case_id': case_id, 'patient_key': subject, 'run': run,
                      'magnitude': mags, 'phase': phases, 'phase4d': phase4d,
                      'filtered_phase': filtered, 'echo_times_ms': tes,
                      'echo_time_source': te_source, 'metadata_files': metadata})
    return cases


def _load_checked(path, reference=None, ndim=3, cfg=CFG, *, require_finite=True):
    image = nib.load(str(path))
    shape = image.shape
    if ndim == 3 and len(shape) == 4 and shape[-1] == 1:
        image = nib.squeeze_image(image)
        shape = image.shape
    if len(shape) != ndim:
        raise ValueError(f'{path}: expected {ndim} dimensions, found {shape}')
    if ndim == 4 and shape[-1] != len(cfg.echo_labels):
        raise ValueError(f'{path}: expected {len(cfg.echo_labels)} echoes on last axis, got {shape}')
    if reference is not None:
        if shape[:3] != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=cfg.affine_tolerance, rtol=0):
            raise ValueError(f'Spatial grid/affine mismatch: {path}')
    if not np.all(np.isfinite(image.affine)) or abs(np.linalg.det(image.affine[:3, :3])) < 1e-10:
        raise ValueError(f'Invalid NIfTI affine: {path}')
    data = image.get_fdata(dtype=np.float32)
    if require_finite and not np.all(np.isfinite(data)):
        raise ValueError(f'Non-finite image data: {path}')
    return image, data


def _save_volume(path, data, reference):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(np.asarray(data, np.float32), reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if qform is not None:
        image.set_qform(qform, int(qcode))
    if sform is not None:
        image.set_sform(sform, int(scode))
    tmp = path.with_name('.tmp_' + path.name)
    nib.save(image, str(tmp))
    tmp.replace(path)


def shell_quote(value):
    return shlex.quote(str(value))


def _shell_environment_lines(cfg=CFG):
    lines = ['set -eo pipefail', 'shopt -s expand_aliases']
    lines += [f'if [[ -f {shell_quote(cfg.bash_aliases)} ]]; then source {shell_quote(cfg.bash_aliases)}; fi']
    candidates = ' '.join(shell_quote(p) for p in cfg.fsl_setup_candidates)
    lines += [f'for setup_file in {candidates}; do',
              '  if [[ -f "$setup_file" ]]; then source "$setup_file"; break; fi', 'done',
              f'export PATH={shell_quote(cfg.clearswi_path_prefix)}:"$PATH"',
              'export FSLOUTPUTTYPE=NIFTI_GZ', 'export LC_ALL=C', 'set -u']
    return lines


def run_embedded_bash(bash_program, *, cwd, environment=None, script_name='run_external.sh'):
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    script = cwd / script_name
    script.write_text(bash_program.rstrip() + '\n')
    env = os.environ.copy()
    env.update(environment or {})
    with (cwd / (script_name + '.log')).open('w') as log:
        process = subprocess.Popen(['bash', str(script.resolve())], cwd=str(cwd), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, ['bash', str(script)])
    return code


def _run_external(args, case_dir, name, cfg=CFG):
    command = shlex.join([str(arg) for arg in args])
    print('$', command, flush=True)
    return run_embedded_bash('\n'.join(['#!/usr/bin/env bash', *_shell_environment_lines(cfg), command]),
                             cwd=case_dir, script_name=name + '.sh')


def external_tool_preflight(cfg=CFG, *, need_clearswi=False):
    commands = ['bet', 'fslmaths'] + (['clearswi'] if need_clearswi else [])
    script = '\n'.join([*_shell_environment_lines(cfg),
                         *[f'command -v {shell_quote(c)} >/dev/null || {{ echo "Missing {c}" >&2; exit 127; }}' for c in commands]])
    subprocess.run(['bash', '-c', script], check=True, text=True)
    print('External commands available:', ', '.join(commands))


class CaseQCError(ValueError):
    """Expected exclusion, recorded separately from execution failures."""


def shell_magnitude_parameters(volumes, mask):
    diagnostics = []
    threshold = MAG_SATURATION_VALUE - MAG_SATURATION_TOLERANCE
    selected = None
    for idx in MAG_REFERENCE_INDICES:
        values = volumes[idx][mask]
        if not values.size or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise CaseQCError('Magnitude requires finite nonnegative in-mask values.')
        lo, hi = np.percentile(values, [MAG_P_LOW, MAG_P_HIGH])
        fraction = float(np.mean(values >= threshold))
        item = {'echo': f'TE{idx+1}', 'p_low': float(lo), 'p_high': float(hi),
                'raw_max': float(values.max()), 'near_saturation_fraction': fraction}
        diagnostics.append(item)
        if hi < threshold and fraction < MAG_MAX_NEAR_SATURATION_FRACTION:
            selected = item
            break
    if selected is None:
        raise CaseQCError('All TE1-TE3 magnitude references failed saturation QC.')
    span = selected['p_high'] - selected['p_low']
    if span <= SHELL_NORMALIZATION_EPSILON:
        raise CaseQCError('Magnitude reference has a degenerate robust range.')
    upper = min(selected['raw_max'], selected['p_high'] + MAG_OUTLIER_MARGIN*span, MAG_SATURATION_VALUE)
    if upper <= SHELL_NORMALIZATION_EPSILON:
        raise CaseQCError('Invalid magnitude upper anchor.')
    return {'reference_echo': selected['echo'], 'lower_anchor': 0.0, 'upper_anchor': upper,
            'center': upper/2, 'half_range': upper/2, 'reference_qc': diagnostics}


def shell_normalize_magnitude(volume, mask, params):
    # Actual clipping: FSL -thr/-uthr zero outliers and are NOT np.clip.
    result = np.clip((np.asarray(volume, np.float64) - params['center']) / params['half_range'], -1, 1)
    result[(volume == 0) | ~mask] = -1.0
    return result.astype(np.float32)


def phase_exponential_parameters(volumes):
    amplitude = float(max(np.max(np.abs(v)) for v in volumes))
    x0, eps = PHASE_EXP_X0, SHELL_NORMALIZATION_EPSILON
    if amplitude <= eps or x0 <= 0:
        k = 1.0
    else:
        ratio = max(amplitude/x0, 1 + 1e-6)
        u = max(1 - 0.5*x0, 1e-6)
        for _ in range(200):
            u = float(np.clip(0.5*u + 0.5*(1-x0+x0*u**ratio), 1e-12, 1-1e-12))
        k = max(float(-x0/np.log(u)), eps)
    return {'A_file': amplitude, 'k_file': k, 'x0': x0,
            'denominator': max(float(-np.expm1(-amplitude/k)), eps), 'method': 'signed_exponential'}


def shell_normalize_phase(volume, params):
    mapped = np.sign(volume) * (-np.expm1(-np.abs(volume.astype(np.float64))/params['k_file'])) / params['denominator']
    return np.clip(mapped, -1, 1).astype(np.float32)


def _eroded_mask_path(case_id, cfg=CFG):
    return cfg.preprocessing_data_root / 'erodemask' / cfg.bet_reference_echo / f'erodemask_{case_id}.nii.gz'


def _prepared_case_path(case_id, modality, echo_label, cfg=CFG):
    if modality == 'magnitude':
        return cfg.magnitude_bet_root / echo_label / f'mag_bet_{case_id}.nii.gz'
    if modality == 'phase':
        return cfg.phase_bet_root / echo_label / f'umf_phase_bet_{case_id}.nii.gz'
    raise ValueError(modality)


def _case_metadata_path(case_id, cfg=CFG):
    return cfg.raw_root / f'{case_id}_preprocessing.json'


def _preparation_signature(case, cfg=CFG):
    paths = case['magnitude'] + case['metadata_files']
    if case['filtered_phase'] is not None:
        paths += [case['filtered_phase']]
    else:
        paths += case['phase'] + ([case['phase4d']] if case['phase4d'] else [])
    return _digest({'revision': CEASET_PIPELINE_REVISION,
        'inputs': [_file_signature(p) for p in paths], 'tes': case['echo_times_ms'],
        'te_source': case['echo_time_source'], 'bet': [cfg.bet_reference_echo, cfg.bet_fraction, cfg.erosion_radius_mm],
        'clearswi': [CLEAR_SWI_EXTRA_ARGS, CLEAR_SWI_REQUIRES_X, PHASE_MODE],
        'normalization': [MAG_P_LOW, MAG_P_HIGH, MAG_REFERENCE_INDICES, MAG_SATURATION_VALUE,
            MAG_SATURATION_TOLERANCE, MAG_MAX_NEAR_SATURATION_FRACTION, MAG_OUTLIER_MARGIN,
            PHASE_EXP_X0, SHELL_NORMALIZATION_EPSILON, EXPORT_SHELL_NORMALIZED]})


def validate_filtered_phase_in_mask(phase4d, mask, source, echo_labels):
    """Keep finite phase samples; zero only invalid background; reject invalid brain data.

    A whole-volume finite check is deferred only for filtered phase, until its
    spatially aligned eroded training mask exists. Never overwrite the source.
    """
    phase4d = np.asarray(phase4d)
    mask_values = np.asarray(mask)
    if mask_values.ndim != 3 or not np.all(np.isfinite(mask_values)):
        raise ValueError('Filtered-phase validation requires a finite 3-D brain mask.')
    mask = mask_values > 0.5
    if (phase4d.ndim != 4 or phase4d.shape[:3] != mask.shape
            or phase4d.shape[-1] != len(echo_labels)):
        raise ValueError(f'Filtered phase/mask shape or echo-count mismatch: {source}')
    if not mask.any():
        raise ValueError('Filtered-phase validation requires a nonempty brain mask.')

    counts = []
    for i, echo in enumerate(echo_labels):
        invalid = ~np.isfinite(phase4d[..., i])
        inside = int(np.count_nonzero(invalid & mask))
        outside = int(np.count_nonzero(invalid & ~mask))
        counts.append({'echo': echo, 'nonfinite_inside_mask': inside,
                       'nonfinite_outside_mask': outside})
    affected = [row for row in counts if row['nonfinite_inside_mask']]
    if affected:
        detail = ', '.join(f"{row['echo']}={row['nonfinite_inside_mask']}"
                           for row in affected)
        raise ValueError(
            f'Filtered phase has non-finite samples INSIDE the eroded brain mask: '
            f'{source}. Invalid voxel counts by echo: {detail}. '
            'Inspect the phase source and its processing before training; '
            'no in-mask values were replaced.'
        )

    outside_total = sum(row['nonfinite_outside_mask'] for row in counts)
    cleaned = phase4d
    if outside_total:
        cleaned = phase4d.copy()
        for i in range(len(echo_labels)):
            volume = cleaned[..., i]
            volume[~np.isfinite(volume)] = 0.0
        print(f'[PHASE QC] {source}: replaced {outside_total} non-finite '
              'background samples with zero in derived outputs; '
              'all in-mask phase samples are finite.', flush=True)
    return cleaned, counts


def preprocess_abrim_case(case, cfg=CFG, *, overwrite=False):
    cid = case['case_id']
    signature = _preparation_signature(case, cfg)
    meta_path = _case_metadata_path(cid, cfg)
    if meta_path.is_file() and not overwrite:
        old = json.loads(meta_path.read_text())
        complete = all(Path(x['path']).is_file() and _file_signature(x['path']) == x for x in old.get('outputs', []))
        if old.get('signature') == signature and old.get('status') == 'accepted' and old.get('outputs') and complete:
            print('[PREP SKIP]', cid)
            return old
        raise RuntimeError(f'{cid}: inputs/settings/products changed or output is incomplete. Set OVERWRITE_PREPROCESSING=True and rebuild derived data.')
    ref, _ = _load_checked(case['magnitude'][0], cfg=cfg)
    magnitudes = [_load_checked(p, ref, cfg=cfg)[1] for p in case['magnitude']]
    case_dir = cfg.swi_temp_root / cid
    case_dir.mkdir(parents=True, exist_ok=True)
    filtered = case['filtered_phase']
    if filtered is None:
        mag4d = case_dir / 'magnitude_4d.nii.gz'
        pha4d = case_dir / 'phase_4d.nii.gz'
        _save_volume(mag4d, np.stack(magnitudes, axis=-1), ref)
        if case['phase4d']:
            _, data = _load_checked(case['phase4d'], ref, ndim=4, cfg=cfg)
        else:
            data = np.stack([_load_checked(p, ref, cfg=cfg)[1] for p in case['phase']], axis=-1)
        _save_volume(pha4d, data, ref)
        del data
        # A new steps folder prevents stale CLEAR-SWI output being mistaken for success.
        steps = case_dir / ('clearswi_' + datetime.now().strftime('%Y%m%dT%H%M%S%f'))
        steps.mkdir()
        args = ['clearswi', '-m', mag4d, '-p', pha4d, '-o', steps/'swi.nii.gz',
                '-t', '['+','.join(map(str, case['echo_times_ms']))+']',
                '--writesteps='+str(steps), '--unwrapping-algorithm', 'laplacian', *CLEAR_SWI_EXTRA_ARGS]
        if CLEAR_SWI_REQUIRES_X:
            args += ['-x', pha4d]
        _run_external(args, case_dir, 'clearswi', cfg)
        filtered = _one_existing(_nii_candidates(steps/'filteredphase_orig'), required=True)
    # Check geometry now; filtered-phase finiteness is checked against the
    # eroded mask below, before any phase output/statistics/cache is written.
    _, phase4d = _load_checked(filtered, ref, ndim=4, cfg=cfg, require_finite=False)
    for i, echo in enumerate(cfg.echo_labels):
        _save_volume(cfg.magnitude_raw_root/echo/f'mag_raw_{cid}.nii.gz', magnitudes[i], ref)
    # Unique mask workspace avoids accepting an old mask after an external-tool failure.
    mask_work = case_dir / ('mask_' + datetime.now().strftime('%Y%m%dT%H%M%S%f'))
    mask_work.mkdir()
    bet_base = mask_work / 'brain'
    ref_path = cfg.magnitude_raw_root/cfg.bet_reference_echo/f'mag_raw_{cid}.nii.gz'
    _run_external(['bet', ref_path, bet_base, '-f', str(cfg.bet_fraction), '-R', '-m'], mask_work, 'bet', cfg)
    bet_path = _one_existing(_nii_candidates(str(bet_base)+'_mask'), required=True)
    _, raw_mask = _load_checked(bet_path, ref, cfg=cfg)
    mask_out = cfg.betmask_root / f'mask_{cid}_mask.nii.gz'
    _save_volume(mask_out, (raw_mask > .5).astype(np.float32), ref)
    eroded_work = mask_work/'eroded.nii.gz'
    _run_external(['fslmaths', mask_out, '-kernel', 'sphere', str(cfg.erosion_radius_mm), '-ero', eroded_work], mask_work, 'erode', cfg)
    _, mask_data = _load_checked(eroded_work, ref, cfg=cfg)
    mask = mask_data > .5
    if not mask.any():
        raise CaseQCError('BET/erosion produced an empty brain mask.')
    eroded = _eroded_mask_path(cid, cfg)
    _save_volume(eroded, mask.astype(np.float32), ref)
    phase4d, phase_nonfinite_qc = validate_filtered_phase_in_mask(
        phase4d, mask, filtered, cfg.echo_labels)
    for i, echo in enumerate(cfg.echo_labels):
        _save_volume(cfg.filtered_phase_root/echo/f'umf_phase_{cid}.nii.gz', phase4d[..., i], ref)
    mags = [np.where(mask, a, 0).astype(np.float32) for a in magnitudes]
    phases = [np.where(mask, phase4d[..., i], 0).astype(np.float32) for i in range(len(cfg.echo_labels))]
    if any(np.any(a[mask] < 0) for a in mags):
        raise CaseQCError('Negative magnitude voxels in brain.')
    mag_params = shell_magnitude_parameters(mags, mask)
    phase_params = phase_exponential_parameters(phases)
    if any(np.percentile(np.abs(a[mask]), PHASE_RATE_PERCENTILE) <= NORMALIZATION_EPSILON for a in phases):
        raise CaseQCError('An echo has no usable phase normalization scale.')
    outputs, details = [mask_out, eroded], []
    for i, echo in enumerate(cfg.echo_labels):
        mpath, ppath = [_prepared_case_path(cid, mod, echo, cfg) for mod in ('magnitude','phase')]
        _save_volume(mpath, mags[i], ref)
        _save_volume(ppath, phases[i], ref)
        outputs += [mpath, ppath, cfg.magnitude_raw_root/echo/f'mag_raw_{cid}.nii.gz',
                    cfg.filtered_phase_root/echo/f'umf_phase_{cid}.nii.gz']
        capped = float(np.mean(mags[i][mask] > mag_params['upper_anchor']))
        row = {'case_id': cid, 'echo': echo, 'magnitude_capped_fraction': capped,
               'phase_min_before': float(phases[i].min()), 'phase_max_before': float(phases[i].max()),
               'phase_nonfinite_inside_mask': phase_nonfinite_qc[i]['nonfinite_inside_mask'],
               'phase_nonfinite_outside_mask': phase_nonfinite_qc[i]['nonfinite_outside_mask']}
        if capped > MAG_HIGH_CAP_WARNING_FRACTION:
            print(f'[QC] {cid} {echo}: {capped:.2%} of brain voxels clipped in the optional magnitude export.')
        if EXPORT_SHELL_NORMALIZED:
            mn, pn = shell_normalize_magnitude(mags[i], mask, mag_params), shell_normalize_phase(phases[i], phase_params)
            mp = cfg.preprocessing_data_root/'mag_norm'/echo/f'bet_mag_{cid}.nii.gz'
            pp = cfg.preprocessing_data_root/'phase_norm'/echo/f'bet_umf_phase_{cid}.nii.gz'
            _save_volume(mp, mn, ref)
            _save_volume(pp, pn, ref)
            outputs += [mp, pp]
            row.update(phase_min_after=float(pn.min()), phase_max_after=float(pn.max()))
        details.append(row)
    te_path = echo_time_sidecar_path(cid, cfg)
    _json_write(te_path, {'echo_times_ms': case['echo_times_ms'], 'source': case['echo_time_source'],
                          'patient_key': case['patient_key'], 'filtered_phase_source': str(filtered)})
    outputs.append(te_path)
    result = {'case_id': cid, 'patient_key': case['patient_key'], 'status': 'accepted',
              'signature': signature, 'filtered_phase_source': str(filtered),
              'magnitude_parameters': mag_params, 'phase_parameters': phase_params,
              'filtered_phase_nonfinite_policy': 'zero_background_reject_inmask_v1',
              'filtered_phase_nonfinite_qc': phase_nonfinite_qc,
              'per_echo_qc': details, 'shape': list(ref.shape[:3]),
              'outputs': [_file_signature(p) for p in outputs]}
    _json_write(meta_path, result)
    print('[PREP OK]', cid)
    return result


def run_abrim_preprocessing(cfg=CFG, *, overwrite=False):
    source, output = cfg.source_nifti_root.resolve(), cfg.preprocessing_data_root.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Input and output roots must be separate, non-overlapping directories.')
    cases = discover_abrim_cases(cfg)
    cfg.preprocessing_data_root.mkdir(parents=True, exist_ok=True)
    external_tool_preflight(cfg, need_clearswi=any(c['filtered_phase'] is None for c in cases))
    rows, reports = [], []
    report_path = cfg.preprocessing_data_root/'dataset_manifest.csv'
    for case in cases:
        row = {'case_id': case['case_id'], 'patient_key': case['patient_key'],
               'echo_times_ms': json.dumps(case['echo_times_ms']), 'echo_time_source': case['echo_time_source']}
        try:
            result = preprocess_abrim_case(case, cfg, overwrite=overwrite)
            row.update(status='accepted', reason='', signature=result['signature'])
            reports.append(result)
        except CaseQCError as exc:
            row.update(status='excluded', reason=str(exc), signature='')
            print('[QC EXCLUDED]', case['case_id'], exc)
        except Exception as exc:
            row.update(status='failed', reason=str(exc), signature='')
            rows.append(row)
            _csv_write(report_path, rows)
            raise
        rows.append(row)
        _csv_write(report_path, rows)
    _csv_write(cfg.preprocessing_data_root/'normalization_qc.csv', [r for report in reports for r in report['per_echo_qc']])
    if not reports:
        raise RuntimeError(f'No cases passed preprocessing/QC. See {report_path}')
    return pd.DataFrame(rows)


def _available_complete_case_ids(cfg=CFG):
    path = cfg.preprocessing_data_root/'dataset_manifest.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Run ABRIM preparation first: {path}')
    frame = pd.read_csv(path, dtype={'case_id': str, 'patient_key': str})
    if (frame['status'] == 'failed').any():
        raise RuntimeError('Preprocessing has failed cases. Resolve them and rerun before training.')
    cases = frame.loc[frame.status == 'accepted', 'case_id'].tolist()
    if not cases or len(cases) != len(set(cases)):
        raise RuntimeError('No unique accepted cases in the preprocessing manifest.')
    for cid in cases:
        required = [_eroded_mask_path(cid, cfg), echo_time_sidecar_path(cid, cfg), _case_metadata_path(cid, cfg)]
        required += [_prepared_case_path(cid, mod, echo, cfg) for mod in ('magnitude','phase') for echo in cfg.echo_labels]
        if any(not p.is_file() for p in required):
            raise FileNotFoundError(f'{cid}: accepted case has missing products; rebuild preparation.')
    return sorted(cases)


def load_float_nifti(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D NIfTI at {path}, got {data.shape}")
    data = np.asarray(data, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite prepared volume: {path}")
    return image, data


def orient_for_processing(volume: np.ndarray) -> np.ndarray:
    return np.transpose(volume, (1, 0, 2))


def orient_for_output(volume: np.ndarray) -> np.ndarray:
    return np.transpose(volume, (1, 0, 2))



def _nearest_digital_ceiling(max_value: float) -> float | None:
    if not np.isfinite(max_value) or max_value <= 0:
        return None
    candidates = np.asarray(MAGNITUDE_CEILING_CANDIDATES, dtype=float)
    idx = int(np.argmin(np.abs(candidates - max_value)))
    candidate = float(candidates[idx])
    # Only call it a digital ceiling when the observed maximum is actually close.
    if abs(max_value - candidate) <= max(16.0, 0.01 * candidate):
        return candidate
    return None


def _magnitude_full_volume_stats(values: np.ndarray) -> dict[str, float | bool]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite in-mask magnitude voxels")
    raw_max = float(np.max(values))
    ceiling = _nearest_digital_ceiling(raw_max)
    saturated = False
    near_fraction = 0.0
    if ceiling is not None:
        threshold = ceiling - MAGNITUDE_CEILING_TOLERANCE
        near_fraction = float(np.mean(values >= threshold))
        saturated = near_fraction >= MAGNITUDE_NEAR_CEILING_FRACTION
    return {
        "mag_p99": float(np.percentile(values, MAGNITUDE_SCALE_PERCENTILE)),
        "mag_p97": float(np.percentile(values, MAGNITUDE_SATURATED_SCALE_PERCENTILE)),
        "mag_p50": float(np.percentile(values, 50.0)),
        "mag_max": raw_max,
        "mag_detected_ceiling": np.nan if ceiling is None else ceiling,
        "mag_near_ceiling_fraction": near_fraction,
        "mag_saturated": bool(saturated),
    }


def build_ceaset_case_stats(cfg: PipelineConfig = CFG, *, overwrite: bool = False) -> pd.DataFrame:
    if CEASET_STATS_CSV.is_file() and not overwrite:
        print(f"[SKIP] Loading existing CEASET stats: {CEASET_STATS_CSV}")
        return pd.read_csv(CEASET_STATS_CSV)

    rows = []
    case_ids = _available_complete_case_ids(cfg)
    for case_id in case_ids:
        echo_times = echo_times_for_case(case_id, cfg)
        _, mask_raw = load_float_nifti(_eroded_mask_path(case_id, cfg))
        mask = orient_for_processing(mask_raw) > 0.5
        for echo_label, te_ms in zip(cfg.echo_labels, echo_times):
            _, mag_raw = load_float_nifti(_prepared_case_path(case_id, "magnitude", echo_label, cfg))
            _, phase_raw = load_float_nifti(_prepared_case_path(case_id, "phase", echo_label, cfg))
            mag = orient_for_processing(mag_raw)
            phase = orient_for_processing(phase_raw)
            if mag.shape != mask.shape or phase.shape != mask.shape:
                raise ValueError(f"Geometry mismatch for {case_id} {echo_label}")
            mag_stats = _magnitude_full_volume_stats(mag[mask])
            phase_values = np.abs(phase[mask])
            phase_values = phase_values[np.isfinite(phase_values)]
            if phase_values.size == 0:
                raise ValueError(f"No finite phase voxels for {case_id} {echo_label}")
            phase_abs_p = float(np.percentile(phase_values, PHASE_RATE_PERCENTILE))
            rows.append({
                "case_id": case_id,
                "echo_label": echo_label,
                "echo_time_ms": float(te_ms),
                "echo_time_source": echo_time_source_for_case(case_id, cfg),
                **mag_stats,
                "phase_abs_p99": phase_abs_p,
                "phase_rate_p99_per_ms": phase_abs_p / max(float(te_ms), NORMALIZATION_EPSILON),
                "mask_voxels": int(mask.sum()),
            })
        print(f"[STATS] {case_id} | TEs={echo_times}")

    frame = pd.DataFrame(rows).sort_values(["case_id", "echo_time_ms"]).reset_index(drop=True)
    CEASET_STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CEASET_STATS_CSV, index=False)
    print(f"[OK] Wrote {CEASET_STATS_CSV}")
    return frame


def choose_reversible_scale(
    case_stats: pd.DataFrame,
    selected_echo_labels: Sequence[str],
    modality: str,
) -> float:
    subset = case_stats[case_stats["echo_label"].isin(list(selected_echo_labels))].copy()
    if len(subset) != len(set(selected_echo_labels)):
        raise ValueError("Missing normalization statistics for one or more selected echoes")
    subset = subset.sort_values("echo_time_ms")

    if modality == "magnitude":
        usable = subset[~subset["mag_saturated"].astype(bool)]
        if not usable.empty:
            scale = float(usable.iloc[0]["mag_p99"])
        else:
            scale = float(subset.iloc[0]["mag_p97"])
    elif modality == "phase":
        rates = pd.to_numeric(subset["phase_rate_p99_per_ms"], errors="coerce").to_numpy(dtype=float)
        rates = rates[np.isfinite(rates) & (rates > NORMALIZATION_EPSILON)]
        if rates.size == 0:
            raise ValueError("No usable phase-rate scale")
        scale = float(np.median(rates) * PHASE_SCALE_REFERENCE_TE_MS)
    else:
        raise ValueError(modality)

    if not np.isfinite(scale) or scale <= NORMALIZATION_EPSILON:
        raise ValueError(f"Invalid reversible {modality} scale: {scale}")
    return scale


def apply_reversible_normalization(volume: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(volume, dtype=np.float32) / float(scale)
    out[~np.isfinite(out)] = 0.0
    return out


def invert_reversible_normalization(volume_norm: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(volume_norm, dtype=np.float32) * float(scale)
    out[~np.isfinite(out)] = 0.0
    return out


def _training_patient_key_from_case_id(case_id):
    # case 001_ses-02_run-03 -> sub-001; all sessions/runs stay together.
    return 'sub-' + re.split(r'_ses-|_run-', str(case_id), maxsplit=1)[0]


def _canonical_split_label(value):
    mapping = {'train':'training', 'training':'training', 'eval':'eval',
               'validation':'eval', 'val':'eval', 'evaluation':'eval', 'test':'test'}
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f'Unknown partition: {value}')
    return mapping[key]


def partition_subjects(patients, percentages, seed):
    """Deterministic subject allocation; closest counts with nonempty active partitions."""
    patients = sorted(set(patients))
    weights = np.asarray(percentages, dtype=float)
    if (weights.shape != (3,) or not np.isfinite(weights).all() or
            (weights[:2] <= 0).any() or weights[2] < 0 or
            not np.isclose(weights.sum(), 100., rtol=0, atol=1e-8)):
        raise ValueError('Invalid partition percentages')
    active = weights > 0
    if len(patients) < int(active.sum()):
        raise ValueError('Too few independent subjects for nonempty requested partitions')
    target = weights * len(patients) / 100.
    counts = active.astype(int)
    for _ in range(len(patients) - int(counts.sum())):
        deficits = np.where(active, target - counts, -np.inf)
        counts[int(np.argmax(deficits))] += 1
    np.random.default_rng(seed).shuffle(patients)
    assignments = {}
    offset = 0
    for label, count in zip(('training', 'eval', 'test'), counts):
        assignments.update({p: label for p in patients[offset:offset+count]})
        offset += count
    return assignments


def load_or_create_ceaset_split(cfg=CFG):
    cases = _available_complete_case_ids(cfg)
    path = TRAIN_EVAL_SPLIT_MANIFEST_XLSX
    expected = {cid: _training_patient_key_from_case_id(cid) for cid in cases}
    percentages = [TRAINING_SPLIT_PERCENT, EVALUATION_SPLIT_PERCENT, TEST_SPLIT_PERCENT]
    settings = dict(zip(('training_percent', 'evaluation_percent', 'test_percent'), percentages))
    settings['seed'] = TRAIN_EVAL_SPLIT_SEED
    rebuild = False
    legacy_two_way = False
    if path.is_file():
        with pd.ExcelFile(path) as book:
            has_settings = 'split_settings' in book.sheet_names
            recorded = (pd.read_excel(book, sheet_name='split_settings').iloc[0].to_dict()
                        if has_settings else {})
        # The original dashboard stored only patient_split and used 80% training /
        # 20% evaluation. Accept its workbook *read-only* in the explicit experiment
        # mode; existing 'test' labels are treated as validation, never final test.
        legacy_two_way = (SMALL_DATASET_MODE and not has_settings and
                          percentages == [80.0, 20.0, 0.0] and TRAIN_EVAL_SPLIT_SEED == 42)
        if legacy_two_way:
            # Only adopt an original workbook when its actual cohort and counts
            # match the requested allocation. Otherwise follow the normal
            # explicit backup-and-repartition path, never silently relabel.
            old = pd.read_excel(path, sheet_name='patient_split', dtype=str)
            if not {'case_id','patient_key','split'} <= set(old.columns):
                legacy_two_way = False
            else:
                old = old[old.case_id.isin(cases)].copy()
                labels = old['split'].map(_canonical_split_label).replace({'test':'eval'})
                actual = old.assign(split=labels).groupby('split')['patient_key'].nunique().to_dict()
                target = partition_subjects(expected.values(), percentages, TRAIN_EVAL_SPLIT_SEED)
                desired = {name: list(target.values()).count(name) for name in ('training','eval')}
                legacy_two_way = (set(old.case_id) == set(cases) and
                                  not old.case_id.duplicated().any() and
                                  actual == desired)
        mismatch = not legacy_two_way and any(recorded.get(k) != v for k, v in settings.items())
        if mismatch:
            if not (REBUILD_PATIENT_SPLIT and OVERWRITE_DERIVED and PIPELINE_ACTION == 'prepare'):
                raise ValueError('Legacy or changed partition settings: enable REBUILD_PATIENT_SPLIT and OVERWRITE_DERIVED, then preprocess. Existing assignments are preserved until an explicit rebuild; do not resume old models after repartitioning.')
            # Validate requested allocation before archiving or changing anything.
            partition_subjects(expected.values(), percentages, TRAIN_EVAL_SPLIT_SEED)
            import shutil, uuid
            backup = path.with_name(path.stem + '.backup-' + uuid.uuid4().hex + path.suffix)
            shutil.copy2(path, backup)
            print(f'[SPLIT] Archived previous workbook: {backup}')
            rebuild = True
    if path.is_file() and not rebuild:
        frame = pd.read_excel(path, sheet_name='patient_split', dtype=str)
        if not {'case_id','patient_key','split'} <= set(frame.columns):
            raise ValueError(f'{path}: patient_split requires case_id, patient_key, split')
        frame = frame[['case_id','patient_key','split']].copy()
        frame['split'] = frame['split'].map(_canonical_split_label)
        if legacy_two_way:
            # A legacy 'test' set was not an independent held-out set in this
            # training architecture: it drives fixed-validation MAE/early stopping.
            frame['split'] = frame['split'].replace({'test': 'eval'})
        if frame['case_id'].duplicated().any():
            raise ValueError('Duplicate cases in patient_split')
        extra = set(frame.case_id) - set(cases)
        missing = set(cases) - set(frame.case_id)
        if missing:
            raise ValueError(f'New accepted cases lack a patient split: {sorted(missing)}. Update the workbook deliberately; existing patients must retain their split.')
        if extra:
            print('[SPLIT] Ignoring unavailable/excluded cases:', sorted(extra))
        frame = frame[frame.case_id.isin(cases)].copy()
        for row in frame.itertuples():
            if row.patient_key != expected[row.case_id]:
                raise ValueError(f'{row.case_id}: patient_key must be {expected[row.case_id]}, got {row.patient_key}')
        if legacy_two_way:
            print('[SPLIT] Reusing legacy two-way workbook without modifying it. '
                  'Its second partition is VALIDATION for checkpoint selection; '
                  'there is no untouched final test set.')
    else:
        if not rebuild and not CREATE_NEW_SPLIT_IF_MANIFEST_MISSING:
            raise FileNotFoundError(path)
        assignments = partition_subjects(expected.values(), percentages, TRAIN_EVAL_SPLIT_SEED)
        frame = pd.DataFrame([{'case_id': c, 'patient_key': expected[c],
                               'split': assignments[expected[c]]} for c in cases])
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            frame.to_excel(writer, sheet_name='patient_split', index=False)
            pd.DataFrame([settings]).to_excel(writer, sheet_name='split_settings', index=False)
    if frame.groupby('patient_key')['split'].nunique().max() > 1:
        raise ValueError('Patient leakage: runs/sessions of one subject span partitions.')
    required = {'training', 'eval'} | ({'test'} if TEST_SPLIT_PERCENT > 0 else set())
    if set(frame['split']) != required:
        raise ValueError('Each requested partition must contain an independent subject.')
    print(frame.groupby('split').agg(subjects=('patient_key','nunique'), cases=('case_id','nunique')))
    return frame.sort_values(['split','patient_key','case_id']).reset_index(drop=True)


def build_chunk_specs(shape, n_chunks, overlap):
    """Cover every voxel, including remainders; pad to the generator's 8-voxel stride."""
    shape, n_chunks = tuple(map(int, shape)), tuple(map(int, n_chunks))
    if len(shape) != 3 or len(n_chunks) != 3 or min(shape) < 1 or min(n_chunks) < 1:
        raise ValueError('Expected positive 3-D shape and grid')
    if overlap < 0 or any(n > s for n, s in zip(n_chunks, shape)):
        raise ValueError('Invalid overlap or more chunks than voxels')
    edges = [np.linspace(0, s, n+1, dtype=int) for s, n in zip(shape, n_chunks)]
    max_core = [int(np.diff(e).max()) for e in edges]
    desired = tuple(int(math.ceil((s+2*overlap)/8)*8) for s in max_core)
    specs = []
    for index in np.ndindex(n_chunks):
        cs = tuple(int(edges[d][index[d]]) for d in range(3))
        ce = tuple(int(edges[d][index[d]+1]) for d in range(3))
        rs = tuple(cs[d]-overlap for d in range(3))
        re_ = tuple(rs[d]+desired[d] for d in range(3))
        specs.append({'index': index, 'core_start': cs, 'core_end': ce,
            'source_start': tuple(max(0, x) for x in rs),
            'source_end': tuple(min(shape[d], re_[d]) for d in range(3)),
            'pad_before': tuple(max(0, -x) for x in rs),
            'pad_after': tuple(max(0, re_[d]-shape[d]) for d in range(3)),
            'chunk_shape': desired})
    return specs


def extract_chunk(volume, spec):
    slices = tuple(slice(a,b) for a,b in zip(spec['source_start'], spec['source_end']))
    pads = tuple(zip(spec['pad_before'], spec['pad_after']))
    if volume.ndim == 4:
        pads += ((0,0),)
    # Zero padding prevents replicated mask values outside the acquired FOV.
    return np.pad(volume[slices], pads, mode='constant', constant_values=0)


def ensure_4d_chunk(chunk):
    result = np.asarray(chunk, np.float32)
    if result.ndim == 3:
        result = result[..., None]
    if result.ndim != 4 or result.shape[-1] != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f'Invalid single-channel chunk: {result.shape}')
    return np.ascontiguousarray(result)


def _derived_signature(cfg=CFG):
    cases = _available_complete_case_ids(cfg)
    metadata = []
    for cid in cases:
        meta = json.loads(_case_metadata_path(cid, cfg).read_text())
        for item in meta['outputs']:
            if not Path(item['path']).is_file() or _file_signature(item['path']) != item:
                raise RuntimeError(f'{cid}: prepared output changed: {item["path"]}. Rebuild preprocessing.')
        metadata.append((cid, meta['signature']))
    split = load_or_create_ceaset_split(cfg)
    return _digest({'cases': metadata, 'split': split.to_dict('records'),
        'echo_labels': cfg.echo_labels, 'grid': cfg.chunk_grid, 'overlap': cfg.chunk_overlap_voxels,
        'normalization': [MAGNITUDE_SCALE_PERCENTILE, MAGNITUDE_SATURATED_SCALE_PERCENTILE,
            MAGNITUDE_NEAR_CEILING_FRACTION, MAGNITUDE_CEILING_CANDIDATES,
            MAGNITUDE_CEILING_TOLERANCE, PHASE_RATE_PERCENTILE, PHASE_SCALE_REFERENCE_TE_MS],
        'revision': CEASET_PIPELINE_REVISION})


def validate_training_artifacts(cfg=CFG, *, sample_builder=None):
    """Validate prepared artifacts; optionally exercise a caller-supplied sampler.

    Preprocessing has no dependency on the TensorFlow training module. The training
    entry point supplies its sampler to additionally validate actual model inputs.
    """
    marker = cfg.preprocessing_data_root/'derived_data_signature.json'
    if not marker.is_file():
        raise FileNotFoundError('Derived data has no completion marker. Run PIPELINE_ACTION="prepare" first.')
    current = _derived_signature(cfg)
    if json.loads(marker.read_text()).get('signature') != current:
        raise RuntimeError('Dataset, split, or chunk/normalization settings changed. Rebuild with OVERWRITE_DERIVED=True.')
    frame = load_active_ceaset_manifest(modality='magnitude')
    load_active_ceaset_manifest(modality='phase')
    split = load_or_create_ceaset_split(cfg).set_index('case_id')['split'].to_dict()
    if set(frame.case_id.astype(str)) != set(split):
        raise RuntimeError('Cached cohort differs from the accepted patient split.')
    if any(str(r.split) != split[str(r.case_id)] for r in frame.itertuples()):
        raise RuntimeError('Cached split assignments disagree with patient_split.')
    cols = ['mask_path'] + [f'{mod}_{echo}_path' for mod in ('magnitude','phase') for echo in cfg.echo_labels]
    for col in cols:
        missing = [p for p in frame[col].astype(str).unique() if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f'Incomplete training cache ({col}): {missing[:3]}')
    stats = pd.read_csv(CEASET_STATS_CSV)
    if set(stats.case_id.astype(str)) != set(split):
        raise RuntimeError('Normalization statistics use a different cohort.')
    for cid, group in stats.groupby('case_id'):
        if set(group.echo_label) != set(cfg.echo_labels) or len(group) != len(cfg.echo_labels):
            raise RuntimeError(f'{cid}: incomplete/duplicate per-echo statistics')
    if sample_builder is not None:
        # Exercise the actual sampler (last acquired echo as target) before a long run.
        for modality in ('magnitude','phase'):
            for partition in ('training','eval'):
                row = frame[frame.split == partition].iloc[0]
                inputs, target, _ = sample_builder(
                    row, modality, stats, len(cfg.echo_labels)-1, [0,1,2])
                if not all(np.all(np.isfinite(a)) for a in [target, *inputs.values()]):
                    raise RuntimeError('Non-finite training sample')
                if any(s % 8 for s in target.shape[:3]) or not inputs['brain_mask'].any():
                    raise RuntimeError('Invalid training chunk geometry/mask')
    print(f'[READY] {len(split)} cases, {len(frame)} chunks, {len(cfg.echo_labels)} acquired echoes.')
    return frame


def prepare_training_data(cfg=CFG):
    manifest = run_abrim_preprocessing(cfg, overwrite=OVERWRITE_PREPROCESSING)
    signature = _derived_signature(cfg)
    marker = cfg.preprocessing_data_root/'derived_data_signature.json'
    if marker.is_file() and not OVERWRITE_DERIVED:
        if json.loads(marker.read_text()).get('signature') != signature:
            raise RuntimeError('Derived data is stale. Set OVERWRITE_DERIVED=True to rebuild.')
        print('[DERIVED SKIP] Existing completed data matches this configuration.')
        return manifest
    # An interrupted build is rebuilt fully; a changed build cannot reuse stale chunks.
    if marker.exists():
        marker.unlink()
    build_ceaset_case_stats(cfg, overwrite=True)
    create_ceaset_chunks(cfg, overwrite=True)
    build_ceaset_fast_npy_cache(cfg, modalities=('magnitude','phase'), overwrite=True)
    _json_write(marker, {'signature': signature, 'revision': CEASET_PIPELINE_REVISION})
    return manifest


def _save_h5_chunk(path: Path, data: np.ndarray, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.asarray(data, dtype=np.float32), compression=CEASET_HDF5_COMPRESSION)


def _save_nii_chunk(path: Path, data: np.ndarray, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine=np.eye(4)), str(path))


def create_ceaset_chunks(
    cfg: PipelineConfig = CFG,
    *,
    overwrite: bool = OVERWRITE_CHUNKS,
) -> pd.DataFrame:
    split = load_or_create_ceaset_split(cfg)
    split_by_case = dict(zip(split["case_id"].astype(str), split["split"].astype(str)))
    rows=[]

    for case_id in sorted(split_by_case):
        split_name = split_by_case[case_id]
        echo_times = echo_times_for_case(case_id, cfg)
        _, mask_raw = load_float_nifti(_eroded_mask_path(case_id, cfg))
        mask = orient_for_processing(mask_raw).astype(np.float32)

        # Load all echoes once per case, then cut the same spatial specs.
        volumes: dict[tuple[str,str], np.ndarray] = {}
        reference_shape = None
        for modality in ("magnitude", "phase"):
            for echo in cfg.echo_labels:
                _, raw = load_float_nifti(_prepared_case_path(case_id, modality, echo, cfg))
                vol = orient_for_processing(raw).astype(np.float32, copy=False)
                if reference_shape is None:
                    reference_shape = vol.shape
                if vol.shape != reference_shape:
                    raise ValueError(f"Shape mismatch for {case_id}: {modality} {echo} {vol.shape} vs {reference_shape}")
                volumes[(modality,echo)] = vol
        if mask.shape != reference_shape:
            raise ValueError(f"Mask shape mismatch for {case_id}: {mask.shape} vs {reference_shape}")

        specs = build_chunk_specs(reference_shape, cfg.chunk_grid, cfg.chunk_overlap_voxels)
        for spec in specs:
            i,j,k = spec["index"]
            chunk_id = f"{case_id}_{i}{j}{k}"
            row = {
                "split": split_name,
                "case_id": case_id,
                "chunk_id": chunk_id,
                "chunk_i": i, "chunk_j": j, "chunk_k": k,
                "source_shape": "x".join(map(str, reference_shape)),
                "chunk_shape": "x".join(map(str, spec["chunk_shape"])),
                "echo_times_ms": json.dumps([float(v) for v in echo_times]),
                "echo_time_source": echo_time_source_for_case(case_id, cfg),
            }
            mask_chunk = ensure_4d_chunk(extract_chunk(mask, spec))
            if not np.any(mask_chunk > 0.5):
                continue
            mask_path = CEASET_CHUNK_ROOT / split_name / "mask" / f"{chunk_id}.h5"
            _save_h5_chunk(mask_path, mask_chunk, overwrite=overwrite)
            row["mask_path"] = str(mask_path)

            for modality in ("magnitude", "phase"):
                for echo in cfg.echo_labels:
                    chunk = ensure_4d_chunk(extract_chunk(volumes[(modality,echo)], spec))
                    base = CEASET_CHUNK_ROOT / split_name / modality / echo / chunk_id
                    if cfg.write_chunk_hdf5:
                        path = base.with_suffix(".h5")
                        _save_h5_chunk(path, chunk, overwrite=overwrite)
                    elif cfg.write_chunk_nifti:
                        path = base.with_suffix(".nii")
                    else:
                        raise RuntimeError("At least one chunk output format must be enabled")
                    if cfg.write_chunk_nifti:
                        _save_nii_chunk(base.with_suffix(".nii"), chunk, overwrite=overwrite)
                    row[f"{modality}_{echo}_path"] = str(path)

            for label, values in (
                ("core_start", spec["core_start"]), ("core_end", spec["core_end"]),
                ("source_start", spec["source_start"]), ("source_end", spec["source_end"]),
                ("pad_before", spec["pad_before"]), ("pad_after", spec["pad_after"]),
            ):
                for axis, value in zip(("H","W","D"), values):
                    row[f"{label}_{axis}"] = int(value)
            rows.append(row)
        print(f"[CHUNKS] {case_id} -> {split_name} | {len(specs)} chunks | shape={reference_shape}")
        del volumes, mask
        gc.collect()

    if not rows:
        raise RuntimeError("No nonempty brain chunks were generated")
    manifest = pd.DataFrame(rows)
    CEASET_CHUNK_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(CEASET_CHUNK_MANIFEST_CSV, index=False)
    print(f"[OK] Wrote chunk manifest: {CEASET_CHUNK_MANIFEST_CSV}")
    return manifest


def _save_npy_float32(path: Path, array: np.ndarray, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, np.asarray(array, dtype=np.float32), allow_pickle=False)
    tmp.replace(path)


def _parse_manifest_source_shape(text: str) -> tuple[int, int, int]:
    vals = tuple(int(v) for v in str(text).split("x"))
    if len(vals) != 3:
        raise ValueError(f"Invalid source_shape: {text!r}")
    return vals


def estimate_fast_npy_cache_gib(
    base_manifest: pd.DataFrame,
    modalities: Sequence[str],
    cfg: PipelineConfig = CFG,
) -> float:
    """
    Approximate output size. Each cached array is raw float32; .npy header
    overhead is negligible relative to a 3-D chunk.
    """
    if base_manifest.empty:
        return 0.0
    first = base_manifest.iloc[0]
    chunk_xyz = tuple(int(v) for v in str(first["chunk_shape"]).split("x"))
    voxels = int(np.prod(chunk_xyz))
    # one mask + all echoes for each requested modality
    arrays_per_chunk = 1 + len(tuple(modalities)) * len(cfg.echo_labels)
    total_bytes = len(base_manifest) * voxels * arrays_per_chunk * 4
    return float(total_bytes) / (1024 ** 3)


def build_ceaset_fast_npy_cache(
    cfg: PipelineConfig = CFG,
    *,
    modalities: Sequence[str] = FAST_NPY_CACHE_MODALITIES,
    overwrite: bool = FAST_NPY_CACHE_OVERWRITE,
) -> pd.DataFrame:
    """
    Build uncompressed training chunks from prepared whole-volume images.

    This intentionally DOES NOT convert the gzip HDF5 files one by one.
    Instead each case's full prepared NIfTI echoes are loaded once, then the
    exact same build_chunk_specs/extract_chunk geometry is reused. This avoids
    tens of thousands of HDF5+gzip reads during cache construction.
    """
    modalities = tuple(str(m).lower() for m in modalities)
    invalid = sorted(set(modalities) - {"magnitude", "phase"})
    if invalid:
        raise ValueError(f"Unsupported cache modalities: {invalid}")

    base_manifest = pd.read_csv(CEASET_CHUNK_MANIFEST_CSV)
    if base_manifest.empty:
        raise RuntimeError("CEASET chunk manifest is empty.")

    # Continue an existing fast manifest so magnitude and phase can be cached
    # in separate runs without losing already-resolved paths.
    if CEASET_FAST_NPY_MANIFEST_CSV.is_file() and not overwrite:
        fast_manifest = pd.read_csv(CEASET_FAST_NPY_MANIFEST_CSV)
        fast_manifest = _resolve_renamed_training_cache_paths(fast_manifest, CEASET_FAST_NPY_MANIFEST_CSV)
        if set(fast_manifest["chunk_id"].astype(str)) != set(base_manifest["chunk_id"].astype(str)):
            raise RuntimeError(
                "Existing fast-cache manifest does not match the current CEASET "
                "chunk manifest. Remove the fast-cache manifest or rebuild after "
                "changing the cohort/chunk geometry."
            )
        fast_manifest = fast_manifest.set_index("chunk_id", drop=False)
    else:
        fast_manifest = base_manifest.copy().set_index("chunk_id", drop=False)

    estimate_gib = estimate_fast_npy_cache_gib(base_manifest, modalities, cfg)
    print(
        f"[FAST CACHE] modalities={modalities} | chunks={len(base_manifest)} | "
        f"approx raw cache size for requested data={estimate_gib:.1f} GiB"
    )
    print(f"[FAST CACHE] root: {CEASET_FAST_NPY_CACHE_ROOT}")

    grouped = base_manifest.groupby("case_id", sort=True)
    for case_number, (case_id, case_rows) in enumerate(grouped, start=1):
        case_id = str(case_id)
        case_rows = case_rows.reset_index(drop=True)
        split_names = sorted(set(case_rows["split"].astype(str)))
        if len(split_names) != 1:
            raise RuntimeError(f"{case_id}: multiple split labels: {split_names}")
        split_name = split_names[0]

        # Geometry is authoritative from the prepared mask/source volumes.
        _, mask_raw = load_float_nifti(_eroded_mask_path(case_id, cfg))
        mask = orient_for_processing(mask_raw).astype(np.float32, copy=False)
        reference_shape = tuple(int(v) for v in mask.shape)
        specs = build_chunk_specs(
            reference_shape,
            cfg.chunk_grid,
            cfg.chunk_overlap_voxels,
        )
        spec_by_index = {
            tuple(int(v) for v in spec["index"]): spec
            for spec in specs
        }

        # Load each requested whole-volume echo ONCE for this case.
        volumes: dict[tuple[str, str], np.ndarray] = {}
        for modality in modalities:
            for echo in cfg.echo_labels:
                _, raw = load_float_nifti(
                    _prepared_case_path(case_id, modality, echo, cfg)
                )
                vol = orient_for_processing(raw).astype(np.float32, copy=False)
                if tuple(vol.shape) != reference_shape:
                    raise ValueError(
                        f"{case_id}: {modality}/{echo} shape {vol.shape} "
                        f"!= mask shape {reference_shape}"
                    )
                volumes[(modality, echo)] = vol

        for _, row in case_rows.iterrows():
            chunk_id = str(row["chunk_id"])
            idx = (
                int(row["chunk_i"]),
                int(row["chunk_j"]),
                int(row["chunk_k"]),
            )
            spec = spec_by_index[idx]

            mask_path = (
                CEASET_FAST_NPY_CACHE_ROOT
                / split_name / "mask" / f"{chunk_id}.npy"
            )
            if overwrite or not mask_path.is_file():
                _save_npy_float32(
                    mask_path,
                    ensure_4d_chunk(extract_chunk(mask, spec)),
                    overwrite=overwrite,
                )
            fast_manifest.at[chunk_id, "mask_path"] = str(mask_path)

            for modality in modalities:
                for echo in cfg.echo_labels:
                    out_path = (
                        CEASET_FAST_NPY_CACHE_ROOT
                        / split_name / modality / echo / f"{chunk_id}.npy"
                    )
                    if overwrite or not out_path.is_file():
                        _save_npy_float32(
                            out_path,
                            ensure_4d_chunk(
                                extract_chunk(volumes[(modality, echo)], spec)
                            ),
                            overwrite=overwrite,
                        )
                    fast_manifest.at[
                        chunk_id,
                        f"{modality}_{echo}_path",
                    ] = str(out_path)

        del volumes, mask
        gc.collect()

        if case_number % 5 == 0 or case_number == len(grouped):
            print(
                f"[FAST CACHE] cases {case_number}/{len(grouped)} complete",
                flush=True,
            )

        # Incremental manifest makes the cache resumable after interruption.
        CEASET_FAST_NPY_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
        fast_manifest.reset_index(drop=True).to_csv(
            CEASET_FAST_NPY_MANIFEST_CSV,
            index=False,
        )

    result = fast_manifest.reset_index(drop=True)
    print(f"[FAST CACHE] manifest: {CEASET_FAST_NPY_MANIFEST_CSV}")
    return result


def _resolve_renamed_training_cache_paths(frame: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    """Rebase old absolute cache references after the directory rename.

    Does not create, move, or overwrite any NPY/HDF5 data. Only persists a
    corrected CSV when every referenced file is verified to exist.
    """
    frame = frame.copy()
    path_cols = [c for c in frame.columns if c == "mask_path" or c.endswith("_path")]
    changes = 0
    missing = []
    aliases = {
        "10_training_cache_npy": "training_cache",
        "09_chunks_553": "training_chunks",
    }

    for col in path_cols:
        for index, value in frame[col].items():
            if pd.isna(value) or not str(value).strip():
                continue
            original = Path(str(value))
            if original.is_file():
                continue

            # First support old absolute paths that differ only in folder name,
            # including projects relocated between disks.
            candidates = []
            parts = original.parts
            for known, selected in {'training_cache': CEASET_FAST_NPY_CACHE_ROOT,
                                    '10_training_cache_npy': CEASET_FAST_NPY_CACHE_ROOT,
                                    'training_chunks': CEASET_CHUNK_ROOT,
                                    '09_chunks_553': CEASET_CHUNK_ROOT}.items():
                if known in parts:
                    candidates.append(selected.joinpath(*parts[parts.index(known)+1:]))
            if "preprocess" in parts:
                rel_parts = parts[parts.index("preprocess") + 1:]
                if rel_parts and rel_parts[0] == "ceaset_abrim_v8":
                    rel_parts = rel_parts[1:]
                if rel_parts:
                    corrected = (aliases.get(rel_parts[0], rel_parts[0]), *rel_parts[1:])
                    candidates.append(CEASET_ROOT.joinpath(*corrected))

            # Also resolve renamed roots relative to the current project.
            for old_name, new_name in aliases.items():
                if old_name in parts:
                    at = parts.index(old_name)
                    candidates.append(CEASET_ROOT / new_name / Path(*parts[at + 1:]))

            matched = next((p for p in candidates if p.is_file()), None)
            if matched is None:
                missing.append((col, str(original), str(candidates[0]) if candidates else "no mapped path"))
                continue
            frame.at[index, col] = str(matched.resolve())
            changes += 1

    if changes:
        print(f"[CACHE PATHS] Resolved {changes} paths in memory; manifest unchanged.", flush=True)

    if missing:
        print(f"[CACHE PATHS] {len(missing)} missing arrays. Examples:", flush=True)
        for col, old, expected in missing[:3]:
            print(f"  {col}: {old} -> {expected}", flush=True)
        print("[CACHE PATHS] If the arrays are still in the old directory, "
              "select its existing path in the dashboard, or rebuild only the NumPy cache.", flush=True)
    return frame


def load_active_ceaset_manifest(
    *,
    modality: str | None = None,
    require_fast: bool | None = None,
) -> pd.DataFrame:
    """
    Resolve the manifest used by training/evaluation and verify that the selected
    modality really points to .npy files when fast-cache mode is required.
    """
    if require_fast is None:
        require_fast = bool(TRAINING_REQUIRE_FAST_NPY_CACHE)

    if TRAINING_USE_FAST_NPY_CACHE:
        if not CEASET_FAST_NPY_MANIFEST_CSV.is_file():
            if require_fast:
                raise FileNotFoundError(
                    "Fast CEASET NumPy cache is required but its manifest is missing:\n"
                    f"  {CEASET_FAST_NPY_MANIFEST_CSV}\n"
                    "Run the FAST NPY CACHE execution cell first."
                )
            path = CEASET_CHUNK_MANIFEST_CSV
        else:
            path = CEASET_FAST_NPY_MANIFEST_CSV
    else:
        path = CEASET_CHUNK_MANIFEST_CSV

    frame = pd.read_csv(path)
    frame = _resolve_renamed_training_cache_paths(frame, path)

    if modality is not None and require_fast:
        modality = str(modality)
        cols = [f"{modality}_{echo}_path" for echo in CFG.echo_labels]
        missing_cols = [c for c in cols if c not in frame.columns]
        if missing_cols:
            raise RuntimeError(f"Fast-cache manifest missing columns: {missing_cols}")

        bad = np.zeros(len(frame), dtype=bool)
        for col in ["mask_path", *cols]:
            bad |= ~frame[col].astype(str).str.endswith(".npy").to_numpy()
        if np.any(bad):
            examples = frame.loc[bad, ["case_id", "chunk_id", *cols[:1]]].head(10)
            raise RuntimeError(
                f"Fast NumPy cache is incomplete for modality={modality}. "
                f"{int(np.sum(bad))} chunks still reference non-NPY data.\n"
                f"Examples:\n{examples.to_string(index=False)}"
            )

    return frame


def load_chunk(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        # Uncompressed NumPy cache: no HDF5 global lock and no gzip decoding.
        arr = np.load(path, allow_pickle=False)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
    elif path.suffix == ".h5":
        with h5py.File(path, "r") as handle:
            key = "data" if "data" in handle else next(iter(handle.keys()))
            arr = np.asarray(handle[key], dtype=np.float32)
    elif path.name.endswith(".nii") or path.name.endswith(".nii.gz"):
        arr = nib.load(str(path)).get_fdata(dtype=np.float32)
    else:
        raise ValueError(f"Unsupported chunk file: {path}")
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4-D chunk, got {arr.shape} from {path}")
    arr[~np.isfinite(arr)] = 0.0
    return np.ascontiguousarray(arr, dtype=np.float32)


def magnitude_physics_baseline_np(
    echo_images: np.ndarray,
    echo_times_ms: np.ndarray,
    echo_mask: np.ndarray,
    target_te_ms: float,
    brain_mask: np.ndarray,
) -> np.ndarray:
    valid = np.flatnonzero(echo_mask > 0.5)
    if valid.size == 0:
        raise ValueError("At least one input echo is required")
    x = np.maximum(echo_images[valid, ..., 0], 0.0).astype(np.float32)
    t = echo_times_ms[valid].astype(np.float32)
    m = brain_mask[..., 0] > 0.5

    if valid.size == 1:
        dt = float(target_te_ms - t[0])
        baseline = x[0] * np.exp(-MAG_SINGLE_ECHO_R2STAR_PRIOR_PER_MS * dt)
    else:
        eps = 1e-5
        y = np.log(np.maximum(x, eps))
        # Magnitude-squared weighting downweights very noisy late echoes.
        w = np.maximum(x * x, eps)
        wsum = np.sum(w, axis=0) + eps
        t_view = t[:, None, None, None]
        tbar = np.sum(w * t_view, axis=0) / wsum
        ybar = np.sum(w * y, axis=0) / wsum
        cov = np.sum(w * (t_view - tbar) * (y - ybar), axis=0)
        var = np.sum(w * (t_view - tbar) ** 2, axis=0) + eps
        slope = cov / var
        # Magnitude decay prior: do not let noisy fits imply explosive positive decay rates.
        slope = np.clip(slope, -MAG_R2STAR_MAX_PER_MS, 0.0)
        intercept = ybar - slope * tbar
        baseline = np.exp(intercept + slope * float(target_te_ms))

    baseline = np.asarray(baseline, dtype=np.float32)
    baseline[~m] = 0.0
    baseline[~np.isfinite(baseline)] = 0.0
    return baseline[..., None]


def phase_physics_baseline_np(
    echo_images: np.ndarray,
    echo_times_ms: np.ndarray,
    echo_mask: np.ndarray,
    target_te_ms: float,
    brain_mask: np.ndarray,
    reliability_images: np.ndarray | None = None,
) -> np.ndarray:
    """
    CLEAR-SWI-inspired phase baseline.

    After phase unwrapping + consistent high-pass filtering, each echo supplies
    a local-frequency estimate omega_i ~= phi_i / TE_i.  We combine those
    frequency estimates and project them to the requested target TE.

    Training can additionally provide corresponding magnitude chunks.  Their
    squared magnitude acts as an SNR proxy, analogous in spirit to CLEAR-SWI's
    inverse-variance weighting.  A robust factor downweights echoes whose
    local-frequency estimate is an outlier relative to the other input echoes.

    This intentionally remains a FIRST-ORDER baseline.  Nonlinear phase
    evolution from multi-compartment susceptibility/microstructure is left for
    the CNN residual to learn, and the V7 physics LOSS only constrains coarse
    spatial scales.
    """
    valid = np.flatnonzero(echo_mask > 0.5)
    if valid.size == 0:
        raise ValueError("At least one input echo is required")

    x = echo_images[valid, ..., 0].astype(np.float32)
    t = echo_times_ms[valid].astype(np.float32)
    m = brain_mask[..., 0] > 0.5

    t_view = np.maximum(t[:, None, None, None], NORMALIZATION_EPSILON)
    omega = x / t_view

    if reliability_images is not None:
        rel = np.asarray(reliability_images, dtype=np.float32)
        if rel.ndim == 5:
            rel = rel[valid, ..., 0]
        elif rel.ndim == 4:
            rel = rel[valid, ...]
        else:
            raise ValueError(
                f"Expected reliability_images [K,H,W,D,(1)], got {rel.shape}"
            )
        weights = np.maximum(rel, 0.0) ** 2
    else:
        weights = np.ones_like(omega, dtype=np.float32)

    if valid.size > 1 and PHASE_PHYSICS_ROBUST_FREQUENCY_WEIGHTING:
        median_omega = np.median(omega, axis=0)
        abs_dev = np.abs(omega - median_omega[None, ...])
        mad = np.median(abs_dev, axis=0)
        robust_scale = (
            1.4826 * mad + np.float32(NORMALIZATION_EPSILON)
        )
        u = abs_dev / (
            float(PHASE_PHYSICS_ROBUST_C) * robust_scale[None, ...]
            + np.float32(NORMALIZATION_EPSILON)
        )
        # Cauchy-like robust weight: smooth and bounded, unlike hard clipping.
        weights = weights / (1.0 + u * u)

    wsum = np.sum(weights, axis=0) + np.float32(NORMALIZATION_EPSILON)
    omega_hat = np.sum(weights * omega, axis=0) / wsum
    baseline = omega_hat * float(target_te_ms)

    baseline = np.asarray(baseline, dtype=np.float32)
    baseline[~m] = 0.0
    baseline[~np.isfinite(baseline)] = 0.0
    return baseline[..., None]


def physics_baseline_np(
    modality: str,
    echo_images: np.ndarray,
    echo_times_ms: np.ndarray,
    echo_mask: np.ndarray,
    target_te_ms: float,
    brain_mask: np.ndarray,
    *,
    reliability_images: np.ndarray | None = None,
) -> np.ndarray:
    if modality == "magnitude":
        return magnitude_physics_baseline_np(
            echo_images,
            echo_times_ms,
            echo_mask,
            target_te_ms,
            brain_mask,
        )
    if modality == "phase":
        return phase_physics_baseline_np(
            echo_images,
            echo_times_ms,
            echo_mask,
            target_te_ms,
            brain_mask,
            reliability_images=reliability_images,
        )
    raise ValueError(modality)

__all__ = [k for k in globals() if not k.startswith("__")]
