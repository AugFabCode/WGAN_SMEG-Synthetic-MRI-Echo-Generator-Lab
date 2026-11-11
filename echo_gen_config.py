"""Shared configuration. No TensorFlow import and no filesystem writes."""
import json
import hashlib
import os
from pathlib import Path

_OVERRIDES = json.loads(Path(os.environ['ECHO_GEN_CONFIG']).read_text()) if os.environ.get('ECHO_GEN_CONFIG') else {}
_ALLOWED = set(json.loads(Path(__file__).with_name('echo_gen_defaults.json').read_text()))
if not isinstance(_OVERRIDES, dict) or set(_OVERRIDES) - _ALLOWED:
    raise ValueError('Configuration must be an object with known settings only')

def _setting(name, default):
    value = _OVERRIDES.get(name, default)
    if isinstance(default, bool) and type(value) is not bool:
        raise ValueError(f'{name} must be boolean')
    if isinstance(default, (int,float)) and not isinstance(default,bool):
        import math
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
        if isinstance(default,int) and type(value) is not int:
            raise ValueError(f'{name} must be an integer')
    if isinstance(default, tuple) and not isinstance(value,(list,tuple)):
        raise ValueError(f'{name} must be an array')
    if isinstance(default,dict) and not isinstance(value,dict):
        raise ValueError(f'{name} must be an object')
    if isinstance(default, Path):
        if value is None:
            return default
        p = Path(value).expanduser()
        return (PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if isinstance(default, tuple):
        return tuple(value)
    if name == 'INPUT_COUNT_PROBABILITIES':
        return {int(k): v for k, v in value.items()}
    return value

def _cache_folder(root, original, repository):
    # Prefer populated manifests; never rename or move a cache.
    explicit = _OVERRIDES.get('CEASET_CHUNK_ROOT' if original == 'training_chunks' else 'CEASET_FAST_NPY_CACHE_ROOT')
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    a, b = root / original, root / repository
    if a.exists() and b.exists() and any(a.glob('*.csv')) and any(b.glob('*.csv')):
        raise ValueError(f'Both cache layouts contain manifests. Set the explicit cache path: {a}, {b}')
    if any(a.glob('*.csv')): return a
    if any(b.glob('*.csv')): return b
    return a if a.exists() else b if b.exists() else a

from pathlib import Path
import os, sys

CEASET_PIPELINE_REVISION = "2026-10-19_ABRIM9_V8_STARTUP_FIX"
PIPELINE_ACTION = os.environ.get("ECHO_GEN_ACTION", "inspect")  # inspect | prepare | train | all
# Resolve project paths from this notebook's working directory, not a disk name.
# Jupyter normally starts in the project directory (or a subdirectory).
# Set ECHO_GEN_PROJECT_ROOT explicitly if Jupyter starts elsewhere.
def _find_project_root():
    override = os.environ.get("ECHO_GEN_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"ECHO_GEN_PROJECT_ROOT does not exist: {root}")
        return root
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".gitignore").exists() or (candidate / "preprocess").is_dir() and (candidate / "models").is_dir():
            return candidate
    raise RuntimeError(
        f"Cannot locate Echo_Gen project root from {cwd}. "
        "Start Jupyter inside the repository or set ECHO_GEN_PROJECT_ROOT."
    )

PROJECT_ROOT = Path(os.environ.get("ECHO_GEN_PROJECT_ROOT", Path(__file__).resolve().parent)).expanduser().resolve()
OUTPUT_BASE = PROJECT_ROOT / "preprocess"
# Original ABRIM input is an external dataset, not part of the Git repository.
SOURCE_ROOT = Path(os.environ.get(
    "ECHO_GEN_ABRIM_ROOT", "/data/disk_3/student_projects/swi/ABRIM"
)).expanduser().resolve()
SOURCE_ROOT = _setting('SOURCE_ROOT', SOURCE_ROOT)
print(f"[PATHS] project={PROJECT_ROOT} | ABRIM={SOURCE_ROOT} | output={OUTPUT_BASE}")
# Separate generation namespace prevents accidental reuse of older MSHC chunks/models.
REFERENCE_ROOT = OUTPUT_BASE
PIPELINE_PROFILE = "SWI-CEASET-ABRIM-V8"
TRAINING_DEVICE_MODE = "single_gpu"  # dual_gpu | single_gpu | cpu (small tests)
TRAINING_DEVICE_MODE = _setting('TRAINING_DEVICE_MODE', TRAINING_DEVICE_MODE)
TRAINING_MODALITIES_TO_RUN = ("magnitude", "phase")
TRAINING_MODALITIES_TO_RUN = _setting('TRAINING_MODALITIES_TO_RUN', TRAINING_MODALITIES_TO_RUN)
OVERWRITE_PREPROCESSING = False
OVERWRITE_PREPROCESSING = _setting('OVERWRITE_PREPROCESSING', OVERWRITE_PREPROCESSING)
OVERWRITE_DERIVED = False
OVERWRITE_DERIVED = _setting('OVERWRITE_DERIVED', OVERWRITE_DERIVED)

ECHO_LABELS = tuple(f"TE{i}" for i in range(1, 10))
DEFAULT_ECHO_TIMES_MS = (6.14, 10.14, 14.14, 18.14, 22.14, 26.14, 30.14, 34.14, 38.14)
DEFAULT_ECHO_TIMES_MS = _setting('DEFAULT_ECHO_TIMES_MS', DEFAULT_ECHO_TIMES_MS)
ALLOW_PROTOCOL_TE_FALLBACK = True  # explicit shell protocol; source recorded per case
ALLOW_PROTOCOL_TE_FALLBACK = _setting('ALLOW_PROTOCOL_TE_FALLBACK', ALLOW_PROTOCOL_TE_FALLBACK)
PHASE_MODE = "auto"  # existing | compute | auto (prefer existing, else compute)
PHASE_MODE = _setting('PHASE_MODE', PHASE_MODE)
# Keys exclude the initial 'sub-': e.g. "001_run-1" or "001_ses-01_run-1".
FILTERED_PHASE_OVERRIDES = {}  # case_id: absolute filteredphase_orig 4-D NIfTI
FILTERED_PHASE_OVERRIDES = _setting('FILTERED_PHASE_OVERRIDES', FILTERED_PHASE_OVERRIDES)
PHASE_4D_OVERRIDES = {}        # case_id: absolute GENUINE raw phase 4-D NIfTI
PHASE_4D_OVERRIDES = _setting('PHASE_4D_OVERRIDES', PHASE_4D_OVERRIDES)
CLEAR_SWI_REQUIRES_X = False   # True only for the modified CLI requiring -x <phase>
CLEAR_SWI_REQUIRES_X = _setting('CLEAR_SWI_REQUIRES_X', CLEAR_SWI_REQUIRES_X)
CLEAR_SWI_EXTRA_ARGS = ()      # e.g. ("--no-phase-rescale",) for phase already in radians
CLEAR_SWI_EXTRA_ARGS = _setting('CLEAR_SWI_EXTRA_ARGS', CLEAR_SWI_EXTRA_ARGS)

EXPORT_SHELL_NORMALIZED = True
EXPORT_SHELL_NORMALIZED = _setting('EXPORT_SHELL_NORMALIZED', EXPORT_SHELL_NORMALIZED)
MAG_P_LOW, MAG_P_HIGH = 1.0, 99.0
MAG_REFERENCE_INDICES = (0, 1, 2)
MAG_REFERENCE_INDICES = _setting('MAG_REFERENCE_INDICES', MAG_REFERENCE_INDICES)
MAG_SATURATION_VALUE = 4095.0
MAG_SATURATION_VALUE = _setting('MAG_SATURATION_VALUE', MAG_SATURATION_VALUE)
MAG_SATURATION_TOLERANCE = 5.0
MAG_SATURATION_TOLERANCE = _setting('MAG_SATURATION_TOLERANCE', MAG_SATURATION_TOLERANCE)
MAG_MAX_NEAR_SATURATION_FRACTION = 0.001
MAG_MAX_NEAR_SATURATION_FRACTION = _setting('MAG_MAX_NEAR_SATURATION_FRACTION', MAG_MAX_NEAR_SATURATION_FRACTION)
MAG_OUTLIER_MARGIN = 0.5
MAG_OUTLIER_MARGIN = _setting('MAG_OUTLIER_MARGIN', MAG_OUTLIER_MARGIN)
MAG_HIGH_CAP_WARNING_FRACTION = 0.005
MAG_HIGH_CAP_WARNING_FRACTION = _setting('MAG_HIGH_CAP_WARNING_FRACTION', MAG_HIGH_CAP_WARNING_FRACTION)
PHASE_EXP_X0 = 0.5
PHASE_EXP_X0 = _setting('PHASE_EXP_X0', PHASE_EXP_X0)
SHELL_NORMALIZATION_EPSILON = 1e-12
SHELL_NORMALIZATION_EPSILON = _setting('SHELL_NORMALIZATION_EPSILON', SHELL_NORMALIZATION_EPSILON)
BET_EROSION_RADIUS_MM = 2.0
BET_EROSION_RADIUS_MM = _setting('BET_EROSION_RADIUS_MM', BET_EROSION_RADIUS_MM)

# Set TensorFlow environment defaults only before its first import. A diagnostic
# import/list_physical_devices call does not by itself initialize the runtime.
# Respect CUDA_VISIBLE_DEVICES supplied by the scheduler/user.
if "tensorflow" not in sys.modules:
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
else:
    print("[CONFIG] TensorFlow is already imported; keeping its runtime environment. "
          "Device setup will check GPU availability and memory growth.", flush=True)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
if PIPELINE_ACTION not in {"inspect", "prepare", "train", "all"}:
    raise ValueError("Invalid PIPELINE_ACTION")
if PHASE_MODE not in {"existing", "compute", "auto"}:
    raise ValueError("Invalid PHASE_MODE")
if TRAINING_DEVICE_MODE not in {"dual_gpu", "single_gpu", "cpu"}:
    raise ValueError("Invalid TRAINING_DEVICE_MODE")

# Model conditioning uses raw milliseconds internally divided by this constant.
TE_EMBEDDING_SCALE_MS = 30.0
TE_EMBEDDING_SCALE_MS = _setting('TE_EMBEDDING_SCALE_MS', TE_EMBEDDING_SCALE_MS)
MAX_INPUT_ECHOES = 4
MAX_INPUT_ECHOES = _setting('MAX_INPUT_ECHOES', MAX_INPUT_ECHOES)
MIN_INPUT_ECHOES = 1
MIN_INPUT_ECHOES = _setting('MIN_INPUT_ECHOES', MIN_INPUT_ECHOES)
INPUT_COUNT_PROBABILITIES = {1: 0.32, 2: 0.30, 3: 0.23, 4: 0.15}
INPUT_COUNT_PROBABILITIES = _setting('INPUT_COUNT_PROBABILITIES', INPUT_COUNT_PROBABILITIES)
SAMPLING_REGIME_WEIGHTS = {
    "forward": 0.35,       # all inputs earlier than target
    "interpolation": 0.30, # target bracketed by inputs
    "backward": 0.20,      # all inputs later than target
    "other": 0.15,
}
SAMPLING_REGIME_WEIGHTS = _setting('SAMPLING_REGIME_WEIGHTS', SAMPLING_REGIME_WEIGHTS)

# -----------------------------------------------------------------------------
# CLEAR-SWI / FSL preprocessing
# -----------------------------------------------------------------------------
CLEAR_SWI_PHASE_SCALING_TYPE = "tanh"       # only relevant to SWI output; filteredphase_orig is used for training
CLEAR_SWI_PHASE_SCALING_TYPE = _setting('CLEAR_SWI_PHASE_SCALING_TYPE', CLEAR_SWI_PHASE_SCALING_TYPE)
CLEAR_SWI_PHASE_SCALING_STRENGTH = 4
CLEAR_SWI_PHASE_SCALING_STRENGTH = _setting('CLEAR_SWI_PHASE_SCALING_STRENGTH', CLEAR_SWI_PHASE_SCALING_STRENGTH)
CLEAR_SWI_UNWRAPPING_ALGORITHM = "laplacian"
CLEAR_SWI_UNWRAPPING_ALGORITHM = _setting('CLEAR_SWI_UNWRAPPING_ALGORITHM', CLEAR_SWI_UNWRAPPING_ALGORITHM)
CLEAR_SWI_PHASE_HP_SIGMA_VOXELS = (4.0, 4.0, 0.0)
CLEAR_SWI_PHASE_HP_SIGMA_VOXELS = _setting('CLEAR_SWI_PHASE_HP_SIGMA_VOXELS', CLEAR_SWI_PHASE_HP_SIGMA_VOXELS)
# Existing shell filtered phase / laplacian CLEAR-SWI processing is used.

BET_REFERENCE_ECHO = "TE3"
BET_REFERENCE_ECHO = _setting('BET_REFERENCE_ECHO', BET_REFERENCE_ECHO)
BET_FRACTION = 0.20
BET_FRACTION = _setting('BET_FRACTION', BET_FRACTION)
BET_ROBUST_CENTER = True
BET_ROBUST_CENTER = _setting('BET_ROBUST_CENTER', BET_ROBUST_CENTER)
BET_CREATE_MASK = True
BET_CREATE_MASK = _setting('BET_CREATE_MASK', BET_CREATE_MASK)
BET_EROSION_KERNEL = "sphere"
BET_EROSION_KERNEL = _setting('BET_EROSION_KERNEL', BET_EROSION_KERNEL)
# Erosion radius in millimetres is configured above.


# -----------------------------------------------------------------------------
# Reversible dynamic normalization
# -----------------------------------------------------------------------------
# Magnitude: x_norm = x / S_mag, with S_mag taken from the earliest usable
# selected input echo's whole-brain robust percentile. No clipping.
MAGNITUDE_SCALE_PERCENTILE = 99.0
MAGNITUDE_SCALE_PERCENTILE = _setting('MAGNITUDE_SCALE_PERCENTILE', MAGNITUDE_SCALE_PERCENTILE)
MAGNITUDE_SATURATED_SCALE_PERCENTILE = 97.0
MAGNITUDE_SATURATED_SCALE_PERCENTILE = _setting('MAGNITUDE_SATURATED_SCALE_PERCENTILE', MAGNITUDE_SATURATED_SCALE_PERCENTILE)
MAGNITUDE_NEAR_CEILING_FRACTION = 0.001
MAGNITUDE_NEAR_CEILING_FRACTION = _setting('MAGNITUDE_NEAR_CEILING_FRACTION', MAGNITUDE_NEAR_CEILING_FRACTION)
MAGNITUDE_CEILING_CANDIDATES = (4095.0, 8191.0, 16383.0, 32767.0)
MAGNITUDE_CEILING_CANDIDATES = _setting('MAGNITUDE_CEILING_CANDIDATES', MAGNITUDE_CEILING_CANDIDATES)
MAGNITUDE_CEILING_TOLERANCE = 5.0
MAGNITUDE_CEILING_TOLERANCE = _setting('MAGNITUDE_CEILING_TOLERANCE', MAGNITUDE_CEILING_TOLERANCE)

# Phase: phi_norm = phi / S_phase. S_phase is derived from the whole-brain
# robust |phi|/TE rate of the selected input echoes and referenced to 30 ms.
PHASE_RATE_PERCENTILE = 99.0
PHASE_RATE_PERCENTILE = _setting('PHASE_RATE_PERCENTILE', PHASE_RATE_PERCENTILE)
PHASE_SCALE_REFERENCE_TE_MS = 30.0
PHASE_SCALE_REFERENCE_TE_MS = _setting('PHASE_SCALE_REFERENCE_TE_MS', PHASE_SCALE_REFERENCE_TE_MS)
NORMALIZATION_EPSILON = 1e-6
NORMALIZATION_EPSILON = _setting('NORMALIZATION_EPSILON', NORMALIZATION_EPSILON)

# Single-echo magnitude physics prior used only for the deterministic baseline.
# The residual network is explicitly allowed to correct it spatially.
MAG_SINGLE_ECHO_R2STAR_PRIOR_PER_MS = 0.040
MAG_SINGLE_ECHO_R2STAR_PRIOR_PER_MS = _setting('MAG_SINGLE_ECHO_R2STAR_PRIOR_PER_MS', MAG_SINGLE_ECHO_R2STAR_PRIOR_PER_MS)
MAG_R2STAR_MAX_PER_MS = 0.25
MAG_R2STAR_MAX_PER_MS = _setting('MAG_R2STAR_MAX_PER_MS', MAG_R2STAR_MAX_PER_MS)

# -----------------------------------------------------------------------------
# Patient split and chunks
REBUILD_PATIENT_SPLIT = _setting("REBUILD_PATIENT_SPLIT", False)
TRAINING_SPLIT_PERCENT = _setting("TRAINING_SPLIT_PERCENT", 70.0)
EVALUATION_SPLIT_PERCENT = _setting("EVALUATION_SPLIT_PERCENT", 15.0)
TEST_SPLIT_PERCENT = _setting("TEST_SPLIT_PERCENT", 15.0)
# -----------------------------------------------------------------------------
TRAIN_EVAL_SPLIT_SEED = 42
TRAIN_EVAL_SPLIT_SEED = _setting('TRAIN_EVAL_SPLIT_SEED', TRAIN_EVAL_SPLIT_SEED)
TRAIN_EVAL_SPLIT_MANIFEST_XLSX = REFERENCE_ROOT / "filelist_traintestpartitions.xlsx"
CREATE_NEW_SPLIT_IF_MANIFEST_MISSING = True
CREATE_NEW_SPLIT_IF_MANIFEST_MISSING = _setting('CREATE_NEW_SPLIT_IF_MANIFEST_MISSING', CREATE_NEW_SPLIT_IF_MANIFEST_MISSING)

CHUNK_GRID = (5, 5, 3)
CHUNK_GRID = _setting('CHUNK_GRID', CHUNK_GRID)
CHUNK_OVERLAP_VOXELS = 20
CHUNK_OVERLAP_VOXELS = _setting('CHUNK_OVERLAP_VOXELS', CHUNK_OVERLAP_VOXELS)
WRITE_CHUNK_NIFTI = False
WRITE_CHUNK_NIFTI = _setting('WRITE_CHUNK_NIFTI', WRITE_CHUNK_NIFTI)
WRITE_CHUNK_HDF5 = True
WRITE_CHUNK_HDF5 = _setting('WRITE_CHUNK_HDF5', WRITE_CHUNK_HDF5)
CEASET_HDF5_COMPRESSION = None  # future chunk creation: favor training throughput over gzip size
CEASET_HDF5_COMPRESSION = _setting('CEASET_HDF5_COMPRESSION', CEASET_HDF5_COMPRESSION)
OVERWRITE_CHUNKS = False
OVERWRITE_CHUNKS = _setting('OVERWRITE_CHUNKS', OVERWRITE_CHUNKS)

CEASET_ROOT = REFERENCE_ROOT
CEASET_STATS_CSV = CEASET_ROOT / "normalization_statistics.csv"
CEASET_CHUNK_ROOT = _cache_folder(CEASET_ROOT, "training_chunks", "09_chunks_553")
CEASET_CHUNK_ROOT = _setting('CEASET_CHUNK_ROOT', CEASET_CHUNK_ROOT)
CEASET_CHUNK_MANIFEST_CSV = CEASET_CHUNK_ROOT / "ceaset_chunk_manifest.csv"

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
TRAINING_ROOT = PROJECT_ROOT / "training" / PIPELINE_PROFILE
TRAINING_MODEL_ROOT = PROJECT_ROOT / "models" / PIPELINE_PROFILE
TRAINING_EVALUATION_ROOT = PROJECT_ROOT / "evaluation" / PIPELINE_PROFILE
ARCHITECTURE_OUTPUT_ROOT = PROJECT_ROOT / "architecture" / PIPELINE_PROFILE
SYNTHETIC_OUTPUT_ROOT = PROJECT_ROOT / "synthetic" / PIPELINE_PROFILE
INFERENCE_OUTPUT_ROOT = PROJECT_ROOT / "inference" / PIPELINE_PROFILE
RADIMAGENET_RESNET50_WEIGHTS = PROJECT_ROOT / "models" / "RadImageNet-ResNet50_notop.h5"

TRAINING_MODALITIES = ("magnitude", "phase")
TRAINING_MODALITIES = _setting('TRAINING_MODALITIES', TRAINING_MODALITIES)
TRAINING_EPOCHS = 40
TRAINING_EPOCHS = _setting('TRAINING_EPOCHS', TRAINING_EPOCHS)

# Memory-safe CEASET batching.
#
# IMPORTANT: the shared 3-D echo encoder executes BEFORE masked set fusion.
# Therefore padding every sample to four echo slots does NOT save encoder work:
# every padded slot still creates a full 3-D encoder activation stream.
#
# The historical epochs 1-26 used batch size 1 with a genuinely dynamic K=1-4
# echo axis. We retain that representation here so only real input echoes are
# encoded. This is substantially more memory efficient for this architecture.
TRAINING_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = _setting('TRAINING_BATCH_SIZE', TRAINING_BATCH_SIZE)
TRAINING_EVAL_BATCH_SIZE = 1
TRAINING_EVAL_BATCH_SIZE = _setting('TRAINING_EVAL_BATCH_SIZE', TRAINING_EVAL_BATCH_SIZE)
TRAINING_SAMPLES_PER_CHUNK_PER_EPOCH = 1
TRAINING_SAMPLES_PER_CHUNK_PER_EPOCH = _setting('TRAINING_SAMPLES_PER_CHUNK_PER_EPOCH', TRAINING_SAMPLES_PER_CHUNK_PER_EPOCH)
TRAINING_RANDOM_CHUNK_SAMPLE = None
TRAINING_RANDOM_CHUNK_SAMPLE = _setting('TRAINING_RANDOM_CHUNK_SAMPLE', TRAINING_RANDOM_CHUNK_SAMPLE)
TRAINING_SEED = 42
TRAINING_SEED = _setting('TRAINING_SEED', TRAINING_SEED)
TRAINING_GENERATOR_LOGICAL_GPU_INDEX = 0
TRAINING_GENERATOR_LOGICAL_GPU_INDEX = _setting('TRAINING_GENERATOR_LOGICAL_GPU_INDEX', TRAINING_GENERATOR_LOGICAL_GPU_INDEX)
TRAINING_CRITIC_LOGICAL_GPU_INDEX = 1
TRAINING_CRITIC_LOGICAL_GPU_INDEX = _setting('TRAINING_CRITIC_LOGICAL_GPU_INDEX', TRAINING_CRITIC_LOGICAL_GPU_INDEX)

# Parallel host-side HDF5/NumPy preparation. Each worker opens its own HDF5
# files; tf.data overlaps this work with GPU training.
TRAINING_IO_PARALLEL_CALLS = 1
TRAINING_IO_PARALLEL_CALLS = _setting('TRAINING_IO_PARALLEL_CALLS', TRAINING_IO_PARALLEL_CALLS)
TRAINING_IO_PRIVATE_THREADPOOL_SIZE = 2
TRAINING_IO_PRIVATE_THREADPOOL_SIZE = _setting('TRAINING_IO_PRIVATE_THREADPOOL_SIZE', TRAINING_IO_PRIVATE_THREADPOOL_SIZE)

# Fast training I/O cache.
# Existing CEASET HDF5 chunks are gzip-compressed; repeated h5py+gzip decoding
# can completely starve the GPU. The fast cache stores the same float32 chunks
# as uncompressed .npy files and is used only for training/evaluation I/O.
CEASET_FAST_NPY_CACHE_ROOT = _cache_folder(CEASET_ROOT, "training_cache", "10_training_cache_npy")
CEASET_FAST_NPY_CACHE_ROOT = _setting('CEASET_FAST_NPY_CACHE_ROOT', CEASET_FAST_NPY_CACHE_ROOT)
CEASET_FAST_NPY_MANIFEST_CSV = CEASET_FAST_NPY_CACHE_ROOT / "ceaset_chunk_manifest_fast_npy.csv"
TRAINING_USE_FAST_NPY_CACHE = True
TRAINING_USE_FAST_NPY_CACHE = _setting('TRAINING_USE_FAST_NPY_CACHE', TRAINING_USE_FAST_NPY_CACHE)
TRAINING_REQUIRE_FAST_NPY_CACHE = True
TRAINING_REQUIRE_FAST_NPY_CACHE = _setting('TRAINING_REQUIRE_FAST_NPY_CACHE', TRAINING_REQUIRE_FAST_NPY_CACHE)
FAST_NPY_CACHE_MODALITIES = ("magnitude","phase")   # add "phase" before phase training
FAST_NPY_CACHE_MODALITIES = _setting('FAST_NPY_CACHE_MODALITIES', FAST_NPY_CACHE_MODALITIES)
FAST_NPY_CACHE_OVERWRITE = False
FAST_NPY_CACHE_OVERWRITE = _setting('FAST_NPY_CACHE_OVERWRITE', FAST_NPY_CACHE_OVERWRITE)

# Fixed small prefetch avoids unbounded AUTOTUNE queue growth while debugging
# native/kernel crashes. The fast .npy cache makes aggressive prefetch unnecessary.
TRAINING_PREFETCH_BATCHES = 1
TRAINING_PREFETCH_BATCHES = _setting('TRAINING_PREFETCH_BATCHES', TRAINING_PREFETCH_BATCHES)
TRAINING_DROP_REMAINDER = False
TRAINING_DROP_REMAINDER = _setting('TRAINING_DROP_REMAINDER', TRAINING_DROP_REMAINDER)

# Fail early rather than silently training on CPU.
TRAINING_REQUIRE_GPU = TRAINING_DEVICE_MODE != "cpu"
TRAINING_ENABLE_MIXED_PRECISION = False
TRAINING_ENABLE_MIXED_PRECISION = _setting('TRAINING_ENABLE_MIXED_PRECISION', TRAINING_ENABLE_MIXED_PRECISION)

# Runtime / logging. The old eager loop printed only once per multi-hour epoch.
TRAINING_COMPILE_STEPS = False
TRAINING_COMPILE_STEPS = _setting('TRAINING_COMPILE_STEPS', TRAINING_COMPILE_STEPS)
# First validate the simplified WGAN-GP path in eager mode. This avoids
# hiding a native crash inside the first tf.function trace/compile. After a full
# epoch is stable, this may be switched back to True for speed.
TRAINING_SYNC_EXECUTION_DEBUG = True
TRAINING_SYNC_EXECUTION_DEBUG = _setting('TRAINING_SYNC_EXECUTION_DEBUG', TRAINING_SYNC_EXECUTION_DEBUG)
TRAINING_REQUIRE_CRASH_PROBE_BEFORE_FULL_RUN = False
TRAINING_REQUIRE_CRASH_PROBE_BEFORE_FULL_RUN = _setting('TRAINING_REQUIRE_CRASH_PROBE_BEFORE_FULL_RUN', TRAINING_REQUIRE_CRASH_PROBE_BEFORE_FULL_RUN)

# Production data path.
#
# The staged crash probe proved that direct NumPy -> TensorFlow -> GPU training
# is stable, while the full tf.data iterator path repeatedly killed the Python
# kernel before batch 1. Production training therefore uses the same direct
# path as the successful probe and does NOT use tf.data/tf.numpy_function.
TRAINING_USE_DIRECT_PYTHON_LOOP = True
TRAINING_USE_DIRECT_PYTHON_LOOP = _setting('TRAINING_USE_DIRECT_PYTHON_LOOP', TRAINING_USE_DIRECT_PYTHON_LOOP)
TRAINING_LOG_EVERY_BATCHES = 5
TRAINING_LOG_EVERY_BATCHES = _setting('TRAINING_LOG_EVERY_BATCHES', TRAINING_LOG_EVERY_BATCHES)
TRAINING_LOG_EVERY_MINUTES = 10.0
TRAINING_LOG_EVERY_MINUTES = _setting('TRAINING_LOG_EVERY_MINUTES', TRAINING_LOG_EVERY_MINUTES)
FIXED_EVAL_SEED = 424242
FIXED_EVAL_SEED = _setting('FIXED_EVAL_SEED', FIXED_EVAL_SEED)
FIXED_EVAL_SHUFFLE = False
FIXED_EVAL_SHUFFLE = _setting('FIXED_EVAL_SHUFFLE', FIXED_EVAL_SHUFFLE)

# Fixed validation enables meaningful checkpoint selection and early stopping.
ENABLE_EARLY_STOPPING = True
ENABLE_EARLY_STOPPING = _setting('ENABLE_EARLY_STOPPING', ENABLE_EARLY_STOPPING)
EARLY_STOPPING_START_EPOCH = 8
EARLY_STOPPING_START_EPOCH = _setting('EARLY_STOPPING_START_EPOCH', EARLY_STOPPING_START_EPOCH)
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_PATIENCE = _setting('EARLY_STOPPING_PATIENCE', EARLY_STOPPING_PATIENCE)
EARLY_STOPPING_MIN_DELTA = 2e-4
EARLY_STOPPING_MIN_DELTA = _setting('EARLY_STOPPING_MIN_DELTA', EARLY_STOPPING_MIN_DELTA)

# -------------------------------------------------------------------------
# Dual-GPU WGAN-GP adversarial training
# -------------------------------------------------------------------------
# Generator and generator backward pass live on logical GPU:0.
# Critic, real/fake scoring, and the second-order gradient penalty live on
# logical GPU:1. This prevents the ~18.4 GiB generator graph and the GP graph
# from competing for the same 48 GiB allocator.
TRAINING_GAN_MODE = "wgan_gp_" + TRAINING_DEVICE_MODE

WGAN_GP_LAMBDA = 10.0
WGAN_GP_LAMBDA = _setting('WGAN_GP_LAMBDA', WGAN_GP_LAMBDA)
WGAN_GP_EPSILON = 1e-12
WGAN_GP_EPSILON = _setting('WGAN_GP_EPSILON', WGAN_GP_EPSILON)
WGAN_GP_FULL_RESOLUTION = True
WGAN_GP_FULL_RESOLUTION = _setting('WGAN_GP_FULL_RESOLUTION', WGAN_GP_FULL_RESOLUTION)
# Emergency fallback only. Keep 1 for exact full-resolution WGAN-GP.
# Setting 2 computes GP on 2x average-pooled volumes while real/fake critic
# scores remain full-resolution. This reduces GP memory substantially but is an
# approximation, so do not enable it unless GPU:1 actually runs out of memory.
WGAN_GP_DOWNSAMPLE_FACTOR = 1
WGAN_GP_DOWNSAMPLE_FACTOR = _setting('WGAN_GP_DOWNSAMPLE_FACTOR', WGAN_GP_DOWNSAMPLE_FACTOR)

# Modality-specific critic schedules.
# Magnitude previously tolerated/benefited from two critic steps; phase showed
# a much stronger discriminator and therefore uses one critic step.
CRITIC_UPDATES_BY_MODALITY = {
    "magnitude": 2,
    "phase": 1,
}
CRITIC_UPDATES_BY_MODALITY = _setting('CRITIC_UPDATES_BY_MODALITY', CRITIC_UPDATES_BY_MODALITY)
CRITIC_WARMUP_EPOCHS = 0
CRITIC_WARMUP_EPOCHS = _setting('CRITIC_WARMUP_EPOCHS', CRITIC_WARMUP_EPOCHS)

# Never restore an optimizer/critic state from the older objective.  V7 changes
# the generator conditioning and the physics loss, so it uses a new state root.
TRAINING_RESET_CRITIC_ON_LEGACY_RESUME = True
TRAINING_RESET_CRITIC_ON_LEGACY_RESUME = _setting('TRAINING_RESET_CRITIC_ON_LEGACY_RESUME', TRAINING_RESET_CRITIC_ON_LEGACY_RESUME)
TRAINING_VARIANT_TAG = "abrim9_wgangp_structphys_rawte_v8"

TRAINING_ENABLE_PERCEPTUAL_LOSS = False  # enable only with local RadImageNet weights
TRAINING_ENABLE_PERCEPTUAL_LOSS = _setting('TRAINING_ENABLE_PERCEPTUAL_LOSS', TRAINING_ENABLE_PERCEPTUAL_LOSS)
TRAINING_ENABLE_EDGE_LOSS = True
TRAINING_ENABLE_EDGE_LOSS = _setting('TRAINING_ENABLE_EDGE_LOSS', TRAINING_ENABLE_EDGE_LOSS)

# The legacy flag name is kept as an alias so older cells/scripts do not break.
# In V7 this is NOT a voxelwise residual-MAE penalty anymore.
TRAINING_ENABLE_COARSE_PHYSICS_LOSS = True
TRAINING_ENABLE_COARSE_PHYSICS_LOSS = _setting('TRAINING_ENABLE_COARSE_PHYSICS_LOSS', TRAINING_ENABLE_COARSE_PHYSICS_LOSS)
TRAINING_ENABLE_PHYSICS_RESIDUAL_LOSS = TRAINING_ENABLE_COARSE_PHYSICS_LOSS

# -------------------------------------------------------------------------
# Explicit physical TE path in addition to the learned TE embedding.
# -------------------------------------------------------------------------
# The learned 48-D embedding stays in place, but every FiLM block also receives
# raw normalized TE through a separate low-order (linear + quadratic) affine
# path.  The new path is zero-initialized so legacy generator checkpoints can
# be migrated without changing their initial prediction.
RAW_TE_FILM_ENABLED = True
RAW_TE_FILM_ENABLED = _setting('RAW_TE_FILM_ENABLED', RAW_TE_FILM_ENABLED)
RAW_TE_PHYSICAL_SCALE_MS = 40.0
RAW_TE_PHYSICAL_SCALE_MS = _setting('RAW_TE_PHYSICAL_SCALE_MS', RAW_TE_PHYSICAL_SCALE_MS)
RAW_TE_FILM_POLYNOMIAL_ORDER = 2
RAW_TE_FILM_POLYNOMIAL_ORDER = _setting('RAW_TE_FILM_POLYNOMIAL_ORDER', RAW_TE_FILM_POLYNOMIAL_ORDER)

# Keep the echo axis genuinely dynamic at K=1-4.
TRAINING_DYNAMIC_ECHO_AXIS = True
TRAINING_DYNAMIC_ECHO_AXIS = _setting('TRAINING_DYNAMIC_ECHO_AXIS', TRAINING_DYNAMIC_ECHO_AXIS)

# -------------------------------------------------------------------------
# Coarse/structural physics prior.
# -------------------------------------------------------------------------
# The physics model is trusted for broad tissue/TE behaviour, NOT for vessels,
# lesions, rims, or other fine susceptibility structures.  Both prediction and
# physics baseline are first blurred in-plane only (axial SWI anisotropy), then:
#   1) masked Huber anchors quantitative coarse intensity/phase,
#   2) masked LNCC checks local structural agreement over a 7x7x1 neighborhood.
PHYSICS_GAUSSIAN_SIGMA_VOXELS = (1.5, 1.5, 0.0)
PHYSICS_GAUSSIAN_SIGMA_VOXELS = _setting('PHYSICS_GAUSSIAN_SIGMA_VOXELS', PHYSICS_GAUSSIAN_SIGMA_VOXELS)
PHYSICS_LNCC_WINDOW_VOXELS = (7, 7, 1)
PHYSICS_LNCC_WINDOW_VOXELS = _setting('PHYSICS_LNCC_WINDOW_VOXELS', PHYSICS_LNCC_WINDOW_VOXELS)
PHYSICS_HUBER_DELTA = 0.05
PHYSICS_HUBER_DELTA = _setting('PHYSICS_HUBER_DELTA', PHYSICS_HUBER_DELTA)
PHYSICS_INTENSITY_FRACTION = 0.95
PHYSICS_INTENSITY_FRACTION = _setting('PHYSICS_INTENSITY_FRACTION', PHYSICS_INTENSITY_FRACTION)
PHYSICS_STRUCTURE_FRACTION = 0.05
PHYSICS_STRUCTURE_FRACTION = _setting('PHYSICS_STRUCTURE_FRACTION', PHYSICS_STRUCTURE_FRACTION)
PHYSICS_LNCC_MIN_MASK_FRACTION = 0.25
PHYSICS_LNCC_MIN_MASK_FRACTION = _setting('PHYSICS_LNCC_MIN_MASK_FRACTION', PHYSICS_LNCC_MIN_MASK_FRACTION)
PHYSICS_LNCC_MIN_VARIANCE = 1e-6
PHYSICS_LNCC_MIN_VARIANCE = _setting('PHYSICS_LNCC_MIN_VARIANCE', PHYSICS_LNCC_MIN_VARIANCE)

# Phase baseline: CLEAR-SWI-style local-frequency averaging phi/TE.
# During training, corresponding magnitude chunks are used as an SNR proxy
# (magnitude^2 weights), with robust downweighting of frequency outliers.
PHASE_PHYSICS_USE_MAGNITUDE_SNR_WEIGHTS = True
PHASE_PHYSICS_USE_MAGNITUDE_SNR_WEIGHTS = _setting('PHASE_PHYSICS_USE_MAGNITUDE_SNR_WEIGHTS', PHASE_PHYSICS_USE_MAGNITUDE_SNR_WEIGHTS)
PHASE_PHYSICS_ROBUST_FREQUENCY_WEIGHTING = True
PHASE_PHYSICS_ROBUST_FREQUENCY_WEIGHTING = _setting('PHASE_PHYSICS_ROBUST_FREQUENCY_WEIGHTING', PHASE_PHYSICS_ROBUST_FREQUENCY_WEIGHTING)
PHASE_PHYSICS_ROBUST_C = 2.5
PHASE_PHYSICS_ROBUST_C = _setting('PHASE_PHYSICS_ROBUST_C', PHASE_PHYSICS_ROBUST_C)

# -------------------------------------------------------------------------
# Continuous-TE auxiliary exposure.
# -------------------------------------------------------------------------
# Main supervised targets remain the genuinely measured echoes.  Every Nth
# training sample additionally receives ONE separate generator update at a
# random TE in 4..40 ms.  There is deliberately NO pseudo target, GAN target,
# edge, or perceptual target for this update: only the new COARSE physics loss
# is used.  This teaches broad interpolation/extrapolation behaviour without
# forcing the physics model's fine-scale errors into synthetic TEs.
CONTINUOUS_AUXILIARY_TRAINING = True
CONTINUOUS_AUXILIARY_TRAINING = _setting('CONTINUOUS_AUXILIARY_TRAINING', CONTINUOUS_AUXILIARY_TRAINING)
CONTINUOUS_TARGET_TRAIN_RANGE_MS = (4.0, 40.0)
CONTINUOUS_TARGET_TRAIN_RANGE_MS = _setting('CONTINUOUS_TARGET_TRAIN_RANGE_MS', CONTINUOUS_TARGET_TRAIN_RANGE_MS)
CONTINUOUS_AUXILIARY_EVERY_N_BATCHES = 4
CONTINUOUS_AUXILIARY_EVERY_N_BATCHES = _setting('CONTINUOUS_AUXILIARY_EVERY_N_BATCHES', CONTINUOUS_AUXILIARY_EVERY_N_BATCHES)
CONTINUOUS_AUXILIARY_WEIGHT_BY_MODALITY = {
    "magnitude": 0.10,
    "phase": 0.05,
}
CONTINUOUS_AUXILIARY_WEIGHT_BY_MODALITY = _setting('CONTINUOUS_AUXILIARY_WEIGHT_BY_MODALITY', CONTINUOUS_AUXILIARY_WEIGHT_BY_MODALITY)
SUPERVISED_TE_RANGE_MS = (
    min(DEFAULT_ECHO_TIMES_MS),
    max(DEFAULT_ECHO_TIMES_MS),
)
TRAINING_SAVE_EVERY_EPOCHS = 1
TRAINING_SAVE_EVERY_EPOCHS = _setting('TRAINING_SAVE_EVERY_EPOCHS', TRAINING_SAVE_EVERY_EPOCHS)
TRAINING_RESUME = True
TRAINING_RESUME = _setting('TRAINING_RESUME', TRAINING_RESUME)

# Spatial augmentation only. Intensity jitter is intentionally avoided because
# quantitative inter-echo intensity/phase evolution is the learning target.
TRAINING_RANDOM_FLIPS = True
TRAINING_RANDOM_FLIPS = _setting('TRAINING_RANDOM_FLIPS', TRAINING_RANDOM_FLIPS)

# WGAN-GP and generator-loss weights. These are starting values, not claimed optima.
TRAINING_HYPERPARAMETERS = {
    "magnitude": {
        "generator_lr": 3e-5,
        "critic_lr": 1e-4,
        "adv_weight": 0.08,
        "rec_weight": 5.6,
        "edge_weight": 0.1,
        "perc_weight": 0.06,
        "physics_residual_weight": 0.35,
    },
    "phase": {
        "generator_lr": 3e-5,
        "critic_lr": 1e-4,
        "adv_weight": 0.06,
        "rec_weight": 5.6,
        "edge_weight": 0.1,
        "perc_weight": 0.03,
        "physics_residual_weight": 0.35,
    },
}
TRAINING_HYPERPARAMETERS = _setting('TRAINING_HYPERPARAMETERS', TRAINING_HYPERPARAMETERS)

MODEL_FILTERS = (16, 24, 32, 48)
MODEL_FILTERS = _setting('MODEL_FILTERS', MODEL_FILTERS)
MODEL_TE_EMBED_DIM = 48
MODEL_TE_EMBED_DIM = _setting('MODEL_TE_EMBED_DIM', MODEL_TE_EMBED_DIM)
MODEL_L2 = 5e-5
MODEL_L2 = _setting('MODEL_L2', MODEL_L2)
MODEL_LEAKY_RELU = 0.2
MODEL_LEAKY_RELU = _setting('MODEL_LEAKY_RELU', MODEL_LEAKY_RELU)

# Evaluation
EVAL_CONFIGS_PER_CHUNK = 2
EVAL_CONFIGS_PER_CHUNK = _setting('EVAL_CONFIGS_PER_CHUNK', EVAL_CONFIGS_PER_CHUNK)
EVAL_MAX_CHUNKS = None
EVAL_MAX_CHUNKS = _setting('EVAL_MAX_CHUNKS', EVAL_MAX_CHUNKS)


CLEARSWI_PATH_PREFIX = Path.home() / "CLEARSWI"
BASH_ALIASES = Path.home() / ".bash_aliases"
FSL_SETUP_CANDIDATES = (
    Path(os.environ.get("FSLDIR", "/usr/local/fsl")) / "etc/fslconf/fsl.sh",
    Path("/etc/fsl/fsl.sh"), Path.home() / "fsl/etc/fslconf/fsl.sh",
)

def assert_direct_loop_runtime_config():
    expected = {"TRAINING_BATCH_SIZE": 1, "TRAINING_EVAL_BATCH_SIZE": 1,
                "TRAINING_DYNAMIC_ECHO_AXIS": True, "TRAINING_USE_DIRECT_PYTHON_LOOP": True}
    for name, value in expected.items():
        if globals()[name] != value:
            raise ValueError(f"This pipeline requires {name}={value}")

print(CEASET_PIPELINE_REVISION, "| action:", PIPELINE_ACTION)
print("Input:", SOURCE_ROOT, "\nPreprocessing:", REFERENCE_ROOT, "\nModels:", TRAINING_MODEL_ROOT)

# Validate before launching external preprocessing or allocating GPU memory.
def validate_settings():
    assert_direct_loop_runtime_config()
    import math
    fractions = (TRAINING_SPLIT_PERCENT, EVALUATION_SPLIT_PERCENT, TEST_SPLIT_PERCENT)
    if fractions[0] <= 0 or fractions[1] <= 0 or fractions[2] < 0 or not math.isclose(sum(fractions), 100., abs_tol=1e-8, rel_tol=0):
        raise ValueError('Partition percentages must total 100; training/evaluation must be positive and test nonnegative')
    if 'TRAIN_EVAL_SPLIT_FRACTION' in _OVERRIDES:
        raise ValueError('TRAIN_EVAL_SPLIT_FRACTION is retired; use TRAINING_SPLIT_PERCENT, EVALUATION_SPLIT_PERCENT and TEST_SPLIT_PERCENT')
    if not PROJECT_ROOT.is_dir(): raise ValueError('Project root does not exist')
    if not TRAINING_MODALITIES_TO_RUN or set(TRAINING_MODALITIES_TO_RUN) - {'magnitude', 'phase'}:
        raise ValueError('Select magnitude and/or phase')
    if len(MODEL_FILTERS) != 4 or any(type(x) is not int or x < 1 for x in MODEL_FILTERS):
        raise ValueError('MODEL_FILTERS requires four positive integers')
    for key in ('MODEL_TE_EMBED_DIM','TRAINING_EPOCHS','TRAINING_SAVE_EVERY_EPOCHS','EARLY_STOPPING_PATIENCE','CONTINUOUS_AUXILIARY_EVERY_N_BATCHES'):
        if type(globals()[key]) is not int or globals()[key] < 1: raise ValueError(f'{key} must be a positive integer')
    if len(CHUNK_GRID)!=3 or any(type(x) is not int or x < 1 for x in CHUNK_GRID): raise ValueError('Invalid chunk grid')
    if CHUNK_OVERLAP_VOXELS < 0: raise ValueError('Chunk overlap must be nonnegative')
    if not 0 < BET_FRACTION < 1: raise ValueError('Invalid BET or split fraction')
    if not 1 <= MIN_INPUT_ECHOES <= MAX_INPUT_ECHOES <= 8: raise ValueError('Invalid echo count range')
    if MODEL_L2 < 0 or not 0 <= MODEL_LEAKY_RELU <= 1: raise ValueError('Invalid model regularization/slope')
    if set(TRAINING_HYPERPARAMETERS) != {'magnitude','phase'}: raise ValueError('Hyperparameters require magnitude and phase')
    for modality in ('magnitude','phase'):
        expected = {'generator_lr','critic_lr','adv_weight','rec_weight','edge_weight','perc_weight','physics_residual_weight'}
        if set(TRAINING_HYPERPARAMETERS[modality]) != expected: raise ValueError(f'Invalid hyperparameter keys for {modality}')
        if CRITIC_UPDATES_BY_MODALITY[modality] < 1: raise ValueError('Critic updates must be positive')
        for key,value in TRAINING_HYPERPARAMETERS[modality].items():
            if not isinstance(value,(int,float)) or not __import__('math').isfinite(value) or value < 0 or (key.endswith('_lr') and value == 0):
                raise ValueError(f'Invalid {modality} {key}')
    if not __import__('math').isfinite(WGAN_GP_LAMBDA) or WGAN_GP_LAMBDA < 0: raise ValueError('Invalid gradient penalty')
    probs = INPUT_COUNT_PROBABILITIES
    if set(probs) != set(range(MIN_INPUT_ECHOES,MAX_INPUT_ECHOES+1)) or any(v < 0 for v in probs.values()) or abs(sum(probs.values())-1)>1e-6:
        raise ValueError('Input-count probabilities must cover the selected echo counts and sum to one')

TRAINING_PREVIEW_ENABLED = _setting('TRAINING_PREVIEW_ENABLED', True)
TRAINING_PREVIEW_INTERVAL_SECONDS = _setting('TRAINING_PREVIEW_INTERVAL_SECONDS', 60.0)
if TRAINING_PREVIEW_INTERVAL_SECONDS < 10:
    raise ValueError('Training preview interval must be at least 10 seconds')
RUN_INSTANCE = _setting('RUN_INSTANCE', '')
if not isinstance(RUN_INSTANCE, str) or (RUN_INSTANCE and not __import__('re').fullmatch(r'[A-Za-z0-9_-]{1,64}',RUN_INSTANCE)):
    raise ValueError('RUN_INSTANCE must be at most 64 letters, digits, underscores or hyphens')
validate_settings()
RESOLVED_CONFIGURATION = {k: v for k,v in list(globals().items()) if k.isupper() and not k.startswith('_')}
# The epoch limit can be extended on resume; all architecture/loss/data settings remain isolated.
_identity = {k:v for k,v in RESOLVED_CONFIGURATION.items() if k not in {'PIPELINE_ACTION','TRAINING_EPOCHS','TRAINING_RESUME','RUN_INSTANCE','TRAINING_PREVIEW_ENABLED','TRAINING_PREVIEW_INTERVAL_SECONDS','REBUILD_PATIENT_SPLIT'}}
_data_fingerprints = {}
for _p in (REFERENCE_ROOT / 'derived_data_signature.json', TRAIN_EVAL_SPLIT_MANIFEST_XLSX, CEASET_STATS_CSV, CEASET_CHUNK_MANIFEST_CSV, CEASET_FAST_NPY_MANIFEST_CSV):
    if _p.is_file():
        _data_fingerprints[str(_p)] = hashlib.sha256(_p.read_bytes()).hexdigest()
_identity['integration_revision'] = '2026-11-11_split_v1'
_identity['data_fingerprints'] = _data_fingerprints
RESOLVED_CONFIGURATION['data_fingerprints'] = _data_fingerprints
CONFIG_HASH = hashlib.sha256(json.dumps(_identity, sort_keys=True, default=str).encode()).hexdigest()[:12]
RUN_NAME = 'wgangp_f' + '-'.join(map(str,MODEL_FILTERS)) + f'_te{MODEL_TE_EMBED_DIM}_{CONFIG_HASH}'
# Fresh runs get a unique suffix. Resume runs retain the deterministic configuration name.
if RUN_INSTANCE:
    RUN_NAME += '_' + RUN_INSTANCE
elif not TRAINING_RESUME:
    RUN_INSTANCE = __import__('uuid').uuid4().hex[:10]
    RUN_NAME += '_' + RUN_INSTANCE
if PIPELINE_ACTION == 'train' and not TRAINING_RESUME:
    for _modality in TRAINING_MODALITIES_TO_RUN:
        _root = TRAINING_MODEL_ROOT / _modality
        if any(_root.glob(f'*{RUN_NAME}*.weights.h5')) or (_root / f'training_state_{RUN_NAME}').exists():
            raise ValueError('This run instance already has checkpoints. Enable resume or choose a new/empty RUN_INSTANCE.')
TRAINING_VARIANT_TAG = RUN_NAME
RESOLVED_CONFIGURATION.update(RUN_INSTANCE=RUN_INSTANCE, RUN_NAME=RUN_NAME, CONFIG_HASH=CONFIG_HASH, TRAINING_VARIANT_TAG=TRAINING_VARIANT_TAG)
__all__ = [k for k in globals() if not k.startswith('__')]
