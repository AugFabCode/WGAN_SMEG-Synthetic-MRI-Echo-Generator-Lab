"""Lossless, signal-preserving rigid augmentation for already-prepared MRI batches.

Random right-angle rotations and integer-voxel translations are applied to
all echo channels, target, physics baseline and brain mask using identical
spatial indices. This operates on the CPU during *batch preparation*, not on
the original NIfTI or canonical HDF5/NumPy caches.

Unlike arbitrary-angle affine resampling this transform cannot blur magnitude
or interpolate wrapped phase. Input arrays remain exactly their original
spatial shape. If the entire signal support will not fit, try another rotation
or a translation-only fallback; never crop occupied voxels.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _spatial_support(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Union of every nonzero value, not only the possibly eroded brain mask."""
    spatial = tuple(int(x) for x in arrays[0].shape[-4:-1])
    support = np.zeros(spatial, dtype=bool)
    for value in arrays:
        arr = np.asarray(value)
        if arr.ndim not in (4, 5) or tuple(arr.shape[-4:-1]) != spatial:
            raise ValueError("All augmentation arrays must share [H,W,D,C] geometry")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Non-finite MRI values cannot be safely augmented")
        # [K,H,W,D,C] versus [H,W,D,C]
        present = np.any(arr != 0, axis=(0, 4) if arr.ndim == 5 else 3)
        support |= present
    return support


def _bbox(support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-open bounds enclosing every signal voxel, without materializing N×3."""
    starts, ends = [], []
    for axis in range(3):
        other_axes = tuple(i for i in range(3) if i != axis)
        occupied = np.flatnonzero(np.any(support, axis=other_axes))
        if occupied.size == 0:
            raise ValueError("Cannot augment a patch without nonzero signal")
        starts.append(int(occupied[0]))
        ends.append(int(occupied[-1]) + 1)
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def _placement(
    original_lo: np.ndarray,
    original_hi: np.ndarray,
    rotated_lo: np.ndarray,
    rotated_hi: np.ndarray,
    shape: tuple[int, int, int],
    max_translation_voxels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find an integer offset preserving the entire transformed signal bounding box."""
    source_center = (original_lo + original_hi - 1) / 2.0
    rotated_center = (rotated_lo + rotated_hi - 1) / 2.0
    # Recenter the rotated anatomy at the original anatomy's location.
    anchor = np.rint(source_center - rotated_center).astype(np.int64)
    safe_low = -rotated_lo
    safe_high = np.asarray(shape, dtype=np.int64) - rotated_hi
    choices_low = np.maximum(safe_low, anchor - max_translation_voxels)
    choices_high = np.minimum(safe_high, anchor + max_translation_voxels)
    if np.any(choices_low > choices_high):
        return None
    offsets = np.asarray([
        int(rng.integers(int(choices_low[d]), int(choices_high[d]) + 1))
        for d in range(3)
    ], dtype=np.int64)
    return offsets, offsets - anchor


def _rotate_insert(
    arr: np.ndarray,
    axis_pair: tuple[int, int] | None,
    quarter_turns: int,
    lo: np.ndarray,
    hi: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Reorient a full tensor, then copy only its guaranteed in-bounds support box."""
    arr = np.asarray(arr)
    offset = 1 if arr.ndim == 5 else 0
    rotated = (
        np.rot90(arr, quarter_turns,
                 axes=(axis_pair[0] + offset, axis_pair[1] + offset))
        if axis_pair is not None and quarter_turns else arr
    )
    src = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    dst = tuple(slice(int(lo[i] + offsets[i]), int(hi[i] + offsets[i])) for i in range(3))
    source_index = (slice(None), *src, slice(None)) if offset else (*src, slice(None))
    target_index = (slice(None), *dst, slice(None)) if offset else (*dst, slice(None))
    output = np.zeros_like(arr)
    output[target_index] = rotated[source_index]
    return np.ascontiguousarray(output)


def augment_rigid_3d(
    arrays: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    enabled: bool = False,
    max_translation_voxels: int = 4,
    rotation_probability: float = 0.75,
    rotation_planes: str = "in_plane",
) -> tuple[list[np.ndarray], dict]:
    """Return co-registered patches, with every original nonzero voxel preserved.

    Rotations are exact multiples of 90° in voxel space. By default only the
    first/second spatial axes are rotated: do not mix an anisotropic slice axis
    into a physical rotation. Use rotation_planes='all' only for isotropic voxels.

    A candidate is *only* applied if the entire nonzero support (union of all
    image inputs, target, model baseline and mask) fits within the original
    [H,W,D] shape. All values are copied exactly with no interpolation. If no
    candidate fits, the original sample is returned intact.
    """
    result = [np.asarray(a) for a in arrays]
    info = dict(
        applied=False,
        rotation_axes=None,
        quarter_turns=0,
        offset_voxels=[0, 0, 0],
        center_translation_voxels=[0, 0, 0],
        outcome="disabled",
    )
    if not enabled:
        return result, info
    if type(max_translation_voxels) is not int or max_translation_voxels < 0:
        raise ValueError("max_translation_voxels must be a nonnegative integer")
    if not np.isfinite(rotation_probability) or not 0 <= rotation_probability <= 1:
        raise ValueError("rotation_probability must be between 0 and 1")
    if rotation_planes not in ("in_plane", "all"):
        raise ValueError("rotation_planes must be in_plane or all")
    if not result:
        raise ValueError("Augmentation needs at least one MRI array")

    shape = tuple(int(x) for x in result[0].shape[-4:-1])
    support = _spatial_support(result)
    if not np.any(support):
        info["outcome"] = "empty_signal_unchanged"
        return result, info

    original_lo, original_hi = _bbox(support)
    candidates: list[tuple[tuple[int, int] | None, int]] = []
    if rng.random() < rotation_probability:
        axes = [(0, 1)] if rotation_planes == "in_plane" else [(0, 1), (0, 2), (1, 2)]
        candidates = [(axes_pair, k) for axes_pair in axes for k in (1, 2, 3)]
        rng.shuffle(candidates)
    candidates.append((None, 0))  # Translation-only or unchanged fallback.

    for plane, turns in candidates:
        rotated_support = (
            np.rot90(support, turns, axes=plane) if plane is not None else support
        )
        rotated_lo, rotated_hi = _bbox(rotated_support)
        placement = _placement(
            original_lo, original_hi, rotated_lo, rotated_hi,
            shape, max_translation_voxels, rng,
        )
        if placement is None:
            continue
        offsets, center_translation = placement
        outputs = [
            _rotate_insert(a, plane, turns, rotated_lo, rotated_hi, offsets)
            for a in result
        ]
        # Every nonzero tensor value must be represented exactly once.
        # Validation is independent of the (possibly eroded) brain-mask quality.
        output_support = _spatial_support(outputs)
        if int(output_support.sum()) != int(support.sum()):
            raise AssertionError("Rigid augmentation unexpectedly lost source voxels")
        info.update(
            applied=bool(plane is not None or np.any(offsets)),
            rotation_axes=list(plane) if plane is not None else None,
            quarter_turns=int(turns),
            offset_voxels=[int(x) for x in offsets],
            center_translation_voxels=[int(x) for x in center_translation],
            outcome=("rigid_rotation" if plane is not None
                     else "translation_only" if np.any(offsets)
                     else "identity_no_safe_motion"),
        )
        return outputs, info

    # This should not be reachable because the translation-only identity fits.
    info["outcome"] = "no_safe_placement"
    return result, info
