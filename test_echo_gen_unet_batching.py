"""Synthetic geometry regression tests; no MRI dataset or TensorFlow required."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import echo_gen_preprocessing as p


def geometry(*, mode="grid_patches", crop=True, margin=2, multiple=8,
             grid=(2, 2, 2), overlap=2):
    return SimpleNamespace(
        unet_batching_mode=mode,
        unet_brain_crop_enabled=crop,
        unet_brain_crop_margin_voxels=margin,
        unet_pad_multiple=multiple,
        chunk_grid=grid,
        chunk_overlap_voxels=overlap,
    )


class UnetBatchingTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((31, 39, 27), dtype=np.float32)
        self.mask[5:19, 8:24, 4:18] = 1.0

    def test_mask_bbox_clamps_and_preserves_half_open_coordinates(self):
        self.assertEqual(
            p.brain_mask_bounding_box(self.mask, margin=2),
            ((3, 6, 2), (21, 26, 20)),
        )
        self.assertEqual(
            p.brain_mask_bounding_box(self.mask, margin=100),
            ((0, 0, 0), self.mask.shape),
        )
        self.assertEqual(
            p.brain_mask_bounding_box(self.mask, margin=0),
            ((5, 8, 4), (19, 24, 18)),
        )

    def test_empty_brain_mask_never_silently_generates_samples(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            p.brain_mask_bounding_box(np.zeros((3, 5, 4), np.float32))
        with self.assertRaises(ValueError):
            p.brain_mask_bounding_box(self.mask, margin=-1)

    def test_patch_specs_cover_crop_roi_in_source_voxel_space(self):
        cfg = geometry()
        specs = p.build_training_spatial_specs(self.mask, cfg)
        self.assertEqual(len(specs), 8)
        coverage = np.zeros_like(self.mask, dtype=np.int32)
        first, last = (3, 6, 2), (21, 26, 20)
        for spec in specs:
            self.assertEqual(spec["roi_start"], first)
            self.assertEqual(spec["roi_end"], last)
            self.assertTrue(all(d % cfg.unet_pad_multiple == 0 for d in spec["chunk_shape"]))
            region = tuple(slice(a, b) for a, b in zip(spec["core_start"], spec["core_end"]))
            coverage[region] += 1
            # No source read may escape the ROI after cropping.
            for axis in range(3):
                self.assertGreaterEqual(spec["source_start"][axis], first[axis])
                self.assertLessEqual(spec["source_end"][axis], last[axis])
        np.testing.assert_array_equal(coverage[3:21, 6:26, 2:20], 1)
        self.assertEqual(int(coverage.sum()), (21-3)*(26-6)*(20-2))

    def test_identical_echo_and_target_coordinate_slices_with_no_interpolation(self):
        cfg = geometry(mode="full_volume", margin=1, multiple=16)
        spec, = p.build_training_spatial_specs(self.mask, cfg)
        h, w, d = np.indices(self.mask.shape)
        raw = (h * 10000 + w * 100 + d).astype(np.float32)
        nine_echoes = np.stack([raw + 1_000_000 * i for i in range(9)], axis=-1)
        cropped = p.extract_chunk(nine_echoes, spec)
        cropped_mask = p.extract_chunk(self.mask, spec)
        self.assertEqual(cropped.shape[-1], 9)
        self.assertEqual(cropped.shape[:3], spec["chunk_shape"])
        self.assertEqual(cropped_mask.shape, spec["chunk_shape"])
        for idx in range(9):
            np.testing.assert_array_equal(
                cropped[..., idx],
                p.extract_chunk(nine_echoes[..., idx], spec),
            )
        # A known source voxel has exactly the same offset in every echo.
        source = (6, 10, 5)
        dest = tuple(
            source[a] - spec["source_start"][a] + spec["pad_before"][a]
            for a in range(3)
        )
        self.assertEqual(cropped_mask[dest], 1)
        for idx in range(9):
            self.assertEqual(cropped[(*dest, idx)], nine_echoes[(*source, idx)])

    def test_full_volume_is_single_stride_aligned_sample(self):
        for multiple in (8, 16, 32):
            spec, = p.build_training_spatial_specs(
                self.mask,
                geometry(mode="full_volume", multiple=multiple),
            )
            self.assertEqual(spec["index"], (0, 0, 0))
            self.assertTrue(all(dim % multiple == 0 for dim in spec["chunk_shape"]))
            self.assertEqual(spec["source_start"], (3, 6, 2))
            self.assertEqual(spec["source_end"], (21, 26, 20))
            self.assertTrue(np.any(p.extract_chunk(self.mask, spec)))

    def test_default_uncropped_grid_reuses_original_splitter(self):
        cfg = geometry(crop=False)
        old = p.build_chunk_specs(self.mask.shape, cfg.chunk_grid, cfg.chunk_overlap_voxels)
        new = p.build_training_spatial_specs(self.mask, cfg)
        self.assertEqual(len(old), len(new))
        for a, b in zip(old, new):
            for field in a:
                self.assertEqual(a[field], b[field])

    def test_nifti_affine_maps_padding_and_swapped_processing_axes(self):
        spec, = p.build_training_spatial_specs(
            self.mask, geometry(mode="full_volume", multiple=32))
        source_affine = np.array([
            [0.7, 0.0, 0.0, 23.0],
            [0.0, -0.9, 0.0, -11.0],
            [0.0, 0.0, 1.4, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        new_affine = p.chunk_nifti_affine(source_affine, spec)
        for voxel in ((0, 0, 0), (2, 3, 4), (10, 9, 7)):
            proc = tuple(
                spec["source_start"][axis] - spec["pad_before"][axis] + voxel[axis]
                for axis in range(3)
            )
            expected = source_affine @ np.array([proc[1], proc[0], proc[2], 1.0])
            actual = new_affine @ np.array([*voxel, 1.0])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_variable_size_cache_estimate_not_assumed_uniform(self):
        manifest = pd.DataFrame({"chunk_shape": ["8x16x24", "16x24x32"]})
        n_voxels = 8*16*24 + 16*24*32
        expected = n_voxels * (1 + 2*9) * 4 / (1024**3)
        self.assertAlmostEqual(
            p.estimate_fast_npy_cache_gib(
                manifest, ("magnitude", "phase"), p.CFG), expected)

    def test_geometry_validation_rejects_unsupported_settings(self):
        with self.assertRaisesRegex(ValueError, "UNET_BATCHING_MODE"):
            p.build_training_spatial_specs(self.mask, geometry(mode="bad"))
        with self.assertRaisesRegex(ValueError, "pad_multiple"):
            p.build_chunk_specs((16, 16, 16), (1, 1, 1), 0, pad_multiple=10)


if __name__ == "__main__":
    unittest.main()
