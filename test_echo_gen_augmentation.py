"""Tests for exact 3-D rigid MRI batch augmentation; no TensorFlow or MRI files."""
import unittest

import numpy as np

from echo_gen_augmentation import augment_rigid_3d


def make_batch(shape=(17, 13, 9)):
    """Co-registered magnitude/phase-like tensors and a binary mask."""
    mask = np.zeros((*shape, 1), np.float32)
    mask[1:6, 2:9, 1:5, 0] = 1.
    h, w, d = np.indices(shape)
    signal = (h * 1000 + w * 10 + d + 1).astype(np.float32)[..., None] * mask
    echo_stack = np.stack([signal, signal * 2, signal * 3], axis=0)
    target = signal * 7
    baseline = signal * 11
    auxiliary = signal * 13
    return [echo_stack, target, baseline, mask, auxiliary]


class SignalSafeRigidAugmentationTests(unittest.TestCase):
    def test_registered_inputs_and_target_have_exact_voxel_value_conservation(self):
        inputs = make_batch()
        for seed in range(12):
            with self.subTest(seed=seed):
                augmented, metadata = augment_rigid_3d(
                    inputs, np.random.default_rng(seed), enabled=True,
                    max_translation_voxels=5, rotation_probability=1.0,
                    rotation_planes="all",
                )
                for original, transformed in zip(inputs, augmented):
                    self.assertEqual(transformed.shape, original.shape)
                    self.assertEqual(transformed.dtype, original.dtype)
                    self.assertTrue(transformed.flags.c_contiguous)
                    np.testing.assert_array_equal(
                        np.sort(original.reshape(-1)), np.sort(transformed.reshape(-1))
                    )
                np.testing.assert_array_equal(augmented[0][0] * 7, augmented[1])
                np.testing.assert_array_equal(augmented[0][1], augmented[0][0] * 2)
                np.testing.assert_array_equal(augmented[0][2], augmented[0][0] * 3)
                np.testing.assert_array_equal(augmented[0][0] * 11, augmented[2])
                np.testing.assert_array_equal(augmented[0][0] * 13, augmented[4])
                self.assertEqual(int(augmented[3].sum()), int(inputs[3].sum()))
                self.assertEqual(metadata["outcome"], "rigid_rotation")
                self.assertEqual(len(metadata["offset_voxels"]), 3)

    def test_translation_only_preserves_boundary_signal_and_values(self):
        inputs = make_batch(shape=(20, 20, 20))
        for seed in range(10):
            out, meta = augment_rigid_3d(
                inputs, np.random.default_rng(seed), enabled=True,
                max_translation_voxels=6, rotation_probability=0.0,
            )
            self.assertEqual(meta["quarter_turns"], 0)
            self.assertTrue(all(abs(v) <= 6 for v in meta["offset_voxels"]))
            self.assertEqual(int(out[3].sum()), int(inputs[3].sum()))
            np.testing.assert_array_equal(
                np.sort(out[1].ravel()), np.sort(inputs[1].ravel())
            )

    def test_full_cube_cannot_translate_or_rotate_between_unequal_spatial_axes(self):
        shape = (13, 17, 7)
        image = np.arange(1, 1 + np.prod(shape), dtype=np.float32).reshape(
            *shape, 1)
        mask = np.ones((*shape, 1), dtype=np.float32)
        for seed in range(12):
            out, meta = augment_rigid_3d(
                [image, mask], np.random.default_rng(seed), enabled=True,
                rotation_probability=1, rotation_planes="in_plane",
                max_translation_voxels=9,
            )
            self.assertEqual(meta['rotation_axes'], [0, 1])
            self.assertEqual(meta['quarter_turns'], 2)
            self.assertEqual(meta['offset_voxels'], [0, 0, 0])
            np.testing.assert_array_equal(np.sort(out[0].ravel()), np.sort(image.ravel()))
            self.assertTrue(np.all(out[1] == 1))

    def test_all_planes_rotation_rejected_if_it_cannot_fit_unchanged_geometry(self):
        shape = (5, 17, 23)
        signal = np.ones((*shape, 1), np.float32)
        out, meta = augment_rigid_3d(
            [signal], np.random.default_rng(123), enabled=True,
            rotation_probability=1.0, rotation_planes="all",
            max_translation_voxels=2,
        )
        self.assertIn(meta['quarter_turns'], [1, 2, 3])
        # A proposed non-fitting swap is rejected, not cropped.
        np.testing.assert_array_equal(np.sort(out[0].ravel()), np.sort(signal.ravel()))
        self.assertTrue(np.all(out[0] == 1))

    def test_signal_outside_eroded_mask_is_also_retained(self):
        arr = make_batch()
        # The source mask may be eroded; a real nonzero phase/magnitude voxel
        # outside that mask must still be protected by union-of-signal checks.
        arr[1][0, 0, 0, 0] = 987654.0
        for seed in range(6):
            out, _ = augment_rigid_3d(
                arr, np.random.default_rng(seed), enabled=True,
                max_translation_voxels=10, rotation_probability=1.0,
            )
            self.assertEqual(int((out[1] == 987654.0).sum()), 1)
            np.testing.assert_array_equal(
                np.sort(out[1].ravel()), np.sort(arr[1].ravel())
            )

    def test_reproducibility_with_identical_rng_and_config(self):
        batch = make_batch()
        x, xm = augment_rigid_3d(
            batch, np.random.default_rng(222), enabled=True,
            max_translation_voxels=6, rotation_probability=0.75)
        y, ym = augment_rigid_3d(
            batch, np.random.default_rng(222), enabled=True,
            max_translation_voxels=6, rotation_probability=0.75)
        self.assertEqual(xm, ym)
        for a, b in zip(x, y):
            np.testing.assert_array_equal(a, b)

    def test_disabled_is_backward_compatible(self):
        batch = make_batch()
        outputs, info = augment_rigid_3d(batch, np.random.default_rng(8))
        self.assertFalse(info['applied'])
        self.assertEqual(info['outcome'], 'disabled')
        for a, b in zip(outputs, batch):
            np.testing.assert_array_equal(a, b)

    def test_all_empty_tensor_is_not_rewritten(self):
        empty = np.zeros((8, 10, 12, 1), dtype=np.float32)
        outputs, info = augment_rigid_3d(
            [empty], np.random.default_rng(9), enabled=True,
            rotation_probability=1.0)
        self.assertEqual(info['outcome'], 'empty_signal_unchanged')
        np.testing.assert_array_equal(outputs[0], empty)

    def test_invalid_arguments_and_nonfinite_data_fail_closed(self):
        batch = make_batch()
        cases = [
            dict(max_translation_voxels=-1),
            dict(max_translation_voxels=1.2),
            dict(rotation_probability=-.01),
            dict(rotation_probability=1.01),
            dict(rotation_planes='unsafe'),
        ]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                augment_rigid_3d(batch, np.random.default_rng(1),
                                 enabled=True, **settings)
        corrupt = [x.copy() for x in batch]
        corrupt[1][1, 2, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            augment_rigid_3d(corrupt, np.random.default_rng(1), enabled=True)
        invalid = [batch[0], np.zeros((8,8,8,1), dtype=np.float32)]
        with self.assertRaisesRegex(ValueError, 'share'):
            augment_rigid_3d(invalid, np.random.default_rng(1), enabled=True)


if __name__ == "__main__":
    unittest.main()
