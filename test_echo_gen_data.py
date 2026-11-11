"""Small synthetic-array checks; uses preprocessing Python dependencies only."""
import unittest
import tempfile
from pathlib import Path
import sys
import numpy as np
import echo_gen_preprocessing as p

class DataTests(unittest.TestCase):
    def test_no_tensorflow_import(self):
        self.assertNotIn('tensorflow',sys.modules)
    def test_chunk_coverage_remainders_and_padding(self):
        shape=(17,19,11);a=np.ones(shape,dtype=np.float32);coverage=np.zeros(shape,int)
        for spec in p.build_chunk_specs(shape,(3,3,2),2):
            slices=tuple(slice(a,b) for a,b in zip(spec['core_start'],spec['core_end']))
            coverage[slices]+=1
            chunk=p.extract_chunk(a,spec)
            self.assertEqual(chunk.shape,spec['chunk_shape'])
            self.assertTrue(all(d%8==0 for d in chunk.shape))
            self.assertEqual(p.ensure_4d_chunk(chunk).shape,chunk.shape+(1,))
        np.testing.assert_array_equal(coverage,np.ones(shape))
    def test_physical_magnitude_decay(self):
        tes=np.array([6.,10.],dtype=np.float32)
        data=np.stack([np.full((3,3,3,1),np.exp(-.04*t),np.float32) for t in tes])
        mask=np.ones((3,3,3,1),np.float32);mask[0]=0
        result=p.magnitude_physics_baseline_np(data,tes,np.ones(2),20.,mask)
        np.testing.assert_allclose(result[1:],np.exp(-.04*20),rtol=1e-3)
        self.assertEqual(np.count_nonzero(result[0]),0)
    def test_fast_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.npy';a=np.arange(512,dtype=np.float32).reshape(8,8,8,1)
            p._save_npy_float32(path,a)
            np.testing.assert_array_equal(p.load_chunk(path),a)

if __name__=='__main__': unittest.main()
