"""Small synthetic-array checks; uses preprocessing Python dependencies only."""
import unittest
import tempfile
from pathlib import Path
import sys
import json
from contextlib import ExitStack
from unittest.mock import patch
from types import SimpleNamespace
import pandas as pd
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
    def _validate_fixture(self, sample_builder=None):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp)
            (root/'derived_data_signature.json').write_text(json.dumps({'signature':'fixture'}))
            array=root/'chunk.npy'
            np.save(array,np.ones((8,8,8,1),np.float32))
            rows=[]
            for case,split in [('001','training'),('002','eval')]:
                row={'case_id':case,'split':split,'mask_path':str(array)}
                row.update({f'{mod}_{echo}_path':str(array) for mod in ('magnitude','phase') for echo in p.CFG.echo_labels})
                rows.append(row)
            frame=pd.DataFrame(rows)
            stats=root/'stats.csv'
            pd.DataFrame([{'case_id':case,'echo_label':echo} for case in ('case001','case002') for echo in p.CFG.echo_labels]).to_csv(stats,index=False)
            # Non-numeric IDs avoid pandas' automatic numeric inference in this fixture.
            frame['case_id']=['case001','case002']
            cfg=SimpleNamespace(preprocessing_data_root=root,echo_labels=p.CFG.echo_labels)
            stack.enter_context(patch.object(p,'_derived_signature',return_value='fixture'))
            stack.enter_context(patch.object(p,'load_active_ceaset_manifest',return_value=frame))
            stack.enter_context(patch.object(p,'load_or_create_ceaset_split',return_value=frame[['case_id','split']]))
            stack.enter_context(patch.object(p,'CEASET_STATS_CSV',stats))
            return p.validate_training_artifacts(cfg,sample_builder=sample_builder)
    def test_preprocessing_validation_without_training_sampler(self):
        self.assertEqual(len(self._validate_fixture()),2)
        self.assertNotIn('tensorflow',sys.modules)
    def test_training_validation_exercises_both_modalities_and_partitions(self):
        calls=[]
        def sampler(row,modality,stats,target,inputs):
            calls.append((modality,row['split']))
            return {'brain_mask':np.ones((8,8,8,1))},np.ones((8,8,8,1)),{}
        self._validate_fixture(sampler)
        self.assertEqual(set(calls),{(m,s) for m in ('magnitude','phase') for s in ('training','eval')})
    def test_training_validation_rejects_invalid_sample(self):
        def sampler(*args):
            return {'brain_mask':np.ones((8,8,8,1))},np.full((8,8,8,1),np.nan),{}
        with self.assertRaisesRegex(RuntimeError,'Non-finite training sample'):
            self._validate_fixture(sampler)
    def test_fast_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.npy';a=np.arange(512,dtype=np.float32).reshape(8,8,8,1)
            p._save_npy_float32(path,a)
            np.testing.assert_array_equal(p.load_chunk(path),a)

if __name__=='__main__': unittest.main()
