"""Preview integration checks without a GPU or an additional generator model."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from echo_gen_preview import axial_chunk_axis, render_preview, PreviewWriter
from echo_gen_dashboard import Jobs

ROOT=Path(__file__).resolve().parent

class PreviewTests(unittest.TestCase):
    def test_axial_axis_accounts_for_chunk_transpose(self):
        self.assertEqual(axial_chunk_axis(np.eye(4)),2)
        affine=np.eye(4);affine[:3,:3]=np.eye(3)[:,[2,0,1]]
        self.assertEqual(axial_chunk_axis(affine),1)
        affine[:3,:3]=np.eye(3)[:,[0,2,1]]
        self.assertEqual(axial_chunk_axis(affine),0)
    def test_png_roundtrip_and_bounds(self):
        from PIL import Image
        import base64,io
        x=np.linspace(-1,1,64*80,dtype=np.float32).reshape(64,80)
        meta={'modality':'phase'}
        result=render_preview(np.stack([x,x,x,np.ones_like(x)]),meta)
        im=Image.open(io.BytesIO(base64.b64decode(result['image'].split(',')[1])))
        self.assertEqual(im.size,(320,64))
        self.assertEqual(result['display_range'][0],-result['display_range'][1])
        np.testing.assert_array_equal(np.array(im)[:,80:160],np.array(im)[:,160:240])
        self.assertFalse(np.array(im)[:,240:].any())
        with self.assertRaises(ValueError):render_preview(np.zeros((4,257,10)),meta)
    def test_empty_mask_and_nonfinite(self):
        result=render_preview(np.zeros((4,16,16)),{'modality':'magnitude'})
        self.assertFalse(result['has_brain'])
        with self.assertRaises(ValueError):render_preview(np.full((4,16,16),np.nan),{'modality':'magnitude'})
    def test_async_single_file_and_throttle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'preview_test.json';writer=PreviewWriter(path,interval=60)
            self.assertTrue(writer.submit(np.ones((4,8,8)),{'modality':'magnitude','run_name':'test'}))
            self.assertFalse(writer.submit(np.ones((4,8,8)),{'modality':'magnitude'}))
            writer.close();self.assertEqual(json.loads(path.read_text())['run_name'],'test')
            self.assertEqual(len(list(Path(tmp).iterdir())),1)
    def test_endpoint_isolated_to_active_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs=Jobs(Path(tmp));jobs.settings={'TRAINING_ROOT':tmp,'TRAINING_MODALITIES_TO_RUN':['magnitude']}
            jobs.action='train';jobs.run_name='current'
            folder=Path(tmp)/'magnitude';folder.mkdir()
            p=folder/'preview_current.json';p.write_text(json.dumps({'run_name':'current','modality':'magnitude','image':'small'}))
            self.assertEqual(jobs.preview('magnitude')['run_name'],'current')
            self.assertIn('magnitude',jobs.status()['previews'])
            self.assertNotIn('image',json.dumps(jobs.status()))
            with self.assertRaises(ValueError):jobs.preview('../../secrets')
            jobs.run_name='other'
            with self.assertRaises(FileNotFoundError):jobs.preview('magnitude')
    def test_critic_loop_preserves_metrics_and_call_counts(self):
        # Execute the actual orchestration method with tiny instrumented updates.
        tree=ast.parse((ROOT/'echo_gen_training.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CEASETTrainer')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_train_batch_impl')
        namespace={};exec(compile(ast.Module(body=[method],type_ignores=[]),'training_method','exec'),namespace)
        class Fake:
            def __init__(self): self.calls=[]
            def _critic_update(self,inputs,real,axis):
                self.calls.append(axis);d={'d_loss':2.}
                if axis>=0:d['_preview']=np.ones((4,3,3))
                return d
            def _generator_update(self,*args):return {'g_total':3.}
            def _continuous_auxiliary_update(self,*args):return {'aux':4.}
        for axis in (-1,2):
            fake=Fake();out=namespace['_train_batch_impl'](fake,None,None,2,axis)
            self.assertEqual(fake.calls,[axis,-1])
            out.pop('_preview',None)
            self.assertEqual(out,{'d_loss':2.,'g_total':3.,'aux':4.})
    def test_middle_slice_and_bounded_transfer(self):
        # Run the actual slice-selection block against a small NumPy TensorFlow adapter.
        source=(ROOT/'echo_gen_training.py').read_text()
        a=source.index('            preview = None\n');b=source.index('            # Full-resolution GP',a)
        import textwrap
        class TF:
            @staticmethod
            def shape(a):return np.array(a.shape)
            @staticmethod
            def stop_gradient(a):return a
            @staticmethod
            def gather(a,index,axis):return np.take(a,index,axis=axis)
            maximum=staticmethod(np.maximum)
            stack=staticmethod(np.stack)
        volume=np.arange(1*300*270*5*1,dtype=np.float32).reshape(1,300,270,5,1)
        for axis in (0,1,2):
            ns={'tf':TF,'preview_axis':axis,'fake_m':volume,'real_m':volume,'mask':np.ones_like(volume),'inputs':{'echo_images':volume[:,None]}}
            exec(textwrap.dedent(source[a:b]),ns)
            result=ns['preview'];self.assertEqual(result.shape[0],4)
            self.assertLessEqual(max(result.shape[1:]),256)
            expected=np.take(volume[0,...,0],volume.shape[axis+1]//2,axis=axis)
            stride=np.maximum((np.array(expected.shape)+255)//256,1)
            np.testing.assert_array_equal(result[1],expected[::stride[0],::stride[1]])

if __name__=='__main__':unittest.main()
