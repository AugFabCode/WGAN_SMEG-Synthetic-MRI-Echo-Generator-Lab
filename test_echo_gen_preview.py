"""Preview integration checks without a GPU or an additional generator model."""
import ast
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
from echo_gen_preview import axial_chunk_axis, render_preview, PreviewWriter
from echo_gen_dashboard import Jobs

ROOT=Path(__file__).resolve().parent

class PreviewTests(unittest.TestCase):
    def job(self, tmp):
        jobs=Jobs(Path(tmp));jobs.settings={'TRAINING_ROOT':tmp,'TRAINING_MODALITIES_TO_RUN':['magnitude']}
        jobs.action='train';jobs.run_name='current';jobs.preview_token='test-token'
        return jobs
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
        self.assertEqual(im.size,(960,192))
        self.assertEqual(result['panel_shape'],[64,80])
        self.assertEqual(result['display_panel_shape'],[192,240])
        self.assertEqual(result['display_scale'],3)
        self.assertEqual(result['input_count'],1)
        self.assertEqual(result['panel_roles'],['input','generated','target','error'])
        self.assertEqual(result['display_interpolation'],'nearest')
        self.assertEqual(result['display_range'][0],-result['display_range'][1])
        np.testing.assert_array_equal(np.array(im)[:,240:480],np.array(im)[:,480:720])
        self.assertFalse(np.array(im)[:,720:].any())
        with self.assertRaises(ValueError):render_preview(np.zeros((4,257,10)),meta)
    def test_exact_pixel_replication_and_panel_isolation(self):
        from PIL import Image
        import base64,io
        x=np.tile(np.array([0.,0.,1.,1.,0.,1.],np.float32),(5,1))
        planes=np.stack([x,np.zeros_like(x),np.ones_like(x),np.ones_like(x)])
        before=planes.copy()
        result=render_preview(planes,{'modality':'magnitude'})
        actual=np.array(Image.open(io.BytesIO(base64.b64decode(result['image'].split(',')[1]))))
        rgb=np.repeat((x[...,None]*255).astype(np.uint8),3,axis=-1)
        expected=np.repeat(np.repeat(rgb,3,axis=0),3,axis=1)
        for y in range(rgb.shape[0]):
            for x in range(rgb.shape[1]):
                block=actual[3*y:3*y+3,3*x:3*x+3]
                self.assertTrue((block == rgb[y,x]).all())
        np.testing.assert_array_equal(actual[:,:18],expected)
        self.assertFalse(actual[:,18:36].any())
        self.assertTrue((actual[:,36:54]==255).all())
        np.testing.assert_array_equal(planes,before)
    def test_full_size_payload_accepted_by_dashboard(self):
        from PIL import Image
        import base64,io
        rng=np.random.default_rng(42)
        # 4 different acquired echoes plus generated/target/mask are the
        # maximum payload: seven 3x pixel-replicated panels.
        planes=rng.normal(size=(7,256,256)).astype(np.float32);planes[6]=1
        payload=render_preview(planes,{'modality':'magnitude','run_name':'current',
               'input_tes_ms':[6.14,10.14,14.14,18.14],
               'input_echoes':['TE1','TE2','TE3','TE4'],
               'target_echo':'TE9','target_te_ms':38.14})
        self.assertEqual(payload['input_count'],4)
        self.assertEqual(payload['panel_roles'],['input']*4+['generated','target','error'])
        im=Image.open(io.BytesIO(base64.b64decode(payload['image'].split(',')[1])))
        self.assertEqual(im.size,(5376,768))
        encoded=json.dumps(payload)
        self.assertLess(len(encoded.encode()),12_000_000)
        with tempfile.TemporaryDirectory() as tmp:
            jobs=self.job(tmp)
            jobs.accept_preview(payload,'test-token')
            self.assertEqual(list(Path(tmp).iterdir()),[])
            self.assertEqual(jobs.preview('magnitude')['display_scale'],3)
    def test_empty_mask_and_nonfinite(self):
        result=render_preview(np.zeros((4,16,16)),{'modality':'magnitude'})
        self.assertFalse(result['has_brain'])
        with self.assertRaises(ValueError):render_preview(np.full((4,16,16),np.nan),{'modality':'magnitude'})
    def test_async_memory_publish_and_throttle(self):
        with tempfile.TemporaryDirectory() as tmp, patch('echo_gen_preview.urlopen') as send:
            send.return_value.__enter__.return_value.read.return_value=b'{}'
            writer=PreviewWriter('http://127.0.0.1:8765/api/preview','test-token',interval=60)
            self.assertTrue(writer.submit(np.ones((4,8,8)),{'modality':'magnitude','run_name':'test'}))
            self.assertFalse(writer.submit(np.ones((4,8,8)),{'modality':'magnitude'}))
            writer.close()
            request=send.call_args.args[0]
            self.assertEqual(json.loads(request.data)['run_name'],'test')
            self.assertEqual(request.get_header('X-echo-preview-token'),'test-token')
            self.assertEqual(list(Path(tmp).iterdir()),[])
    def test_no_receiver_skips_preview(self):
        writer=PreviewWriter()
        self.assertFalse(writer.due())
        self.assertFalse(writer.submit(np.ones((4,8,8)),{}))
    def test_endpoint_isolated_to_active_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs=self.job(tmp)
            payload={'run_name':'current','modality':'magnitude','image':'data:image/png;base64,first'}
            with self.assertRaises(PermissionError):jobs.accept_preview(payload,'wrong')
            jobs.accept_preview(payload,'test-token')
            self.assertEqual(jobs.preview('magnitude')['run_name'],'current')
            self.assertIn('magnitude',jobs.status()['previews'])
            self.assertNotIn('image',json.dumps(jobs.status()))
            jobs.accept_preview(dict(payload,image='data:image/png;base64,second'),'test-token')
            self.assertEqual(len(jobs.previews),1)
            self.assertEqual(jobs.preview('magnitude')['image'],'data:image/png;base64,second')
            self.assertEqual(jobs.status()['previews']['magnitude'],2)
            self.assertEqual(list(Path(tmp).iterdir()),[])
            with self.assertRaises(ValueError):jobs.preview('../../secrets')
            jobs.run_name='other'
            with self.assertRaises(FileNotFoundError):jobs.preview('magnitude')
            with self.assertRaises(ValueError):jobs.accept_preview(payload,'test-token')
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
        # Exercise the actual GPU preview slice-selection block with a NumPy
        # TensorFlow adapter. Every active input has a distinct source value.
        source=(ROOT/'echo_gen_training.py').read_text()
        a=source.index('            preview = None\n')
        b=source.index('            # Full-resolution GP',a)
        import textwrap
        from contextlib import nullcontext
        from types import SimpleNamespace
        class TF:
            @staticmethod
            def shape(a):return np.array(a.shape)
            @staticmethod
            def stop_gradient(a):return a
            @staticmethod
            def gather(a,index,axis):return np.take(a,index,axis=axis)
            @staticmethod
            def boolean_mask(a,condition,axis=0):
                return np.compress(np.asarray(condition,dtype=bool),a,axis=axis)
            @staticmethod
            def device(*args):return nullcontext()
            maximum=staticmethod(np.maximum)
            stack=staticmethod(np.stack)
            concat=staticmethod(np.concatenate)
        volume=np.arange(1*300*270*5*1,dtype=np.float32).reshape(1,300,270,5,1)
        echoes=np.stack([volume[0]+float(k)*1000000 for k in range(4)],axis=0)[None,...]
        for axis in (0,1,2):
            ns={'tf':TF,'self':SimpleNamespace(generator_device='/CPU:0'),
                'preview_axis':axis,'fake_m':volume,'real_m':volume,
                'mask':np.ones_like(volume),
                'inputs':{'echo_images':echoes,'echo_mask':np.ones((1,4),np.float32)}}
            exec(textwrap.dedent(source[a:b]),ns)
            result=ns['preview']
            self.assertEqual(result.shape[0],7)
            self.assertLessEqual(max(result.shape[1:]),256)
            expected=np.take(volume[0,...,0],volume.shape[axis+1]//2,axis=axis)
            stride=np.maximum((np.array(expected.shape)+255)//256,1)
            for k in range(4):
                np.testing.assert_array_equal(
                    result[k],expected[::stride[0],::stride[1]]+float(k)*1000000)
            np.testing.assert_array_equal(result[4],expected[::stride[0],::stride[1]])
            # A padded stack still displays only active inputs.
            ns['inputs']['echo_mask']=np.array([[1,1,0,0]],np.float32)
            exec(textwrap.dedent(source[a:b]),ns)
            self.assertEqual(ns['preview'].shape[0],5)

    def test_variable_echo_count_and_metadata(self):
        from PIL import Image
        import base64,io
        mask=np.ones((5,6),np.float32)
        for k in range(1,5):
            planes=np.stack([np.full((5,6),i/10,dtype=np.float32)
                             for i in range(k)]+[
                             np.full((5,6),.35,dtype=np.float32),
                             np.full((5,6),.5,dtype=np.float32),mask])
            metadata={'modality':'magnitude','input_tes_ms':[float(6+4*i) for i in range(k)],
                      'input_echoes':[f'TE{i+1}' for i in range(k)],'target_te_ms':38.0,
                      'target_echo':'TE9'}
            result=render_preview(planes,metadata)
            self.assertEqual(result['input_count'],k)
            self.assertEqual(result['panel_count'],k+3)
            self.assertEqual(result['input_tes_ms'],metadata['input_tes_ms'])
            self.assertEqual(result['input_echoes'],metadata['input_echoes'])
            im=Image.open(io.BytesIO(base64.b64decode(result['image'].split(',')[1])))
            self.assertEqual(im.size,((k+3)*18,15))
            with self.assertRaisesRegex(ValueError,'input_tes_ms'):
                render_preview(planes,dict(metadata,input_tes_ms=[6.0]))

if __name__=='__main__':unittest.main()
