"""Dependency-free integration checks: python -m unittest -v test_echo_gen.py."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import ast

HERE=Path(__file__).resolve().parent

class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def resolve(self, changes=None, action='train', ok=True):
        path=self.root/'config.json';path.write_text(json.dumps(changes or {}))
        script='train_echo_gen.py' if action=='train' else 'preprocess_echo_gen.py'
        p=subprocess.run([sys.executable,str(HERE/script),'--dry-run','--project-root',str(self.root),'--config',str(path)],capture_output=True,text=True)
        if not ok:
            self.assertNotEqual(p.returncode,0);return p.stderr
        self.assertEqual(p.returncode,0,p.stderr)
        return json.loads(p.stdout[p.stdout.index('{'):])
    def test_partition_config_validation_and_identity(self):
        base=self.resolve()
        self.assertEqual([base[k] for k in ('TRAINING_SPLIT_PERCENT','EVALUATION_SPLIT_PERCENT','TEST_SPLIT_PERCENT')],[70,15,15])
        small=self.resolve({'SMALL_DATASET_MODE':True})
        self.assertEqual([small[k] for k in ('TRAINING_SPLIT_PERCENT','EVALUATION_SPLIT_PERCENT','TEST_SPLIT_PERCENT')],[80,20,0])
        self.assertNotEqual(base['RUN_NAME'],small['RUN_NAME'])
        self.resolve({'SMALL_DATASET_MODE':True,'TEST_SPLIT_PERCENT':10},ok=False)
        self.resolve({'SMALL_DATASET_MODE':'true'},ok=False)
        changed=self.resolve({'TRAINING_SPLIT_PERCENT':60,'TEST_SPLIT_PERCENT':25})
        self.assertNotEqual(base['RUN_NAME'],changed['RUN_NAME'])
        for settings in ({'TEST_SPLIT_PERCENT':30},{'TRAINING_SPLIT_PERCENT':0,'EVALUATION_SPLIT_PERCENT':85}, {'TEST_SPLIT_PERCENT':-1}, {'TEST_SPLIT_PERCENT':True}, {'TRAIN_EVAL_SPLIT_FRACTION':.2}):
            self.resolve(settings,ok=False)

    def test_configuration_names_and_resume(self):
        first=self.resolve()
        self.assertEqual(first['RUN_NAME'],self.resolve({'TRAINING_EPOCHS':60})['RUN_NAME'])
        self.assertNotEqual(first['RUN_NAME'],self.resolve({'MODEL_FILTERS':[8,16,24,32]})['RUN_NAME'])
        self.assertEqual(first['RUN_NAME'],self.resolve(action='prepare')['RUN_NAME'])
        self.assertNotEqual(self.resolve({'TRAINING_RESUME':False})['RUN_NAME'],self.resolve({'TRAINING_RESUME':False})['RUN_NAME'])
    def test_fresh_instance_can_resume_and_cannot_overwrite(self):
        fresh=self.resolve({'TRAINING_RESUME':False})
        instance=fresh['RUN_INSTANCE']
        resumed=self.resolve({'TRAINING_RESUME':True,'RUN_INSTANCE':instance})
        self.assertEqual(fresh['RUN_NAME'],resumed['RUN_NAME'])
        model_root=Path(fresh['TRAINING_MODEL_ROOT'])/'magnitude'
        model_root.mkdir(parents=True)
        (model_root/f"generator_{fresh['RUN_NAME']}_epoch001.weights.h5").touch()
        self.resolve({'TRAINING_RESUME':False,'RUN_INSTANCE':instance},ok=False)
    def test_preview_settings_do_not_change_checkpoint_identity(self):
        first=self.resolve()
        other=self.resolve({'TRAINING_PREVIEW_ENABLED':False,'TRAINING_PREVIEW_INTERVAL_SECONDS':120.0})
        self.assertEqual(first['RUN_NAME'],other['RUN_NAME'])
        self.resolve({'TRAINING_PREVIEW_INTERVAL_SECONDS':1.0},ok=False)
    def test_repository_and_notebook_cache_layout(self):
        for folder in ['09_chunks_553','10_training_cache_npy']:(self.root/'preprocess'/folder).mkdir(parents=True)
        r=self.resolve();self.assertTrue(r['CEASET_CHUNK_ROOT'].endswith('09_chunks_553'))
        self.assertTrue(r['CEASET_FAST_NPY_CACHE_ROOT'].endswith('10_training_cache_npy'))
        (self.root/'preprocess/training_chunks').mkdir()
        (self.root/'preprocess/training_chunks/a.csv').touch()
        self.assertTrue(self.resolve()['CEASET_CHUNK_ROOT'].endswith('training_chunks'))
        (self.root/'preprocess/09_chunks_553/b.csv').touch()
        self.resolve(ok=False)
        self.assertTrue(self.resolve({'CEASET_CHUNK_ROOT':'preprocess/09_chunks_553'})['CEASET_CHUNK_ROOT'].endswith('09_chunks_553'))
    def test_invalid_settings_fail_before_processing(self):
        for value in [{'TRAINING_BATCH_SIZE':2},{'MODEL_FILTERS':[8,16]},{'TRAINING_EPOCHS':0},{'UNKNOWN':1},{'MODEL_L2':-1},{'TRAINING_EPOCHS':2.5},{'TRAINING_RESUME':'false'},{'WGAN_GP_LAMBDA':float('nan')}]:
            with self.subTest(value=value):self.resolve(value,ok=False)
    def test_data_manifest_changes_isolate_run(self):
        a=self.resolve()['RUN_NAME']
        p=self.root/'preprocess/normalization_statistics.csv';p.parent.mkdir();p.write_text('changed')
        self.assertNotEqual(a,self.resolve()['RUN_NAME'])
    def test_dry_run_does_not_create_output_folders(self):
        self.resolve()
        self.assertEqual([p.name for p in self.root.iterdir()],['config.json'])
    def test_split_has_no_training_import_in_preprocessing(self):
        tree=ast.parse((HERE/'echo_gen_preprocessing.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertNotIn('tensorflow',imports)
        self.assertNotIn('echo_gen_training',imports)

if __name__=='__main__':unittest.main()
