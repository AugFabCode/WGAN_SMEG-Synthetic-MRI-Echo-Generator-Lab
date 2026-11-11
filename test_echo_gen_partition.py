"""Subject partition persistence, leakage prevention and cooperative-stop regressions."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, Mock
import pandas as pd
import echo_gen_preprocessing as p
from echo_gen_dashboard import Jobs
from echo_gen_control import check_training_stop, request_stop, reset_stop, TrainingStopped


class PartitionTests(unittest.TestCase):
    def test_counts_determinism_and_small_cohorts(self):
        subjects=[f'sub-{i:03}' for i in range(20)]
        result=p.partition_subjects(subjects,[70,15,15],42)
        self.assertEqual({k:list(result.values()).count(k) for k in set(result.values())}, {'training':14,'eval':3,'test':3})
        self.assertEqual(result,p.partition_subjects(reversed(subjects),[70,15,15],42))
        self.assertEqual(set(p.partition_subjects(subjects[:3],[70,15,15],42).values()),{'training','eval','test'})
        with self.assertRaises(ValueError):p.partition_subjects(subjects[:2],[70,15,15],42)
        self.assertEqual(set(p.partition_subjects(subjects[:2],[80,20,0],42).values()),{'training','eval'})
        for weights in ([70,20,20],[0,50,50],[70,31,-1],[float('nan'),15,15]):
            with self.assertRaises(ValueError):p.partition_subjects(subjects,weights,42)

    def fixture(self, directory):
        stack=ExitStack();self.addCleanup(stack.close)
        cases=[f'{i:03}_ses-{session}_run-1' for i in range(20) for session in ('01','02')]
        values={'TRAIN_EVAL_SPLIT_MANIFEST_XLSX':Path(directory)/'filelist_traintestpartitions.xlsx',
                'TRAINING_SPLIT_PERCENT':70.,'EVALUATION_SPLIT_PERCENT':15.,'TEST_SPLIT_PERCENT':15.,
                'TRAIN_EVAL_SPLIT_SEED':42,'REBUILD_PATIENT_SPLIT':False,'OVERWRITE_DERIVED':False,
                'CREATE_NEW_SPLIT_IF_MANIFEST_MISSING':True,'PIPELINE_ACTION':'prepare'}
        for key,value in values.items():stack.enter_context(patch.object(p,key,value))
        stack.enter_context(patch.object(p,'_available_complete_case_ids',return_value=cases))
        return values['TRAIN_EVAL_SPLIT_MANIFEST_XLSX']

    def test_workbook_roundtrip_and_explicit_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.fixture(tmp)
            first=p.load_or_create_ceaset_split()
            self.assertEqual(first.groupby('patient_key')['split'].nunique().max(),1)
            self.assertEqual(first.groupby('split').patient_key.nunique().to_dict(),{'training':14,'eval':3,'test':3})
            before=path.read_bytes()
            pd.testing.assert_frame_equal(first,p.load_or_create_ceaset_split())
            self.assertEqual(before,path.read_bytes())
            with patch.object(p,'TRAINING_SPLIT_PERCENT',60.),patch.object(p,'TEST_SPLIT_PERCENT',25.):
                with self.assertRaisesRegex(ValueError,'REBUILD_PATIENT_SPLIT'):p.load_or_create_ceaset_split()
                self.assertEqual(before,path.read_bytes())
                with patch.object(p,'REBUILD_PATIENT_SPLIT',True),patch.object(p,'OVERWRITE_DERIVED',True):
                    with patch.object(p,'PIPELINE_ACTION','train'):
                        with self.assertRaises(ValueError):p.load_or_create_ceaset_split()
                    changed=p.load_or_create_ceaset_split()
                    self.assertEqual(changed.groupby('split').patient_key.nunique().to_dict(),{'training':12,'eval':3,'test':5})
                    backups=list(Path(tmp).glob('*.backup-*.xlsx'))
                    self.assertEqual(len(backups),1);self.assertEqual(backups[0].read_bytes(),before)
                    p.load_or_create_ceaset_split()
                    self.assertEqual(len(list(Path(tmp).glob('*.backup-*.xlsx'))),1)

    def test_small_dataset_two_way_legacy_workbook_reused_without_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.fixture(tmp)
            cases=p._available_complete_case_ids()
            subjects=sorted({p._training_patient_key_from_case_id(case) for case in cases})
            assigned=p.partition_subjects(subjects,[80,20,0],42)
            frame=pd.DataFrame([{'case_id':c,'patient_key':p._training_patient_key_from_case_id(c),
                                 'split':'test' if assigned[p._training_patient_key_from_case_id(c)]=='eval' else 'train'}
                                for c in cases])
            with pd.ExcelWriter(path,engine='openpyxl') as writer:
                frame.to_excel(writer,sheet_name='patient_split',index=False)
            before=path.read_bytes()
            with patch.object(p,'SMALL_DATASET_MODE',True),\
                 patch.object(p,'TRAINING_SPLIT_PERCENT',80.),\
                 patch.object(p,'EVALUATION_SPLIT_PERCENT',20.),\
                 patch.object(p,'TEST_SPLIT_PERCENT',0.):
                result=p.load_or_create_ceaset_split()
                self.assertEqual(result.groupby('split').patient_key.nunique().to_dict(),
                                 {'training':16,'eval':4})
                self.assertEqual(before,path.read_bytes())
                self.assertFalse(list(Path(tmp).glob('*.backup-*.xlsx')))
                # A wrong legacy ratio is rejected rather than silently accepted.
                with patch.object(p,'TRAINING_SPLIT_PERCENT',75.),\
                     patch.object(p,'EVALUATION_SPLIT_PERCENT',25.):
                    with self.assertRaisesRegex(ValueError,'REBUILD_PATIENT_SPLIT'):
                        p.load_or_create_ceaset_split()

    def test_legacy_labels_and_patient_leakage(self):
        self.assertEqual(p._canonical_split_label('test'),'test')
        with tempfile.TemporaryDirectory() as tmp:
            path=self.fixture(tmp);frame=p.load_or_create_ceaset_split()
            with pd.ExcelWriter(path) as writer:frame.to_excel(writer,sheet_name='patient_split',index=False)
            with self.assertRaisesRegex(ValueError,'Legacy'):p.load_or_create_ceaset_split()
            with patch.object(p,'REBUILD_PATIENT_SPLIT',True),patch.object(p,'OVERWRITE_DERIVED',True):
                frame=p.load_or_create_ceaset_split()
            frame.loc[0,'split']='eval' if frame.loc[0,'split']!='eval' else 'training'
            with pd.ExcelWriter(path,mode='a',engine='openpyxl',if_sheet_exists='replace') as writer:
                frame.to_excel(writer,sheet_name='patient_split',index=False)
            with self.assertRaisesRegex(ValueError,'Patient leakage'):p.load_or_create_ceaset_split()


class StopTests(unittest.TestCase):
    def tearDown(self):reset_stop()
    def test_deferred_stop_and_idempotent_dashboard_request(self):
        reset_stop();check_training_stop();request_stop()
        with self.assertRaises(TrainingStopped):check_training_stop()
        jobs=Jobs(Path('.'));jobs.action='prepare';jobs.process=Mock();jobs.process.poll.return_value=None
        with self.assertRaises(ValueError):jobs.stop()
        jobs.action='train';jobs.stop();jobs.stop()
        jobs.process.send_signal.assert_called_once_with(signal.SIGINT)
        self.assertEqual(jobs.status()['state'],'stopping')
        jobs.process.poll.return_value=130
        self.assertEqual(jobs.status()['state'],'stopped')
        with self.assertRaises(ValueError):jobs.stop()

    @unittest.skipUnless(os.name=='posix','Linux process signal integration')
    def test_stop_releases_runner_lock_and_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'previous.weights.h5';checkpoint.write_bytes(b'previous checkpoint')
            script='''
import sys,time,types
from pathlib import Path
from echo_gen_control import check_training_stop
root=Path(sys.argv[1]);sys.argv=['train_echo_gen.py','--project-root',str(root)]
def train(modality):
    (root/'ready').touch()
    while True:
        time.sleep(.01)
        check_training_stop()
t=types.SimpleNamespace(CFG=None,assert_training_hardware=lambda:None,
    validate_training_artifacts=lambda *a,**kw:None,make_ceaset_sample_from_configuration=None,
    train_ceaset_modality=train)
sys.modules['echo_gen_training']=t
from echo_gen_runner import main
main('train')
'''
            with (root/'worker.log').open('wb') as log:
                proc=subprocess.Popen([sys.executable,'-c',script,tmp],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=Path(__file__).resolve().parent)
                try:
                    deadline=time.monotonic()+10
                    while not (root/'ready').exists() and proc.poll() is None and time.monotonic()<deadline:time.sleep(.01)
                    self.assertTrue((root/'ready').exists(),(root/'worker.log').read_text())
                    jobs=Jobs(root);jobs.process=proc;jobs.action='train';jobs.stop()
                    self.assertEqual(proc.wait(timeout=10),130)
                    self.assertFalse((root/'.echo_gen_job.lock').exists())
                    self.assertEqual(checkpoint.read_bytes(),b'previous checkpoint')
                    self.assertIn('[STOPPED]',(root/'worker.log').read_text())
                finally:
                    if proc.poll() is None:proc.kill();proc.wait()

if __name__=='__main__':unittest.main()
