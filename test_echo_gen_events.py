"""Event parsing and raw-log opt-out checks without a training dependency."""
import tempfile
import unittest
from pathlib import Path
from echo_gen_events import parse_log_events
from echo_gen_dashboard import Jobs

class EventTests(unittest.TestCase):
    def test_recognized_messages_and_unknown_fallback(self):
        lines=['[MAGNITUDE] E003 B0012/0100 12.0% | ETA≈1min',
               '[PHASE] epoch=003 COMPLETE | fixed_eval_MAE=0.1',
               '[SAVE] generator: /models/a.h5', '[SPLIT] subjects: 20',
               '[PREVIEW] Skipped: orientation unavailable',
               'ValueError: missing input', '[STOPPED] Interrupted', 'unrecognized detail']
        events=parse_log_events('\n'.join(lines))
        self.assertEqual([e['category'] for e in events],['Training progress','Epoch completed','Checkpoint saved','Preprocessing','Warning','Error','Training stopped','Output'])
        self.assertEqual(events[0]['batch'],12);self.assertEqual(events[0]['epoch'],3)
        self.assertEqual(events[0]['modality'],'magnitude')
        self.assertEqual(events[4]['level'],'warning');self.assertEqual(events[5]['level'],'error')

    def test_stable_ids_tail_limit_and_safe_plain_text(self):
        prefix='first≈line\n';line='\x1b[31mRuntimeError: <script>x</script>\x1b[0m\n'
        whole=parse_log_events(prefix+line,source='run')
        tail=parse_log_events(line,offset=len(prefix.encode()),source='run')
        self.assertEqual(whole[-1],tail[-1])
        self.assertNotIn('\x1b',tail[0]['message'])
        self.assertIn('<script>',tail[0]['message']) # UI uses textContent, not HTML.
        self.assertEqual(len(parse_log_events('line\n'*1000)),100)
        self.assertEqual(len(parse_log_events('x'*10000)[0]['message']),2000)
        self.assertEqual(parse_log_events('\n \n'),[])

    def test_raw_log_default_opt_out_and_event_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs=Jobs(Path(tmp));jobs.log=Path(tmp)/'job.log'
            raw='[INIT] start\n[SAVE] generator: /a.h5\n'
            jobs.log.write_text(raw)
            a=jobs.status();b=jobs.status()
            self.assertEqual(a['log'],'');self.assertEqual(a['events'],b['events'])
            self.assertEqual(jobs.status(include_log=True)['log'],raw)
            jobs.log=Path(tmp)/'new.log';jobs.log.write_text('new run\n')
            self.assertEqual(len(jobs.status()['events']),1)
            jobs.log.write_text('')
            self.assertEqual(jobs.status()['events'],[])

    def test_bounded_tail_discards_partial_first_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs=Jobs(Path(tmp));jobs.log=Path(tmp)/'job.log'
            jobs.log.write_text('x'*40000+'\n[READY] prepared\n')
            status=jobs.status(include_log=True)
            self.assertEqual(status['log'],'[READY] prepared\n')
            self.assertEqual(status['events'][0]['category'],'Preprocessing')

if __name__=='__main__':unittest.main()
