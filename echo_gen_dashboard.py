"""Local Echo Gen dashboard; standard library only. Start in the training environment."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPOSITORY = 'https://github.com/AugFabCode/WGAN_SMEG-Synthetic-MRI-Echo-Generator-Lab'

class Jobs:
    def __init__(self, project):
        self.project = project
        self.process = None
        self.log = None
        self.settings = None
        self.action = None
        self.run_name = None
        self.lock = threading.Lock()

    def start(self, action, config):
        if action not in ('prepare', 'train'): raise ValueError('Invalid action')
        with self.lock:
            if self.process and self.process.poll() is None: raise ValueError('A job is already running')
            if (self.project / '.echo_gen_job.lock').exists(): raise ValueError('A job owns the project lock; see terminal/README')
            directory = self.project / 'training' / 'SWI-CEASET-ABRIM-V8'
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + secrets.token_hex(3)
            config = dict(config)
            if config.get('TRAINING_RESUME',True) is False and not config.get('RUN_INSTANCE'):
                config['RUN_INSTANCE'] = stamp
            config_path = directory / f'dashboard_config_{stamp}.json'
            config_path.write_text(json.dumps(config, indent=2, allow_nan=False))
            command = [sys.executable, '-u', str(HERE / ('preprocess_echo_gen.py' if action == 'prepare' else 'train_echo_gen.py')),
                       '--project-root', str(self.project), '--config', str(config_path)]
            env = dict(os.environ, ECHO_GEN_RUN_ID=stamp)
            checked = subprocess.run(command + ['--dry-run'], env=env, capture_output=True, text=True, timeout=30)
            if checked.returncode: raise ValueError(checked.stderr[-5000:] or checked.stdout[-5000:])
            # Dry-run emits diagnostic lines before the JSON object.
            resolved = json.loads(checked.stdout[checked.stdout.index('{'):])
            self.run_name = resolved['RUN_NAME']
            self.settings = resolved
            self.action = action
            self.log = directory / f'dashboard_{action}_{stamp}.log'
            with self.log.open('wb') as log:
                self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)

    def status(self):
        with self.lock:
            code = self.process.poll() if self.process else None
            text = ''
            if self.log and self.log.exists():
                with self.log.open('rb') as f:
                    f.seek(max(0, self.log.stat().st_size - 32000))
                    text = f.read().decode('utf-8', 'replace')
            histories = {}
            if self.settings and self.action == 'train':
                for modality in self.settings['TRAINING_MODALITIES_TO_RUN']:
                    p = Path(self.settings['TRAINING_ROOT']) / modality / f'history_{self.run_name}.csv'
                    if p.is_file(): histories[modality] = list(csv.DictReader(io.StringIO(p.read_text())))
            previews = {}
            if self.settings and self.action == 'train' and self.settings.get('TRAINING_PREVIEW_ENABLED', True):
                for modality in self.settings['TRAINING_MODALITIES_TO_RUN']:
                    path = Path(self.settings['TRAINING_ROOT']) / modality / f'preview_{self.run_name}.json'
                    if path.is_file():
                        stat = path.stat()
                        previews[modality] = f'{stat.st_mtime_ns}-{stat.st_size}'
            return {'previews': previews, 'state':'idle' if not self.process else 'running' if code is None else 'completed' if code == 0 else 'failed',
                    'exit_code':code, 'action':self.action, 'run_name':self.run_name, 'log':text,
                    'histories':histories, 'project':str(self.project),
                    'run_instance': self.settings.get('RUN_INSTANCE') if self.settings else '',
                    'model_root': self.settings.get('TRAINING_MODEL_ROOT') if self.settings else None}

    def preview(self, modality):
        if modality not in ('magnitude', 'phase'): raise ValueError('Invalid modality')
        with self.lock:
            if not self.settings or self.action != 'train': raise FileNotFoundError('No training preview')
            if not self.settings.get('TRAINING_PREVIEW_ENABLED', True): raise FileNotFoundError('Previews disabled')
            path = Path(self.settings['TRAINING_ROOT']) / modality / f'preview_{self.run_name}.json'
            # Four 768x768 RGB panels are <9.5 MB even as uncompressed base64.
            if path.stat().st_size > 12_000_000: raise ValueError('Preview exceeds size limit')
            payload = json.loads(path.read_text())
            if payload.get('run_name') != self.run_name or payload.get('modality') != modality:
                raise ValueError('Preview identity mismatch')
            return payload


def serve(project, port):
    jobs = Jobs(project)
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, data, mime='application/json'):
            body = data.encode() if isinstance(data, str) else json.dumps(data, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', mime+'; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)
        def trusted(self):
            return self.headers.get('Host') in {f'127.0.0.1:{port}', f'localhost:{port}'}
        def do_GET(self):
            if not self.trusted(): return self.reply(403, {'error':'Invalid host'})
            path = urlparse(self.path).path
            if path == '/':
                page = (HERE/'echo_gen_dashboard.html').read_text().replace('__TOKEN__',token)
                return self.reply(200,page,'text/html')
            if path == '/api/defaults': return self.reply(200,json.loads((HERE/'echo_gen_defaults.json').read_text()))
            if path == '/api/status': return self.reply(200,jobs.status())
            if path == '/api/preview':
                try:
                    return self.reply(200,jobs.preview(parse_qs(urlparse(self.path).query).get('modality',[''])[0]))
                except FileNotFoundError as exc: return self.reply(404,{'error':str(exc)})
                except (ValueError, OSError) as exc: return self.reply(400,{'error':str(exc)})
            self.reply(404, {'error':'Not found'})
        def do_POST(self):
            if not self.trusted() or self.headers.get('X-Echo-Token') != token: return self.reply(403,{'error':'Invalid request token'})
            if self.path != '/api/start': return self.reply(404,{'error':'Not found'})
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0 < size < 100000: raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(size))
                jobs.start(payload['action'], payload['config'])
                self.reply(200,jobs.status())
            except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as exc:
                self.reply(400,{'error':str(exc)})
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(('127.0.0.1',port), Handler)
    print(f'Echo Gen dashboard: http://127.0.0.1:{port}\nProject: {project}\nRepository: {REPOSITORY}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: print('Dashboard stopped. An active worker continues; its project lock protects the data.')
    finally: server.server_close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=HERE)
    parser.add_argument('--port',type=int,default=8765)
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    if not project.is_dir(): parser.error('Project root must exist')
    serve(project,args.port)
