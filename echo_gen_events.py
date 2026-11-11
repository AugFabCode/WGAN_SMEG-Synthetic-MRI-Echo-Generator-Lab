"""Best-effort, bounded events from the existing process log; no new log files."""
import re
from collections import deque

_ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
_PROGRESS = re.compile(r'\bE(\d+)\s+B(\d+)/(\d+)')


def parse_log_events(text, *, offset=0, source='', limit=100):
    """Stable byte-position IDs; unknown output remains visible as an Output event.

    The caller provides at most the recent log tail. No timestamps are invented:
    order is log order, and any timestamps in the original message are retained.
    """
    events = deque(maxlen=limit)
    position = offset
    for raw in text.splitlines(keepends=True):
        event_id = f'{source}:{position}'
        position += len(raw.encode('utf-8', errors='replace'))
        message = _ANSI.sub('', raw).strip()
        if not message:
            continue
        level, category = 'info', 'Output'
        lower = message.lower()
        if re.search(r'\b\w*(?:error|exception)\b\s*:|traceback \(most recent call last\)|\bfatal\b', lower):
            level, category = 'error', 'Error'
        elif '[stopped]' in lower or '[early stop]' in lower:
            category = 'Training stopped'
        elif '[warn' in lower or 'warning' in lower or ('[preview]' in lower and any(w in lower for w in ('skipped', 'unavailable', 'failed'))):
            level, category = 'warning', 'Warning'
        elif '[save]' in lower:
            category = 'Checkpoint saved'
        elif 'complete' in lower and 'epoch=' in lower:
            category = 'Epoch completed'
        elif _PROGRESS.search(message) or 'first-batch-diag' in lower:
            category = 'Training progress'
        elif 'eval' in lower and any(w in lower for w in ('batch', 'validation', 'start', 'complete')):
            category = 'Evaluation'
        elif any(tag in lower for tag in ('[split]', '[chunks]', '[ready]', '[derived', '[preprocess')):
            category = 'Preprocessing'
        elif any(tag in lower for tag in ('[init]', '[resume]', '[cleanup]', '[paths]')):
            category = 'Setup'
        event = {'id': event_id, 'level': level, 'category': category, 'message': message[:2000]}
        progress = _PROGRESS.search(message)
        if progress:
            event.update(epoch=int(progress[1]), batch=int(progress[2]), batches=int(progress[3]))
        modality = re.search(r'\[(MAGNITUDE|PHASE)\]', message)
        if modality:
            event['modality'] = modality[1].lower()
        events.append(event)
    return list(events)
