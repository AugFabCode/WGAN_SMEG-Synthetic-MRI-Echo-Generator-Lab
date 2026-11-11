"""Cooperative training interruption at safe Python boundaries."""
import threading

_requested = threading.Event()

class TrainingStopped(Exception):
    pass

def request_stop(*_):
    _requested.set()

def reset_stop():
    _requested.clear()

def check_training_stop():
    if _requested.is_set():
        raise TrainingStopped('Training interrupted. Resume uses the last saved checkpoint; the unfinished epoch is not saved.')
