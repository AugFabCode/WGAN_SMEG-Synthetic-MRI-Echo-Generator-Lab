"""Bounded CPU-only image publishing for existing training tensors. No TensorFlow import."""
import base64
import json
import struct
import threading
import time
import zlib
from urllib.request import Request, urlopen
import numpy as np


def axial_chunk_axis(affine):
    """Chunks transpose source axes (1,0,2); flips do not change anatomical plane."""
    from nibabel.orientations import io_orientation
    orientation = io_orientation(np.asarray(affine))
    source = np.flatnonzero(orientation[:, 0] == 2)
    if len(source) != 1:
        raise ValueError('Cannot identify the superior-inferior source axis')
    return (1, 0, 2).index(int(source[0]))


def png_rgb(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3: raise ValueError('RGB image required')
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    raw = b''.join(b'\0' + row.tobytes() for row in rgb)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))
            + chunk(b'IDAT',zlib.compress(raw,1)) + chunk(b'IEND',b''))


def render_preview(planes, metadata):
    """Render K actual input echoes plus generated, target and error.

    Each 2-D panel is bounded to 256 pixels per side on the GPU. The CPU
    performs exact 3x nearest-neighbor pixel replication ONLY for display;
    no image intensities or tensors used for training are interpolated.
    """
    planes = np.asarray(planes, dtype=np.float32)
    if planes.ndim != 3 or not (4 <= planes.shape[0] <= 7) or max(planes.shape[1:]) > 256:
        raise ValueError('Expected 1-4 input planes plus generated, target and mask, at most 256 pixels per side')
    if not np.isfinite(planes).all():
        raise ValueError('Non-finite preview')
    count = planes.shape[0] - 3
    for field in ('input_tes_ms', 'input_echoes'):
        if field in metadata and len(metadata[field]) != count:
            raise ValueError(f'{field} does not match the number of input planes')

    inputs = planes[:count]
    generated, target, mask = planes[count:]
    mask = mask > .5
    # A common display window across every acquired input and target makes
    # magnitude/phase comparisons meaningful even when each TE differs.
    values = np.concatenate([plane[mask] for plane in (*inputs, target)])
    if values.size:
        low, high = np.percentile(values, [1, 99]).tolist()
        if metadata['modality'] == 'phase':
            high = max(abs(low), abs(high), 1e-6)
            low = -high
        elif high <= low:
            low = min(0., low)
            high = max(high, low + 1e-6)
    else:
        low, high = 0., 1.

    def gray(a):
        a = np.clip((a - low) / max(high - low, 1e-6), 0, 1) * mask
        return np.repeat((a[..., None] * 255).astype(np.uint8), 3, axis=-1)

    error = np.abs(generated - target) * mask
    error_max = max(float(np.percentile(error[mask], 99)) if mask.any() else 0., 1e-6)
    e = np.clip(error / error_max, 0, 1)
    heat = np.stack([255*e, 100*e, np.zeros_like(e)], axis=-1).astype(np.uint8)
    panels = [gray(a) for a in inputs] + [gray(generated), gray(target), heat]
    scale = 3
    # Independent replication prevents any interpolation between MRI panels.
    rgb = np.concatenate([
        np.repeat(np.repeat(panel, scale, axis=0), scale, axis=1)
        for panel in panels
    ], axis=1)
    return dict(
        metadata,
        image='data:image/png;base64,' + base64.b64encode(png_rgb(rgb)).decode(),
        input_count=count,
        panel_count=len(panels),
        panel_roles=['input'] * count + ['generated', 'target', 'error'],
        display_range=[low, high], error_range=[0., error_max],
        has_brain=bool(mask.any()), units='sample-normalized',
        panel_shape=list(inputs.shape[1:]),
        display_panel_shape=[inputs.shape[1]*scale, inputs.shape[2]*scale],
        display_scale=scale, display_interpolation='nearest',
        captured_at=time.time(),
    )


class PreviewWriter:
    """At most one small CPU snapshot in flight; never queue training tensors."""
    def __init__(self, url=None, token=None, interval=60.):
        self.url=url; self.token=token; self.interval=interval
        self.last=-float('inf'); self.thread=None
    def due(self):
        return bool(self.url and self.token) and time.monotonic()-self.last >= self.interval and not (self.thread and self.thread.is_alive())
    def submit(self, planes, metadata):
        if not self.due(): return False
        snapshot=np.array(planes,dtype=np.float32,copy=True)
        self.last=time.monotonic()
        def publish():
            try:
                payload=render_preview(snapshot,metadata)
                request=Request(self.url, data=json.dumps(payload,allow_nan=False).encode(),
                                headers={'Content-Type':'application/json','X-Echo-Preview-Token':self.token}, method='POST')
                with urlopen(request, timeout=2) as response:
                    response.read(1024)
            except Exception as exc:
                print(f'[PREVIEW] Skipped: {exc}',flush=True)
        self.thread=threading.Thread(target=publish,name='echo-preview',daemon=True)
        self.thread.start()
        return True
    def close(self):
        if self.thread: self.thread.join(timeout=5)
