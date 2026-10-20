"""Execution-owned optional tracker; disabled path imports/installs nothing."""
from __future__ import annotations

import base64
from contextlib import AbstractContextManager
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tempfile


class OcclusionTracker(AbstractContextManager):
    def __init__(self, root, *, cpu_threads=0):
        from .cpu_threads import resolve_cpu_threads
        self.root = root
        self.cpu_threads = resolve_cpu_threads(cpu_threads)
        self.last_metrics = {}
        self.process = None
        self.responses = queue.Queue(maxsize=4)
        self.errors = []
        self.temp = None
        self.regions = ()
        self.automatic = False
        self.last_reason = ''

    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix='minimax-tracking-')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(self.cpu_threads),
                   MKL_NUM_THREADS=str(self.cpu_threads),
                   PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONUNBUFFERED='1')
        try:
            self.process = subprocess.Popen([sys.executable, '-u', '-m',
                'mosaichunk_h3.occlusion_worker', str(self.root), str(self.cpu_threads)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', env=env,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except BaseException:
            self.close()
            raise
        def stdout():
            for line in self.process.stdout:
                self.responses.put(line)
        def stderr():
            for line in self.process.stderr:
                self.errors.append(line.rstrip())
                self.errors[:] = self.errors[-8:]
        threading.Thread(target=stdout, daemon=True).start()
        threading.Thread(target=stderr, daemon=True).start()
        try:
            response = self.wait(180)
            if not response.get('ready'):
                raise RuntimeError('SAM 2 worker did not initialize')
            if response.get('cpu_threads', self.cpu_threads) != self.cpu_threads:
                raise RuntimeError('SAM 2 worker thread configuration does not match the requested budget')
        except BaseException:
            self.close()
            raise
        return self

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                from comfy.model_management import throw_exception_if_processing_interrupted
                throw_exception_if_processing_interrupted()
            except ImportError:
                pass
            try:
                result = json.loads(self.responses.get(timeout=.25))
                if 'error' in result:
                    raise RuntimeError(result['error'])
                return result
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError('SAM 2 worker exited: '+'; '.join(self.errors)[-1000:])
        self.close()
        raise TimeoutError('Optional SAM 2 tracker timed out; disable occlusion_aware_matching to continue')

    def track(self, frames, regions, sources=(0,)):
        started = time.perf_counter()
        from PIL import Image
        import torch
        self.automatic = not bool(regions)
        regions = tuple(regions) or self.regions
        # Include every protected source; remaining slots sample the whole clip.
        ids = sorted(set([int(v) for v in sources if 0 <= v < len(frames)]
                         + torch.linspace(0,len(frames)-1,max(1,16-len(sources))).round().int().tolist()))[:16]
        source = min(sources or (0,))
        if source not in ids:
            raise ValueError('Tracker source is outside the decoded video')
        encoded = []
        for frame_id in ids:
            frame = frames[frame_id].detach().cpu()
            if frame.dtype != torch.uint8:
                frame = frame.float().clamp(0,1).mul(255).byte()
            image = Image.fromarray(frame.permute(1,2,0).numpy())
            image.thumbnail((384,384))
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=85)
            encoded.append(base64.b64encode(stream.getvalue()).decode())
        request = dict(frames=encoded, ids=ids, source=ids.index(source), temp_root=self.temp.name, regions=[dict(name=r.name, box=r.box,
            source=ids.index(source)) for r in regions])
        self.process.stdin.write(json.dumps(request)+'\n')
        self.process.stdin.flush()
        result = self.wait(300)
        self.last_metrics = dict(result.get('metrics',{}), wall_seconds=time.perf_counter()-started,
                                 cpu_threads=self.cpu_threads)
        from .region_memory import Region
        self.regions = tuple(Region(r['name'],tuple(r['box'])) for r in result.get('regions',[]))
        self.last_reason = result.get('reason','') + f'; CPU threads {self.cpu_threads}'
        return {(r['frame'],r['region']):r for r in result['records']}

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            for pipe in (self.process.stdin,self.process.stdout,self.process.stderr):
                pipe.close()
        if self.temp is not None:
            self.temp.cleanup()

    def __exit__(self, *args):
        self.close()
