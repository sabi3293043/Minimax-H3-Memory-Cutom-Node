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
        self._wide_next = False

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

    def track(self, frames, regions, sources=(0,), *, frame_pairs=(), frame_budget=8,
              selection='retrieved', step=None):
        started = time.perf_counter()
        from PIL import Image
        from .tracking_selection import (select_tracking_frames, expand_tracking_frames,
                                         tracking_uncertain, tracking_record_trusted)
        from .region_memory import Region
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError('Tracker consumes decoded [frames,3,height,width] RGB observations')
        self.automatic = not bool(regions)
        regions = tuple(regions) or self.regions
        pairs = tuple(frame_pairs)
        ids = select_tracking_frames(len(frames), sources=sources, pairs=pairs,
                                     budget=frame_budget, mode=selection)
        initial_ids = ids
        source = min(sources or (0,))
        if selection == 'retrieved' and self._wide_next:
            ids = expand_tracking_frames(len(frames), ids)
        expanded = ids != initial_ids
        observation = dict(step=step, frame_count=len(frames), resolution=list(frames.shape[-2:]))
        # Encoding is transport only. All indices refer to the caller's existing
        # current-step RGB; no VAE, separate capture, or persistent image bank.
        import torch
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
        request = dict(frames=encoded, ids=ids, source=ids.index(source),
            observation=observation, temp_root=self.temp.name,
            regions=[dict(name=r.name,box=r.box,source=ids.index(source)) for r in regions])
        self.process.stdin.write(json.dumps(request)+'\n')
        self.process.stdin.flush()
        result = self.wait(300)
        if result.get('observation') != observation:
            raise RuntimeError('Tracker response does not match the current observation step/resolution')
        records = {(r['frame'],r['region']):r for r in result['records']}
        if any(frame not in ids for frame, _ in records):
            raise RuntimeError('Tracker response contains frames outside the shared selection')
        regions = tuple(Region(r['name'],tuple(r['box'])) for r in result.get('regions',[]))
        # A hidden object is a normal case, not a reason to rerun an expensive
        # backbone on the same observation. Broaden the NEXT fresh observation.
        # Only a coverage flag survives; RGB and masks are never reused later.
        uncertain = tracking_uncertain(records, ids, regions)
        for record in records.values():
            record['trusted'] = tracking_record_trusted(record)
        self._wide_next = selection == 'retrieved' and uncertain
        self.regions = regions
        self.last_metrics = dict(wall_seconds=time.perf_counter()-started,cpu_threads=self.cpu_threads,
            selection=selection, requested_frames=frame_budget, initial_frames=len(initial_ids),
            selected_frames=len(ids), frame_ids=list(ids), initial_frame_ids=list(initial_ids),
            processed_frames=len(ids), encoded_frames=len(encoded), fallback_expanded=expanded,
            next_coverage_wide=self._wide_next, uncertain=uncertain, observation=observation,
            retrieved_pairs=len(pairs), covered_pairs=sum(s in ids and t in ids for s,t in pairs),
            refreshes=1, **{key:result.get('metrics',{}).get(key,0.)
                for key in ('detection_seconds','propagation_seconds','worker_seconds')})
        self.last_reason = (result.get('reason','') + f'; {selection} {len(ids)}/{len(frames)} shared frames'
                            + ('; broader coverage after uncertainty' if expanded else '')
                            + f'; CPU threads {self.cpu_threads}')
        return records

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
