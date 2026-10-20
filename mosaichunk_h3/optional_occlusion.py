"""Explicit, isolated SAM 2 installation. Importing this module never downloads."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.request

REVISION = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
MODEL_REVISION = 'de431c4043854a71d8101e17995dfe596bf101a5'
MODEL_SHA256 = '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
MODEL_URL = ('https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/'
             + MODEL_REVISION + '/sam2.1_hiera_tiny.pt')
_LOCK = threading.Lock()


def location():
    # Inside the node so portable ComfyUI installations remain self-contained.
    return Path(__file__).resolve().parents[1] / '.optional' / 'sam2'


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def status(root=None):
    root = Path(root) if root else location()
    try:
        manifest = json.loads((root / 'ready.json').read_text())
        return (manifest.get('installer_schema') == 2 and manifest['revision'] == REVISION and manifest['model_sha256'] == MODEL_SHA256
                and (root / 'packages/sam2/build_sam.py').is_file()
                and (root / 'sam2.1_hiera_tiny.pt').stat().st_size == 156008466)
    except (OSError, ValueError, KeyError):
        return False


def ensure_installed(root=None, progress=lambda message: None):
    """Only called by explicit opt-in or the standalone installation command.

    --no-deps and --target prevent pip from upgrading ComfyUI's torch/numpy.
    A subprocess smoke test must pass before the atomic ready marker appears.
    """
    root = Path(root) if root else location()
    with _LOCK:
        if status(root):
            return root
        root.mkdir(parents=True, exist_ok=True)
        # O_EXCL serializes installers in independent ComfyUI processes too.
        lock = root / 'install.lock'
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f'SAM 2 installation already running. If its process exited, remove {lock}') from exc
        os.close(fd)
        try:
            target = root / 'packages'
            env = dict(os.environ, SAM2_BUILD_CUDA='0')
            packages = [
                f'https://github.com/facebookresearch/sam2/archive/{REVISION}.zip',
                'hydra-core==1.3.2', 'omegaconf==2.3.0',
                'antlr4-python3-runtime==4.9.3', 'iopath==0.1.10',
                'portalocker==3.2.0', 'packaging==25.0', 'PyYAML==6.0.2',
            ]
            if sys.platform == 'win32':
                packages.append('pywin32==311')
            progress('Installing optional SAM 2 packages; existing PyTorch is preserved')
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                '--no-deps', '--no-cache-dir', '--no-build-isolation', '--upgrade', '--target', str(target), *packages],
                env=env, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            checkpoint = root / 'sam2.1_hiera_tiny.pt'
            if not checkpoint.is_file() or digest(checkpoint) != MODEL_SHA256:
                progress('Downloading official SAM 2.1 tiny tracker (156 MB)')
                partial = root / 'checkpoint.partial'
                try:
                    with urllib.request.urlopen(MODEL_URL, timeout=60) as response, partial.open('wb') as dest:
                        while data := response.read(1024 * 1024):
                            dest.write(data)
                    if digest(partial) != MODEL_SHA256:
                        raise RuntimeError('SAM 2 checkpoint checksum mismatch; download discarded')
                    partial.replace(checkpoint)
                finally:
                    partial.unlink(missing_ok=True)
            progress('Checking the optional tracker without using CUDA')
            probe = ('import sys,site; site.addsitedir(sys.argv[1]); sys.path.insert(0,sys.argv[1]); '
                     'import torch, torchvision; from sam2.build_sam import build_sam2_video_predictor; '
                     'm=build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml",'
                     'sys.argv[2],device="cpu",apply_postprocessing=False); '
                     'print("SAM2_CPU_MODEL_READY")')
            subprocess.run([sys.executable, '-c', probe, str(target), str(checkpoint)],
                           check=True, env=env,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            manifest = dict(installer_schema=2, revision=REVISION, model_revision=MODEL_REVISION,
                            model_sha256=MODEL_SHA256, optional=True)
            pending = root / 'ready.partial'
            pending.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            pending.replace(root / 'ready.json')
            progress('Optional SAM 2 tracker ready')
            return root
        finally:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    ensure_installed(progress=print)
