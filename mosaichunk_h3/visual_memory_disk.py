"""Execution-scoped disk storage for visual observations, not RAVEN KV tensors.

Each run holds an interprocess lock. Normal completion, errors and interruptions
remove its images; a later startup reaps unlocked abandoned session directories.
Neither user outputs nor model files are ever used as cache directories.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import uuid

LOG = logging.getLogger(__name__)
_SESSION_NAME = re.compile(r"^run-[0-9a-f]{32}$")
_ALLOWED_FORMATS = {"png", "webp", "jpg", "jpeg"}


def _lock_file(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class VisualRecord:
    step: int
    frame: int
    filename: str
    size: int


class VisualMemoryDisk:
    """Bounded, execution-local storage for already encoded preview images.

    The caller chooses which images deserve storage. The session must be closed
    or used with a context manager around the sampler, including cancellation.
    """

    def __init__(self, root=None, *, max_bytes=2 * 1024**3):
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.root = Path(root or Path(tempfile.gettempdir()) / "mosaichunk-h3-visual-memory")
        self.root.mkdir(parents=True, exist_ok=True)
        self.cleanup_abandoned(self.root)
        self.directory = self.root / ("run-" + uuid.uuid4().hex)
        self.directory.mkdir(mode=0o700)
        self._lock_handle = (self.directory / ".lock").open("w+b")
        self._lock_handle.write(b"0")
        self._lock_handle.flush()
        try:
            _lock_file(self._lock_handle)
        except BaseException:
            self._lock_handle.close()
            shutil.rmtree(self.directory)
            raise
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self.records: list[VisualRecord] = []
        self._mutex = threading.Lock()
        self._closed = False

    @staticmethod
    def cleanup_abandoned(root):
        """Only remove UUID sessions that no running process has locked."""
        root = Path(root)
        if not root.is_dir():
            return 0
        removed = 0
        for child in root.iterdir():
            if not _SESSION_NAME.fullmatch(child.name) or child.is_symlink() or not child.is_dir():
                continue
            lock_path = child / ".lock"
            if lock_path.is_symlink() or not lock_path.is_file():
                continue
            try:
                with lock_path.open("r+b") as lock:
                    try:
                        _lock_file(lock)
                    except (OSError, BlockingIOError):
                        continue
                    _unlock_file(lock)
                # Windows cannot remove a directory while its lock file is open.
                shutil.rmtree(child)
                removed += 1
            except OSError as exc:
                LOG.warning("Visual-memory orphan cleanup skipped %s: %s", child, exc)
        return removed

    def save(self, *, step: int, frame: int, data: bytes, fmt: str = "png") -> VisualRecord:
        if step < 0 or frame < 0:
            raise ValueError("step and frame must be nonnegative")
        if fmt not in _ALLOWED_FORMATS:
            raise ValueError("unsupported visual-memory image format")
        if not isinstance(data, bytes) or not data:
            raise ValueError("data must be nonempty encoded image bytes")
        with self._mutex:
            if self._closed:
                raise RuntimeError("visual-memory session is closed")
            if self.used_bytes + len(data) > self.max_bytes:
                raise MemoryError("visual-memory disk budget exceeded")
            name = f"s{step:05d}-f{frame:06d}-{uuid.uuid4().hex}.{fmt}"
            path = self.directory / name
            partial = path.with_suffix(path.suffix + ".part")
            try:
                with partial.open("xb") as fh:
                    fh.write(data)
                os.replace(partial, path)
            finally:
                partial.unlink(missing_ok=True)
            record = VisualRecord(step, frame, name, len(data))
            self.records.append(record)
            self.used_bytes += len(data)
            return record

    def read(self, record: VisualRecord) -> bytes:
        with self._mutex:
            if self._closed:
                raise RuntimeError("visual-memory session is closed")
            if record not in self.records:
                raise ValueError("record belongs to another session")
            return (self.directory / record.filename).read_bytes()

    def delete(self, record: VisualRecord):
        """Evict one observation and reclaim its disk budget immediately."""
        with self._mutex:
            if self._closed:
                raise RuntimeError("visual-memory session is closed")
            if record not in self.records:
                raise ValueError("record belongs to another session")
            (self.directory / record.filename).unlink()
            self.records.remove(record)
            self.used_bytes -= record.size

    def close(self):
        with self._mutex:
            if self._closed:
                return
            self._closed = True
            try:
                _unlock_file(self._lock_handle)
            finally:
                self._lock_handle.close()
                try:
                    shutil.rmtree(self.directory)
                except OSError as exc:
                    LOG.warning("Visual-memory cleanup deferred for %s: %s", self.directory, exc)
                self.records.clear()
                self.used_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _tb):
        self.close()
        return False
