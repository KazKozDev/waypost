"""Single process instance (AR-06 / risk §10).

A second process against the same SQLite would split quota accounting:
each reserves from its own in-memory snapshot and overwrites the other
on write. A flock file lock is enough — cross-node coordination is not
needed (D4).
"""
from __future__ import annotations

import errno
import fcntl
from pathlib import Path


class InstanceLockError(RuntimeError):
    """A second instance against the same var/ directory is already running."""


class InstanceLock:
    """Exclusive flock on a marker file. Held until close().

    The PID inside is for human diagnostics, not for locking: the kernel
    releases the flock only when the fd is closed, so a crashed process
    leaves no lock behind, and a held file is clearly visible.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._file = None
        self._fd: int | None = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self._path, "a+", encoding="utf-8")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            f.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                holder = self._path.read_text().strip() or "?"
                raise InstanceLockError(
                    f"waypost is already running (lock {self._path}, pid {holder})"
                ) from exc
            raise
        import os

        f.seek(0)
        f.truncate()
        f.write(f"{os.getpid()}\n")
        f.flush()
        # The file object is intentionally kept open: the lock lives as
        # long as the fd is open. Garbage collection would close it and
        # release the lock.
        self._file = f
        self._fd = f.fileno()

    def release(self) -> None:
        if self._file is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                self._file.close()
                self._file = None
                self._fd = None
                try:
                    self._path.unlink()
                except OSError:
                    pass

    def __enter__(self) -> "InstanceLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
