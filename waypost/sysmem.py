"""Physical memory of a local engine, without adding a dependency.

RSS is the wrong number here. mlx-lm keeps model weights in memory the
kernel does not charge to the resident set: `ps` reports ~10 MB for a 27B
model that actually holds 15 GB. macOS `footprint` reports the physical
footprint Activity Monitor shows, and returns in ~0.2s — `vmmap` gives one
more decimal but takes 2.5s, which a page polling every 5s cannot pay.

psutil would be the portable answer, but it is not in the dependency set
and one dashboard number does not justify adding it.

Every failure path returns None. The panel then prints "not reported",
which is the honest output when nothing was measured.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

_TTL_S = 30.0
_LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
_UNITS = {"KB": 1 / 1024 / 1024, "MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}

_cache: dict[str, tuple[float, float | None]] = {}
_lock = threading.Lock()
_total_gb: float | None = None


def available() -> bool:
    return (
        sys.platform == "darwin"
        and shutil.which("lsof") is not None
        and shutil.which("footprint") is not None
    )


def _run(cmd: list[str], timeout: float) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _listener_pid(port: int) -> int | None:
    out = _run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], 3.0)
    if not out:
        return None
    first = out.strip().splitlines()
    try:
        return int(first[0]) if first else None
    except ValueError:
        return None


def _footprint_gb(pid: int) -> float | None:
    out = _run(["footprint", "-p", str(pid)], 5.0)
    if not out:
        return None
    m = re.search(r"Footprint:\s*([\d.]+)\s*(KB|MB|GB|TB)", out)
    if not m:
        return None
    try:
        return float(m.group(1)) * _UNITS[m.group(2)]
    except (ValueError, KeyError):
        return None


def engine_footprint_gb(base_url: str) -> float | None:
    """Memory held by whatever listens on this local engine's port.

    Only loopback URLs are probed — a remote base_url is somebody else's
    machine and there is nothing here to measure.
    """
    if not available():
        return None
    parsed = urlparse(base_url)
    if parsed.hostname not in _LOOPBACK or not parsed.port:
        return None

    now = time.monotonic()
    with _lock:
        cached = _cache.get(base_url)
        if cached and now - cached[0] < _TTL_S:
            return cached[1]

    pid = _listener_pid(parsed.port)
    value = _footprint_gb(pid) if pid else None
    with _lock:
        _cache[base_url] = (now, value)
    return value


def total_ram_gb() -> float | None:
    """Installed RAM. Constant for the life of the process."""
    global _total_gb
    if _total_gb is not None:
        return _total_gb
    if sys.platform != "darwin" or shutil.which("sysctl") is None:
        return None
    out = _run(["sysctl", "-n", "hw.memsize"], 3.0)
    if not out:
        return None
    try:
        _total_gb = int(out.strip()) / (1024**3)
    except ValueError:
        return None
    return _total_gb
