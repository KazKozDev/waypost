"""Provider keys in macOS Keychain.

`.env` is convenient, but it is a plaintext secrets file sitting next to
the code. Keychain solves exactly this problem and nothing more: keys
live in the system key store, and the router asks `security` for them by
name.

Key resolution order (registry.Offering.api_keys):
    environment variable → Keychain(service="waypost", account=NAME) →
    numbered variants NAME_2, NAME_3 … (multiple keys per provider)

On non-macOS the module silently returns None, leaving `.env` in play.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading

SERVICE = "waypost"
_cache: dict[str, str | None] = {}
_lock = threading.Lock()


def available() -> bool:
    return sys.platform == "darwin" and shutil.which("security") is not None


def get(account: str, service: str = SERVICE) -> str | None:
    """Read a key. The result is cached per process: `security` is an
    external call, and the hot path calls api_key on every request."""
    cache_key = f"{service}/{account}"
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]
    value: str | None = None
    if available():
        try:
            out = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s",
                    service,
                    "-a",
                    account,
                    "-w",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0:
                value = out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            value = None
    with _lock:
        _cache[cache_key] = value
    return value


def put(account: str, secret: str, service: str = SERVICE) -> bool:
    """Write a key (overwriting any existing one)."""
    if not available():
        return False
    try:
        subprocess.run(
            [
                "security",
                "add-generic-password",
                "-U",
                "-s",
                service,
                "-a",
                account,
                "-w",
                secret,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    with _lock:
        _cache.pop(f"{service}/{account}", None)
    return True


def delete(account: str, service: str = SERVICE) -> bool:
    if not available():
        return False
    try:
        subprocess.run(
            ["security", "delete-generic-password", "-s", service, "-a", account],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    with _lock:
        _cache.pop(f"{service}/{account}", None)
    return True


def clear_cache() -> None:
    with _lock:
        _cache.clear()
