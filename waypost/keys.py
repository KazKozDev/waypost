"""Secure API key storage and management for Waypost providers.

Handles safe persistence to macOS Keychain and local project .env file (chmod 0600),
updating active in-memory environment variables, and masking keys for UI display.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import keychain

_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = _ROOT / ".env"

# Curated list of providers with FREE models or FREE rate-limited tiers
PROVIDER_DEFS: list[dict[str, Any]] = [
    {
        "id": "groq",
        "name": "Groq",
        "env_var": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "description": "Ultra-fast LPU inference (Kimi, Compound, Llama 3.3)",
        "doc_url": "https://console.groq.com/keys",
    },
    {
        "id": "gemini",
        "name": "Google Gemini",
        "env_var": "GEMINI_API_KEY",
        "alt_env_vars": ["GOOGLE_API_KEY"],
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "description": "Google AI Studio free tier (Gemini 2.5 / 3.5 / 3.7 Flash)",
        "doc_url": "https://aistudio.google.com/app/apikey",
    },
    {
        "id": "nvidia",
        "name": "NVIDIA NIM",
        "env_var": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "description": "Free cloud tier for Nemotron, Llama, Qwen models",
        "doc_url": "https://build.nvidia.com",
    },
    {
        "id": "cerebras",
        "name": "Cerebras",
        "env_var": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "description": "Wafer-scale high-throughput free tier (GPT-OSS, Llama, Gemma)",
        "doc_url": "https://cloud.cerebras.ai",
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "env_var": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "description": "Free model catalog (Gemma, Minimax, Nemotron, GLM :free)",
        "doc_url": "https://openrouter.ai/keys",
    },
    {
        "id": "mistral",
        "name": "Mistral AI",
        "env_var": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "description": "Free experimentation tier (Codestral, Mistral Small & Large)",
        "doc_url": "https://console.mistral.ai/api-keys",
    },
    {
        "id": "siliconflow",
        "name": "SiliconFlow",
        "env_var": "SILICONFLOW_API_KEY",
        "base_url": "https://api.siliconflow.cn/v1",
        "description": "Free cloud inference for DeepSeek-R1 and Qwen models",
        "doc_url": "https://cloud.siliconflow.cn",
    },
    {
        "id": "zhipu",
        "name": "Zhipu AI (GLM)",
        "env_var": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "description": "Free tier for GLM-4 Flash with 200K context window",
        "doc_url": "https://open.bigmodel.cn",
    },
    {
        "id": "ollama",
        "name": "Ollama Cloud",
        "env_var": "OLLAMA_API_KEY",
        "base_url": "https://ollama.com/v1",
        "description": "Cloud hosting with free quotas for cloud-prefixed models",
        "doc_url": "https://ollama.com",
    },
]


def mask_key(val: str | None) -> str:
    """Returns a masked key for safe UI display (e.g. gsk_•••••••• or ••••3aF9)."""
    if not val:
        return ""
    s = val.strip()
    if len(s) <= 8:
        return "••••••••"
    prefix = s[:4]
    suffix = s[-4:]
    return f"{prefix}••••••••{suffix}"


def get_active_key(env_var: str, alt_env_vars: list[str] | None = None) -> str | None:
    """Reads key value from environment, Keychain, or .env file."""
    candidates = [env_var] + (alt_env_vars or [])
    for var in candidates:
        v = os.environ.get(var)
        if v:
            return v
        kc_val = keychain.get(var)
        if kc_val:
            return kc_val
    return None


def read_env_file() -> dict[str, str]:
    """Reads key-value pairs from .env."""
    if not ENV_FILE.exists():
        return {}
    out: dict[str, str] = {}
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip("'\"")
                if k:
                    out[k] = v
    except Exception:
        pass
    return out


def write_env_file(data: dict[str, str]) -> None:
    """Writes key-value pairs to .env with 0600 permissions."""
    lines = [
        "# Waypost secure configuration",
        "# Automatically managed — keep confidential",
    ]
    for k, v in sorted(data.items()):
        if v:
            lines.append(f"{k}={v}")
    content = "\n".join(lines) + "\n"
    ENV_FILE.write_text(content, encoding="utf-8")
    try:
        os.chmod(ENV_FILE, 0o600)
    except Exception:
        pass


def save_key(env_var: str, key_value: str) -> bool:
    """Stores key in Keychain, updates .env with 0600 perms, and sets
    os.environ. Returns whether the Keychain write succeeded — the caller
    reports it instead of promising "saved to Keychain" unconditionally.
    """
    clean_val = key_value.strip()
    if not clean_val:
        return False

    # 1. Update in-memory environment
    os.environ[env_var] = clean_val

    # 2. Save in macOS Keychain if available
    in_keychain = keychain.put(env_var, clean_val)

    # 3. Save to local .env
    env_data = read_env_file()
    env_data[env_var] = clean_val
    write_env_file(env_data)
    return in_keychain


def delete_key(env_var: str, alt_env_vars: list[str] | None = None) -> None:
    """Removes key from Keychain, .env and os.environ."""
    candidates = [env_var] + (alt_env_vars or [])
    env_data = read_env_file()

    for var in candidates:
        os.environ.pop(var, None)
        keychain.delete(var)
        env_data.pop(var, None)

    write_env_file(env_data)
