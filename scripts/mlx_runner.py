#!/usr/bin/env python3
"""MLX LM Server Runner with Memory Constraints for 32GB Mac Studio.

Enforces:
- 4 GB cache limit
- 22 GB memory limit
- Default model: mlx-community/Qwen2.5-Coder-32B-Instruct-4bit
- Port: 8081 (to avoid conflict with Waypost on 8080)
- Per-request dynamic thinking support
"""
from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen3.6-27B-4bit")
DEFAULT_PORT = int(os.environ.get("MLX_PORT", "8081"))
DEFAULT_HOST = os.environ.get("MLX_HOST", "127.0.0.1")

CACHE_LIMIT_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
MEMORY_LIMIT_BYTES = 22 * 1024 * 1024 * 1024  # 22 GB


def apply_memory_limits() -> None:
    """Set hard Metal memory limits on Apple Silicon unified memory."""
    try:
        import mlx.core as mx

        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(CACHE_LIMIT_BYTES)
        elif hasattr(mx.metal, "set_cache_limit"):
            mx.metal.set_cache_limit(CACHE_LIMIT_BYTES)

        if hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(MEMORY_LIMIT_BYTES)
        elif hasattr(mx.metal, "set_memory_limit"):
            mx.metal.set_memory_limit(MEMORY_LIMIT_BYTES)

        print(
            f"[mlx_runner] Applied Metal memory caps: cache={CACHE_LIMIT_BYTES / (1024**3):.1f}GB, "
            f"memory={MEMORY_LIMIT_BYTES / (1024**3):.1f}GB"
        )
    except ImportError:
        print(
            "[mlx_runner] Warning: mlx package not installed or not running on Apple Silicon. "
            "Proceeding without Metal memory limits."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Waypost MLX LM Server Runner")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Hugging Face model ID or path"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface to bind")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to listen on"
    )
    parser.add_argument(
        "--adapter-path", default=None, help="Optional LoRA adapter path"
    )
    parser.add_argument(
        "--chat-template-args",
        default=os.environ.get(
            "MLX_CHAT_TEMPLATE_ARGS", '{"enable_thinking":false}'
        ),
        help="JSON string of chat template args, e.g. '{\"enable_thinking\":false}'",
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=int(os.environ.get("MLX_PROMPT_CACHE_SIZE", "2")),
        help="Maximum number of distinct KV caches in prompt cache",
    )
    parser.add_argument(
        "--prompt-cache-bytes",
        type=int,
        default=int(os.environ.get("MLX_PROMPT_CACHE_BYTES", str(1024 * 1024 * 1024))),
        help="Maximum bytes for prompt cache (default: 1GB)",
    )
    args = parser.parse_args()

    apply_memory_limits()

    # mlx_lm.server invocation:
    try:
        from mlx_lm import server
    except ImportError:
        print("[mlx_runner] Error: mlx_lm is not installed. Run: pip install mlx-lm")
        sys.exit(1)

    sys.argv = [
        "mlx_lm.server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--prompt-cache-size",
        str(args.prompt_cache_size),
        "--prompt-cache-bytes",
        str(args.prompt_cache_bytes),
        "--decode-concurrency",
        "1",
        "--prompt-concurrency",
        "1",
    ]
    if args.adapter_path:
        sys.argv.extend(["--adapter-path", args.adapter_path])
    if args.chat_template_args:
        sys.argv.extend(["--chat-template-args", args.chat_template_args])

    print(
        f"[mlx_runner] Starting mlx_lm server for {args.model} on {args.host}:{args.port}..."
    )
    server.main()


if __name__ == "__main__":
    main()
