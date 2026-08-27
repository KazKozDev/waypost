#!/bin/zsh
# Interactive chat with the router — by double-clicking from Finder.
# The server must be running (./waypost-server.command).

cd "${0:A:h}" || exit 1
PORT="${ROUTER_PORT:-8080}"
URL="http://127.0.0.1:${PORT}"

printf '\033]0;waypost chat\007'

if ! curl -sf -m 2 "${URL}/health" >/dev/null 2>&1; then
    printf '\n  Server at %s is not responding.\n' "$URL"
    printf '  Start it: ./waypost-server.command\n\n'
    printf '  Press Enter to close this window.'
    read -r _
    exit 1
fi

if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    PY=python3
fi

exec "$PY" -m scripts.chat --url "${URL}/v1" "$@"
