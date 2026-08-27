#!/bin/zsh
# Launch waypost by double-clicking from Finder.
#
# The file lives in the project root and computes its own path, so it
# can be put in the Dock or on the Desktop as an alias — cd will still
# be correct.

cd "${0:A:h}" || exit 1
PORT="${ROUTER_PORT:-8080}"
URL="http://127.0.0.1:${PORT}"

# Window title: with three such windows, "Terminal" cannot tell them apart.
printf '\033]0;waypost :%s\007' "$PORT"

printf '\n  waypost — local router for free LLMs\n'
printf '  %s\n\n' "$(pwd)"

# If already running, kill the old process so we restart cleanly.
LOCK_FILE="var/router.lock"
if [[ -f "$LOCK_FILE" ]]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null | tr -d '[:space:]')
    if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
        printf '  Stopping previous waypost instance (PID %s)...\n' "$LOCK_PID"
        kill -15 "$LOCK_PID" 2>/dev/null
        sleep 0.5
        if kill -0 "$LOCK_PID" 2>/dev/null; then
            kill -9 "$LOCK_PID" 2>/dev/null
        fi
    fi
fi

# Ensure port is free (kill any process holding the port)
PIDS=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null)
if [[ -n "$PIDS" ]]; then
    printf '  Freeing port %s (PID: %s)...\n' "$PORT" "$PIDS"
    kill -15 $PIDS 2>/dev/null
    sleep 0.5
    PIDS_STILL=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null)
    if [[ -n "$PIDS_STILL" ]]; then
        kill -9 $PIDS_STILL 2>/dev/null
        sleep 0.5
    fi
fi

if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
    printf '  .venv not found — using system python3\n\n'
else
    printf '  Python not found. Install it and retry.\n'
    printf '  Press Enter to close this window.'
    read -r _
    exit 1
fi

printf '  Address:   %s\n' "$URL"
printf '  Models:  %s/v1/models\n' "$URL"
printf '  Pricing:    %s/v1/pricing\n' "$URL"
printf '  Chat:    ./waypost-chat.command (in a separate window)\n'
printf '  Press Ctrl-C to stop; window will close.\n\n'

"$PY" -m waypost.cli
CODE=$?

# Code 3 — instance lock: a second server is already up somewhere.
if [[ $CODE -eq 3 ]]; then
    printf '\n  Server already running in another window (instance lock).\n'
elif [[ $CODE -ne 0 ]]; then
    printf '\n  Server exited with code %s. See log above.\n' "$CODE"
    printf '  Press Enter to close this window.'
    read -r _
fi
