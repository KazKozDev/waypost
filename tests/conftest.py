"""Isolate the suite from whatever is in the developer's working copy.

Settings read `.env`, and `load_env()` copies it into the environment. So
a test asserting a feature's behaviour was really asserting that the
person running it had not switched that feature off — which is how
test_pii_forces_strict_without_explicit_flag came to fail on one machine
and pass in CI for weeks. The failure was real and the code was fine: a
local `.env` carried ROUTER_ENABLE_PII=false.

Pointing the env file at a path that does not exist makes every test
start from the declared defaults, and anything a test needs beyond them
it sets itself, visibly.
"""
import os

import pytest

os.environ["ROUTER_ENV_FILE"] = "/nonexistent/waypost-tests.env"


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Strip ROUTER_* out of the environment for the duration of a test.

    Both halves matter: the env file is ignored above, and the variables
    it may already have exported into this process are removed here.
    """
    for name in list(os.environ):
        if name.startswith("ROUTER_") and name != "ROUTER_ENV_FILE":
            monkeypatch.delenv(name, raising=False)
    yield
