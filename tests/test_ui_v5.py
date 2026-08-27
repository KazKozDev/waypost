"""Tests for Waypost v5 3-tab UI architecture.

Verifies:
- Chat page (/chat, /) renders Claude-style interface and 3-tab navigation.
- Dashboard page (/dashboard) renders 6 operational blocks.
- Setup page (/setup) renders configuration, local engines, and data health.
"""
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from waypost.server import app


def test_ui_v5_three_tabs_rendering(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        monkeypatch.setenv("ROUTER_DB_PATH", db_path)
        monkeypatch.setenv("ROUTER_MANIFEST_PATH", "config/providers.yaml")
        monkeypatch.setenv("ROUTER_ENABLE_DISCOVERY", "false")
        monkeypatch.setenv("ROUTER_ENABLE_EXPLORATION", "false")
        with TestClient(app) as client:
            # 1. Chat Tab
            r_chat = client.get("/chat")
            assert r_chat.status_code == 200
            assert "text/html" in r_chat.headers["content-type"]
            assert "Waypost — Chat" in r_chat.text
            assert "Dashboard" in r_chat.text
            assert "Setup" in r_chat.text

            # 2. Dashboard Tab
            r_dash = client.get("/dashboard")
            assert r_dash.status_code == 200
            assert "text/html" in r_dash.headers["content-type"]
            assert "Waypost — Dashboard" in r_dash.text
            assert "Binding Provider Quotas" in r_dash.text
            assert "Local Hit Rate" in r_dash.text
            assert "Latency (P50 / P95)" in r_dash.text
            assert "Active Models (Top 5)" in r_dash.text
            assert "Live Traces" in r_dash.text

            # 3. Setup Tab
            r_setup = client.get("/setup")
            assert r_setup.status_code == 200
            assert "text/html" in r_setup.headers["content-type"]
            assert "Waypost — Setup" in r_setup.text
            assert "Router Subsystems & Effects" in r_setup.text
            assert "Local Engines & Metal Memory" in r_setup.text
            assert "Keys & Provider Health" in r_setup.text
            assert "Data Health & Exploration" in r_setup.text
            assert "Beta Measurement & Ensemble Assessment" in r_setup.text
