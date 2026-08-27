"""UI Components and 3-Tab HTML Templates for Waypost v5.

Implements Part II of Waypost Spec v5:
- Tab 1: Chat (/chat, /)
- Tab 2: Dashboard (/dashboard, /v1/stats)
- Tab 3: Setup (/setup, /v1/models, /v1/probe, /v1/pricing)
"""
from __future__ import annotations

import html
from typing import Any

FAVICON = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%232b2620'/%3E%3Cpath d='M8 10h16M8 16h16M8 22h10' stroke='%23fbfaf7' stroke-width='2.5' stroke-linecap='round'/%3E%3C/svg%3E"

LOGO = """<svg width="22" height="22" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="border-radius:5px;flex-shrink:0;">
  <rect width="32" height="32" rx="6" fill="#2b2620"/>
  <path d="M8 10h16M8 16h16M8 22h10" stroke="#fbfaf7" stroke-width="2.5" stroke-linecap="round"/>
</svg>"""


def theme_css() -> str:
    return """
:root {
  --bg: #f7f6f2;
  --bg-subtle: #eeede8;
  --card: #ffffff;
  --border: #e4e2d8;
  --border-light: #eceae1;
  --text: #21201c;
  --text-secondary: #636159;
  --text-muted: #8e8b82;
  --accent: #2d2c28;
  --accent-fg: #fbfaf7;
  --terracotta: #c85a32;
  --green: #2a7a4c;
  --green-bg: #edf7f0;
  --amber: #b45309;
  --amber-bg: #fef3c7;
  --red: #c53030;
  --red-bg: #fee2e2;
  --blue: #2563eb;
  --blue-bg: #eff6ff;
  --radius-sm: 6px;
  --radius-md: 10px;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-mono: "SF Mono", Monaco, Menlo, Consolas, monospace;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 13px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

/* Nav Header */
.top-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 24px;
  background: var(--card);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 100;
}
.brand-group {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
  color: var(--text);
}
.brand-title {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: -0.01em;
}
.brand-tag {
  font-size: 11px;
  background: var(--bg-subtle);
  color: var(--text-secondary);
  padding: 2px 7px;
  border-radius: 4px;
  font-weight: 600;
}
.nav-links {
  display: flex;
  gap: 4px;
  background: var(--bg-subtle);
  padding: 3px;
  border-radius: var(--radius-sm);
}
.nav-link {
  padding: 6px 14px;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
  text-decoration: none;
  border-radius: 4px;
  transition: all 0.15s ease;
}
.nav-link:hover {
  color: var(--text);
}
.nav-link.active {
  background: var(--card);
  color: var(--text);
  font-weight: 600;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

/* Common Layout */
.page-container {
  max-width: 1160px;
  margin: 0 auto;
  padding: 24px 20px 60px;
}
.section-title {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
  margin-bottom: 12px;
}
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 18px 20px;
  margin-bottom: 20px;
}
.card-plain {
  padding: 16px 0;
  margin-bottom: 20px;
}

/* Badges & Tables */
.badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 7px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
}
.badge-pass { background: var(--green-bg); color: var(--green); }
.badge-warn { background: var(--amber-bg); color: var(--amber); }
.badge-fail { background: var(--red-bg); color: var(--red); }
.badge-info { background: var(--blue-bg); color: var(--blue); }
.badge-neutral { background: var(--bg-subtle); color: var(--text-secondary); }

table.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.data-table th {
  text-align: left;
  padding: 8px 12px;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--border);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
table.data-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border-light);
}
table.data-table tr:last-child td {
  border-bottom: none;
}
.mono {
  font-family: var(--font-mono);
  font-size: 12px;
}

/* Accordion Disclosure Cards */
details.accordion-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  margin-bottom: 20px;
  overflow: hidden;
  transition: all 0.15s ease;
}
details.accordion-card summary {
  padding: 16px 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  user-select: none;
  list-style: none;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
  transition: background 0.15s ease;
}
details.accordion-card summary::-webkit-details-marker {
  display: none;
}
details.accordion-card summary:hover {
  color: var(--text);
  background: var(--bg-subtle);
}
.accordion-arrow {
  display: inline-block;
  transition: transform 0.2s ease;
  font-size: 11px;
}
details[open] .accordion-arrow {
  transform: rotate(180deg);
}
.accordion-content {
  padding: 0 20px 18px;
  border-top: 1px solid var(--border-light);
}
"""


def nav_header(active: str = "chat") -> str:
    return f"""<header class="top-header claude-header">
  <a href="/chat" class="brand-group brand">
    {LOGO}
    <span class="brand-title brand-name">Waypost</span>
    <span class="brand-tag brand-badge">v5</span>
  </a>
  <nav class="nav-links nav-tabs">
    <a href="/chat" class="nav-link nav-tab {'active' if active == 'chat' else ''}">Chat</a>
    <a href="/dashboard" class="nav-link nav-tab {'active' if active == 'dashboard' else ''}">Dashboard</a>
    <a href="/setup" class="nav-link nav-tab {'active' if active == 'setup' else ''}">Setup</a>
  </nav>
</header>"""


def render_dashboard_html(data: dict[str, Any]) -> str:
    """Renders 6-block operational Dashboard per Spec v5 Section 2.1."""
    summary = data.get("summary", {})
    quotas = data.get("quotas", [])
    latencies = data.get("latencies", {})
    escalations = data.get("escalations", {})
    top_models = data.get("top_models", [])
    recent_traces = data.get("recent_traces", [])
    coverage = data.get("outcome_coverage", 1.0)

    # Block 1: Status Bar
    total_reqs = summary.get("total_requests", 0)
    esc_rate = summary.get("escalation_rate_pct", 0.0)
    urgent_warning = data.get("urgent_warning", "All systems operational")

    coverage_alert = ""
    if coverage < 0.70:
        coverage_alert = f'<span class="badge badge-warn" style="margin-left:12px">outcome coverage low: {coverage * 100:.0f}%</span>'

    status_bar = f"""
    <div class="card" style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px;margin-bottom:16px;">
      <div style="display:flex;align-items:center;gap:16px;">
        <span style="font-weight:600">Requests: <span class="mono">{total_reqs}</span></span>
        <span style="color:var(--text-secondary)">·</span>
        <span style="font-weight:600">Escalation Rate: <span class="mono">{esc_rate:.1f}%</span></span>
        {coverage_alert}
      </div>
      <div>
        <span class="badge {'badge-pass' if 'operational' in urgent_warning.lower() else 'badge-warn'}">{urgent_warning}</span>
      </div>
    </div>
    """

    # Block 2: Funnel
    funnel = summary.get("funnel", {})
    in_cnt = funnel.get("in", total_reqs)
    cache_cnt = funnel.get("cache", 0)
    local_cnt = funnel.get("local", 0)
    cloud_cnt = funnel.get("cloud", 0)
    esc_cnt = funnel.get("escalated", 0)
    fail_cnt = funnel.get("failed", 0)
    local_hit_rate = (local_cnt + cache_cnt) / max(1, in_cnt) * 100

    funnel_block = f"""
    <div class="card" style="display:flex;justify-content:space-between;align-items:center;padding:18px 24px;">
      <div style="font-size:14px;display:flex;flex-wrap:wrap;gap:12px;align-items:center;">
        <b>IN</b> <span class="mono">{in_cnt}</span>
        <span style="color:var(--text-muted)">→</span>
        <b>CACHE</b> <span class="mono">{cache_cnt}</span>
        <span style="color:var(--text-muted)">→</span>
        <b>LOCAL</b> <span class="mono">{local_cnt}</span>
        <span style="color:var(--text-muted)">→</span>
        <b>CLOUD</b> <span class="mono">{cloud_cnt}</span>
        <span style="color:var(--text-muted)">→</span>
        <b>ESCALATED</b> <span class="mono" style="color:var(--amber)">{esc_cnt}</span>
        <span style="color:var(--text-muted)">→</span>
        <b>FAILED</b> <span class="mono" style="color:var(--red)">{fail_cnt}</span>
      </div>
      <div style="text-align:right;border-left:1px solid var(--border);padding-left:24px;">
        <div style="font-size:11px;text-transform:uppercase;color:var(--text-secondary);font-weight:600;">Local Hit Rate</div>
        <div style="font-size:24px;font-weight:800;color:var(--green);">{local_hit_rate:.0f}%</div>
      </div>
    </div>
    """

    # Block 3: Quota rows
    quota_rows_html = []
    for q in quotas:
        name = q.get("provider", "—")
        pct = q.get("used_pct", 0)
        burn_rate = q.get("burn_rate", "—")
        binding = q.get("binding", "requests")
        eta = q.get("exhaustion_eta", "will last until reset")
        is_warn = pct > 80
        bar_fill = int(pct / 10)
        bar_str = "█" * bar_fill + "░" * (10 - bar_fill)
        quota_rows_html.append(
            f"""
        <tr>
          <td style="font-weight:600">{html.escape(name)}</td>
          <td class="mono">{bar_str} {pct:.0f}%</td>
          <td class="mono">{burn_rate}</td>
          <td><span class="badge badge-neutral">binding: {binding}</span></td>
          <td style="color:{'var(--amber)' if is_warn else 'var(--text-secondary)'};font-weight:{'600' if is_warn else 'normal'}">
            → {html.escape(eta)} {'⚠️' if is_warn else ''}
          </td>
        </tr>
        """
        )
    quota_table = f"""
    <details class="accordion-card">
      <summary>
        <span>Binding Provider Quotas ({len(quotas)})</span>
        <span class="accordion-arrow">▼</span>
      </summary>
      <div class="accordion-content">
        <table class="data-table" style="margin-top:8px;">
          <thead>
            <tr>
              <th>Provider</th><th>Usage</th><th>Burn Rate</th><th>Binding Limit</th><th>Exhaustion Forecast</th>
            </tr>
          </thead>
          <tbody>
            {''.join(quota_rows_html) if quota_rows_html else '<tr><td colspan="5" style="color:var(--text-muted)">No active quotas configured.</td></tr>'}
          </tbody>
        </table>
      </div>
    </details>
    """

    # Block 4: Latency & Escalations
    loc_p50 = latencies.get("local_p50", 0)
    loc_p95 = latencies.get("local_p95", 0)
    cld_p50 = latencies.get("cloud_p50", 0)
    cld_p95 = latencies.get("cloud_p95", 0)

    esc_reasons_html = []
    for r_name, r_cnt in escalations.items():
        esc_reasons_html.append(
            f"""
        <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--border-light)">
          <span>{html.escape(r_name)}</span>
          <span class="mono" style="font-weight:600">{r_cnt}</span>
        </div>
        """
        )

    lat_esc_block = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
      <div class="card">
        <div class="section-title">Latency (P50 / P95)</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px;">
          <div style="background:var(--bg-subtle);padding:12px;border-radius:var(--radius-sm);">
            <div style="font-size:11px;color:var(--text-secondary)">Local P50 / P95</div>
            <div class="mono" style="font-size:16px;font-weight:700">{loc_p50}ms / {loc_p95}ms</div>
          </div>
          <div style="background:var(--bg-subtle);padding:12px;border-radius:var(--radius-sm);">
            <div style="font-size:11px;color:var(--text-secondary)">Cloud P50 / P95</div>
            <div class="mono" style="font-size:16px;font-weight:700">{cld_p50}ms / {cld_p95}ms</div>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="section-title">Escalation Reasons</div>
        <div style="margin-top:8px;">
          {''.join(esc_reasons_html) if esc_reasons_html else '<div style="color:var(--text-muted)">No escalations recorded.</div>'}
        </div>
      </div>
    </div>
    """

    # Block 5: Top-5 Models Table
    model_rows_html = []
    for m in top_models[:5]:
        m_name = m.get("model", "—")
        backend = m.get("backend", "cloud")
        thk = "yes" if m.get("thinking_supported") else "no"
        primary = m.get("primary_calls", 0)
        esc = m.get("escalation_calls", 0)
        success = m.get("success_rate_pct", 100.0)
        p50 = m.get("p50_ms", 0)
        p95 = m.get("p95_ms", 0)
        tokens = m.get("total_tokens", 0)

        model_rows_html.append(
            f"""
        <tr>
          <td style="font-weight:600">{html.escape(m_name)}</td>
          <td><span class="badge badge-neutral">{html.escape(backend)}</span></td>
          <td>{thk}</td>
          <td class="mono">{primary}</td>
          <td class="mono">{esc}</td>
          <td class="mono" style="color:var(--green)">{success:.0f}%</td>
          <td class="mono">{p50}ms</td>
          <td class="mono">{p95}ms</td>
          <td class="mono">{tokens:,}</td>
        </tr>
        """
        )

    models_block = f"""
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div class="section-title" style="margin:0">Active Models (Top 5)</div>
        <a href="/setup" style="font-size:12px;color:var(--text-secondary);text-decoration:none">View all in Setup →</a>
      </div>
      <table class="data-table">
        <thead>
          <tr>
            <th>Model</th><th>Backend</th><th>Thinking</th><th>Primary</th><th>Escalated</th><th>Success</th><th>P50</th><th>P95</th><th>Tokens</th>
          </tr>
        </thead>
        <tbody>
          {''.join(model_rows_html) if model_rows_html else '<tr><td colspan="9" style="color:var(--text-muted)">No model execution data.</td></tr>'}
        </tbody>
      </table>
    </div>
    """

    # Block 6: Live Trace (15 rows)
    trace_rows_html = []
    for t in recent_traces[:15]:
        ts_str = t.get("time_str", "")
        req_id = t.get("request_id", "")
        path_str = t.get("path_summary", "cache miss → local → accepted")
        duration = t.get("duration_s", 0.0)
        is_explore = t.get("is_exploration", False)

        trace_rows_html.append(
            f"""
        <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border-light);font-size:12px;">
          <div style="display:flex;gap:12px;align-items:center;">
            <span class="mono" style="color:var(--text-muted)">{ts_str}</span>
            <span class="mono" style="font-weight:600">{req_id}</span>
            <span>{html.escape(path_str)}</span>
            {'<span class="badge badge-info">explore</span>' if is_explore else ''}
          </div>
          <div class="mono">{duration:.2f}s</div>
        </div>
        """
        )

    traces_block = f"""
    <div class="card">
      <div class="section-title">Live Traces (Last 15)</div>
      <div style="margin-top:6px;">
        {''.join(trace_rows_html) if trace_rows_html else '<div style="color:var(--text-muted)">No recent traces.</div>'}
      </div>
    </div>
    """

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Waypost — Dashboard</title>
<link rel=icon href="{FAVICON}">
<style>{theme_css()}</style>
</head>
<body>
{nav_header('dashboard')}
<div class="page-container">
  {status_bar}
  {funnel_block}
  {lat_esc_block}
  {models_block}
  {traces_block}
  {quota_table}
</div>
</body></html>"""


def render_setup_html(data: dict[str, Any]) -> str:
    """Renders comprehensive Setup tab per Spec v5 Section 2.2."""
    subsystems = data.get("subsystems", [])
    engines = data.get("engines", [])
    keys_health = data.get("keys_health", [])
    data_health = data.get("data_health", {})
    beta_metric = data.get("beta_metric", {})
    pricing = data.get("pricing", {})

    # Subsystems table
    sub_rows = []
    for s in subsystems:
        st = s.get("status", "off")
        eff = s.get("effect", "—")
        badge = (
            "badge-pass"
            if st == "on"
            else ("badge-warn" if "0" in st else "badge-neutral")
        )
        sub_rows.append(
            f"""
        <tr>
          <td style="font-weight:600">{html.escape(s.get('name', ''))}</td>
          <td><span class="badge {badge}">{html.escape(st)}</span></td>
          <td style="color:var(--text-secondary)">{html.escape(eff)}</td>
        </tr>
        """
        )

    # Local engines & memory
    engine_cards = []
    for e in engines:
        engine_cards.append(
            f"""
        <div style="background:var(--bg-subtle);padding:14px;border-radius:var(--radius-sm)">
          <div style="font-weight:700;margin-bottom:4px">{html.escape(e.get('name', ''))}</div>
          <div style="font-size:12px;color:var(--text-secondary);margin-bottom:2px">Model: <b>{html.escape(e.get('model', ''))}</b></div>
          <div class="mono" style="font-size:12px">Memory: {e.get('mem_used_gb', 0):.1f} GB / {e.get('mem_limit_gb', 22):.1f} GB cap</div>
          <div style="font-size:11px;color:var(--text-muted);margin-top:4px">Warmup status: {html.escape(e.get('warmup_status', 'ready'))}</div>
        </div>
        """
        )

    # Data health
    cov = data_health.get("outcome_coverage", 1.0)
    attempts_cnt = data_health.get("attempt_log_count", 0)
    exp_rate = data_health.get("exploration_rate_pct", 10.0)

    # Beta
    p_best = beta_metric.get("p_best", 0.85)
    beta = beta_metric.get("beta", 0.05)
    ceil_beta = 1.0 - beta
    ensemble_verdict = beta_metric.get(
        "verdict", "Single best model sufficient; ensemble deferred."
    )

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Waypost — Setup</title>
<link rel=icon href="{FAVICON}">
<style>{theme_css()}</style>
</head>
<body>
{nav_header('setup')}
<div class="page-container">

  <!-- Section 1: Subsystems -->
  <div class="card">
    <div class="section-title">Router Subsystems & Effects</div>
    <table class="data-table">
      <thead><tr><th>Subsystem</th><th>Status</th><th>Effect / Diagnostics</th></tr></thead>
      <tbody>{''.join(sub_rows)}</tbody>
    </table>
  </div>

  <!-- Section 2: Local Engines & Memory -->
  <div class="card">
    <div class="section-title">Local Engines & Metal Memory (32 GB Machine)</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:16px;margin-top:10px;">
      {''.join(engine_cards)}
    </div>
  </div>

  <!-- Section 3: Keys & Providers -->
  <div class="card">
    <div class="section-title">Keys & Provider Health</div>
    <table class="data-table">
      <thead><tr><th>Provider</th><th>Base URL</th><th>Key Status</th><th>Active Models</th></tr></thead>
      <tbody>
        {''.join(f'<tr><td style="font-weight:600">{html.escape(k.get("provider", ""))}</td><td class="mono">{html.escape(k.get("base_url", ""))}</td><td><span class="badge {k.get("key_badge", "badge-pass")}">{html.escape(k.get("key_status", "valid"))}</span></td><td class="mono">{k.get("model_count", 0)}</td></tr>' for k in keys_health)}
      </tbody>
    </table>
  </div>

  <!-- Section 4: Data Health & Beta Measurement -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
    <div class="card">
      <div class="section-title">Data Health & Exploration</div>
      <div style="display:flex;flex-direction:column;gap:8px;margin-top:8px;">
        <div style="display:flex;justify-content:space-between"><span>Outcome Coverage:</span><b class="mono">{cov * 100:.1f}%</b></div>
        <div style="display:flex;justify-content:space-between"><span>Attempt Log Rows:</span><b class="mono">{attempts_cnt:,}</b></div>
        <div style="display:flex;justify-content:space-between"><span>Exploration Target Rate:</span><b class="mono">{exp_rate:.1f}%</b></div>
      </div>
    </div>
    <div class="card">
      <div class="section-title">Beta Measurement & Ensemble Assessment</div>
      <div style="display:flex;flex-direction:column;gap:8px;margin-top:8px;">
        <div style="display:flex;justify-content:space-between"><span>P_best:</span><b class="mono">{p_best:.3f}</b></div>
        <div style="display:flex;justify-content:space-between"><span>Beta (Error Overlap):</span><b class="mono">{beta:.3f}</b></div>
        <div style="display:flex;justify-content:space-between"><span>Ensemble Ceiling (1 - Beta):</span><b class="mono">{ceil_beta:.3f}</b></div>
        <div style="font-size:12px;color:var(--text-secondary);margin-top:4px;border-top:1px solid var(--border-light);padding-top:6px;">
          Verdict: <b>{html.escape(ensemble_verdict)}</b>
        </div>
      </div>
    </div>
  </div>

  <!-- Section 5: Pricing & Savings -->
  <div class="card">
    <div class="section-title">Pricing & Financial Savings</div>
    <div style="display:flex;justify-content:space-around;padding:12px 0;">
      <div style="text-align:center">
        <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase">Estimated Saved USD</div>
        <div style="font-size:24px;font-weight:800;color:var(--green)">${pricing.get('saved_usd', 0.0):.2f}</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase">Total Routed Tokens</div>
        <div style="font-size:24px;font-weight:800">{pricing.get('total_tokens', 0):,}</div>
      </div>
    </div>
  </div>

</div>
</body></html>"""


def render_chat_html() -> str:
    """Renders the Anthropic Claude-style interactive web chat interface."""
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Waypost — Chat</title>
<link rel=icon href="{FAVICON}">
<style>
{theme_css()}

html, body {{
  height: 100%;
  overflow: hidden;
}}
.chat-app {{
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg);
}}
.chat-topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 24px;
  background: var(--card);
  border-bottom: 1px solid var(--border);
  gap: 12px;
  flex-wrap: wrap;
}}
.model-pill {{
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 4px 10px;
  font-size: 13px;
  font-weight: 500;
}}
.model-select {{
  background: transparent;
  border: none;
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  outline: none;
  cursor: pointer;
  font-family: var(--font-sans);
}}
.chat-options {{
  display: flex;
  align-items: center;
  gap: 14px;
  font-size: 12px;
  color: var(--text-secondary);
}}
.chat-options label {{
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
}}
.btn-icon {{
  background: transparent;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  padding: 4px 10px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.15s ease;
}}
.btn-icon:hover {{
  background: var(--bg-subtle);
  color: var(--text);
}}

/* Message stream */
.chat-messages {{
  flex: 1;
  overflow-y: auto;
  padding: 24px 20px;
  scroll-behavior: smooth;
}}
.messages-inner {{
  max-width: 820px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
}}

/* Welcome Hero */
.welcome-hero {{
  text-align: center;
  padding: 48px 20px 20px;
  margin: auto 0;
}}
.welcome-icon {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 56px;
  height: 56px;
  border-radius: var(--radius-md);
  background: var(--bg-subtle);
  color: var(--text);
  border: 1px solid var(--border);
  margin-bottom: 16px;
}}
.welcome-title {{
  font-family: var(--font-sans);
  font-size: 24px;
  font-weight: 700;
  color: var(--text);
  margin: 0 0 8px;
  letter-spacing: -0.02em;
}}
.welcome-sub {{
  color: var(--text-secondary);
  font-size: 13.5px;
  max-width: 500px;
  margin: 0 auto 32px;
  line-height: 1.5;
}}
.prompt-chips {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
  max-width: 660px;
  margin: 0 auto;
}}
.prompt-chip {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 14px 16px;
  font-size: 13px;
  color: var(--text);
  text-align: left;
  cursor: pointer;
  transition: all 0.15s ease;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.prompt-chip:hover {{
  border-color: var(--accent);
  background: var(--card);
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.06);
}}

/* Message items */
.message-row {{
  display: flex;
  gap: 14px;
  width: 100%;
}}
.message-row.user {{
  justify-content: flex-end;
}}
.message-bubble {{
  max-width: 85%;
  font-size: 14px;
  line-height: 1.6;
}}
.message-row.user .message-bubble {{
  background: var(--card);
  border: 1px solid var(--border);
  padding: 12px 18px;
  border-radius: 16px 16px 4px 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  color: var(--text);
  white-space: pre-wrap;
  font-weight: 450;
}}
.message-row.assistant {{
  justify-content: flex-start;
}}
.assistant-avatar {{
  width: 32px;
  height: 32px;
  border-radius: var(--radius-sm);
  background: var(--card);
  color: var(--text);
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 2px;
}}
.message-row.assistant .message-bubble {{
  flex: 1;
  color: var(--text);
}}
.router-pill {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 4px 10px;
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 10px;
  font-family: var(--font-mono);
}}
.router-pill b {{ color: var(--text); }}

/* Markdown typography */
.message-bubble h1, .message-bubble h2, .message-bubble h3 {{
  font-family: var(--font-sans);
  margin: 16px 0 8px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
}}
.message-bubble h1 {{ font-size: 19px; }}
.message-bubble h2 {{ font-size: 16px; }}
.message-bubble h3 {{ font-size: 14px; }}
.message-bubble p {{ margin: 0 0 12px; }}
.message-bubble p:last-child {{ margin-bottom: 0; }}
.message-bubble ul, .message-bubble ol {{
  margin: 8px 0 12px;
  padding-left: 22px;
}}
.message-bubble li {{ margin-bottom: 4px; }}
.message-bubble blockquote {{
  margin: 12px 0;
  padding: 6px 14px;
  border-left: 3px solid var(--text-muted);
  color: var(--text-secondary);
  background: var(--bg-subtle);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}}
.message-bubble code {{
  font-family: var(--font-mono);
  font-size: 12.5px;
  background: var(--bg-subtle);
  border: 1px solid var(--border-light);
  padding: 2px 6px;
  border-radius: 4px;
}}
.code-block-wrap {{
  position: relative;
  margin: 14px 0;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  overflow: hidden;
  background: var(--card);
}}
.code-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  background: var(--bg-subtle);
  border-bottom: 1px solid var(--border-light);
  font-size: 11px;
  color: var(--text-muted);
  font-family: var(--font-mono);
  text-transform: uppercase;
}}
.btn-copy {{
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-secondary);
  font-size: 11px;
  cursor: pointer;
  padding: 2px 8px;
  border-radius: 4px;
  transition: all 0.15s ease;
}}
.btn-copy:hover {{ background: var(--card); color: var(--text); }}
.code-block-wrap pre {{
  margin: 0;
  padding: 14px 16px;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.5;
}}
.code-block-wrap pre code {{
  background: transparent;
  border: none;
  padding: 0;
}}

/* Thinking Disclosure Box */
.thinking-box {{
  margin: 8px 0 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-subtle);
  font-size: 12.5px;
  color: var(--text-secondary);
  overflow: hidden;
}}
.thinking-box summary {{
  padding: 8px 12px;
  cursor: pointer;
  font-weight: 600;
  color: var(--text-secondary);
  user-select: none;
  outline: none;
  background: var(--card);
  border-bottom: 1px solid var(--border-light);
}}
.thinking-box summary:hover {{
  color: var(--text);
}}
.thinking-content {{
  padding: 10px 14px;
  white-space: pre-wrap;
  font-family: var(--font-mono);
  font-size: 11.5px;
  line-height: 1.5;
  color: var(--text-muted);
  max-height: 200px;
  overflow-y: auto;
}}

/* Input bar */
.chat-bottom {{
  padding: 14px 20px 22px;
  background: var(--bg);
  border-top: 1px solid var(--border-light);
}}
.input-container {{
  max-width: 820px;
  margin: 0 auto;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 10px 14px 10px 18px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  display: flex;
  align-items: flex-end;
  gap: 10px;
  transition: all 0.15s ease;
}}
.input-container:focus-within {{
  border-color: var(--accent);
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}}
#chat-input {{
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  font-size: 14px;
  font-family: var(--font-sans);
  color: var(--text);
  resize: none;
  max-height: 180px;
  min-height: 26px;
  line-height: 1.5;
  padding: 3px 0;
}}
.btn-send {{
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: var(--accent);
  color: var(--accent-fg);
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 14px;
  font-weight: 700;
  transition: all 0.15s ease;
  flex-shrink: 0;
}}
.btn-send:hover {{
  opacity: 0.9;
  transform: scale(1.04);
}}
.btn-send:disabled {{
  background: var(--border);
  color: var(--text-muted);
  cursor: not-allowed;
  transform: none;
}}
.input-hint {{
  text-align: center;
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 8px;
}}
.cursor-blink {{
  display: inline-block;
  width: 6px;
  height: 14px;
  background: var(--text);
  margin-left: 2px;
  vertical-align: -2px;
  animation: blink 0.9s infinite;
}}
@keyframes blink {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0; }}
}}
</style></head><body>
<div class="chat-app">
  {nav_header('chat')}
  <div class="chat-topbar">
    <div class="model-pill">
      <span style="color:var(--accent);font-size:12px">⚡</span>
      <select id="model-select" class="model-select">
        <optgroup label="Virtual Smart Routing" id="group-virtual">
          <option value="auto">auto (Smart Router)</option>
          <option value="Tier S">Tier S · Fast</option>
          <option value="Tier M">Tier M · General</option>
          <option value="Tier L">Tier L · Reasoning</option>
        </optgroup>
        <optgroup label="Local Models" id="group-local"></optgroup>
        <optgroup label="Cloud Models" id="group-cloud"></optgroup>
      </select>
    </div>
    <div class="chat-options">
      <label><input type="checkbox" id="opt-stream" checked> Streaming</label>
      <label><input type="checkbox" id="opt-thinking"> Thinking</label>
      <label>Privacy:
        <select id="opt-privacy" style="background:transparent;border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:11px;padding:2px 4px">
          <option value="default">default</option>
          <option value="strict">strict (no cloud)</option>
        </select>
      </label>
      <button class="btn-icon" id="btn-clear" title="Clear chat (⌘K)">Clear</button>
    </div>
  </div>

  <div class="chat-messages" id="messages-container">
    <div class="messages-inner" id="messages-list">
      <div class="welcome-hero" id="welcome-hero">
        <div class="welcome-icon">
          {LOGO}
        </div>
        <h2 class="welcome-title">How can I help you today?</h2>
        <p class="welcome-sub">Waypost routes prompts across local and cloud models, choosing the fastest free engine with automatic escalation.</p>
        <div class="prompt-chips">
          <div class="prompt-chip" onclick="usePrompt(this.innerText)">Compare Rust vs Go for high-throughput networking services</div>
          <div class="prompt-chip" onclick="usePrompt(this.innerText)">Write a Python decorator to rate-limit async functions with token buckets</div>
          <div class="prompt-chip" onclick="usePrompt(this.innerText)">Explain transformer self-attention and KV cache mechanisms simply</div>
          <div class="prompt-chip" onclick="usePrompt(this.innerText)">How do circuit breakers prevent cascading failures in microservices?</div>
        </div>
      </div>
    </div>
  </div>

  <div class="chat-bottom">
    <div class="input-container">
      <textarea id="chat-input" placeholder="Message Waypost..." rows="1"></textarea>
      <button class="btn-send" id="btn-send" title="Send message (Enter)">↑</button>
    </div>
    <div class="input-hint">Waypost Smart Router · Enter to send · Shift+Enter for new line · ⌘K to clear</div>
  </div>
</div>

<script>
window.onerror = function(msg, url, line, col, error) {{
  const div = document.createElement("div");
  div.style = "position:fixed;top:10px;left:10px;z-index:9999;background:rgba(255,0,0,0.8);color:white;padding:10px;font-family:monospace;border-radius:4px;max-width:80%;word-break:break-all;";
  div.innerText = "Error: " + msg + " at " + line + ":" + col;
  document.body.appendChild(div);
}};
window.addEventListener("unhandledrejection", function(e) {{
  const div = document.createElement("div");
  div.style = "position:fixed;top:60px;left:10px;z-index:9999;background:rgba(255,0,0,0.8);color:white;padding:10px;font-family:monospace;border-radius:4px;max-width:80%;word-break:break-all;";
  div.innerText = "Unhandled Rejection: " + (e.reason && e.reason.message ? e.reason.message : e.reason);
  document.body.appendChild(div);
}});
const messagesList = document.getElementById('messages-list');
const messagesContainer = document.getElementById('messages-container');
const chatInput = document.getElementById('chat-input');
const btnSend = document.getElementById('btn-send');
const btnClear = document.getElementById('btn-clear');
const modelSelect = document.getElementById('model-select');
const groupLocal = document.getElementById('group-local');
const groupCloud = document.getElementById('group-cloud');
const optStream = document.getElementById('opt-stream');
const optThinking = document.getElementById('opt-thinking');
const optPrivacy = document.getElementById('opt-privacy');
const welcomeHero = document.getElementById('welcome-hero');

let history = [];
let isGenerating = false;
let abortController = null;

// Populate specific models from /v1/pricing
fetch('/v1/pricing').then(r => r.json()).then(res => {{
  if (res && res.models) {{
    res.models.forEach(m => {{
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.id + (m.free ? ' · free' : '');
      if (m.is_local) {{
        groupLocal.appendChild(opt);
      }} else {{
        groupCloud.appendChild(opt);
      }}
    }});
  }}
}}).catch(() => {{}});

// Auto-expand textarea
chatInput.addEventListener('input', () => {{
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
}});

chatInput.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    sendMessage();
  }}
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {{
    e.preventDefault();
    clearChat();
  }}
}});

btnSend.addEventListener('click', () => {{
  if (isGenerating) {{
    stopGeneration();
  }} else {{
    sendMessage();
  }}
}});

btnClear.addEventListener('click', clearChat);

function usePrompt(text) {{
  chatInput.value = text;
  sendMessage();
}}

function clearChat() {{
  history = [];
  messagesList.innerHTML = '';
  if (welcomeHero) messagesList.appendChild(welcomeHero);
  chatInput.focus();
}}

function escapeHtml(str) {{
  if (str === null || str === undefined) return '';
  const s = typeof str === 'string' ? str : String(str);
  return s.replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#039;');
}}

function extractText(obj) {{
  if (!obj) return '';
  if (typeof obj === 'string') return obj;
  if (Array.isArray(obj)) {{
    return obj.map(item => {{
      if (typeof item === 'string') return item;
      if (item && item.text) return item.text;
      if (item && item.content) return extractText(item.content);
      return '';
    }}).join('');
  }}
  if (obj.text) return typeof obj.text === 'string' ? obj.text : extractText(obj.text);
  if (obj.content) return typeof obj.content === 'string' ? obj.content : extractText(obj.content);
  if (obj.reasoning) return typeof obj.reasoning === 'string' ? obj.reasoning : extractText(obj.reasoning);
  if (obj.reasoning_content) return typeof obj.reasoning_content === 'string' ? obj.reasoning_content : extractText(obj.reasoning_content);
  return '';
}}

// Lightweight Markdown renderer
function renderMarkdown(md) {{
  if (md === null || md === undefined) return '';
  let str = typeof md === 'string' ? md : String(md);
  if (!str) return '';

  const codeBlockCount = (str.match(/```/g) || []).length;
  if (codeBlockCount % 2 === 1) {{
    str = str + '\\n```';
  }}

  // 1. Code blocks
  let text = str.replace(/```([a-zA-Z0-9_-]*)\\n([\\s\\S]*?)```/g, function(match, lang, code) {{
    const l = lang ? lang.trim() : 'text';
    const escaped = escapeHtml(code.replace(/\\n$/, ''));
    return '<div class="code-block-wrap"><div class="code-header"><span>' + l + '</span><button class="btn-copy" onclick="copyCode(this)">Copy</button></div><pre><code>' + escaped + '</code></pre></div>';
  }});

  // 2. Inline code
  text = text.replace(/`([^`]+)`/g, function(match, code) {{
    return '<code>' + escapeHtml(code) + '</code>';
  }});

  // 3. Headings
  text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>')
             .replace(/^## (.*$)/gim, '<h2>$1</h2>')
             .replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // 4. Bold & italic
  text = text.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
             .replace(/\\*(.*?)\\*/g, '<em>$1</em>');

  // 5. Blockquotes
  text = text.replace(/^\\> (.*$)/gim, '<blockquote>$1</blockquote>');

  // 6. Lists
  text = text.replace(/^\\s*[-*+] (.*$)/gim, '<li>$1</li>');
  text = text.replace(/(<li>.*<\\/li>)/s, '<ul>$1</ul>');

  // 7. Paragraphs
  const paragraphs = text.split(/\\n\\n+/);
  return paragraphs.map(p => {{
    p = p.trim();
    if (!p) return '';
    if (p.startsWith('<div') || p.startsWith('<h') || p.startsWith('<ul') || p.startsWith('<blockquote') || p.startsWith('<details')) {{
      return p;
    }}
    return '<p>' + p.replace(/\\n/g, '<br>') + '</p>';
  }}).join('');
}}

function copyCode(btn) {{
  const pre = btn.parentElement.nextElementSibling;
  if (pre) {{
    navigator.clipboard.writeText(pre.innerText).then(() => {{
      const orig = btn.innerText;
      btn.innerText = 'Copied!';
      setTimeout(() => {{ btn.innerText = orig; }}, 1500);
    }});
  }}
}}

function appendUserMessage(content) {{
  if (welcomeHero && welcomeHero.parentNode) {{
    welcomeHero.parentNode.removeChild(welcomeHero);
  }}
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = '<div class="message-bubble">' + escapeHtml(content) + '</div>';
  messagesList.appendChild(row);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}}

function createAssistantMessage() {{
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.innerHTML = `
    <div class="assistant-avatar">
      {LOGO}
    </div>
    <div class="message-bubble"><div class="bubble-content"><span class="cursor-blink"></span></div><div class="bubble-meta"></div></div>
  `;
  messagesList.appendChild(row);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return {{
    row: row,
    contentEl: row.querySelector('.bubble-content'),
    metaEl: row.querySelector('.bubble-meta')
  }};
}}

function stopGeneration() {{
  if (abortController) {{
    abortController.abort();
    abortController = null;
  }}
  setGenerating(false);
}}

function setGenerating(gen) {{
  isGenerating = gen;
  if (gen) {{
    btnSend.textContent = '■';
    btnSend.title = 'Stop generating';
  }} else {{
    btnSend.textContent = '↑';
    btnSend.title = 'Send message';
  }}
}}

async function sendMessage() {{
  const text = chatInput.value.trim();
  if (!text || isGenerating) return;

  chatInput.value = '';
  chatInput.style.height = 'auto';

  appendUserMessage(text);
  history.push({{ role: 'user', content: text }});

  const isStream = optStream ? optStream.checked : true;
  const isThinking = optThinking ? optThinking.checked : false;
  const model = (modelSelect && modelSelect.value) ? modelSelect.value : 'auto';
  const privacy = (optPrivacy && optPrivacy.value) ? optPrivacy.value : 'default';

  const cleanMessages = history.filter(m => m && m.content && String(m.content).trim().length > 0);

  const payload = {{
    model: model,
    messages: cleanMessages,
    stream: isStream,
    enable_thinking: isThinking,
    privacy: privacy,
  }};

  const assistantMsg = createAssistantMessage();
  let fullContent = '';
  let routerMeta = null;

  setGenerating(true);
  abortController = new AbortController();

  try {{
    const response = await fetch('/v1/chat/completions', {{
      method: 'POST',
      headers: {{
        'Content-Type': 'application/json',
        'Accept': isStream ? 'text/event-stream, application/json' : 'application/json'
      }},
      body: JSON.stringify(payload),
      signal: abortController.signal
    }});

    if (!response.ok) {{
      const errJson = await response.json().catch(() => ({{}}));
      const errMsg = (errJson.error && errJson.error.message) ? errJson.error.message : response.statusText;
      assistantMsg.contentEl.innerHTML = '<span style="color:var(--red);background:var(--red-bg);padding:8px 12px;border-radius:6px;display:inline-block">Error: ' + escapeHtml(errMsg) + '</span>';
      setGenerating(false);
      return;
    }}

    if (isStream) {{
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {{
        const {{ value, done }} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {{ stream: true }});
        const lines = buffer.split('\\n');
        buffer = lines.pop() || '';

        for (const line of lines) {{
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const dataStr = trimmed.slice(5).trim();
          if (dataStr === '[DONE]') break;
          try {{
            const chunk = JSON.parse(dataStr);
            if (chunk.error) {{
              const errMsg = typeof chunk.error === 'string' ? chunk.error : (chunk.error.message || JSON.stringify(chunk.error));
              assistantMsg.contentEl.innerHTML = '<span style="color:var(--red);background:var(--red-bg);padding:8px 12px;border-radius:6px;display:inline-block">Error: ' + escapeHtml(errMsg) + '</span>';
              return;
            }}
            const c0 = (chunk.choices && chunk.choices[0]) || {{}};
            const d = c0.delta || {{}};
            const delta = extractText(d) || (typeof d.content === 'string' ? d.content : '');
            if (delta) {{
              fullContent += delta;
              assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent) + '<span class="cursor-blink"></span>';
              messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }}
            if (chunk.router) {{
              routerMeta = chunk.router;
            }}
          }} catch (e) {{}}
        }}
      }}
      if (!fullContent) {{
        assistantMsg.contentEl.innerHTML = '<span style="color:var(--text-muted)">(Empty response from model)</span>';
      }} else {{
        assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent);
      }}
    }} else {{
      const data = await response.json();
      if (data.error) {{
        const errMsg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
        assistantMsg.contentEl.innerHTML = '<span style="color:var(--red);background:var(--red-bg);padding:8px 12px;border-radius:6px;display:inline-block">Error: ' + escapeHtml(errMsg) + '</span>';
        setGenerating(false);
        return;
      }}
      const m0 = (data.choices && data.choices[0] && data.choices[0].message) || {{}};
      fullContent = extractText(m0) || (typeof m0.content === 'string' ? m0.content : '');
      routerMeta = data.router;
      if (!fullContent) {{
        assistantMsg.contentEl.innerHTML = '<span style="color:var(--text-muted)">(Empty response from model)</span>';
      }} else {{
        assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent);
      }}
    }}

    if (fullContent && fullContent.trim()) {{
      history.push({{ role: 'assistant', content: fullContent }});
    }}

    if (routerMeta) {{
      const prov = routerMeta.provider || '—';
      const mName = routerMeta.model || '—';
      const tier = routerMeta.complexity_tier || '—';
      const lat = routerMeta.latency_ms ? routerMeta.latency_ms + 'ms' : '';
      const cached = routerMeta.cache === 'hit' ? 'cached' : 'fresh';
      assistantMsg.metaEl.innerHTML = `
        <div class="router-pill">
          <span>⚡ <b>Router:</b> ${{escapeHtml(prov)}} · ${{escapeHtml(mName)}} · Tier ${{escapeHtml(tier)}} ${{lat ? '· ' + lat : ''}} · ${{cached}}</span>
        </div>
      `;
    }}
  }} catch (err) {{
    if (err.name !== 'AbortError') {{
      assistantMsg.contentEl.innerHTML = '<span style="color:var(--red);background:var(--red-bg);padding:8px 12px;border-radius:6px;display:inline-block">Connection error: ' + escapeHtml(err.message) + '</span>';
    }} else {{
      assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent) + ' <span style="color:var(--text-muted);font-size:12px">(stopped)</span>';
    }}
  }} finally {{
    setGenerating(false);
    abortController = null;
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }}
}}
</script>
</body></html>"""

