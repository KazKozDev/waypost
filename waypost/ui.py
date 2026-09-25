"""UI Components and 3-Tab HTML Templates for Waypost v5.

Implements Part II of Waypost Spec v5:
- Tab 1: Chat (/chat, /)
- Tab 2: Dashboard (/dashboard, /v1/stats)
- Tab 3: Setup (/setup, /v1/models, /v1/probe, /v1/pricing)
"""
from __future__ import annotations

import base64
import html
import os
from typing import Any

# Load branded Waypost logo PNG (macos/icon-96.png or asesst/waypost.png)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATE_LOGOS = [
    os.path.join(_ROOT, "macos", "icon-96.png"),
    os.path.join(_ROOT, "asesst", "waypost.png"),
    os.path.join(_ROOT, "macos", "icon.png"),
    "macos/icon-96.png",
    "asesst/waypost.png",
]

_LOGO_URI = None
for _p in _CANDIDATE_LOGOS:
    if os.path.exists(_p):
        try:
            with open(_p, "rb") as _f:
                _LOGO_URI = "data:image/png;base64," + base64.b64encode(
                    _f.read()
                ).decode("utf-8")
            break
        except Exception:
            pass

if not _LOGO_URI:
    _LOGO_URI = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%232b2620'/%3E%3Cpath d='M8 10h16M8 16h16M8 22h10' stroke='%23fbfaf7' stroke-width='2.5' stroke-linecap='round'/%3E%3C/svg%3E"

# Pure transparent glyph for large center badge
_GLYPH_PATH = os.path.join(_ROOT, "macos", "glyph.png")
_GLYPH_URI = None
if os.path.exists(_GLYPH_PATH):
    try:
        with open(_GLYPH_PATH, "rb") as _gf:
            _GLYPH_URI = "data:image/png;base64," + base64.b64encode(_gf.read()).decode(
                "utf-8"
            )
    except Exception:
        pass
if not _GLYPH_URI:
    _GLYPH_URI = _LOGO_URI

FAVICON = _LOGO_URI
LOGO = f'<img src="{_LOGO_URI}" width="22" height="22" alt="Waypost" style="border-radius:4px;flex-shrink:0;vertical-align:middle;object-fit:cover;">'
# Chat avatar: the signpost drawn as an outline instead of the app icon.
# The app icon is a dark rounded square — at 22px next to a line of text it
# reads as a black block, and it cannot follow the light/dark theme. This
# inherits currentColor, so one mark serves both themes.
AVATAR_MARK = (
    '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" aria-label="Waypost" role="img">'
    '<path d="M7 3v18"/>'
    '<path d="M7 5.5h9.5l2.5 2.75-2.5 2.75H7"/>'
    '<path d="M7 13.5h6.5l2.5 2.75-2.5 2.75H7"/>'
    "</svg>"
)

LOGO_LARGE = f'<img src="{_GLYPH_URI}" width="70" height="70" alt="Waypost" style="object-fit:contain;">'


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
  --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "SF Pro", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --font-mono: "SF Mono", SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}

*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html, body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 13px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

input, button, select, textarea, optgroup, option {
  font-family: inherit;
  font-size: inherit;
  color: inherit;
}

code, pre, kbd, samp, .mono, .font-mono, .mono-tag, .data-mono {
  font-family: var(--font-mono);
}

h1, h2, h3, h4, h5, h6, .brand-title, .brand-name, .page-title, .section-title {
  font-family: var(--font-sans);
}

/* Nav Header */
.top-header {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  padding: 8px 24px;
  background: var(--card);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 100;
}
.macos-app .top-header,
html.macos-app .top-header,
body.macos-app .top-header {
  padding-left: 20px;
  -webkit-app-region: drag;
  user-select: none;
}
.macos-app .top-header .header-left {
  width: 72px;
}
.top-header .header-left {
  display: flex;
  align-items: center;
  grid-column: 1;
}
.top-header .brand-center,
.top-header .brand-group,
.top-header .brand {
  display: flex;
  align-items: center;
  justify-content: center;
  text-decoration: none;
  color: var(--text);
  grid-column: 2;
}
.brand-title, .brand-name {
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--text);
}
.brand-tag {
  display: none;
}
.top-header .nav-links,
.top-header .nav-tabs {
  display: flex;
  gap: 4px;
  background: var(--bg-subtle);
  padding: 3px;
  border-radius: var(--radius-sm);
  justify-self: end;
  grid-column: 3;
}
.macos-app .top-header .brand-center,
.macos-app .top-header .brand-group,
.macos-app .top-header .brand,
.macos-app .top-header .nav-links,
.macos-app .top-header .nav-tabs,
.macos-app .top-header a,
.macos-app .top-header button,
.macos-app .top-header input,
.macos-app .top-header select {
  -webkit-app-region: no-drag;
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
.live-dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 2px rgba(42, 122, 76, 0.2);
  animation: pulse-dot 2s infinite ease-in-out;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.45; transform: scale(0.85); }
}

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
  <div class="header-left"></div>
  <a href="/chat" class="brand-center brand">
    <span class="brand-title brand-name">Waypost</span>
  </a>
  <nav class="nav-links nav-tabs">
    <a href="/chat" class="nav-link nav-tab {'active' if active == 'chat' else ''}">Chat</a>
    <a href="/swarm" class="nav-link nav-tab {'active' if active == 'swarm' else ''}">Рой</a>
    <a href="/dashboard" class="nav-link nav-tab {'active' if active == 'dashboard' else ''}">Dashboard</a>
    <a href="/providers" class="nav-link nav-tab {'active' if active == 'providers' else ''}">Providers</a>
    <a href="/setup" class="nav-link nav-tab {'active' if active == 'setup' else ''}">Setup</a>
  </nav>
</header>"""


def render_swarm_html() -> str:
    """Live swarm transcript, progress journal, and run controls."""
    page = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Waypost — Рой</title><link rel="icon" href="__FAVICON__">
<style>
__THEME_CSS__
html,body{height:100%;overflow:hidden}
.swarm-app{height:100vh;display:flex;flex-direction:column;background:var(--bg)}
.swarm-toolbar{display:flex;align-items:center;gap:12px;padding:10px 24px;border-bottom:1px solid var(--border-light);background:var(--card);min-height:52px}
.swarm-title{font-size:14px;font-weight:700;white-space:nowrap}
.swarm-run-select{min-width:150px;max-width:300px;flex:1;border:1px solid var(--border);background:var(--card);border-radius:7px;padding:7px 10px;color:var(--text);font:inherit}
.swarm-status{display:inline-flex;align-items:center;gap:7px;color:var(--text-secondary);font-size:12px;margin-left:auto;white-space:nowrap}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--text-muted)}
.status-dot.running{background:var(--green)}.status-dot.paused{background:var(--amber)}.status-dot.failed,.status-dot.interrupted,.status-dot.needs_attention,.status-dot.budget_exhausted{background:var(--red)}.status-dot.completed{background:var(--green)}
.swarm-button{border:1px solid var(--border);background:var(--card);color:var(--text);padding:7px 11px;border-radius:7px;font:inherit;font-weight:550;cursor:pointer;white-space:nowrap}
.swarm-button:hover{background:var(--bg-subtle)}.swarm-button:disabled{opacity:.4;cursor:default}
.swarm-button.danger{color:var(--red)}
.swarm-feed{flex:1;overflow:auto;padding:30px 20px 22px}.feed-inner{max-width:850px;margin:0 auto}
.swarm-empty{text-align:center;color:var(--text-secondary);padding:16vh 20px 0}.swarm-empty h1{font-size:25px;color:var(--text);margin:14px 0 5px}.swarm-empty p{font-size:14px;line-height:1.6}
.swarm-mark{width:60px;height:60px;margin:auto;display:flex;align-items:center;justify-content:center;background:var(--card);border:1px solid var(--border);border-radius:18px;box-shadow:0 6px 20px rgba(0,0,0,.06)}
.swarm-mark img{width:42px;height:42px;object-fit:contain}
.feed-item{display:flex;gap:12px;margin:0 0 18px;align-items:flex-start}.feed-item.user{justify-content:flex-end}.feed-body{max-width:85%;min-width:0}
.feed-role{font-size:11px;color:var(--text-muted);font-weight:650;margin:0 0 4px}.feed-item.user .feed-role{text-align:right}
.feed-text{font-size:13px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere}.feed-item.user .feed-text{background:var(--bg-subtle);border:1px solid var(--border-light);border-radius:14px;padding:10px 13px}
.feed-item.monitor .feed-text{border-left:2px solid var(--terracotta);padding:3px 0 3px 12px}
.feed-item.result .feed-text{border-left:2px solid var(--green);padding:3px 0 3px 12px}
.feed-log{border-top:1px solid var(--border-light);padding:9px 0;color:var(--text-secondary);font-size:12px;display:flex;gap:12px;align-items:baseline;min-width:0}
.feed-log .log-time{font-family:var(--font-mono);font-size:10px;color:var(--text-muted);width:58px;flex:none}.feed-log .log-label{font-weight:650;color:var(--text);min-width:110px}.feed-log .log-detail{white-space:pre-wrap;overflow-wrap:anywhere;min-width:0}
.swarm-board{border-bottom:1px solid var(--border-light);background:var(--card);padding:0 24px}
.swarm-board summary{cursor:pointer;font-size:12px;font-weight:650;color:var(--text-secondary);padding:8px 0;list-style:none}
.swarm-board summary::-webkit-details-marker{display:none}.swarm-board summary::before{content:"▸ ";color:var(--text-muted)}.swarm-board[open] summary::before{content:"▾ "}
.board-list{max-width:850px;margin:0 auto;max-height:30vh;overflow:auto;padding:0 0 10px}
.board-entry{display:flex;gap:10px;font-size:12px;line-height:1.5;padding:4px 0;border-top:1px solid var(--border-light);min-width:0}
.board-kind{flex:none;width:86px;font-weight:650;color:var(--text)}.board-kind.dead_end{color:var(--red)}.board-kind.assumption{color:var(--amber)}.board-kind.decision{color:var(--green)}
.board-text{white-space:pre-wrap;overflow-wrap:anywhere;min-width:0;color:var(--text-secondary)}.board-author{color:var(--text-muted)}
.swarm-composer{padding:12px 20px 15px;background:var(--card);border-top:1px solid var(--border-light)}
.composer-inner{max-width:850px;margin:auto}.composer-box{display:flex;gap:10px;align-items:flex-end;border:1px solid var(--border);border-radius:12px;background:var(--card);box-shadow:0 2px 7px rgba(0,0,0,.035);padding:9px 9px 9px 14px}
.composer-box textarea{width:100%;border:0;outline:0;resize:none;min-height:30px;max-height:145px;font:inherit;font-size:13px;line-height:1.5;color:var(--text);background:transparent}
.composer-send{width:34px;height:34px;flex:none;border:0;border-radius:50%;background:var(--accent);color:var(--accent-fg);font-size:17px;cursor:pointer}.composer-send:disabled{opacity:.5;cursor:default}
.composer-help{font-size:11px;color:var(--text-muted);text-align:center;margin-top:7px}.composer-error{font-size:12px;color:var(--red);max-width:850px;margin:0 auto 7px;display:none}
@media(max-width:760px){.swarm-board{padding:0 12px}.board-kind{width:70px}.swarm-toolbar{padding:9px 12px;gap:6px;flex-wrap:wrap}.swarm-run-select{min-width:110px}.swarm-title{display:none}.swarm-status{order:2;margin-left:0}.swarm-button{padding:6px 8px}.swarm-feed{padding:18px 13px}.feed-body{max-width:94%}.swarm-composer{padding:10px 12px}}
</style></head><body><div class="swarm-app">
__NAV_HEADER__
<div class="swarm-toolbar">
  <span class="swarm-title">Рой агентов</span>
  <select id="run-select" class="swarm-run-select" aria-label="Задачи роя"><option value="">Новая задача</option></select>
  <button id="new-button" class="swarm-button" title="Новая задача">Новая</button>
  <span id="run-status" class="swarm-status"><span class="status-dot"></span><span>Готов к задаче</span></span>
  <button id="pause-button" class="swarm-button" disabled>Пауза</button>
  <button id="resume-button" class="swarm-button" disabled>Продолжить</button>
  <button id="interrupt-button" class="swarm-button danger" disabled>Прервать</button>
  <button id="clear-button" class="swarm-button" title="Остановить задачу и начать с чистого листа (Cmd+K)">Clear</button>
</div>
<details id="board-panel" class="swarm-board" hidden><summary>Доска роя <span id="board-count"></span></summary><div id="board-list" class="board-list"></div></details>
<main id="feed" class="swarm-feed"><div id="feed-inner" class="feed-inner">
  <div id="empty" class="swarm-empty"><div class="swarm-mark"><img src="__GLYPH_URI__" alt=""></div><h1>Что поручить рою?</h1>
    <p>Опишите результат. Агенты спланируют работу, покажут действия и проверят итог.<br>Вы можете уточнять задачу и управлять выполнением.</p></div>
</div></main>
<div class="swarm-composer"><div id="composer-error" class="composer-error"></div><div class="composer-inner">
  <div class="composer-box"><textarea id="message" rows="1" placeholder="Задача для роя или уточнение…" aria-label="Сообщение рою"></textarea><button id="send-button" class="composer-send" aria-label="Отправить">↑</button></div>
  <div class="composer-help">Enter — отправить · Shift+Enter — новая строка · Пауза и продолжение сохраняют ход работы</div>
</div></div></div>
<script>
const $ = id => document.getElementById(id);
let runId = localStorage.getItem('waypost-swarm-run') || '';
let offset = 0, busy = false, currentStatus = '', pollInFlight = false, initialized = false, newRunMode = false, finalRound = null;
const escapeHtml = text => String(text ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
function showError(message){$('composer-error').textContent=message;$('composer-error').style.display=message?'block':'none'}
function scrollDown(){const feed=$('feed');if(feed.scrollHeight-feed.scrollTop-feed.clientHeight<240)feed.scrollTop=feed.scrollHeight}
function addBubble(role,text,type='agent'){
  $('empty').style.display='none';const row=document.createElement('div');row.className='feed-item '+type;
  row.innerHTML='<div class="feed-body"><div class="feed-role">'+escapeHtml(role)+'</div><div class="feed-text">'+escapeHtml(text)+'</div></div>';
  $('feed-inner').appendChild(row);scrollDown();
}
function addLog(time,label,detail){
  $('empty').style.display='none';const row=document.createElement('div');row.className='feed-log';
  row.innerHTML='<span class="log-time">'+escapeHtml((time||'').slice(11,19))+'</span><span class="log-label">'+escapeHtml(label)+'</span><span class="log-detail">'+escapeHtml(detail)+'</span>';
  $('feed-inner').appendChild(row);scrollDown();
}
const agentOf=e=>(e.session||'').split(':').pop()||'агент';
const secs=e=>e.seconds!=null?' · '+e.seconds+' с':'';
const ladder=e=>{const p=e.router?.fallback_path||[];return p.length>1?' · перебрано: '+p.join(' → '):''};
const BOARD_KINDS={fact:'Факт',decision:'Решение',assumption:'Допущение',dead_end:'Тупик'};
function renderBoard(board){const panel=$('board-panel');if(!board||!board.length){panel.hidden=true;return}
  panel.hidden=false;$('board-count').textContent='· '+board.length;
  $('board-list').innerHTML=board.slice().reverse().map(b=>'<div class="board-entry"><span class="board-kind '+escapeHtml(b.kind)+'">'+escapeHtml(BOARD_KINDS[b.kind]||b.kind)+'</span><span class="board-text">'+escapeHtml(b.text)+' <span class="board-author">— '+escapeHtml(b.author||'')+', раунд '+escapeHtml(b.round||'')+'</span></span></div>').join('')}
function eventView(e){
  const kind=e.event||'';const task=e.task||e.role||'';
  if(kind==='user_task')return addBubble('Вы',e.text,'user');
  if(kind==='user_message_queued')return addBubble('Вы · уточнение',e.text,'user');
  if(kind==='plan')return addBubble('План роя', (e.plan?.tasks||[]).map((t,i)=>(i+1)+'. '+t.role+' — '+t.instruction).join('\n'), 'monitor');
  if(kind==='progress_assessment')return addBubble('Агент прогресса · '+e.action,e.reason+(e.guidance?'\n'+e.guidance:''),'monitor');
  if(kind==='task_done')return addLog(e.time,'Завершил: '+task,e.role||'');
  if(kind==='tool_started')return addLog(e.time,task+' · '+e.tool, e.arguments?.path||e.arguments?.code?.slice(0,100)||'');
  if(kind==='tool_finished')return addLog(e.time,task+' · результат',e.observation||'');
  if(kind==='llm_request')return addLog(e.time,'Запрос · '+agentOf(e),'ждём ответ модели…');
  if(kind==='llm_response')return addLog(e.time,'Ответ · '+agentOf(e),(e.router?.provider||'')+' / '+(e.router?.model||'')+secs(e)+ladder(e));
  if(kind==='llm_error')return addLog(e.time,'Ошибка · '+agentOf(e),(e.status?e.status+' · ':'')+(e.error||'')+secs(e)+ladder(e));
  if(kind==='invalid_output')return addLog(e.time,'Исправление ответа',e.error||'Неверный формат');
  if(kind==='collective')return addLog(e.time,'Коллектив · '+(e.role||''),(e.size||0)+' из '+(e.width||e.size||0)+(e.families?.length?': '+e.families.filter(Boolean).join(', '):'')+(e.reason?' · сузился: '+e.reason:''));
  if(kind==='collective_narrowed')return;
  if(kind==='review_panel')return addBubble('Комиссия проверки · '+(e.passed?'прошло':'не прошло'),(e.votes||[]).map(v=>(v.passed?'✓ ':'✗ ')+(v.family||'модель')+(v.findings?.length?': '+v.findings.join('; '):'')).join('\n')+(e.confirmed?.length?'\nПодтверждено ≥2: '+e.confirmed.join('; '):''),'monitor');
  if(kind==='proposals_judged')return addBubble('Судья · '+(e.subject==='plan'?'план':'итог'),'Варианты от: '+(e.families||[]).map(f=>f||'модель').join(', ')+'\n'+(e.merged?'Решение: объединить сильные стороны':'Выбран вариант №'+((e.chosen||0)+1))+(e.judge_family?' · судья: '+e.judge_family:'')+(e.critiques||[]).map(c=>'\n№'+(c.proposal+1)+': + '+(c.strengths||[]).join('; ')+(c.flaws?.length?' / − '+c.flaws.join('; '):'')).join(''),'monitor');
  if(kind==='judge_failed')return addLog(e.time,'Судья недоступен',(e.error||'')+' · взят первый вариант');
  if(kind==='debate')return e.contradictions?.length?addBubble('Спор · раунд '+e.round,'Противоречия:\n'+e.contradictions.map(c=>'- '+c).join('\n'),'monitor'):addLog(e.time,'Спор · раунд '+e.round,'противоречий нет');
  if(kind==='debate_speaker_failed')return addLog(e.time,'Спор · пропущен',e.task+' · '+(e.error||''));
  if(kind==='board_post')return addLog(e.time,'Доска · '+(BOARD_KINDS[e.entry_kind]||e.entry_kind),e.text+' — '+(e.author||''));
  if(kind==='lesson_saved')return addLog(e.time,'Память',(e.passed?'урок успеха':'урок неудачи')+' сохранён'+(e.dead_ends?' · тупиков: '+e.dead_ends:''));
  if(kind==='autonomous_finish')return addLog(e.time,e.delivered===false?'Завершено без результата':'Завершено автономно',e.reason||'');
  if(kind==='model_call_failed')return addLog(e.time,'Повтор · '+(e.role||''),'попытка '+e.failure+' · '+(e.error||''));
  if(kind==='router_unreachable')return addLog(e.time,'Роутер недоступен',(e.role||'')+' · ждём '+e.next_try_s+' с (прошло '+e.waited+' с)');
  if(kind==='waiting_for_models')return addLog(e.time,'Нет живых моделей','пауза '+e.seconds+' с, затем тот же шаг снова · '+(e.reason||''));
  if(kind==='run_stopped')return addLog(e.time,'Статус',(e.status||'')+(e.error?' · '+e.error:''));
  if(kind==='paused'||kind==='resumed'||kind==='replan'||kind==='interrupt_requested'||kind==='pause_requested'||kind==='resume_requested')return addLog(e.time,'Управление',kind+(e.reason?' · '+e.reason:''));
}
async function api(path,options={}){const response=await fetch(path,options);let data=await response.json();if(!response.ok)throw new Error(data.detail||'Ошибка запроса');return data}
function setStatus(s){currentStatus=s.status;const labels={starting:'Запуск',running:'Работает',paused:'Пауза',interrupted:'Прерван',completed:'Завершён',failed:'Ошибка',needs_attention:'Нужно уточнение',budget_exhausted:'Лимит достигнут'};
  const c=s.collective;const coll=c?' · коллектив '+c.size+'/'+c.width:'';
  const collTitle=c?('Последний коллектив: '+c.role+' — '+(c.families||[]).filter(Boolean).join(', ')+(c.reason?'\nСузился: '+c.reason:'')):'';
  $('run-status').innerHTML='<span class="status-dot '+escapeHtml(s.status)+'"></span><span title="'+escapeHtml(collTitle)+'">'+escapeHtml(labels[s.status]||s.status)+' · '+escapeHtml(s.phase||'')+' · вызовов '+escapeHtml(s.calls||0)+escapeHtml(coll)+'</span>';
  renderBoard(s.board);
  $('pause-button').disabled=!s.running||s.paused;$('resume-button').disabled=!(s.paused||['interrupted','failed','needs_attention'].includes(s.status));$('interrupt-button').disabled=!s.running;
  if(s.status==='completed'&&s.draft&&finalRound!==s.round){finalRound=s.round;addBubble('Итог роя',s.draft,'result')}
  if(['failed','needs_attention','budget_exhausted'].includes(s.status)&&s.error)showError(s.error);
  else if(s.status!=='starting')showError('');
}
async function refreshRuns(){const data=await api('/v1/swarm/runs');const select=$('run-select');const old=runId;
  select.innerHTML='<option value="">Новая задача</option>'+data.runs.map(r=>'<option value="'+escapeHtml(r.id)+'">'+escapeHtml((r.task||'').slice(0,65))+'</option>').join('');
  if(!initialized&&!runId&&!newRunMode&&data.runs.length)runId=data.runs[0].id;
  initialized=true;select.value=runId||'';if(runId!==old)resetFeed();
}
function resetFeed(){offset=0;finalRound=null;$('feed-inner').innerHTML='<div id="empty" class="swarm-empty"><div class="swarm-mark"><img src="__GLYPH_URI__" alt=""></div><h1>Что поручить рою?</h1><p>Опишите результат. Агенты спланируют работу, покажут действия и проверят итог.</p></div>';showError('');renderBoard([])}
async function poll(){if(pollInFlight||!runId)return;pollInFlight=true;try{
  const [status,events]=await Promise.all([api('/v1/swarm/runs/'+runId),api('/v1/swarm/runs/'+runId+'/events?offset='+offset)]);
  for(const e of events.events)eventView(e);offset=events.next_offset;setStatus(status);
}catch(err){showError(err.message)}finally{pollInFlight=false}}
async function send(){const text=$('message').value.trim();if(!text||busy)return;busy=true;$('send-button').disabled=true;showError('');try{
  if(!runId){const state=await api('/v1/swarm/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:text})});runId=state.id;newRunMode=false;localStorage.setItem('waypost-swarm-run',runId);resetFeed();await refreshRuns()}
  else await api('/v1/swarm/runs/'+runId+'/messages',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  $('message').value='';await poll();
}catch(err){showError(err.message)}finally{busy=false;$('send-button').disabled=false}}
async function control(action){if(!runId)return;try{await api('/v1/swarm/runs/'+runId+'/'+action,{method:'POST'});await poll()}catch(err){showError(err.message)}}
$('send-button').onclick=send;$('message').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
$('pause-button').onclick=()=>control('pause');$('resume-button').onclick=()=>control('resume');$('interrupt-button').onclick=()=>control('interrupt');
function startNew(){runId='';newRunMode=true;localStorage.removeItem('waypost-swarm-run');$('run-select').value='';resetFeed();$('run-status').innerHTML='<span class="status-dot"></span><span>Готов к задаче</span>';currentStatus='';for(const id of ['pause-button','resume-button','interrupt-button'])$(id).disabled=true;$('message').focus()}
$('new-button').onclick=startNew;
// Clear, like the chat's: the running task is stopped, the screen starts
// over. The run itself stays in the list with its artifacts.
async function clearRun(){if(runId&&currentStatus==='running'){try{await api('/v1/swarm/runs/'+runId+'/interrupt',{method:'POST'})}catch(err){}}$('message').value='';showError('');startNew()}
$('clear-button').onclick=clearRun;
document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();clearRun()}});
$('run-select').onchange=e=>{runId=e.target.value;newRunMode=!runId;if(runId)localStorage.setItem('waypost-swarm-run',runId);else localStorage.removeItem('waypost-swarm-run');resetFeed();poll()};
refreshRuns().then(poll).catch(err=>showError(err.message));setInterval(poll,1500);setInterval(()=>refreshRuns().catch(()=>{}),12000);
</script></body></html>"""
    return (page.replace("__THEME_CSS__", theme_css())
            .replace("__NAV_HEADER__", nav_header("swarm"))
            .replace("__FAVICON__", FAVICON)
            .replace("__GLYPH_URI__", _GLYPH_URI))


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
      <div style="display:flex;align-items:center;gap:10px;">
        <span class="live-dot" title="Live Auto-Polling Active (4s)"></span>
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
        burn_rate = q.get("burn_rate", "0/min")
        binding = q.get("binding", "—")
        eta = q.get("exhaustion_eta", "100% capacity available")
        rem_summary = q.get("remaining_summary", "")
        used_summary = q.get("used_summary", "")
        is_warn = pct > 80 or q.get("is_blocked", False)

        if pct > 0:
            bar_fill = max(1, min(10, int(pct / 10)))
            bar_str = "█" * bar_fill + "░" * (10 - bar_fill)
            usage_html = f'<span class="mono" style="font-weight:600">{bar_str} {pct:.0f}%</span> <span class="mono" style="color:var(--text-secondary);font-size:11px">({html.escape(used_summary)})</span>'
        else:
            usage_html = f'<span class="badge badge-success">100% free</span> <span class="mono" style="color:var(--text-secondary);font-size:11px">({html.escape(rem_summary)})</span>'

        quota_rows_html.append(
            f"""
        <tr>
          <td style="font-weight:600">{html.escape(name)}</td>
          <td>{usage_html}</td>
          <td class="mono">{html.escape(burn_rate)}</td>
          <td><span class="badge badge-neutral">{html.escape(binding)}</span></td>
          <td style="color:{'var(--amber)' if is_warn else 'var(--text-secondary)'};font-weight:{'600' if is_warn else 'normal'}">
            → {html.escape(eta)}
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
    loc_n = latencies.get("local_n", 0)
    cld_n = latencies.get("cloud_n", 0)

    def _lat(ms: float, n: int) -> str:
        """A local answer runs into tens of seconds — printing that as a
        five-digit ms number hides the scale. And zero samples is "no
        data", not "instant"."""
        if not n:
            return "no data"
        return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{int(ms)}ms"

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
            <div class="mono" style="font-size:16px;font-weight:700">{_lat(loc_p50, loc_n)} / {_lat(loc_p95, loc_n)}</div>
            <div style="font-size:10px;color:var(--text-muted)">{loc_n} samples · 24h</div>
          </div>
          <div style="background:var(--bg-subtle);padding:12px;border-radius:var(--radius-sm);">
            <div style="font-size:11px;color:var(--text-secondary)">Cloud P50 / P95</div>
            <div class="mono" style="font-size:16px;font-weight:700">{_lat(cld_p50, cld_n)} / {_lat(cld_p95, cld_n)}</div>
            <div style="font-size:10px;color:var(--text-muted)">{cld_n} samples · 24h</div>
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
      <div style="margin-bottom:12px;">
        <div class="section-title" style="margin:0">Active Models (Top 5)</div>
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
<div class="page-container" id="dashboard-container">
  {status_bar}
  {funnel_block}
  {lat_esc_block}
  {models_block}
  {traces_block}
  {quota_table}
</div>
<script>
(function() {{
  let isPolling = false;
  async function refreshDashboard() {{
    if (document.hidden || isPolling) return;
    isPolling = true;
    try {{
      const res = await fetch(window.location.href, {{
        headers: {{ 'X-Requested-With': 'WaypostLivePoll' }}
      }});
      if (!res.ok) return;
      const htmlText = await res.text();
      const parser = new DOMParser();
      const doc = parser.parseFromString(htmlText, 'text/html');
      const newContainer = doc.getElementById('dashboard-container');
      const currentContainer = document.getElementById('dashboard-container');
      if (newContainer && currentContainer) {{
        // Preserve accordion open state
        const openSet = new Set();
        currentContainer.querySelectorAll('details[open]').forEach(d => {{
          const sumText = d.querySelector('summary') ? d.querySelector('summary').textContent.trim() : '';
          if (sumText) openSet.add(sumText);
        }});
        newContainer.querySelectorAll('details').forEach(d => {{
          const sumText = d.querySelector('summary') ? d.querySelector('summary').textContent.trim() : '';
          if (sumText && openSet.has(sumText)) {{
            d.open = true;
          }}
        }});
        currentContainer.innerHTML = newContainer.innerHTML;
      }}
    }} catch (e) {{
      // Ignore background fetch errors
    }} finally {{
      isPolling = false;
    }}
  }}
  // Auto-poll every 4 seconds
  setInterval(refreshDashboard, 4000);
}})();
</script>
</body></html>"""


def render_setup_html(data: dict[str, Any]) -> str:
    """Renders comprehensive Setup tab per Spec v5 Section 2.2."""
    subsystems = data.get("subsystems", [])
    engines = data.get("engines", [])
    keys_health = data.get("keys_health", [])
    data_health = data.get("data_health", {})
    beta_metric = data.get("beta_metric", {})
    pricing = data.get("pricing", {})
    pricing_coverage = float(pricing.get("usage_coverage_pct", 0.0) or 0.0)

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
        used, cap = e.get("mem_used_gb"), e.get("mem_limit_gb")
        # Physical footprint, not RSS — see waypost/sysmem.py. Still
        # "not reported" when nothing could be measured: an engine nobody
        # measured must not be dressed up as a zero.
        if used is None:
            mem = "Memory: not reported"
        elif cap:
            mem = f"Memory: {used:.1f} GB of {cap:.0f} GB RAM"
        else:
            mem = f"Memory: {used:.1f} GB"
        engine_cards.append(
            f"""
        <div style="background:var(--bg-subtle);padding:14px;border-radius:var(--radius-sm)">
          <div style="font-weight:700;margin-bottom:4px">{html.escape(e.get('name', ''))}</div>
          <div style="font-size:12px;color:var(--text-secondary);margin-bottom:2px">Model: <b>{html.escape(e.get('model', ''))}</b></div>
          <div class="mono" style="font-size:12px">{html.escape(mem)}</div>
          <div style="font-size:11px;color:var(--text-muted);margin-top:4px">Status: {html.escape(e.get('warmup_status', 'ready'))}</div>
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
<div class="page-container" id="setup-container">

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
    <div class="section-title">Local Engines</div>
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
    <div style="text-align:center;font-size:12px;color:var(--text-secondary);border-top:1px solid var(--border-light);padding-top:10px;">
      Exact token usage coverage: <b>{pricing_coverage:.1f}%</b>
      ({pricing.get('measured_requests', 0):,}/{pricing.get('total_requests', 0):,} requests)
      · baseline: <b>{html.escape(str(pricing.get('baseline', 'Waypost commercial baseline v1')))}</b>
    </div>
  </div>

</div>
<script>
(function() {{
  let isPolling = false;
  async function refreshSetup() {{
    if (document.hidden || isPolling) return;
    isPolling = true;
    try {{
      const res = await fetch(window.location.href, {{
        headers: {{ 'X-Requested-With': 'WaypostLivePoll' }}
      }});
      if (!res.ok) return;
      const htmlText = await res.text();
      const parser = new DOMParser();
      const doc = parser.parseFromString(htmlText, 'text/html');
      const newContainer = doc.getElementById('setup-container');
      const currentContainer = document.getElementById('setup-container');
      if (newContainer && currentContainer) {{
        currentContainer.innerHTML = newContainer.innerHTML;
      }}
    }} catch (e) {{
      // Ignore background fetch errors
    }} finally {{
      isPolling = false;
    }}
  }}
  // Auto-poll every 5 seconds
  setInterval(refreshSetup, 5000);
}})();
</script>
</body></html>"""


def render_providers_html(data: dict[str, Any]) -> str:
    """Renders an ultra-minimalist Providers & Keys interface."""
    providers = data.get("providers", [])
    summary = data.get("summary", {})
    total_provs = summary.get("total_providers", len(providers))
    configured_keys = summary.get("configured_keys", 0)

    rows = []
    for p in providers:
        p_id = html.escape(p.get("id", ""))
        name = html.escape(p.get("name", ""))
        env_var = html.escape(p.get("env_var", ""))
        doc_url = html.escape(p.get("doc_url", ""))
        has_key = p.get("has_key", False)
        masked = html.escape(p.get("masked_key", ""))

        dot_style = (
            "background:var(--green);box-shadow:0 0 0 2px rgba(42,122,76,0.15);"
            if has_key
            else "background:var(--border);"
        )
        remove_style = "" if has_key else "display:none;"

        link_html = (
            f"""<a href="{doc_url}" target="_blank" rel="noopener noreferrer" class="provider-link-arrow" title="Get API key for {name} &rarr;" style="color:var(--text-muted);display:inline-flex;align-items:center;text-decoration:none;transition:all 0.15s ease;padding:2px;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 17l9.2-9.2M17 17V7H7"/></svg></a>"""
            if doc_url
            else ""
        )

        rows.append(
            f"""
      <div class="prov-row" id="row-{p_id}" style="display:flex;align-items:center;justify-content:space-between;padding:12px 24px;border-bottom:1px solid var(--border-light);gap:20px;">
        <div style="display:flex;align-items:center;gap:10px;width:190px;min-width:190px;flex-shrink:0;">
          <span class="status-dot" id="dot-{p_id}" style="width:7px;height:7px;border-radius:50%;{dot_style}flex-shrink:0;"></span>
          <span style="font-size:14px;font-weight:600;color:var(--text);white-space:nowrap;">{name}</span>
          {link_html}
        </div>

        <div style="display:flex;align-items:center;gap:10px;flex:1;">
          <input type="password" id="input-{p_id}" class="mono" style="padding:8px 14px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--bg);color:var(--text);font-size:12.5px;width:100%;outline:none;transition:border-color 0.15s ease;" placeholder="{masked or 'Paste ' + env_var + '...'}" autocomplete="off" onkeydown="if(event.key==='Enter') saveKey('{p_id}', '{env_var}')" />
          <button type="button" class="btn-save" style="padding:8px 18px;background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:var(--radius-sm);font-size:12.5px;font-weight:600;cursor:pointer;white-space:nowrap;transition:all 0.15s ease;" onclick="saveKey('{p_id}', '{env_var}')">Save</button>
          <button type="button" class="btn-remove" id="remove-{p_id}" style="padding:8px 14px;background:var(--card);color:var(--text-muted);border:1px solid var(--border);border-radius:var(--radius-sm);font-size:12.5px;font-weight:600;cursor:pointer;white-space:nowrap;transition:all 0.15s ease;{remove_style}" onclick="removeKey('{p_id}', '{env_var}', '{name}')">Remove</button>
        </div>
      </div>
"""
        )

    rows_html = "\n".join(rows)

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Waypost — Providers</title>
<link rel=icon href="{FAVICON}">
<style>
{theme_css()}
.prov-row:last-child {{
  border-bottom: none !important;
}}
.provider-link-arrow:hover {{
  color: var(--terracotta) !important;
  transform: translate(1px, -1px);
}}
.btn-save:hover {{
  background: var(--bg-subtle) !important;
  border-color: var(--text-muted) !important;
}}
input:focus {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(217, 119, 87, 0.15) !important;
}}
</style>
</head>
<body>
{nav_header('providers')}
<div class="page-container" id="providers-container">

  <div class="card" style="padding:0;border-radius:var(--radius-md);overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.03);">
    <div style="padding:14px 24px;border-bottom:1px solid var(--border-light);display:flex;align-items:center;justify-content:space-between;background:var(--bg-subtle);">
      <span style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-secondary);">Providers & API Keys</span>
      <span style="font-size:12px;color:var(--text-muted);font-family:var(--font-mono);font-weight:600;">{configured_keys} / {total_provs} active</span>
    </div>

    <div>
      {rows_html}
    </div>
  </div>

</div>

<div id="toast" style="position:fixed;bottom:24px;right:24px;background:var(--card);border:1px solid var(--border);box-shadow:0 8px 24px rgba(0,0,0,0.15);border-radius:var(--radius-md);padding:12px 18px;font-size:13px;font-weight:500;color:var(--text);transform:translateY(100px);opacity:0;transition:all 0.25s cubic-bezier(0.16, 1, 0.3, 1);z-index:1000;"></div>

<script>
function showToast(msg, isErr) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.borderColor = isErr ? 'var(--red)' : 'var(--accent)';
  t.style.transform = 'translateY(0)';
  t.style.opacity = '1';
  setTimeout(() => {{
    t.style.transform = 'translateY(100px)';
    t.style.opacity = '0';
  }}, 3500);
}}

async function saveKey(pId, envVar) {{
  const inp = document.getElementById('input-' + pId);
  const val = inp ? inp.value.trim() : '';
  // The field is cleared after every save, so an empty one is the normal
  // resting state — it must never be read as "delete this key".
  if (!val) {{
    showToast('Nothing to save — paste a key first', true);
    return;
  }}
  try {{
    const res = await fetch('/v1/providers/keys', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ env_var: envVar, api_key: val, provider: pId }})
    }});
    const d = await res.json();
    if (res.ok) {{
      showToast(d.message || 'Key saved successfully');
      inp.value = '';
      if (d.masked_key) inp.placeholder = d.masked_key;
      const dotEl = document.getElementById('dot-' + pId);
      if (dotEl) {{
        dotEl.style.background = 'var(--green)';
        dotEl.style.boxShadow = '0 0 0 2px rgba(42,122,76,0.15)';
      }}
      const rmEl = document.getElementById('remove-' + pId);
      if (rmEl) rmEl.style.display = '';
    }} else {{
      showToast(d.detail || 'Failed to update key', true);
    }}
  }} catch (err) {{
    showToast('Error: ' + err.message, true);
  }}
}}

async function removeKey(pId, envVar, name) {{
  // Deleting a key wipes it from .env, the Keychain and the running
  // process at once. It is not recoverable from here — ask first.
  if (!confirm('Remove the ' + name + ' key (' + envVar + ')? '
      + 'It will be deleted from .env and the macOS Keychain, '
      + 'and you will have to paste it again to restore it.')) return;
  try {{
    const res = await fetch('/v1/providers/keys', {{
      method: 'DELETE',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ env_var: envVar }})
    }});
    const d = await res.json();
    if (res.ok) {{
      showToast(d.message || 'Key removed');
      const inp = document.getElementById('input-' + pId);
      if (inp) {{ inp.value = ''; inp.placeholder = 'Paste ' + envVar + '...'; }}
      const dotEl = document.getElementById('dot-' + pId);
      if (dotEl) {{
        dotEl.style.background = 'var(--border)';
        dotEl.style.boxShadow = 'none';
      }}
      const rmEl = document.getElementById('remove-' + pId);
      if (rmEl) rmEl.style.display = 'none';
    }} else {{
      showToast(d.detail || 'Failed to remove key', true);
    }}
  }} catch (err) {{
    showToast('Error: ' + err.message, true);
  }}
}}
</script>
</body></html>"""


def render_chat_html() -> str:
    """Renders the Anthropic Claude / Waypost-styled interactive web chat interface."""
    tmpl = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Waypost — Chat</title>
<link rel=icon href="__FAVICON__">
<style>
__THEME_CSS__

html, body {
  height: 100%;
  overflow: hidden;
}
.chat-app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg);
  position: relative;
}
.chat-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 24px;
  background: var(--bg);
  border-bottom: 1px solid var(--border-light);
  gap: 12px;
  flex-wrap: wrap;
}
.model-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 5px 12px;
  font-size: 12.5px;
  font-weight: 500;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.model-indicator {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--terracotta);
  flex-shrink: 0;
}
.model-select {
  background: transparent;
  border: none;
  font-size: 12.5px;
  font-weight: 500;
  color: var(--text);
  outline: none;
  cursor: pointer;
  font-family: var(--font-sans);
}
.chat-options {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 12.5px;
  color: var(--text-secondary);
}
.chat-options label {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  user-select: none;
  font-weight: 500;
}
.chat-options input[type="checkbox"] {
  accent-color: var(--terracotta);
  cursor: pointer;
}
.btn-icon {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  padding: 5px 14px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.15s ease;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.btn-icon:hover {
  background: var(--bg-subtle);
}

/* Message stream */
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px 20px;
  scroll-behavior: smooth;
}
.messages-inner {
  max-width: 820px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* Welcome Hero - Double Substrate Badge */
.welcome-hero {
  text-align: center;
  padding: 36px 20px 20px;
  margin: auto 0;
}
.welcome-badge-wrap {
  margin-bottom: 22px;
  display: inline-block;
}
.welcome-badge-outer {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 96px;
  height: 96px;
  border-radius: 26px;
  background: var(--card);
  border: 1px solid var(--border);
  box-shadow: 0 10px 28px -4px rgba(0,0,0,0.06), 0 2px 8px -1px rgba(0,0,0,0.03);
  padding: 8px;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}
.welcome-badge-outer:hover {
  transform: translateY(-2px) scale(1.02);
  box-shadow: 0 16px 36px -4px rgba(0,0,0,0.08), 0 4px 12px -2px rgba(0,0,0,0.04);
}
.welcome-badge-inner {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  border-radius: 18px;
  background: #1c1b18;
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: inset 0 1px 1px rgba(255,255,255,0.15), 0 2px 8px rgba(0,0,0,0.25);
}
.welcome-badge-inner img {
  width: 68px;
  height: 68px;
  object-fit: contain;
}
.welcome-title {
  font-family: var(--font-sans);
  font-size: 26px;
  font-weight: 700;
  color: var(--text);
  margin: 0 0 10px;
  letter-spacing: -0.02em;
}
.welcome-sub {
  color: var(--text-secondary);
  font-size: 13.5px;
  max-width: 560px;
  margin: 0 auto 34px;
  line-height: 1.5;
}
.prompt-chips {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
  max-width: 740px;
  margin: 0 auto;
}
@media (max-width: 680px) {
  .prompt-chips {
    grid-template-columns: 1fr;
  }
}
.prompt-chip {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 16px 18px;
  font-size: 13px;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 14px;
  text-align: left;
  cursor: pointer;
  transition: all 0.15s ease;
  box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.prompt-chip:hover {
  border-color: var(--terracotta);
  background: var(--card);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.prompt-chip-icon {
  color: var(--terracotta);
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}
.prompt-chip-text {
  flex: 1;
  line-height: 1.4;
  font-weight: 500;
}

/* Message items */
.message-row {
  display: flex;
  gap: 14px;
  width: 100%;
}
.message-row.user {
  justify-content: flex-end;
}
.message-bubble {
  max-width: 85%;
  font-size: 14px;
  line-height: 1.6;
}
.message-row.user .message-bubble {
  background: var(--card);
  border: 1px solid var(--border);
  padding: 12px 18px;
  border-radius: 16px 16px 4px 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  color: var(--text);
  white-space: pre-wrap;
  font-weight: 450;
}
.message-row.assistant {
  justify-content: flex-start;
  align-items: flex-start;
}
.assistant-avatar {
  /* The first line of the reply is 14px * 1.6 = 22.4px tall. A 22px mark
     with no top offset puts its centre on that line's centre; the old
     32px box sat ~7px below the text it belongs to. No background or
     border: the mark is a bare outline that inherits this colour. */
  width: 22px;
  height: 22px;
  color: var(--text);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.message-row.assistant .message-bubble {
  flex: 1;
  color: var(--text);
}
.router-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 4px 10px;
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 10px;
  font-family: var(--font-mono);
}
.router-dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--text-secondary);
}
.router-pill b { color: var(--text); }

/* Markdown typography */
.message-bubble h1, .message-bubble h2, .message-bubble h3 {
  font-family: var(--font-sans);
  margin: 16px 0 8px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
}
.message-bubble h1 { font-size: 19px; }
.message-bubble h2 { font-size: 16px; }
.message-bubble h3 { font-size: 14px; }
.message-bubble p { margin: 0 0 12px; }
.message-bubble p:last-child { margin-bottom: 0; }
.message-bubble ul, .message-bubble ol {
  margin: 8px 0 12px;
  padding-left: 22px;
}
.message-bubble li { margin-bottom: 4px; }
.message-bubble blockquote {
  margin: 12px 0;
  padding: 6px 14px;
  border-left: 3px solid var(--text-muted);
  color: var(--text-secondary);
  background: var(--bg-subtle);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}
.message-bubble code {
  font-family: var(--font-mono);
  font-size: 12.5px;
  background: var(--bg-subtle);
  border: 1px solid var(--border-light);
  padding: 2px 6px;
  border-radius: 4px;
}
.code-block-wrap {
  position: relative;
  margin: 14px 0;
  background: #18181b;
  border-radius: var(--radius-md);
  border: 1px solid #27272a;
  overflow: hidden;
}
.code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  background: #27272a;
  font-family: var(--font-mono);
  font-size: 11px;
  color: #a1a1aa;
  border-bottom: 1px solid #3f3f46;
}
.btn-copy {
  background: transparent;
  border: 1px solid #3f3f46;
  border-radius: 4px;
  color: #d4d4d8;
  font-size: 10.5px;
  padding: 2px 8px;
  cursor: pointer;
}
.btn-copy:hover {
  background: #3f3f46;
  color: #fff;
}
.code-block-wrap pre {
  margin: 0;
  padding: 14px 16px;
  overflow-x: auto;
  background: transparent;
}
.code-block-wrap code {
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.55;
  color: #f4f4f5;
  background: transparent;
  padding: 0;
  border: none;
  white-space: pre;
}

/* Thinking accordion */
.thinking-box {
  margin-bottom: 12px;
  background: var(--bg-subtle);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
}
.thinking-box summary {
  padding: 6px 12px;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--text-secondary);
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid var(--border-light);
}
.thinking-box summary:hover {
  color: var(--text);
}
.thinking-content {
  padding: 10px 14px;
  white-space: pre-wrap;
  font-family: var(--font-mono);
  font-size: 11.5px;
  line-height: 1.5;
  color: var(--text-muted);
  max-height: 200px;
  overflow-y: auto;
}

/* Attachments & input bar */
.chat-bottom {
  padding: 12px 20px 20px;
  background: var(--bg);
  border-top: 1px solid var(--border-light);
  position: relative;
}
.attachments-tray {
  max-width: 820px;
  margin: 0 auto 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 0 4px;
}
.attachment-chip {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 4px 10px;
  font-size: 12px;
  color: var(--text);
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  max-width: 260px;
}
.att-ext-tag {
  font-family: var(--font-mono);
  font-size: 9.5px;
  font-weight: 700;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: 1px 4px;
  border-radius: 3px;
  letter-spacing: 0.02em;
  line-height: 1.2;
  flex-shrink: 0;
}
.attachment-chip .att-img-thumb {
  width: 22px;
  height: 22px;
  border-radius: 4px;
  object-fit: cover;
  flex-shrink: 0;
}
.attachment-chip .att-name {
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.attachment-chip .att-size {
  color: var(--text-muted);
  font-size: 11px;
  flex-shrink: 0;
}
.attachment-chip .btn-remove-att {
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 14px;
  cursor: pointer;
  padding: 0 2px;
  line-height: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}
.attachment-chip .btn-remove-att:hover {
  color: var(--red);
}
.btn-attach {
  width: 32px;
  height: 32px;
  border-radius: var(--radius-sm);
  background: transparent;
  border: none;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.15s ease;
  flex-shrink: 0;
  margin-bottom: 1px;
  user-select: none;
}
.btn-attach:hover {
  background: var(--bg-subtle);
  color: var(--text);
}
.btn-attach svg {
  width: 17px;
  height: 17px;
}
.drag-drop-overlay {
  position: absolute;
  top: 8px;
  left: 8px;
  right: 8px;
  bottom: 8px;
  background: rgba(247, 246, 242, 0.96);
  backdrop-filter: blur(4px);
  border: 2px dashed var(--border);
  border-radius: var(--radius-md);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  z-index: 500;
  pointer-events: none;
  color: var(--text);
  font-size: 13.5px;
  font-weight: 600;
}
.user-attachments {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
}
.user-att-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 3px 8px;
  font-size: 11.5px;
  font-family: var(--font-mono);
  color: var(--text-secondary);
}
.user-att-thumb {
  max-width: 220px;
  max-height: 160px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  display: block;
  margin-top: 6px;
}
.input-container {
  max-width: 820px;
  margin: 0 auto;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 8px 12px 8px 10px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  display: flex;
  align-items: flex-end;
  gap: 8px;
  transition: all 0.15s ease;
  position: relative;
}
.input-container:focus-within {
  border-color: var(--accent);
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
#chat-input {
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
  padding: 4px 0;
}
.btn-send {
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
  margin-bottom: 1px;
}
.btn-send:hover {
  opacity: 0.9;
  transform: scale(1.04);
}
.btn-send:disabled {
  background: var(--border);
  color: var(--text-muted);
  cursor: not-allowed;
  transform: none;
}
.input-hint {
  text-align: center;
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 8px;
}
.cursor-blink {
  display: inline-block;
  width: 6px;
  height: 14px;
  background: var(--text);
  margin-left: 2px;
  vertical-align: -2px;
  animation: blink 0.9s infinite;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
</style></head><body>
<div class="chat-app" id="chat-app">
  __NAV_HEADER__
  <div class="chat-topbar">
    <div class="model-pill">
      <span class="model-indicator"></span>
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
      <label title="Strict Privacy: Keep all prompts local and disable cloud routing"><input type="checkbox" id="opt-privacy"> Strict Privacy</label>
      <button class="btn-icon" id="btn-clear" title="Clear chat (Cmd+K)">Clear</button>
    </div>
  </div>

  <div class="chat-messages" id="messages-container">
    <div class="messages-inner" id="messages-list">
      <div class="welcome-hero" id="welcome-hero">
        <div class="welcome-badge-wrap">
          <div class="welcome-badge-outer">
            <div class="welcome-badge-inner">
              <img src="__GLYPH_URI__" width="70" height="70" alt="Waypost Logo">
            </div>
          </div>
        </div>
        <h2 class="welcome-title">How can I help you today?</h2>
        <p class="welcome-sub">Waypost routes prompts across local and cloud models, choosing the fastest free engine with automatic escalation.</p>
        <div class="prompt-chips">
          <div class="prompt-chip" onclick="usePrompt(this.querySelector('.prompt-chip-text').innerText)">
            <span class="prompt-chip-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
            </span>
            <span class="prompt-chip-text">Compare Rust vs Go for high-throughput networking services</span>
          </div>
          <div class="prompt-chip" onclick="usePrompt(this.querySelector('.prompt-chip-text').innerText)">
            <span class="prompt-chip-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/><line x1="14" y1="4" x2="10" y2="20"/></svg>
            </span>
            <span class="prompt-chip-text">Write a Python decorator to rate-limit async functions with token buckets</span>
          </div>
          <div class="prompt-chip" onclick="usePrompt(this.querySelector('.prompt-chip-text').innerText)">
            <span class="prompt-chip-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
            </span>
            <span class="prompt-chip-text">Explain transformer self-attention and KV cache mechanisms simply</span>
          </div>
          <div class="prompt-chip" onclick="usePrompt(this.querySelector('.prompt-chip-text').innerText)">
            <span class="prompt-chip-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>
            </span>
            <span class="prompt-chip-text">How do circuit breakers prevent cascading failures in microservices?</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="chat-bottom" id="chat-bottom">
    <div class="attachments-tray" id="attachments-tray" style="display:none"></div>
    <div class="input-container">
      <label for="file-input" class="btn-attach" id="btn-attach" title="Attach files">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.49-8.49a4 4 0 0 1 5.66 5.66l-8.49 8.49a2 2 0 0 1-2.83-2.83l7.78-7.78"/></svg>
      </label>
      <input type="file" id="file-input" multiple style="position:absolute;left:-9999px;opacity:0;width:1px;height:1px;">
      <textarea id="chat-input" placeholder="Message Waypost or drop files..." rows="1"></textarea>
      <button class="btn-send" id="btn-send" title="Send message (Enter)">↑</button>
    </div>
    <div class="input-hint">Waypost Smart Router · Enter to send · Shift+Enter for new line · Drag & drop or click Attach</div>
  </div>
  <div id="drag-drop-overlay" class="drag-drop-overlay" style="display:none">
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="color:var(--text-secondary)"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
    <span>Drop files here to attach to conversation</span>
  </div>
</div>

<script>
window.onerror = function(msg, url, line, col, error) {
  const div = document.createElement("div");
  div.style = "position:fixed;top:10px;left:10px;z-index:9999;background:rgba(255,0,0,0.8);color:white;padding:10px;font-family:monospace;border-radius:4px;max-width:80%;word-break:break-all;";
  div.innerText = "Error: " + msg + " at " + line + ":" + col;
  document.body.appendChild(div);
};
window.addEventListener("unhandledrejection", function(e) {
  const div = document.createElement("div");
  div.style = "position:fixed;top:60px;left:10px;z-index:9999;background:rgba(255,0,0,0.8);color:white;padding:10px;font-family:monospace;border-radius:4px;max-width:80%;word-break:break-all;";
  div.innerText = "Unhandled Rejection: " + (e.reason && e.reason.message ? e.reason.message : e.reason);
  document.body.appendChild(div);
});
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
const attachmentsTray = document.getElementById('attachments-tray');
const fileInput = document.getElementById('file-input');
const btnAttach = document.getElementById('btn-attach');
const dragDropOverlay = document.getElementById('drag-drop-overlay');
const chatApp = document.getElementById('chat-app');

let history = [];
let transcript = [];
let attachedFiles = [];
let isGenerating = false;
let abortController = null;

// The nav tabs are ordinary links, so moving to Dashboard and back reloads
// the page and drops every JS variable with it. sessionStorage keeps the
// conversation across that reload and lets go of it when the window closes:
// a transcript should survive a tab switch, not outlive the session on disk.
const SESSION_KEY = 'waypost.chat.v1';

function stripImageParts(msg) {
  if (!msg || !Array.isArray(msg.content)) return msg;
  const textOnly = msg.content.filter(p => p && p.type === 'text');
  return { role: msg.role, content: textOnly.map(p => p.text).join(' ') };
}

function saveSession() {
  try {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify({ history, transcript }));
  } catch (err) {
    // Base64 images exhaust the quota quickly. Drop the pixels and keep the
    // words — losing a thumbnail beats losing the conversation.
    try {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify({
        history: history.map(stripImageParts),
        transcript: transcript.map(t => Object.assign({}, t, {
          attachments: (t.attachments || []).map(a => Object.assign({}, a, { dataUrl: null }))
        }))
      }));
    } catch (err2) {
      // Private window, disabled storage, or still too large: carry on
      // without persistence rather than breaking the chat.
    }
  }
}

function restoreSession() {
  let saved = null;
  try {
    saved = JSON.parse(sessionStorage.getItem(SESSION_KEY) || 'null');
  } catch (err) {
    saved = null;
  }
  if (!saved || !Array.isArray(saved.transcript) || saved.transcript.length === 0) return;
  history = Array.isArray(saved.history) ? saved.history : [];
  transcript = saved.transcript;
  if (welcomeHero && welcomeHero.parentNode) {
    welcomeHero.parentNode.removeChild(welcomeHero);
  }
  for (const item of transcript) {
    if (item.role === 'user') {
      appendUserMessage(item.text || '', item.attachments || []);
    } else {
      const msg = createAssistantMessage();
      msg.contentEl.innerHTML = item.text
        ? renderMarkdown(item.text)
        : '<span style="color:var(--text-muted)">(Empty response from model)</span>';
      if (item.router) msg.metaEl.innerHTML = routerPillHtml(item.router);
    }
  }
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateModelIndicator() {
  const selectedOpt = modelSelect.options[modelSelect.selectedIndex];
  const optGroup = selectedOpt ? selectedOpt.parentElement : null;
  const modelVal = modelSelect.value;
  const ind = document.querySelector('.model-indicator');

  let mode = 'auto';
  if (modelVal === 'auto' || modelVal.startsWith('Tier')) {
    mode = 'auto';
    if (ind) ind.style.background = 'var(--terracotta)';
  } else if (optGroup && optGroup.id === 'group-cloud') {
    mode = 'cloud';
    if (ind) ind.style.background = 'var(--blue)';
  } else {
    mode = 'local';
    if (ind) ind.style.background = 'var(--green)';
  }

  fetch('/v1/active-model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: modelVal, mode: mode })
  }).catch(() => {});
}

modelSelect.addEventListener('change', updateModelIndicator);

// Populate specific models from /v1/pricing
fetch('/v1/pricing').then(r => r.json()).then(res => {
  if (res && res.models) {
    res.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.id + (m.free ? ' · free' : '');
      if (m.is_local) {
        groupLocal.appendChild(opt);
      } else {
        groupCloud.appendChild(opt);
      }
    });
    updateModelIndicator();
  }
}).catch(() => {});

// Auto-expand textarea
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault();
    clearChat();
  }
});

btnSend.addEventListener('click', () => {
  if (isGenerating) {
    stopGeneration();
  } else {
    sendMessage();
  }
});

btnClear.addEventListener('click', clearChat);

if (fileInput) {
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(Array.from(e.target.files));
      fileInput.value = '';
    }
  });
}

// Drag & drop handlers
let dragCounter = 0;
if (chatApp) {
  ['dragenter', 'dragover'].forEach(eventName => {
    chatApp.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter++;
      if (dragDropOverlay) dragDropOverlay.style.display = 'flex';
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    chatApp.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter--;
      if (dragCounter <= 0 && dragDropOverlay) {
        dragCounter = 0;
        dragDropOverlay.style.display = 'none';
      }
    }, false);
  });

  chatApp.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter = 0;
    if (dragDropOverlay) dragDropOverlay.style.display = 'none';
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(Array.from(e.dataTransfer.files));
    }
  });
}

// Clipboard paste handler
window.addEventListener('paste', (e) => {
  if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
    addFiles(Array.from(e.clipboardData.files));
  }
});

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function getFileBadge(name) {
  const ext = (name.split('.').pop() || '').toUpperCase().slice(0, 4) || 'FILE';
  return `<span class="att-ext-tag">${escapeHtml(ext)}</span>`;
}

function addFiles(files) {
  for (const file of files) {
    const isImage = file.type.startsWith('image/');
    const item = {
      id: 'att_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
      file: file,
      name: file.name,
      size: file.size,
      type: file.type,
      isImage: isImage,
      dataUrl: null,
      textContent: null,
    };

    if (isImage) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        item.dataUrl = ev.target.result;
        renderAttachmentsTray();
      };
      reader.readAsDataURL(file);
    } else {
      if (file.size < 10 * 1024 * 1024) {
        const reader = new FileReader();
        reader.onload = (ev) => {
          item.textContent = ev.target.result;
        };
        reader.readAsText(file);
      }
    }
    attachedFiles.push(item);
  }
  renderAttachmentsTray();
}

window.removeAttachment = function(index) {
  attachedFiles.splice(index, 1);
  renderAttachmentsTray();
};

function renderAttachmentsTray() {
  if (!attachmentsTray) return;
  if (attachedFiles.length === 0) {
    attachmentsTray.style.display = 'none';
    attachmentsTray.innerHTML = '';
    return;
  }
  attachmentsTray.style.display = 'flex';
  attachmentsTray.innerHTML = attachedFiles.map((att, idx) => {
    const icon = (att.isImage && att.dataUrl)
      ? `<img class="att-img-thumb" src="${att.dataUrl}" alt="">`
      : getFileBadge(att.name);
    return `
      <div class="attachment-chip" title="${escapeHtml(att.name)} (${formatBytes(att.size)})">
        ${icon}
        <span class="att-name">${escapeHtml(att.name)}</span>
        <span class="att-size">${formatBytes(att.size)}</span>
        <button type="button" class="btn-remove-att" onclick="removeAttachment(${idx})" title="Remove">×</button>
      </div>
    `;
  }).join('');
}

function usePrompt(text) {
  chatInput.value = text;
  sendMessage();
}

function clearChat() {
  history = [];
  transcript = [];
  try { sessionStorage.removeItem(SESSION_KEY); } catch (err) {}
  attachedFiles = [];
  renderAttachmentsTray();
  messagesList.innerHTML = '';
  if (welcomeHero) messagesList.appendChild(welcomeHero);
  chatInput.focus();
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  const s = typeof str === 'string' ? str : String(str);
  return s.replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#039;');
}

function extractText(obj) {
  if (!obj) return '';
  if (typeof obj === 'string') return obj;
  if (Array.isArray(obj)) {
    return obj.map(item => {
      if (typeof item === 'string') return item;
      if (item && item.text) return item.text;
      if (item && item.content) return extractText(item.content);
      return '';
    }).join('');
  }
  if (obj.text) return typeof obj.text === 'string' ? obj.text : extractText(obj.text);
  if (obj.content) return typeof obj.content === 'string' ? obj.content : extractText(obj.content);
  if (obj.reason) return typeof obj.reason === 'string' ? obj.reason : extractText(obj.reason);
  if (obj.reasoning) return typeof obj.reasoning === 'string' ? obj.reasoning : extractText(obj.reasoning);
  if (obj.reasoning_content) return typeof obj.reasoning_content === 'string' ? obj.reasoning_content : extractText(obj.reasoning_content);
  return '';
}

// Lightweight Markdown renderer
function renderMarkdown(md) {
  if (md === null || md === undefined) return '';
  let str = typeof md === 'string' ? md : String(md);
  if (!str) return '';

  const codeBlockCount = (str.match(/```/g) || []).length;
  if (codeBlockCount % 2 === 1) {
    str = str + '\n```';
  }

  // 1. Code blocks
  let text = str.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, function(match, lang, code) {
    const l = lang ? lang.trim() : 'text';
    const escaped = escapeHtml(code.replace(/\n$/, ''));
    return '<div class="code-block-wrap"><div class="code-header"><span>' + l + '</span><button class="btn-copy" onclick="copyCode(this)">Copy</button></div><pre><code>' + escaped + '</code></pre></div>';
  });

  // 2. Inline code
  text = text.replace(/`([^`]+)`/g, function(match, code) {
    return '<code>' + escapeHtml(code) + '</code>';
  });

  // 3. Headings
  text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>')
             .replace(/^## (.*$)/gim, '<h2>$1</h2>')
             .replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // 4. Bold & italic
  text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
             .replace(/\*(.*?)\*/g, '<em>$1</em>');

  // 5. Blockquotes
  text = text.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

  // 6. Lists
  text = text.replace(/^\s*[-*+] (.*$)/gim, '<li>$1</li>');
  text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // 7. Paragraphs
  const paragraphs = text.split(/\n\n+/);
  return paragraphs.map(p => {
    p = p.trim();
    if (!p) return '';
    if (p.startsWith('<div') || p.startsWith('<h') || p.startsWith('<ul') || p.startsWith('<blockquote') || p.startsWith('<details')) {
      return p;
    }
    return '<p>' + p.replace(/\n/g, '<br>') + '</p>';
  }).join('');
}

function routerPillHtml(meta) {
  const prov = meta.provider || '—';
  const mName = meta.model || '—';
  const tier = meta.complexity_tier || '—';
  const lat = meta.latency_ms ? meta.latency_ms + 'ms' : '';
  const cached = meta.cache === 'hit' ? 'cached' : 'fresh';
  return `
        <div class="router-pill">
          <span class="router-dot"></span>
          <span><b>Router:</b> ${escapeHtml(prov)} · ${escapeHtml(mName)} · Tier ${escapeHtml(tier)} ${lat ? '· ' + lat : ''} · ${cached}</span>
        </div>
      `;
}

function copyCode(btn) {
  const pre = btn.parentElement.nextElementSibling;
  if (pre) {
    navigator.clipboard.writeText(pre.innerText).then(() => {
      const orig = btn.innerText;
      btn.innerText = 'Copied!';
      setTimeout(() => { btn.innerText = orig; }, 1500);
    });
  }
}

function appendUserMessage(content, attachments) {
  if (welcomeHero && welcomeHero.parentNode) {
    welcomeHero.parentNode.removeChild(welcomeHero);
  }
  const row = document.createElement('div');
  row.className = 'message-row user';

  let attHtml = '';
  if (attachments && attachments.length > 0) {
    attHtml += '<div class="user-attachments">';
    for (const a of attachments) {
      if (a.isImage && a.dataUrl) {
        attHtml += `<div><span class="user-att-pill">${getFileBadge(a.name)} ${escapeHtml(a.name)} (${formatBytes(a.size)})</span><img src="${a.dataUrl}" class="user-att-thumb" alt="${escapeHtml(a.name)}"></div>`;
      } else {
        attHtml += `<span class="user-att-pill">${getFileBadge(a.name)} ${escapeHtml(a.name)} (${formatBytes(a.size)})</span>`;
      }
    }
    attHtml += '</div>';
  }

  const textHtml = content ? escapeHtml(content) : '';
  row.innerHTML = `<div class="message-bubble">${attHtml}${textHtml}</div>`;
  messagesList.appendChild(row);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function createAssistantMessage() {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.innerHTML = `
    <div class="assistant-avatar">
      __AVATAR_MARK__
    </div>
    <div class="message-bubble"><div class="bubble-content"><span class="cursor-blink"></span></div><div class="bubble-meta"></div></div>
  `;
  messagesList.appendChild(row);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return {
    row: row,
    contentEl: row.querySelector('.bubble-content'),
    metaEl: row.querySelector('.bubble-meta')
  };
}

function stopGeneration() {
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
  setGenerating(false);
}

function setGenerating(gen) {
  isGenerating = gen;
  if (gen) {
    btnSend.textContent = '■';
    btnSend.title = 'Stop generating';
  } else {
    btnSend.textContent = '↑';
    btnSend.title = 'Send message';
  }
}

async function sendMessage() {
  const text = chatInput.value.trim();
  const currentAtts = [...attachedFiles];
  if ((!text && currentAtts.length === 0) || isGenerating) return;

  chatInput.value = '';
  chatInput.style.height = 'auto';
  attachedFiles = [];
  renderAttachmentsTray();

  appendUserMessage(text, currentAtts);

  let messageText = text;
  const fileContextParts = [];

  for (const att of currentAtts) {
    if (!att.isImage && att.textContent) {
      const ext = (att.name.split('.').pop() || '').toLowerCase();
      fileContextParts.push(`[Attached File: ${att.name} (${formatBytes(att.size)})]\n\`\`\`${ext}\n${att.textContent}\n\`\`\``);
    }
  }

  if (fileContextParts.length > 0) {
    messageText = (text ? text + '\n\n' : '') + fileContextParts.join('\n\n');
  }

  const hasImages = currentAtts.some(a => a.isImage && a.dataUrl);
  let userPayloadContent = messageText;

  if (hasImages) {
    const parts = [];
    if (messageText) {
      parts.push({ type: 'text', text: messageText });
    }
    for (const a of currentAtts) {
      if (a.isImage && a.dataUrl) {
        parts.push({
          type: 'image_url',
          image_url: { url: a.dataUrl }
        });
      }
    }
    userPayloadContent = parts;
  }

  history.push({ role: 'user', content: userPayloadContent });
  transcript.push({ role: 'user', text: text, attachments: currentAtts });
  saveSession();

  const isStream = optStream ? optStream.checked : true;
  const isThinking = optThinking ? optThinking.checked : false;
  const model = (modelSelect && modelSelect.value) ? modelSelect.value : 'auto';
  const privacy = (optPrivacy && optPrivacy.checked) ? 'strict' : 'default';

  const cleanMessages = history.filter(m => m && m.content);

  const payload = {
    model: model,
    messages: cleanMessages,
    stream: isStream,
    enable_thinking: isThinking,
    privacy: privacy,
  };

  const assistantMsg = createAssistantMessage();
  let fullContent = '';
  let routerMeta = null;

  setGenerating(true);
  abortController = new AbortController();

  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': isStream ? 'text/event-stream, application/json' : 'application/json'
      },
      body: JSON.stringify(payload),
      signal: abortController.signal
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      const errMsg = (errJson.error && errJson.error.message) ? errJson.error.message : response.statusText;
      assistantMsg.contentEl.innerHTML = '<span style="color:var(--red);background:var(--red-bg);padding:8px 12px;border-radius:6px;display:inline-block">Error: ' + escapeHtml(errMsg) + '</span>';
      setGenerating(false);
      return;
    }

    if (isStream) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const dataStr = trimmed.slice(5).trim();
          if (dataStr === '[DONE]') break;
          try {
            const chunk = JSON.parse(dataStr);
            if (chunk.error) {
              const errMsg = typeof chunk.error === 'string' ? chunk.error : (chunk.error.message || JSON.stringify(chunk.error));
              assistantMsg.contentEl.innerHTML = '<span style="color:var(--red);background:var(--red-bg);padding:8px 12px;border-radius:6px;display:inline-block">Error: ' + escapeHtml(errMsg) + '</span>';
              return;
            }
            const c0 = (chunk.choices && chunk.choices[0]) || {};
            const d = c0.delta || {};
            const delta = extractText(d) || (typeof d.content === 'string' ? d.content : '');
            if (delta) {
              fullContent += delta;
              assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent) + '<span class="cursor-blink"></span>';
              messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            if (chunk.router) {
              routerMeta = chunk.router;
            }
          } catch (e) {}
        }
      }
      if (!fullContent) {
        assistantMsg.contentEl.innerHTML = '<span style="color:var(--text-muted)">(Empty response from model)</span>';
      } else {
        assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent);
      }
    } else {
      const data = await response.json();
      if (data.error) {
        const errMsg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
        assistantMsg.contentEl.innerHTML = '<span style="color:var(--red);background:var(--red-bg);padding:8px 12px;border-radius:6px;display:inline-block">Error: ' + escapeHtml(errMsg) + '</span>';
        setGenerating(false);
        return;
      }
      const m0 = (data.choices && data.choices[0] && data.choices[0].message) || {};
      fullContent = extractText(m0) || (typeof m0.content === 'string' ? m0.content : '');
      routerMeta = data.router;
      if (!fullContent) {
        assistantMsg.contentEl.innerHTML = '<span style="color:var(--text-muted)">(Empty response from model)</span>';
      } else {
        assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent);
      }
    }

    if (fullContent && fullContent.trim()) {
      history.push({ role: 'assistant', content: fullContent });
    }

    if (routerMeta) {
      assistantMsg.metaEl.innerHTML = routerPillHtml(routerMeta);
    }

    transcript.push({ role: 'assistant', text: fullContent, router: routerMeta || null });
    saveSession();
  } catch (err) {
    if (err.name !== 'AbortError') {
      assistantMsg.contentEl.innerHTML = '<span style="color:var(--red);background:var(--red-bg);padding:8px 12px;border-radius:6px;display:inline-block">Connection error: ' + escapeHtml(err.message) + '</span>';
    } else {
      assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent) + ' <span style="color:var(--text-muted);font-size:12px">(stopped)</span>';
    }
  } finally {
    setGenerating(false);
    abortController = null;
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
}

restoreSession();
</script>
</body></html>"""
    return (
        tmpl.replace("__FAVICON__", FAVICON)
        .replace("__THEME_CSS__", theme_css())
        .replace("__NAV_HEADER__", nav_header("chat"))
        .replace("__AVATAR_MARK__", AVATAR_MARK)
        .replace("__LOGO_URI__", _LOGO_URI)
        .replace("__GLYPH_URI__", _GLYPH_URI)
    )
