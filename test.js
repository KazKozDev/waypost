
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

let history = [];
let isGenerating = false;
let abortController = null;

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

function usePrompt(text) {
  chatInput.value = text;
  sendMessage();
}

function clearChat() {
  history = [];
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
    str = str + '\\n```';
  }

  // 1. Code blocks
  let text = str.replace(/```([a-zA-Z0-9_-]*)\\n([\\s\\S]*?)```/g, function(match, lang, code) {
    const l = lang ? lang.trim() : 'text';
    const escaped = escapeHtml(code.replace(/\\n$/, ''));
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
  text = text.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
             .replace(/\\*(.*?)\\*/g, '<em>$1</em>');

  // 5. Blockquotes
  text = text.replace(/^\\> (.*$)/gim, '<blockquote>$1</blockquote>');

  // 6. Lists
  text = text.replace(/^\\s*[-*+] (.*$)/gim, '<li>$1</li>');
  text = text.replace(/(<li>.*<\\/li>)/s, '<ul>$1</ul>');

  // 7. Paragraphs
  const paragraphs = text.split(/\\n\\n+/);
  return paragraphs.map(p => {
    p = p.trim();
    if (!p) return '';
    if (p.startsWith('<div') || p.startsWith('<h') || p.startsWith('<ul') || p.startsWith('<blockquote') || p.startsWith('<details')) {
      return p;
    }
    return '<p>' + p.replace(/\\n/g, '<br>') + '</p>';
  }).join('');
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

function appendUserMessage(content) {
  if (welcomeHero && welcomeHero.parentNode) {
    welcomeHero.parentNode.removeChild(welcomeHero);
  }
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = '<div class="message-bubble">' + escapeHtml(content) + '</div>';
  messagesList.appendChild(row);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function createAssistantMessage() {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.innerHTML = `
    <div class="assistant-avatar">
      ${LOGO}
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
  if (!text || isGenerating) return;

  chatInput.value = '';
  chatInput.style.height = 'auto';

  appendUserMessage(text);
  history.push({ role: 'user', content: text });

  const isStream = optStream ? optStream.checked : true;
  const isThinking = optThinking ? optThinking.checked : false;
  const model = (modelSelect && modelSelect.value) ? modelSelect.value : 'auto';
  const privacy = (optPrivacy && optPrivacy.value) ? optPrivacy.value : 'default';

  const cleanMessages = history.filter(m => m && m.content && String(m.content).trim().length > 0);

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
        const lines = buffer.split('\\n');
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
      const prov = routerMeta.provider || '—';
      const mName = routerMeta.model || '—';
      const tier = routerMeta.complexity_tier || '—';
      const lat = routerMeta.latency_ms ? routerMeta.latency_ms + 'ms' : '';
      const cached = routerMeta.cache === 'hit' ? 'cached' : 'fresh';
      assistantMsg.metaEl.innerHTML = `
        <div class="router-pill">
          <span>⚡ <b>Router:</b> ${escapeHtml(prov)} · ${escapeHtml(mName)} · Tier ${escapeHtml(tier)} ${lat ? '· ' + lat : ''} · ${cached}</span>
        </div>
      `;
    }
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
