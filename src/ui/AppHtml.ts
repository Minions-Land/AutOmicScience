export const APP_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutOmicScience Console</title>
<style>
:root {
  color-scheme: light;
  --bg:#f7f7f4;
  --ink:#202426;
  --muted:#687076;
  --line:#d8d8d0;
  --panel:#ffffff;
  --soft:#fbfbfa;
  --accent:#0f766e;
  --accent2:#8b5cf6;
  --bad:#b42318;
}
* { box-sizing:border-box; }
body { margin:0; font-family:ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
.shell { display:grid; grid-template-columns:260px 1fr; min-height:100vh; }
aside { border-right:1px solid var(--line); padding:20px 16px; background:#eeeeea; }
main { padding:22px; }
h1 { font-size:20px; margin:0 0 16px; }
h2 { font-size:15px; margin:0 0 10px; }
.brand { display:flex; align-items:center; gap:10px; margin:0 0 18px; min-height:44px; }
.brand-title { font-size:20px; font-weight:760; letter-spacing:0; }
nav button { width:100%; display:block; text-align:left; border:0; background:transparent; padding:10px 12px; border-radius:6px; color:var(--ink); cursor:pointer; }
nav button.active { background:var(--panel); box-shadow:inset 3px 0 0 var(--accent); }
.grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; min-width:0; }
.wide { grid-column:1 / -1; }
.kv { color:var(--muted); font-size:13px; }
.value { font-weight:650; color:var(--ink); }
textarea, input, select { width:100%; border:1px solid var(--line); border-radius:6px; padding:10px; font:inherit; background:white; color:var(--ink); }
textarea { min-height:110px; resize:vertical; }
.chat-log { display:grid; gap:12px; }
.msg { border:1px solid var(--line); border-radius:8px; padding:12px; background:var(--soft); }
.msg.user { background:#f0fdfa; border-color:#99f6e4; }
.msg.assistant { background:#ffffff; }
.msg.error { background:#fff1f2; border-color:#fecdd3; color:var(--bad); }
.msg-head { display:flex; align-items:center; gap:8px; font-size:12px; color:var(--muted); margin-bottom:8px; font-weight:650; }
.msg .body { white-space:pre-wrap; line-height:1.5; }
.thinking { display:flex; align-items:center; gap:8px; color:var(--muted); }
.thinking-bars { display:flex; gap:3px; height:18px; align-items:center; }
.thinking-bars span { width:4px; height:7px; border-radius:999px; background:linear-gradient(180deg, #0f766e, #8b5cf6); animation:thinkPulse 1s ease-in-out infinite; }
.thinking-bars span:nth-child(2) { animation-delay:.12s; }
.thinking-bars span:nth-child(3) { animation-delay:.24s; }
.thinking-bars span:nth-child(4) { animation-delay:.36s; }
.tool-timeline { display:grid; gap:8px; margin-top:8px; }
.tool-event { border:1px solid var(--line); border-radius:6px; padding:8px; background:#f8fafc; font-size:12px; }
.tool-event summary { cursor:pointer; font-weight:650; color:#0f766e; }
.raw-events { margin-top:10px; }
.raw-events summary { cursor:pointer; color:var(--muted); font-size:12px; }
.model-grid { display:grid; grid-template-columns:1fr 1fr auto; gap:8px; align-items:end; }
.btn { border:0; border-radius:6px; padding:10px 12px; background:var(--accent); color:white; cursor:pointer; font-weight:650; }
.btn:disabled { opacity:.55; cursor:not-allowed; }
.btn.secondary { background:#475569; }
pre { white-space:pre-wrap; word-break:break-word; margin:0; font-size:12px; line-height:1.45; color:#111827; }
.list { display:grid; gap:8px; }
.item { border:1px solid var(--line); border-radius:6px; padding:10px; background:var(--soft); }
.badge { display:inline-block; padding:2px 7px; border-radius:999px; background:#e4f4f2; color:#0f766e; font-size:12px; margin-right:6px; }
.hidden { display:none; }
@media (max-width:900px) { .shell { grid-template-columns:1fr; } aside { border-right:0; border-bottom:1px solid var(--line); } .grid { grid-template-columns:1fr; } }
@media (max-width:700px) { .model-grid { grid-template-columns:1fr; } }
@keyframes thinkPulse {
  0%, 100% { height:6px; opacity:.45; }
  50% { height:18px; opacity:1; }
}
</style>
</head>
<body>
<div class="shell">
<aside>
  <div class="brand">
    <div class="brand-title">AutOmicScience</div>
  </div>
  <nav id="nav">
    <button data-view="overview" class="active">Overview</button>
    <button data-view="chat">Agent Run</button>
    <button data-view="permissions">Permissions</button>
    <button data-view="tasks">Tasks</button>
    <button data-view="plugins">Plugins</button>
    <button data-view="sessions">Sessions</button>
    <button data-view="project">Project Rules</button>
    <button data-view="hooks">Hooks</button>
  </nav>
</aside>
<main>
  <section id="overview" class="view grid"></section>
  <section id="chat" class="view hidden grid">
    <div class="panel wide">
      <h2>Model</h2>
      <div class="model-grid">
        <label>
          <div class="kv">Candidate model</div>
          <select id="modelSelect"></select>
        </label>
        <label>
          <div class="kv">Custom model or chain</div>
          <input id="modelInput" placeholder="gpt-5.5 or gpt-5.5 -> gpt-5.4">
        </label>
        <button class="btn secondary" onclick="setModel()">Switch Model</button>
      </div>
      <div style="height:8px"></div>
      <div class="kv" id="modelStatus"></div>
    </div>
    <div class="panel wide">
      <h2>Agent Run</h2>
      <textarea id="chatInput" placeholder="Ask AutOmicScience to plan, inspect files, run tools, or execute a bioinformatics workflow."></textarea>
      <div style="height:8px"></div>
      <button id="sendButton" class="btn" onclick="sendChat()">Run</button>
    </div>
    <div class="panel wide"><h2>Conversation</h2><div id="chatOutput" class="chat-log"></div></div>
  </section>
  <section id="permissions" class="view hidden grid">
    <div class="panel">
      <h2>Permission Mode</h2>
      <select id="permissionMode"><option>default</option><option>plan</option><option>auto</option><option>bypassPermissions</option></select>
      <div style="height:8px"></div>
      <button class="btn" onclick="setPermissionMode()">Update Mode</button>
    </div>
    <div class="panel">
      <h2>Add Rule</h2>
      <input id="permissionRule" placeholder="Example: deny shell:rm or ask destructive">
      <div style="height:8px"></div>
      <button class="btn secondary" onclick="addPermissionRule()">Add Rule</button>
    </div>
    <div class="panel wide"><h2>Current Rules</h2><div id="permissionRules" class="list"></div></div>
  </section>
  <section id="tasks" class="view hidden grid"><div class="panel wide"><h2>Background Tasks</h2><div id="taskList" class="list"></div></div></section>
  <section id="plugins" class="view hidden grid">
    <div class="panel">
      <h2>Load Plugin</h2>
      <input id="pluginTarget" placeholder="Plugin name or path">
      <div style="height:8px"></div>
      <button class="btn" onclick="loadPlugin()">Load</button>
    </div>
    <div class="panel wide"><h2>Loaded Plugins</h2><div id="pluginList" class="list"></div></div>
  </section>
  <section id="sessions" class="view hidden grid">
    <div class="panel">
      <h2>Save Current Session</h2>
      <input id="sessionName" placeholder="Session name">
      <div style="height:8px"></div>
      <button class="btn" onclick="saveSession()">Save</button>
    </div>
    <div class="panel wide"><h2>Sessions</h2><div id="sessionList" class="list"></div></div>
  </section>
  <section id="project" class="view hidden grid"><div class="panel wide"><h2>Project Instructions</h2><div id="projectFiles" class="list"></div></div></section>
  <section id="hooks" class="view hidden grid"><div class="panel wide"><h2>Hook Events</h2><div id="hookList" class="list"></div></div></section>
</main>
</div>
<script>
let state = null;
let chatTranscript = [];
let chatBusy = false;
const nav = document.getElementById('nav');
nav.addEventListener('click', (event) => {
  const btn = event.target.closest('button[data-view]');
  if (!btn) return;
  document.querySelectorAll('nav button').forEach((b) => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.view').forEach((v) => v.classList.add('hidden'));
  document.getElementById(btn.dataset.view).classList.remove('hidden');
});
async function api(path, options) {
  const res = await fetch(path, options);
  const json = await res.json();
  if (!res.ok) throw new Error(json.error || res.statusText);
  return json;
}
async function refresh() {
  state = await api('/api/state');
  renderOverview();
  renderPermissions();
  renderTasks();
  renderPlugins();
  renderSessions();
  renderHooks();
  document.getElementById('permissionMode').value = state.permissions.mode;
  renderModels();
}
function renderModels() {
  const select = document.getElementById('modelSelect');
  const models = state.models || state.agent.models || [];
  const current = state.agent.models.join(' -> ');
  select.innerHTML = models.map((model) => '<option value="'+escapeHtml(model)+'">'+escapeHtml(model)+'</option>').join('');
  select.value = state.agent.models[0];
  document.getElementById('modelInput').value = current;
  document.getElementById('modelStatus').textContent = 'Current model: ' + current;
}
function renderOverview() {
  const a = state.agent;
  document.getElementById('overview').innerHTML = [
    card('Agent', '<div class="kv">Name</div><div class="value">'+escapeHtml(a.name)+'</div><div class="kv">Model</div><div>'+escapeHtml(a.models.join(' -> '))+'</div>'),
    card('Context', '<div class="kv">Messages</div><div class="value">'+a.messageCount+'</div><div class="kv">Skills</div><div>'+a.skillCount+'</div>'),
    card('Tools', '<div class="kv">Count</div><div class="value">'+a.toolCount+'</div><div class="kv">Permission mode</div><div>'+escapeHtml(state.permissions.mode)+'</div>'),
    card('Skills', list(a.skills.map((s) => '<span class="badge">'+escapeHtml(s.name)+'</span>'+escapeHtml(s.description)))),
    card('Tool Catalog', list(a.tools.slice(0, 24).map((t) => '<span class="badge">'+escapeHtml(t.name)+'</span>'+escapeHtml(t.description)))),
    card('Commands', list(state.commands.map((c) => '<span class="badge">'+escapeHtml(c.name)+'</span>'+escapeHtml(c.description))), 'wide')
  ].join('');
}
function renderPermissions() {
  document.getElementById('permissionRules').innerHTML = list(state.permissions.rules.map((r) => '<pre>'+escapeHtml(JSON.stringify(r, null, 2))+'</pre>'));
}
function renderTasks() {
  document.getElementById('taskList').innerHTML = list(state.tasks.map((t) => '<span class="badge">'+escapeHtml(t.state)+'</span>'+escapeHtml(t.id+' '+t.name)));
}
function renderPlugins() {
  document.getElementById('pluginList').innerHTML = list(state.plugins.map((p) => escapeHtml(p)));
}
function renderSessions() {
  document.getElementById('sessionList').innerHTML = list(state.sessions.map((s) => escapeHtml(s)));
}
function renderHooks() {
  document.getElementById('hookList').innerHTML = list(state.hooks.map((h) => '<span class="badge">'+escapeHtml(h.event)+'</span><pre>'+escapeHtml(JSON.stringify(h.data, null, 2))+'</pre>'));
}
async function renderProject() {
  const data = await api('/api/project-instructions');
  document.getElementById('projectFiles').innerHTML = list(data.files.map((f) => '<span class="badge">'+escapeHtml(f.path)+'</span><pre>'+escapeHtml(f.content)+'</pre>'));
}
async function sendChat() {
  if (chatBusy) return;
  const input = document.getElementById('chatInput').value;
  if (!input.trim()) return;
  chatBusy = true;
  document.getElementById('sendButton').disabled = true;
  const output = document.getElementById('chatOutput');
  chatTranscript.push({ role:'user', content:input });
  chatTranscript.push({ role:'assistant', pending:true, content:'' });
  output.innerHTML = renderTranscript();
  try {
    const data = await api('/api/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ input }) });
    chatTranscript.pop();
    chatTranscript.push(buildAssistantTurn(data.events || []));
    output.innerHTML = renderTranscript();
    document.getElementById('chatInput').value = '';
    await refresh();
  } catch (err) {
    chatTranscript.pop();
    chatTranscript.push({ role:'error', content: err && err.message ? err.message : String(err) });
    output.innerHTML = renderTranscript();
    await refresh().catch(() => {});
  } finally {
    chatBusy = false;
    document.getElementById('sendButton').disabled = false;
  }
}
function buildAssistantTurn(events) {
  const text = events.filter((event) => event.type === 'text').map((event) => event.data || '').join('');
  const done = [...events].reverse().find((event) => event.type === 'done');
  const finalText = (done && done.data) ? done.data : text || 'Completed, but the model returned no final text.';
  const tools = [];
  for (const event of events) {
    if (event.type === 'tool_call') {
      for (const call of event.data || []) tools.push({ kind:'call', name:call.name, data:call.arguments || {} });
    } else if (event.type === 'tool_result') {
      const payload = parseToolPayload(event.data && event.data.content);
      tools.push({ kind:'result', name:event.data && event.data.tool_call_id ? event.data.tool_call_id : 'tool_result', data:payload });
    }
  }
  return { role:'assistant', content:finalText, tools, events };
}
function renderTranscript() {
  return chatTranscript.map((turn) => {
    if (turn.role === 'user') return '<div class="msg user"><div class="msg-head">You</div><div class="body">'+escapeHtml(turn.content)+'</div></div>';
    if (turn.role === 'error') return '<div class="msg error"><div class="msg-head">Error</div><div class="body">'+escapeHtml(turn.content)+'</div></div>';
    return renderAssistantTurn(turn);
  }).join('');
}
function renderAssistantTurn(turn) {
  const head = '<div class="msg-head"><span>AutOmicScience</span></div>';
  const body = turn.pending
    ? '<div class="thinking"><div class="thinking-bars"><span></span><span></span><span></span><span></span></div><span>Thinking and checking tools...</span></div>'
    : '<div class="body">'+escapeHtml(turn.content)+'</div>';
  return [
    '<div class="msg assistant">'+head+body+'</div>',
    turn.tools && turn.tools.length ? '<div class="tool-timeline">'+turn.tools.map(renderToolEvent).join('')+'</div>' : '',
    turn.events ? '<details class="raw-events"><summary>Debug: raw AgentEvents JSON</summary><pre>'+escapeHtml(JSON.stringify(turn.events, null, 2))+'</pre></details>' : ''
  ].join('');
}
function renderToolEvent(event) {
  const title = event.kind === 'call' ? 'Tool call: ' + event.name : 'Tool result: ' + event.name;
  return '<details class="tool-event"><summary>'+escapeHtml(title)+'</summary><pre>'+escapeHtml(JSON.stringify(event.data, null, 2))+'</pre></details>';
}
function parseToolPayload(content) {
  if (typeof content !== 'string') return content || {};
  try { return JSON.parse(content); } catch { return content; }
}
async function setModel() {
  const typed = document.getElementById('modelInput').value.trim();
  const selected = document.getElementById('modelSelect').value;
  const model = typed || selected;
  const data = await api('/api/model', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model }) });
  document.getElementById('modelStatus').textContent = 'Current model: ' + data.models.join(' -> ');
  await refresh();
}
async function setPermissionMode() {
  await api('/api/permissions/mode', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ mode: document.getElementById('permissionMode').value }) });
  await refresh();
}
async function addPermissionRule() {
  await api('/api/permissions/rules', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ rule: document.getElementById('permissionRule').value }) });
  document.getElementById('permissionRule').value = '';
  await refresh();
}
async function loadPlugin() {
  await api('/api/plugins/load', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ target: document.getElementById('pluginTarget').value }) });
  await refresh();
}
async function saveSession() {
  await api('/api/session/save', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name: document.getElementById('sessionName').value }) });
  await refresh();
}
function card(title, html, extra) { return '<div class="panel '+(extra || '')+'"><h2>'+escapeHtml(title)+'</h2>'+html+'</div>'; }
function list(items) { return items.length ? '<div class="list">'+items.map((x) => '<div class="item">'+x+'</div>').join('')+'</div>' : '<div class="kv">No data</div>'; }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
setInterval(refresh, 6000);
refresh().then(renderProject);
</script>
</body>
</html>`;
