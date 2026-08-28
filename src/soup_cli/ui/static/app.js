/* Soup Web UI — Frontend Application */

const API = '';  // same origin

// v0.53.9 #94 — Lightweight EventSource consumer for /api/train/stream.
// Opens on demand (call `startTrainEventStream()`) and dispatches parsed
// payloads to `onTrainEvent(payload)` which other modules can override.
// Auto-closes on `status=done` or `status=timeout`.
let _trainEventSource = null;
window.onTrainEvent = window.onTrainEvent || function (_payload) {};
function startTrainEventStream() {
  if (_trainEventSource) return _trainEventSource;
  try {
    const es = new EventSource('/api/train/stream');
    _trainEventSource = es;
    es.onmessage = function (event) {
      if (!event.data) return;
      try {
        const payload = JSON.parse(event.data);
        if (payload && typeof window.onTrainEvent === 'function') {
          window.onTrainEvent(payload);
        }
        if (payload && payload.type === 'status' &&
            (payload.message === 'done' || payload.message === 'timeout')) {
          es.close();
          _trainEventSource = null;
        }
      } catch (e) {
        // Ignore malformed frames.
      }
    };
    es.onerror = function () {
      try { es.close(); } catch (e) {}
      _trainEventSource = null;
    };
    return es;
  } catch (e) {
    return null;
  }
}
function stopTrainEventStream() {
  if (_trainEventSource) {
    try { _trainEventSource.close(); } catch (e) {}
    _trainEventSource = null;
  }
}
window.startTrainEventStream = startTrainEventStream;
window.stopTrainEventStream = stopTrainEventStream;

// v0.53.9 #95 — Pick up Bearer token from `?token=…` (phone QR landing)
// or from sessionStorage on subsequent navigations. Stripped from the URL
// after read so the token doesn't sit in browser history.
(function _bootstrapAuthToken() {
  try {
    const params = new URLSearchParams(window.location.search);
    const fromUrl = params.get('token');
    if (fromUrl) {
      window._authToken = fromUrl;
      try { sessionStorage.setItem('soup_auth_token', fromUrl); } catch (e) {}
      // Drop ?token=… from the URL so refresh history doesn't leak it.
      params.delete('token');
      const qs = params.toString();
      const clean = window.location.pathname + (qs ? '?' + qs : '') +
        window.location.hash;
      window.history.replaceState(null, '', clean);
    } else {
      try {
        const saved = sessionStorage.getItem('soup_auth_token');
        if (saved) window._authToken = saved;
      } catch (e) {}
    }
  } catch (e) {
    // Defensive: never block app load.
  }
  // Proactive: don't wait for the first 401 to tell the user something is
  // wrong — if we land with no token from either source, every button is
  // about to fail, so say so immediately instead of letting them click
  // around and hit "Unauthorized" one card at a time.
  if (!window._authToken) {
    document.addEventListener('DOMContentLoaded', () => showAuthBanner());
  }
})();

function showAuthBanner() {
  const banner = document.getElementById('auth-banner');
  if (banner) banner.style.display = 'flex';
}

function hideAuthBanner() {
  const banner = document.getElementById('auth-banner');
  if (banner) banner.style.display = 'none';
}

function submitAuthToken() {
  const input = document.getElementById('auth-banner-input');
  const token = (input.value || '').trim();
  if (!token) return;
  window._authToken = token;
  try { sessionStorage.setItem('soup_auth_token', token); } catch (e) {}
  input.value = '';
  hideAuthBanner();
  pushToast('Token saved for this session.', 'success');
  // Best-effort refresh of whatever page is showing, now that requests
  // should actually authenticate.
  try { navigate(currentPage); } catch (e) {}
}

// --- State ---
let currentPage = 'dashboard';
let runsData = [];
let systemInfo = null;
let chatMessages = [];
let chatEndpoint = null;

// --- Navigation ---
function navigate(page) {
  currentPage = page;
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  document.querySelector(`[data-page="${page}"]`).classList.add('active');

  if (page === 'dashboard') loadDashboard();
  else if (page === 'training') loadTrainingPage();
  else if (page === 'data') { /* loaded on demand */ }
  else if (page === 'chat') loadChatPage();
  else if (page === 'tools') loadToolOutputs();
  else if (page === 'modelhub') loadModelHubPage();
  else if (page === 'help') loadHelpPage();
  // v0.53.10 #155 — pause Tool Outputs polling when navigating away so we
  // don't keep firing fetch() against /api/tool-outputs from background tabs.
  if (page !== 'tools') stopToolOutputsPolling();
  if (page !== 'modelhub') stopHubDownloadsPolling();
  if (page !== 'training') stopCompressJobsPolling();
  if (page !== 'training') stopTrainingProgressPolling();
  if (page !== 'dashboard') stopDashboardLivePolling();
}

// --- Tool Outputs panel (v0.53.10 #155) ---
// Polls /api/tool-outputs every 3 s while the page is active. XSS-safe via
// textContent / .appendChild (no innerHTML for user-controlled fields).
let _toolsPollHandle = null;

function loadToolOutputs() {
  renderToolOutputs();
  if (_toolsPollHandle === null) {
    _toolsPollHandle = setInterval(renderToolOutputs, 3000);
  }
}

function stopToolOutputsPolling() {
  if (_toolsPollHandle !== null) {
    clearInterval(_toolsPollHandle);
    _toolsPollHandle = null;
  }
}

async function renderToolOutputs() {
  const container = document.getElementById('tools-content');
  if (!container) return;
  let payload;
  try {
    const headers = {};
    if (window._authToken) {
      headers['Authorization'] = 'Bearer ' + window._authToken;
    }
    const resp = await fetch('/api/tool-outputs?limit=100', { headers });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    payload = await resp.json();
  } catch (err) {
    container.textContent = 'Failed to load tool outputs: ' + err.message;
    return;
  }
  const records = (payload && Array.isArray(payload.records)) ? payload.records : [];
  // Build the table via DOM APIs so user-controlled fields stay XSS-safe.
  container.replaceChildren();
  if (records.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    const t = document.createElement('div');
    t.className = 'empty-state-text';
    t.textContent = 'No tool calls observed yet.';
    empty.appendChild(t);
    container.appendChild(empty);
    return;
  }
  const wrap = document.createElement('div');
  wrap.className = 'table-wrap';
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const head = document.createElement('tr');
  ['Name', 'Started', 'Duration (ms)', 'OK', 'Output'].forEach(label => {
    const th = document.createElement('th');
    th.textContent = label;
    head.appendChild(th);
  });
  thead.appendChild(head);
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  records.forEach(rec => {
    const tr = document.createElement('tr');
    const cells = [
      String(rec.name || ''),
      rec.started_ts ? new Date(rec.started_ts * 1000).toLocaleTimeString() : '-',
      (typeof rec.duration_ms === 'number') ? rec.duration_ms.toFixed(1) : '-',
      rec.success ? '✓' : '✗',
      String(rec.output_preview || rec.error || ''),
    ];
    cells.forEach(value => {
      const td = document.createElement('td');
      td.textContent = value;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
}

// --- API Helpers ---
// Bug fix: every mutating endpoint (train/start, train/stop, config/validate,
// data/inspect, config/from-form, config/patch-training, hf/download) is
// registered server-side with `Depends(_verify_token)` and 401s without a
// Bearer header. This helper is the single place nearly every page routes
// requests through, but it never attached the token that `_bootstrapAuthToken`
// (top of this file) already captures — so Start Training / Stop Training /
// Validate Config / Data Inspect all 401'd silently from the browser even
// after a valid token was on `window._authToken`. Fixed once, here, instead
// of patching each of the ~8 call sites separately.
async function api(path, opts = {}) {
  const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
  if (window._authToken) {
    headers['Authorization'] = 'Bearer ' + window._authToken;
  }
  const resp = await fetch(API + path, { ...opts, headers });
  if (resp.status === 401) {
    // Reactive path: covers a token going stale mid-session (e.g. the
    // `soup ui` process restarted and rotated the token) — the proactive
    // check in `_bootstrapAuthToken` only catches "no token at all" at
    // load time. Clear the stale one so `showAuthBanner` isn't fighting a
    // token that will just 401 again.
    window._authToken = null;
    try { sessionStorage.removeItem('soup_auth_token'); } catch (e) {}
    showAuthBanner();
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || 'Unauthorized — session token needed (see banner above).');
  }
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || 'API error');
  }
  return resp.json();
}

function formatDuration(secs) {
  if (!secs) return '-';
  if (secs < 60) return `${Math.round(secs)}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

function formatDate(iso) {
  if (!iso) return '-';
  const d = new Date(iso);
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function statusBadge(status) {
  const map = {
    completed: 'badge-success',
    failed: 'badge-danger',
    running: 'badge-warning',
  };
  return `<span class="badge ${map[status] || 'badge-info'}">${escapeHtml(status)}</span>`;
}

function truncate(str, len = 30) {
  if (!str) return '-';
  return str.length > len ? str.substring(0, len) + '...' : str;
}

// --- Dashboard ---
let _dashboardLivePollHandle = null;

async function loadDashboard() {
  try {
    const [runsResp, sysResp] = await Promise.all([
      api('/api/runs?limit=100'),
      api('/api/system'),
    ]);
    runsData = runsResp.runs;
    systemInfo = sysResp;
    renderDashboard();
  } catch (err) {
    document.getElementById('dashboard-content').innerHTML =
      `<div class="empty-state"><div class="empty-state-text">Error loading dashboard: ${escapeHtml(err.message)}</div></div>`;
  }
  renderHealthBanner();
  if (_dashboardLivePollHandle === null) {
    renderLiveResources();
    _dashboardLivePollHandle = setInterval(renderLiveResources, 4000);
  }
}

function stopDashboardLivePolling() {
  if (_dashboardLivePollHandle !== null) {
    clearInterval(_dashboardLivePollHandle);
    _dashboardLivePollHandle = null;
  }
}

async function renderHealthBanner() {
  const el = document.getElementById('health-banner');
  if (!el) return;
  try {
    const health = await api('/api/system/health');
    if (health.ok) {
      el.innerHTML = '';
      return;
    }
    el.innerHTML = `<div class="card" style="border-left:3px solid var(--danger);margin-bottom:1rem">` +
      `<div class="card-title" style="margin-bottom:0.35rem">⚠ Check before you train</div>` +
      health.issues.map(i => `<div style="font-size:0.85rem;color:var(--text-dim)">${escapeHtml(i)}</div>`).join('') +
      `<div style="font-size:0.8rem;margin-top:0.4rem"><code>soup doctor</code> for the full diagnostic.</div></div>`;
  } catch (err) {
    el.innerHTML = '';
  }
}

async function renderLiveResources() {
  const dashboardContent = document.getElementById('dashboard-content');
  if (!dashboardContent) return;
  let live;
  try {
    live = await api('/api/system/live');
  } catch (err) {
    return;
  }
  let el = document.getElementById('live-resources');
  if (!el) {
    el = document.createElement('div');
    el.id = 'live-resources';
    el.className = 'card';
    dashboardContent.prepend(el);
  }
  const bar = (label, pct) => `
    <div style="margin-bottom:0.5rem">
      <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:var(--text-dim)">
        <span>${escapeHtml(label)}</span><span>${pct == null ? '-' : pct.toFixed(0) + '%'}</span>
      </div>
      <div class="progress-bar"><div class="progress-bar-fill" style="width:${pct || 0}%"></div></div>
    </div>`;
  let html = '<div class="card-title">Live resources</div>';
  html += bar('CPU', live.cpu_pct);
  html += bar('RAM', live.ram_pct);
  (live.gpu || []).forEach(g => {
    html += bar(`GPU ${g.index} (${escapeHtml(g.name)}) — ${g.memory_used_gb}/${g.memory_total_gb} GB`, g.memory_pct);
  });
  if (!live.gpu || live.gpu.length === 0) {
    html += '<div class="empty-state-hint">No CUDA GPU detected.</div>';
  }
  el.innerHTML = html;
}

function renderDashboard() {
  const completed = runsData.filter(r => r.status === 'completed');
  const failed = runsData.filter(r => r.status === 'failed');
  const running = runsData.filter(r => r.status === 'running');
  const bestLoss = completed.length
    ? Math.min(...completed.map(r => r.final_loss).filter(Boolean)).toFixed(4)
    : '-';

  document.getElementById('dashboard-content').innerHTML = `
    <div class="stats-row">
      <div class="card stat-card">
        <div class="stat-value">${runsData.length}</div>
        <div class="stat-label">Total Runs</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">${completed.length}</div>
        <div class="stat-label">Completed</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">${running.length}</div>
        <div class="stat-label">Running</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">${bestLoss}</div>
        <div class="stat-label">Best Loss</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">System</div>
      <div style="font-size:0.9rem; color: var(--text-dim)">
        Device: <strong style="color:var(--text)">${escapeHtml(systemInfo.device_name)}</strong> &nbsp;|&nbsp;
        GPU Memory: <strong style="color:var(--text)">${escapeHtml(systemInfo.gpu_info.memory_total)}</strong> &nbsp;|&nbsp;
        Python: <strong style="color:var(--text)">${escapeHtml(systemInfo.python_version)}</strong> &nbsp;|&nbsp;
        Soup: <strong style="color:var(--text)">v${escapeHtml(systemInfo.version)}</strong>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Recent Runs</div>
      ${runsData.length === 0
        ? '<div class="empty-state"><div class="empty-state-text">No runs yet</div><div class="empty-state-hint">Start training with "soup train" or use the New Training page</div></div>'
        : renderRunsTable(runsData.slice(0, 20))
      }
    </div>
  `;
}

function renderRunsTable(runs) {
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Name</th>
            <th>Model</th>
            <th>Task</th>
            <th>Status</th>
            <th>Loss</th>
            <th>Duration</th>
            <th>Date</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${runs.map(r => `
            <tr style="cursor:pointer" onclick="showRunDetail('${escapeHtml(r.run_id)}')">
              <td><code style="font-size:0.8rem">${escapeHtml(r.run_id.substring(0, 20))}...</code></td>
              <td>${escapeHtml(r.experiment_name || '-')}</td>
              <td>${escapeHtml(truncate(r.base_model))}</td>
              <td>${escapeHtml(r.task || 'sft')}</td>
              <td>${statusBadge(r.status)}</td>
              <td>${r.final_loss ? r.final_loss.toFixed(4) : '-'}</td>
              <td>${formatDuration(r.duration_secs)}</td>
              <td>${formatDate(r.created_at)}</td>
              <td>
                <button class="btn btn-danger btn-sm" onclick="event.stopPropagation(); deleteRun('${escapeHtml(r.run_id)}')">Delete</button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

async function deleteRun(runId) {
  if (!confirm('Delete this run and all its metrics?')) return;
  try {
    await api(`/api/runs/${runId}`, { method: 'DELETE' });
    loadDashboard();
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

// --- Run Detail Modal ---
let lossChart = null;

async function showRunDetail(runId) {
  const modal = document.getElementById('run-modal');
  const body = document.getElementById('run-modal-body');
  modal.classList.add('active');

  body.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--text-dim)">Loading...</div>';

  try {
    const [run, metricsResp, evalResp] = await Promise.all([
      api(`/api/runs/${runId}`),
      api(`/api/runs/${runId}/metrics`),
      api(`/api/runs/${runId}/eval`),
    ]);

    const config = run.config_json ? JSON.parse(run.config_json) : {};
    const metrics = metricsResp.metrics;

    body.innerHTML = `
      <div class="grid-2" style="margin-bottom:1rem">
        <div>
          <div class="form-label">Run ID</div>
          <div><code>${escapeHtml(run.run_id)}</code></div>
        </div>
        <div>
          <div class="form-label">Status</div>
          <div>${statusBadge(run.status)}</div>
        </div>
        <div>
          <div class="form-label">Model</div>
          <div>${escapeHtml(run.base_model || '-')}</div>
        </div>
        <div>
          <div class="form-label">Task</div>
          <div>${escapeHtml(run.task || 'sft')}</div>
        </div>
        <div>
          <div class="form-label">Device</div>
          <div>${escapeHtml(run.device_name || run.device || '-')}</div>
        </div>
        <div>
          <div class="form-label">Duration</div>
          <div>${formatDuration(run.duration_secs)}</div>
        </div>
        <div>
          <div class="form-label">Initial Loss</div>
          <div>${run.initial_loss ? run.initial_loss.toFixed(4) : '-'}</div>
        </div>
        <div>
          <div class="form-label">Final Loss</div>
          <div>${run.final_loss ? run.final_loss.toFixed(4) : '-'}</div>
        </div>
      </div>

      ${metrics.length > 0 ? `
        <div class="chart-grid">
          <div class="card">
            <div class="card-title">Loss</div>
            <div class="chart-container"><canvas id="loss-chart"></canvas></div>
          </div>
          <div class="card">
            <div class="card-title">Learning Rate</div>
            <div class="chart-container"><canvas id="lr-chart"></canvas></div>
          </div>
          <div class="card">
            <div class="card-title">Gradient Norm</div>
            <div class="chart-container"><canvas id="gradnorm-chart"></canvas></div>
          </div>
          <div class="card">
            <div class="card-title">Throughput</div>
            <div class="chart-container"><canvas id="speed-chart"></canvas></div>
          </div>
        </div>
        ${metrics.some(m => m.gpu_mem) ? `
          <div class="card">
            <div class="card-title">GPU Memory</div>
            <div class="chart-container"><canvas id="gpumem-chart"></canvas></div>
          </div>
        ` : ''}
      ` : ''}

      ${evalResp.eval_results && evalResp.eval_results.length > 0 ? `
        <div class="card">
          <div class="card-title">Eval Results</div>
          <div class="table-wrap">
            <table class="eval-table">
              <thead><tr><th>Benchmark</th><th>Score</th><th>Details</th></tr></thead>
              <tbody>
                ${evalResp.eval_results.map(er => `
                  <tr>
                    <td>${escapeHtml(er.benchmark)}</td>
                    <td>${typeof er.score === 'number' ? er.score.toFixed(4) : escapeHtml(String(er.score))}</td>
                    <td><code style="font-size:0.75rem">${er.details_json ? escapeHtml(String(er.details_json).substring(0, 100)) : '-'}</code></td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      ` : ''}

      <div class="card">
        <div class="card-title">Config</div>
        <pre style="font-size:0.8rem;color:var(--text-dim);white-space:pre-wrap;max-height:300px;overflow-y:auto">${JSON.stringify(config, null, 2)}</pre>
      </div>
    `;

    if (metrics.length > 0) {
      renderCharts(metrics);
    }
  } catch (err) {
    body.innerHTML = `<div class="empty-state"><div class="empty-state-text">Error: ${escapeHtml(err.message)}</div></div>`;
  }
}

function renderCharts(metrics) {
  const steps = metrics.map(m => m.step);
  const losses = metrics.map(m => m.loss);
  const lrs = metrics.map(m => m.lr);
  const gradNorms = metrics.map(m => m.grad_norm || 0);
  const speeds = metrics.map(m => m.speed || 0);

  const chartOpts = (yLabel) => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { title: { display: true, text: 'Step', color: '#a09088' }, ticks: { color: '#a09088' }, grid: { color: 'rgba(58,48,64,0.5)' } },
      y: { title: { display: true, text: yLabel, color: '#a09088' }, ticks: { color: '#a09088' }, grid: { color: 'rgba(58,48,64,0.5)' } },
    },
  });

  const makeDataset = (label, data, color) => ({
    label, data, borderColor: color,
    backgroundColor: color.replace(')', ', 0.1)').replace('rgb', 'rgba'),
    fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
  });

  // Loss chart
  const lossCtx = document.getElementById('loss-chart');
  if (lossCtx) {
    if (lossChart) lossChart.destroy();
    lossChart = new Chart(lossCtx, {
      type: 'line',
      data: { labels: steps, datasets: [makeDataset('Loss', losses, 'rgb(192, 81, 45)')] },
      options: chartOpts('Loss'),
    });
  }

  // LR chart
  const lrCtx = document.getElementById('lr-chart');
  if (lrCtx) {
    new Chart(lrCtx, {
      type: 'line',
      data: { labels: steps, datasets: [makeDataset('LR', lrs, 'rgb(232, 151, 90)')] },
      options: chartOpts('LR'),
    });
  }

  // Gradient Norm chart
  const gnCtx = document.getElementById('gradnorm-chart');
  if (gnCtx) {
    new Chart(gnCtx, {
      type: 'line',
      data: { labels: steps, datasets: [makeDataset('Grad Norm', gradNorms, 'rgb(100, 180, 220)')] },
      options: chartOpts('Grad Norm'),
    });
  }

  // Speed chart
  const spCtx = document.getElementById('speed-chart');
  if (spCtx) {
    new Chart(spCtx, {
      type: 'line',
      data: { labels: steps, datasets: [makeDataset('Speed', speeds, 'rgb(120, 200, 130)')] },
      options: chartOpts('Tokens/sec'),
    });
  }

  // GPU Memory chart (optional — parse numeric values from strings like "4.2GB")
  const gmCtx = document.getElementById('gpumem-chart');
  if (gmCtx) {
    const gpuVals = metrics.map(m => {
      if (!m.gpu_mem) return 0;
      const match = String(m.gpu_mem).match(/([\d.]+)/);
      return match ? parseFloat(match[1]) : 0;
    });
    new Chart(gmCtx, {
      type: 'line',
      data: { labels: steps, datasets: [makeDataset('GPU Mem', gpuVals, 'rgb(200, 130, 200)')] },
      options: chartOpts('GPU Memory (GB)'),
    });
  }
}

function closeModal() {
  document.getElementById('run-modal').classList.remove('active');
}

// --- New Training Page ---
async function loadTrainingPage() {
  // Bug fix (this session): renderTrainingPage() rebuilds #training-content
  // from scratch every call — including the config editor, reset to blank
  // (or, before this session, silently to an arbitrary template). Since
  // startTraining() calls loadTrainingPage() right after POSTing the
  // config to actually start the run, the editor used to visibly wipe
  // itself immediately after Start Training — confusing (looks like
  // something failed) even though the run had already started correctly
  // with the old content. Capture and restore it across the re-render.
  const existingEditor = document.getElementById('config-editor');
  const preservedYaml = existingEditor ? existingEditor.value : null;
  const existingTemplateSel = document.getElementById('template-select');
  const preservedTemplate = existingTemplateSel ? existingTemplateSel.value : null;

  try {
    const [templatesResp, statusResp, recipesResp] = await Promise.all([
      api('/api/templates'),
      api('/api/train/status'),
      api('/api/recipes'),
    ]);
    window._recipes = recipesResp.recipes || [];
    renderTrainingPage(templatesResp.templates, statusResp);
    if (preservedYaml) {
      document.getElementById('config-editor').value = preservedYaml;
      if (preservedTemplate) document.getElementById('template-select').value = preservedTemplate;
      syncQuicksetFromEditor();
    }
    initStreamingPanel();
    loadCompressPage();       // Compress section lives inside New Training now
    renderQuickReference();
    renderCalculator();
    if (statusResp.running) {
      connectTrainingSSE();
      startTrainingProgressPolling();
    }
  } catch (err) {
    document.getElementById('training-content').innerHTML =
      `<div class="empty-state"><div class="empty-state-text">Error: ${escapeHtml(err.message)}</div></div>`;
  }
}

// --- Calculator: estimated model size after training/quantization/compress ---
// (this session — see index.html's #calculator-section)

const CALC_BYTES_PER_PARAM = {
  fp32: 4.0, fp16: 2.0, int8: 1.0,
  gptq4: 0.5, gptq3: 0.375, gptq2: 0.25, awq4: 0.5,
  q8_0: 1.06, q5_k_m: 0.73, q4_k_m: 0.63, q2_k: 0.35,
};

let _calcModelLookupTimer = null;

function onCalcModelInput() {
  // Debounced — don't fire a lookup on every keystroke.
  clearTimeout(_calcModelLookupTimer);
  _calcModelLookupTimer = setTimeout(async () => {
    const model = document.getElementById('calc-model').value.trim();
    if (!model) return;
    try {
      const result = await api('/api/calculator/model-size?model=' + encodeURIComponent(model));
      document.getElementById('calc-params-b').value = result.params_b;
      renderCalculator();
    } catch (err) {
      // Silent — the params field just stays whatever the user last typed;
      // they can always enter a param count manually.
    }
  }, 500);
}

function renderCalculator() {
  const paramsB = parseFloat(document.getElementById('calc-params-b').value);
  const resultEl = document.getElementById('calculator-result');
  if (!resultEl) return;
  if (!paramsB || paramsB <= 0) {
    resultEl.innerHTML = '<div class="empty-state-hint">Enter a model or a parameter count to calculate.</div>';
    return;
  }
  const precision = document.getElementById('calc-precision').value;
  const bytesPerParam = CALC_BYTES_PER_PARAM[precision] || 2.0;
  const compressPct = parseFloat(document.getElementById('calc-compress-pct').value || '0') / 100;

  const baseGb = paramsB * 2.0; // FP16, the typical post-SFT checkpoint size
  const quantizedGb = paramsB * bytesPerParam;
  const compressedParamsB = paramsB * (1 - compressPct);
  const finalGb = compressedParamsB * bytesPerParam;

  const row = (label, gb, note) =>
    `<tr><td>${escapeHtml(label)}</td><td style="text-align:right">${gb.toFixed(2)} GB</td>` +
    `<td style="color:var(--text-dim);font-size:0.8rem">${escapeHtml(note || '')}</td></tr>`;

  resultEl.innerHTML = `
    <div class="table-wrap">
    <table>
      <thead><tr><th>Stage</th><th style="text-align:right">Est. size</th><th>Note</th></tr></thead>
      <tbody>
        ${row('After training (FP16)', baseGb, `${paramsB}B params × 2 bytes`)}
        ${row('After quantization only', quantizedGb, `${paramsB}B params × ${bytesPerParam} bytes`)}
        ${compressPct > 0
          ? row('After quantization + compress', finalGb, `${compressedParamsB.toFixed(2)}B effective params × ${bytesPerParam} bytes`)
          : ''}
      </tbody>
    </table>
    </div>
    <div class="empty-state-hint" style="margin-top:0.5rem">
      "After training" assumes a typical FP16/BF16 checkpoint straight out of SFT — LoRA
      adapters alone are far smaller (tens to low hundreds of MB) until merged into the base.
    </div>
  `;
}

// --- Quick Reference: moved to the bottom of the page and expanded to
// actually match the current schema (was a small, partial, increasingly
// stale list squeezed next to Training Status). ---

function renderQuickReference() {
  const el = document.getElementById('quick-reference-card');
  if (!el) return;
  el.innerHTML = `
    <div class="card-title">Quick Reference</div>
    <div style="font-size:0.85rem; color:var(--text-dim); line-height:1.9">
      <strong style="color:var(--text)">Tasks:</strong> sft, dpo, grpo, ppo, reward_model, kto,
      orpo, simpo, ipo, bco, preference, pretrain, embedding, prm, tts, classifier, reranker,
      cross_encoder, distill, unlearn, moe_lora_routing, online_dpo, asr<br>
      <strong style="color:var(--text)">Backends:</strong> transformers, unsloth<br>
      <strong style="color:var(--text)">Data formats:</strong> alpaca, sharegpt, chatml, dpo, kto,
      llava, sharegpt4v, plaintext, embedding, audio, tool-calling, prm, pre_tokenized,
      input_output, video, multimodal, raft, asr, auto (auto-detects)<br>
      <strong style="color:var(--text)">Quantization (training):</strong> 4bit, 8bit, none
      (bitsandbytes, for QLoRA-style training)<br>
      <strong style="color:var(--text)">Quantization (export, post-training):</strong>
      AWQ (4-bit only), GPTQ (2/3/4/8-bit), GGUF k-quants, GGUF i-quants/UD — see the
      Quantization card above and <a href="#" onclick="navigate('help');return false;">Help</a>.<br>
      <strong style="color:var(--text)">Training objectives</strong>
      (<code>training.objectives</code>): code, tool_call, reasoning, chat, general — freely
      combinable under sft/distill; orpo — alone only, under task: orpo.<br>
      <strong style="color:var(--text)">Compression pipeline</strong>
      (<code>training.pipeline</code>): activation_scan → compress → distill, run via
      <code>soup pipeline run</code>. See the Compress section above and
      <a href="#" onclick="navigate('help');return false;">Help</a> for details.<br>
      <strong style="color:var(--text)">Multi-dataset:</strong> data.train / data.val /
      data.calibration each accept a single path or a list.
    </div>
  `;
}


// Sets the RAM-cache slider's max from real available host RAM, and
// applies the slider/dropdown onto the `training:` block of the currently
// loaded config via /api/config/patch-ram-prefetch and /api/config/patch-quant
// (server-side, re-validated against the real schema — see soup_cli/ui/app.py).
async function initStreamingPanel() {
  try {
    const ram = await api('/api/system/ram');
    const slider = document.getElementById('ram-cache-slider');
    const availEl = document.getElementById('ram-cache-avail');
    if (ram.safe_max_gb) {
      slider.max = ram.safe_max_gb;
      availEl.textContent = `${ram.available_gb} GB available (safe max ${ram.safe_max_gb} GB)`;
    } else {
      availEl.textContent = 'psutil not installed — RAM detection unavailable';
    }
  } catch (err) {
    // Non-fatal: keep the default 0-16 GB slider range.
  }

  try {
    window._quantFormats = await api('/api/quant/formats');
  } catch (err) {
    window._quantFormats = null;
  }

  // Bug fix (this session): the RAM Prefetch slider and Quantization
  // dropdown always reset to their defaults (0 GB / none) on load,
  // regardless of what the current config editor's YAML actually already
  // says — "Apply to config" was one-way (UI -> YAML), so loading a
  // template/recipe that already set e.g. custom_quant_strategy: gptq
  // showed "None" in the dropdown, an inconsistency between what's
  // displayed and what's actually configured that could send someone
  // looking for a bug that wasn't there. Now reads the editor first.
  syncStreamingPanelFromEditor();
}

function syncStreamingPanelFromEditor() {
  const editor = document.getElementById('config-editor');
  const yamlStr = editor ? editor.value : '';

  const ramMatch = yamlStr.match(/^\s+ram_cache_gb:\s*([\d.]+)/m);
  const slider = document.getElementById('ram-cache-slider');
  if (slider) slider.value = ramMatch ? ramMatch[1] : '0';
  onRamCacheSliderInput();

  const strategyMatch = yamlStr.match(/^\s+custom_quant_strategy:\s*["']?([\w-]+)["']?/m);
  const detailMatch = yamlStr.match(/^\s+custom_quant_detail:\s*["']?([\w.-]+)["']?/m);
  const strategySel = document.getElementById('quant-strategy-select');
  if (strategySel) strategySel.value = strategyMatch ? strategyMatch[1] : 'none';
  onQuantStrategyChange();
  if (detailMatch) {
    const detailSel = document.getElementById('quant-detail-select');
    if (detailSel) detailSel.value = detailMatch[1];
  }
}

function onQuantStrategyChange() {
  const strategy = document.getElementById('quant-strategy-select').value;
  const group = document.getElementById('quant-detail-group');
  const label = document.getElementById('quant-detail-label');
  const select = document.getElementById('quant-detail-select');
  select.innerHTML = '';

  if (strategy === 'none' || !window._quantFormats || !window._quantFormats[strategy]) {
    group.style.display = 'none';
    return;
  }
  group.style.display = '';
  const spec = window._quantFormats[strategy];
  if (spec.kind === 'bits') {
    label.textContent = 'Bits';
    spec.options.forEach(bits => {
      const opt = document.createElement('option');
      opt.value = String(bits);
      opt.textContent = `${bits}-bit`;
      select.appendChild(opt);
    });
    // 4-bit is the common default across AWQ/GPTQ.
    select.value = spec.options.includes(4) ? '4' : String(spec.options[0]);
  } else {
    label.textContent = 'Quant type';
    spec.options.forEach(o => {
      const opt = document.createElement('option');
      opt.value = o.name;
      opt.textContent = `${o.name} (~${o.bits}-bit) — ${o.description}`;
      select.appendChild(opt);
    });
  }
}

function renderConfigDiff(before, after, targetId = 'config-diff') {
  const el = document.getElementById(targetId);
  if (!el) return;
  const beforeLines = before.split('\n');
  const afterLines = after.split('\n');
  const beforeSet = new Set(beforeLines);
  const afterSet = new Set(afterLines);
  const added = afterLines.filter(l => l.trim() && !beforeSet.has(l));
  const removed = beforeLines.filter(l => l.trim() && !afterSet.has(l));
  if (added.length === 0 && removed.length === 0) {
    el.innerHTML = '<div class="empty-state-hint">No change.</div>';
    return;
  }
  let html = '<div class="code-block" style="font-size:0.8rem">';
  removed.forEach(l => { html += `<div style="color:var(--danger)">- ${escapeHtml(l)}</div>`; });
  added.forEach(l => { html += `<div style="color:var(--accent)">+ ${escapeHtml(l)}</div>`; });
  html += '</div>';
  el.innerHTML = html;
}
function onRamCacheSliderInput() {
  const v = parseFloat(document.getElementById('ram-cache-slider').value || '0');
  document.getElementById('ram-cache-val').textContent =
    v > 0 ? `${v.toFixed(1)} GB` : '0 GB (disabled)';
}

async function applyRamPrefetchSettings() {
  const statusEl = document.getElementById('ram-prefetch-status');
  const editor = document.getElementById('config-editor');
  if (!editor) return;
  const ramGb = parseFloat(document.getElementById('ram-cache-slider').value || '0');
  statusEl.textContent = 'Applying...';
  statusEl.style.color = 'var(--text-dim)';
  const before = editor.value;
  try {
    const result = await api('/api/config/patch-ram-prefetch', {
      method: 'POST',
      body: JSON.stringify({ yaml: editor.value, ram_cache_gb: ramGb }),
    });
    editor.value = result.yaml;
    statusEl.textContent = 'Applied to config below.';
    statusEl.style.color = 'var(--accent)';
    renderConfigDiff(before, result.yaml, 'ram-prefetch-diff');
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
    statusEl.style.color = 'var(--danger)';
  }
}

async function applyQuantSettings() {
  const statusEl = document.getElementById('quant-settings-status');
  const editor = document.getElementById('config-editor');
  if (!editor) return;
  const quant = document.getElementById('quant-strategy-select').value;
  const detailSelect = document.getElementById('quant-detail-select');
  const detail = (quant !== 'none' && detailSelect.options.length) ? detailSelect.value : '';
  statusEl.textContent = 'Applying...';
  statusEl.style.color = 'var(--text-dim)';
  const before = editor.value;
  try {
    const result = await api('/api/config/patch-quant', {
      method: 'POST',
      body: JSON.stringify({
        yaml: editor.value,
        custom_quant_strategy: quant,
        custom_quant_detail: detail,
      }),
    });
    editor.value = result.yaml;
    statusEl.textContent = 'Applied to config below.';
    statusEl.style.color = 'var(--accent)';
    renderConfigDiff(before, result.yaml, 'quant-diff');
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
    statusEl.style.color = 'var(--danger)';
  }
}

function renderTrainingPage(templates, status) {
  const templateNames = Object.keys(templates);
  const editorId = 'config-editor';

  document.getElementById('training-content').innerHTML = `
    <!-- Training Status: full-width, first thing on the page — it used to
         sit in a side column at the same height as Template/Config, easy
         to miss and easy to mistake for secondary. Progress/log detail
         still lives in the Training Progress / Training Logs cards right
         below it once a run is active. -->
    <div class="card" id="training-status-card">
      <div class="card-title">Training Status</div>
      <div id="train-status-panel">
        ${status.running
          ? `<div><span class="badge badge-warning">${status.paused ? 'Paused' : 'Running'}</span> PID: ${escapeHtml(String(status.pid))}
               ${status.phase ? ` — ${escapeHtml(status.phase)}` : ''}</div>
             <div style="margin-top:0.5rem;font-size:0.8rem;color:var(--text-dim)">
               Live progress, phase, and logs are below (Training Progress / Training Logs cards).
             </div>`
          : '<div style="color:var(--text-dim)">No training in progress — configure a run below.</div>'
        }
      </div>
    </div>

    <div class="card">
      <div class="card-title">1. Template / Recipe</div>
      <div class="empty-state-hint" style="margin-bottom:0.6rem">
        Pick one to pre-fill the config below with a working starting point, or write your
        own from scratch. Templates marked "needs data" use a placeholder dataset path you
        must change before training — the ones without that marker point at a small bundled
        fixture so they run end-to-end as-is (good for verifying your setup; the fixtures are
        far too small to produce a useful model). Recipes are hyperparameter presets for 140+
        specific models — all of them need a real dataset, by design.
      </div>
      <div class="grid-2" style="margin-bottom:0">
        <div class="form-group" style="margin-bottom:0">
          <label class="form-label" style="font-size:0.8rem">Template</label>
          <select id="template-select" onchange="loadTemplate()">
            <option value="">-- Select a template --</option>
            ${templateNames.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}${_templateNeedsData(templates[t]) ? ' (needs data)' : ''}</option>`).join('')}
          </select>
        </div>
        <div class="form-group" style="margin-bottom:0">
          <label class="form-label" style="font-size:0.8rem">Recipe</label>
          <select id="recipe-select" onchange="loadRecipe()">
            <option value="">-- Select a recipe --</option>
            ${(window._recipes || []).map(r => `<option value="${escapeHtml(r.name)}">${escapeHtml(r.name)} (${escapeHtml(r.task)})</option>`).join('')}
          </select>
        </div>
      </div>
      <div id="template-warning" style="margin-top:0.6rem"></div>
    </div>

    <div class="card">
      <div class="card-title">2. Dataset, model &amp; output paths</div>
      <div class="empty-state-hint" style="margin-bottom:0.6rem">
        These edit the same YAML below — use them or edit the YAML directly, both stay in
        sync. "Browse" lists real files/directories on this machine so you don't have to
        remember or type exact paths.
      </div>
      <div class="grid-3">
        <div class="form-group">
          <label class="form-label">Base model (HF id or local path)</label>
          <div style="display:flex;gap:0.4rem">
            <input type="text" id="quickset-base" placeholder="e.g. meta-llama/Llama-3.1-8B-Instruct">
            <button class="btn btn-sm" onclick="browsePath('quickset-base', {mode:'dir', label:'model directory'})">Browse</button>
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">Training data (data.train)</label>
          <div style="display:flex;gap:0.4rem">
            <input type="text" id="quickset-train" placeholder="e.g. examples/data/alpaca_tiny.jsonl">
            <button class="btn btn-sm" onclick="browsePath('quickset-train', {mode:'file', extensions:['.jsonl','.json','.csv','.parquet','.txt'], label:'dataset file'})">Browse</button>
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">Output directory (output)</label>
          <div style="display:flex;gap:0.4rem">
            <input type="text" id="quickset-output" placeholder="e.g. ./output">
            <button class="btn btn-sm" onclick="browsePath('quickset-output', {mode:'dir', label:'output directory', allowNew:true})">Browse</button>
          </div>
        </div>
      </div>
      <button class="btn btn-primary btn-sm" onclick="applyQuicksetPaths()">Apply to config</button>
      <span id="quickset-status" style="margin-left:0.75rem;font-size:0.85rem;color:var(--text-dim)"></span>
    </div>

    <div class="card">
      <div class="card-title">3. Config (YAML)</div>
      <textarea id="${editorId}" rows="22" placeholder="Select a template above, or paste/write your own soup.yaml config here..."></textarea>
    </div>

    <div style="display:flex; gap:0.75rem; margin-top:0.75rem">
      <button class="btn btn-primary" onclick="validateConfig()">Validate</button>
      <button class="btn btn-primary" onclick="startTraining()">Start Training</button>
    </div>
    <div id="config-status" style="margin-top:0.75rem; font-size:0.85rem"></div>
  `;

  // Store templates globally
  window._templates = templates;
}

function _templateNeedsData(yamlStr) {
  // Templates mark their own placeholders explicitly (see config/schema.py's
  // TEMPLATES). Recipes (config/recipes/catalog.py — 140+ model/hyperparameter
  // presets) don't carry that marker text, but every single one consistently
  // uses a bare `./data/...` placeholder path by convention (verified: none
  // of the bundled, real `examples/data/*.jsonl` fixtures use that prefix),
  // so that pattern catches recipes too without needing a network round-trip
  // to check file existence for every dropdown selection.
  return yamlStr.includes('# <-- change this to your dataset')
    || yamlStr.includes('PLACEHOLDER PATH')
    || /^\s+train:\s*\.\/data\//m.test(yamlStr);
}

function loadTemplate() {
  const sel = document.getElementById('template-select');
  const editor = document.getElementById('config-editor');
  const warningEl = document.getElementById('template-warning');
  if (sel.value && window._templates[sel.value]) {
    const yamlStr = window._templates[sel.value];
    editor.value = yamlStr;
    if (warningEl) {
      warningEl.innerHTML = _templateNeedsData(yamlStr)
        ? `<div class="empty-state-hint" style="color:var(--warning);border:1px solid var(--warning);border-radius:6px;padding:0.5rem 0.75rem">
             ⚠️ This template needs a real dataset — the path in <code>data.train</code> is a
             placeholder that doesn't exist. Use "Dataset, model &amp; output paths" below (or
             edit the YAML directly) before starting training.
           </div>`
        : '';
    }
    syncQuicksetFromEditor();
    syncStreamingPanelFromEditor();
  } else if (warningEl) {
    warningEl.innerHTML = '';
  }
}

function loadRecipe() {
  const sel = document.getElementById('recipe-select');
  const editor = document.getElementById('config-editor');
  if (!sel.value || !window._recipes) return;
  const recipe = window._recipes.find(r => r.name === sel.value);
  if (recipe && recipe.yaml) {
    editor.value = recipe.yaml;
    const warningEl = document.getElementById('template-warning');
    if (warningEl) {
      warningEl.innerHTML = _templateNeedsData(recipe.yaml)
        ? `<div class="empty-state-hint" style="color:var(--warning);border:1px solid var(--warning);border-radius:6px;padding:0.5rem 0.75rem">
             ⚠️ This recipe needs a real dataset — the path in <code>data.train</code> is a
             placeholder that doesn't exist. Recipes are hyperparameter presets for specific
             models, not runnable-as-is fixtures. Use "Dataset, model &amp; output paths"
             below (or edit the YAML directly) before starting training.
           </div>`
        : '';
    }
    syncQuicksetFromEditor();
    syncStreamingPanelFromEditor();
  }
}

// --- Dataset/model/output quick-set paths (this session) ---
//
// Small, targeted YAML text edits (not a full YAML AST) — reads/writes
// only the `base:`, `data:\n  train:`, and `output:` lines. Good enough
// for the shape every template/recipe here actually uses; a config with
// wildly different structure just won't populate the quick-set fields
// (syncQuicksetFromEditor leaves them blank), and applying still only
// ever touches those three specific keys.

function syncQuicksetFromEditor() {
  const yamlStr = document.getElementById('config-editor').value;
  const base = yamlStr.match(/^base:\s*(.+)$/m);
  const train = yamlStr.match(/^\s+train:\s*(.+)$/m);
  const output = yamlStr.match(/^output:\s*(.+)$/m);
  const baseEl = document.getElementById('quickset-base');
  const trainEl = document.getElementById('quickset-train');
  const outputEl = document.getElementById('quickset-output');
  if (baseEl) baseEl.value = base ? base[1].trim() : '';
  if (trainEl) trainEl.value = train ? train[1].trim() : '';
  if (outputEl) outputEl.value = output ? output[1].trim() : '';
}

function applyQuicksetPaths() {
  const editor = document.getElementById('config-editor');
  const statusEl = document.getElementById('quickset-status');
  if (!editor.value.trim()) {
    statusEl.textContent = 'Select a template or write a config first.';
    statusEl.style.color = 'var(--warning)';
    return;
  }
  let yamlStr = editor.value;
  const base = document.getElementById('quickset-base').value.trim();
  const train = document.getElementById('quickset-train').value.trim();
  const output = document.getElementById('quickset-output').value.trim();

  if (base) {
    yamlStr = yamlStr.match(/^base:\s*.+$/m)
      ? yamlStr.replace(/^base:\s*.+$/m, `base: ${base}`)
      : `base: ${base}\n` + yamlStr;
  }
  if (train) {
    if (yamlStr.match(/^(\s+)train:\s*.+$/m)) {
      yamlStr = yamlStr.replace(/^(\s+)train:\s*.+$/m, `$1train: ${train}`);
    } else if (yamlStr.match(/^data:\s*$/m)) {
      yamlStr = yamlStr.replace(/^data:\s*$/m, `data:\n  train: ${train}`);
    } else {
      // No `train:` key AND no bare `data:` block header at all — the
      // previous version silently dropped the path here instead of
      // falling through to this case, so a config with no `data:` section
      // yet (e.g. a from-scratch config, or one where `data:` has other
      // keys inline on the same line) never got a train path applied with
      // no indication why.
      yamlStr = yamlStr + `\ndata:\n  train: ${train}\n`;
    }
  }
  if (output) {
    yamlStr = yamlStr.match(/^output:\s*.+$/m)
      ? yamlStr.replace(/^output:\s*.+$/m, `output: ${output}`)
      : yamlStr + `\noutput: ${output}\n`;
  }
  editor.value = yamlStr;
  const warningEl = document.getElementById('template-warning');
  if (warningEl && !_templateNeedsData(yamlStr)) warningEl.innerHTML = '';
  statusEl.textContent = 'Applied to config below.';
  statusEl.style.color = 'var(--accent)';
}

async function browsePath(targetInputId, opts) {
  opts = opts || {};
  const startPath = document.getElementById(targetInputId).value.trim() || '.';
  openFileBrowserModal(startPath, opts, (chosenPath) => {
    document.getElementById(targetInputId).value = chosenPath;
  });
}

// --- Generic file/directory browser modal (this session) ---
//
// mode: 'dir' (Select button picks the currently-listed directory) or
// 'file' (clicking a file selects it directly; extensions filters what's
// shown). allowNew: true also accepts a not-yet-existing path typed into
// the manual input (for a fresh output directory that doesn't exist yet).

let _fsBrowserState = { path: '.', opts: {}, onSelect: null };

async function openFileBrowserModal(startPath, opts, onSelect) {
  _fsBrowserState = { path: startPath, opts, onSelect };
  document.getElementById('fs-browser-title').textContent =
    'Browse for ' + (opts.label || 'a path');
  document.getElementById('fs-browser-select-btn').style.display =
    opts.mode === 'file' ? 'none' : '';
  document.getElementById('fs-browser-modal').style.display = 'flex';
  await fsBrowserNavigate(startPath);
}

function closeFileBrowserModal() {
  document.getElementById('fs-browser-modal').style.display = 'none';
}

async function fsBrowserNavigate(path) {
  const listEl = document.getElementById('fs-browser-list');
  const pathEl = document.getElementById('fs-browser-path');
  listEl.innerHTML = '<div class="empty-state-hint" style="padding:0.75rem">Loading...</div>';
  try {
    const opts = _fsBrowserState.opts;
    const params = new URLSearchParams({ path });
    if (opts.mode === 'file' && opts.extensions) {
      params.set('extensions', opts.extensions.join(','));
    }
    const data = await api('/api/fs/browse?' + params.toString());
    _fsBrowserState.path = data.path;
    pathEl.textContent = data.path;
    document.getElementById('fs-browser-manual').value = data.path;

    let html = '';
    if (data.parent) {
      html += `<div class="fs-browser-item" onclick="fsBrowserNavigate('${escapeHtml(data.parent).replace(/'/g, "\\'")}')">
                 <span class="fs-icon">↑</span> ..</div>`;
    }
    for (const entry of data.entries) {
      const icon = entry.type === 'dir' ? '📁' : '📄';
      const escPath = escapeHtml(entry.path).replace(/'/g, "\\'");
      if (entry.type === 'dir') {
        html += `<div class="fs-browser-item" onclick="fsBrowserNavigate('${escPath}')">
                   <span class="fs-icon">${icon}</span> ${escapeHtml(entry.name)}/</div>`;
      } else {
        html += `<div class="fs-browser-item" onclick="fsBrowserPickFile('${escPath}')">
                   <span class="fs-icon">${icon}</span> ${escapeHtml(entry.name)}</div>`;
      }
    }
    listEl.innerHTML = html || '<div class="empty-state-hint" style="padding:0.75rem">Empty directory.</div>';
  } catch (err) {
    listEl.innerHTML = `<div class="empty-state-hint" style="padding:0.75rem;color:var(--danger)">${escapeHtml(err.message)}</div>`;
  }
}

function fsBrowserGo() {
  const manual = document.getElementById('fs-browser-manual').value.trim();
  if (!manual) return;
  const opts = _fsBrowserState.opts;
  if (opts.mode === 'dir' && opts.allowNew) {
    // Accept a path that doesn't exist yet (e.g. a fresh output dir) —
    // try to navigate; if the backend 404s (doesn't exist), just select it
    // directly instead of forcing the user to pre-create it out of band.
    fsBrowserNavigate(manual).catch(() => {});
  } else {
    fsBrowserNavigate(manual);
  }
}

function fsBrowserPickFile(path) {
  if (_fsBrowserState.onSelect) _fsBrowserState.onSelect(path);
  closeFileBrowserModal();
}

function fsBrowserSelectCurrent() {
  const opts = _fsBrowserState.opts;
  const manual = document.getElementById('fs-browser-manual').value.trim();
  const chosen = (opts.allowNew && manual) ? manual : _fsBrowserState.path;
  if (_fsBrowserState.onSelect) _fsBrowserState.onSelect(chosen);
  closeFileBrowserModal();
}

async function validateConfig() {
  const yaml = document.getElementById('config-editor').value;
  const statusEl = document.getElementById('config-status');
  try {
    const result = await api('/api/config/validate', {
      method: 'POST',
      body: JSON.stringify({ yaml }),
    });
    if (result.valid) {
      statusEl.innerHTML = '<span style="color:var(--accent)">Config is valid!</span>';
    } else {
      statusEl.innerHTML = `<span style="color:var(--danger)">Invalid: ${escapeHtml(result.error)}</span>`;
    }
  } catch (err) {
    statusEl.innerHTML = `<span style="color:var(--danger)">Error: ${escapeHtml(err.message)}</span>`;
  }
}

async function startTraining() {
  const yaml = document.getElementById('config-editor').value;
  if (!yaml.trim()) {
    alert('Please enter a config');
    return;
  }
  if (!confirm('Start training with this config?')) return;

  try {
    const result = await api('/api/train/start', {
      method: 'POST',
      body: JSON.stringify({ config_yaml: yaml }),
    });
    document.getElementById('config-status').innerHTML =
      `<span style="color:var(--accent)">Training started! PID: ${escapeHtml(String(result.pid))}</span>`;
    // Bug fix (this session): this used to only refresh the static status
    // text ("Running, PID: X") via a full loadTrainingPage() reload — the
    // Training Progress bar and Training Logs panel below (which already
    // existed in the page markup) were never actually connected, so
    // nothing visibly changed after the PID appeared. Connect them
    // immediately instead of waiting for the next manual page reload.
    connectTrainingSSE();
    startTrainingProgressPolling();
    loadTrainingPage();
  } catch (err) {
    document.getElementById('config-status').innerHTML =
      `<span style="color:var(--danger)">Error: ${escapeHtml(err.message)}</span>`;
  }
}

async function stopTraining() {
  if (!confirm('Stop the current training run?')) return;
  try {
    await api('/api/train/stop', { method: 'POST' });
    stopTrainingProgressPolling();
    loadTrainingPage();
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

// --- Pause/Resume + live progress polling (this session) ---
//
// Pause suspends the OS process (SIGSTOP/SIGCONT) — it frees compute, NOT
// VRAM (model/optimizer state stay allocated the whole time). See
// /api/train/pause's docstring in ui/app.py.
let _trainProgressPollHandle = null;

async function togglePauseTraining() {
  const btn = document.getElementById('pause-resume-btn');
  const isPaused = btn && btn.textContent.trim() === 'Resume';
  try {
    await api(isPaused ? '/api/train/resume' : '/api/train/pause', { method: 'POST' });
    if (btn) btn.textContent = isPaused ? 'Pause' : 'Resume';
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

function startTrainingProgressPolling() {
  stopTrainingProgressPolling();
  const startedAt = Date.now();
  _trainProgressPollHandle = setInterval(async () => {
    try {
      const runId = window._currentRunId || null;
      const data = await api('/api/train/progress' + (runId ? `?run_id=${encodeURIComponent(runId)}` : ''));
      if (!data.running) {
        stopTrainingProgressPolling();
        loadTrainingPage();
        return;
      }
      const phaseEl = document.getElementById('progress-phase');
      if (phaseEl && data.phase) phaseEl.textContent = data.phase;

      const btn = document.getElementById('pause-resume-btn');
      if (btn) btn.textContent = data.paused ? 'Resume' : 'Pause';

      const speedEl = document.getElementById('progress-speed');
      if (speedEl) speedEl.textContent = data.speed ? `${data.speed.toFixed(2)} it/s` : '';

      const elapsed = (Date.now() - startedAt) / 1000;
      updateProgressBar(
        data.current_step || 0,
        data.total_steps || 0,
        elapsed,
        data.eta_seconds != null ? data.eta_seconds : null,
      );
    } catch (err) {
      // Transient network hiccup — keep polling, don't spam the user.
    }
  }, 2000);
}

function stopTrainingProgressPolling() {
  if (_trainProgressPollHandle !== null) {
    clearInterval(_trainProgressPollHandle);
    _trainProgressPollHandle = null;
  }
}

// --- Data Explorer ---
async function inspectData() {
  const path = document.getElementById('data-path').value;
  if (!path.trim()) { alert('Enter a file path'); return; }

  const limit = parseInt(document.getElementById('data-limit').value) || 50;
  const content = document.getElementById('data-content');
  content.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--text-dim)">Loading...</div>';

  try {
    const result = await api('/api/data/inspect', {
      method: 'POST',
      body: JSON.stringify({ path, limit }),
    });
    renderDataResults(result);
  } catch (err) {
    content.innerHTML = `<div class="empty-state"><div class="empty-state-text">Error: ${escapeHtml(err.message)}</div></div>`;
  }
}

function renderDataResults(data) {
  const content = document.getElementById('data-content');

  content.innerHTML = `
    <div class="stats-row" style="margin-bottom:1rem">
      <div class="card stat-card">
        <div class="stat-value">${data.total}</div>
        <div class="stat-label">Total Entries</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value" style="font-size:1.5rem">${data.format}</div>
        <div class="stat-label">Detected Format</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">${data.keys.length}</div>
        <div class="stat-label">Fields</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Fields: ${data.keys.join(', ')}</div>
    </div>

    <div class="card">
      <div class="card-title">Sample Data (${data.sample.length} of ${data.total})</div>
      ${data.sample.map((entry, idx) => `
        <div class="data-entry">
          <div style="font-size:0.75rem; color:var(--text-dim); margin-bottom:0.5rem">#${idx + 1}</div>
          ${Object.entries(entry).map(([key, val]) => `
            <div class="data-entry-field">
              <span class="data-entry-key">${key}:</span>
              <span>${typeof val === 'object' ? JSON.stringify(val).substring(0, 200) : String(val).substring(0, 200)}</span>
            </div>
          `).join('')}
        </div>
      `).join('')}
    </div>
  `;
}

// --- Model Chat ---
let chatAbortController = null;

function loadChatPage() {
  renderChatMessages();
}

function renderChatMessages() {
  const container = document.getElementById('chat-messages');
  if (!container) return;

  if (chatMessages.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-text">No messages yet</div>
        <div class="empty-state-hint">Enter a server URL and start chatting</div>
      </div>
    `;
    return;
  }

  container.innerHTML = chatMessages.map(msg => `
    <div class="chat-msg ${msg.role}">
      <div class="chat-msg-role">${msg.role}</div>
      <div class="chat-msg-content chat-markdown">${msg.role === 'assistant' ? renderMarkdown(msg.content) : escapeHtml(msg.content)}</div>
    </div>
  `).join('');

  container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderMarkdown(text) {
  let html = escapeHtml(text);
  // Code blocks (``` ... ```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="code-block"><code>$2</code></pre>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // List items
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  // Line breaks
  html = html.replace(/\n/g, '<br>');
  return html;
}

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const serverUrl = document.getElementById('chat-server').value.trim();
  const msg = input.value.trim();
  if (!msg) return;
  if (!serverUrl) { alert('Enter a server URL'); return; }

  // Build messages with optional system prompt
  const systemPrompt = document.getElementById('chat-system')?.value?.trim();
  const allMessages = [];
  if (systemPrompt) {
    allMessages.push({ role: 'system', content: systemPrompt });
  }
  chatMessages.push({ role: 'user', content: msg });
  allMessages.push(...chatMessages.map(m => ({ role: m.role, content: m.content })));
  input.value = '';
  renderChatMessages();

  // Show typing indicator, switch buttons
  const typing = document.getElementById('typing-indicator');
  const sendBtn = document.getElementById('chat-send-btn');
  const cancelBtn = document.getElementById('chat-cancel-btn');
  if (typing) typing.style.display = 'flex';
  if (sendBtn) sendBtn.style.display = 'none';
  if (cancelBtn) cancelBtn.style.display = 'inline-flex';

  chatAbortController = new AbortController();

  const temperature = parseFloat(document.getElementById('chat-temperature')?.value || '0.7');
  const maxTokens = parseInt(document.getElementById('chat-max-tokens')?.value || '512');
  const topP = parseFloat(document.getElementById('chat-top-p')?.value || '0.9');
  const adapter = document.getElementById('chat-adapter')?.value?.trim() || undefined;

  try {
    const resp = await fetch(API + '/api/chat/send', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + (window._authToken || ''),
      },
      body: JSON.stringify({
        messages: allMessages,
        endpoint: serverUrl,
        temperature: temperature,
        max_tokens: maxTokens,
        top_p: topP,
        adapter: adapter,
      }),
      signal: chatAbortController.signal,
    });

    // Stream tokens
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let assistantMsg = '';
    chatMessages.push({ role: 'assistant', content: '' });

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.done) break;
            if (data.delta) {
              assistantMsg += data.delta;
              chatMessages[chatMessages.length - 1].content = assistantMsg;
              renderChatMessages();
            }
            if (data.error) {
              assistantMsg += '[Error: ' + data.error + ']';
              chatMessages[chatMessages.length - 1].content = assistantMsg;
              renderChatMessages();
            }
          } catch (parseErr) { /* ignore malformed lines */ }
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      chatMessages.push({ role: 'assistant', content: `[Error: ${err.message}]` });
      renderChatMessages();
    }
  } finally {
    chatAbortController = null;
    if (typing) typing.style.display = 'none';
    if (sendBtn) sendBtn.style.display = 'inline-flex';
    if (cancelBtn) cancelBtn.style.display = 'none';
  }
}

function cancelChat() {
  if (chatAbortController) chatAbortController.abort();
}

function clearChat() {
  chatMessages = [];
  renderChatMessages();
}

function exportChat() {
  if (chatMessages.length === 0) return;
  const blob = new Blob([JSON.stringify(chatMessages, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'chat-export.json';
  a.click();
  URL.revokeObjectURL(url);
}

function handleChatKey(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendChatMessage();
  }
}

// --- Training Live Monitor (SSE) ---
let logEventSource = null;
let metricsEventSource = null;
const LOG_MAX_LINES = 500;

function connectTrainingSSE() {
  disconnectTrainingSSE();

  const logPanel = document.getElementById('train-log-panel');
  const progressPanel = document.getElementById('train-progress');
  const liveBadge = document.getElementById('live-badge');
  if (!logPanel) return;

  logPanel.style.display = 'block';
  progressPanel.style.display = 'block';
  if (liveBadge) liveBadge.style.display = 'inline-flex';

  // Connect to log SSE
  logEventSource = new EventSource(API + '/api/train/logs');
  logEventSource.onmessage = function(ev) {
    const data = JSON.parse(ev.data);
    appendLogLine(data.line);
  };
  logEventSource.addEventListener('done', function() {
    appendLogLine('[Training finished]');
    disconnectTrainingSSE();
  });
  logEventSource.onerror = function() {
    setTimeout(function() {
      if (logEventSource && logEventSource.readyState === EventSource.CLOSED) {
        disconnectTrainingSSE();
      }
    }, 5000);
  };
}

function disconnectTrainingSSE() {
  if (logEventSource) { logEventSource.close(); logEventSource = null; }
  if (metricsEventSource) { metricsEventSource.close(); metricsEventSource = null; }
  const liveBadge = document.getElementById('live-badge');
  if (liveBadge) liveBadge.style.display = 'none';
}

function appendLogLine(text) {
  const output = document.getElementById('log-output');
  if (!output) return;
  output.textContent += text + '\n';
  // Ring buffer: trim to max lines
  const lines = output.textContent.split('\n');
  if (lines.length > LOG_MAX_LINES) {
    output.textContent = lines.slice(lines.length - LOG_MAX_LINES).join('\n');
  }
  // Auto-scroll
  const autoScroll = document.getElementById('log-autoscroll');
  if (autoScroll && autoScroll.checked) {
    output.scrollTop = output.scrollHeight;
  }
}

function updateProgressBar(step, total, elapsed, eta) {
  const fill = document.getElementById('progress-fill');
  const stepEl = document.getElementById('progress-step');
  const elapsedEl = document.getElementById('progress-elapsed');
  const etaEl = document.getElementById('progress-eta');
  if (!fill) return;

  const pct = total > 0 ? Math.min(100, (step / total) * 100) : 0;
  fill.style.width = pct.toFixed(1) + '%';
  if (stepEl) stepEl.textContent = 'Step ' + step + (total ? '/' + total : '');
  if (elapsedEl) elapsedEl.textContent = 'Elapsed: ' + formatDuration(elapsed);
  if (etaEl) etaEl.textContent = 'ETA: ' + formatDuration(eta);
}

// --- HF Hub: HuggingFace model / dataset / benchmark search + download (renamed from "Model Hub" this session — clearer since it also covers datasets/benchmarks, not just models) ---
// All search calls are read-only GETs (no auth needed, same tier as
// /api/templates). Downloads go through the authed api() helper and land
// under ./models/<org>__<name> or ./datasets/<org>__<name> by default (see
// soup_cli/ui/app.py — path-validated to stay under the working directory).
window._hubTab = 'model';
let _hubPollHandle = null;

// --- Help / Tutorial page (this session) ---
//
// A beginner-oriented, step-by-step walkthrough of what each part of the
// app does and why — separate from Quick Reference (a fast lookup table
// for people who already know what they're doing).

function loadHelpPage() {
  const el = document.getElementById('help-content');
  if (!el) return;
  el.innerHTML = HELP_SECTIONS.map((s, i) => `
    <details class="card" id="help-section-${i}" ${i === 0 ? 'open' : ''} style="padding:0">
      <summary class="card-title" style="cursor:pointer;padding:1rem 1.25rem;margin-bottom:0">
        ${escapeHtml(s.title)}
      </summary>
      <div style="padding:0 1.25rem 1.25rem 1.25rem;font-size:0.9rem;line-height:1.7">
        ${s.body}
      </div>
    </details>
  `).join('');
}

const HELP_SECTIONS = [
  {
    title: '1. What is Soup, in one paragraph',
    body: `
      <p>Soup fine-tunes a language model on your own data. You write a small YAML config
      (which base model, which dataset, which task), run <code>soup train</code> (or press
      "Start Training" here), and it produces a fine-tuned checkpoint. This Web UI is a
      convenience layer over that same CLI — every button here corresponds to a real
      <code>soup</code> command you could also type yourself.</p>
      <p>The whole point of the config file is that it's the <strong>one place</strong> that
      describes an entire run: what model, what data, how long, what compression, what
      quantization. You never edit Python code to fine-tune a model.</p>`,
  },
  {
    title: '2. Step-by-step: your first training run',
    body: `
      <ol style="padding-left:1.25rem">
        <li><strong>Pick a base model.</strong> Go to <a href="#" onclick="navigate('modelhub');return false">HF Hub</a>,
          search for a model (e.g. "llama 3.1 8b instruct"), and copy its id
          (looks like <code>meta-llama/Llama-3.1-8B-Instruct</code>). Smaller models (1-8B)
          train faster and need less VRAM — good for a first run.</li>
        <li><strong>Prepare your dataset.</strong> A JSONL file, one training example per line.
          The simplest format (Alpaca) looks like:
          <pre class="log-panel" style="margin:0.5rem 0">{"instruction": "Summarize this text", "input": "...", "output": "..."}</pre>
          Don't have data yet? The bundled example at <code>examples/data/alpaca_tiny.jsonl</code>
          lets you test the whole pipeline end-to-end (it won't produce a <em>useful</em>
          model — just proves your setup works).</li>
        <li><strong>Go to New Training</strong> and pick a template close to your use case
          (chat, code, etc.) — it pre-fills a sensible starting config.</li>
        <li><strong>Edit the config</strong> in the text box: at minimum, set
          <code>base:</code> to your model id and <code>data.train:</code> to your dataset path.</li>
        <li><strong>Click Start Training.</strong> Watch the Training Progress card below —
          it shows the current phase, a progress bar (once step count is known), and live
          logs. You can Pause (frees compute, keeps GPU memory allocated) or Stop at any
          time.</li>
        <li><strong>When it finishes</strong>, the checkpoint is in the directory you set as
          <code>output:</code> in the config (default <code>./output</code>). From there you
          can quantize it (see section 5) or chat with it directly.</li>
      </ol>`,
  },
  {
    title: '3. What each New Training section does',
    body: `
      <p><strong>Template / Config editor</strong> — the YAML that fully describes the run.
      Everything else on this page is a convenience form that edits this same YAML for you
      (so it always stays the single source of truth — you can always hand-edit it directly
      instead).</p>
      <p><strong>Layer RAM Prefetch</strong> — only matters for models too large to fit
      entirely in VRAM. A background thread reads upcoming layers into pinned RAM ahead of
      when they're needed, so streaming from RAM/disk doesn't stall the GPU waiting. Leave at
      0 (disabled) unless you've hit an out-of-memory error with a large model.</p>
      <p><strong>Model Quantization</strong> — <em>after</em> training finishes, optionally
      shrink the checkpoint for deployment (smaller file, faster inference, some quality
      loss). This does not affect training itself — training always runs at the precision set
      elsewhere in the config. See section 5 for which format to pick.</p>
      <p><strong>Compress (optional)</strong> — a separate, opt-in step for shrinking the
      <em>model architecture itself</em> (fewer/smaller weight matrices), independent of
      quantization. See section 6 — most people can skip this entirely.</p>
      <p><strong>Calculator</strong> — before you commit to a long training run, estimate how
      big the final checkpoint will be under different quantization/compression choices, so
      you know if it'll fit where you plan to deploy it.</p>`,
  },
  {
    title: '4. Training Status: phases, progress, pause',
    body: `
      <p>Once training starts, the <strong>Training Progress</strong> card shows:</p>
      <ul style="padding-left:1.25rem">
        <li><strong>Phase</strong> — a plain-language label for what's happening right now
          (loading the config, verifying your dataset, loading the model, actually training).
          Early phases can take a while for a large model — that's normal, not a hang.</li>
        <li><strong>Progress bar / ETA</strong> — shown once the total step count is known.
          This is only computable when your dataset is a local <code>.jsonl</code> file and
          batch size isn't set to <code>"auto"</code> — otherwise you'll see step count and
          speed (iterations/sec) without a percentage, which is expected, not a bug.</li>
        <li><strong>Pause</strong> — suspends the training process at the OS level. No
          progress is lost, but GPU memory stays allocated the whole time (this frees your
          CPU/compute, not VRAM). Press Resume to continue exactly where it left off.
          For actually freeing VRAM, Stop instead and resume later from a checkpoint.</li>
        <li><strong>Training Logs</strong> — the raw output of the training process, live.
          If something goes wrong, the error will be here.</li>
      </ul>`,
  },
  {
    title: '5. Quantization: which format should I pick?',
    body: `
      <table><thead><tr><th>Format</th><th>Bits</th><th>When to use it</th></tr></thead>
      <tbody>
        <tr><td>None</td><td>16</td><td>Best quality, largest file. Use if you have the VRAM/disk and want maximum quality.</td></tr>
        <tr><td>AWQ</td><td>4 only</td><td>Good quality-per-byte, fast inference with the right runtime (vLLM, TGI). 4-bit is a hard limit of the method itself, not a Soup restriction.</td></tr>
        <tr><td>GPTQ</td><td>2/3/4/8</td><td>Similar niche to AWQ, more bit-width flexibility if you need to go below 4-bit for a very tight memory budget (larger quality loss at 2-3 bit).</td></tr>
        <tr><td>GGUF k-quants</td><td>~2-8 (varies by type)</td><td>For <code>llama.cpp</code>-based local inference (LM Studio, Ollama, etc.) — CPU-friendly.</td></tr>
        <tr><td>GGUF i-quants / UD</td><td>~1-8 (varies)</td><td>Newer, generally better quality-per-byte than k-quants at the same size, same llama.cpp ecosystem.</td></tr>
      </tbody></table>
      <p style="margin-top:0.5rem">Not sure? Start with <strong>None</strong> to verify quality, then try
      <strong>AWQ 4-bit</strong> or a <strong>GGUF Q4_K_M</strong> if the full-size checkpoint is too
      large for where you want to deploy it. Use the Calculator to see the size difference before
      committing to a long run.</p>`,
  },
  {
    title: '6. Compress: do I need this?',
    body: `
      <p><strong>Short answer: probably not, unless you specifically need a smaller model
      architecture</strong> (not just a smaller file — quantization already does that with much
      less risk). Compress changes the actual weight matrices, which is a bigger intervention.</p>
      <p><strong>Neuron importance scan</strong> — ranks which neurons matter least, informational
      only, doesn't change anything by itself. Useful before deciding merge thresholds.</p>
      <p><strong>Similar-neuron merging</strong> — finds near-duplicate neurons and merges them,
      shrinking the model's intermediate size. Conservative default (0.98 similarity threshold)
      keeps quality loss small (~0.3-1.3% typical). Always writes to a NEW directory — your
      original checkpoint is never touched.</p>
      <p><strong>SVD compression</strong> — "Denoise" mode is always safe (same shape, can only
      help or be a no-op). "Factorize" mode genuinely shrinks the file but needs custom loading
      code afterward — advanced use only.</p>
      <p><strong>Distillation recovery</strong> — after compressing, quality can be recovered by
      distilling from the original (uncompressed) model as a teacher. Generates a ready-to-run
      distillation config for you.</p>
      <p>For the full ordered pipeline (scan → compress → distill, all in one training YAML), see
      <code>docs/pipeline.md</code> in the repo and the <code>training.pipeline</code> config block —
      run explicitly with <code>soup pipeline run config.yaml</code>, not automatically.</p>`,
  },
  {
    title: '7. HF Hub: finding models and datasets',
    body: `
      <p>Search across three tabs: <strong>Models</strong>, <strong>Datasets</strong>, and
      <strong>Benchmarks</strong> (a filtered view of datasets tagged as evaluation benchmarks).</p>
      <ul style="padding-left:1.25rem">
        <li><strong>Task</strong> uses different vocabularies for models (e.g.
          <code>text-generation</code>) vs datasets (e.g. <code>question-answering</code> as
          a task <em>category</em>) — the suggestions shown adapt to which tab you're on.</li>
        <li><strong>Library</strong> only applies to model search (e.g. filter to only
          <code>gguf</code>-format repos).</li>
        <li><strong>Language</strong> and <strong>License</strong> work on both tabs.</li>
        <li>All filters use Hugging Face's own vocabulary — the suggestions are common values,
          not an exhaustive list. Anything valid on huggingface.co's own filters works here too.</li>
      </ul>
      <p>Click a result to see details and download it locally, or copy its id straight into a
      training config's <code>base:</code> / <code>data.train:</code> field.</p>`,
  },
  {
    title: '8. Common problems',
    body: `
      <p><strong>"Unauthorized" on every button</strong> — you opened the UI without the
      session token. Use the full URL <code>soup ui</code> printed
      (ends in <code>?token=...</code>), or paste the token into the banner that appears at
      the top of the page.</p>
      <p><strong>Training seems stuck at "Starting"</strong> — check the Training Logs panel
      for the actual reason; large models can take a while to load, but a real error will show
      there.</p>
      <p><strong>Dataset verification failed before training started</strong> — this is
      intentional: Soup checks your train/val/calibration files for zero rows, missing files,
      or fully-duplicated data before spending GPU time on them. The error message names the
      exact problem and file. Set <code>data.verify_before_training: false</code> in the config
      to skip this check if you're confident the data is fine (e.g. an intentionally-unusual
      test fixture).</p>
      <p><strong>Progress bar shows step count but no percentage/ETA</strong> — expected when
      your dataset isn't a local <code>.jsonl</code> file, or <code>training.batch_size</code>
      is <code>"auto"</code> — the total step count can't be computed in advance in either case.</p>`,
  },
];

function loadModelHubPage() {
  switchHubTab(window._hubTab);
  renderHubDownloads();
  if (_hubPollHandle === null) {
    _hubPollHandle = setInterval(renderHubDownloads, 3000);
  }
}

function stopHubDownloadsPolling() {
  if (_hubPollHandle !== null) {
    clearInterval(_hubPollHandle);
    _hubPollHandle = null;
  }
}

function switchHubTab(tab) {
  window._hubTab = tab;
  document.querySelectorAll('.hub-tab').forEach(b => {
    b.classList.toggle('active', b.dataset.hubTab === tab);
  });
  // Library filtering only applies to model search on the Hub API.
  document.getElementById('hub-library-group').style.display = tab === 'model' ? '' : 'none';
  document.getElementById('hub-task-group').querySelector('label').textContent =
    tab === 'model' ? 'Task' : 'Task category';
  document.getElementById('hub-task').placeholder =
    tab === 'model' ? 'text-generation' : 'question-answering';

  // Bug fix (this session): the task field used the SAME free-text input
  // for both tabs, but models and datasets use different HF taxonomies
  // (pipeline_tag vs task_categories) — a value valid for one silently
  // returns zero results on the other with no indication why. Swapping the
  // <datalist> suggestions per tab doesn't restrict input, just points at
  // the right vocabulary.
  const taskOptions = tab === 'model'
    ? ['text-generation', 'text-classification', 'question-answering', 'summarization',
       'translation', 'text2text-generation', 'feature-extraction', 'image-text-to-text',
       'automatic-speech-recognition', 'text-to-speech']
    : ['text-generation', 'question-answering', 'summarization', 'translation',
       'text-classification', 'token-classification', 'text-to-speech',
       'automatic-speech-recognition', 'visual-question-answering'];
  document.getElementById('hub-task-options').innerHTML =
    taskOptions.map(t => `<option value="${escapeHtml(t)}"></option>`).join('');

  // Language filter is real for both tabs (models via a `language:<code>`
  // tag, datasets via list_datasets' native `language=` kwarg — see
  // ui/app.py's _hf_build_filter) — used to silently do nothing for
  // models despite the field being shown there too.
  const langHint = document.getElementById('hub-language-hint');
  if (langHint) {
    langHint.textContent = tab === 'model'
      ? 'Filters by the model repo\'s declared language tag.'
      : 'Filters by the dataset\'s declared language metadata.';
  }
}

async function runHubSearch() {
  const container = document.getElementById('hub-results');
  container.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--text-dim)">Searching...</div>';
  const q = document.getElementById('hub-q').value.trim();
  const task = document.getElementById('hub-task').value.trim();
  const language = document.getElementById('hub-language').value.trim();
  const license = document.getElementById('hub-license').value.trim();
  const sort = document.getElementById('hub-sort').value;
  const tab = window._hubTab;

  const params = new URLSearchParams();
  if (q) params.set('q', q);
  if (task) params.set('task', task);
  if (license) params.set('license', license);
  if (language) params.set('language', language);
  params.set('sort', sort);
  params.set('limit', '30');

  let path;
  if (tab === 'model') {
    const library = document.getElementById('hub-library').value.trim();
    if (library) params.set('library', library);
    path = '/api/hf/models/search?' + params.toString();
  } else {
    if (tab === 'benchmark') params.set('benchmark_only', 'true');
    path = '/api/hf/datasets/search?' + params.toString();
  }

  try {
    const resp = await api(path);
    renderHubResults(resp.results || [], tab === 'model' ? 'model' : 'dataset');
  } catch (err) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-text">Error: ${escapeHtml(err.message)}</div></div>`;
  }
}

function renderHubResults(results, repoType) {
  const container = document.getElementById('hub-results');
  if (results.length === 0) {
    container.innerHTML = '<div class="empty-state"><div class="empty-state-text">No results</div></div>';
    return;
  }
  container.innerHTML = '<div class="grid-3" id="hub-results-grid"></div>';
  const grid = document.getElementById('hub-results-grid');
  results.forEach(item => {
    const card = document.createElement('div');
    card.className = 'card';
    const title = document.createElement('div');
    title.className = 'card-title';
    title.textContent = item.id;
    card.appendChild(title);

    const meta = document.createElement('div');
    meta.style.cssText = 'font-size:0.8rem;color:var(--text-dim);margin-bottom:0.5rem';
    const bits = [`↓ ${item.downloads ?? 0}`, `♥ ${item.likes ?? 0}`];
    if (item.pipeline_tag) bits.push(item.pipeline_tag);
    if (item.gated) bits.push('gated');
    meta.textContent = bits.join(' · ');
    card.appendChild(meta);

    if (item.tags && item.tags.length) {
      const tagsWrap = document.createElement('div');
      tagsWrap.style.cssText = 'margin-bottom:0.75rem';
      item.tags.slice(0, 6).forEach(t => {
        const b = document.createElement('span');
        b.className = 'badge badge-info';
        b.style.marginRight = '0.25rem';
        b.textContent = t;
        tagsWrap.appendChild(b);
      });
      card.appendChild(tagsWrap);
    }

    const btn = document.createElement('button');
    btn.className = 'btn btn-primary btn-sm';
    btn.textContent = 'Download';
    btn.onclick = () => downloadHubItem(item.id, repoType, btn);
    card.appendChild(btn);

    grid.appendChild(card);
  });
}

async function downloadHubItem(repoId, repoType, btnEl) {
  if (btnEl) { btnEl.disabled = true; btnEl.textContent = 'Starting...'; }
  try {
    await api('/api/hf/download', {
      method: 'POST',
      body: JSON.stringify({ repo_id: repoId, repo_type: repoType }),
    });
    if (btnEl) { btnEl.textContent = 'Queued ✓'; }
    renderHubDownloads();
  } catch (err) {
    if (btnEl) { btnEl.disabled = false; btnEl.textContent = 'Download'; }
    alert('Download failed: ' + err.message);
  }
}

async function renderHubDownloads() {
  const container = document.getElementById('hub-downloads');
  if (!container) return;
  let payload;
  try {
    payload = await api('/api/hf/download/jobs');
  } catch (err) {
    return; // keep last-known state on transient errors
  }
  const jobs = payload.jobs || [];
  if (jobs.length === 0) {
    container.innerHTML = '<div class="empty-state-hint">No downloads yet.</div>';
    return;
  }
  const statusBadgeMap = { queued: 'badge-info', downloading: 'badge-warning', done: 'badge-success', error: 'badge-danger' };
  container.innerHTML = '<div class="table-wrap"><table><thead><tr><th>Repo</th><th>Type</th><th>Status</th><th>Local dir</th></tr></thead><tbody></tbody></table></div>';
  const tbody = container.querySelector('tbody');
  jobs.forEach(j => {
    const tr = document.createElement('tr');
    const cells = [j.repo_id, j.repo_type, null, j.local_dir];
    const tdRepo = document.createElement('td'); tdRepo.textContent = cells[0]; tr.appendChild(tdRepo);
    const tdType = document.createElement('td'); tdType.textContent = cells[1]; tr.appendChild(tdType);
    const tdStatus = document.createElement('td');
    tdStatus.innerHTML = `<span class="badge ${statusBadgeMap[j.status] || 'badge-info'}">${escapeHtml(j.status)}</span>`;
    if (j.status === 'error' && j.error) tdStatus.title = j.error;
    tr.appendChild(tdStatus);
    const tdDir = document.createElement('td'); tdDir.textContent = cells[3]; tr.appendChild(tdDir);
    tbody.appendChild(tr);
  });
}

// --- Compress: neuron importance + similar-neuron merging ---
let _compressPollHandle = null;

function loadCompressPage() {
  renderCompressJobs();
  if (_compressPollHandle === null) {
    _compressPollHandle = setInterval(renderCompressJobs, 3000);
  }
}

function stopCompressJobsPolling() {
  if (_compressPollHandle !== null) {
    clearInterval(_compressPollHandle);
    _compressPollHandle = null;
  }
}

// Auto-fill Compress section paths from the current training config's
// `output:` field (this session) — "quando si vuole utilizzare la
// compressione venga automaticamente scelta la directory giusta relativa
// al file yaml". Compress typically runs on the checkpoint a training run
// just produced, i.e. that config's `output:` dir, not the pre-training
// `base:` — so that's what gets used here. Only fills fields that are
// still empty, so it never clobbers something the person already typed.
function autofillCompressDirs() {
  const editor = document.getElementById('config-editor');
  if (!editor) return;
  const match = editor.value.match(/^output:\s*(.+)$/m);
  if (!match) return;
  const outputDir = match[1].trim();

  const modelEl = document.getElementById('compress-model');
  if (modelEl && !modelEl.value.trim()) modelEl.value = outputDir;

  const mergeOutEl = document.getElementById('merge-outputdir');
  if (mergeOutEl && !mergeOutEl.value.trim()) mergeOutEl.value = outputDir + '/compressed-merge';

  const svdOutEl = document.getElementById('svd-outputdir');
  if (svdOutEl && !svdOutEl.value.trim()) svdOutEl.value = outputDir + '/compressed-svd';
}

function _requireCompressModel() {
  const model = document.getElementById('compress-model').value.trim();
  if (!model) {
    alert('Enter a model (HF Hub id or local directory) first.');
    return null;
  }
  return model;
}

async function _pollCompressJob(jobId, onDone) {
  // Simple bounded poll: every 1.5s, up to ~5 minutes, stop on done/error.
  for (let i = 0; i < 200; i++) {
    await new Promise(res => setTimeout(res, 1500));
    let payload;
    try {
      payload = await api('/api/compress/jobs');
    } catch (err) {
      continue;
    }
    const job = (payload.jobs || []).find(j => j.job_id === jobId);
    if (job && (job.status === 'done' || job.status === 'error')) {
      onDone(job);
      return;
    }
  }
  onDone(null); // timed out — the Compress jobs table below still has it
}

function onImportanceMetricChange() {
  const metric = document.getElementById('importance-metric').value;
  document.getElementById('importance-calib-group').style.display = metric === 'wanda' ? '' : 'none';
}

async function runImportanceScan() {
  const model = _requireCompressModel();
  if (!model) return;
  const statusEl = document.getElementById('importance-status');
  const resultsEl = document.getElementById('importance-results');
  const metric = document.getElementById('importance-metric').value;

  let calibrationTexts = null;
  let calibrationDatasetPaths = null;
  if (metric === 'wanda') {
    calibrationTexts = document.getElementById('importance-calib-text').value
      .split('\n').map(s => s.trim()).filter(Boolean);
    calibrationDatasetPaths = document.getElementById('importance-calib-datasets').value
      .split('\n').map(s => s.trim()).filter(Boolean);
    if (calibrationTexts.length === 0 && calibrationDatasetPaths.length === 0) {
      alert('Wanda needs calibration text and/or at least one calibration dataset file.');
      return;
    }
  }

  statusEl.textContent = metric === 'wanda'
    ? 'Loading model + scoring on calibration data (heavier than magnitude)...'
    : 'Scanning (streamed, one weight matrix at a time)...';
  resultsEl.innerHTML = '';
  try {
    const { job_id } = await api('/api/compress/importance/scan', {
      method: 'POST',
      body: JSON.stringify({
        model,
        modules: document.getElementById('importance-modules').value,
        bottom_k: parseInt(document.getElementById('importance-bottomk').value || '10', 10),
        metric,
        calibration_texts: calibrationTexts,
        calibration_dataset_paths: calibrationDatasetPaths,
        calibration_samples_per_dataset: parseInt(
          document.getElementById('importance-calib-samples-per-file')?.value || '64', 10
        ),
      }),
    });
    renderCompressJobs();
    _pollCompressJob(job_id, job => {
      if (!job) { statusEl.textContent = 'Still running — check the jobs table below.'; return; }
      if (job.status === 'error') {
        statusEl.textContent = 'Error: ' + job.error;
        statusEl.style.color = 'var(--danger)';
        pushToast('Importance scan failed: ' + job.error, 'error');
        return;
      }
      statusEl.textContent = 'Done.';
      statusEl.style.color = 'var(--accent)';
      renderImportanceResult(job.result);
      pushToast('Importance scan complete (' + job.result.metric + ')', 'success');
    });
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  }
}

function renderImportanceResult(result) {
  const el = document.getElementById('importance-results');
  if (!result || !result.layers || result.layers.length === 0) {
    el.innerHTML = '<div class="empty-state-hint">No scannable weight matrices found.</div>';
    return;
  }
  let html = '<div class="table-wrap"><table><thead><tr><th>Param</th><th>Group</th><th>Neurons</th><th>Least-important indices (norm)</th></tr></thead><tbody>';
  result.layers.forEach(r => {
    const least = r.least_important.slice(0, 5).map(x => `#${x.index} (${x.norm.toFixed(3)})`).join(', ');
    html += `<tr><td>${escapeHtml(r.param_name)}</td><td>${escapeHtml(r.group)}</td><td>${r.n_neurons}</td><td>${escapeHtml(least)}</td></tr>`;
  });
  html += '</tbody></table></div>';
  el.innerHTML = html;
}

async function runNeuronScan() {
  const model = _requireCompressModel();
  if (!model) return;
  const statusEl = document.getElementById('merge-status');
  const resultsEl = document.getElementById('merge-results');
  statusEl.textContent = 'Scanning MLP layers...';
  resultsEl.innerHTML = '';
  try {
    const { job_id } = await api('/api/compress/neurons/scan', {
      method: 'POST',
      body: JSON.stringify({
        model,
        threshold: parseFloat(document.getElementById('merge-threshold').value),
        max_pairs_per_layer: parseInt(document.getElementById('merge-maxpairs').value || '50', 10),
      }),
    });
    renderCompressJobs();
    _pollCompressJob(job_id, job => {
      if (!job) { statusEl.textContent = 'Still running — check the jobs table below.'; return; }
      if (job.status === 'error') {
        statusEl.textContent = 'Error: ' + job.error;
        statusEl.style.color = 'var(--danger)';
        return;
      }
      statusEl.textContent = `Done — ${job.result.total_pairs} candidate pair(s) across ${job.result.layers_with_candidates} layer(s).`;
      statusEl.style.color = 'var(--accent)';
      renderMergeCandidates(job.result.candidates);
    });
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  }
}

function renderMergeCandidates(candidates) {
  const el = document.getElementById('merge-results');
  const layers = Object.keys(candidates || {});
  if (layers.length === 0) {
    el.innerHTML = '<div class="empty-state-hint">No candidates at this threshold — try lowering it.</div>';
    return;
  }
  let html = '<div class="table-wrap"><table><thead><tr><th>Layer</th><th>Pairs</th><th>Best sim</th><th>Worst sim</th></tr></thead><tbody>';
  layers.forEach(idx => {
    const sims = candidates[idx].map(c => c.joint_similarity);
    html += `<tr><td>${idx}</td><td>${sims.length}</td><td>${Math.max(...sims).toFixed(4)}</td><td>${Math.min(...sims).toFixed(4)}</td></tr>`;
  });
  html += '</tbody></table></div>';
  el.innerHTML = html;
}

async function runNeuronApply() {
  const model = _requireCompressModel();
  if (!model) return;
  const outputDir = document.getElementById('merge-outputdir').value.trim();
  if (!outputDir) {
    alert('Set an output directory first — Apply writes a new checkpoint there.');
    return;
  }
  if (!confirm(`This will write a new, smaller checkpoint to "${outputDir}". Continue?`)) return;

  const evalEnabled = document.getElementById('merge-eval-enable').checked;
  let evalTexts = null;
  if (evalEnabled) {
    evalTexts = document.getElementById('merge-eval-text').value
      .split('\n').map(s => s.trim()).filter(Boolean);
    if (evalTexts.length === 0) evalTexts = null;
  }

  const statusEl = document.getElementById('merge-status');
  document.getElementById('merge-eval-result').innerHTML = '';
  document.getElementById('merge-distill-suggest').innerHTML = '';
  statusEl.textContent = evalTexts
    ? 'Applying merge + quick eval (loads original AND merged model — slower)...'
    : 'Applying merge (this holds the model in memory once, like any checkpoint save)...';
  try {
    const { job_id } = await api('/api/compress/neurons/apply', {
      method: 'POST',
      body: JSON.stringify({
        model,
        threshold: parseFloat(document.getElementById('merge-threshold').value),
        max_pairs_per_layer: parseInt(document.getElementById('merge-maxpairs').value || '50', 10),
        output_dir: outputDir,
        allow_nonuniform: document.getElementById('merge-allow-nonuniform').checked,
        eval_texts: evalTexts,
      }),
    });
    renderCompressJobs();
    _pollCompressJob(job_id, job => {
      if (!job) { statusEl.textContent = 'Still running — check the jobs table below.'; return; }
      if (job.status === 'error') {
        statusEl.textContent = 'Error: ' + job.error;
        statusEl.style.color = 'var(--danger)';
        pushToast('Merge apply failed: ' + job.error, 'error');
        return;
      }
      statusEl.textContent = `Wrote merged model to ${job.result.output_dir}. Re-run your eval suite before shipping.`;
      statusEl.style.color = 'var(--accent)';
      pushToast('Merge applied: ' + job.result.output_dir, 'success');
      if (job.result.quick_eval) {
        const qe = job.result.quick_eval;
        document.getElementById('merge-eval-result').innerHTML =
          `<div class="empty-state-hint">Perplexity: ${qe.perplexity_before.toFixed(3)} -> ` +
          `${qe.perplexity_after.toFixed(3)} (${qe.relative_increase_pct >= 0 ? '+' : ''}${qe.relative_increase_pct.toFixed(2)}%) ` +
          `over ${qe.n_texts} samples.</div>`;
      }
      document.getElementById('merge-distill-suggest').innerHTML =
        '<button class="btn btn-sm">Prefill distillation config with this result</button>';
      document.getElementById('merge-distill-suggest').querySelector('button').onclick = () => {
        document.getElementById('distill-student').value = job.result.output_dir;
        document.getElementById('distill-teacher').value = model;
        document.getElementById('distill-student').scrollIntoView({ behavior: 'smooth', block: 'center' });
      };
    });
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  }
}

// --- SVD compression ---

async function runSvdScan() {
  const model = _requireCompressModel();
  if (!model) return;
  const statusEl = document.getElementById('svd-status');
  const resultsEl = document.getElementById('svd-results');
  statusEl.textContent = 'Analyzing singular value spectra...';
  resultsEl.innerHTML = '';
  try {
    const { job_id } = await api('/api/compress/svd/scan', {
      method: 'POST',
      body: JSON.stringify({ model, modules: 'mlp,attn', energy_thresholds: [0.90, 0.95, 0.99] }),
    });
    renderCompressJobs();
    _pollCompressJob(job_id, job => {
      if (!job) { statusEl.textContent = 'Still running — check the jobs table below.'; return; }
      if (job.status === 'error') {
        statusEl.textContent = 'Error: ' + job.error;
        statusEl.style.color = 'var(--danger)';
        pushToast('SVD scan failed: ' + job.error, 'error');
        return;
      }
      statusEl.textContent = `Done — ${job.result.matrices.length} matrices analyzed.`;
      statusEl.style.color = 'var(--accent)';
      renderSvdResults(job.result.matrices);
      pushToast('SVD analysis complete', 'success');
    });
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  }
}

function renderSvdResults(matrices) {
  const el = document.getElementById('svd-results');
  if (!matrices || matrices.length === 0) {
    el.innerHTML = '<div class="empty-state-hint">No matrices found.</div>';
    return;
  }
  let html = '<div class="table-wrap"><table><thead><tr><th>Param</th><th>Shape</th><th>Rank@90%</th><th>Rank@95%</th><th>Rank@99%</th></tr></thead><tbody>';
  matrices.forEach(m => {
    const r = m.rank_at_energy;
    html += `<tr><td>${escapeHtml(m.param_name)}</td><td>${m.shape[0]}x${m.shape[1]}</td>` +
      `<td>${r['0.9'] ?? '-'}</td><td>${r['0.95'] ?? '-'}</td><td>${r['0.99'] ?? '-'}</td></tr>`;
  });
  html += '</tbody></table></div>';
  el.innerHTML = html;
}

async function runSvdApply() {
  const model = _requireCompressModel();
  if (!model) return;
  const outputDir = document.getElementById('svd-outputdir').value.trim();
  if (!outputDir) {
    alert('Set an output directory first — Apply writes a new checkpoint there.');
    return;
  }
  const mode = document.getElementById('svd-mode').value;
  if (!confirm(`This will write a new ${mode} checkpoint to "${outputDir}". Continue?`)) return;

  const statusEl = document.getElementById('svd-status');
  statusEl.textContent = `Applying SVD (${mode})...`;
  try {
    const { job_id } = await api('/api/compress/svd/apply', {
      method: 'POST',
      body: JSON.stringify({
        model,
        modules: 'mlp,attn',
        rank_at_energy: parseFloat(document.getElementById('svd-energy').value),
        mode,
        output_dir: outputDir,
      }),
    });
    renderCompressJobs();
    _pollCompressJob(job_id, job => {
      if (!job) { statusEl.textContent = 'Still running — check the jobs table below.'; return; }
      if (job.status === 'error') {
        statusEl.textContent = 'Error: ' + job.error;
        statusEl.style.color = 'var(--danger)';
        pushToast('SVD apply failed: ' + job.error, 'error');
        return;
      }
      statusEl.textContent = `Wrote ${mode} checkpoint to ${job.result.output_dir}.` +
        (mode === 'factorize' ? ' See svd_manifest.json — needs custom loading code.' : '');
      statusEl.style.color = 'var(--accent)';
      pushToast('SVD ' + mode + ' applied: ' + job.result.output_dir, 'success');
    });
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  }
}

// --- Distillation config generator ---

async function runDistillConfig() {
  const student = document.getElementById('distill-student').value.trim();
  const teacher = document.getElementById('distill-teacher').value.trim();
  if (!student || !teacher) {
    alert('Set both student and teacher model paths.');
    return;
  }
  const statusEl = document.getElementById('distill-status');
  statusEl.textContent = 'Generating...';
  try {
    const result = await api('/api/compress/distill-config', {
      method: 'POST',
      body: JSON.stringify({
        student, teacher,
        mode: document.getElementById('distill-mode').value,
      }),
    });
    statusEl.textContent = 'Generated.';
    statusEl.style.color = 'var(--accent)';
    const el = document.getElementById('distill-result');
    el.innerHTML =
      '<div class="code-block" style="white-space:pre-wrap;max-height:300px;overflow:auto"></div>' +
      '<button class="btn btn-sm" style="margin-top:0.5rem" onclick="_openDistillInTraining()">Open in New Training</button>';
    el.querySelector('.code-block').textContent = result.yaml;
    window._lastDistillYaml = result.yaml;
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  }
}

function _openDistillInTraining() {
  navigate('training');
  // config-editor is only guaranteed to exist once loadTrainingPage()
  // finishes rendering — give it a beat rather than racing it.
  setTimeout(() => {
    const editor = document.getElementById('config-editor');
    if (editor && window._lastDistillYaml) editor.value = window._lastDistillYaml;
  }, 300);
}

async function renderCompressJobs() {
  const el = document.getElementById('compress-jobs');
  if (!el) return;
  let payload;
  try {
    payload = await api('/api/compress/jobs');
  } catch (err) {
    return;
  }
  const jobs = payload.jobs || [];
  if (jobs.length === 0) {
    el.innerHTML = '<div class="empty-state-hint">No jobs yet.</div>';
    return;
  }
  const badgeMap = { running: 'badge-warning', done: 'badge-success', error: 'badge-danger' };
  let html = '<div class="table-wrap"><table><thead><tr><th>Kind</th><th>Model</th><th>Status</th></tr></thead><tbody>';
  jobs.forEach(j => {
    html += `<tr><td>${escapeHtml(j.kind)}</td><td>${escapeHtml(j.model || '')}</td>` +
      `<td><span class="badge ${badgeMap[j.status] || 'badge-info'}" title="${escapeHtml(j.error || '')}">${escapeHtml(j.status)}</span></td></tr>`;
  });
  html += '</tbody></table></div>';
  el.innerHTML = html;
}

// --- Toast notifications (global — job completion is visible even if
// you've navigated to a different page since starting it) ---

function pushToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const colors = { success: 'var(--accent)', error: 'var(--danger)', info: 'var(--text-dim)' };
  const toast = document.createElement('div');
  toast.className = 'card';
  toast.style.cssText = `padding:0.6rem 1rem;border-left:3px solid ${colors[type] || colors.info};` +
    'box-shadow:0 4px 12px rgba(0,0,0,0.3);max-width:320px;font-size:0.85rem;animation:none';
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.4s'; }, 5000);
  setTimeout(() => toast.remove(), 5400);
}

// Tracks job ids we've already shown a toast for, so a job sitting in
// "done" state across multiple poll ticks doesn't re-notify every tick.
const _notifiedJobIds = new Set();

async function _globalJobWatcherTick() {
  try {
    const [hf, compress] = await Promise.all([
      api('/api/hf/download/jobs').catch(() => ({ jobs: [] })),
      api('/api/compress/jobs').catch(() => ({ jobs: [] })),
    ]);
    const allJobs = [
      ...(hf.jobs || []).map(j => ({ ...j, _kind: 'download', _label: j.repo_id })),
      ...(compress.jobs || []).map(j => ({ ...j, _kind: j.kind, _label: j.model || j.output_dir || '' })),
    ];
    for (const j of allJobs) {
      const terminal = j.status === 'done' || j.status === 'error';
      if (!terminal || _notifiedJobIds.has(j.job_id)) continue;
      _notifiedJobIds.add(j.job_id);
      if (j.status === 'done') {
        pushToast(`${j._kind} finished: ${j._label}`, 'success');
      } else {
        pushToast(`${j._kind} failed: ${j._label} — ${j.error || ''}`, 'error');
      }
    }
  } catch (err) {
    // silent — this is a background convenience, not critical path
  }
}

setInterval(_globalJobWatcherTick, 4000);

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
  navigate('dashboard');
});
